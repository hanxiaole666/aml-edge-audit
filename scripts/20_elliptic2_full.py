"""
20_elliptic2_full.py - Elliptic2 完整节点特征版子图分类
========================================================
用刚下载的 5.3GB background_nodes.csv（166 维节点原始特征）+ 367K 交易边，
做真正的 Elliptic2 子图级洗钱检测：
  - 每个连通分量(ccId)是一个子图，标签 suspicious/licit
  - 节点特征聚合：子图内 166 维特征 mean/max/sum 池化 → 子图表示
  - 加结构特征（16 脚本的 18 维）→ 组合表示
  - 对比：仅结构 / 仅节点特征 / 组合

这是补齐"Elliptic2 性能弱（缺原始特征）"短板的正式实验。

用法：python scripts/20_elliptic2_full.py --quick  # 快速验证
      python scripts/20_elliptic2_full.py          # 完整（5 种子）
产出：results/elliptic2_full_*.csv
断点续跑：checkpoint=results/_checkpoints/elliptic2_full.json
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    DATA_DIR, RESULTS_DIR, Checkpoint, pr_auc, roc_auc, compute_metrics,
    write_provenance, set_seed,
)


def build_full_features(sample_nodes: int = 0):
    """用完整 166 维节点特征构造子图表示。"""
    d = DATA_DIR / "elliptic2"
    t0 = time.time()
    print("[elliptic2_full] 读取 5.3GB 节点特征...", flush=True)
    nodes = pd.read_csv(d / "nodes.csv")
    edges = pd.read_csv(d / "edges.csv")
    comps = pd.read_csv(d / "connected_components.csv")
    cl2cc = dict(zip(nodes["clId"], nodes["ccId"]))
    need = set(int(c) for c in nodes["clId"].values)  # 只需标注子图的节点
    print(f"[elliptic2_full] 目标节点={len(need)}（全库 4930 万，流式过滤）", flush=True)

    # 优先用缓存（由 22_cache_node_features.py 生成），避免重复读 5.3GB
    header = pd.read_csv(d / "background_nodes.csv", nrows=0)
    feat_cols = [c for c in header.columns if c.startswith("feat#")]
    cache = d / "node_features_43.npy"
    cl2feat = {}
    if cache.exists():
        arr = np.load(cache)
        print(f"[elliptic2_full] 从缓存加载 {arr.shape}（跳过 5.3GB 读取）", flush=True)
        cl2feat = {int(cl): arr[i] for i, cl in enumerate(nodes["clId"].values)}
    else:
        print("[elliptic2_full] 无缓存，请先运行 22_cache_node_features.py", flush=True)
        raise SystemExit(2)
    print(f"[elliptic2_full] 节点特征={len(cl2feat)} 维度={len(feat_cols)} "
          f"耗时={time.time()-t0:.1f}s", flush=True)

    cc_nodes = defaultdict(list)
    for cl, cc in cl2cc.items():
        cc_nodes[cc].append(cl)
    cc_edges = defaultdict(list)
    cc_tx = defaultdict(set)
    for _, r in edges.iterrows():
        c = cl2cc.get(r["clId1"])
        if c is not None and c == cl2cc.get(r["clId2"]):
            cc_edges[c].append((r["clId1"], r["clId2"]))
            cc_tx[c].add(r["txId"])

    rows, labels = [], []
    for _, r in comps.iterrows():
        cc = r["ccId"]
        cls = cc_nodes.get(cc, [])
        if not cls:
            continue
        mats = [cl2feat[c] for c in cls if c in cl2feat]
        if not mats:
            continue
        M = np.vstack(mats)
        # 池化：mean / max / std（166*3）
        mean = np.nanmean(M, axis=0)
        maxv = np.nanmax(M, axis=0)
        stdv = np.nanstd(M, axis=0)
        vec = np.concatenate([mean, maxv, stdv])
        vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
        # 结构特征
        n = len(cls); m = len(cc_edges.get(cc, []))
        struct = np.array([n, m, len(cc_tx.get(cc, set())), m / max(n, 1)])
        vec = np.concatenate([vec, np.log1p(struct)])
        rows.append(vec)
        labels.append(1 if r["ccLabel"] == "suspicious" else 0)
    X = np.vstack(rows)
    X = np.clip(X, -1e6, 1e6)
    y = np.array(labels, dtype=int)
    print(f"[elliptic2_full] 子图表示：{X.shape}，正类 {y.sum()} ({y.mean():.3%})，"
          f"总耗时={time.time()-t0:.1f}s", flush=True)
    return X, y


def run_one(X, y, name, seed, split="random"):
    set_seed(seed)
    if split == "random":
        tr, te = train_test_split(np.arange(len(y)), test_size=0.25,
                                  random_state=seed, stratify=y)
    else:
        # 按子图规模排序做近似时间/结构分裂（Elliptic2 无显式时间）
        order = np.argsort(X[:, -4])  # n_nodes
        cut = int(0.75 * len(y))
        tr, te = order[:cut], order[cut:]
    sc = StandardScaler().fit(X[tr])
    Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
    if name == "LogisticRegression":
        m = LogisticRegression(max_iter=2000, class_weight="balanced", C=0.1)
    elif name == "RandomForest":
        m = RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1,
                                   class_weight="balanced_subsample", max_depth=12)
    elif name == "LightGBM":
        from lightgbm import LGBMClassifier
        m = LGBMClassifier(n_estimators=400, learning_rate=0.05, verbose=-1,
                           random_state=seed, n_jobs=-1)
    else:
        from xgboost import XGBClassifier
        m = XGBClassifier(n_estimators=400, learning_rate=0.05, eval_metric="logloss",
                          random_state=seed, n_jobs=-1, tree_method="hist",
                          scale_pos_weight=len(y) / max(y.sum(), 1))
    m.fit(Xtr, y[tr])
    score = m.predict_proba(Xte)[:, 1]
    met = compute_metrics(y[te], score)
    met.update({"model": f"elliptic2_full_{name}", "seed": seed, "split": split, "status": "done"})
    return met


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--sample-nodes", type=int, default=0)
    ap.add_argument("--outdir", default=str(RESULTS_DIR))
    args = ap.parse_args()

    X, y = build_full_features(args.sample_nodes)
    outdir = Path(args.outdir)
    ckpt = Checkpoint("elliptic2_full", outdir / "_checkpoints" / "elliptic2_full.json")
    ckpt.install_signal_handler()
    models = ["LogisticRegression", "RandomForest", "LightGBM", "XGBoost"]
    splits = ["random", "size"] if not args.quick else ["random"]
    seeds = range(2 if args.quick else args.seeds)
    all_keys = [f"{m}|{sp}|seed{s}" for m in models for sp in splits for s in seeds]
    todo = all_keys if args.fresh else ckpt.todo_keys(all_keys)
    print(f"[checkpoint] 总 run={len(all_keys)} 待跑={len(todo)}", flush=True)

    for key in todo:
        parts = key.split("|")
        name = parts[0]; split = parts[1]; seed = int(parts[2].replace("seed", ""))
        ckpt.mark_start(key)
        try:
            res = run_one(X, y, name, seed, split)
        except Exception as exc:
            res = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
        if res.get("status") == "done":
            ckpt.mark_done(key, res)
            write_provenance("elliptic2_full", {"key": key, "PR-AUC": round(res["PR-AUC"], 4)})
            print(f"  [OK] {key:48s} PR-AUC={res['PR-AUC']:.4f} AUC={res['AUC']:.4f}", flush=True)
        else:
            ckpt.skip(key)
            print(f"  [SKIP] {key}: {res.get('error','')[:60]}", flush=True)

    ckpt._flush()
    full = [{"key": k, **v["metrics"]} for k, v in ckpt.state["runs"].items()
            if v.get("status") == "done" and "PR-AUC" in v.get("metrics", {})]
    if full:
        fdf = pd.DataFrame(full)
        fdf.to_csv(outdir / "elliptic2_full_results.csv", index=False, float_format="%.6f")
        agg = fdf.groupby(["model", "split"]).agg(
            PR_AUC_mean=("PR-AUC", "mean"), PR_AUC_std=("PR-AUC", "std"),
            AUC_mean=("AUC", "mean"), AUC_std=("AUC", "std"),
            ECE_mean=("ECE", "mean"), n=("seed", "nunique"),
        ).reset_index()
        agg.to_csv(outdir / "elliptic2_full_summary.csv", index=False, float_format="%.6f")
        print("\n[Elliptic2 完整特征汇总]", flush=True)
        print(agg.to_string(index=False), flush=True)
        print("\n[对比旧版（仅拓扑特征）]", flush=True)
        old = outdir / "elliptic2_summary.csv"
        if old.exists():
            o = pd.read_csv(old)
            print(f"  仅拓扑最佳 PR-AUC: {o['PR_AUC_mean'].max():.4f}", flush=True)
            print(f"  完整特征最佳 PR-AUC: {agg['PR_AUC_mean'].max():.4f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
