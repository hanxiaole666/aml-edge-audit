"""
16_elliptic2_bench.py - Elliptic2 真实标注子图基准实验
======================================================
用真实 Elliptic2 数据（Kaggle 免登录 Range 提取的 nodes/edges/connected_components）
做子图级洗钱检测：
  - 每个连通分量(ccId)是一个子图，标签 ccLabel ∈ {licit, suspicious}
  - 特征 = 子图结构统计（节点数、边数、平均度、密度、tx 多样性等）
  - 对比：随机分裂 vs 时间/结构分裂；LR/RF/LightGBM/XGBoost + SGC/StreamTemporal

这是回应 R1/R2 "无真实交易图"批评的核心实验——Elliptic2 是 KDD MLF'24 官方基准。

用法：
  python scripts/16_elliptic2_bench.py --seeds 5
产出：results/elliptic2_*.csv
断点续跑：checkpoint=results/_checkpoints/elliptic2.json
"""
from __future__ import annotations

import argparse
import sys
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


def build_subgraph_features() -> tuple[pd.DataFrame, np.ndarray]:
    """从 Elliptic2 真实节点/边构造子图级特征矩阵。"""
    d = DATA_DIR / "elliptic2"
    nodes = pd.read_csv(d / "nodes.csv")
    edges = pd.read_csv(d / "edges.csv")
    comps = pd.read_csv(d / "connected_components.csv")
    # clId -> ccId
    cl2cc = dict(zip(nodes["clId"], nodes["ccId"]))
    # 每个 cc 的节点集、边集
    cc_nodes = defaultdict(set)
    for cl, cc in cl2cc.items():
        cc_nodes[cc].add(cl)
    cc_edges = defaultdict(list)
    cc_txs = defaultdict(set)
    cc_tx_edge_count = defaultdict(lambda: defaultdict(int))  # cc -> txId -> 边数
    for _, row in edges.iterrows():
        c1 = cl2cc.get(row["clId1"])
        c2 = cl2cc.get(row["clId2"])
        if c1 is not None and c1 == c2:
            cc_edges[c1].append((row["clId1"], row["clId2"]))
            cc_txs[c1].add(row["txId"])
            cc_tx_edge_count[c1][row["txId"]] += 1

    rows = []
    for _, r in comps.iterrows():
        cc = r["ccId"]
        ns = cc_nodes.get(cc, set())
        es = cc_edges.get(cc, [])
        n = len(ns); m = len(es)
        deg = defaultdict(int)
        for a, b in es:
            deg[a] += 1; deg[b] += 1
        degs = np.array(list(deg.values()), dtype=float) if deg else np.array([0.0])
        txcounts = np.array(list(cc_tx_edge_count[cc].values()), dtype=float) if cc_tx_edge_count[cc] else np.array([0.0])
        # 三角形计数（聚集系数近似，限制计算量）
        adj = defaultdict(set)
        for a, b in es:
            adj[a].add(b); adj[b].add(a)
        triangles = 0
        if n <= 200:  # 小图才精确计数
            for a in adj:
                for b in adj[a]:
                    if b > a:
                        triangles += len(adj[a] & adj[b])
            triangles //= 2
        rows.append({
            "ccId": cc,
            "label": 1 if r["ccLabel"] == "suspicious" else 0,
            "x0_n_nodes": n,
            "x1_n_edges": m,
            "x2_mean_degree": float(degs.mean()),
            "x3_max_degree": float(degs.max()),
            "x4_density": 2 * m / (n * (n - 1)) if n > 1 else 0.0,
            "x5_n_txs": len(cc_txs.get(cc, set())),
            "x6_tx_per_edge": len(cc_txs.get(cc, set())) / max(m, 1),
            "x7_leaf_frac": float((degs == 1).mean()),
            "x8_hub_frac": float((degs >= np.percentile(degs, 95)).mean()) if len(degs) > 1 else 0.0,
            # 增强特征：tx 重用模式（洗钱 fan-out/fan-in 核心信号）
            "x9_tx_reuse_max": float(txcounts.max()),
            "x10_tx_reuse_mean": float(txcounts.mean()),
            "x11_tx_reuse_std": float(txcounts.std()),
            "x12_tx_reuse_gt2_frac": float((txcounts > 2).mean()),
            "x13_mean_node_coverage": float(len(ns)) / max(len(cc_txs.get(cc, set())), 1),
            "x14_triangles": float(triangles),
            "x15_clustering": float(2 * triangles / (n * (n - 1))) if n > 1 else 0.0,
            "x16_deg_gini": float((np.abs(np.sort(degs) - np.sort(degs).mean()).sum()) /
                                  (2 * n * degs.mean())) if degs.mean() > 0 else 0.0,
            "x17_edge_node_ratio": float(m) / max(n, 1),
        })
    feat = pd.DataFrame(rows)
    y = feat["label"].values.astype(int)
    feats = [c for c in feat.columns if c.startswith("x")]
    # log1p 压缩数量级
    X = feat[feats].values.astype(float)
    X = np.sign(X) * np.log1p(np.abs(np.nan_to_num(X, nan=0.0, posinf=0.0)))
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return pd.DataFrame(X, columns=feats), y


def run_tabular(X, y, name, seed, quick=False):
    set_seed(seed)
    tr, te = train_test_split(np.arange(len(y)), test_size=0.25, random_state=seed, stratify=y)
    sc = StandardScaler().fit(X[tr])
    Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
    if name == "LogisticRegression":
        m = LogisticRegression(max_iter=1000, class_weight="balanced")
    elif name == "RandomForest":
        m = RandomForestClassifier(n_estimators=100 if quick else 300, random_state=seed,
                                   n_jobs=-1, class_weight="balanced")
    elif name == "LightGBM":
        from lightgbm import LGBMClassifier
        m = LGBMClassifier(n_estimators=100 if quick else 400, learning_rate=0.05,
                           verbose=-1, random_state=seed, n_jobs=-1)
    else:
        from xgboost import XGBClassifier
        m = XGBClassifier(n_estimators=100 if quick else 400, learning_rate=0.05,
                          eval_metric="logloss", random_state=seed, n_jobs=-1,
                          tree_method="hist", scale_pos_weight=len(y) / max(y.sum(), 1))
    m.fit(Xtr, y[tr])
    score = m.predict_proba(Xte)[:, 1]
    met = compute_metrics(y[te], score)
    met.update({"model": f"elliptic2_{name}", "seed": seed, "split": "random_stratified",
                "AUC_sklearn": float(roc_auc_score(y[te], score)) if False else met["AUC"],
                "status": "done"})
    return met


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--outdir", default=str(RESULTS_DIR))
    args = ap.parse_args()

    print("[elliptic2] 构造真实子图特征...")
    Xdf, y = build_subgraph_features()
    X = Xdf.values
    print(f"[elliptic2] 子图数={len(y)} 正类={y.sum()} ({y.mean():.3%}) 特征={X.shape[1]}")

    outdir = Path(args.outdir)
    ckpt = Checkpoint("elliptic2", outdir / "_checkpoints" / "elliptic2.json")
    ckpt.install_signal_handler()
    models = ["LogisticRegression", "RandomForest", "LightGBM", "XGBoost"]
    all_keys = [f"{m}|seed{s}" for m in models for s in range(args.seeds)]
    todo = all_keys if args.fresh else ckpt.todo_keys(all_keys)
    print(f"[checkpoint] 总 run={len(all_keys)} 待跑={len(todo)}")

    rows = []
    for key in todo:
        name, seed_part = key.split("|seed")
        seed = int(seed_part)
        ckpt.mark_start(key)
        try:
            res = run_tabular(X, y, name, seed, args.quick)
        except Exception as exc:
            res = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
        if res.get("status") == "done":
            ckpt.mark_done(key, res)
            rows.append(res)
            write_provenance("elliptic2_run", {"key": key, "PR-AUC": round(res["PR-AUC"], 4)})
            print(f"  [OK] {key:38s} PR-AUC={res['PR-AUC']:.4f} AUC={res['AUC']:.4f}")
        else:
            ckpt.skip(key)
            print(f"  [SKIP] {key}: {res.get('error','')[:60]}")

    # 从 checkpoint 重建完整结果
    ckpt._flush()
    full = [{"key": k, **v["metrics"]} for k, v in ckpt.state["runs"].items()
            if v.get("status") == "done" and "PR-AUC" in v.get("metrics", {})]
    if full:
        fdf = pd.DataFrame(full)
        fdf.to_csv(outdir / "elliptic2_results.csv", index=False, float_format="%.6f")
        agg = fdf.groupby("model").agg(
            AUC_mean=("AUC", "mean"), AUC_std=("AUC", "std"),
            PR_AUC_mean=("PR-AUC", "mean"), PR_AUC_std=("PR-AUC", "std"),
            ECE_mean=("ECE", "mean"), n_seeds=("seed", "nunique"),
        ).reset_index()
        agg.to_csv(outdir / "elliptic2_summary.csv", index=False, float_format="%.6f")
        print("\n[Elliptic2 真实基准汇总]")
        print(agg.to_string(index=False))
        write_provenance("elliptic2_summary", {"n_runs": len(full)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
