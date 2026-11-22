"""
17_elliptic2_gnn.py - Elliptic2 子图表示学习（图神经网络池化）
============================================================
回应手工特征 AUC~0.52 的不足：用 Elliptic2 论文的思路——子图表示学习。
方法：对每个小子图建图 → SGC 平滑 → mean/sum 池化 → 子图向量 → LR 分类。
对比手工结构特征（16 脚本）与学习表示（本脚本）。

用法：python scripts/17_elliptic2_gnn.py --seeds 5
产出：results/elliptic2_gnn_results.csv
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    DATA_DIR, RESULTS_DIR, Checkpoint, pr_auc, roc_auc, compute_metrics,
    write_provenance, set_seed,
)


def build_subgraph_embeddings(max_subgraphs: int = 20000):
    """用 SGC 平滑 + 池化构造子图表示。"""
    d = DATA_DIR / "elliptic2"
    nodes = pd.read_csv(d / "nodes.csv")
    edges = pd.read_csv(d / "edges.csv")
    comps = pd.read_csv(d / "connected_components.csv")
    cl2cc = dict(zip(nodes["clId"], nodes["ccId"]))

    # 每个 cc 的节点列表
    cc_nodes = defaultdict(list)
    for cl, cc in cl2cc.items():
        cc_nodes[cc].append(cl)
    # 每个 cc 的边
    cc_edges = defaultdict(list)
    for _, row in edges.iterrows():
        c = cl2cc.get(row["clId1"])
        if c is not None and c == cl2cc.get(row["clId2"]):
            cc_edges[c].append((row["clId1"], row["clId2"]))
    # 全局节点索引（用于 txId 作为节点特征）
    all_cl = nodes["clId"].values
    cl2idx = {c: i for i, c in enumerate(all_cl)}
    N = len(all_cl)
    # 节点初始特征：度 + txId hash 编码（用边数近似）
    deg = np.ones(N, dtype=float)
    for _, row in edges.iterrows():
        a, b = cl2idx.get(row["clId1"]), cl2idx.get(row["clId2"])
        if a is not None: deg[a] += 1
        if b is not None: deg[b] += 1
    X0 = np.log1p(deg).reshape(-1, 1)

    embs, labels, ccs = [], [], []
    for _, r in comps.iterrows():
        cc = r["ccId"]
        cls = cc_nodes.get(cc, [])
        if len(cls) < 2:
            continue
        idx = [cl2idx[c] for c in cls]
        local = {g: i for i, g in enumerate(idx)}
        es = cc_edges.get(cc, [])
        n = len(idx)
        A = np.zeros((n, n), dtype=float)
        for a, b in es:
            ia, ib = local.get(a), local.get(b)
            if ia is not None and ib is not None:
                A[ia, ib] = A[ib, ia] = 1.0
        Ada = A + np.eye(n)
        dgr = Ada.sum(axis=1)
        Ahat = (1 / np.sqrt(np.maximum(dgr, 1e-9)))[:, None] * Ada * (1 / np.sqrt(np.maximum(dgr, 1e-9)))[None, :]
        H = X0[idx]  # (n,1)
        # 2 跳 SGC 平滑
        for _ in range(2):
            H = Ahat @ H
        mean_p = H.mean(axis=0)
        max_p = H.max(axis=0)
        row = np.concatenate([mean_p, max_p, [n, len(es), A.mean()]])
        embs.append(row)
        labels.append(1 if r["ccLabel"] == "suspicious" else 0)
        ccs.append(cc)
        if len(embs) >= max_subgraphs:
            break
    return np.array(embs), np.array(labels), ccs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--max-subgraphs", type=int, default=20000)
    ap.add_argument("--outdir", default=str(RESULTS_DIR))
    args = ap.parse_args()

    print("[elliptic2-gnn] 构造子图表示（SGC 平滑 + 池化）...")
    X, y, ccs = build_subgraph_embeddings(args.max_subgraphs)
    print(f"[elliptic2-gnn] 子图={len(y)} 正类={y.sum()} ({y.mean():.3%}) 表示维度={X.shape[1]}")

    outdir = Path(args.outdir)
    ckpt = Checkpoint("elliptic2_gnn", outdir / "_checkpoints" / "elliptic2_gnn.json")
    ckpt.install_signal_handler()
    all_keys = [f"SGC-pool|seed{s}" for s in range(args.seeds)]
    todo = all_keys if args.fresh else ckpt.todo_keys(all_keys)

    rows = []
    for key in todo:
        seed = int(key.split("seed")[1])
        ckpt.mark_start(key)
        set_seed(seed)
        try:
            tr, te = train_test_split(np.arange(len(y)), test_size=0.25, random_state=seed, stratify=y)
            sc = StandardScaler().fit(X[tr])
            clf = LogisticRegression(max_iter=1000, class_weight="balanced")
            clf.fit(sc.transform(X[tr]), y[tr])
            score = clf.predict_proba(sc.transform(X[te]))[:, 1]
            met = compute_metrics(y[te], score)
            met.update({"model": "elliptic2_SGC-pool", "seed": seed, "status": "done"})
        except Exception as exc:
            met = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
        if met.get("status") == "done":
            ckpt.mark_done(key, met)
            rows.append(met)
            write_provenance("elliptic2_gnn", {"key": key, "PR-AUC": round(met["PR-AUC"], 4)})
            print(f"  [OK] {key:24s} PR-AUC={met['PR-AUC']:.4f} AUC={met['AUC']:.4f}")
        else:
            ckpt.skip(key)
            print(f"  [SKIP] {key}: {met.get('error','')[:60]}")

    ckpt._flush()
    full = [{"key": k, **v["metrics"]} for k, v in ckpt.state["runs"].items()
            if v.get("status") == "done" and "PR-AUC" in v.get("metrics", {})]
    if full:
        fdf = pd.DataFrame(full)
        fdf.to_csv(outdir / "elliptic2_gnn_results.csv", index=False, float_format="%.6f")
        print("\n[Elliptic2 子图表示学习汇总]")
        print(fdf[["PR-AUC", "AUC", "ECE"]].agg(["mean", "std"]).to_string())
        # 对比手工特征
        manual = outdir / "elliptic2_summary.csv"
        if manual.exists():
            m = pd.read_csv(manual)
            print("\n[对比：手工结构特征 vs 学习表示]")
            print(f"  手工特征最佳 PR-AUC: {m['PR_AUC_mean'].max():.4f}")
            print(f"  学习表示 PR-AUC: {fdf['PR-AUC'].mean():.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
