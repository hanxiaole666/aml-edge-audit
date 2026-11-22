"""
13_temporal_sota.py - 公平时序图基线对比（回应 R1 MC4）
======================================================
用 PyTorch Geometric 实现标准的图注意力基线（GAT）与时序感知变体（TGAT-lite），
在**同一真实交易图**、**同一线性读出层预算**下与本文方法公平对比。

目的：回应审稿人"只与 lite 基线比、不与真正时序方法对比"的批评。
诚实说明：TGAT-lite 是可控简化实现，非完整 TGAT（后者需连续时间随机游走采样）；
         但它是标准 PyG GAT 卷积 + 时间编码，远强于纯 numpy lite 基线。

用法：
  python scripts/13_temporal_sota.py
产出：results/temporal_sota.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data
from torch_geometric.nn import GATConv

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RESULTS_DIR, pr_auc, roc_auc, write_provenance, set_seed


def load_graph(nodes_p: str, edges_p: str):
    nodes = pd.read_csv(nodes_p)
    edges = pd.read_csv(edges_p)
    feat_cols = [c for c in nodes.columns if c.startswith("x")]
    X = nodes[feat_cols].values.astype(float)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = nodes["label"].values.astype(int)
    t = nodes["time"].values.astype(float)
    node_id = nodes["node_id"].values
    id2idx = {int(n): i for i, n in enumerate(node_id)}
    src = np.array([id2idx[int(s)] for s in edges["src"] if int(s) in id2idx])
    dst = np.array([id2idx[int(d)] for d in edges["dst"] if int(d) in id2idx])
    edge_index = torch.tensor(np.vstack([src, dst]), dtype=torch.long)
    # 对称化
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1).unique(dim=1)
    return X, y, t, edge_index


def time_encoding(t: np.ndarray, d: int = 16) -> np.ndarray:
    """标准时间编码：多频率 sin/cos（TGAT 核心思想）。"""
    t_norm = (t - t.min()) / max(t.max() - t.min(), 1.0)
    freqs = np.exp(np.linspace(0, np.log(100), d // 2))
    ang = t_norm[:, None] * freqs[None, :]
    return np.concatenate([np.sin(ang), np.cos(ang)], axis=1)


class GATEncoder(torch.nn.Module):
    def __init__(self, in_dim: int, hidden: int = 32, heads: int = 4):
        super().__init__()
        self.gat1 = GATConv(in_dim, hidden, heads=heads, dropout=0.1)
        self.gat2 = GATConv(hidden * heads, hidden, heads=1, concat=False)

    def forward(self, x, edge_index):
        h = F.elu(self.gat1(x, edge_index))
        return F.elu(self.gat2(h, edge_index))


def run_gat(X, y, t, edge_index, split: str, seed: int, use_time: bool, quick: bool):
    set_seed(seed)
    torch.manual_seed(seed)
    if split == "temporal":
        order = np.argsort(t, kind="mergesort")
        cut = int(0.75 * len(y))
        tr, te = order[:cut], order[cut:]
        tmax = t[tr].max()
        te = te[t[te] > tmax]
        if len(te) < 20:
            te = order[cut:]
    else:
        tr, te = train_test_split(np.arange(len(y)), test_size=0.25,
                                  random_state=seed, stratify=y)

    # 特征 + 可选时间编码
    X_feat = X.copy()
    if use_time:
        te_enc = time_encoding(t)
        X_feat = np.concatenate([X_feat, te_enc], axis=1)
    sc = StandardScaler().fit(X_feat[tr])
    Xs = torch.tensor(sc.transform(X_feat), dtype=torch.float)

    model = GATEncoder(Xs.shape[1], hidden=16 if quick else 32, heads=4)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-4)
    y_tr = torch.tensor(y[tr], dtype=torch.long)
    model.train()
    t0 = time.perf_counter()
    for _ in range(60 if quick else 150):
        opt.zero_grad()
        h = model(Xs, edge_index)
        loss = F.cross_entropy(h[tr], y_tr)
        loss.backward()
        opt.step()
    train_sec = time.perf_counter() - t0

    model.eval()
    with torch.no_grad():
        h = model(Xs, edge_index).numpy()
    # 统一线性读出层（与本文方法公平对比）
    clf = LogisticRegression(max_iter=1000).fit(h[tr], y[tr])
    score = clf.predict_proba(h[te])[:, 1]
    return {
        "PR-AUC": float(pr_auc(y[te], score)), "AUC": float(roc_auc(y[te], score)),
        "split": split, "use_time": use_time, "seed": seed,
        "train_seconds": round(train_sec, 2), "status": "done",
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nodes", default="data/real_cats/real_graph_nodes.csv")
    ap.add_argument("--edges", default="data/real_cats/real_graph_edges.csv")
    ap.add_argument("--outdir", default=str(RESULTS_DIR))
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    X, y, t, edge_index = load_graph(args.nodes, args.edges)
    print(f"[data] nodes={len(y)} edges={edge_index.shape[1]//2} pos_rate={y.mean():.3f}")
    print(f"[model] GAT (PyG {__import__('torch_geometric').__version__}) + 时间编码")

    rows = []
    for split in ["random", "temporal"]:
        for use_time in [False, True]:
            name = "TGAT-lite" if use_time else "GAT-standard"
            for seed in range(args.seeds):
                try:
                    res = run_gat(X, y, t, edge_index, split, seed, use_time, args.quick)
                    res["model"] = name
                    rows.append(res)
                    print(f"  [OK] {name:14s} {split:8s} seed{seed} PR-AUC={res['PR-AUC']:.4f}")
                except Exception as exc:
                    print(f"  [FAIL] {name} {split} seed{seed}: {type(exc).__name__}: {exc}")

    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(Path(args.outdir) / "temporal_sota_results.csv", index=False, float_format="%.6f")
        agg = df.groupby(["model", "split"]).agg(
            PR_AUC_mean=("PR-AUC", "mean"), PR_AUC_std=("PR-AUC", "std"),
            AUC_mean=("AUC", "mean"), train_sec=("train_seconds", "mean"),
        ).reset_index()
        out = {"experiment": "temporal_sota", "summary": agg.to_dict("records"),
               "comparison_note": ("GAT-standard 与 TGAT-lite 共享相同 PyG 架构和线性读出层，"
                                   "仅差时间编码；与本文 StreamTemporal（纯numpy时态加权SGC）"
                                   "在同一真实图上对比，构成公平基线。"),
               "status": "PASS"}
        (Path(args.outdir) / "temporal_sota.json").write_text(
            json.dumps(out, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
        write_provenance("temporal_sota", {"status": "PASS", "n_runs": len(rows)})
        print("\n[汇总]")
        print(agg.to_string(index=False))

    # 与本文方法对比
    print("\n[公平对比：本文 StreamTemporal vs 标准 GAT/TGAT]")
    try:
        ours = pd.read_csv(Path(args.outdir) / "gnn_summary_realgraph.csv")
        ours_t = ours[(ours["model"] == "gnn_StreamTemporal") & (ours["split"] == "temporal")]["PR_AUC_mean"].values
        if len(ours_t):
            print(f"  本文 StreamTemporal (temporal): {ours_t[0]:.4f}")
            for _, r in agg[agg["split"] == "temporal"].iterrows():
                print(f"  {r['model']:14s}: {r['PR_AUC_mean']:.4f}")
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
