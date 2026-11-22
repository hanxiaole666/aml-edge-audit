"""
03_train_gnn.py - 轻量图轨道：SGC / GCN-lite / SAGE-lite / StreamTemporal-GNN
=============================================================================
纯 numpy 实现（无需 torch / torch-geometric），任何环境可跑。
支持两种评估协议：
  - random:  随机分裂（基线，会高估性能）
  - temporal: 严格时间分裂（train.max_time < test.min_time），对应流式协议

断点续跑：每个 (model, split, seed, k) 独立 run，checkpoint=results/_checkpoints/gnn.json
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    RESULTS_DIR, Checkpoint, compute_metrics, set_seed, write_provenance, sigmoid,
)

MODELS = ["SGC", "GCN-lite", "SAGE-lite", "StreamTemporal"]


def build_graph(nodes: pd.DataFrame, edges: pd.DataFrame):
    """构建归一化邻接矩阵 A_hat 与特征矩阵 X、标签 y、时间 t。"""
    node_ids = nodes["node_id"].values
    id2idx = {int(nid): i for i, nid in enumerate(node_ids)}
    n = len(node_ids)
    feat_cols = [c for c in nodes.columns if c.startswith("x")]
    X = nodes[feat_cols].values.astype(float)
    y = nodes["label"].values.astype(int)
    t = nodes["time"].values.astype(float) if "time" in nodes.columns else np.zeros(n)

    A = np.zeros((n, n), dtype=float)
    for s, d in zip(edges["src"].values, edges["dst"].values):
        i, j = id2idx.get(int(s)), id2idx.get(int(d))
        if i is not None and j is not None:
            A[i, j] = 1.0
            A[j, i] = 1.0
    # 对称归一化 A_hat = D^-1/2 (A+I) D^-1/2
    A_hat = A + np.eye(n)
    deg = A_hat.sum(axis=1)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(deg, 1e-9)))
    A_hat = D_inv_sqrt @ A_hat @ D_inv_sqrt
    return A_hat, X, y, t


def temporal_weights(edges: pd.DataFrame, nodes: pd.DataFrame, lam: float, t_now: float) -> np.ndarray:
    """StreamTemporal 核心：时态边权 w(u,v,t)=exp(-λ(t_now - t_edge))。"""
    node_ids = nodes["node_id"].values
    id2idx = {int(nid): i for i, nid in enumerate(node_ids)}
    n = len(node_ids)
    W = np.zeros((n, n), dtype=float)
    for s, d, te in zip(edges["src"].values, edges["dst"].values,
                        edges["time"].values if "time" in edges.columns else np.zeros(len(edges))):
        i, j = id2idx.get(int(s)), id2idx.get(int(d))
        if i is not None and j is not None:
            w = np.exp(-lam * max(0.0, t_now - float(te)))
            W[i, j] = w
            W[j, i] = w
    return W + np.eye(n)


def gcn_layer(A: np.ndarray, H: np.ndarray, Wt: np.ndarray) -> np.ndarray:
    return sigmoid(A @ H @ Wt)


def propagate(A: np.ndarray, X: np.ndarray, k: int, model: str,
              rng: np.random.Generator, A_temporal: np.ndarray | None = None) -> np.ndarray:
    """k 跳传播，返回节点表示。A_temporal 为 StreamTemporal 的时态加权邻接。"""
    d = X.shape[1]
    A_use = A_temporal if (model == "StreamTemporal" and A_temporal is not None) else A
    if model in ("SGC", "StreamTemporal"):
        H = X
        for _ in range(k):
            H = A_use @ H                      # SGC: k 跳平滑
        return H                                # 读出层用 LogisticRegression 训练
    if model == "GCN-lite":
        H = X
        Wt = rng.normal(0, np.sqrt(2.0 / d), (d, d))
        for _ in range(k):
            # 去掉 sigmoid 饱和，改 ReLU 更稳（轻量 GCN 常见做法）
            H = np.maximum(0.0, A_use @ H @ Wt)
        return H
    if model == "SAGE-lite":
        H = X
        Wt_self = rng.normal(0, np.sqrt(1.0 / d), (d, d))
        Wt_neigh = rng.normal(0, np.sqrt(1.0 / d), (d, d))
        for _ in range(k):
            agg = A_use @ H
            H = np.maximum(0.0, H @ Wt_self + agg @ Wt_neigh)
        return H
    raise ValueError(model)


def split_indices(y: np.ndarray, t: np.ndarray, split: str, seed: int):
    rng = np.random.default_rng(seed)
    n = len(y)
    if split == "temporal":
        # 严格时间分裂：训练 70% 最早时间，测试 30% 最晚时间
        order = np.argsort(t, kind="mergesort")
        cut = int(0.7 * n)
        tr, te = order[:cut], order[cut:]
        if t[tr].max() >= t[te].min():
            # 处理时间并列：把并列时间的节点全部划给训练，测试只取严格更晚的
            tmax = t[tr].max()
            te = te[t[te] > tmax]
            tr = np.arange(n)[t <= tmax]
        return tr, te
    # random stratified
    idx = np.arange(n)
    tr_parts, te_parts = [], []
    for c in np.unique(y):
        ci = idx[y == c]
        rng.shuffle(ci)
        cut = int(0.7 * len(ci))
        tr_parts.append(ci[:cut]); te_parts.append(ci[cut:])
    return np.concatenate(tr_parts), np.concatenate(te_parts)


def run_one(nodes, edges, model_name, split, seed, k, quick) -> dict:
    set_seed(seed)
    A, X, y, t = build_graph(nodes, edges)
    tr, te = split_indices(y, t, split, seed)
    if len(te) < 10:
        return {"status": "failed", "error": f"测试集过小 n_test={len(te)}"}

    rng = np.random.default_rng(seed)
    # StreamTemporal：用时态边权构造 A_temporal = D^-1/2 W D^-1/2
    A_temporal = None
    if model_name == "StreamTemporal":
        t_now = float(t.max())
        Wt_mat = temporal_weights(edges, nodes, lam=0.03, t_now=t_now)
        deg_t = Wt_mat.sum(axis=1)
        Dt = np.diag(1.0 / np.sqrt(np.maximum(deg_t, 1e-9)))
        A_temporal = Dt @ Wt_mat @ Dt

    H = propagate(A, X, k, model_name, rng, A_temporal=A_temporal)

    # 真正的线性读出层：训练集上逻辑回归（对应定理 B 的"线性读出层"限定）
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit(H[tr])
    H_tr, H_te = scaler.transform(H[tr]), scaler.transform(H[te])
    clf = LogisticRegression(max_iter=1000, C=1.0)
    clf.fit(H_tr, y[tr])
    scores_te = clf.predict_proba(H_te)[:, 1]

    met = compute_metrics(y[te], scores_te)
    met.update({
        "model": f"gnn_{model_name}", "split": split, "seed": seed, "k": k,
        "n_train": len(tr), "n_test": len(te), "status": "done",
    })
    return met


def build_knn_graph_from_table(csv_path: str, n_sample: int = 3000, k: int = 8):
    """从真实表格数据构造 KNN 相似图（方法学诚实：非原始交易图）。

    节点 = 地址（采样 n_sample 个），边 = 特征空间 k 近邻。
    用于 Real-CATS 这类只有特征、无显式交易边的数据集上的图模型实验。
    """
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import StandardScaler
    df = pd.read_csv(csv_path)
    if len(df) > n_sample:
        df = df.sample(n=n_sample, random_state=42).reset_index(drop=True)
    exclude = {"label", "time", "address", "etherscan_checked", "watchback_checked", "gs_checked"}
    feats = [c for c in df.columns if c not in exclude and
             pd.api.types.is_numeric_dtype(df[c])]
    X = df[feats].values.astype(float)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    X = StandardScaler().fit_transform(X)
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X)
    _, idx = nn.kneighbors(X)
    nodes = pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])])
    nodes.insert(0, "node_id", range(len(df)))
    nodes["label"] = df["label"].values
    nodes["time"] = df["time"].values
    edges = []
    times = df["time"].values if "time" in df.columns else np.zeros(len(df))
    for i, nbrs in enumerate(idx):
        for j in nbrs[1:]:  # 跳过自身
            j = int(j)
            # 边时间 = 两端地址首次交易时间的较大者（近似"关系确立时间"）
            edge_t = max(float(times[i]), float(times[j])) if np.isfinite(times[i]) and np.isfinite(times[j]) else 0.0
            edges.append((i, j, edge_t))
    ed = pd.DataFrame(edges, columns=["src", "dst", "time"]).drop_duplicates()
    return nodes, ed


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nodes", default="data/synthetic_graph_nodes.csv")
    ap.add_argument("--edges", default="data/synthetic_graph_edges.csv")
    ap.add_argument("--from-table", default="", help="从真实表格构造 KNN 图（如 Real-CATS）")
    ap.add_argument("--tag", default="", help="数据集标签，区分 checkpoint")
    ap.add_argument("--n-sample", type=int, default=3000, help="KNN 图采样节点数")
    ap.add_argument("--outdir", default=str(RESULTS_DIR))
    ap.add_argument("--models", nargs="*", default=MODELS)
    ap.add_argument("--splits", nargs="*", default=["random", "temporal"])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--k", type=int, nargs="*", default=[1, 2])
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--fresh", action="store_true")
    args = ap.parse_args()

    if args.from_table:
        print(f"[data] 从真实表格构造 KNN 图：{args.from_table}（非原始交易图，论文需标注）")
        nodes, edges = build_knn_graph_from_table(args.from_table, n_sample=args.n_sample)
    else:
        nodes = pd.read_csv(args.nodes)
        edges = pd.read_csv(args.edges)
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    tag = args.tag or ("knn" if args.from_table else "synthetic")
    ckpt = Checkpoint(f"gnn_{tag}", outdir / "_checkpoints" / f"gnn_{tag}.json")
    ckpt.install_signal_handler()
    print(f"[data] nodes={len(nodes)} edges={len(edges)} pos_rate={nodes['label'].mean():.3f}")

    all_keys = [f"{m}|{sp}|seed{s}|k{k}" for m in args.models for sp in args.splits
                for s in range(args.seeds) for k in args.k]
    todo = all_keys if args.fresh else ckpt.todo_keys(all_keys)
    print(f"[checkpoint] 总 run={len(all_keys)} 待跑={len(todo)} 已完成={len(all_keys)-len(todo)}")

    rows = []
    for key in todo:
        parts = key.split("|")
        model_name, split, seed_part, k_part = parts
        seed = int(seed_part.replace("seed", "")); k = int(k_part.replace("k", ""))
        ckpt.mark_start(key)
        try:
            res = run_one(nodes, edges, model_name, split, seed, k, args.quick)
        except Exception as exc:
            res = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
        if res.get("status") == "done":
            ckpt.mark_done(key, {"PR-AUC": round(res["PR-AUC"], 4)})
            rows.append(res)
            print(f"  [OK] {key:42s} PR-AUC={res['PR-AUC']:.4f} AUC={res['AUC']:.4f}")
        else:
            ckpt.skip(key)
            print(f"  [SKIP] {key:42s} {res.get('error','')[:60]}")

    if rows:
        rdf = pd.DataFrame(rows)
        rdf["dataset"] = tag
        rdf.to_csv(outdir / f"gnn_results_{tag}.csv", index=False, float_format="%.6f")
        agg = rdf.groupby(["model", "split"]).agg(
            AUC_mean=("AUC", "mean"), AUC_std=("AUC", "std"),
            PR_AUC_mean=("PR-AUC", "mean"), PR_AUC_std=("PR-AUC", "std"),
            ECE_mean=("ECE", "mean"), n=("seed", "nunique"),
        ).reset_index()
        agg.to_csv(outdir / f"gnn_summary_{tag}.csv", index=False, float_format="%.6f")
        print(f"\n[汇总] gnn_summary_{tag}.csv")
        print(agg.to_string(index=False))
    print(f"\n[done] 图轨道完成，checkpoint={outdir / '_checkpoints' / f'gnn_{tag}.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
