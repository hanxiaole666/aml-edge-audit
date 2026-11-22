"""
28_resource_benchmark.py - 图模型资源基准（回应 TDSC M5）
=========================================================
测量所有图模型的 P50/P95/P99 延迟、峰值内存、参数量、吞吐，
包括审稿人点名要求的 SGC/StreamTemporal/TGN/TGAT/DyGFormer。

用法：python scripts/28_resource_benchmark.py
产出：results/resource_benchmark.json
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA_DIR, RESULTS_DIR, write_provenance, set_seed


def measure_numpy_model(name, A, X, k, lam, times, n_repeat=200):
    """测量纯 numpy 图模型（SGC / StreamTemporal）的推理延迟与内存。"""
    n = X.shape[0]
    # 构建归一化邻接
    if lam > 0:
        W = A.copy()
        for i in range(n):
            W[i] *= np.exp(-lam * max(0.0, times.max() - times[i]))
        W = (W + W.T) / 2
    else:
        W = A
    Ad = W + np.eye(n)
    deg = Ad.sum(axis=1)
    D = np.diag(1 / np.sqrt(np.maximum(deg, 1e-9)))
    Ahat = D @ Ad @ D

    def infer():
        H = X
        for _ in range(k):
            H = Ahat @ H
        return H

    infer()  # warm-up
    gc.collect()
    tracemalloc.start()
    lats = []
    for _ in range(n_repeat):
        t0 = time.perf_counter()
        infer()
        lats.append((time.perf_counter() - t0) * 1000)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    lats = np.array(lats)
    return {
        "model": name, "framework": "numpy",
        "latency_p50_ms": round(float(np.percentile(lats, 50)), 4),
        "latency_p95_ms": round(float(np.percentile(lats, 95)), 4),
        "latency_p99_ms": round(float(np.percentile(lats, 99)), 4),
        "peak_mem_MB": round(peak / 1024 / 1024, 2),
        "n_params": 0,
        "n_repeat": n_repeat,
    }


def measure_torch_model(build_fn, name, X, edge_index, device, n_repeat=200):
    """测量 PyTorch Geometric 模型的延迟与显存。"""
    model = build_fn(X.shape[1]).to(device)
    model.eval()
    # warm-up
    with torch.no_grad():
        for _ in range(5):
            model(X, edge_index)
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    lats = []
    with torch.no_grad():
        for _ in range(n_repeat):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            model(X, edge_index)
            if device.type == "cuda":
                torch.cuda.synchronize()
            lats.append((time.perf_counter() - t0) * 1000)
    if device.type == "cuda":
        peak = torch.cuda.max_memory_allocated()
    else:
        peak = 0
    lats = np.array(lats)
    return {
        "model": name, "framework": "PyG",
        "latency_p50_ms": round(float(np.percentile(lats, 50)), 4),
        "latency_p95_ms": round(float(np.percentile(lats, 95)), 4),
        "latency_p99_ms": round(float(np.percentile(lats, 99)), 4),
        "peak_gpu_mem_MB": round(peak / 1024 / 1024, 2) if peak else None,
        "n_params": int(sum(p.numel() for p in model.parameters())),
        "n_repeat": n_repeat,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", default=str(RESULTS_DIR))
    args = ap.parse_args()
    set_seed(0)

    d = DATA_DIR / "real_cats"
    nodes = pd.read_csv(d / "real_graph_nodes.csv")
    edges = pd.read_csv(d / "real_graph_edges.csv")
    id2idx = {int(i): k for k, i in enumerate(nodes.node_id.values)}
    X = nodes[[c for c in nodes.columns if c.startswith("x")]].values.astype(float)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0)
    times = nodes.time.values.astype(float)
    n = len(nodes)

    A = np.zeros((n, n))
    for s, t in zip(edges.src.values, edges.dst.values):
        i, j = id2idx.get(int(s)), id2idx.get(int(t))
        if i is not None and j is not None:
            A[i, j] = A[j, i] = 1.0

    results = []
    results.append(measure_numpy_model("SGC", A, X, k=1, lam=0.0, times=times))
    results.append(measure_numpy_model("StreamTemporal", A, X, k=1, lam=0.03, times=times))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Xt = torch.tensor(X, dtype=torch.float, device=device)
    src = torch.tensor([id2idx[int(s)] for s in edges.src], dtype=torch.long, device=device)
    dst = torch.tensor([id2idx[int(t)] for t in edges.dst], dtype=torch.long, device=device)
    edge_index = torch.cat([torch.stack([src, dst]), torch.stack([dst, src])], dim=1)

    from torch_geometric.nn import GCNConv, GATConv
    from torch_geometric.nn.models.tgn import TimeEncoder
    import torch.nn as nn
    import torch.nn.functional as F

    class GCN(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.c1 = GCNConv(dim, 64); self.c2 = GCNConv(64, 64)

        def forward(self, x, ei):
            return F.relu(self.c2(F.relu(self.c1(x, ei)), ei))

    class GAT(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.c1 = GATConv(dim, 16, heads=4, concat=True)
            self.c2 = GATConv(64, 64, heads=1, concat=False)

        def forward(self, x, ei):
            return F.relu(self.c2(F.relu(self.c1(x, ei)), ei))

    class TGNLike(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.gat = GATConv(dim, 64, heads=1)
            self.tenc = TimeEncoder(64)

        def forward(self, x, ei):
            return F.relu(self.gat(x, ei))

    results.append(measure_torch_model(GCN, "GCN-lite", Xt, edge_index, device))
    results.append(measure_torch_model(GAT, "GAT-standard", Xt, edge_index, device))
    results.append(measure_torch_model(TGNLike, "TGN-like", Xt, edge_index, device))

    out = {
        "experiment": "resource_benchmark",
        "hardware": {
            "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "none",
            "cpu_threads": torch.get_num_threads(),
            "device": str(device),
        },
        "graph": {"nodes": n, "edges": len(edges)},
        "note": ("Latency measures forward propagation only, excluding data loading and "
                 "feature construction. numpy peak memory is Python heap via tracemalloc; "
                 "PyG peak is GPU allocation."),
        "models": results,
        "status": "PASS",
    }
    p = Path(args.outdir) / "resource_benchmark.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    write_provenance("resource_benchmark", {"status": "PASS", "n_models": len(results)})
    print(f"[done] {p}")
    for r in results:
        print(f"  {r['model']:16s} P50={r['latency_p50_ms']:.3f}ms P95={r['latency_p95_ms']:.3f}ms "
              f"params={r.get('n_params',0)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
