"""
21_gpu_gnn.py - GPU 加速的 Elliptic2 真实图 GNN 实验
=====================================================
利用 RTX 2060 6GB CUDA，在 Elliptic2 真实交易图上跑标准 GCN/GAT/GraphSAGE，
做节点级可疑性分类（用连通分量标签作节点弱标签），与 CPU 基线对比。

数据：nodes.csv(444K 节点→ccId) + edges.csv(367K 边) + connected_components.csv(122K 标签)
节点特征：从 background_nodes.csv 流式提取的 43 维（或拓扑特征）
GPU：torch.cuda，自动混合精度可选

用法：
  python scripts/21_gpu_gnn.py --model gcn --epochs 30
  python scripts/21_gpu_gnn.py --model gat --epochs 30
产出：results/gpu_gnn_results.csv
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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GATConv, SAGEConv

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA_DIR, RESULTS_DIR, pr_auc, roc_auc, write_provenance, set_seed


class GNN(torch.nn.Module):
    def __init__(self, in_dim: int, hidden: int, model: str, dropout: float = 0.2):
        super().__init__()
        conv = {"gcn": GCNConv, "gat": GATConv, "sage": SAGEConv}[model]
        self.c1 = conv(in_dim, hidden)
        self.c2 = conv(hidden, hidden)
        self.out = torch.nn.Linear(hidden, 1)  # 节点级二分类输出
        self.dropout = dropout

    def forward(self, x, edge_index):
        h = F.relu(self.c1(x, edge_index))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = F.relu(self.c2(h, edge_index))
        return self.out(h).squeeze(-1)  # (N,)


def load_graph(device, use_full_feat: bool = False):
    d = DATA_DIR / "elliptic2"
    nodes = pd.read_csv(d / "nodes.csv")
    edges = pd.read_csv(d / "edges.csv")
    comps = pd.read_csv(d / "connected_components.csv")
    n = len(nodes)

    # 节点特征：优先用完整特征缓存（若 20 脚本已生成），否则拓扑特征
    cache = d / "node_features_43.npy"
    if use_full_feat and cache.exists():
        X = np.load(cache).astype(np.float32)
        print(f"[gpu_gnn] 使用完整特征缓存 {X.shape}")
    else:
        # 拓扑特征：度 + ccId hash（弱特征）
        deg = np.zeros(n, dtype=np.float32)
        src = edges["clId1"].map(dict(zip(nodes.clId, range(n)))).dropna().astype(int).values
        dst = edges["clId2"].map(dict(zip(nodes.clId, range(n)))).dropna().astype(int).values
        np.add.at(deg, src, 1); np.add.at(deg, dst, 1)
        X = np.log1p(deg).reshape(-1, 1)
        # cc 规模作为弱特征
        cc_size = nodes.groupby("ccId").size()
        X = np.hstack([X, np.log1p(nodes.ccId.map(cc_size).values.astype(np.float32)).reshape(-1, 1)])
        print(f"[gpu_gnn] 使用拓扑特征 {X.shape}（完整特征用 --use-full 先运行 20 脚本）")

    sc = StandardScaler().fit(X)
    X = torch.tensor(sc.transform(X), dtype=torch.float, device=device)

    id2idx = {int(c): i for i, c in enumerate(nodes.clId.values)}
    src_i = np.array([id2idx[int(c)] for c in edges.clId1 if int(c) in id2idx], dtype=np.int64)
    dst_i = np.array([id2idx[int(c)] for c in edges.clId2 if int(c) in id2idx], dtype=np.int64)
    edge_index = torch.tensor(np.vstack([src_i, dst_i]), dtype=torch.long, device=device)
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1).unique(dim=1)

    # 节点标签：ccLabel 传播到节点
    cc2lab = dict(zip(comps.ccId, (comps.ccLabel == "suspicious").astype(int)))
    y = torch.tensor([cc2lab.get(int(c), 0) for c in nodes.ccId.values],
                     dtype=torch.long, device=device)
    print(f"[gpu_gnn] 节点={n} 边={edge_index.shape[1]//2} 正类率={y.float().mean():.3%}")
    return Data(x=X, edge_index=edge_index, y=y)


def train_eval(data, model_name, device, epochs, seed, lr=0.01):
    set_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    idx = np.arange(data.y.shape[0])
    tr, te = train_test_split(idx, test_size=0.2, random_state=seed,
                              stratify=data.y.cpu().numpy())
    tr_t = torch.tensor(tr, dtype=torch.long, device=device)
    te_t = torch.tensor(te, dtype=torch.long, device=device)

    model = GNN(data.x.shape[1], hidden=64, model=model_name).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-4)
    y_pos = float(data.y[tr_t].float().mean())
    pos_weight = torch.tensor([(1 - y_pos) / max(y_pos, 1e-6)], device=device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    t0 = time.perf_counter()
    model.train()
    for ep in range(epochs):
        opt.zero_grad()
        logits = model(data.x, data.edge_index)
        loss = criterion(logits[tr_t], data.y[tr_t].float())
        loss.backward()
        opt.step()
        if (ep + 1) % 10 == 0:
            print(f"  epoch {ep+1}/{epochs} loss={loss.item():.4f}", flush=True)
    train_sec = time.perf_counter() - t0

    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        prob = torch.sigmoid(logits[te_t]).cpu().numpy()
    yte = data.y[te_t].cpu().numpy()
    return {
        "model": f"gpu_{model_name}", "seed": seed, "epochs": epochs,
        "PR-AUC": float(pr_auc(yte, prob)), "AUC": float(roc_auc(yte, prob)),
        "train_seconds": round(train_sec, 2),
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
        "status": "done",
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", choices=["gcn", "gat", "sage"], default="gcn")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--use-full", action="store_true")
    ap.add_argument("--outdir", default=str(RESULTS_DIR))
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[gpu_gnn] device={device}", flush=True)
    data = load_graph(device, args.use_full)

    rows = []
    for seed in range(args.seeds):
        try:
            res = train_eval(data, args.model, device, args.epochs, seed, args.lr)
            rows.append(res)
            print(f"  [OK] {args.model} seed{seed}: PR-AUC={res['PR-AUC']:.4f} "
                  f"AUC={res['AUC']:.4f} time={res['train_seconds']}s", flush=True)
        except Exception as exc:
            print(f"  [FAIL] seed{seed}: {type(exc).__name__}: {exc}", flush=True)

    if rows:
        df = pd.DataFrame(rows)
        out = Path(args.outdir) / "gpu_gnn_results.csv"
        if out.exists():
            old = pd.read_csv(out)
            df = pd.concat([old, df], ignore_index=True).drop_duplicates(
                subset=["model", "seed", "epochs"], keep="last")
        df.to_csv(out, index=False, float_format="%.6f")
        print(f"\n[汇总] {out}")
        print(df.groupby("model").agg(PR_AUC_mean=("PR-AUC", "mean"),
                                      AUC_mean=("AUC", "mean"),
                                      sec=("train_seconds", "mean")).to_string())
        write_provenance("gpu_gnn", {"model": args.model, "n_runs": len(rows),
                                     "device": str(device)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
