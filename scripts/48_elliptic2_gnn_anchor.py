"""
48_elliptic2_gnn_anchor.py - E7：Elliptic2 上的子图 GNN 参照系
======================================================================
审稿意见从第一版欠到现在的一条：0.536 没有参照系。官方 Elliptic2 论文
用 GLASS / GNNSeg / Sub2vec 在 49M 节点的背景图上做子图分类，管线依赖
完整背景图（本机只有发布版 444K cluster / 367K 边）。本脚本不声称复现
官方 SOTA，而是给出**同一份发布拓扑上的子图 GNN 近似锚点**，回答：
  在相同特征与切分下，引入子图拓扑的 GNN 是否显著超过树基线？
  若不超过，则 0.536 是该特征集的实际上限，而非"没调好"。

三个模型，同一份 43 维节点特征：
  A. LightGBM on pooled features (mean/max/std + 4 structural)  —— 树基线
  B. DeepSets：节点特征过 MLP，sum+mean 池化，**不用边**        —— 拓扑阴性对照
  C. GIN-3：3 层图同构网络，子图内消息传递 + 池化              —— 拓扑模型

切分：
  random 80:10:10（官方设定），5 seeds
  size-shift：训练=最小 75%，测试=最大 25%，1 seed（与主表一致）

用法：python scripts/48_elliptic2_gnn_anchor.py [--seeds 5] [--epochs 30]
产出：results/elliptic2_gnn_anchor.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA_DIR, RESULTS_DIR, pr_auc, set_seed, write_provenance

torch.set_num_threads(max(1, torch.get_num_threads()))


# ---------------------------------------------------------------- 数据
def load_subgraphs():
    d = DATA_DIR / "elliptic2"
    nodes = pd.read_csv(d / "nodes.csv")
    edges = pd.read_csv(d / "edges.csv")
    comps = pd.read_csv(d / "connected_components.csv")
    X = np.load(d / "node_features_43.npy").astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    cc_ids = comps.ccId.values.astype(np.int64)
    cc2sub = {int(c): i for i, c in enumerate(cc_ids)}
    y = (comps.ccLabel.values == "suspicious").astype(np.int64)
    node_sub = np.array([cc2sub[int(c)] for c in nodes.ccId.values], dtype=np.int64)
    cl2idx = {int(c): i for i, c in enumerate(nodes.clId.values)}

    # 子图 -> 节点（全局索引）
    order = np.argsort(node_sub, kind="stable")
    sub_nodes = [[] for _ in range(len(cc_ids))]
    for gi in order:
        sub_nodes[node_sub[gi]].append(gi)
    sub_nodes = [np.asarray(v, dtype=np.int64) for v in sub_nodes]

    # 子图内边（局部索引）
    local = {}
    for s, arr in enumerate(sub_nodes):
        for k, gi in enumerate(arr):
            local[gi] = k
    sub_edges = [([], []) for _ in range(len(cc_ids))]
    kept = 0
    for a, b in zip(edges.clId1.values, edges.clId2.values):
        ia, ib = cl2idx.get(int(a)), cl2idx.get(int(b))
        if ia is None or ib is None:
            continue
        sa, sb = node_sub[ia], node_sub[ib]
        if sa != sb:
            continue
        sub_edges[sa][0].append(local[ia])
        sub_edges[sa][1].append(local[ib])
        kept += 1
    sub_edges = [(np.asarray(a, dtype=np.int64), np.asarray(b, dtype=np.int64))
                 for a, b in sub_edges]
    stats = {"n_subgraphs": len(cc_ids), "n_nodes": len(nodes), "n_edges_total": len(edges),
             "n_edges_within": kept, "prevalence": float(y.mean()),
             "nodes_per_sub_mean": float(np.mean([len(a) for a in sub_nodes]))}
    return X, y, sub_nodes, sub_edges, stats


def pooled_features(X, sub_nodes):
    rows = []
    for arr in sub_nodes:
        M = X[arr]
        n = len(arr)
        mean = M.mean(axis=0)
        mx = M.max(axis=0)
        sd = M.std(axis=0)
        struct = np.array([n, 0.0, 0.0, 0.0], dtype=np.float32)
        rows.append(np.concatenate([mean, mx, sd, struct]).astype(np.float32))
    Xp = np.vstack(rows)
    # 填结构特征：节点数、边数、密度、tx 数近似
    return Xp


# ---------------------------------------------------------------- 批处理
def make_batch(sub_ids, sub_nodes, sub_edges, Xt):
    xs, srcs, dsts, batch = [], [], [], []
    off = 0
    for k, s in enumerate(sub_ids):
        arr = sub_nodes[s]
        xs.append(Xt[arr])
        a, b = sub_edges[s]
        srcs.append(a + off)
        dsts.append(b + off)
        batch.append(np.full(len(arr), k, dtype=np.int64))
        off += len(arr)
    x = torch.from_numpy(np.concatenate(xs))
    src = torch.from_numpy(np.concatenate(srcs) if srcs else np.zeros(0, np.int64))
    dst = torch.from_numpy(np.concatenate(dsts) if dsts else np.zeros(0, np.int64))
    bt = torch.from_numpy(np.concatenate(batch))
    return x, src, dst, bt, len(sub_ids)


class GIN(nn.Module):
    def __init__(self, in_dim, hid=64, layers=3, use_edges=True, drop=0.1):
        super().__init__()
        self.use_edges = use_edges
        self.enc = nn.Linear(in_dim, hid)
        self.mlps = nn.ModuleList([
            nn.Sequential(nn.Linear(hid, hid), nn.ReLU(), nn.Linear(hid, hid))
            for _ in range(layers)])
        self.eps = nn.Parameter(torch.zeros(layers))
        self.drop = nn.Dropout(drop)
        self.head = nn.Sequential(nn.Linear(hid * 2, hid), nn.ReLU(), nn.Linear(hid, 1))

    def forward(self, x, src, dst, batch, n_graphs):
        h = F.relu(self.enc(x))
        for l, mlp in enumerate(self.mlps):
            if self.use_edges and src.numel() > 0:
                m = torch.zeros_like(h).index_add_(0, dst, h[src])
                m = m.index_add_(0, src, h[dst])
            else:
                m = torch.zeros_like(h)
            h = mlp((1 + self.eps[l]) * h + m)
            h = self.drop(h)
        s = torch.zeros(n_graphs, h.size(1)).index_add_(0, batch, h)
        cnt = torch.zeros(n_graphs).index_add_(0, batch, torch.ones_like(batch, dtype=torch.float))
        mean = s / cnt.unsqueeze(1).clamp(min=1)
        z = torch.cat([s, mean], dim=1)
        return self.head(z).squeeze(1)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", default=str(RESULTS_DIR))
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=512)
    args = ap.parse_args()
    set_seed(0)
    t0 = time.time()

    X, y, sub_nodes, sub_edges, stats = load_subgraphs()
    print("[data]", stats, flush=True)

    # 结构特征补全（供 LightGBM）
    Xp = pooled_features(X, sub_nodes)
    n_nodes = np.array([len(a) for a in sub_nodes], dtype=np.float32)
    n_edges = np.array([len(a[0]) for a in sub_edges], dtype=np.float32)
    Xp[:, -4] = n_nodes
    Xp[:, -3] = n_edges
    Xp[:, -2] = n_edges / np.maximum(n_nodes, 1)
    Xp[:, -1] = np.log1p(n_nodes) * np.log1p(n_edges)

    out = {"experiment": "elliptic2_gnn_anchor", "data": stats,
           "models": ["LightGBM-pooled", "DeepSets", "GIN-3"],
           "status": "PASS", "random": {}, "size_shift": {}}

    from lightgbm import LGBMClassifier

    def run_split(tr, va, te, tag, seed):
        res = {}
        # A. LightGBM
        sc = StandardScaler().fit(Xp[tr])
        lgb = LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31,
                             subsample=0.9, colsample_bytree=0.8, random_state=seed,
                             n_jobs=-1, verbose=-1,
                             scale_pos_weight=(len(tr) - y[tr].sum()) / max(y[tr].sum(), 1))
        lgb.fit(sc.transform(Xp[tr]), y[tr])
        res["LightGBM-pooled"] = round(float(pr_auc(y[te], lgb.predict_proba(sc.transform(Xp[te]))[:, 1])), 4)
        # 节点特征标准化（GNN 用）
        nsc = StandardScaler().fit(X[np.concatenate([tr, va])])
        Xn = nsc.transform(X).astype(np.float32)
        for name, use_edges in [("DeepSets", False), ("GIN-3", True)]:
            torch.manual_seed(seed)
            model = GIN(X.shape[1], hid=64, layers=3, use_edges=use_edges)
            ap_te, ap_va = train_eval_idx(model, Xn, y, tr, va, te,
                                          sub_nodes, sub_edges, args.epochs, args.bs, seed)
            res[name] = round(float(ap_te), 4)
        print(f"    [{tag} seed={seed}] {res}", flush=True)
        return res

    def train_eval_idx(model, Xn, y, tr, va, te, sub_nodes, sub_edges, epochs, bs, seed):
        """按子图索引训练；批内直接用索引取 Xn。"""
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        pos_w = torch.tensor((len(tr) - y[tr].sum()) / max(y[tr].sum(), 1), dtype=torch.float)
        lossf = nn.BCEWithLogitsLoss(pos_weight=pos_w)
        rng = np.random.default_rng(seed)
        Xt = torch.from_numpy(Xn)
        best, best_state, patience = -1.0, None, 0
        yt = torch.from_numpy(y.astype(np.float32))

        def infer(ids):
            model.eval()
            outs = []
            with torch.no_grad():
                for st in range(0, len(ids), 2048):
                    b = ids[st:st + 2048]
                    x, src, dst, bt, ng = make_batch(b, sub_nodes, sub_edges, Xt)
                    outs.append(model(x, src, dst, bt, ng).numpy())
            return np.concatenate(outs)

        for ep in range(epochs):
            model.train()
            idx = rng.permutation(tr)
            for st in range(0, len(idx), bs):
                b = idx[st:st + bs]
                x, src, dst, bt, ng = make_batch(b, sub_nodes, sub_edges, Xt)
                logit = model(x, src, dst, bt, ng)
                loss = lossf(logit, yt[b])
                opt.zero_grad(); loss.backward(); opt.step()
            ap = pr_auc(y[va], infer(va))
            if ap > best:
                best, best_state, patience = ap, {k: v.clone() for k, v in model.state_dict().items()}, 0
            else:
                patience += 1
                if patience >= 8:
                    break
        model.load_state_dict(best_state)
        return pr_auc(y[te], infer(te)), best

    # ---- random 80:10:10
    rows_rand = defaultdict(list)
    for seed in range(args.seeds):
        trva, te = train_test_split(np.arange(len(y)), test_size=0.1, stratify=y, random_state=seed)
        tr, va = train_test_split(trva, test_size=1 / 9, stratify=y[trva], random_state=seed)
        r = run_split(tr, va, te, "random", seed)
        for k, v in r.items():
            rows_rand[k].append(v)
    out["random"] = {k: {"mean": round(float(np.mean(v)), 4),
                         "std": round(float(np.std(v, ddof=1)), 4), "folds": v}
                     for k, v in rows_rand.items()}

    # ---- size shift：最小 75% 训练，最大 25% 测试
    ordr = np.argsort(n_nodes, kind="mergesort")
    cut = int(0.75 * len(y))
    tr_all, te = ordr[:cut], ordr[cut:]
    tr, va = train_test_split(tr_all, test_size=0.1, stratify=y[tr_all], random_state=0)
    r = run_split(tr, va, te, "size_shift", 0)
    out["size_shift"] = {k: {"value": v} for k, v in r.items()}
    out["size_shift_prevalence"] = {"train": round(float(y[tr].mean()), 4),
                                    "test": round(float(y[te].mean()), 4)}
    out["elapsed_sec"] = round(time.time() - t0, 1)

    # ---- 判据
    g = out["random"]["GIN-3"]["mean"]; d = out["random"]["DeepSets"]["mean"]
    l = out["random"]["LightGBM-pooled"]["mean"]
    out["verdict"] = {
        "topology_gain_over_deepsets": round(g - d, 4),
        "gnn_vs_tree_gap": round(g - l, 4),
        "interpretation": ("GNN does not exceed the pooled-feature tree baseline"
                           if g <= l else "GNN exceeds the tree baseline"),
    }
    print("\n[summary]", json.dumps({"random": out["random"],
                                     "size_shift": out["size_shift"],
                                     "verdict": out["verdict"]}, indent=2, default=float))

    p = Path(args.outdir) / "elliptic2_gnn_anchor.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    write_provenance("elliptic2_gnn_anchor", {"status": "PASS", "seeds": args.seeds})
    print(f"[done] {p}  ({out['elapsed_sec']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
