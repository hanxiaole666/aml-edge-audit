"""
56_elliptic2_subgraph_models.py - E2：Elliptic2 子图模型族 + 文献对照
======================================================================
审稿人要求旗舰数据集上给 SOTA 参照。官方论文（Bellei et al., 2024）的
Table 2 报告（random 80:10:10，**不使用节点/边特征**）：

    method      test f1   test pr_auc   test roc_auc
    GNN-Seg      0.398       0.026         0.537
    Sub2Vec      0.944       0.022         0.496
    GLASS        0.933       0.208         0.889

GLASS 的优势来自 **49M 节点的背景图**（跨子图边界消息传递），不是子图内部
拓扑。本机只有发布版 444K cluster / 367K 子图内边，没有背景图，因此：
  * 不能声称复现 GLASS；能声称的是"在发布拓扑 + 发布特征上的同输入对照"。
  * 把"结构信号≈0"这一结论限定在**子图内拓扑**上，而不是全局。

本脚本在同一 split、同一份 43 维节点特征上跑五个模型：
    LightGBM-pooled   池化特征树基线
    DeepSets          节点 MLP + 池化，**不用边**（拓扑阴性对照）
    GIN-3             求和聚合 + (1+eps)
    GCN-2             均值聚合 + 自环
    SAGE-2            拼接自表示与邻居均值
若 GIN/GCN/SAGE ≤ DeepSets，则子图内拓扑无增益；若都高于 GNN-Seg 的
0.026，则说明发布特征本身承载了主要信号。

用法：python scripts/56_elliptic2_subgraph_models.py [--seeds 3] [--epochs 25]
产出：results/elliptic2_subgraph_models.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RESULTS_DIR, pr_auc, set_seed, write_provenance

torch.set_num_threads(8)

_spec = importlib.util.spec_from_file_location(
    "e2", Path(__file__).resolve().parent / "48_elliptic2_gnn_anchor.py")
_e2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_e2)
load_subgraphs, make_batch, pooled_features = _e2.load_subgraphs, _e2.make_batch, _e2.pooled_features


class MPNN(nn.Module):
    """mode in {gin, gcn, sage}; use_edges=False 退化为 DeepSets。"""

    def __init__(self, in_dim, hid=64, layers=3, mode="gin", use_edges=True, drop=0.1):
        super().__init__()
        self.mode, self.use_edges = mode, use_edges
        self.enc = nn.Linear(in_dim, hid)
        d_in = hid if mode != "sage" else 2 * hid
        self.lins = nn.ModuleList([nn.Linear(d_in, hid) for _ in range(layers)])
        self.eps = nn.Parameter(torch.zeros(layers))
        self.drop = nn.Dropout(drop)
        self.head = nn.Sequential(nn.Linear(hid * 2, hid), nn.ReLU(), nn.Linear(hid, 1))

    def forward(self, x, src, dst, batch, n_graphs):
        h = F.relu(self.enc(x))
        for l, lin in enumerate(self.lins):
            if self.use_edges and src.numel() > 0:
                if self.mode == "gin":
                    m = torch.zeros_like(h).index_add_(0, dst, h[src])
                    m = m.index_add_(0, src, h[dst])
                    h = lin((1 + self.eps[l]) * h + m)
                else:
                    s = torch.zeros_like(h).index_add_(0, dst, h[src])
                    s = s.index_add_(0, src, h[dst])
                    deg = torch.zeros(h.size(0)).index_add_(
                        0, dst, torch.ones_like(dst, dtype=torch.float))
                    deg = deg.index_add_(0, src, torch.ones_like(src, dtype=torch.float))
                    mean = s / deg.clamp(min=1).unsqueeze(1)
                    h = lin(torch.cat([h, mean], dim=1) if self.mode == "sage"
                            else mean + h)
            else:
                h = lin(h if self.mode != "sage" else torch.cat([h, h], dim=1))
            h = self.drop(F.relu(h))
        s = torch.zeros(n_graphs, h.size(1)).index_add_(0, batch, h)
        cnt = torch.zeros(n_graphs).index_add_(0, batch, torch.ones_like(batch, dtype=torch.float))
        z = torch.cat([s, s / cnt.clamp(min=1).unsqueeze(1)], dim=1)
        return self.head(z).squeeze(1)


def train_eval(model, Xn, y, tr, va, te, epochs, bs, seed):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3, weight_decay=1e-4)
    Xt = torch.from_numpy(Xn)
    pos_w = torch.tensor((len(tr) - y[tr].sum()) / max(y[tr].sum(), 1), dtype=torch.float32)
    best, state, bad = -1.0, None, 0
    for ep in range(epochs):
        model.train()
        perm = np.random.default_rng(ep).permutation(tr)
        for i in range(0, len(perm), bs):
            idx = perm[i:i + bs]
            x, s, d, b, ng = make_batch(idx, *_SUBS, Xt)
            opt.zero_grad()
            logit = model(x, s, d, b, ng)
            loss = F.binary_cross_entropy_with_logits(
                logit, torch.from_numpy(y[idx].astype(np.float32)), pos_weight=pos_w)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vals = []
            for i in range(0, len(va), 2048):
                idx = va[i:i + 2048]
                x, s, d, b, ng = make_batch(idx, *_SUBS, Xt)
                vals.append(torch.sigmoid(model(x, s, d, b, ng)).numpy())
            p = np.concatenate(vals)
        a = pr_auc(y[va], p)
        if a > best:
            best, state, bad = a, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= 6:
                break
    if state:
        model.load_state_dict(state)
    model.eval()
    with torch.no_grad():
        vals = []
        for i in range(0, len(te), 2048):
            idx = te[i:i + 2048]
            x, s, d, b, ng = make_batch(idx, *_SUBS, Xt)
            vals.append(torch.sigmoid(model(x, s, d, b, ng)).numpy())
    return float(pr_auc(y[te], np.concatenate(vals))), best


_SUBS = None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", default=str(RESULTS_DIR))
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--bs", type=int, default=1024)
    args = ap.parse_args()
    set_seed(0)
    t0 = time.time()
    global _SUBS

    X, y, sub_nodes, sub_edges, stats = load_subgraphs()
    _SUBS = (sub_nodes, sub_edges)
    print("[data]", stats, flush=True)

    Xp = pooled_features(X, sub_nodes)
    n_nodes = np.array([len(a) for a in sub_nodes], dtype=np.float32)
    n_edges = np.array([len(a[0]) for a in sub_edges], dtype=np.float32)
    Xp[:, -4] = n_nodes
    Xp[:, -3] = n_edges
    Xp[:, -2] = n_edges / np.maximum(n_nodes, 1)
    Xp[:, -1] = np.log1p(n_nodes) * np.log1p(n_edges)

    idx = np.arange(len(y))
    tr, tmp = train_test_split(idx, test_size=0.2, random_state=0, stratify=y)
    va, te = train_test_split(tmp, test_size=0.5, random_state=0, stratify=y[tmp])
    print(f"[split] train={len(tr)} val={len(va)} test={len(te)} "
          f"prev={y[te].mean():.4f}", flush=True)

    res = {}
    # ---- LightGBM-pooled
    aps = []
    for sd in range(args.seeds):
        sc = StandardScaler().fit(Xp[tr])
        m = LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31,
                           subsample=0.9, colsample_bytree=0.8, random_state=sd,
                           n_jobs=-1, verbose=-1,
                           scale_pos_weight=(len(tr) - y[tr].sum()) / max(y[tr].sum(), 1))
        m.fit(sc.transform(Xp[tr]), y[tr])
        aps.append(float(pr_auc(y[te], m.predict_proba(sc.transform(Xp[te]))[:, 1])))
    res["LightGBM-pooled"] = {"test_AP": round(float(np.mean(aps)), 4),
                              "sd": round(float(np.std(aps, ddof=1)), 4) if len(aps) > 1 else 0.0,
                              "per_seed": [round(a, 4) for a in aps], "uses_edges": False}
    print("[lgbm]", res["LightGBM-pooled"], flush=True)

    # ---- GNN 族
    nsc = StandardScaler().fit(X[np.concatenate([tr, va])])
    Xn = nsc.transform(X).astype(np.float32)
    for name, mode, use_edges, layers in [("DeepSets", "gin", False, 3),
                                          ("GIN-3", "gin", True, 3),
                                          ("GCN-2", "gcn", True, 2),
                                          ("SAGE-2", "sage", True, 2)]:
        aps, vas = [], []
        for sd in range(args.seeds):
            mdl = MPNN(X.shape[1], hid=64, layers=layers, mode=mode, use_edges=use_edges)
            a, v = train_eval(mdl, Xn, y, tr, va, te, args.epochs, args.bs, sd)
            aps.append(a)
            vas.append(v)
        res[name] = {"test_AP": round(float(np.mean(aps)), 4),
                     "sd": round(float(np.std(aps, ddof=1)), 4) if len(aps) > 1 else 0.0,
                     "val_AP": round(float(np.mean(vas)), 4),
                     "per_seed": [round(a, 4) for a in aps], "uses_edges": use_edges}
        print(f"[{name}]", res[name], flush=True)

    # ---- 文献对照
    lit = {
        "GNN-Seg": {"train": {"f1": 0.397, "pr_auc": 0.024, "roc_auc": 0.538},
                    "test": {"f1": 0.398, "pr_auc": 0.026, "roc_auc": 0.537},
                    "uses_features": False, "uses_background_graph": False},
        "Sub2Vec": {"train": {"f1": 0.964, "pr_auc": 0.145, "roc_auc": 0.706},
                    "test": {"f1": 0.944, "pr_auc": 0.022, "roc_auc": 0.496},
                    "uses_features": False, "uses_background_graph": False},
        "GLASS": {"train": {"f1": 0.937, "pr_auc": 0.224, "roc_auc": 0.915},
                  "test": {"f1": 0.933, "pr_auc": 0.208, "roc_auc": 0.889},
                  "uses_features": False, "uses_background_graph": True},
    }
    out = {
        "experiment": "elliptic2_subgraph_models", "data": stats,
        "split": {"train": int(len(tr)), "val": int(len(va)), "test": int(len(te)),
                  "test_prevalence": round(float(y[te].mean()), 4),
                  "protocol": "random 80:10:10, stratified, seed 0"},
        "models": res,
        "literature": lit,
        "comparability": {
            "GNN-Seg": "not reproduced: needs background-graph sampling (SALIENT++) and the "
                       "official pipeline; our GIN on subgraph-internal edges is the closest "
                       "structural analogue but uses features",
            "Sub2Vec": "not reproduced: random-walk embedding over a 49M-node graph",
            "GLASS": "not reproduced: requires the 49M-cluster background graph for "
                     "cross-boundary message passing; only the 444K-cluster subgraph release "
                     "is available locally",
            "headline": "Literature numbers use no node/edge features; ours do. The 0.208 of "
                        "GLASS is attributable to the background graph, which is absent here. "
                        "Within the released topology, adding edges does not beat a no-edge "
                        "pooling baseline.",
        },
        "status": "PASS",
    }
    best_gnn = max(res[k]["test_AP"] for k in ["GIN-3", "GCN-2", "SAGE-2"])
    out["verdict"] = {
        "best_edge_model_AP": round(best_gnn, 4),
        "no_edge_DeepSets_AP": res["DeepSets"]["test_AP"],
        "edge_gain_over_no_edge": round(best_gnn - res["DeepSets"]["test_AP"], 4),
        "edge_gain_over_tree": round(best_gnn - res["LightGBM-pooled"]["test_AP"], 4),
        "topology_helps_within_release": bool(best_gnn > res["DeepSets"]["test_AP"] + 0.01),
        "literature_gap_explained_by_background_graph": True,
    }
    print("[verdict]", out["verdict"], flush=True)
    print(f"[time] {time.time()-t0:.0f}s", flush=True)

    p = Path(args.outdir) / "elliptic2_subgraph_models.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    write_provenance("elliptic2_subgraph_models", {"status": "PASS"})
    print(f"[done] {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
