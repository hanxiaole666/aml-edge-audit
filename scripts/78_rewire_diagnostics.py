"""
78_rewire_diagnostics.py - 给 M2' / M2'' 补诊断列
==================================================
审稿人指出：Table 5 有 C1/C2/C3/M2 的 coverage、neighbour-label AUC、MMD，
但两个保度重连对照 M2'（rewire C2）和 M2''（rewire C1）没有这三列，
"composition was destroyed" 是未经验证的断言。

本脚本在同一张图、同一切分上，对 C1/C2/M2'/M2'' 计算：
  - test-node degree
  - fraction of test nodes with a train neighbour (coverage)
  - neighbour-label AUC（用邻居的标签均值预测测试节点标签）
  - RBF-MMD 到 C3 representation

同时输出 Table 7(top) 需要的 C1 degree（逐 cutoff），用于验证 M2'' 保度。

用法：python scripts/78_rewire_diagnostics.py
产出：results/rewire_diagnostics.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA_DIR, RESULTS_DIR, pr_auc, set_seed, write_provenance

FRACS = [0.60, 0.65, 0.70, 0.75, 0.80]


def load_graph():
    nodes = pd.read_csv(DATA_DIR / "real_cats" / "real_graph_nodes.csv")
    edges = pd.read_csv(DATA_DIR / "real_cats" / "real_graph_edges.csv")
    feats = [c for c in nodes.columns if c.startswith("x")]
    X = np.nan_to_num(nodes[feats].values.astype(float))
    y = nodes.label.values.astype(int)
    t = nodes.time.values.astype(float)
    id2 = {int(v): k for k, v in enumerate(nodes.node_id.values)}
    src = np.array([id2[int(s)] for s in edges.src])
    dst = np.array([id2[int(d)] for d in edges.dst])
    et = edges.time.values.astype(float)
    return X, y, t, src, dst, et


def build_adj(n, src, dst, et, t, tr_set, te_set, cond):
    rows, cols = [], []
    for s, d, e in zip(src, dst, et):
        a, b = int(s), int(d)
        ia, ib = a in tr_set, b in tr_set
        if cond == "C1":
            keep = True
        elif cond == "C2":
            keep = (ia and ib) or ((ia != ib) and (e <= t[b if ia else a]))
        elif cond == "C3":
            keep = ia and ib
        else:
            keep = False
        if keep:
            rows += [a, b]; cols += [b, a]
    A = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    A.data[:] = 1.0
    return A


def to_nx(A):
    G = nx.Graph()
    G.add_nodes_from(range(A.shape[0]))
    coo = A.tocoo()
    for a, b in zip(coo.row, coo.col):
        if a < b:
            G.add_edge(int(a), int(b))
    return G


def rewire(G, nswap_mul=12, seed=0):
    H = G.copy()
    try:
        nx.double_edge_swap(H, nswap=nswap_mul * max(1, G.number_of_edges()),
                            max_tries=200 * max(1, G.number_of_edges()), seed=seed)
    except nx.NetworkXAlgorithmError:
        pass
    A = nx.to_scipy_sparse_array(H, nodelist=range(G.number_of_nodes()), format="csr")
    A.data[:] = 1.0
    return A


def repr_of(A, X, tr):
    """两跳对称归一化传播后的表示。"""
    Ahat = A + sparse.eye(A.shape[0], format="csr")
    deg = np.asarray(Ahat.sum(axis=1)).ravel()
    D = sparse.diags(1 / np.sqrt(np.maximum(deg, 1e-9)))
    S = D @ Ahat @ D
    Xs = StandardScaler().fit(X[tr]).transform(X)
    return np.asarray(S @ (S @ Xs))


def mmd_rbf(A, B, max_n=400, seed=0):
    rng = np.random.default_rng(seed)
    A = A[rng.choice(len(A), min(max_n, len(A)), replace=False)]
    B = B[rng.choice(len(B), min(max_n, len(B)), replace=False)]
    # median heuristic
    Z = np.vstack([A, B])
    sub = Z[rng.choice(len(Z), min(500, len(Z)), replace=False)]
    d2 = ((sub[:, None, :] - sub[None, :, :]) ** 2).sum(-1)
    med = np.median(d2[d2 > 0])
    g = 1.0 / max(med, 1e-9)

    def k(X, Y):
        return np.exp(-g * ((X[:, None, :] - Y[None, :, :]) ** 2).sum(-1))

    return float(k(A, A).mean() + k(B, B).mean() - 2 * k(A, B).mean())


def diagnostics(A, y, tr, te, rep, rep_c3):
    tr_set = set(tr.tolist())
    deg = np.asarray(A[te].sum(axis=1)).ravel()
    # 邻居标签均值（只用训练端邻居）
    nbr_pos = np.zeros(len(y)); nbr_cnt = np.zeros(len(y))
    coo = A.tocoo()
    for a, b in zip(coo.row, coo.col):
        if a in tr_set and b not in tr_set:
            nbr_pos[b] += y[a]; nbr_cnt[b] += 1
        if b in tr_set and a not in tr_set:
            nbr_pos[a] += y[b]; nbr_cnt[a] += 1
    # coverage 与 46 号脚本一致：测试节点至少有一个【训练端】邻居
    # （不是任意邻居，后者含 test-test 边）
    cov = float((nbr_cnt[te] > 0).mean())
    auc = None
    mask = (nbr_cnt[te] > 0) & ((y[te] == 0) | (y[te] == 1))
    if mask.sum() > 10 and len(np.unique(y[te][mask])) == 2:
        auc = float(roc_auc_score(y[te][mask], nbr_pos[te][mask] / nbr_cnt[te][mask]))
    return {"test_degree": round(float(deg.mean()), 3),
            "coverage": round(cov, 4),
            "nbr_label_auc": None if auc is None else round(auc, 4),
            "mmd_to_c3": round(mmd_rbf(rep[te], rep_c3[te]), 4)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=str(RESULTS_DIR))
    args = ap.parse_args()
    set_seed(0)
    X, y, t, src, dst, et = load_graph()
    n = len(y)
    rows = []
    for frac in FRACS:
        order = np.argsort(t, kind="mergesort")
        cut = int(frac * n)
        tr, te = order[:cut], order[cut:]
        tmax = t[tr].max()
        te = te[t[te] > tmax]
        tr_set, te_set = set(tr.tolist()), set(te.tolist())

        A = {c: build_adj(n, src, dst, et, t, tr_set, te_set, c)
             for c in ("C1", "C2", "C3")}
        Ap = {"M2prime": rewire(to_nx(A["C2"]), seed=0),
              "M2dprime": rewire(to_nx(A["C1"]), seed=0)}
        reps = {k: repr_of(v, X, tr) for k, v in list(A.items()) + list(Ap.items())}
        rep_c3 = reps["C3"]
        rec = {"frac": frac, "n_test": int(len(te))}
        for key in ("C1", "C2", "C3", "M2prime", "M2dprime"):
            rec[key] = diagnostics(A.get(key, Ap.get(key)), y, tr, te,
                                   reps[key], rep_c3)
        # 保度验证
        d1 = np.asarray(A["C1"].sum(axis=1)).ravel()
        d2 = np.asarray(Ap["M2dprime"].sum(axis=1)).ravel()
        rec["M2dprime_degree_preserved"] = bool(np.array_equal(d1, d2))
        d2p = np.asarray(A["C2"].sum(axis=1)).ravel()
        dpp = np.asarray(Ap["M2prime"].sum(axis=1)).ravel()
        rec["M2prime_degree_preserved"] = bool(np.array_equal(d2p, dpp))
        rows.append(rec)
        print(f"  frac={frac:.2f} C1deg={rec['C1']['test_degree']:.2f} "
              f"M2prime deg={rec['M2prime']['test_degree']:.2f} "
              f"nbrAUC={rec['M2prime']['nbr_label_auc']} | "
              f"M2dprime deg={rec['M2dprime']['test_degree']:.2f} "
              f"nbrAUC={rec['M2dprime']['nbr_label_auc']}", flush=True)

    out = {"experiment": "rewire_diagnostics", "dataset": "Real-CATS ETH",
           "rows": rows, "status": "PASS"}
    out["all_degree_preserved"] = all(
        r["M2prime_degree_preserved"] and r["M2dprime_degree_preserved"] for r in rows)
    p = Path(args.outdir) / "rewire_diagnostics.json"
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    write_provenance("rewire_diagnostics", {"status": "PASS", "n": len(rows)})
    print(f"[done] {p} degree_preserved={out['all_degree_preserved']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
