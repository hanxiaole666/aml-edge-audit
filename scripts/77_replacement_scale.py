"""
77_replacement_scale.py - E-1 替换式 λ 扫描 + E-2 C1 保度 rewiring
===================================================================
背景：M2′（rewire C2，度不变）把跨切边成分从 excess=-0.044 打到 ~0，
      AP 变化只有 +0.011；而 λ 注射往图上新增 K=2000/6000 条边，
      AP 跨度 0.68。两者不在同一尺度，不能用来证明同一个主张。

E-1：把 C2 的跨切边**替换**成 λ 控制的同数边，保留每个端点在跨切边上的
     stub 数（因此节点度、边数都与 C2 完全相同）。这条曲线与 M2′ 同尺度。
E-2：对 C1 做保度 double-edge swap，与 C1 配对，分解 C3-C1=0.115。

用法：python scripts/77_replacement_scale.py
产出：results/replacement_scale.json
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
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA_DIR, RESULTS_DIR, pr_auc, set_seed, write_provenance

FRACS = [0.60, 0.65, 0.70, 0.75, 0.80]
LAMS = [-1.0, -0.5, 0.0, 0.5, 1.0]


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


def split_at(t, frac, n):
    order = np.argsort(t, kind="mergesort")
    cut = int(frac * n)
    tr, te = order[:cut], order[cut:]
    tmax = t[tr].max()
    te = te[t[te] > tmax]
    return tr, te


def build_adj(n, src, dst, et, t, tr_set, te_set, cond):
    """与 44 号完全一致的 C1/C2/C3 邻接（稀疏）。"""
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
        if keep:
            rows.append(a); cols.append(b)
            rows.append(b); cols.append(a)
    A = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    A.data[:] = 1.0
    return A


def to_nx(A):
    coo = A.tocoo()
    G = nx.Graph()
    G.add_nodes_from(range(A.shape[0]))
    for a, b in zip(coo.row, coo.col):
        if a < b:
            G.add_edge(int(a), int(b))
    return G


def score(A, X, y, tr, te):
    n = A.shape[0]
    Ahat = A + sparse.eye(n, format="csr")
    deg = np.asarray(Ahat.sum(axis=1)).ravel()
    S = sparse.diags(1 / np.sqrt(np.maximum(deg, 1e-9))) @ Ahat @ sparse.diags(
        1 / np.sqrt(np.maximum(deg, 1e-9)))
    Xs = StandardScaler().fit(X[tr]).transform(X)
    H = np.asarray(S @ (S @ Xs))
    clf = LogisticRegression(max_iter=1000, class_weight="balanced", C=0.1)
    clf.fit(H[tr], y[tr])
    return clf.predict_proba(H[te])[:, 1]


def paired(y, sa, sb, seed=0, B=400):
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(B):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) < 2:
            continue
        d.append(pr_auc(y[i], sa[i]) - pr_auc(y[i], sb[i]))
    d = np.array(d)
    lo, hi = np.percentile(d, [2.5, 97.5])
    return {"diff": round(float(d.mean()), 4), "ci95": [round(float(lo), 4), round(float(hi), 4)],
            "excludes_zero": bool(lo > 0 or hi < 0)}


def replacement_lambda(A_c2, y, tr, te, lam, rng_seed):
    """受限 double-edge swap：只在跨切边之间交换端点。

    选两条跨切边 (u1,v1)、(u2,v2)，统一成 u 在训练、v 在测试，
    重连为 (u1,v2)、(u2,v1)。每个节点的度严格不变，边数严格不变，
    且拒绝会产生重边的交换。λ 通过接受准则推动同标签率：
      lam<0 只接受同标签对数下降的交换；
      lam>0 只接受上升的交换；
      lam=0 只接受不改变同标签对数的交换。
    """
    tr_set, te_set = set(tr.tolist()), set(te.tolist())
    G = to_nx(A_c2)
    cross = [(u, v) for u, v in G.edges()
             if ((u in tr_set) != (v in tr_set))]
    cross = [(u, v) if u in tr_set else (v, u) for u, v in cross]
    if len(cross) < 2:
        return A_c2.copy(), {"K": len(cross), "excess": 0.0, "degree_ok": True}

    rng = np.random.default_rng(rng_seed)
    n_swaps = 0
    n_try = 0
    max_try = 60 * len(cross)
    while n_try < max_try and n_swaps < 12 * len(cross):
        n_try += 1
        i, j = rng.integers(0, len(cross), size=2)
        if i == j:
            continue
        u1, v1 = cross[int(i)]
        u2, v2 = cross[int(j)]
        if u1 == u2 or v1 == v2:
            continue
        if G.has_edge(u1, v2) or G.has_edge(u2, v1):
            continue
        cur = (1 if y[u1] == y[v1] else 0) + (1 if y[u2] == y[v2] else 0)
        new = (1 if y[u1] == y[v2] else 0) + (1 if y[u2] == y[v1] else 0)
        if lam < 0 and new >= cur:
            continue
        if lam > 0 and new <= cur:
            continue
        if lam == 0 and new != cur:
            continue
        G.remove_edge(u1, v1); G.remove_edge(u2, v2)
        G.add_edge(u1, v2); G.add_edge(u2, v1)
        cross[int(i)] = (u1, v2)
        cross[int(j)] = (u2, v1)
        n_swaps += 1

    A = nx.to_scipy_sparse_array(G, nodelist=range(A_c2.shape[0]), format="csr")
    A.data[:] = 1.0
    d0 = np.asarray(A_c2.sum(axis=1)).ravel()
    d1 = np.asarray(A.sum(axis=1)).ravel()
    if not np.array_equal(d0, d1):
        raise RuntimeError("degree-preserving swap failed")
    if A.sum() != A_c2.sum():
        raise RuntimeError("edge count changed")

    prev_tr = y[tr].mean(); prev_te = y[te].mean()
    p_ind = prev_tr * prev_te + (1 - prev_tr) * (1 - prev_te)
    obs = sum(1 for u, v in cross if y[u] == y[v]) / len(cross)
    return A, {"K": len(cross), "q_realised": round(obs, 4),
               "excess": round(obs - p_ind, 4), "degree_ok": True,
               "n_swaps": n_swaps}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=str(RESULTS_DIR))
    args = ap.parse_args()
    set_seed(0)
    X, y, t, src, dst, et = load_graph()
    n = len(y)
    rows = []

    for frac in FRACS:
        tr, te = split_at(t, frac, n)
        tr_set, te_set = set(tr.tolist()), set(te.tolist())
        A1 = build_adj(n, src, dst, et, t, tr_set, te_set, "C1")
        A2 = build_adj(n, src, dst, et, t, tr_set, te_set, "C2")
        A3 = build_adj(n, src, dst, et, t, tr_set, te_set, "C3")
        s1, s2, s3 = (score(A, X, y, tr, te) for A in (A1, A2, A3))
        a1, a2, a3 = (float(pr_auc(y[te], s)) for s in (s1, s2, s3))
        deg1 = float(np.asarray(A1[te].sum(axis=1)).mean())
        deg3 = float(np.asarray(A3[te].sum(axis=1)).mean())

        # ---- E-1 替换式 λ 扫描（相对 C2）
        e1 = []
        for lam in LAMS:
            A, info = replacement_lambda(A2, y, tr, te, lam,
                                         rng_seed=hash((frac, lam)) % (2**32))
            s = score(A, X, y, tr, te)
            ap_ = float(pr_auc(y[te], s))
            e1.append({"lambda": lam, "AP": round(ap_, 4),
                       "delta_vs_C2": round(ap_ - a2, 4),
                       "excess": info["excess"], "K": info["K"],
                       "degree": round(float(np.asarray(A[te].sum(axis=1)).mean()), 3),
                       "paired_vs_C2": paired(y[te], s, s2, seed=hash((frac, lam, 7)) % 10000)})

        # ---- E-2 对 C1 做保度 rewiring
        G1 = to_nx(A1)
        Gp = G1.copy()
        try:
            nx.double_edge_swap(Gp, nswap=10 * G1.number_of_edges(),
                                max_tries=200 * G1.number_of_edges(), seed=0)
        except nx.NetworkXAlgorithmError:
            pass
        Ap = nx.to_scipy_sparse_array(Gp, nodelist=range(n), format="csr")
        sp_ = score(Ap, X, y, tr, te)
        app_ = float(pr_auc(y[te], sp_))
        degp = float(np.asarray(Ap[te].sum(axis=1)).mean())

        rows.append({"frac": frac, "n_test": int(len(te)), "prevalence": round(float(y[te].mean()), 4),
                     "C1": round(a1, 4), "C2": round(a2, 4), "C3": round(a3, 4),
                     "C1_degree": round(deg1, 3), "C3_degree": round(deg3, 3),
                     "C3_minus_C1": paired(y[te], s3, s1, seed=0),
                     "E1_replacement": e1,
                     "E2_C1_rewire": {"AP": round(app_, 4), "degree": round(degp, 3),
                                      "delta_vs_C1": round(app_ - a1, 4),
                                      "paired_vs_C1": paired(y[te], sp_, s1, seed=0)}})
        print(f"  frac={frac:.2f} C1={a1:.3f} C2={a2:.3f} C3={a3:.3f} | "
              f"E2 M2''={app_:.3f} (deg {degp:.2f}) | "
              f"E1 λ=-1 {e1[0]['delta_vs_C2']:+.3f} λ=+1 {e1[-1]['delta_vs_C2']:+.3f}",
              flush=True)

    out = {"experiment": "replacement_scale", "dataset": "Real-CATS ETH",
           "protocol": ("E-1: replace the C2 cross-split edges with lambda-controlled edges at "
                        "identical count and per-node cross-degree; E-2: degree-preserving "
                        "double-edge swap of C1"),
           "rows": rows}

    def avg(key, sub=None):
        vals = []
        for r in rows:
            v = r[key] if sub is None else sub(r)
            vals.append(v)
        return float(np.mean(vals))

    e1sum = []
    for lam in LAMS:
        vals = [next(e for e in r["E1_replacement"] if e["lambda"] == lam)["delta_vs_C2"]
                for r in rows]
        e1sum.append({"lambda": lam, "mean_delta_vs_C2": round(float(np.mean(vals)), 4),
                      "excludes_zero": sum(
                          1 for e in (next(x for x in r["E1_replacement"] if x["lambda"] == lam)
                                      for r in rows) if e["paired_vs_C2"]["excludes_zero"]),
                      "n": len(vals)})
    out["E1_by_lambda"] = e1sum
    e2d = [r["E2_C1_rewire"]["delta_vs_C1"] for r in rows]
    gain = [r["C3_minus_C1"]["diff"] for r in rows]
    out["summary"] = {
        "C3_minus_C1": round(float(np.mean(gain)), 4),
        "E1_span_vs_C2": round(e1sum[-1]["mean_delta_vs_C2"] - e1sum[0]["mean_delta_vs_C2"], 4),
        "E2_M2pprime_minus_C1": round(float(np.mean(e2d)), 4),
        "degree_share_of_headline": round(
            float(np.mean(gain) - np.mean(e2d)) / max(abs(float(np.mean(gain))), 1e-9), 4),
        "E2_excludes_zero": sum(1 for r in rows
                                if r["E2_C1_rewire"]["paired_vs_C1"]["excludes_zero"]),
        "n_splits": len(rows),
    }
    out["verdict"] = {
        "E1_monotone": all(b["mean_delta_vs_C2"] >= a["mean_delta_vs_C2"] - 1e-9
                           for a, b in zip(e1sum, e1sum[1:])),
        "E2_sig_splits": out["summary"]["E2_excludes_zero"],
        "note": ("E2 delta is the composition part of the headline gain at C1's own degree; "
                 "the remainder is the degree part."),
    }
    out["status"] = "PASS"
    p = Path(args.outdir) / "replacement_scale.json"
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    write_provenance("replacement_scale", {"status": "PASS"})
    print(f"[done] {p}")
    print(json.dumps(out["summary"], indent=1), json.dumps(out["verdict"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
