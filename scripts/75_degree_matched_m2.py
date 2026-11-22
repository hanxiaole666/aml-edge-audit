"""
75_degree_matched_m2.py - E-C 度匹配的随机图对照 M2'
=====================================================
Table 5 的 M2 只匹配了边数：C2 的测试节点度是 2.44、覆盖率 17.6%，
而均匀随机配对给出 7.40、99.2%。"同样的边数，度差 3 倍"意味着
M2 与 C1 的对比证否的只是"度单独决定"，不是"度无关"。

M2' 用配置模型在**保持 C2 每个节点度不变**的前提下随机重连：
边数与度都匹配，只有"谁和谁连"被随机化。若 M2' 仍接近 M2，
则成分是主因；若 M2' 升到接近 C2，则度确实在起作用。

实现：networkx.double_edge_swap 在 C2 图上做 10x|E| 次交换，
保持简单图、保持度序列。同时报告交换后的同行标签 excess。

用法：python scripts/75_degree_matched_m2.py
产出：results/degree_matched_m2.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
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


def c2_graph(n, src, dst, et, t, tr_set, te_set):
    """C2：与 scripts/44_edge_protocol_final.py 完全相同的构造。

    44 号用 e <= t[test-endpoint] 过滤跨切边；边时间来自 edges.csv，
    节点时间来自 real_cats_eth.csv，两者并不恒等，因此这一步会剔除
    约 1000 条跨切边。此前 75 号缺少这个过滤，导致 C2 边数 6392 而
    44 号是 5349，两张表报出两套数。这里以 44 号为准并轨。
    """
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for s, d, e in zip(src, dst, et):
        a, b = int(s), int(d)
        ia, ib = a in tr_set, b in tr_set
        if ia and ib:
            G.add_edge(a, b)
        elif ia != ib:
            tn = b if ia else a
            if e <= t[tn]:
                G.add_edge(a, b)
        # 两端都在测试集：排除
    return G


def excess_of(G, y, tr_set, te_set):
    """跨切分边的同行标签率与独立期望。"""
    same = tot = 0
    for a, b in G.edges():
        if (a in tr_set and b in te_set) or (a in te_set and b in tr_set):
            same += int(y[a] == y[b]); tot += 1
    # 全图同行率也报一份
    same_all = sum(int(y[a] == y[b]) for a, b in G.edges())
    n_all = G.number_of_edges()
    return {"n_cross": tot,
            "cross_observed_same": round(same / tot, 4) if tot else None,
            "observed_same_all": round(same_all / max(n_all, 1), 4)}


def eval_graph(G, X, y, tr, te):
    A = nx.to_numpy_array(G, nodelist=range(G.number_of_nodes()))
    Ahat = A + np.eye(A.shape[0])
    deg = Ahat.sum(axis=1)
    D = np.diag(1 / np.sqrt(np.maximum(deg, 1e-9)))
    S = D @ Ahat @ D
    Xs = StandardScaler().fit(X[tr]).transform(X)
    H = S @ (S @ Xs)
    clf = LogisticRegression(max_iter=1000, class_weight="balanced", C=0.1)
    clf.fit(H[tr], y[tr])
    s = clf.predict_proba(H[te])[:, 1]
    return s, float(np.mean(A[te].sum(axis=1))), int(A.sum() / 2)


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
        tr = tr[t[tr] <= tmax]
        tr_set, te_set = set(tr.tolist()), set(te.tolist())

        G = c2_graph(n, src, dst, et, t, tr_set, te_set)
        deg_seq = [d for _, d in G.degree()]
        s_c2, d_c2, e_c2 = eval_graph(G, X, y, tr, te)
        ap_c2 = float(pr_auc(y[te], s_c2))

        # M2'：保持度序列的随机重连
        Gm = G.copy()
        try:
            nx.double_edge_swap(Gm, nswap=10 * G.number_of_edges(), max_tries=200 * G.number_of_edges(),
                                seed=0)
        except nx.NetworkXAlgorithmError as e:
            print(f"  [warn] swap failed: {e}")
        deg_m = [d for _, d in Gm.degree()]
        s_m, d_m, e_m = eval_graph(Gm, X, y, tr, te)
        ap_m = float(pr_auc(y[te], s_m))

        # 均匀随机配对（原 M2）作参照
        from common import random_pair_edges
        Ar = np.zeros((n, n))
        for a, b in random_pair_edges(n, e_c2):
            Ar[a, b] = Ar[b, a] = 1.0
        Ahat = Ar + np.eye(n); dg = Ahat.sum(axis=1)
        D = np.diag(1 / np.sqrt(np.maximum(dg, 1e-9)))
        sc = StandardScaler().fit(X[tr])
        Hr = D @ Ahat @ D
        Hr = Hr @ (Hr @ sc.transform(X))
        clf = LogisticRegression(max_iter=1000, class_weight="balanced", C=0.1)
        clf.fit(Hr[tr], y[tr])
        s_r = clf.predict_proba(Hr[te])[:, 1]
        ap_r = float(pr_auc(y[te], s_r))
        d_r = float(np.mean(Ar[te].sum(axis=1)))

        from common import pr_auc as _ap
        from common import RESULTS_DIR as _rd
        rng = np.random.default_rng(0)
        dif = []
        for _ in range(400):
            ii = rng.integers(0, len(te), len(te))
            if len(np.unique(y[te][ii])) < 2:
                continue
            dif.append(_ap(y[te][ii], s_m[ii]) - _ap(y[te][ii], s_c2[ii]))
        dif = np.array(dif)
        ci = [round(float(np.percentile(dif, 2.5)), 4), round(float(np.percentile(dif, 97.5)), 4)]

        rows.append({"frac": frac, "n_test": int(len(te)), "n_edges": e_c2,
                     "C2": {"AP": round(ap_c2, 4), "test_degree": round(d_c2, 3)},
                     "M2_degree_matched": {"AP": round(ap_m, 4), "test_degree": round(d_m, 3),
                                           "degree_seq_equal": bool(sorted(deg_seq) == sorted(deg_m)),
                                           **excess_of(Gm, y, tr_set, te_set)},
                     "M2_uniform": {"AP": round(ap_r, 4), "test_degree": round(d_r, 3)},
                     "paired_M2p_minus_C2": {"diff": round(float(ap_m - ap_c2), 4),
                                             "boot_ci95": ci,
                                             "excludes_zero": bool(ci[0] > 0 or ci[1] < 0)}})
        print(f"  frac={frac:.2f} C2={ap_c2:.3f}(d={d_c2:.2f}) "
              f"M2'={ap_m:.3f}(d={d_m:.2f}) M2={ap_r:.3f}(d={d_r:.2f})", flush=True)

    out = {"experiment": "degree_matched_m2", "dataset": "Real-CATS ETH", "rows": rows}
    c2 = np.mean([r["C2"]["AP"] for r in rows])
    m2p = np.mean([r["M2_degree_matched"]["AP"] for r in rows])
    m2u = np.mean([r["M2_uniform"]["AP"] for r in rows])
    out["means"] = {"C2": round(float(c2), 4), "M2_degree_matched": round(float(m2p), 4),
                    "M2_uniform": round(float(m2u), 4)}
    pdiff = [r["paired_M2p_minus_C2"]["diff"] for r in rows]
    out["paired_summary"] = {"mean_diff": round(float(np.mean(pdiff)), 4),
                             "n_splits": len(pdiff)}
    out["verdict"] = {
        "gap_C2_minus_M2p": round(float(c2 - m2p), 4),
        "gap_M2p_minus_M2u": round(float(m2p - m2u), 4),
        "degree_matched_recovers_C2": bool(c2 - m2p < 0.10),
        "note": ("If M2' stays near the uniform M2, degree is not the channel. If it rises "
                 "toward C2, degree matters and the claim must be softened."),
    }
    out["status"] = "PASS"
    p = Path(args.outdir) / "degree_matched_m2.json"
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    write_provenance("degree_matched_m2", {"status": "PASS"})
    print(json.dumps(out["means"], indent=1), json.dumps(out["verdict"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
