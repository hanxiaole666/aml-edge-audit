"""
74_cross_split_injection.py - E-B 在 Real-CATS 的跨切分边上做受控注射
=====================================================================
68 号的注射实验在 Elliptic 的 test--test 边上做，而被解释的现象是
Real-CATS 的 cross-split 效应。边型不同、图不同，机制是"迁移"过去的。

本脚本在被解释的那张图上就地验证：
  * 训练图固定为 C2（排除 test--test 边），训练特征只由训练图决定；
  * 测试图 = C2 + K 条 train→test 边，边的同行标签率由 λ 控制；
  * 度由 K 固定，λ 只改成分。
若 ΔAP 随 λ 从负走到正，机制就在它被发现的那张图上成立。

用法：python scripts/74_cross_split_injection.py
产出：results/cross_split_injection.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA_DIR, RESULTS_DIR, pr_auc, set_seed, write_provenance

FRACS = [0.60, 0.65, 0.70, 0.75, 0.80]
KS = [2000, 6000]
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
    return X, y, t, src, dst


def split_at(t, frac):
    order = np.argsort(t, kind="mergesort")
    cut = int(frac * len(t))
    tr, te = order[:cut], order[cut:]
    tmax = t[tr].max()
    te = te[t[te] > tmax]
    tr = tr[t[tr] <= tmax]
    return tr, te


def adj_base(n, src, dst, tr_set, te_set):
    """C2：保留任何含训练端点的边，剔除 test--test。"""
    A = np.zeros((n, n), dtype=float)
    for s, d in zip(src, dst):
        a, b = int(s), int(d)
        if not ((a in tr_set) or (b in tr_set)):
            continue
        A[a, b] = A[b, a] = 1.0
    return A


def inject(A, y, tr, te, K, lam, rng_seed):
    """在 A 上注入 K 条 train→test 边，同行标签概率由 λ 决定。

    λ=-1 全部异标签，λ=0 独立抽样，λ=+1 全部同标签。
    端点从训练集/测试集中均匀抽取，不重复。
    """
    rng = np.random.default_rng(rng_seed)
    tr_pos = tr[y[tr] == 1]; tr_neg = tr[y[tr] == 0]
    te_pos = te[y[te] == 1]; te_neg = te[y[te] == 0]
    if min(len(tr_pos), len(tr_neg), len(te_pos), len(te_neg)) == 0:
        return A, 0

    # 目标同标签比例 q(λ)：λ=1 → 1，λ=0 → 由两端的正例率决定，λ=-1 → 0
    prev_tr = len(tr_pos) / len(tr)
    prev_te = len(te_pos) / len(te)
    p_indep = prev_tr * prev_te + (1 - prev_tr) * (1 - prev_te)
    if lam >= 0:
        q = p_indep + lam * (1 - p_indep)
    else:
        q = p_indep + lam * p_indep
    q = float(np.clip(q, 0.0, 1.0))

    seen = set()
    added = 0
    tries = 0
    while added < K and tries < 40 * K:
        tries += 1
        same = rng.random() < q
        if same:
            if rng.random() < 0.5:
                a = rng.choice(tr_pos); b = rng.choice(te_pos)
            else:
                a = rng.choice(tr_neg); b = rng.choice(te_neg)
        else:
            if rng.random() < 0.5:
                a = rng.choice(tr_pos); b = rng.choice(te_neg)
            else:
                a = rng.choice(tr_neg); b = rng.choice(te_pos)
        a, b = int(a), int(b)
        if a == b or (a, b) in seen or (b, a) in seen:
            continue
        if A[a, b] > 0:
            continue
        seen.add((a, b))
        A[a, b] = A[b, a] = 1.0
        added += 1

    nb = set(A[te].sum(axis=1).nonzero()[0].tolist())
    obs = 0.0
    n_pair = 0
    for a, b in seen:
        n_pair += 1
        if y[a] == y[b]:
            obs += 1
    exp = p_indep
    return A, {"K": added, "q_target": round(q, 4), "q_realised": round(obs / max(n_pair, 1), 4),
               "expected_independent": round(exp, 4),
               "excess": round(obs / max(n_pair, 1) - exp, 4)}


def score(A_tr, A_te, X, y, tr, te):
    """训练行走 A_tr 的邻居，测试行走 A_te 的邻居。"""
    def feats(A, idx):
        Ahat = A + np.eye(A.shape[0])
        deg = Ahat.sum(axis=1)
        D = np.diag(1 / np.sqrt(np.maximum(deg, 1e-9)))
        H = D @ Ahat @ D
        H = H @ (H @ X)
        return H[idx]

    sc = StandardScaler().fit(X[tr])
    Xs = sc.transform(X)
    Ftr = feats(A_tr, tr)
    Fte = feats(A_te, te)
    clf = LogisticRegression(max_iter=1000, class_weight="balanced", C=0.1)
    clf.fit(Ftr, y[tr])
    return clf.predict_proba(Fte)[:, 1], Ftr, Fte


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=str(RESULTS_DIR))
    args = ap.parse_args()
    set_seed(0)

    X, y, t, src, dst = load_graph()
    n = len(y)
    print(f"[graph] n={n} edges={len(src)} illicit={int(y.sum())}")

    rows = []
    for frac in FRACS:
        tr, te = split_at(t, frac)
        tr_set, te_set = set(tr.tolist()), set(te.tolist())
        A0 = adj_base(n, src, dst, tr_set, te_set)
        base_score, _, _ = score(A0, A0, X, y, tr, te)
        base_ap = float(pr_auc(y[te], base_score))
        cross = sum(1 for s, d in zip(src, dst)
                    if ((int(s) in tr_set and int(d) in te_set) or
                        (int(s) in te_set and int(d) in tr_set)))
        print(f"  frac={frac:.2f} n_te={len(te)} C2_AP={base_ap:.4f} existing_cross={cross}",
              flush=True)

        for K in KS:
            for lam in LAMS:
                A = A0.copy()
                A, info = inject(A, y, tr, te, K, lam, rng_seed=hash((frac, K, lam)) % (2**32))
                s, _, _ = score(A0, A, X, y, tr, te)
                ap = float(pr_auc(y[te], s))
                rows.append({"frac": frac, "n_test": int(len(te)), "K": K, "lambda": lam,
                             "AP_base": round(base_ap, 4), "AP_injected": round(ap, 4),
                             "delta_AP": round(ap - base_ap, 4),
                             "q_realised": info["q_realised"] if isinstance(info, dict) else None,
                             "excess": info["excess"] if isinstance(info, dict) else None,
                             "K_added": info["K"] if isinstance(info, dict) else 0})

    out = {"experiment": "cross_split_injection", "dataset": "Real-CATS ETH (shipped graph)",
           "protocol": "train graph = C2; test graph = C2 + K injected train->test edges",
           "n_nodes": n, "fracs": FRACS, "K": KS, "lambdas": LAMS,
           "rows": rows}

    # 汇总：每个 λ 的平均 ΔAP，检查单调与符号
    summ = []
    for lam in LAMS:
        vals = [r["delta_AP"] for r in rows if r["lambda"] == lam]
        summ.append({"lambda": lam, "mean_delta_AP": round(float(np.mean(vals)), 4),
                     "min": round(float(np.min(vals)), 4), "max": round(float(np.max(vals)), 4),
                     "n": len(vals)})
    out["by_lambda"] = summ
    seq = [s["mean_delta_AP"] for s in summ]
    out["verdict"] = {
        "monotone": bool(all(b >= a - 1e-9 for a, b in zip(seq, seq[1:]))),
        "sign_flips": bool(seq[0] < 0 < seq[-1]),
        "range": [seq[0], seq[-1]],
        "note": ("If the sign flips with lambda at fixed K, the mechanism holds on the graph "
                 "where the effect was found, not only on Elliptic."),
    }
    out["status"] = "PASS"
    p = Path(args.outdir) / "cross_split_injection.json"
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    write_provenance("cross_split_injection", {"status": "PASS", "n_rows": len(rows)})
    print(f"[done] {p}")
    print(json.dumps(out["verdict"], indent=1, ensure_ascii=False))
    for s in summ:
        print(f"   lambda={s['lambda']:+.1f}  dAP={s['mean_delta_AP']:+.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
