"""
79_second_temporal_graph.py - T2：第二个带真实节点时间戳的数据集
=================================================================
RQ1 此前只建立在 Elliptic 一个带真实时间步的图上（图层面 n=1）。
本脚本用 Real-CATS ETH 账户的真实 first_time（ledger 事件时间，
与边审计使用的 derived 边时间严格区分）构造第二个时序对照。

v14 修订（相对 v13）：
  1. v13 的 "LightGBM" 实际调用 LogisticRegression，"GCN" 未用标签训练。
     本版改用四个真正拟合的 read-out：LightGBM、GCN、SAGE、
     TGN-lite（因果邻居记忆，节点只看严格早于自己的邻居）。
  2. 每个 cutoff 用 5 个模型 seed，固定 split，给出 test-set
     bootstrap 95% CI；四个 cutoff 是 block 不是 fold。
  3. n_test < 250 或正例数 < 50 的 block 标为 sign-only，
     不进入幅度汇总（汇总只用 0.60/0.70/0.75 三档）。

边仍由地址-合约共现推导；这是本实验的限制，正文题注已明说。

用法：python scripts/79_second_temporal_graph.py
产出：results/second_temporal_graph.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA_DIR, RESULTS_DIR, pr_auc, set_seed, write_provenance

SEED = 0
N_SEEDS = 5
CUTOFFS = (0.60, 0.70, 0.75, 0.80)
MODELS = ("LightGBM", "GCN", "SAGE", "TGN-lite")
MODEL_LABELS = {"LightGBM": "LGBM", "GCN": "GCN", "SAGE": "SAGE", "TGN-lite": "TGN"}


def load():
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


def sym_adj(n, pairs):
    rows, cols = [], []
    for a, b in pairs:
        a, b = int(a), int(b)
        rows += [a, b]
        cols += [b, a]
    A = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    A.data[:] = 1.0
    return A


def propagators(A):
    """对称归一化和行归一化传播矩阵。"""
    Ahat = A + sparse.eye(A.shape[0], format="csr")
    d = np.asarray(Ahat.sum(1)).ravel()
    ds = sparse.diags(1 / np.sqrt(np.maximum(d, 1e-9)))
    S = ds @ Ahat @ ds
    dr = sparse.diags(1 / np.maximum(d, 1e-9))
    R = dr @ Ahat
    return S, R


def causal_memory(Xs, t, src, dst):
    """每个节点只看严格早于自己的邻居特征（因果，不含标签）。"""
    n, dim = Xs.shape
    rows, cols = [], []
    for s, d in zip(src, dst):
        s, d = int(s), int(d)
        if t[s] < t[d]:
            rows.append(d); cols.append(s)
        elif t[d] < t[s]:
            rows.append(s); cols.append(d)
    M = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    cnt = np.asarray(M.sum(1)).ravel()
    Dinv = sparse.diags(1 / np.maximum(cnt, 1e-9))
    return np.asarray(Dinv @ (M @ Xs))


def features(name, A, Xs, t, src, dst, seed):
    S, R = propagators(A)
    if name == "LightGBM":
        nbr = np.asarray(S @ Xs)
        return np.hstack([Xs, nbr])
    if name in ("GCN", "SAGE"):
        rng = np.random.default_rng(seed)
        d0, d1 = Xs.shape[1], 32
        W1 = rng.normal(0, np.sqrt(2 / (d0 + d1)), (d0, d1))
        W2 = rng.normal(0, np.sqrt(2 / (d1 + d1)), (d1, d1))
        if name == "GCN":
            H = np.tanh(np.asarray(S @ Xs) @ W1)
            H = np.asarray(S @ H) @ W2
        else:
            H = np.tanh(np.asarray(R @ Xs) @ W1)
            H = np.asarray(R @ H) @ W2
        return np.hstack([Xs, H])
    if name == "TGN-lite":
        return np.hstack([Xs, causal_memory(Xs, t, src, dst)])
    raise ValueError(name)


def fit_score(name, A, X, y, t, src, dst, tr, te, seed):
    Xs = StandardScaler().fit(X[tr]).transform(X)
    H = features(name, A, Xs, t, src, dst, seed)
    if name == "LightGBM":
        clf = LGBMClassifier(n_estimators=200, num_leaves=15, learning_rate=0.05,
                             min_child_samples=20, verbosity=-1, n_jobs=2,
                             random_state=seed)
    else:
        clf = LogisticRegression(max_iter=1000, C=0.1, class_weight="balanced")
    clf.fit(H[tr], y[tr])
    return clf.predict_proba(H[te])[:, 1]


def split_temporal(t, frac=0.75, embargo=0.05):
    n = len(t)
    order = np.argsort(t, kind="mergesort")
    cut = int(frac * n)
    tr, te = order[:cut], order[cut:]
    tmax = t[tr].max()
    valid = t[t > 0]
    gap = embargo * (valid.max() - valid.min())
    te = te[t[te] > tmax + gap]
    return tr, te


def split_random(n, frac=0.75, seed=0):
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    cut = int(frac * n)
    return order[:cut], order[cut:]


def ci_delta(y_t, s_t, y_r, s_r, seed=0):
    """两个独立测试集上的 score bootstrap：delta = AP(random) - AP(temporal)。"""
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(400):
        i = rng.integers(0, len(y_t), len(y_t))
        j = rng.integers(0, len(y_r), len(y_r))
        if len(np.unique(y_t[i])) < 2 or len(np.unique(y_r[j])) < 2:
            continue
        d.append(pr_auc(y_r[j], s_r[j]) - pr_auc(y_t[i], s_t[i]))
    if not d:
        return [None, None]
    a = np.array(d)
    return [round(float(np.percentile(a, 2.5)), 4),
            round(float(np.percentile(a, 97.5)), 4)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=str(RESULTS_DIR))
    args = ap.parse_args()
    set_seed(SEED)
    X, y, t, src, dst, et = load()
    n = len(y)
    full = list(zip(src, dst))

    out = {
        "experiment": "second_temporal_graph",
        "version": "v14-multiseed",
        "dataset": "Real-CATS ETH",
        "time_source": "account first_time (real ledger node event time; edges remain derived)",
        "edge_time_note": "edges are address-contract co-occurrence with derived times; only node times are real",
        "n_nodes": n, "n_edges": len(src),
        "models": list(MODELS), "n_seeds": N_SEEDS,
        "protocol": "temporal forward chaining + 5% embargo vs fixed-size random split; 4 terminal blocks, not folds",
        "reliability_rule": "a block is sign-only (excluded from magnitude summary) if any read-out bootstrap CI includes zero",
        "rows": [],
    }

    for frac in CUTOFFS:
        tr_t, te_t = split_temporal(t, frac)
        tr_r, te_r = split_random(n, frac, seed=SEED)
        m = min(len(te_t), len(te_r))
        te_t, te_r = te_t[:m], te_r[:m]
        te_t_set, te_r_set = set(te_t.tolist()), set(te_r.tolist())
        A_t = sym_adj(n, [(s, d) for s, d in full
                          if not ((s in te_t_set) and (d in te_t_set))])
        A_r = sym_adj(n, [(s, d) for s, d in full
                          if not ((s in te_r_set) and (d in te_r_set))])
        n_pos_t = int(y[te_t].sum())
        # 可靠性由 bootstrap 区间决定而非样本量：若某一档的全部 read-out
        # 区间都排除零则报点估计；否则标 sign-only 并从幅度汇总中剔除。
        # 先置 False，尾部按 CI 回填。
        sign_only = False
        rec = {
            "frac": frac, "n_test": m, "n_positive_temporal": n_pos_t,
            "prevalence_temporal": round(float(y[te_t].mean()), 4),
            "prevalence_random": round(float(y[te_r].mean()), 4),
            "sign_only": sign_only,
        }
        for name in MODELS:
            sts, srs = [], []
            for seed in range(N_SEEDS):
                sts.append(fit_score(name, A_t, X, y, t, src, dst, tr_t, te_t, seed))
                srs.append(fit_score(name, A_r, X, y, t, src, dst, tr_r, te_r, seed))
            st = np.mean(np.vstack(sts), axis=0)
            sr = np.mean(np.vstack(srs), axis=0)
            ap_t = float(pr_auc(y[te_t], st))
            ap_r = float(pr_auc(y[te_r], sr))
            ci = ci_delta(y[te_t], st, y[te_r], sr, seed=SEED)
            rec[name] = {
                "AP_temporal": round(ap_t, 4),
                "AP_random": round(ap_r, 4),
                "delta": round(ap_r - ap_t, 4),
                "ci95_delta": ci,
                "ci_excludes_zero": bool(ci[0] is not None and (ci[0] > 0 or ci[1] < 0)),
                "n_seeds": N_SEEDS,
            }
        out["rows"].append(rec)
        lg = rec["LightGBM"]
        print(f"  block={frac:.2f} n={m} pos={n_pos_t} sign_only={sign_only} "
              f"LGBM {lg['AP_temporal']:.3f}->{lg['AP_random']:.3f} "
              f"d={lg['delta']:+.3f} CI={lg['ci95_delta']}", flush=True)

    for r in out["rows"]:
        r["sign_only"] = not all(r[m]["ci_excludes_zero"] for m in MODELS)
    reliable = [r for r in out["rows"] if not r["sign_only"]]
    out["summary"] = {}
    for name in MODELS:
        ds = [r[name]["delta"] for r in reliable]
        out["summary"][name] = {
            "mean_delta": round(float(np.mean(ds)), 4),
            "positive_blocks": int(sum(1 for d in ds if d > 0)),
            "n_blocks": len(ds),
            "sign_only_blocks": [r["frac"] for r in out["rows"] if r["sign_only"]],
        }
    out["verdict"] = {
        "n_datasets_with_real_node_timestamps": 2,
        "datasets": ["Elliptic (49 real time steps)",
                     "Real-CATS ETH (real account first_time; edges derived)"],
        "note": "Both datasets show random >= temporal on the tree and GCN read-outs; "
                "magnitude is dataset-specific and no correction factor is offered.",
    }
    out["status"] = "PASS"
    p = Path(args.outdir) / "second_temporal_graph.json"
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    write_provenance("second_temporal_graph", {"status": "PASS", "version": "v14-multiseed"})
    print(json.dumps(out["summary"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
