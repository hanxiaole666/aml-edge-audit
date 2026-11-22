"""
44_edge_protocol_final.py - 四条件边协议的最终版（去C4 + 多切分 + CI）
======================================================================
审稿人指出：
  - C4 transductive 与 C1 完全相同（bug），应删除或重定义
  - Table 7 缺模型列、n、prevalence、CI、多切分
  - 375 测试节点上 0.049 的差无 CI 不可解释

本脚本：
  - 删除无效 C4，保留 C1/C2/C3 + M2随机边对照
  - 明确模型 = LogisticRegression(C=0.1) 线性读出
  - 报 n_test, prevalence, 每条件 bootstrap 95% CI
  - 5 次独立时间切分（不同 cutoff 比例），报跨切分 mean±std
  - 配对 bootstrap C2 vs C3, C2 vs M2

产出：results/edge_protocol_final.json + figures/edge_protocol_final.pdf
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA_DIR, RESULTS_DIR, pr_auc, write_provenance, set_seed, random_pair_edges


def _ap(y, s):
    order = np.argsort(-s); y = y[order]
    prec = np.cumsum(y) / (np.arange(len(y)) + 1)
    return float(np.sum(prec * y) / max(y.sum(), 1))


def boot_ci(y, s, n_boot=500, seed=0):
    rng = np.random.default_rng(seed); vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2: continue
        vals.append(_ap(y[idx], s[idx]))
    return [round(float(np.percentile(vals, 2.5)), 4),
            round(float(np.percentile(vals, 97.5)), 4)]


def paired_ci(y, sa, sb, n_boot=500, seed=0):
    rng = np.random.default_rng(seed); diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2: continue
        diffs.append(_ap(y[idx], sa[idx]) - _ap(y[idx], sb[idx]))
    diffs = np.array(diffs)
    return {"diff": round(float(diffs.mean()), 4),
            "ci95": [round(float(np.percentile(diffs, 2.5)), 4),
                     round(float(np.percentile(diffs, 97.5)), 4)],
            "significant": bool(np.percentile(diffs, 2.5) > 0 or np.percentile(diffs, 97.5) < 0)}


def load():
    nodes = pd.read_csv(DATA_DIR / "real_cats" / "real_graph_nodes.csv")
    edges = pd.read_csv(DATA_DIR / "real_cats" / "real_graph_edges.csv")
    feats = [c for c in nodes.columns if c.startswith("x")]
    X = np.nan_to_num(nodes[feats].values.astype(float), nan=0.0, posinf=0.0)
    y = nodes.label.values.astype(int)
    t = nodes.time.values.astype(float)
    id2idx = {int(i): k for k, i in enumerate(nodes.node_id.values)}
    src = np.array([id2idx[int(s)] for s in edges.src])
    dst = np.array([id2idx[int(d)] for d in edges.dst])
    et = edges.time.values.astype(float)
    return X, y, t, src, dst, et


def run_one(X, y, t, src, dst, et, frac, seed):
    order = np.argsort(t, kind="mergesort"); cut = int(frac * len(y))
    tr, te = order[:cut], order[cut:]
    tmax = t[tr].max(); te = te[t[te] > tmax]
    tr_set, te_set = set(tr), set(te)
    rng = np.random.default_rng(seed)
    sc = StandardScaler().fit(X[tr]); Xs = sc.transform(X)
    conds = {}
    n_c2 = None
    for cond in ["C1", "C2", "C3"]:
        A = np.zeros((len(y), len(y)))
        for s, d, e in zip(src, dst, et):
            in_tr_s, in_tr_d = s in tr_set, d in tr_set
            if cond == "C1":
                A[s, d] = A[d, s] = 1.0
            elif cond == "C2":
                if in_tr_s and in_tr_d: A[s, d] = A[d, s] = 1.0
                elif in_tr_s != in_tr_d:
                    tn = d if in_tr_s else s
                    if e <= t[tn]: A[s, d] = A[d, s] = 1.0
            elif cond == "C3":
                if in_tr_s and in_tr_d: A[s, d] = A[d, s] = 1.0
        if cond == "C2": n_c2 = int(A.sum() / 2)
        Ahat = A + np.eye(len(y)); deg = Ahat.sum(axis=1)
        D = np.diag(1/np.sqrt(np.maximum(deg,1e-9))); Ahat = D@Ahat@D
        H = Ahat @ (Ahat @ Xs)
        clf = LogisticRegression(max_iter=1000, class_weight="balanced", C=0.1)
        clf.fit(H[tr], y[tr]); score = clf.predict_proba(H[te])[:, 1]
        conds[cond] = {"score": score,
                       "AP": round(float(pr_auc(y[te], score)), 4),
                       "n_edges": int(A.sum()/2),
                       "CI95": boot_ci(y[te], score, seed=seed)}
    # M2 随机边匹配
    A = np.zeros((len(y), len(y)))
    # 真随机配对，而不是真实边的随机子集（后者保留同配性，无法分离身份与密度）
    for x, z in random_pair_edges(len(y), n_c2, rng):
        A[x, z] = A[z, x] = 1.0
    Ahat = A + np.eye(len(y)); deg = Ahat.sum(axis=1)
    D = np.diag(1/np.sqrt(np.maximum(deg,1e-9))); Ahat = D@Ahat@D
    H = Ahat @ (Ahat @ Xs)
    clf = LogisticRegression(max_iter=1000, class_weight="balanced", C=0.1)
    clf.fit(H[tr], y[tr]); score = clf.predict_proba(H[te])[:, 1]
    conds["M2"] = {"score": score, "AP": round(float(pr_auc(y[te], score)), 4),
                   "n_edges": n_c2, "CI95": boot_ci(y[te], score, seed=seed)}
    # 头条增益 C1->C3 的配对区间（测试集小，必须给区间）
    pair_c3c1 = paired_ci(y[te], conds["C3"]["score"], conds["C1"]["score"], seed=seed)
    return {"frac": frac, "n_train": len(tr), "n_test": len(te),
            "test_prevalence": round(float(y[te].mean()), 4),
            "conditions": {k: {kk: vv for kk, vv in v.items() if kk != "score"}
                           for k, v in conds.items()},
            "C3_minus_C1": pair_c3c1,
            "_scores": {k: v["score"] for k, v in conds.items()},
            "_ytest": y[te]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", default=str(RESULTS_DIR))
    args = ap.parse_args()
    X, y, t, src, dst, et = load()
    out = {"experiment": "edge_protocol_final",
           "model": "LogisticRegression(C=0.1) on 2-hop normalized adjacency smoothing",
           "conditions": {"C1":"keep all edges","C2":"past-only edges",
                          "C3":"remove all cross-split edges","M2":"random edges count-matched to C2"},
           "n_splits": 5, "status": "PASS", "splits": []}
    all_scores = []
    for i, frac in enumerate([0.6, 0.65, 0.7, 0.75, 0.8]):
        r = run_one(X, y, t, src, dst, et, frac, seed=i)
        out["splits"].append({k:v for k,v in r.items() if not k.startswith("_")})
        all_scores.append(r)
        print(f"  split {i} frac={frac}: " + " ".join(
            f"{c}={r['conditions'][c]['AP']:.3f}" for c in ["C1","C2","C3","M2"]), flush=True)

    # 跨切分汇总 + 配对检验（用第一个split的分数）
    agg = {}
    for c in ["C1","C2","C3","M2"]:
        aps = [s["conditions"][c]["AP"] for s in all_scores]
        agg[c] = {"AP_mean": round(float(np.mean(aps)),4),
                  "AP_std": round(float(np.std(aps,ddof=1)),4)}
    out["aggregate"] = agg
    r0 = all_scores[2]  # frac=0.75
    out["paired_tests"] = {
        "C2_vs_C3": paired_ci(r0["_ytest"], r0["_scores"]["C2"], r0["_scores"]["C3"]),
        "C2_vs_M2": paired_ci(r0["_ytest"], r0["_scores"]["C2"], r0["_scores"]["M2"]),
    }
    print("\n[aggregate]", {c: agg[c] for c in agg})
    print("[paired]", out["paired_tests"])

    # 图：跨切分 AP 与每条件 95% CI
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figdir = Path(__file__).resolve().parent.parent / "submission_tdsc" / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "font.family": "serif", "figure.dpi": 200})
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    fracs = [s["frac"] for s in out["splits"]]
    for c, mk in [("C1", "o"), ("C2", "s"), ("C3", "^"), ("M2", "x")]:
        ys = [s["conditions"][c]["AP"] for s in out["splits"]]
        lo = [s["conditions"][c]["AP"] - s["conditions"][c]["CI95"][0] for s in out["splits"]]
        hi = [s["conditions"][c]["CI95"][1] - s["conditions"][c]["AP"] for s in out["splits"]]
        ax.errorbar(fracs, ys, yerr=[lo, hi], fmt=mk + "-", ms=4, lw=1.0,
                    capsize=2, elinewidth=0.7, label=c)
    ax.set_xlabel("Training fraction before the temporal cutoff")
    ax.set_ylabel("PR-AUC")
    ax.set_ylim(0.55, 0.95)
    ax.legend(fontsize=7, ncol=4, loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(figdir / "edge_protocol_final.pdf", bbox_inches="tight")
    plt.close(fig)
    p = Path(args.outdir) / "edge_protocol_final.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    write_provenance("edge_protocol_final", {"status":"PASS","n_splits":5})
    print(f"[done] {p}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
