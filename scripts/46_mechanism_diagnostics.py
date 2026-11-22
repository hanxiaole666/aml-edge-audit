"""
46_mechanism_diagnostics.py - E3 机制诊断（把"有害结构"从一句话变成机制）
======================================================================
E2 已确立 C1 < C2 < C3 且 M2（随机等量边）低于 C2，说明效应来自"保留哪些边"
而非"边数多少"。E3 追问：那些被 C1 保留、被 C2 删除的跨切分边，究竟通过什么
通道伤害测试期排序？四个候选通道，逐一量化：

  D1 度通道：C1 让测试节点获得大量跨切分邻居，度数抬升 → 2-hop 平滑把测试节点
     的特征过度平均化。报各条件测试节点度分布（mean/median/p90/max），以及
     "度本身"对测试标签的 AUC（度若是标签代理，则排序被度支配）。

  D2 邻居标签通道（仅用训练标签，不看测试标签）：对每个测试节点统计其邻居中
     训练正例占比 frac_pos。报覆盖率、均值，以及 frac_pos 对测试标签的 AUC。
     若 C1 的 AUC 明显低于 C2，说明 C1 引入的邻居标签构成与测试期标签不一致，
     即"训练期的标签格局被搬到测试期"，这是分布错配而非未来信息。

  D3 表示通道：各条件 2-hop 表示 H 与最干净条件 C3 的 RBF-MMD（测试集上）。
     MMD 越大，表示被跨切分边拉得越远。

  D4 归一化通道：三种传播算子（raw sum / 对称归一化 / 行归一化）下重跑
     C1/C2/C3。若排序在三种算子下都保持，则 C1<C2 不是归一化伪影。

另报跨切分边的标签同质性：C1 保留的 tr-te 边两端标签一致率 vs 随机配对期望，
用于区分"标签同质传导"与"纯拓扑噪声"。

用法：python scripts/46_mechanism_diagnostics.py
产出：results/mechanism_diagnostics.json + figures/mechanism_diagnostics.pdf
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
from scipy.spatial.distance import cdist
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA_DIR, RESULTS_DIR, pr_auc, write_provenance, set_seed, random_pair_edges


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


def build_adj(n, src, dst, et, t, tr_set, te_set, condition, rng, target_edges=None):
    A = np.zeros((n, n))
    if condition == "C1":
        for s, d in zip(src, dst):
            A[s, d] = A[d, s] = 1.0
    elif condition == "C2":
        for s, d, e in zip(src, dst, et):
            if s in tr_set and d in tr_set:
                A[s, d] = A[d, s] = 1.0
            elif (s in tr_set) != (d in tr_set):
                tn = d if s in tr_set else s
                if e <= t[tn]:
                    A[s, d] = A[d, s] = 1.0
    elif condition == "C3":
        for s, d in zip(src, dst):
            if s in tr_set and d in tr_set:
                A[s, d] = A[d, s] = 1.0
    elif condition == "M2":
        for s, d in random_pair_edges(n, target_edges, rng):
            A[s, d] = A[d, s] = 1.0
    return A


def propagate(A, Xs, mode):
    """三种传播算子。返回 2-hop 表示。"""
    n = A.shape[0]
    Ahat = A + np.eye(n)
    if mode == "sym":
        deg = Ahat.sum(axis=1)
        D = np.diag(1.0 / np.sqrt(np.maximum(deg, 1e-9)))
        P = D @ Ahat @ D
    elif mode == "row":
        deg = Ahat.sum(axis=1)
        P = np.diag(1.0 / np.maximum(deg, 1e-9)) @ Ahat
    elif mode == "raw":
        P = Ahat
    else:
        raise ValueError(mode)
    return P @ (P @ Xs)


def mmd_rbf(Xa, Xb, max_n=400, seed=0):
    rng = np.random.default_rng(seed)
    if len(Xa) > max_n:
        Xa = Xa[rng.choice(len(Xa), max_n, replace=False)]
    if len(Xb) > max_n:
        Xb = Xb[rng.choice(len(Xb), max_n, replace=False)]
    Z = np.vstack([Xa, Xb])
    d2 = cdist(Z, Z, "sqeuclidean")
    med = np.median(d2[d2 > 0])
    gamma = 1.0 / max(med, 1e-12)
    K = np.exp(-gamma * d2)
    n = len(Xa)
    Kxx, Kyy, Kxy = K[:n, :n], K[n:, n:], K[:n, n:]
    return float(Kxx.mean() + Kyy.mean() - 2.0 * Kxy.mean())


def fit_score(H, y, tr, te):
    clf = LogisticRegression(max_iter=3000, class_weight="balanced", C=0.1)
    clf.fit(H[tr], y[tr])
    s = clf.predict_proba(H[te])[:, 1]
    return s, float(pr_auc(y[te], s))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", default=str(RESULTS_DIR))
    ap.add_argument("--frac", type=float, default=0.75)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    set_seed(args.seed)

    X, y, t, src, dst, et = load()
    n = len(y)
    order = np.argsort(t, kind="mergesort")
    cut = int(args.frac * n)
    tr, te = order[:cut], order[cut:]
    tmax = t[tr].max()
    te = te[t[te] > tmax]
    tr_set, te_set = set(tr.tolist()), set(te.tolist())
    rng = np.random.default_rng(args.seed)
    sc = StandardScaler().fit(X[tr])
    Xs = sc.transform(X)

    out = {
        "experiment": "mechanism_diagnostics",
        "split": {"frac": args.frac, "n_train": int(len(tr)), "n_test": int(len(te)),
                  "test_prevalence": round(float(y[te].mean()), 4)},
        "conditions": {}, "status": "PASS",
    }

    # ---- 先建 C2 以取边数，供 M2 匹配
    A_c2 = build_adj(n, src, dst, et, t, tr_set, te_set, "C2", rng)
    n_c2 = int(A_c2.sum() / 2)
    conds = {
        "C1": build_adj(n, src, dst, et, t, tr_set, te_set, "C1", rng),
        "C2": A_c2,
        "C3": build_adj(n, src, dst, et, t, tr_set, te_set, "C3", rng),
        "M2": build_adj(n, src, dst, et, t, tr_set, te_set, "M2", rng, target_edges=n_c2),
    }

    # 参照表示：C3
    H_ref = propagate(conds["C3"], Xs, "sym")[te]

    for name, A in conds.items():
        H = propagate(A, Xs, "sym")
        s, ap = fit_score(H, y, tr, te)
        deg_all = A.sum(axis=1)
        deg_te = deg_all[te]
        # 度对标签的 AUC（测试集内）
        deg_auc = (float(roc_auc_score(y[te], deg_te))
                   if len(np.unique(y[te])) == 2 else float("nan"))
        # 邻居标签构成（仅训练标签）
        tr_idx = np.array(sorted(tr_set))
        A_te_tr = A[np.ix_(te, tr_idx)]
        pos = (y[tr_idx] == 1).astype(float)
        n_pos = A_te_tr @ pos
        n_nbr = A_te_tr.sum(axis=1)
        frac_pos = np.where(n_nbr > 0, n_pos / np.maximum(n_nbr, 1), np.nan)
        has = n_nbr > 0
        nb_auc = (float(roc_auc_score(y[te][has], frac_pos[has]))
                  if has.sum() > 10 and len(np.unique(y[te][has])) == 2 else float("nan"))
        # 邻居比例与度
        out["conditions"][name] = {
            "n_edges": int(A.sum() / 2),
            "PR_AUC": round(ap, 4),
            "test_degree": {
                "mean": round(float(deg_te.mean()), 3),
                "median": float(np.median(deg_te)),
                "p90": float(np.percentile(deg_te, 90)),
                "max": float(deg_te.max()),
                "zero_frac": round(float((deg_te == 0).mean()), 4),
            },
            "degree_label_AUC": round(deg_auc, 4) if np.isfinite(deg_auc) else None,
            "neighbor_label": {
                "coverage": round(float(has.mean()), 4),
                "mean_frac_pos": round(float(np.nanmean(frac_pos)), 4) if has.any() else None,
                "frac_pos_label_AUC": round(nb_auc, 4) if np.isfinite(nb_auc) else None,
            },
            "MMD_to_C3": round(mmd_rbf(H[te], H_ref), 6),
        }
        print(f"  {name}: AP={ap:.3f} deg={deg_te.mean():.2f} "
              f"nbr_cov={has.mean():.2f} nbrAUC={nb_auc:.3f} "
              f"MMD={out['conditions'][name]['MMD_to_C3']:.4f}", flush=True)

    # ---- D4 归一化通道消融
    norm_ablation = {}
    for mode in ["sym", "row", "raw"]:
        row = {}
        for name in ["C1", "C2", "C3", "M2"]:
            H = propagate(conds[name], Xs, mode)
            _, ap = fit_score(H, y, tr, te)
            row[name] = round(ap, 4)
        row["order_holds"] = bool(row["C1"] < row["C2"] < row["C3"])
        norm_ablation[mode] = row
        print(f"  norm={mode}: {row}", flush=True)
    out["normalization_ablation"] = norm_ablation

    # ---- 跨切分边标签同质性（C1 的 tr-te 边）
    tr_arr = np.array(sorted(tr_set))
    tr_pos_frac = float(y[tr_arr].mean())
    cross_same, cross_n = 0, 0
    for s, d in zip(src, dst):
        if (s in tr_set) != (d in tr_set):
            cross_n += 1
            a, b = (s, d) if s in tr_set else (d, s)
            cross_same += int(y[a] == y[b])
    # 随机配对期望：训练正例率与测试正例率的混合
    p_tr, p_te = float(y[tr_arr].mean()), float(y[te].mean())
    expected_same = p_tr * p_te + (1 - p_tr) * (1 - p_te)
    out["cross_edge_homophily"] = {
        "n_cross_edges": int(cross_n),
        "observed_same_label": round(cross_same / max(cross_n, 1), 4),
        "expected_if_independent": round(float(expected_same), 4),
        "excess": round(cross_same / max(cross_n, 1) - float(expected_same), 4),
    }
    print(f"  cross-edge homophily: obs={out['cross_edge_homophily']['observed_same_label']} "
          f"exp={out['cross_edge_homophily']['expected_if_independent']}", flush=True)

    # ---- 结论判据
    c = out["conditions"]
    out["verdict"] = {
        "density_alone_insufficient": bool(c["M2"]["PR_AUC"] < c["C2"]["PR_AUC"]),
        "over_smoothing": bool(c["C1"]["test_degree"]["mean"] > 3 * max(c["C3"]["test_degree"]["mean"], 0.1)),
        "neighbor_label_mismatch": bool(
            c["C1"]["neighbor_label"]["frac_pos_label_AUC"] is not None
            and c["C2"]["neighbor_label"]["frac_pos_label_AUC"] is not None
            and c["C1"]["neighbor_label"]["frac_pos_label_AUC"]
            < c["C2"]["neighbor_label"]["frac_pos_label_AUC"]),
        "representation_shift": bool(c["C1"]["MMD_to_C3"] > c["C2"]["MMD_to_C3"]),
        "normalization_robust": bool(all(v["order_holds"] for v in norm_ablation.values())),
    }

    # ---- 图：度分布 + 邻居标签构成
    figdir = Path(__file__).resolve().parent.parent / "submission_tdsc" / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "font.family": "serif", "figure.dpi": 200})
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.4))
    labels = ["C1", "C2", "C3", "M2"]
    data = [conds[k].sum(axis=1)[te] for k in labels]
    bp = axes[0].boxplot(data, labels=labels, showfliers=False, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("#cfe3f3")
    axes[0].set_ylabel("Test-node degree")
    axes[0].set_title("(a) Degree under each condition")
    axes[0].grid(alpha=0.3, axis="y")

    for k, mk in zip(labels, ["o", "s", "^", "x"]):
        A = conds[k]
        A_te_tr = A[np.ix_(te, tr_arr)]
        n_nbr = A_te_tr.sum(axis=1)
        fp = np.where(n_nbr > 0, (A_te_tr @ (y[tr_arr] == 1)) / np.maximum(n_nbr, 1), np.nan)
        axes[1].hist(fp[np.isfinite(fp)], bins=20, alpha=0.45, label=k, density=True)
    axes[1].set_xlabel("Fraction of positive training neighbours")
    axes[1].set_title("(b) Neighbour label composition")
    axes[1].legend(fontsize=7)
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(figdir / "mechanism_diagnostics.pdf", bbox_inches="tight")
    plt.close(fig)

    p = Path(args.outdir) / "mechanism_diagnostics.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    write_provenance("mechanism_diagnostics", {"status": "PASS", "frac": args.frac})
    print(f"[done] {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
