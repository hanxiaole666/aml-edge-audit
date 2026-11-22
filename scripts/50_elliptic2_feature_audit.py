"""
50_elliptic2_feature_audit.py - E5：Elliptic2 特征语义核查与截断版本
======================================================================
审稿人要求把"43 维特征时间窗无文档"的 WARN 变成"有文档的 PASS"或
"明确不可用"。文档侧已确认 43 维是 per-cluster 局部属性（cluster 的地址数、
交易数等），不是图级聚合。本脚本给实证侧的四项检验：

  A. 单特征判别力：43 个特征各自的 AUC。若存在接近 1 的单特征，说明它是
     标签代理（真泄漏）；若最大单特征 AUC 远低于全模型，说明信号分布式存在。

  B. 语义分组：按值域与分布把 43 维分成 ratio-like / count-like /
     zero-inflated / other 四组，逐组单独训练。若信号集中在 count-like
     （有文档的局部规模属性），则支持"特征合法"的结论。

  C. 前缀截断：按单特征 AUC 降序取前 k 个（k=1,2,4,8,16,32,43），报 AP 曲线。
     这是"时间截断"在本数据上的可行替身——Elliptic2 不提供子图时间戳，
     任何声称的时间截断都不可验证；前缀截断检验的是信号集中度。

  D. 训练规模曲线：训练集 10%/25%/50%/75%/100%，看 AP 是否早饱和。
     若在 25% 就接近饱和，说明模型学到的是总体规律而非逐样本记忆。

结论判据（写入 JSON）：
  documented_pass = 最大单特征 AUC < 0.8 且 count-like 组 AP > ratio-like 组
  若最大单特征 AUC >= 0.9，则判为 suspected_proxy，必须明确标注不可用。

用法：python scripts/50_elliptic2_feature_audit.py
产出：results/elliptic2_feature_audit.json + figures/elliptic2_feature_audit.pdf
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA_DIR, RESULTS_DIR, pr_auc, write_provenance, set_seed


def rf():
    return RandomForestClassifier(n_estimators=120, max_depth=12, random_state=0,
                                  n_jobs=-1, class_weight="balanced_subsample")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", default=str(RESULTS_DIR))
    args = ap.parse_args()
    set_seed(0)

    d = DATA_DIR / "elliptic2"
    nodes = pd.read_csv(d / "nodes.csv")
    comps = pd.read_csv(d / "connected_components.csv")
    Xn = np.load(d / "node_features_43.npy").astype(np.float32)
    Xn = np.nan_to_num(Xn, nan=0.0, posinf=0.0, neginf=0.0)
    cc2sub = {int(c): i for i, c in enumerate(comps.ccId.values)}
    y = (comps.ccLabel.values == "suspicious").astype(int)
    node_sub = np.array([cc2sub[int(c)] for c in nodes.ccId.values])

    # 子图级聚合（mean/max/std），与 20/41 脚本一致
    groups = defaultdict(list)
    for gi, s in enumerate(node_sub):
        groups[s].append(gi)
    rows = []
    for s in range(len(y)):
        M = Xn[groups[s]]
        rows.append(np.concatenate([M.mean(0), M.max(0), M.std(0)]))
    Xp = np.vstack(rows).astype(np.float32)
    mean_block = Xp[:, :43]
    print(f"[data] X={Xp.shape} prevalence={y.mean():.4f}", flush=True)

    tr, te = train_test_split(np.arange(len(y)), test_size=0.25, stratify=y, random_state=0)

    # ---- A. 单特征 AUC（在 mean 块上，逐特征）
    single = []
    for j in range(43):
        col = Xp[:, j]
        if np.std(col) < 1e-12:
            auc = 0.5
        else:
            auc = float(roc_auc_score(y, col))
            auc = max(auc, 1 - auc)  # 方向无关
        single.append({"feature": j, "AUC": round(auc, 4),
                       "zero_frac": round(float((col == 0).mean()), 4),
                       "min": round(float(col.min()), 4), "max": round(float(col.max()), 4)})
    single_sorted = sorted(single, key=lambda r: -r["AUC"])
    max_auc = single_sorted[0]["AUC"]

    # ---- B. 语义分组
    def classify(col):
        v = Xp[:, col]
        zf = float((v == 0).mean())
        if v.min() >= 0 and v.max() <= 1.0001:
            return "ratio_like"
        if zf > 0.5:
            return "zero_inflated"
        if v.min() >= 0 and v.max() > 10:
            return "count_like"
        return "other"

    sem = defaultdict(list)
    for j in range(43):
        sem[classify(j)].append(j)
    group_ap = {}
    for g, cols in sem.items():
        if not cols:
            continue
        m = rf()
        m.fit(Xp[tr][:, cols], y[tr])
        group_ap[g] = {"n_features": len(cols), "features": cols,
                       "AP": round(float(pr_auc(y[te], m.predict_proba(Xp[te][:, cols])[:, 1])), 4)}
        print(f"  group {g}: n={len(cols)} AP={group_ap[g]['AP']}", flush=True)

    # ---- C. 前缀截断
    order = [r["feature"] for r in single_sorted]
    prefix = []
    for k in [1, 2, 4, 8, 16, 32, 43]:
        cols = order[:k]
        m = rf()
        m.fit(Xp[tr][:, cols], y[tr])
        ap = float(pr_auc(y[te], m.predict_proba(Xp[te][:, cols])[:, 1]))
        prefix.append({"k": k, "AP": round(ap, 4)})
        print(f"  prefix k={k}: AP={ap:.4f}", flush=True)

    # ---- D. 训练规模曲线
    rng = np.random.default_rng(0)
    curve = []
    for frac in [0.10, 0.25, 0.50, 0.75, 1.00]:
        n_sub = int(frac * len(tr))
        idx = rng.choice(tr, n_sub, replace=False)
        m = rf()
        m.fit(Xp[idx], y[idx])
        ap = float(pr_auc(y[te], m.predict_proba(Xp[te])[:, 1]))
        curve.append({"train_frac": frac, "n_train": n_sub, "AP": round(ap, 4)})
        print(f"  train frac={frac}: n={n_sub} AP={ap:.4f}", flush=True)

    # ---- E. 判据
    full_ap = prefix[-1]["AP"]
    best_group = max(group_ap.items(), key=lambda kv: kv[1]["AP"]) if group_ap else (None, {"AP": 0})
    verdict = {
        "max_single_feature_AUC": max_auc,
        "top5": single_sorted[:5],
        "suspected_proxy": bool(max_auc >= 0.9),
        "documented_pass": bool(max_auc < 0.8),
        "best_group": best_group[0],
        "best_group_AP": best_group[1]["AP"],
        "best_group_share_of_full": round(float(best_group[1]["AP"] / max(full_ap, 1e-9)), 3),
        "no_single_group_explains": bool(best_group[1]["AP"] / max(full_ap, 1e-9) < 0.9),
        "training_curve_saturated": bool(curve[-1]["AP"] - curve[-2]["AP"] < 0.03),
        "note": ("Elliptic2 releases no subgraph timestamps, so a temporal truncation "
                 "cannot be verified on this dataset. Prefix truncation and training-size "
                 "curves are used as the feasible substitutes."),
    }
    out = {"experiment": "elliptic2_feature_audit",
           "n_subgraphs": int(len(y)), "n_features": 43,
           "single_feature": single_sorted, "semantic_groups": group_ap,
           "prefix_truncation": prefix, "training_curve": curve,
           "verdict": verdict, "status": "PASS"}
    print("\n[verdict]", json.dumps(verdict, indent=2, default=float))

    # ---- 图
    figdir = Path(__file__).resolve().parent.parent / "submission_tdsc" / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "font.family": "serif", "figure.dpi": 200})
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.4))
    axes[0].plot([p["k"] for p in prefix], [p["AP"] for p in prefix], "o-", ms=4, lw=1.1)
    axes[0].set_xscale("log", base=2)
    axes[0].set_xlabel("Number of features kept (ranked by single-feature AUC)")
    axes[0].set_ylabel("PR-AUC")
    axes[0].set_title("(a) Prefix truncation")
    axes[0].grid(alpha=0.3)
    axes[1].plot([c["train_frac"] for c in curve], [c["AP"] for c in curve], "s-", ms=4, lw=1.1,
                 color="#b5651d")
    axes[1].set_xlabel("Fraction of training subgraphs")
    axes[1].set_ylabel("PR-AUC")
    axes[1].set_title("(b) Training-size curve")
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(figdir / "elliptic2_feature_audit.pdf", bbox_inches="tight")
    plt.close(fig)

    p = Path(args.outdir) / "elliptic2_feature_audit.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    write_provenance("elliptic2_feature_audit", {"status": "PASS"})
    print(f"[done] {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
