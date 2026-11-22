"""
36_extrapolation_curve.py - 外推距离曲线 + embargo 敏感性（E2）
==============================================================
审稿人要求：把"时间泄漏被低估"做成可外推曲线，而非单点。
数据：Real-CATS ETH 真实 first_time（2016-2024，8年跨度，真实时间戳）

实验1 外推距离扫描：
  train = 截至 t0 的样本；test = (t0, t0+Δ]，Δ ∈ {1,3,7,14,30,90,180,365} 天
  对比 random split 的平坦线，画 PR-AUC 随外推距离衰减曲线

实验2 embargo 敏感性：
  embargo ∈ {0,1%,5%,10%,20%} 时 random-vs-temporal gap 如何变化

用法：python scripts/36_extrapolation_curve.py
产出：results/extrapolation_curve.json + figures/extrapolation.pdf + figures/embargo.pdf
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA_DIR, RESULTS_DIR, pr_auc, write_provenance, set_seed

DAY = 86400.0


def train_eval(X, y, t, tr, te):
    if len(te) < 20 or len(np.unique(y[te])) < 2:
        return None
    sc = StandardScaler().fit(X[tr])
    m = LogisticRegression(max_iter=1000, class_weight="balanced", C=0.1)
    m.fit(sc.transform(X[tr]), y[tr])
    return float(pr_auc(y[te], m.predict_proba(sc.transform(X[te]))[:, 1]))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", default=str(RESULTS_DIR))
    args = ap.parse_args()
    set_seed(0)

    df = pd.read_csv(DATA_DIR / "real_cats_eth.csv")
    feats = [c for c in df.columns if c not in {"label", "time", "address", "etherscan_checked", "watchback_checked", "gs_checked"}
             and pd.api.types.is_numeric_dtype(df[c])]
    X = df[feats].values.astype(float)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0)
    y = df.label.values.astype(int)
    t = df.time.values.astype(float)
    valid = t > 0
    t0 = np.percentile(t[valid], 60)  # 训练截止点

    out = {"experiment": "extrapolation_curve",
           "data": "Real-CATS ETH, real first_time timestamps (2016-2024)",
           "t0": float(t0), "extrapolation": {}, "embargo": {}, "status": "PASS"}

    # 实验1: 外推距离扫描
    deltas_days = [1, 3, 7, 14, 30, 90, 180, 365]
    print("[extrap] scanning extrapolation horizons...", flush=True)
    for d in deltas_days:
        tr = np.where(valid & (t <= t0))[0]
        te = np.where(valid & (t > t0) & (t <= t0 + d * DAY))[0]
        ap = train_eval(X, y, t, tr, te)
        if ap:
            out["extrapolation"][str(d)] = {"PR-AUC": round(ap, 4), "n_test": len(te)}
            print(f"  +{d:3d}d: PR-AUC={ap:.4f} n={len(te)}", flush=True)
    # random 基线（平坦线参考）
    rtr, rte = train_test_split(np.arange(len(y)), test_size=0.25, random_state=0, stratify=y)
    rand_ap = train_eval(X, y, t, rtr, rte)
    out["random_reference"] = round(rand_ap, 4)

    # 实验2: embargo 敏感性
    print("[extrap] embargo sensitivity...", flush=True)
    span = t[valid].max() - t[valid].min()
    for eb in [0.0, 0.01, 0.05, 0.10, 0.20]:
        tr = np.where(valid & (t <= t0))[0]
        te = np.where(valid & (t > t0 + eb * span))[0]
        ap = train_eval(X, y, t, tr, te)
        if ap:
            out["embargo"][str(eb)] = {"PR-AUC": round(ap, 4), "n_test": len(te)}
            print(f"  embargo={eb:.0%}: PR-AUC={ap:.4f} n={len(te)}", flush=True)

    # 图
    figdir = Path(__file__).resolve().parent.parent / "submission_tdsc" / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "font.family": "serif", "figure.dpi": 200})

    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    ds = [d for d in deltas_days if str(d) in out["extrapolation"]]
    vals = [out["extrapolation"][str(d)]["PR-AUC"] for d in ds]
    ax.plot(ds, vals, "o-", color="#404040", label="Temporal extrapolation")
    ax.axhline(rand_ap, ls="--", color="#808080", label=f"Random split ({rand_ap:.3f})")
    ax.set_xscale("log")
    ax.set_xlabel("Extrapolation horizon (days)")
    ax.set_ylabel("PR-AUC")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(figdir / "extrapolation.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    ebs = [e for e in [0.0, 0.01, 0.05, 0.10, 0.20] if str(e) in out["embargo"]]
    evs = [out["embargo"][str(e)]["PR-AUC"] for e in ebs]
    ax.plot([e * 100 for e in ebs], evs, "s-", color="#404040")
    ax.set_xlabel("Embargo (% of time span)")
    ax.set_ylabel("PR-AUC")
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(figdir / "embargo.pdf", bbox_inches="tight")
    plt.close(fig)

    p = Path(args.outdir) / "extrapolation_curve.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    write_provenance("extrapolation_curve", {"status": "PASS"})
    print(f"[done] {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
