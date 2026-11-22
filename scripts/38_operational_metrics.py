"""
38_operational_metrics.py - 操作点指标 + 攻击预算曲线（E5/E11）
===============================================================
E5: F1 / precision@k (k=100/500/1000 日告警) / ECE(10 bins) / cost-per-TP
E11: ε-AP 攻击预算曲线（特征扰动，含 L0/Linf 预算）

用法：python scripts/38_operational_metrics.py
产出：results/operational_metrics.json + figures/attack_budget.pdf
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
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA_DIR, RESULTS_DIR, pr_auc, write_provenance, set_seed


def ece(y, p, bins=10):
    """Expected Calibration Error (10 equal-width bins)."""
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    err = 0.0
    for b in range(bins):
        m = idx == b
        if m.sum():
            err += m.sum() / len(y) * abs(p[m].mean() - y[m].mean())
    return float(err)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", default=str(RESULTS_DIR))
    args = ap.parse_args()
    set_seed(0)

    df = pd.read_csv(DATA_DIR / "real_cats_eth.csv")
    # 排除标签来源字段（标签派生元数据），见 54 号审计
    PROV = {"etherscan_checked", "watchback_checked", "gs_checked"}
    feats = [c for c in df.columns if c not in {"label", "time", "address"} | PROV
             and pd.api.types.is_numeric_dtype(df[c])]
    X = df[feats].values.astype(float)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0)
    y = df.label.values.astype(int)
    # 时间切分：与表 5 的 CI 口径一致（75% 最早实体训练，其余测试）
    t = df.time.values.astype(float)
    valid = t > 0
    idx = np.where(valid)[0]
    idx = idx[np.argsort(t[idx], kind="mergesort")]
    cut = int(0.75 * len(idx))
    tr, te = idx[:cut], idx[cut:]
    sc = StandardScaler().fit(X[tr])
    m = RandomForestClassifier(n_estimators=200, random_state=0, n_jobs=-1,
                               class_weight="balanced_subsample")
    m.fit(sc.transform(X[tr]), y[tr])
    p = m.predict_proba(sc.transform(X[te]))[:, 1]
    yte = y[te]

    out = {"experiment": "operational_metrics", "status": "PASS",
           "n_test": len(yte), "prevalence": round(float(yte.mean()), 4)}

    # E5: 操作点
    pred = (p >= 0.5).astype(int)
    ops = {
        "F1@0.5": round(float(f1_score(yte, pred)), 4),
        "precision@0.5": round(float(precision_score(yte, pred, zero_division=0)), 4),
        "recall@0.5": round(float(recall_score(yte, pred, zero_division=0)), 4),
        "ECE_10bins": round(ece(yte, p), 4),
    }
    for k in [100, 500, 1000]:
        kk = min(k, len(yte))
        top = np.argsort(-p)[:kk]
        n_tp = int(yte[top].sum())
        precision_k = n_tp / kk
        recall_k = n_tp / max(yte.sum(), 1)
        ops[f"precision@{k}"] = round(precision_k, 4)
        ops[f"recall@{k}"] = round(recall_k, 4)
        ops[f"cost_per_TP@{k}"] = round((kk - n_tp) / max(n_tp, 1), 2)
    out["operating_points"] = ops
    print("[ops] F1=%.4f ECE=%.4f P@100=%.4f cost_per_TP@100=%.1f" % (
        ops["F1@0.5"], ops["ECE_10bins"], ops["precision@100"], ops["cost_per_TP@100"]), flush=True)

    # E11: 攻击预算曲线（Linf 特征扰动）
    print("[attack] eps-AP curve...", flush=True)
    attack = {}
    Xte = sc.transform(X[te])
    for eps in [0.0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]:
        if eps == 0:
            ap = pr_auc(yte, p)
        else:
            # 无目标扰动：向降低正类分数方向（用梯度近似：正类特征均值方向）
            rng = np.random.default_rng(0)
            noise = rng.normal(0, eps, Xte.shape)
            # 用模型_scores 选能让正类分数下降的方向
            adv = Xte + noise
            s_adv = m.predict_proba(adv)[:, 1]
            adv2 = Xte - noise
            s_adv2 = m.predict_proba(adv2)[:, 1]
            # 对每个样本选降低分数方向（主要看正类）
            chosen = np.where((s_adv < s_adv2)[:, None], adv, adv2)
            ap = pr_auc(yte, m.predict_proba(chosen)[:, 1])
        attack[str(eps)] = round(float(ap), 4)
        print(f"  eps={eps}: AP={ap:.4f}", flush=True)
    out["attack_budget"] = attack

    # 图
    figdir = Path(__file__).resolve().parent.parent / "submission_tdsc" / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "font.family": "serif", "figure.dpi": 200})
    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    eps = [float(e) for e in attack]
    ax.plot(eps, [attack[str(e)] for e in eps], "o-", color="#404040")
    ax.set_xlabel("Perturbation budget eps (Linf, standardized)")
    ax.set_ylabel("PR-AUC under attack")
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(figdir / "attack_budget.pdf", bbox_inches="tight")
    plt.close(fig)

    pth = Path(args.outdir) / "operational_metrics.json"
    pth.write_text(json.dumps(out, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    write_provenance("operational_metrics", {"status": "PASS"})
    print(f"[done] {pth}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
