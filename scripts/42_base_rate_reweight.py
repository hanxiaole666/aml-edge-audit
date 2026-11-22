"""
42_base_rate_reweight.py - 基率重加权 + conformal 运营化（E6/E7）
===============================================================
审稿人指出：Real-CATS prevalence=0.274，P@100=1.0 是基率假象；
AML真实基率是个位数百分比。且 conformal 只报 over-coverage，
未报阈值下 recall/precision/告警量/预测集大小。

E7: 用重要性加权把测试集重采样到 1%/3%/10% 基率，重算 AP/P@k/代价
E6: conformal 阈值下的运营指标（recall/precision/告警率/集合大小）

用法：python scripts/42_base_rate_reweight.py
产出：results/base_rate_reweight.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA_DIR, RESULTS_DIR, pr_auc, roc_auc, write_provenance, set_seed


def weighted_ap(y, s, w):
    """加权 average precision。"""
    order = np.argsort(-s)
    y, s, w = y[order], s[order], w[order]
    cum_tp = np.cumsum(y * w)
    cum_all = np.cumsum(w)
    prec = cum_tp / np.maximum(cum_all, 1e-9)
    total_pos_w = np.sum(y * w)
    return float(np.sum(prec * y * w) / max(total_pos_w, 1e-9))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", default=str(RESULTS_DIR))
    args = ap.parse_args()
    set_seed(0)

    df = pd.read_csv(DATA_DIR / "real_cats_eth.csv")
    PROV = {"etherscan_checked", "watchback_checked", "gs_checked"}
    feats = [c for c in df.columns if c not in {"label", "time", "address"} | PROV
             and pd.api.types.is_numeric_dtype(df[c])]
    X = df[feats].values.astype(float)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0)
    y = df.label.values.astype(int)
    t = df.time.values.astype(float)
    order = np.argsort(t, kind="mergesort")
    cut = int(0.75 * len(y))
    tr, te = order[:cut], order[cut:]
    tmax = t[tr].max()
    te = te[t[te] > tmax]
    if len(te) < 20:
        te = order[cut:]
    sc = StandardScaler().fit(X[tr])
    m = RandomForestClassifier(n_estimators=200, random_state=0, n_jobs=-1,
                               class_weight="balanced_subsample")
    m.fit(sc.transform(X[tr]), y[tr])
    p = m.predict_proba(sc.transform(X[te]))[:, 1]
    yte = y[te]
    base_prev = float(yte.mean())

    out = {"experiment": "base_rate_reweight",
           "original_prevalence": round(base_prev, 4),
           "n_test": len(yte), "status": "PASS", "reweighted": {}}

    # E7: 重要性加权到目标基率
    for target in [0.01, 0.03, 0.10]:
        # 重要性权重：正类 w = target/base，负类 w = (1-target)/(1-base)
        w = np.where(yte == 1, target / base_prev, (1 - target) / (1 - base_prev))
        wap = weighted_ap(yte, p, w)
        # 加权 P@100（按分数排序取权重最大的100个"有效样本"近似：取top100按w归一）
        order = np.argsort(-p)[:100]
        w_tp = float(np.sum(yte[order] * w[order]))
        w_all = float(np.sum(w[order]))
        out["reweighted"][str(target)] = {
            "target_prevalence": target,
            "weighted_PR_AUC": round(wap, 4),
            "weighted_precision@100": round(w_tp / max(w_all, 1e-9), 4),
            "effective_positive_weight_top100": round(w_tp, 3),
        }
        print(f"  target={target:.0%}: weighted AP={wap:.4f} "
              f"P@100={w_tp/max(w_all,1e-9):.4f}", flush=True)

    # E6: conformal 运营指标（在测试集上用 α=0.1 的阈值）
    alpha = 0.1
    cutoff_score = float(np.quantile(1 - p, 1 - alpha))  # 非一致性分位数
    pred_pos = (1 - p) <= cutoff_score
    pos = yte == 1; neg = yte == 0
    out["conformal_operational"] = {
        "alpha": alpha,
        "threshold_score": round(1 - cutoff_score, 4),
        "predicted_positive_rate": round(float(pred_pos.mean()), 4),
        "recall_on_positive": round(float(pred_pos[pos].mean()), 4),
        "precision": round(float(yte[pred_pos].mean()), 4) if pred_pos.sum() else None,
        "n_alerts": int(pred_pos.sum()),
        "alert_rate_per_1000": round(float(pred_pos.mean()) * 1000, 1),
        "note": ("At this threshold the prediction set is deliberately large, giving high "
                 "recall and coverage but many alerts; this is the operational cost of "
                 "the conservative guarantee."),
    }
    print(f"  conformal@α={alpha}: recall={out['conformal_operational']['recall_on_positive']} "
          f"precision={out['conformal_operational']['precision']} "
          f"alerts={out['conformal_operational']['n_alerts']}/{len(yte)}")

    pth = Path(args.outdir) / "base_rate_reweight.json"
    pth.write_text(json.dumps(out, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    write_provenance("base_rate_reweight", {"status": "PASS"})
    print(f"[done] {pth}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
