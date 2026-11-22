"""
31_rolling_origin.py - Rolling-origin 前向链式多折评估（回应审稿人 P1-2）
=======================================================================
审稿人要求：单一时间切分改为多折 rolling-origin，报告跨折 mean±std。
本脚本在 Real-CATS ETH 上做 5 折扩展窗口前向评估：
  fold i: train = 时间 <= t_i, test = (t_i, t_{i+1}]
  相邻折不重叠，符合真实部署的时间推进。

用法：python scripts/31_rolling_origin.py
产出：results/rolling_origin.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA_DIR, RESULTS_DIR, pr_auc, roc_auc, write_provenance, set_seed


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--folds", type=int, default=5)
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

    order = np.argsort(t, kind="mergesort")
    # 5 个等宽时间折，扩展窗口
    bounds = np.quantile(t[y == 0], np.linspace(0, 1, args.folds + 1))
    report = {
        "experiment": "rolling_origin",
        "dataset": "Real-CATS ETH", "n_folds": args.folds,
        "protocol": "expanding window; train = all entities with time <= fold start; "
                    "test = entities in (fold start, fold end]",
        "folds": [], "status": "PASS",
    }

    models = {
        "XGBoost": lambda: XGBClassifier(n_estimators=400, learning_rate=0.05,
                                           max_depth=5, subsample=0.9, colsample_bytree=0.9,
                                           eval_metric="logloss", random_state=0,
                                           n_jobs=-1, tree_method="hist"),
        "RandomForest": lambda: RandomForestClassifier(n_estimators=200, random_state=0, n_jobs=-1),
        "LogisticRegression": lambda: LogisticRegression(max_iter=1000),
    }

    for i in range(args.folds):
        lo, hi = bounds[i], bounds[i + 1]
        tr = np.where(t <= lo)[0]
        te = np.where((t > lo) & (t <= hi))[0]
        if len(tr) < 100 or len(te) < 20 or len(np.unique(y[te])) < 2:
            report["folds"].append({"fold": i, "status": "skip", "n_train": len(tr), "n_test": len(te)})
            continue
        sc = StandardScaler().fit(X[tr])
        Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
        fold_res = {"fold": i, "n_train": len(tr), "n_test": len(te),
                    "prevalence": round(float(y[te].mean()), 4), "models": {}}
        for name, mk in models.items():
            m = mk(); m.fit(Xtr, y[tr])
            score = m.predict_proba(Xte)[:, 1]
            fold_res["models"][name] = {
                "PR-AUC": round(float(pr_auc(y[te], score)), 4),
                "AUC": round(float(roc_auc(y[te], score)), 4),
            }
        report["folds"].append(fold_res)
        print(f"  fold {i}: train={len(tr)} test={len(te)} "
              f"XGB PR={fold_res['models']['XGBoost']['PR-AUC']:.4f}", flush=True)

    # 跨折汇总
    valid = [f for f in report["folds"] if f.get("status") != "skip"]
    summary = {}
    for name in models:
        vals = [f["models"][name]["PR-AUC"] for f in valid if name in f["models"]]
        if vals:
            summary[name] = {
                "PR_AUC_mean": round(float(np.mean(vals)), 4),
                "PR_AUC_std": round(float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0, 4),
                "n_folds": len(vals),
            }
    report["summary"] = summary
    print("\n[rolling-origin summary]")
    for name, v in summary.items():
        print(f"  {name:20s} {v['PR_AUC_mean']:.4f} ± {v['PR_AUC_std']:.4f} ({v['n_folds']} folds)")

    p = Path(args.outdir) / "rolling_origin.json"
    p.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    write_provenance("rolling_origin", {"status": "PASS", "n_folds": len(valid)})
    print(f"[done] {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
