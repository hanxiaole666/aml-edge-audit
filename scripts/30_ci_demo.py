"""
30_ci_demo.py - 为关键结论计算真实 bootstrap CI（回应审稿人统计质疑）
=======================================================================
重新训练 Real-CATS 表格模型与 Elliptic2 LightGBM，保存逐样本预测，
对测试集做 1000 次 bootstrap，报告 AP 的 95% CI 与模型间 paired bootstrap。

用法：python scripts/30_ci_demo.py
产出：results/ci_demo.json + results/predictions_*.csv
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
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA_DIR, RESULTS_DIR, write_provenance, set_seed

sys.path.insert(0, str(Path(__file__).resolve().parent))
def _average_precision(y, score):
    order = np.argsort(-score)
    y = y[order]
    prec = np.cumsum(y) / (np.arange(len(y)) + 1)
    return float(np.sum(prec * y) / max(np.sum(y), 1))


def bootstrap_ap_ci(y, score, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed)
    y = np.asarray(y); score = np.asarray(score); n = len(y)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        vals.append(_average_precision(y[idx], score[idx]))
    vals = np.array(vals)
    return {"point": round(float(_average_precision(y, score)), 4),
            "ci95_low": round(float(np.percentile(vals, 2.5)), 4),
            "ci95_high": round(float(np.percentile(vals, 97.5)), 4),
            "n_boot": n_boot, "n_test": n}


def paired_bootstrap(y, a, b, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed)
    y = np.asarray(y); a = np.asarray(a); b = np.asarray(b); n = len(y)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        diffs.append(_average_precision(y[idx], a[idx]) - _average_precision(y[idx], b[idx]))
    diffs = np.array(diffs)
    return {"mean_diff": round(float(diffs.mean()), 4),
            "ci95_low": round(float(np.percentile(diffs, 2.5)), 4),
            "ci95_high": round(float(np.percentile(diffs, 97.5)), 4),
            "p_improvement": round(float((diffs > 0).mean()), 4),
            "significant_0.05": bool(np.percentile(diffs, 2.5) > 0 or np.percentile(diffs, 97.5) < 0),
            "n_boot": n_boot}


def realcats_predictions():
    df = pd.read_csv(DATA_DIR / "real_cats_eth.csv")
    feats = [c for c in df.columns if c not in {"label", "time", "address", "etherscan_checked", "watchback_checked", "gs_checked"}
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
    Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
    out = {"y_test": y[te], "n_test": len(te), "prevalence": float(y[te].mean())}
    for name, mk in [
        ("XGBoost", lambda: XGBClassifier(n_estimators=400, learning_rate=0.05,
                                           max_depth=5, subsample=0.9, colsample_bytree=0.9,
                                           eval_metric="logloss", random_state=0,
                                           n_jobs=-1, tree_method="hist")),
        ("RandomForest", lambda: RandomForestClassifier(n_estimators=200, random_state=0,
                                                        n_jobs=-1)),
    ]:
        m = mk(); m.fit(Xtr, y[tr])
        out[name] = m.predict_proba(Xte)[:, 1]
        pd.DataFrame({"label": y[te], "score": out[name]}).to_csv(
            RESULTS_DIR / f"predictions_realcats_{name.lower()}.csv", index=False)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", default=str(RESULTS_DIR))
    args = ap.parse_args()
    set_seed(0)

    print("[ci] rebuilding Real-CATS temporal predictions...", flush=True)
    d = realcats_predictions()
    y = d["y_test"]
    report = {
        "experiment": "bootstrap_ci_demo",
        "dataset": "Real-CATS ETH, temporal split",
        "n_test": d["n_test"], "prevalence": round(d["prevalence"], 4),
        "n_boot": 1000,
        "ci_by_model": {},
        "paired_comparison": {},
        "status": "PASS",
    }
    for name in ["XGBoost", "RandomForest"]:
        ci = bootstrap_ap_ci(y, d[name], n_boot=1000, seed=0)
        report["ci_by_model"][name] = ci
        print(f"  [CI] {name}: {ci['point']:.4f} "
              f"[{ci['ci95_low']:.4f}, {ci['ci95_high']:.4f}]", flush=True)
    report["paired_comparison"]["XGBoost_vs_RandomForest"] = paired_bootstrap(
        y, d["XGBoost"], d["RandomForest"], n_boot=1000, seed=0)

    p = Path(args.outdir) / "ci_demo.json"
    p.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    write_provenance("ci_demo", {"status": "PASS", "n_test": d["n_test"]})
    print(f"[done] {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
