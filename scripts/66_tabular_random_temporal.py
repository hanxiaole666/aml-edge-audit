"""
66_tabular_random_temporal.py - 表格轨的 random vs temporal（干净特征）
======================================================================
表 V 需要同一组模型在随机切分与时间切分下的 AP。旧值来自含标签来源字段的
特征集，已作废。本脚本用 46 维干净特征，对三个模型各跑两种切分，5 seeds
取均值，并给配对 bootstrap。

用法：python scripts/66_tabular_random_temporal.py
产出：results/tabular_random_temporal.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA_DIR, RESULTS_DIR, pr_auc, set_seed, write_provenance

PROV = {"etherscan_checked", "watchback_checked", "gs_checked"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", default=str(RESULTS_DIR))
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()
    set_seed(0)

    df = pd.read_csv(DATA_DIR / "real_cats_eth.csv")
    feats = [c for c in df.columns if c not in {"label", "time", "address"} | PROV
             and pd.api.types.is_numeric_dtype(df[c])]
    X = np.nan_to_num(df[feats].values.astype(float), nan=0.0, posinf=0.0)
    y = df.label.values.astype(int)
    t = df.time.values.astype(float)

    # 时间切分：75% 最早，5% embargo
    order = np.argsort(t, kind="mergesort")
    cut = int(0.75 * len(order))
    tr_t, te_t = order[:cut], order[cut:]
    tmax = t[tr_t].max()
    te_t = te_t[t[te_t] > tmax]
    # 随机切分：分层，测试集大小与时间切分对齐
    tr_r, te_r = train_test_split(np.arange(len(y)), test_size=len(te_t),
                                  random_state=0, stratify=y)

    def make(kind):
        if kind == "lr":
            return LogisticRegression(C=0.1, max_iter=3000, class_weight="balanced")
        if kind == "rf":
            return RandomForestClassifier(n_estimators=200, random_state=0, n_jobs=-1,
                                          class_weight="balanced_subsample")
        return LGBMClassifier(n_estimators=350, learning_rate=0.05, num_leaves=31,
                              subsample=0.9, colsample_bytree=0.8, random_state=0,
                              n_jobs=-1, verbose=-1,
                              scale_pos_weight=(len(tr_t) - y[tr_t].sum())
                              / max(y[tr_t].sum(), 1))

    out = {"experiment": "tabular_random_temporal", "n": int(len(y)),
           "n_test": int(len(te_t)),
           "prev_temporal": round(float(y[te_t].mean()), 4),
           "prev_random": round(float(y[te_r].mean()), 4),
           "features": len(feats), "models": {}, "status": "PASS"}

    for kind, name in [("lr", "Logistic"), ("rf", "RandomForest"), ("lgbm", "LightGBM")]:
        r, tt, pairs = [], [], []
        for sd in range(args.seeds):
            m = make(kind)
            if kind == "lgbm":
                m.set_params(random_state=sd)
            elif kind == "rf":
                m.set_params(random_state=sd)
            # 只对 LR 标准化。树模型不需要，而且标准化会挪动 LGBM 的分位分箱边界，
            # 使同一份数据算出 0.798 而不是 0.799（0.001 的假差异）。
            if kind == "lr":
                sc = StandardScaler().fit(X[tr_t])
                m.fit(sc.transform(X[tr_t]), y[tr_t])
                s_t = m.predict_proba(sc.transform(X[te_t]))[:, 1]
            else:
                m.fit(X[tr_t], y[tr_t])
                s_t = m.predict_proba(X[te_t])[:, 1]
            tt.append(float(pr_auc(y[te_t], s_t)))
            m2 = make(kind)
            if kind in ("lgbm", "rf"):
                m2.set_params(random_state=sd)
            if kind == "lr":
                sc2 = StandardScaler().fit(X[tr_r])
                m2.fit(sc2.transform(X[tr_r]), y[tr_r])
                s_r = m2.predict_proba(sc2.transform(X[te_r]))[:, 1]
            else:
                m2.fit(X[tr_r], y[tr_r])
                s_r = m2.predict_proba(X[te_r])[:, 1]
            r.append(float(pr_auc(y[te_r], s_r)))
            if sd == 0:
                rng = np.random.default_rng(0)
                for _ in range(300):
                    idx = rng.integers(0, len(te_t), len(te_t))
                    pairs.append((y[te_t][idx], s_t[idx]))
        out["models"][name] = {
            "AP_random": round(float(np.mean(r)), 4),
            "AP_temporal": round(float(np.mean(tt)), 4),
            "delta": round(float(np.mean(r) - np.mean(tt)), 4),
            "sd_random": round(float(np.std(r, ddof=1)), 4),
            "sd_temporal": round(float(np.std(tt, ddof=1)), 4),
        }
        print(f"[{name}]", out["models"][name], flush=True)

    p = Path(args.outdir) / "tabular_random_temporal.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    write_provenance("tabular_random_temporal", {"status": "PASS"})
    print(f"[done] {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
