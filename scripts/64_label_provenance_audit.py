"""
64_label_provenance_audit.py - E3 附加：标签来源字段审计（自查发现）
======================================================================
做时间窗审计时顺带扫了 Real-CATS ETH 的 49 个数值列，发现三列是**标签来源
元数据**而不是行为特征：

    etherscan_checked   |corr(label)| = 0.705
    gs_checked          |corr(label)| = 0.553
    watchback_checked   |corr(label)| = 0.012

前两列记录"该地址被哪个榜单确认"，与标签同源。把它们当特征等于把标签的
一部分直接喂给模型。这是标签重构探针要抓的东西（E3 层次 3），只是发生在
我们自己的管线上，而不是被审计的数据集里。

本脚本量化影响并固化处置：
  1. 三列的单特征 AUC；
  2. 时间切分下 49 维（含）vs 46 维（不含）的 AP / AUC，LightGBM 与 RF 各一；
  3. 结论：全项目移除这三列（07/10/38/42 已改），并把 49→46 的差异作为
     "标签派生特征贡献"的度量写进审计表。

用法：python scripts/64_label_provenance_audit.py
产出：results/label_provenance_audit.json
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
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA_DIR, RESULTS_DIR, pr_auc, set_seed, write_provenance

PROV = ["etherscan_checked", "watchback_checked", "gs_checked"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", default=str(RESULTS_DIR))
    args = ap.parse_args()
    set_seed(0)

    df = pd.read_csv(DATA_DIR / "real_cats_eth.csv")
    numeric = [c for c in df.columns if c not in {"label", "time", "address"}
               and pd.api.types.is_numeric_dtype(df[c])]
    clean = [c for c in numeric if c not in PROV]
    y = df.label.values.astype(int)
    t = df.time.values.astype(float)
    order = np.argsort(t, kind="mergesort")
    cut = int(0.75 * len(order))
    tr, te = order[:cut], order[cut:]
    # 边界 embargo：与 66 号脚本一致，同一时间戳的全部划入训练。
    # 不加这一步，测试集会包含与训练末端同刻的节点，LGBM 的 AP 会高 0.003（0.801 vs 0.798）。
    te = te[t[te] > t[tr].max()]

    singles = {}
    for c in PROV:
        v = df[c].values.astype(float)
        singles[c] = {"single_feature_AUC": round(float(roc_auc_score(y, v)), 4),
                      "abs_corr_with_label": round(float(abs(np.corrcoef(v, y)[0, 1])), 4)}

    # 全列单特征 AUC 前五（标签代理筛查）
    aucs = []
    for c in numeric:
        v = df[c].values.astype(float)
        if np.std(v) < 1e-12:
            continue
        a = roc_auc_score(y, v)
        aucs.append((c, max(a, 1 - a)))
    aucs.sort(key=lambda z: -z[1])

    def run(cols, model, seed=0, return_scores=False):
        X = np.nan_to_num(df[cols].values.astype(float), nan=0.0, posinf=0.0)
        if model == "lr":
            from sklearn.linear_model import LogisticRegression
            from sklearn.preprocessing import StandardScaler
            sc_ = StandardScaler().fit(X[tr])
            m = LogisticRegression(C=0.1, max_iter=3000, class_weight="balanced")
            m.fit(sc_.transform(X[tr]), y[tr])
            s_ = m.predict_proba(sc_.transform(X[te]))[:, 1]
            out_ = {"AP": round(float(pr_auc(y[te], s_)), 4),
                    "AUC": round(float(roc_auc_score(y[te], s_)), 4)}
            if return_scores:
                out_["_scores"] = s_
            return out_
        if model == "rf":
            m = RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1,
                                       class_weight="balanced_subsample")
        else:
            m = LGBMClassifier(n_estimators=350, learning_rate=0.05, num_leaves=31,
                               subsample=0.9, colsample_bytree=0.8, random_state=seed,
                               n_jobs=-1, verbose=-1,
                               scale_pos_weight=(len(tr) - y[tr].sum())
                               / max(y[tr].sum(), 1))
        m.fit(X[tr], y[tr])
        s = m.predict_proba(X[te])[:, 1]
        out = {"AP": round(float(pr_auc(y[te], s)), 4),
               "AUC": round(float(roc_auc_score(y[te], s)), 4)}
        if return_scores:
            out["_scores"] = s
        return out

    def run_seeds(cols, model, n=5):
        """五种子均值，与 66 号脚本同口径；单种子会在第三位小数量级上漂移 0.001。"""
        rs = [run(cols, model, sd) for sd in range(n)]
        aps = np.array([r["AP"] for r in rs])
        aucs_ = np.array([r["AUC"] for r in rs])
        return {"AP": round(float(aps.mean()), 4),
                "AP_sd": round(float(aps.std(ddof=1)), 4),
                "AUC": round(float(aucs_.mean()), 4),
                "n_seeds": n}

    out = {"experiment": "label_provenance_audit",
           "n_rows": int(len(df)), "n_features_all": len(numeric),
           "n_features_clean": len(clean),
           "test": {"n": int(len(te)), "prevalence": round(float(y[te].mean()), 4)},
           "label_provenance_columns": singles,
           "top_single_feature_AUC": [{"column": c, "AUC": round(float(a), 4)}
                                      for c, a in aucs[:6]],
           "models": {}, "status": "PASS"}

    for model in ["lr", "rf", "lgbm"]:
        allr = run_seeds(numeric, model)
        clr = run_seeds(clean, model)
        # bootstrap 95% CI：用 seed 0 的测试分数重采样，与点估计同一条流水线
        r0 = run(clean, model, 0, return_scores=True)
        sc = r0["_scores"]
        rng = np.random.default_rng(0)
        bs = []
        for _ in range(400):
            idx = rng.integers(0, len(te), len(te))
            if len(np.unique(y[te][idx])) < 2:
                continue
            bs.append(pr_auc(y[te][idx], sc[idx]))
        clr["CI95"] = [round(float(np.percentile(bs, 2.5)), 4),
                       round(float(np.percentile(bs, 97.5)), 4)]
        clr["CI_width"] = round(clr["CI95"][1] - clr["CI95"][0], 4)
        out["models"][model] = {
            "with_provenance": allr, "without_provenance": clr,
            "AP_gap": round(allr["AP"] - clr["AP"], 4),
            "AUC_gap": round(allr["AUC"] - clr["AUC"], 4)}
        print(f"[{model}] all={allr} clean={clr}", flush=True)

    out["verdict"] = {
        "action": "excluded from all pipelines (07, 10, 38, 42, 55..60)",
        "graph_features": {"before": 49, "after": 46},
        "note": ("Two of the three columns are label sources, not behaviour. Their removal "
                 "lowers tabular AP by %.3f (RF) and %.3f (LightGBM), which is the size of "
                 "the leak. Every Real-CATS number in the paper is recomputed after removal."
                 % (out["models"]["rf"]["AP_gap"], out["models"]["lgbm"]["AP_gap"])),
    }
    print("[verdict]", out["verdict"], flush=True)

    p = Path(args.outdir) / "label_provenance_audit.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    write_provenance("label_provenance_audit", {"status": "PASS"})
    print(f"[done] {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
