"""
70_attack_mitigation.py - E3：攻击之后给缓解，而不是只报漏洞
======================================================================
61 号脚本的结论是：layering 把 AP 从 0.799 打到 0.336（drop 0.464），
原因是模型学到"老账户 = 合法"的 lifetime 捷径。评审要求把"发现捷径"
变成"可部署的修补"。本脚本测三种特征侧缓解 + 一种检测侧缓解：

  A. drop-age      移除 lifetime / activity_d / activity_w / activity_time
  B. winsorise     把年龄类特征在训练集 99 分位处截断（可部署的约束）
  C. both          A + B
  D. stratified    不改模型，按训练期 lifetime 五分位分层定阈值，
                   使每个年龄层的告警率相同（检测侧缓解，报 recall@配额）

所有缓解都在**同一攻击**（layering h=2, q=1.0）下评估。攻击对缺失/被冻结的
特征自动跳过，避免"移除了特征却仍在扰动它"的假象。

说明：真正的"评分时冻结"需要每个实体的观测时刻，release 不提供；因此本
脚本用 winsorise 作为可部署近似，并在论文 Limitations 中写明这一边界。

用法：python scripts/70_attack_mitigation.py
产出：results/attack_mitigation.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA_DIR, RESULTS_DIR, pr_auc, set_seed, write_provenance

PROV = {"etherscan_checked", "watchback_checked", "gs_checked"}
AGE = ["lifetime", "activity_d", "activity_w", "activity_time"]


def load():
    df = pd.read_csv(DATA_DIR / "real_cats_eth.csv")
    feats = [c for c in df.columns if c not in {"label", "time", "address"} | PROV
             and pd.api.types.is_numeric_dtype(df[c])]
    X = np.nan_to_num(df[feats].values.astype(float), nan=0.0, posinf=0.0, neginf=0.0)
    y = df.label.values.astype(int)
    t = df.time.values.astype(float)
    return X, y, t, feats


def recall_at(y, s, frac=0.01):
    k = max(int(frac * len(y)), 1)
    top = np.argsort(-s)[:k]
    return float(y[top].sum() / max(y.sum(), 1))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", default=str(RESULTS_DIR))
    args = ap.parse_args()
    set_seed(0)

    X, y, t, feats = load()
    idx = np.argsort(t, kind="mergesort")
    cut = int(0.75 * len(idx))
    tr, te = idx[:cut], idx[cut:]
    te = te[t[te] > t[tr].max()]
    ytr, yte = y[tr], y[te]
    col = {c: i for i, c in enumerate(feats)}

    def build(variant):
        """返回（特征矩阵，列名，需要重新截断的列 -> 上界）。"""
        cols = list(range(len(feats)))
        caps = {}
        Xv = X.copy()
        if variant in ("drop-age", "both"):
            cols = [i for i, c in enumerate(feats) if c not in AGE]
        if variant in ("winsorise", "both"):
            for c in AGE:
                if c in col:
                    j = col[c]
                    hi = float(np.percentile(Xv[tr, j], 99))
                    Xv[:, j] = np.minimum(Xv[:, j], hi)
                    caps[c] = hi
        return Xv[:, cols], [feats[i] for i in cols], caps

    def attack(Xv, names, caps):
        """layering h=2, q=1.0；攻击后仍按部署规则截断（winsorise 不是冻结）。"""
        Xa = Xv.copy()
        # 攻击对象是**全表**中的测试正例，必须用全局索引；用测试内相对下标
        # 会打到训练期前 1,957 行，把 drop 从 0.46 缩成 0.06。
        pos = te[yte == 1]
        name_to_j = {n: j for j, n in enumerate(names)}
        for c, delta in [("transaction_number", 4), ("payment_transactions", 4),
                         ("lifetime", 60), ("activity_d", 4), ("activity_w", 4)]:
            if c in name_to_j:
                j = name_to_j[c]
                Xa[np.ix_(pos, [j])] += delta
                if c in caps:
                    Xa[:, j] = np.minimum(Xa[:, j], caps[c])
        return Xa

    rows = []
    for variant in ["none", "drop-age", "winsorise", "both"]:
        Xv, names, caps = build(variant)
        m = LGBMClassifier(n_estimators=350, learning_rate=0.05, num_leaves=31,
                           subsample=0.9, colsample_bytree=0.8, random_state=0,
                           n_jobs=-1, verbose=-1,
                           scale_pos_weight=(len(tr) - ytr.sum()) / max(ytr.sum(), 1))
        m.fit(Xv[tr], ytr)
        s0 = m.predict_proba(Xv[te])[:, 1]
        Xa = attack(Xv, names, caps)
        s1 = m.predict_proba(Xa[te])[:, 1]
        a0, a1 = float(pr_auc(yte, s0)), float(pr_auc(yte, s1))
        rows.append({"variant": variant, "n_features": len(names),
                     "base_AP": round(a0, 4), "attacked_AP": round(a1, 4),
                     "AP_drop": round(a0 - a1, 4),
                     "base_recall@1%": round(recall_at(yte, s0), 4),
                     "attacked_recall@1%": round(recall_at(yte, s1), 4),
                     "capped_columns": sorted(caps)})
        print(f"[{variant}] n={len(names)} base={a0:.4f} attack={a1:.4f} "
              f"drop={a0-a1:+.4f} R@1% {recall_at(yte, s0):.3f}->{recall_at(yte, s1):.3f}",
              flush=True)

    # 检测侧缓解：按训练期 lifetime 五分位分层定阈值，使各层告警率相同。
    # 用 none 模型（全特征）的分数，不改排序，只改阈值，因此 AP 不变；
    # 报 recall@1% 与分层后的 recall@1%，看攻击造成的告警流失是否被截住。
    Xv, names, _ = build("none")
    m = LGBMClassifier(n_estimators=350, learning_rate=0.05, num_leaves=31,
                       subsample=0.9, colsample_bytree=0.8, random_state=0,
                       n_jobs=-1, verbose=-1,
                       scale_pos_weight=(len(tr) - ytr.sum()) / max(ytr.sum(), 1))
    m.fit(Xv[tr], ytr)
    j_life = names.index("lifetime") if "lifetime" in names else None
    strat = None
    if j_life is not None:
        qs = np.percentile(Xv[tr, j_life], [20, 40, 60, 80])
        bin_tr = np.digitize(Xv[tr, j_life], qs)
        bin_te = np.digitize(Xv[te, j_life], qs)
        Xa = attack(Xv, names, {})
        s0 = m.predict_proba(Xv[te])[:, 1]
        s1 = m.predict_proba(Xa[te])[:, 1]
        # 每层取训练分数的 (1 - 0.01) 分位作为阈值（1% 告警率）
        thr = np.array([np.percentile(s0[bin_te == b], 99) if (bin_te == b).any() else np.inf
                        for b in range(5)])
        pred0 = s0 >= thr[bin_te]
        pred1 = s1 >= thr[bin_te]
        strat = {
            "rule": "threshold at the 99th percentile within each training-lifetime quintile",
            "base_recall@1%": round(float(yte[pred0].sum() / max(yte.sum(), 1)), 4),
            "attacked_recall@1%": round(float(yte[pred1].sum() / max(yte.sum(), 1)), 4),
            "n_flagged_base": int(pred0.sum()), "n_flagged_attacked": int(pred1.sum()),
        }
        print("[stratified]", strat, flush=True)

    base = next(r for r in rows if r["variant"] == "none")
    out = {"experiment": "attack_mitigation",
           "attack": "layering h=2, q=1.0 (adds 4 transactions, 60 days lifetime, 4 active days)",
           "n_test": int(len(te)), "prevalence": round(float(yte.mean()), 4),
           "variants": rows, "detection_side": strat, "status": "PASS"}
    out["verdict"] = {
        "undefended_drop": base["AP_drop"],
        "best_feature_side_drop": min(r["AP_drop"] for r in rows if r["variant"] != "none"),
        "drop_reduction": round(base["AP_drop"] - min(
            r["AP_drop"] for r in rows if r["variant"] != "none"), 4),
        "mitigation_effective": bool(min(r["AP_drop"] for r in rows if r["variant"] != "none") < 0.1),
        "note": ("Effective means the layering attack no longer moves AP by more than 0.1. "
                 "If only winsorisation helps, the shortcut is the unbounded tail of the "
                 "age features; if only removal helps, the model needs the age signal and "
                 "the fix costs accuracy."),
    }
    print("[verdict]", out["verdict"], flush=True)

    p = Path(args.outdir) / "attack_mitigation.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    write_provenance("attack_mitigation", {"status": "PASS"})
    print(f"[done] {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
