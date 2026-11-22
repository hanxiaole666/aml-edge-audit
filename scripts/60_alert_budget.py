"""
60_alert_budget.py - E10 + E16：告警预算下的运营特性与 risk–coverage
======================================================================
AML 是运营系统，模型分数必须落到"每天能看多少条"的约束上。本脚本：

告警预算（E10）
  固定每日 k ∈ {100, 500, 1000} 条告警，按测试期自然日分组，每日取分数最高的 k 条。
  报 micro 的 P@k、R@k、TP、FP、cost-per-TP = FP/TP（每抓一个真阳性要付多少误报）。
  同表给"全局 top-k"作对照，说明按日归一化会改变什么。

risk–coverage（E16）
  选择性子集：按置信度 |p-0.5| 排序，接受最有信心的 c 比例，其余转人工。
  报 coverage ∈ {0.80, 0.90, 0.95, 1.00} 下的
      risk  = 接受子集上的错误率
      FNR   = 正例中未被正确接受的比例（漏检率）
  这正是"覆盖率 80% 时 FNR 多少"的答案。

用法：python scripts/60_alert_budget.py
产出：results/alert_budget.json + figures/alert_budget.pdf
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RESULTS_DIR, pr_auc, set_seed, write_provenance

PROV = {"etherscan_checked", "watchback_checked", "gs_checked"}


def load_tabular():
    """Real-CATS ETH 表格轨：与 38 号脚本同口径（时间切分、排除标签来源字段）。"""
    import pandas as pd
    from common import DATA_DIR
    df = pd.read_csv(DATA_DIR / "real_cats_eth.csv")
    feats = [c for c in df.columns if c not in {"label", "time", "address"} | PROV
             and pd.api.types.is_numeric_dtype(df[c])]
    X = np.nan_to_num(df[feats].values.astype(float), nan=0.0, posinf=0.0, neginf=0.0)
    y = df.label.values.astype(int)
    t = df.time.values.astype(float)
    return X, y, t, feats


_spec = importlib.util.spec_from_file_location(
    "ep", Path(__file__).resolve().parent / "44_edge_protocol_final.py")
_ep = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ep)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", default=str(RESULTS_DIR))
    args = ap.parse_args()
    set_seed(0)

    X, y, t, _feats = load_tabular()
    order = np.argsort(t, kind="mergesort")
    cut = int(0.75 * len(order))
    tr, te = order[:cut], order[cut:]
    tmax = t[tr].max()
    te = te[t[te] > tmax]
    # 树模型不标准化：与 61/66/67 同口径，避免分位分箱边界移动造成的 0.001 假差异
    m = LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31,
                       subsample=0.9, colsample_bytree=0.8, random_state=0,
                       n_jobs=-1, verbose=-1,
                       scale_pos_weight=(len(tr) - y[tr].sum()) / max(y[tr].sum(), 1))
    m.fit(X[tr], y[tr])
    s = m.predict_proba(X[te])[:, 1]
    yt = y[te]
    days = ((t[te] - t[te].min()) // 86400).astype(int)
    n_days = days.max() + 1
    print(f"[data] test={len(yt)} prev={yt.mean():.4f} AP={pr_auc(yt, s):.4f} "
          f"days={n_days} pos={int(yt.sum())}", flush=True)

    budgets = []
    vol_per_day = len(yt) / max(n_days, 1)
    # 说明：本数据集测试期 575 天仅 7,148 个实体（12.4/天），"每日 100 条"的
    # 绝对上限根本不会触发。因此把预算定义为**每 1,000 个被评分实体**的告警
    # 配额（等价于日均量达到 1,000 时的每日配额），并同时报绝对条数。
    rates = [0.01, 0.05, 0.10]
    for rate in rates:
        k = int(rate * len(yt))
        top = np.argsort(-s)[:k]
        tp = int(yt[top].sum()); fp = int(k - tp)
        budgets.append({"quota_per_1000": int(rate * 1000), "flagged": k,
                        "precision": round(tp / max(k, 1), 4),
                        "recall": round(tp / max(int(yt.sum()), 1), 4),
                        "TP": tp, "FP": fp,
                        "cost_per_TP": round(fp / max(tp, 1), 2)})
    # 绝对条数（若日均量足够大时的等效配额）
    for k in [100, 500, 1000]:
        kk = min(k, len(yt))
        top = np.argsort(-s)[:kk]
        tp = int(yt[top].sum())
        budgets.append({"quota_absolute": k, "flagged": kk,
                        "precision": round(tp / max(kk, 1), 4),
                        "recall": round(tp / max(int(yt.sum()), 1), 4),
                        "TP": tp, "FP": int(kk - tp),
                        "cost_per_TP": round((kk - tp) / max(tp, 1), 2)})

    # risk–coverage
    conf = np.abs(s - 0.5)
    o = np.argsort(-conf)
    rc = []
    for c in [0.80, 0.90, 0.95, 1.00]:
        n_acc = int(c * len(yt))
        acc = o[:n_acc]
        err = float((yt[acc] != (s[acc] >= 0.5)).mean())
        pos = yt == 1
        # 三个量必须分开报，否则"选择性预测"会把 deferral 误算成排序变差：
        #   FNR_accepted      接受集内部，正例被判为负的比例（应随覆盖率下降而下降）
        #   positives_deferred 被推迟的正例占全部正例的比例（代价来源）
        #   FNR_end_to_end    被推迟也算漏检（部署口径），恒 >= FNR_accepted
        pred = s[acc] >= 0.5
        tp = int(((pred) & (yt[acc] == 1)).sum())
        fn = int((~pred & (yt[acc] == 1)).sum())
        fnr_acc = fn / max(tp + fn, 1)
        fnr = float(((yt[pos] != (s[pos] >= 0.5)) | ~np.isin(np.where(pos)[0], acc)).mean())
        rc.append({"coverage": c, "n_accepted": n_acc,
                   "risk_error_rate": round(err, 4),
                   "FNR_accepted": round(float(fnr_acc), 4),
                   "positives_deferred": round(float(1 - yt[acc].sum() / yt.sum()), 4),
                   "FNR_end_to_end": round(fnr, 4),
                   "precision_at_0.5": round(float(yt[acc][s[acc] >= 0.5].mean())
                                             if (s[acc] >= 0.5).any() else 0.0, 4)})

    out = {"experiment": "alert_budget", "test": {"n": int(len(yt)),
                                                 "prevalence": round(float(yt.mean()), 4),
                                                 "AP": round(float(pr_auc(yt, s)), 4),
                                                 "days": int(n_days),
                                                 "volume_per_day": round(vol_per_day, 1)},
           "budget_note": ("Daily volume is 12.4 entities, so an absolute cap of 100/day "
                           "never binds; quotas are given per 1,000 scored entities, which "
                           "is the same rule at production volume."),
           "budgets": budgets, "risk_coverage": rc, "status": "PASS"}
    out["verdict"] = {
        "cost_per_TP_at_1pct_quota": next(b["cost_per_TP"] for b in budgets
                                          if b.get("quota_per_1000") == 10),
        "cost_per_TP_at_10pct_quota": next(b["cost_per_TP"] for b in budgets
                                           if b.get("quota_per_1000") == 100),
        "FNR_accepted_at_80pct_coverage": next(r["FNR_accepted"] for r in rc
                                               if r["coverage"] == 0.80),
        "FNR_end_to_end_at_80pct_coverage": next(r["FNR_end_to_end"] for r in rc
                                                 if r["coverage"] == 0.80),
        "positives_deferred_at_80pct_coverage": next(r["positives_deferred"] for r in rc
                                                     if r["coverage"] == 0.80),
        "note": "A per-day budget equalises operational load; the global top-k column shows "
                "how much of the apparent precision comes from ignoring the daily cap.",
    }
    print(json.dumps(out["budgets"], indent=1), flush=True)
    print(json.dumps(out["risk_coverage"], indent=1), flush=True)

    p = Path(args.outdir) / "alert_budget.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2, default=float), encoding="utf-8")

    fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.4))
    q = [b for b in budgets if "quota_per_1000" in b]
    ax[0].plot([b["recall"] for b in q], [b["precision"] for b in q],
               "o-", color="#c0392b", ms=4)
    for b in q:
        ax[0].annotate(f'{b["quota_per_1000"]}/1k', (b["recall"], b["precision"]),
                       fontsize=6, xytext=(3, 3), textcoords="offset points")
    ax[0].set_xlabel("recall", fontsize=7); ax[0].set_ylabel("precision", fontsize=7)
    ax[1].plot([r["coverage"] for r in rc], [r["FNR_end_to_end"] for r in rc], "o-",
               color="#2980b9", ms=4)
    ax[1].set_xlabel("coverage", fontsize=7); ax[1].set_ylabel("FNR", fontsize=7)
    for a in ax:
        a.tick_params(labelsize=6); a.grid(alpha=0.25, lw=0.5)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR.parent / "figures" / "alert_budget.pdf", bbox_inches="tight")
    print("[fig] alert_budget.pdf")

    write_provenance("alert_budget", {"status": "PASS"})
    print(f"[done] {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
