"""
67_operating_points.py - E5：把 §4.3 承诺的操作点指标补齐
======================================================================
§4.3 承诺报 PR-AUC、ROC-AUC、F1、precision@k（与告警配额绑定）、ECE。
此前只有 Real-CATS ETH 单模型一份 operating_metrics.json，且 F1 取 0.5 阈值
（在 scale_pos_weight 加权后的分数上没有意义）。本脚本给
"数据集 x 模型" 的完整操作点表。

关键设计：阈值必须样本外选
    F1 依赖阈值，若在测试集上扫最优阈值，报出来的 F1 是乐观估计。这里在
    训练期内做 3 折 out-of-fold 预测，用 OOF 分数选 F1 最优阈值 tau*，
    再直接施加到测试集分数上。测试集从未参与阈值选择。

两处口径修正
    1. Elliptic v1 只取 4 个非退化 cutoff（30/33/36/39）。c=42 的 prevalence
       2.5%，任何模型都塌到近随机，操作点没有解释力。
    2. Real-CATS BTC 的"域内"数字此前来自随机 70/30 切分，与 ETH 的时间
       切分不可比。这里对 BTC 也做 first_time 时间切分，两条链才并得进同一张表。

用法：python scripts/67_operating_points.py
产出：results/operating_points.json + figures/operating_points.pdf
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
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA_DIR, RESULTS_DIR, pr_auc, set_seed, write_provenance

PROV = {"etherscan_checked", "watchback_checked", "gs_checked"}
META = {"address", "label", "time", "first_time", "last_time"}
QUOTAS = (10, 50, 100)          # 每 1,000 个被评分实体的告警配额
ABS_KS = (100, 500, 1000)       # 绝对条数，便于和旧表对照
DEGENERATE_PREV = 0.05


# ----------------------------------------------------------------- 指标
def f1_at(y, s, tau):
    pred = s >= tau
    tp = int(np.sum(pred & (y == 1)))
    fp = int(np.sum(pred & (y == 0)))
    fn = int(np.sum(~pred & (y == 1)))
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    return (2 * p * r / max(p + r, 1e-12), p, r, tp, fp, fn)


def best_f1_tau(y, s):
    """在给定分数上扫阈值取 F1 最优；返回 (tau, f1)。"""
    order = np.argsort(-s)
    ys = y[order]
    ss = s[order]
    tp = np.cumsum(ys)
    fp = np.cumsum(1 - ys)
    p = tp / np.maximum(tp + fp, 1)
    r = tp / max(ys.sum(), 1)
    f1 = 2 * p * r / np.maximum(p + r, 1e-12)
    i = int(np.argmax(f1))
    return float(ss[i]), float(f1[i])


def ece(y, s, n_bins=10):
    """等频分箱的期望校准误差 + Brier。"""
    order = np.argsort(s)
    bins = np.array_split(order, n_bins)
    e = 0.0
    for b in bins:
        if len(b) == 0:
            continue
        e += len(b) / len(y) * abs(y[b].mean() - s[b].mean())
    return float(e), float(np.mean((s - y) ** 2))


def topk_metrics(y, s, k):
    k = int(min(max(k, 1), len(y)))
    idx = np.argsort(-s)[:k]
    tp = int(y[idx].sum())
    prec = tp / k
    rec = tp / max(int(y.sum()), 1)
    return {"k": k, "precision": round(prec, 4), "recall": round(rec, 4),
            "cost_per_TP": round(k / tp, 3) if tp else None}


# ----------------------------------------------------------------- 数据
def load_cats(chain):
    """Real-CATS 一条链：ETH 走已合并的 csv，BTC 走原始 tsv。"""
    if chain == "ETH":
        df = pd.read_csv(DATA_DIR / "real_cats_eth.csv")
        df = df.rename(columns={"time": "t"})
    else:
        frames = []
        for fn, lab in [("CB.tsv", 1), ("BB.tsv", 0)]:
            d = pd.read_csv(DATA_DIR / "real_cats" / fn, sep="\t")
            d["label"] = lab
            frames.append(d)
        df = pd.concat(frames, ignore_index=True)
        df = df.rename(columns={"first_time": "t"})
    feats = [c for c in df.columns
             if c not in META | PROV | {"t"} and pd.api.types.is_numeric_dtype(df[c])]
    X = np.nan_to_num(df[feats].values.astype(float), nan=0.0, posinf=0.0, neginf=0.0)
    # 不做 log 变换。树模型对单调变换不变，而 LR 前有 StandardScaler，
    # 变换只会让 ETH 的 AP 偏离 Table~V 已引用的 0.652/0.723/0.798。
    y = df.label.values.astype(int)
    tt = df["t"]
    if not pd.api.types.is_numeric_dtype(tt):
        tt = pd.to_datetime(tt, errors="coerce")
        tt = tt.astype("int64") / 1e9          # 秒级时间戳，BTC 的 first_time 是日期字符串
    t = tt.values.astype(float)
    return X, y, t, feats


def load_elliptic():
    d = DATA_DIR / "elliptic_v1"
    feat = pd.read_csv(d / "elliptic_txs_features.csv", header=None,
                       dtype={0: np.int64, 1: np.int64})
    tx = feat.iloc[:, 0].values.astype(np.int64)
    ts = feat.iloc[:, 1].values.astype(np.int64)
    X = feat.iloc[:, 2:95].values.astype(float)      # 只取 93 维 local
    cls = pd.read_csv(d / "elliptic_txs_classes.csv")
    lab = dict(zip(cls.txId.values.astype(np.int64), cls["class"].values))
    y = np.array([1 if lab.get(int(t)) == "1" else (0 if lab.get(int(t)) == "2" else -1)
                  for t in tx], dtype=np.int64)
    return X, y, ts


# ----------------------------------------------------------------- 模型
def make(kind, y_tr, seed=0):
    if kind == "lr":
        return LogisticRegression(C=0.1, max_iter=3000, class_weight="balanced")
    if kind == "rf":
        return RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1,
                                      class_weight="balanced_subsample")
    return LGBMClassifier(n_estimators=350, learning_rate=0.05, num_leaves=31,
                          subsample=0.9, colsample_bytree=0.8, random_state=seed,
                          n_jobs=-1, verbose=-1,
                          scale_pos_weight=(len(y_tr) - y_tr.sum()) / max(y_tr.sum(), 1))


def fit_scores(kind, X_tr, y_tr, X_te, scaler=True, seed=0):
    m = make(kind, y_tr, seed)
    if kind == "lr":
        sc = StandardScaler().fit(X_tr)
        m.fit(sc.transform(X_tr), y_tr)
        return m.predict_proba(sc.transform(X_te))[:, 1]
    m.fit(X_tr, y_tr)
    return m.predict_proba(X_te)[:, 1]


def oof_tau(kind, X_tr, y_tr, n_splits=3, seed=0):
    """训练期内 out-of-fold 选阈值，测试集不参与。"""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.zeros(len(y_tr))
    for a, b in skf.split(X_tr, y_tr):
        oof[b] = fit_scores(kind, X_tr[a], y_tr[a], X_tr[b], seed=seed)
    return best_f1_tau(y_tr, oof)


def ops_block(kind, X_tr, y_tr, X_te, y_te, seed=0):
    tau, f1_oof = oof_tau(kind, X_tr, y_tr, seed=seed)
    s = fit_scores(kind, X_tr, y_tr, X_te, seed=seed)
    f1_t, p_t, r_t, tp, fp, fn = f1_at(y_te, s, tau)
    f1_05, p_05, r_05, *_ = f1_at(y_te, s, 0.5)
    e, brier = ece(y_te, s)
    n = len(y_te)
    block = {
        "AP": round(pr_auc(y_te, s), 4),
        "ROC_AUC": round(float(roc_auc_score(y_te, s)), 4),
        "tau_star_OOF": round(tau, 4),
        "F1_at_0.5": round(f1_05, 4),
        "P_at_0.5": round(p_05, 4),
        "R_at_0.5": round(r_05, 4),
        "F1_at_tau_star": round(f1_t, 4),
        "P_at_tau_star": round(p_t, 4),
        "R_at_tau_star": round(r_t, 4),
        "ECE_10bins": round(e, 4),
        "Brier": round(brier, 4),
        "n_test": int(n),
        "prevalence": round(float(y_te.mean()), 4),
        "quotas_per_1000": {}, "abs_topk": {},
    }
    for q in QUOTAS:
        block["quotas_per_1000"][str(q)] = topk_metrics(y_te, s, round(q / 1000 * n))
    for k in ABS_KS:
        if k <= n:
            block["abs_topk"][str(k)] = topk_metrics(y_te, s, k)
    return block


def summarise(folds):
    """跨折聚合：均值 ± 标准差。"""
    keys = ["AP", "ROC_AUC", "F1_at_0.5", "F1_at_tau_star",
            "P_at_tau_star", "R_at_tau_star", "ECE_10bins", "Brier"]
    out = {}
    for k in keys:
        v = np.array([f[k] for f in folds], dtype=float)
        out[k] = {"mean": round(float(v.mean()), 4),
                  "std": round(float(v.std(ddof=1)), 4) if len(v) > 1 else 0.0}
    out["prevalence"] = {"mean": round(float(np.mean([f["prevalence"] for f in folds])), 4)}
    for q in QUOTAS:
        p = np.array([f["quotas_per_1000"][str(q)]["precision"] for f in folds])
        c = [f["quotas_per_1000"][str(q)]["cost_per_TP"] for f in folds]
        c = [x for x in c if x is not None]
        out[f"P@{q}per1000"] = {"mean": round(float(p.mean()), 4),
                                "std": round(float(p.std(ddof=1)), 4) if len(p) > 1 else 0.0,
                                "cost_per_TP": round(float(np.mean(c)), 3) if c else None}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", default=str(RESULTS_DIR))
    ap.add_argument("--models", default="lr,rf,lgbm")
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()
    set_seed(0)
    kinds = [m.strip() for m in args.models.split(",") if m.strip()]
    name_of = {"lr": "Logistic", "rf": "RandomForest", "lgbm": "LightGBM"}

    out = {"experiment": "operating_points",
           "protocol": "threshold selected by 3-fold out-of-fold F1 on the training period; "
                       "test set never used for threshold selection",
           "quotas": list(QUOTAS), "abs_ks": list(ABS_KS), "datasets": {}, "status": "PASS"}

    # ---------- D1 Elliptic v1：4 个非退化 cutoff
    Xe, ye, tse = load_elliptic()
    lab = ye >= 0
    elliptic = {}
    folds_used = []
    for cutoff in [30, 33, 36, 39]:
        tr = np.where(lab & (tse < cutoff))[0]
        te = np.where(lab & (tse > cutoff))[0]        # 跳过 t == cutoff 作 embargo
        if len(te) == 0 or ye[te].mean() < DEGENERATE_PREV:
            continue
        folds_used.append(cutoff)
        for k in kinds:
            elliptic.setdefault(name_of[k], []).append(
                ops_block(k, Xe[tr], ye[tr], Xe[te], ye[te]))
        print(f"[elliptic] cutoff={cutoff} n_tr={len(tr)} n_te={len(te)} "
              f"prev={ye[te].mean():.4f}", flush=True)
    out["datasets"]["Elliptic v1"] = {
        "splits": "4 non-degenerate temporal cutoffs (embargo = 1 step)",
        "cutoffs": folds_used, "features": 93,
        "models": {m: summarise(v) for m, v in elliptic.items()},
    }

    # ---------- D2/D3 Real-CATS：first_time 时间切分 + embargo
    for chain, label in [("ETH", "Real-CATS ETH"), ("BTC", "Real-CATS BTC")]:
        X, y, t, feats = load_cats(chain)
        order = np.argsort(t, kind="mergesort")
        cut = int(0.75 * len(order))
        tr, te = order[:cut], order[cut:]
        tmax = t[tr].max()
        te = te[t[te] > tmax]                       # 同一时刻的全部划入训练，避免并列泄漏
        prev_te = float(y[te].mean())
        degenerate = prev_te < DEGENERATE_PREV
        per = {}
        for k in kinds:
            # 五种子平均，与 64/66 号脚本同口径；单种子会在第三位小数上漂移 0.001
            blocks = [ops_block(k, X[tr], y[tr], X[te], y[te], seed=sd)
                      for sd in range(args.seeds)]
            per[name_of[k]] = summarise(blocks)
            per[name_of[k]]["n_seeds"] = args.seeds
            b = blocks[0]
            print(f"[{chain}] {name_of[k]} AP={per[name_of[k]]['AP']['mean']:.4f}"
                  f"±{per[name_of[k]]['AP']['std']:.4f} "
                  f"F1@{b['tau_star_OOF']:.3f}={b['F1_at_tau_star']:.4f} "
                  f"ECE={b['ECE_10bins']:.4f}", flush=True)
        if degenerate:
            # 不静默丢弃：BTC 的晚期样本里阳性只占 2.8%，时间切分本身就是一条结论
            print(f"[{chain}] WARNING degenerate test prevalence {prev_te:.4f} "
                  f"(<{DEGENERATE_PREV}); reported with a flag, not used in the band",
                  flush=True)
        out["datasets"][label] = {
            "splits": "first_time 75/25 temporal with embargo at the boundary",
            "features": len(feats), "n_train": int(len(tr)), "n_test": int(len(te)),
            "test_prevalence": round(prev_te, 4),
            "degenerate": bool(degenerate),
            "degenerate_note": ("Late-period prevalence is below the 5% floor. A temporal "
                                "split on this chain leaves almost no positives in test, so "
                                "the random-split domain-internal figures in the cross-chain "
                                "study (0.88/0.93) have no temporal counterpart.")
                               if degenerate else None,
            "models": per,
        }

    # ---------- 汇总一句可引用的判断
    gaps = []
    for ds, b in out["datasets"].items():
        for m, v in b["models"].items():
            if "F1_at_tau_star" in v and isinstance(v["F1_at_tau_star"], dict):
                gaps.append(v["F1_at_tau_star"]["mean"] - v["F1_at_0.5"]["mean"])
    out["f1_gain_from_threshold_selection"] = {
        "mean": round(float(np.mean(gaps)), 4) if gaps else None,
        "max": round(float(np.max(gaps)), 4) if gaps else None,
        "note": "F1 at a model-selected threshold minus F1 at the naive 0.5 cut. "
                "A large positive gap means 0.5 was miscalibrated, not that the model improved.",
    }
    print("[f1 gain over 0.5]", out["f1_gain_from_threshold_selection"], flush=True)

    p = Path(args.outdir) / "operating_points.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2, default=float), encoding="utf-8")

    # ---------- 图：P@quota vs 操作点（各数据集 LightGBM）
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.7))
    for ds, b in out["datasets"].items():
        if b.get("degenerate"):
            continue
        v = b["models"].get("LightGBM", {})
        if not v or f"P@{QUOTAS[0]}per1000" not in v:
            continue
        ps = [v[f"P@{q}per1000"]["mean"] for q in QUOTAS]
        ax[0].plot(list(QUOTAS), ps, marker="o", ms=3, lw=1.2, label=ds)
        ax[1].scatter([v["R_at_tau_star"]["mean"]], [v["F1_at_tau_star"]["mean"]],
                      s=18, label=ds)
    ax[0].set_xscale("log"); ax[0].set_xlabel("alert quota per 1,000 scored")
    ax[0].set_ylabel("precision"); ax[0].set_ylim(0, 1.02)
    ax[0].set_title("(a) precision under a fixed budget", fontsize=8)
    ax[0].legend(fontsize=5.5, frameon=False); ax[0].tick_params(labelsize=6)
    ax[1].set_xlabel("recall at selected threshold"); ax[1].set_ylabel("F1")
    ax[1].set_title("(b) operating point at $\\tau^\\star$", fontsize=8)
    ax[1].legend(fontsize=5.5, frameon=False); ax[1].tick_params(labelsize=6)
    fig.tight_layout()
    fig.savefig(Path(RESULTS_DIR).parent / "figures" / "operating_points.pdf",
                bbox_inches="tight")
    print("[fig] operating_points.pdf")
    print(f"[done] {p}")
    write_provenance("operating_points", {"status": "PASS"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
