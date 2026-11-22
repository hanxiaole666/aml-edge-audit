"""
47_aci_conformal.py - E6 conformal 的漂移修正：从 split conformal 到 ACI
======================================================================
split conformal 的覆盖保证要求 exchangeability。AML 数据是时间流，
该假设不成立，固定阈值在漂移下会失效。Gibbs & Candes (2021) 的
Adaptive Conformal Inference (ACI) 在每个时间步更新误覆盖率参数：

    alpha_{t+1} = alpha_t + gamma * (alpha - err_t)

其中 err_t = 1 表示第 t 个测试实体是正类但未被告警（漏报）。该更新
对任意（可自适应选择的）数据序列给出长期平均误覆盖率 <= alpha 的
保证，不需要 exchangeability。代价是阈值随流波动，覆盖率在短窗口内
仍会偏离。

本脚本在同一时间折上并列报告：
  1. split conformal（固定阈值，exchangeability 假设）
  2. ACI（在线更新，无 exchangeability 假设）
  3. oracle 固定阈值（用测试集自身分位数，事后上界参照）
并报覆盖率、末段 20% 窗口覆盖率、告警率、precision、以及 gamma 敏感性。

用法：python scripts/47_aci_conformal.py
产出：results/aci_conformal.json + figures/aci_coverage.pdf
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
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA_DIR, RESULTS_DIR, write_provenance, set_seed


def sorted_quantile(sorted_scores: np.ndarray, q: float) -> float:
    """线性插值分位数，输入已排序，O(1)。"""
    q = min(max(q, 0.0), 1.0)
    n = len(sorted_scores)
    if n == 1:
        return float(sorted_scores[0])
    pos = q * (n - 1)
    lo = int(np.floor(pos))
    hi = int(np.ceil(pos))
    if lo == hi:
        return float(sorted_scores[lo])
    w = pos - lo
    return float(sorted_scores[lo] * (1 - w) + sorted_scores[hi] * w)


def run_stream(scores: np.ndarray, labels: np.ndarray, cal_sorted: np.ndarray,
               alpha: float, mode: str, gamma: float = 0.001,
               window: int = 1000) -> dict:
    """按时间顺序跑一个测试流，返回覆盖/告警指标。

    mode: fixed  = split conformal，阈值固定
          aci    = adaptive conformal inference，alpha_t 在线更新
          swc    = sliding-window conformal，校准集为最近 window 个已观测样本
          oracle = 事后用测试集自身分位数（上界参照）
    """
    n = len(scores)
    hist = list(cal_sorted[-window:]) if len(cal_sorted) > window else list(cal_sorted)
    flags = np.zeros(n, dtype=bool)
    q_trace = np.zeros(n)
    at = alpha
    q_fixed = sorted_quantile(cal_sorted, 1 - alpha)
    for i in range(n):
        if mode == "fixed":
            q = q_fixed
        elif mode == "aci":
            q = sorted_quantile(cal_sorted, 1 - at)
        elif mode == "swc":
            q = float(np.quantile(np.asarray(hist), 1 - alpha))
        elif mode == "oracle":
            q = None  # 事后统一处理
        else:
            raise ValueError(mode)
        if mode != "oracle":
            flags[i] = scores[i] <= q
            q_trace[i] = q
            err = 1.0 if (labels[i] == 1 and not flags[i]) else 0.0
            if mode == "aci":
                at = float(np.clip(at + gamma * (alpha - err), 1e-3, 0.5))
            if mode == "swc":
                hist.append(float(scores[i]))
                if len(hist) > window:
                    hist.pop(0)
    if mode == "oracle":
        q = sorted_quantile(np.sort(scores), 1 - alpha)
        flags = scores <= q
        q_trace[:] = q

    pos = labels == 1
    neg = labels == 0
    coverage = float(flags[pos].mean()) if pos.sum() else float("nan")
    alert_rate = float(flags.mean())
    precision = float(labels[flags].mean()) if flags.sum() else float("nan")
    tail = slice(int(0.8 * n), n)
    tail_cov = float(flags[tail][pos[tail]].mean()) if pos[tail].sum() else float("nan")
    head_cov = float(flags[: int(0.2 * n)][pos[: int(0.2 * n)]].mean()) if pos[: int(0.2 * n)].sum() else float("nan")
    # 滚动窗口覆盖率（窗口 = 10% 流长）
    w = max(int(0.1 * n), 20)
    roll = []
    for s in range(0, n - w + 1, max(w // 4, 1)):
        seg_pos = pos[s:s + w]
        if seg_pos.sum() >= 3:
            roll.append(round(float(flags[s:s + w][seg_pos].mean()), 3))
    return {
        "coverage": round(coverage, 4) if np.isfinite(coverage) else None,
        "tail20_coverage": round(tail_cov, 4) if np.isfinite(tail_cov) else None,
        "head20_coverage": round(head_cov, 4) if np.isfinite(head_cov) else None,
        "alert_rate": round(alert_rate, 4),
        "precision": round(precision, 4) if np.isfinite(precision) else None,
        "rolling_coverage": roll,
        "q_final": round(float(q_trace[-1]), 4) if len(q_trace) else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", default=str(RESULTS_DIR))
    args = ap.parse_args()
    set_seed(0)

    df = pd.read_csv(DATA_DIR / "real_cats_eth.csv")
    PROV = {"etherscan_checked", "watchback_checked", "gs_checked"}
    feats = [c for c in df.columns if c not in {"label", "time", "address"} | PROV
             and pd.api.types.is_numeric_dtype(df[c])]
    X = np.nan_to_num(df[feats].values.astype(float), nan=0.0, posinf=0.0)
    y = df.label.values.astype(int)
    t = df.time.values.astype(float)
    valid = t > 0

    out = {
        "experiment": "aci_conformal",
        "method": ("split conformal (fixed threshold) vs adaptive conformal inference "
                   "(Gibbs & Candes 2021). ACI updates alpha_t = alpha_t-1 + gamma*(alpha - err_t) "
                   "and guarantees long-run average miscoverage <= alpha without exchangeability."),
        "alphas": [0.05, 0.10, 0.20],
        "gammas": [0.001, 0.005, 0.01, 0.02],
        "folds": [], "status": "PASS",
    }

    cache = []
    q = np.quantile(t[valid], np.linspace(0, 1, 7))
    for i in range(4):
        lo_cal, hi_cal = q[i + 1], q[i + 2]
        lo_te, hi_te = q[i + 2], q[i + 3]
        tr = np.where(valid & (t <= lo_cal))[0]
        cal = np.where(valid & (t > lo_cal) & (t <= hi_cal))[0]
        te = np.where(valid & (t > lo_te) & (t <= hi_te))[0]
        if len(tr) < 100 or len(cal) < 50 or len(te) < 50:
            continue
        te = te[np.argsort(t[te], kind="mergesort")]  # 流按时间排序
        sc = StandardScaler().fit(X[tr])
        m = RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1,
                                   class_weight="balanced_subsample")
        m.fit(sc.transform(X[tr]), y[tr])
        s_cal = 1 - m.predict_proba(sc.transform(X[cal]))[:, 1]
        s_te = 1 - m.predict_proba(sc.transform(X[te]))[:, 1]
        cal_sorted = np.sort(s_cal)
        fold = {"fold": i, "n_train": len(tr), "n_calib": len(cal), "n_test": len(te),
                "test_prevalence": round(float(y[te].mean()), 4),
                "stream_order": "ascending first-seen time",
                "alphas": {}}
        for alpha in out["alphas"]:
            res = {
                "split_conformal": run_stream(s_te, y[te], cal_sorted, alpha, "fixed"),
                "aci": run_stream(s_te, y[te], cal_sorted, alpha, "aci", gamma=0.001),
                "swc": run_stream(s_te, y[te], cal_sorted, alpha, "swc", window=1000),
                "oracle_fixed": run_stream(s_te, y[te], cal_sorted, alpha, "oracle"),
            }
            res["target_coverage"] = round(1 - alpha, 4)
            res["aci_minus_fixed_coverage"] = round(
                (res["aci"]["coverage"] or 0) - (res["split_conformal"]["coverage"] or 0), 4)
            fold["alphas"][str(alpha)] = res
        out["folds"].append(fold)
        cache.append((s_te, y[te].copy(), cal_sorted))
        print(f"  fold {i}: n={len(te)} "
              f"fixed@0.1={fold['alphas']['0.1']['split_conformal']['coverage']} "
              f"aci@0.1={fold['alphas']['0.1']['aci']['coverage']}", flush=True)

    # ---- gamma 敏感性（alpha=0.1，全部折平均）
    gamma_sens = {}
    for g in out["gammas"]:
        covs, alerts, tails = [], [], []
        for s_te, y_te, cal_sorted in cache:
            r = run_stream(s_te, y_te, cal_sorted, 0.10, "aci", gamma=g)
            if r["coverage"] is not None:
                covs.append(r["coverage"])
            alerts.append(r["alert_rate"])
            if r["tail20_coverage"] is not None:
                tails.append(r["tail20_coverage"])
        gamma_sens[str(g)] = {
            "mean_coverage": round(float(np.mean(covs)), 4) if covs else None,
            "mean_tail20_coverage": round(float(np.mean(tails)), 4) if tails else None,
            "mean_alert_rate": round(float(np.mean(alerts)), 4),
            "max_abs_gap": round(float(np.max(np.abs(np.array(covs) - 0.90))), 4) if covs else None,
        }
        print(f"  gamma={g}: cov={gamma_sens[str(g)]['mean_coverage']} "
              f"alerts={gamma_sens[str(g)]['mean_alert_rate']}", flush=True)
    out["gamma_sensitivity"] = gamma_sens

    # ---- 汇总
    summary = {}
    for alpha in out["alphas"]:
        a = str(alpha)
        cov_f = [f["alphas"][a]["split_conformal"]["coverage"] for f in out["folds"]]
        cov_a = [f["alphas"][a]["aci"]["coverage"] for f in out["folds"]]
        cov_o = [f["alphas"][a]["oracle_fixed"]["coverage"] for f in out["folds"]]
        cov_s = [f["alphas"][a]["swc"]["coverage"] for f in out["folds"]]
        alert_s = [f["alphas"][a]["swc"]["alert_rate"] for f in out["folds"]]
        alert_a = [f["alphas"][a]["aci"]["alert_rate"] for f in out["folds"]]
        alert_f = [f["alphas"][a]["split_conformal"]["alert_rate"] for f in out["folds"]]
        tail_a = [f["alphas"][a]["aci"]["tail20_coverage"] for f in out["folds"]]
        summary[a] = {
            "target": round(1 - alpha, 4),
            "fixed_mean_coverage": round(float(np.mean(cov_f)), 4),
            "aci_mean_coverage": round(float(np.mean(cov_a)), 4),
            "swc_mean_coverage": round(float(np.mean(cov_s)), 4),
            "swc_mean_alert_rate": round(float(np.mean(alert_s)), 4),
            "aci_tail20_mean_coverage": round(float(np.nanmean(tail_a)), 4),
            "oracle_mean_coverage": round(float(np.mean(cov_o)), 4),
            "fixed_mean_alert_rate": round(float(np.mean(alert_f)), 4),
            "aci_mean_alert_rate": round(float(np.mean(alert_a)), 4),
            "fixed_max_abs_gap": round(float(np.max(np.abs(np.array(cov_f) - (1 - alpha)))), 4),
            "aci_max_abs_gap": round(float(np.max(np.abs(np.array(cov_a) - (1 - alpha)))), 4),
        }
    out["summary"] = summary
    print("\n[summary]")
    for a, v in summary.items():
        print(f"  alpha={a}: fixed={v['fixed_mean_coverage']}/{v['fixed_mean_alert_rate']} "
              f"aci={v['aci_mean_coverage']}/{v['aci_mean_alert_rate']} "
              f"swc={v['swc_mean_coverage']}/{v['swc_mean_alert_rate']} target={v['target']}")

    # ---- 图：滚动覆盖率 fixed vs aci（alpha=0.1，fold 0）
    figdir = Path(__file__).resolve().parent.parent / "submission_tdsc" / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "font.family": "serif", "figure.dpi": 200})
    fig, ax = plt.subplots(figsize=(3.5, 2.3))
    f0 = out["folds"][0]["alphas"]["0.1"]
    xs_f = np.linspace(0, 1, len(f0["split_conformal"]["rolling_coverage"]))
    xs_a = np.linspace(0, 1, len(f0["aci"]["rolling_coverage"]))
    ax.plot(xs_f, f0["split_conformal"]["rolling_coverage"], "s-", ms=3, lw=0.9,
            label="split conformal")
    ax.plot(xs_a, f0["aci"]["rolling_coverage"], "o-", ms=3, lw=0.9, label="ACI")
    xs_s = np.linspace(0, 1, len(f0["swc"]["rolling_coverage"]))
    ax.plot(xs_s, f0["swc"]["rolling_coverage"], "^-", ms=3, lw=0.9, label="sliding window")
    ax.axhline(0.90, ls="--", color="#808080", lw=0.8, label="target 0.90")
    ax.set_xlabel("Position in test stream")
    ax.set_ylabel("Rolling coverage")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(figdir / "aci_coverage.pdf", bbox_inches="tight")
    plt.close(fig)

    p = Path(args.outdir) / "aci_conformal.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    write_provenance("aci_conformal", {"status": "PASS", "n_folds": len(out["folds"])})
    print(f"[done] {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
