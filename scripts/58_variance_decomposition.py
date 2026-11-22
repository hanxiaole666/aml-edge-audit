"""
58_variance_decomposition.py - E8：多折 + 三源方差分解
======================================================================
现有 4 折 rolling-origin 的 SD 是 0.131，最后一折崩到 0.688，且没有解释。
"0.893" 这种三位小数在 4 折 SD=0.131 的情况下是不可辩护的。本脚本：

1. 把 rolling-origin 扩到 10 折（Real-CATS ETH，按 first-seen 时间排序）。
2. 每折报 n_test、test prevalence、时间跨度（天）。
3. 三源方差分解（两因子随机效应，矩估计）：
       AP_{f,s} = mu + a_f + b_s + e_{fs}
       a_f ~ 折（切分）效应，b_s ~ 种子效应，e ~ 残差
       sigma2_fold  = max(0, MS_fold - MS_resid) / S
       sigma2_seed  = max(0, MS_seed - MS_resid) / F
       sigma2_resid = MS_resid
   另报 bootstrap 方差：每折内对测试集重采样的 AP 分布方差，取均值。
4. 用分解结果给"该报几位小数"的结论：若 sigma_fold 占主导，则点估计不可
   精确到小数点后三位，应报区间。

用法：python scripts/58_variance_decomposition.py [--folds 10] [--seeds 3]
产出：results/variance_decomposition.json + figures/variance_folds.pdf
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
    ap.add_argument("--folds", type=int, default=10)
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()
    set_seed(0)

    X, y, t, _feats = load_tabular()
    order = np.argsort(t, kind="mergesort")
    print(f"[data] n={len(y)} prev={y.mean():.4f} span_days="
          f"{(t.max()-t.min())/86400:.0f}", flush=True)

    fracs = np.linspace(0.45, 0.90, args.folds)
    folds = []
    for k, frac in enumerate(fracs):
        cut = int(frac * len(order))
        tr, te = order[:cut], order[cut:]
        tmax = t[tr].max()
        te = te[t[te] > tmax]                    # embargo：严格晚于训练末尾
        if len(te) < 150 or len(np.unique(y[te])) < 2:
            print(f"[skip] frac={frac:.2f} n_test={len(te)}", flush=True)
            continue
        sc = StandardScaler().fit(X[tr])
        aps = []
        for sd in range(args.seeds):
            m = LGBMClassifier(n_estimators=350, learning_rate=0.05, num_leaves=31,
                               subsample=0.9, colsample_bytree=0.8, random_state=sd,
                               n_jobs=-1, verbose=-1,
                               scale_pos_weight=(len(tr) - y[tr].sum()) / max(y[tr].sum(), 1))
            m.fit(sc.transform(X[tr]), y[tr])
            s = m.predict_proba(sc.transform(X[te]))[:, 1]
            aps.append(float(pr_auc(y[te], s)))
            if sd == 0:
                # 折内 bootstrap 方差
                rng = np.random.default_rng(k)
                bv = []
                for _ in range(300):
                    idx = rng.integers(0, len(te), len(te))
                    if len(np.unique(y[te][idx])) < 2:
                        continue
                    bv.append(pr_auc(y[te][idx], s[idx]))
                boot_var = float(np.var(bv))
        folds.append({"fold": k, "frac": round(float(frac), 3), "n_train": int(len(tr)),
                      "n_test": int(len(te)), "prev_test": round(float(y[te].mean()), 4),
                      "span_days": round(float((t[te].max() - t[te].min()) / 86400), 1),
                      "t_end": round(float(t[te].max()), 0),
                      "AP_mean": round(float(np.mean(aps)), 4),
                      "AP_per_seed": [round(a, 4) for a in aps],
                      "boot_var": boot_var})
        print(f"  fold{k} frac={frac:.2f} n={len(te)} prev={y[te].mean():.3f} "
              f"AP={np.mean(aps):.4f}", flush=True)

    F, S = len(folds), args.seeds
    M = np.array([[f["AP_per_seed"][s] for s in range(S)] for f in folds])
    grand = M.mean()
    ms_fold = S * ((M.mean(axis=1) - grand) ** 2).sum() / max(F - 1, 1)
    ms_seed = F * ((M.mean(axis=0) - grand) ** 2).sum() / max(S - 1, 1)
    resid = M - M.mean(axis=1, keepdims=True) - M.mean(axis=0, keepdims=True) + grand
    ms_res = (resid ** 2).sum() / max((F - 1) * (S - 1), 1)
    s2_fold = max(0.0, (ms_fold - ms_res) / S)
    s2_seed = max(0.0, (ms_seed - ms_res) / F)
    s2_res = ms_res
    s2_boot = float(np.mean([f["boot_var"] for f in folds]))
    tot = s2_fold + s2_seed + s2_res + s2_boot

    out = {"experiment": "variance_decomposition", "folds": folds,
           "grand_mean_AP": round(float(grand), 4),
           "fold_SD": round(float(np.std(M.mean(axis=1), ddof=1)), 4),
           "variance_components": {
               "fold": round(s2_fold, 6), "seed": round(s2_seed, 6),
               "residual": round(s2_res, 6), "bootstrap": round(s2_boot, 6)},
           "share": {k: round(v / tot, 3) for k, v in
                     [("fold", s2_fold), ("seed", s2_seed),
                      ("residual", s2_res), ("bootstrap", s2_boot)]},
           "status": "PASS"}
    # 可信区间：grand mean ± 1.96 * sqrt(s2_fold/F + s2_res/(F*S))
    se = np.sqrt(s2_fold / max(F, 1) + s2_res / max(F * S, 1))
    out["grand_mean_CI95"] = [round(float(grand - 1.96 * se), 4),
                              round(float(grand + 1.96 * se), 4)]
    out["recommended_precision"] = ("one decimal" if np.sqrt(s2_fold) > 0.05 else "two decimals")
    out["verdict"] = {
        "dominant_source": max(out["share"], key=out["share"].get),
        "fold_SD": out["fold_SD"],
        "note": ("Point estimates to three decimals are not defensible when the fold "
                 "component dominates; we report the grand mean with a CI and keep "
                 "three decimals only for within-fold comparisons."),
    }
    print("[components]", out["variance_components"], out["share"], flush=True)
    print("[CI]", out["grand_mean_CI95"], out["verdict"], flush=True)

    p = Path(args.outdir) / "variance_decomposition.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2, default=float), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(4.2, 2.5))
    xs = [f["fold"] for f in folds]
    ax.plot(xs, [f["AP_mean"] for f in folds], "o-", color="#c0392b", ms=4)
    ax.fill_between(xs, [f["AP_mean"] - 1.96 * np.sqrt(f["boot_var"]) for f in folds],
                    [f["AP_mean"] + 1.96 * np.sqrt(f["boot_var"]) for f in folds],
                    color="#c0392b", alpha=0.18, lw=0)
    ax.axhline(out["grand_mean_AP"], ls="--", lw=1, color="#34495e")
    ax.set_xlabel("rolling-origin fold", fontsize=8)
    ax.set_ylabel("PR-AUC", fontsize=8)
    ax.tick_params(labelsize=6)
    ax.grid(alpha=0.25, lw=0.5)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR.parent / "figures" / "variance_folds.pdf", bbox_inches="tight")
    print("[fig] variance_folds.pdf")

    write_provenance("variance_decomposition", {"status": "PASS"})
    print(f"[done] {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
