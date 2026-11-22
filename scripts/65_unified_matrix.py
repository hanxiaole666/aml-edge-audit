"""
65_unified_matrix.py - E6：跨数据集统一结果矩阵 + E1 的非退化折汇总
======================================================================
把四个数据集、同一套切分语言的结果并入一张表。协议标签写清楚，避免把
不可比的数字并排放：

    D1  Elliptic v1        49 真实时间步，6 折 rolling-origin（5 折非退化）
    D2  Elliptic2          发布版子图拓扑，random 80:10:10
    D3  Real-CATS ETH      地址 first-seen 时间切分 75/25
    D4  Real-CATS BTC      地址 first-seen 时间切分 75/25（跨链脚本的域内列）

同时修正 E1 的汇总量：最后一折 test prevalence 0.011，属于退化折，必须
从 mean±sd 中剔除，否则 0.77±0.35 这种数字没有意义。

用法：python scripts/65_unified_matrix.py
产出：results/unified_matrix.json + figures/random_vs_temporal.pdf
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RESULTS_DIR, set_seed, write_provenance

DEGENERATE_PREV = 0.05


def load(name):
    p = RESULTS_DIR / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", default=str(RESULTS_DIR))
    args = ap.parse_args()
    set_seed(0)

    em = load("elliptic_rolling_matrix.json")
    e2 = load("elliptic2_subgraph_models.json")
    lp = load("label_provenance_audit.json")
    cc = load("cross_chain.json")
    ep = load("edge_protocol_final.json")

    # ---------------- E1 非退化折汇总
    folds = [f for f in em["folds_detail"] if f["test_prevalence"] >= DEGENERATE_PREV]
    dropped = [f["cutoff"] for f in em["folds_detail"]
               if f["test_prevalence"] < DEGENERATE_PREV]
    summ = {}
    for model in ["lightgbm", "gcn", "sage", "tgn_lite"]:
        t = np.array([f["models"][model]["temporal"] for f in folds])
        r = np.array([f["models"][model]["random"] for f in folds])
        d = r - t
        n = len(d)
        se = float(np.std(d, ddof=1) / np.sqrt(n)) if n > 1 else 0.0
        summ[model] = {
            "temporal_mean": round(float(t.mean()), 4),
            "temporal_sd": round(float(t.std(ddof=1)), 4) if n > 1 else 0.0,
            "random_mean": round(float(r.mean()), 4),
            "delta_mean": round(float(d.mean()), 4),
            "delta_ci95": [round(float(d.mean() - 2.776 * se), 4),
                           round(float(d.mean() + 2.776 * se), 4)],
            "folds_with_positive_delta": int((d > 0).sum()),
            "n_folds": n,
        }
    em["summary_nondegenerate"] = summ
    em["dropped_folds"] = dropped
    (RESULTS_DIR / "elliptic_rolling_matrix.json").write_text(
        json.dumps(em, ensure_ascii=False, indent=2, default=float), encoding="utf-8")

    # ---------------- 统一矩阵
    rows = []
    for model in ["lightgbm", "gcn", "sage", "tgn_lite"]:
        rows.append({"dataset": "Elliptic v1", "model": model,
                     "AP": summ[model]["temporal_mean"],
                     "n_test_mean": int(np.mean([f["n_test"] for f in folds])),
                     "prevalence_mean": round(float(np.mean(
                         [f["test_prevalence"] for f in folds])), 4),
                     "protocol": "6-fold rolling origin, 5 non-degenerate",
                     "AP_random": summ[model]["random_mean"]})
    if e2:
        for model, v in e2["models"].items():
            rows.append({"dataset": "Elliptic2", "model": model, "AP": v["test_AP"],
                         "n_test_mean": e2["split"]["test"],
                         "prevalence_mean": e2["split"]["test_prevalence"],
                         "protocol": "random 80:10:10", "AP_random": None})
    # Real-CATS 的表格轨统一取 66 号脚本（5 seeds 的 temporal/random 配对），
    # 不再用 64 号 provenance 审计里的单次运行值：同一名义数字出现过 0.798/0.799/0.801
    # 三个版本，表格里只能留一个。
    tt = load("tabular_random_temporal.json")
    if tt:
        for mdl in ["LightGBM", "RandomForest", "Logistic"]:
            v = tt["models"][mdl]
            rows.append({"dataset": "Real-CATS ETH", "model": mdl,
                         "AP": v["AP_temporal"], "n_test_mean": tt["n_test"],
                         "prevalence_mean": tt["prev_temporal"],
                         "protocol": "time 75/25, provenance removed",
                         "AP_random": v["AP_random"]})
    # BTC 的域内数字此前只有随机切分（0.88/0.93）。67 号脚本按 first_time 做时间
    # 切分后才给出可比的域内值，且它同时是退化折：晚期阳性率 2.8%。
    op = load("operating_points.json")
    if op and op.get("datasets", {}).get("Real-CATS BTC"):
        b = op["datasets"]["Real-CATS BTC"]
        v = b["models"].get("LightGBM", {})
        if v:
            rows.append({"dataset": "Real-CATS BTC", "model": "LightGBM",
                         "AP": v["AP"]["mean"], "n_test_mean": b["n_test"],
                         "prevalence_mean": b["test_prevalence"],
                         "protocol": ("time 75/25 on first_time; degenerate "
                                      f"(prev {b['test_prevalence']:.3f})"),
                         "AP_random": cc["models"]["RandomForest"]["BTC_internal"] if cc else None})

    out = {"experiment": "unified_matrix", "rows": rows,
           "provenance_removed": True,
           "caveat": ("Protocols differ across datasets; the table is a map of what exists, "
                      "not a leaderboard. Cross-dataset comparisons inherit the split "
                      "difference and are labelled accordingly."),
           "status": "PASS"}
    if ep:
        out["real_cats_edge_conditions"] = {
            k: v["AP_mean"] for k, v in ep["aggregate"].items()}
    print(json.dumps(rows, indent=1), flush=True)

    p = Path(args.outdir) / "unified_matrix.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2, default=float), encoding="utf-8")

    # ---------------- E1 图：random vs temporal
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.4))
    xs = np.arange(len(folds))
    colors = {"lightgbm": "#c0392b", "gcn": "#2980b9", "sage": "#27ae60",
              "tgn_lite": "#8e44ad"}
    for model in ["lightgbm", "gcn", "sage", "tgn_lite"]:
        t = [f["models"][model]["temporal"] for f in folds]
        r = [f["models"][model]["random"] for f in folds]
        ax[0].plot(xs, t, "o-", ms=3, lw=1, color=colors[model], label=f"{model} temporal")
        ax[0].plot(xs, r, "s--", ms=2.5, lw=0.8, color=colors[model], alpha=0.65)
    ax[0].set_xticks(xs)
    ax[0].set_xticklabels([f"c={f['cutoff']}" for f in folds], fontsize=6)
    ax[0].set_ylabel("PR-AUC", fontsize=7)
    ax[0].tick_params(labelsize=6)
    ax[0].legend(fontsize=4.2, frameon=False, ncol=2)
    ax[0].grid(alpha=0.25, lw=0.5)

    names = ["lightgbm", "gcn", "sage", "tgn_lite"]
    vals = [summ[m]["delta_mean"] for m in names]
    err = [[summ[m]["delta_mean"] - summ[m]["delta_ci95"][0] for m in names],
           [summ[m]["delta_ci95"][1] - summ[m]["delta_mean"] for m in names]]
    ax[1].bar(range(len(names)), vals, yerr=err, capsize=3, color="#c0392b", alpha=0.8)
    ax[1].axhline(0, color="#34495e", lw=0.8)
    ax[1].set_xticks(range(len(names)))
    ax[1].set_xticklabels(names, fontsize=6, rotation=15)
    ax[1].set_ylabel("random − temporal AP", fontsize=7)
    ax[1].tick_params(labelsize=6)
    ax[1].grid(alpha=0.25, lw=0.5)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR.parent / "figures" / "random_vs_temporal.pdf",
                bbox_inches="tight")
    print("[fig] random_vs_temporal.pdf")

    print("[e1 summary]", json.dumps(summ, indent=1), flush=True)
    write_provenance("unified_matrix", {"status": "PASS"})
    print(f"[done] {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
