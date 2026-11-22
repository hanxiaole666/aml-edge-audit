"""
25_generate_figures.py - 生成论文中的所有图表（figures/*.pdf）
=============================================================
从 results/ 的实际结果文件生成 submission_tdsc/figures/ 中的 PDF 图，
保证论文图表与数据一一对应、可复现。

用法：python scripts/25_generate_figures.py
产出：submission_tdsc/figures/{elliptic2_ablation,split_comparison}.pdf
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RESULTS_DIR, write_provenance

ROOT = Path(__file__).resolve().parent.parent
FIGDIR = ROOT / "submission_tdsc" / "figures"


def main() -> int:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "font.family": "serif", "figure.dpi": 200})

    # Fig. 1: Elliptic2 节点特征消融
    e2 = pd.read_csv(RESULTS_DIR / "elliptic2_full_summary.csv")
    lg = e2[e2.model == "elliptic2_full_LightGBM"]
    full_pr = float(lg[lg.split == "random"].PR_AUC_mean.iloc[0])
    topo_pr = 0.0262  # results/elliptic2_summary.csv 中最佳 PR-AUC（仅拓扑特征）
    topo_df = pd.read_csv(RESULTS_DIR / "elliptic2_summary.csv")
    topo_pr = float(topo_df.PR_AUC_mean.max())

    fig, ax = plt.subplots(figsize=(3.4, 1.9))
    bars = ax.bar(["Topology only", "Full features"], [topo_pr, full_pr],
                  color=["#b0b0b0", "#404040"], edgecolor="black", linewidth=0.5, width=0.5)
    for bar, value in zip(bars, [topo_pr, full_pr]):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.015,
                f"{value:.3f}", ha="center", fontsize=8)
    ax.set_ylabel("PR-AUC (LightGBM)")
    ax.set_ylim(0, 0.65)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGDIR / "elliptic2_ablation.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] elliptic2_ablation.pdf: topo={topo_pr:.4f} full={full_pr:.4f}")

    # Fig. 2: 随机 vs 时间分裂
    g = pd.read_csv(RESULTS_DIR / "gnn_summary_realgraph.csv")
    t = pd.read_csv(RESULTS_DIR / "temporal_sota_v2.csv")
    rows = []
    for _, r in g.iterrows():
        rows.append((r["model"].replace("gnn_", ""), r["split"], r["PR_AUC_mean"]))
    for _, r in t.iterrows():
        rows.append((r["model"], r["split"], r["PR-AUC"]))
    df = pd.DataFrame(rows, columns=["model", "split", "pr"])
    piv = df.groupby(["model", "split"])["pr"].mean().unstack()
    order = ["StreamTemporal", "tgn", "dyformer", "tgat", "SGC", "GCN-lite", "SAGE-lite"]
    order = [o for o in order if o in piv.index]
    piv = piv.loc[order]

    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    x = np.arange(len(piv))
    w = 0.35
    ax.bar(x - w / 2, piv["random"], w, label="Random split",
           color="#b0b0b0", edgecolor="black", linewidth=0.5)
    ax.bar(x + w / 2, piv["temporal"], w, label="Temporal split",
           color="#404040", edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([o.replace("StreamTemporal", "Ours") for o in order],
                       rotation=35, ha="right", fontsize=7.5)
    ax.set_ylabel("PR-AUC")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGDIR / "split_comparison.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] split_comparison.pdf: {len(order)} models")

    write_provenance("generate_figures", {"status": "PASS", "n_figures": 2})
    print("[done] figures regenerated from results/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
