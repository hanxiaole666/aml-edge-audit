"""
76_make_figures.py - 论文三张核心图
====================================
F1 审稿意见：全文 14 表 0 图，而最有说服力的三个结果本质上是图。
本脚本生成：
  fig_injection.pdf   λ -> ΔAP，两张图并排（Elliptic 与 Real-CATS）
  fig_conditions.pdf  两图四条件并排条形
  fig_rolling.pdf     十折 temporal vs random（LightGBM）

用法：python scripts/76_make_figures.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 300, "savefig.bbox": "tight", "pdf.fonttype": 42,
})

C = {"C1": "#4C72B0", "C2": "#55A868", "C3": "#C44E52", "M2": "#8172B2"}


def load(name):
    with open(RES / f"{name}.json", encoding="utf-8") as f:
        return json.load(f)


def fig_injection():
    el = load("step_coarsening")
    rc = load("cross_split_injection")

    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.5), sharey=True)
    lams = [-1, -0.5, 0, 0.5, 1]

    for ax, K, ls, mk in ((axes[0], 8000, "-", "o"), (axes[0], 40000, "--", "s")):
        y = [next(r["delta_AP"] for r in el["injection"]
                  if r["K_target"] == K and abs(r["lambda"] - l) < 1e-9) for l in lams]
        ax.plot(lams, y, ls, marker=mk, ms=3, lw=1.2,
                label=f"$K$={K:,}", color="#C44E52" if K == 8000 else "#4C72B0")
    ax = axes[0]
    ax.axhline(0, color="0.6", lw=0.7, zorder=0)
    ax.set_title("(a) Elliptic, test–test edges")
    ax.set_xlabel("same-label rate $\\lambda$")
    ax.set_ylabel("$\\Delta$AP vs no injection")
    ax.set_xticks(lams)
    ax.legend(frameon=False, loc="upper left")

    ax = axes[1]
    for K, ls, mk, col in ((2000, "-", "o", "#C44E52"), (6000, "--", "s", "#4C72B0")):
        y = [next(r["delta_AP"] for r in rc["rows"]
                  if r["K"] == K and abs(r["lambda"] - l) < 1e-9) for l in lams]
        ax.plot(lams, y, ls, marker=mk, ms=3, lw=1.2, label=f"$K$={K:,}", color=col)
    ax.axhline(0, color="0.6", lw=0.7, zorder=0)
    ax.set_title("(b) Real-CATS, train$\\to$test edges")
    ax.set_xlabel("same-label rate $\\lambda$")
    ax.set_xticks(lams)
    ax.legend(frameon=False, loc="upper left")

    fig.tight_layout()
    fig.savefig(FIG / "fig_injection.pdf")
    plt.close(fig)
    print("[ok] fig_injection.pdf")


def fig_conditions():
    ep = load("edge_protocol_final")["aggregate"]
    el = load("elliptic_v1_strong_model")["aggregate_nondegenerate"]

    fig, ax = plt.subplots(figsize=(3.45, 2.4))
    conds = ["C1", "C2", "C3", "M2"]
    x = np.arange(len(conds))
    w = 0.36
    rc_v = [ep[c]["AP_mean"] for c in conds]
    el_v = [el[c]["mean"] for c in conds]

    b1 = ax.bar(x - w / 2, rc_v, w, label="Real-CATS", color="#C44E52", alpha=0.9)
    b2 = ax.bar(x + w / 2, el_v, w, label="Elliptic", color="#4C72B0", alpha=0.9)
    for bars in (b1, b2):
        for r in bars:
            ax.annotate(f"{r.get_height():.3f}", (r.get_x() + r.get_width() / 2, r.get_height()),
                        xytext=(0, 2), textcoords="offset points", ha="center", fontsize=6)

    ax.set_xticks(x)
    ax.set_xticklabels(conds)
    ax.set_ylabel("PR-AUC")
    ax.set_ylim(0.35, 0.86)
    ax.legend(frameon=False, ncol=2, loc="lower left")
    ax.set_title("Same protocol change, opposite sign")
    fig.tight_layout()
    fig.savefig(FIG / "fig_conditions.pdf")
    plt.close(fig)
    print("[ok] fig_conditions.pdf")


def fig_rolling():
    d = load("elliptic_rolling_matrix")
    fd = d["folds_detail"]
    cuts = [r["cutoff"] for r in fd]
    models = [("lightgbm", "LightGBM"), ("gcn", "GCN"),
              ("sage", "SAGE"), ("tgn_lite", "TGN-lite")]

    fig, axes = plt.subplots(2, 2, figsize=(3.45, 3.8), sharex=True)
    for ax, (key, lab) in zip(axes.ravel(), models):
        t = [r["models"][key]["temporal"] for r in fd]
        r = [r["models"][key]["random"] for r in fd]
        sm = d["summary"][key]
        ax.plot(cuts, r, "-o", ms=2.2, lw=1.0, color="#C44E52", label="random")
        ax.plot(cuts, t, "-s", ms=2.2, lw=1.0, color="#4C72B0", label="temporal")
        ax.fill_between(cuts, t, r, where=np.array(r) > np.array(t),
                        color="#C44E52", alpha=0.12, interpolate=True)
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(f"{lab}: $\\Delta$={sm['delta_mean']:+.3f}, "
                     f"{sm['delta_positive_folds']}/10 folds", fontsize=7)
        ax.legend(frameon=False, fontsize=6, loc="lower left")
    for ax in axes[1, :]:
        ax.set_xlabel("cutoff step $c$")
    for ax in axes[:, 0]:
        ax.set_ylabel("PR-AUC")
    fig.tight_layout()
    fig.savefig(FIG / "fig_rolling.pdf")
    plt.close(fig)
    print("[ok] fig_rolling.pdf")


def main() -> int:
    fig_injection()
    fig_conditions()
    fig_rolling()
    return 0


if __name__ == "__main__":
    sys.exit(main())
