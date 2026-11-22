"""
53_plot_elliptic_strong.py - 从结果文件绘制 Elliptic 四条件图
======================================================
数据源：results/elliptic_v1_strong_model.json（51 脚本产出）。
图用于论文 Fig. 2：五个时间切分上四条件的 AP，展示与 Real-CATS 相反的排序。

用法：python scripts/53_plot_elliptic_strong.py
产出：submission_tdsc/figures/elliptic_v1_strong.pdf
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RESULTS_DIR


def main() -> int:
    d = json.loads((RESULTS_DIR / "elliptic_v1_strong_model.json").read_text(encoding="utf-8"))
    splits = d["splits"]
    xs = [s["cutoff"] for s in splits]
    plt.rcParams.update({"font.size": 9, "font.family": "serif", "figure.dpi": 200})
    fig, ax = plt.subplots(figsize=(3.2, 2.3))
    for c, mk, col in [("C1", "o", "#1f4e79"), ("C2", "s", "#c0504d"),
                       ("C3", "^", "#4f8a4f"), ("M2", "x", "#7f7f7f")]:
        ys = [s["conditions"][c]["AP"] for s in splits]
        ax.plot(xs, ys, mk + "-", ms=4, lw=1.1, color=col, label=c)
    ax.set_xlabel("Training cutoff (time step)")
    ax.set_ylabel("PR-AUC on future steps")
    ax.set_ylim(-0.03, 0.98)
    ax.legend(fontsize=6.5, ncol=2, loc="center left", framealpha=0.95)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = Path(__file__).resolve().parent.parent / "submission_tdsc" / "figures" / "elliptic_v1_strong.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[done] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
