"""
05_evaluate.py - 统一评估：汇总所有轨道结果 + 泄露量化 + 生成论文级表格
=============================================================================
用法：
    python scripts/05_evaluate.py                          # 汇总 results/ 下所有 *_results.csv
    python scripts/05_evaluate.py --leakage                # 额外生成泄露量化表（随机 vs 时间分裂）

产出：
    results/final_summary.csv        统一汇总（所有模型/轨道）
    results/leakage_quantification.csv  泄露量化表（C2 主张证据）
    results/paper_tables.md          论文可直接引用的 Markdown 表格
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

sys_path = str(Path(__file__).resolve().parent)
import sys
sys.path.insert(0, sys_path)
from common import RESULTS_DIR, write_provenance  # noqa: E402


def load_all(outdir: Path) -> pd.DataFrame:
    frames = []
    # 匹配带 tag 后缀的结果文件（如 tabular_results_realcats_random.csv）
    for p in sorted(outdir.glob("*_results*.csv")):
        try:
            df = pd.read_csv(p)
            if len(df):
                df["_source"] = p.name
                frames.append(df)
        except Exception as exc:
            print(f"[warn] 读取 {p.name} 失败：{exc}")
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def fmt(x, nd=4):
    if pd.isna(x):
        return "—"
    return f"{x:.{nd}f}"


def build_paper_tables(df: pd.DataFrame, outdir: Path) -> None:
    lines = ["# TinyAML-Bench 实验结果表（自动生成，冒烟/论文级以 provenance 为准）", ""]
    lines.append("## 表1 主结果（各轨道，mean±std over seeds）")
    lines.append("")
    if "model" in df.columns:
        group_cols = ["model"]
        if "split" in df.columns:
            group_cols.append("split")
        agg = df.groupby(group_cols).agg(
            AUC_mean=("AUC", "mean"), AUC_std=("AUC", "std"),
            PR_AUC_mean=("PR-AUC", "mean"), PR_AUC_std=("PR-AUC", "std"),
            ECE_mean=("ECE", "mean"), n=("seed", "nunique") if "seed" in df.columns else ("AUC", "size"),
        ).reset_index()
        cols = group_cols + ["AUC_mean", "AUC_std", "PR_AUC_mean", "PR_AUC_std", "ECE_mean", "n"]
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
        for _, r in agg.iterrows():
            vals = [str(r[c]) if c in group_cols else fmt(r[c], 4) for c in cols]
            lines.append("| " + " | ".join(vals) + " |")
    lines.append("")

    # 表2 泄露量化（按数据集分组，避免混合失真）
    if {"model", "split"}.issubset(df.columns):
        lines.append("## 表2 泄露量化：随机分裂 vs 时间分裂（C2 证据，按数据集分组）")
        lines.append("")
        ds_col = "dataset" if "dataset" in df.columns else None
        groups = list(df.groupby(ds_col)) if ds_col else [("all", df)]
        wrote = False
        for ds_name, g in groups:
            gg = g[g["split"].isin(["random", "temporal"])]
            if gg.empty:
                continue
            agg = gg.groupby(["model", "split"])["PR-AUC"].mean().unstack()
            if "random" not in agg.columns or "temporal" not in agg.columns:
                continue
            lines.append(f"### 数据集：{ds_name}")
            lines.append("| model | random PR-AUC | temporal PR-AUC | 落差 | 落差% |")
            lines.append("| --- | --- | --- | --- | --- |")
            for model_name in agg.index:
                rnd = float(agg.loc[model_name, "random"])
                tmp = float(agg.loc[model_name, "temporal"])
                drop = rnd - tmp
                pct = 100 * drop / rnd if rnd else 0.0
                lines.append(f"| {model_name} | {fmt(rnd)} | {fmt(tmp)} | {fmt(drop)} | {fmt(pct, 1)}% |")
            lines.append("")
            wrote = True
        if wrote:
            lines.append("> 解读：随机分裂系统性高估性能；落差即为「标签泄露水分」的量化。")
        else:
            lines.append("_（尚无同时包含 random 与 temporal 分裂的结果）_")
    (outdir / "paper_tables.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[paper_tables] 已生成 {outdir / 'paper_tables.md'}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", default=str(RESULTS_DIR))
    ap.add_argument("--leakage", action="store_true", default=True)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    df = load_all(outdir)
    if df.empty:
        print("[ERROR] results/ 下无 *_results.csv，请先运行 02/03/04", file=sys.stderr)
        return 2

    # 统一汇总
    keep = [c for c in ["model", "split", "seed", "k", "AUC", "PR-AUC", "F1",
                        "P@k", "ECE", "n", "pos_rate", "status"] if c in df.columns]
    summary = df[keep].copy()
    summary.to_csv(outdir / "final_summary.csv", index=False, float_format="%.6f")

    print(f"[汇总] 共 {len(summary)} 条结果，来自 {summary['_source'].nunique() if '_source' in summary else '?'} 个文件")
    print(summary.groupby("model")["PR-AUC"].agg(["mean", "std", "count"]).to_string())

    # 泄露量化表
    if args.leakage and "split" in df.columns:
        gnn = df[df["split"].isin(["random", "temporal"])]
        if not gnn.empty:
            # 按数据集分组分别量化（不同数据集混合会失真，论文必须分组报告）
            ds_col = "dataset" if "dataset" in gnn.columns else None
            groups = list(gnn.groupby(ds_col)) if ds_col else [("all", gnn)]
            tables = []
            for ds_name, g in groups:
                piv = g.pivot_table(index="model", columns="split", values="PR-AUC", aggfunc="mean")
                if {"random", "temporal"}.issubset(piv.columns):
                    piv["PR_AUC_drop"] = piv["random"] - piv["temporal"]
                    piv["drop_pct"] = 100 * piv["PR_AUC_drop"] / piv["random"]
                    piv.insert(0, "dataset", ds_name)
                    tables.append(piv.reset_index())
            if tables:
                leak = pd.concat(tables, ignore_index=True)
                leak.to_csv(outdir / "leakage_quantification.csv", index=False, float_format="%.6f")
                print("\n[泄露量化] leakage_quantification.csv（按数据集分组）")
                print(leak.to_string(index=False))

    build_paper_tables(df, outdir)

    # Wilcoxon 符号秩检验：同 split 下两两模型 PR-AUC 显著性（补实验 #9）
    sig_path = outdir / "significance_wilcoxon.csv"
    try:
        from scipy.stats import wilcoxon
        rows_sig = []
        # 按 split 分组；无 split 列时整体作为一组
        groups = list(df.groupby("split")) if "split" in df.columns else [("all", df)]
        for sp_name, g in groups:
            pivot = g.pivot_table(index="seed", columns="model", values="PR-AUC", aggfunc="mean") \
                if "seed" in g.columns else None
            if pivot is None or pivot.shape[1] < 2:
                continue
            models = list(pivot.columns)
            for i in range(len(models)):
                for j in range(i + 1, len(models)):
                    a, b = pivot[models[i]].dropna(), pivot[models[j]].dropna()
                    common_idx = a.index.intersection(b.index)
                    if len(common_idx) < 3:
                        continue
                    diff = a[common_idx] - b[common_idx]
                    if (diff == 0).all():
                        pval = 1.0
                    else:
                        try:
                            pval = float(wilcoxon(a[common_idx], b[common_idx]).pvalue)
                        except Exception:
                            pval = float("nan")
                    rows_sig.append({"split": sp_name, "model_a": models[i],
                                     "model_b": models[j], "mean_diff": float(diff.mean()),
                                     "p_value": pval, "significant_0.05": bool(pval < 0.05)})
        if rows_sig:
            pd.DataFrame(rows_sig).to_csv(sig_path, index=False, float_format="%.6f")
            print(f"\n[显著性] {sig_path}（{len(rows_sig)} 对比较）")
    except Exception as exc:
        print(f"[warn] Wilcoxon 检验跳过：{exc}")

    write_provenance("evaluation", {"n_results": len(df), "models": df["model"].nunique() if "model" in df else 0})
    print(f"\n[done] 评估完成：final_summary.csv / paper_tables.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
