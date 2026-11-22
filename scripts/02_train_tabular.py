"""
02_train_tabular.py - 表格轨道：LR / RF / LightGBM / XGBoost（多模型 × 多种子 × 断点续跑）
=====================================================================================
用法：
    python scripts/02_train_tabular.py --data data/german_credit.csv --label label --seeds 5
    python scripts/02_train_tabular.py --data data/german_credit.csv --quick        # 快速冒烟
    python scripts/02_train_tabular.py --data data/german_credit.csv --resume       # 断点续跑（默认开启）

断点续跑：每个 (model, split, seed) 是独立 run，状态写入 results/_checkpoints/tabular.json。
         中断后重跑自动跳过已完成 run；Ctrl+C 安全落盘。
降级：缺 lightgbm/xgboost 时自动跳过对应模型并记录，不影响其他模型。
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    RESULTS_DIR, Checkpoint, compute_metrics, set_seed, write_provenance, append_jsonl,
)
import json

MODELS = ["LogisticRegression", "RandomForest", "LightGBM", "XGBoost"]


def build_model(name: str, seed: int, quick: bool):
    """构建模型；返回 None 表示依赖缺失需跳过。"""
    if name == "LogisticRegression":
        return LogisticRegression(max_iter=1000, C=1.0)
    if name == "RandomForest":
        return RandomForestClassifier(
            n_estimators=60 if quick else 300, max_depth=8,
            min_samples_leaf=2, random_state=seed, n_jobs=-1,
        )
    if name == "LightGBM":
        try:
            from lightgbm import LGBMClassifier
        except ImportError:
            return None
        return LGBMClassifier(
            n_estimators=80 if quick else 400, learning_rate=0.05,
            num_leaves=15, max_depth=6, min_child_samples=10,
            verbose=-1, random_state=seed, n_jobs=-1,
        )
    if name == "XGBoost":
        try:
            from xgboost import XGBClassifier
        except ImportError:
            return None
        return XGBClassifier(
            n_estimators=80 if quick else 400, learning_rate=0.05,
            max_depth=5, subsample=0.9, colsample_bytree=0.9,
            eval_metric="logloss", random_state=seed, n_jobs=-1,
            tree_method="hist",
        )
    raise ValueError(f"未知模型：{name}")


def run_one(df: pd.DataFrame, label: str, model_name: str, seed: int, quick: bool,
            split: str = "random", time_col: str = "time") -> dict:
    set_seed(seed)
    exclude = {label, time_col, "address"}
    feat_cols = [c for c in df.columns if c not in exclude and
                 pd.api.types.is_numeric_dtype(df[c])]
    X = df[feat_cols].values.astype(float)
    y = df[label].values.astype(int)

    if split == "temporal" and time_col in df.columns:
        # 严格时间分裂：train.max_time < test.min_time（无泄露）
        t = df[time_col].values.astype(float)
        n = len(y)
        order = np.argsort(t, kind="mergesort")
        cut = int(0.75 * n)
        tr, te = order[:cut], order[cut:]
        tmax = t[tr].max()
        # 处理并列时间：训练包含所有 <= tmax，测试只取严格更晚
        te = te[t[te] > tmax]
        tr = np.where(t <= tmax)[0]
        if len(te) < 20:
            te = order[cut:]
        X_tr, X_te, y_tr, y_te = X[tr], X[te], y[tr], y[te]
        split_name = "temporal"
    else:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.25, random_state=seed, stratify=y,
        )
        split_name = "random"
    sc = StandardScaler().fit(X_tr)
    X_tr_s, X_te_s = sc.transform(X_tr), sc.transform(X_te)

    model = build_model(model_name, seed, quick)
    if model is None:
        return {"status": "skipped", "reason": "依赖缺失"}

    t0 = time.perf_counter()
    model.fit(X_tr_s, y_tr)
    # 概率为正类
    proba = model.predict_proba(X_te_s)[:, 1]
    elapsed = time.perf_counter() - t0

    met = compute_metrics(y_te, proba)
    met.update({
        "model": f"tabular_{model_name}",
        "seed": seed,
        "split": split_name,
        "fit_seconds": round(elapsed, 3),
        "AUC_sklearn": round(float(roc_auc_score(y_te, proba)), 4),
        "AP_sklearn": round(float(average_precision_score(y_te, proba)), 4),
        "status": "done",
    })
    return met


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data/german_credit.csv")
    ap.add_argument("--label", default="label")
    ap.add_argument("--split", default="random", choices=["random", "temporal"],
                    help="temporal 需数据含 time 列（Real-CATS 支持）")
    ap.add_argument("--time-col", default="time")
    ap.add_argument("--tag", default="", help="数据集标签，用于区分不同数据源的 checkpoint，如 realcats")
    ap.add_argument("--outdir", default=str(RESULTS_DIR))
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--models", nargs="*", default=MODELS)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--resume", action="store_true", default=True, help="断点续跑（默认开启）")
    ap.add_argument("--fresh", action="store_true", help="忽略已有 checkpoint，全部重跑")
    args = ap.parse_args()

    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = Path.cwd() / data_path
    if not data_path.exists():
        print(f"[ERROR] 数据不存在：{data_path}，请先运行 01_download_data.py", file=sys.stderr)
        return 2

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    tag = args.tag or data_path.stem
    ckpt_path = outdir / "_checkpoints" / f"tabular_{tag}_{args.split}.json"
    ckpt = Checkpoint(f"tabular_{tag}_{args.split}", ckpt_path)
    ckpt.install_signal_handler()

    df = pd.read_csv(data_path)
    print(f"[data] {data_path.name}: n={len(df)} features={len(df.columns)-1} pos_rate={df[args.label].mean():.3f}")

    seeds = range(args.seeds)
    all_keys = [f"{m}|{args.split}|seed{s}" for m in args.models for s in seeds]
    todo = ckpt.todo_keys(all_keys) if not args.fresh else all_keys
    print(f"[checkpoint] 总 run={len(all_keys)} 待跑={len(todo)} 已完成={len(all_keys)-len(todo)}")

    preds_rows = []
    for key in todo:
        model_name, _, seed_part = key.split("|")
        seed = int(seed_part.replace("seed", ""))
        ckpt.mark_start(key)
        try:
            res = run_one(df, args.label, model_name, seed, args.quick,
                          split=args.split, time_col=args.time_col)
        except Exception as exc:  # 单 run 失败不拖垮全局
            res = {"status": "failed", "error": f"{type(exc).__name__}: {exc}",
                   "model": f"tabular_{model_name}", "seed": seed}
        if res.get("status") == "done":
            ckpt.mark_done(key, res)  # 存完整指标，便于从 checkpoint 重建汇总
            preds_rows.append(res)
            append_jsonl(outdir / "_checkpoints" / "tabular_metrics.jsonl",
                         {"key": key, **res})
            write_provenance("tabular_run", {"key": key, "metrics": res})
            print(f"  [OK] {key:38s} PR-AUC={res['PR-AUC']:.4f} AUC={res['AUC']:.4f}")
        else:
            ckpt.skip(key)
            write_provenance("tabular_skip", {"key": key, "reason": res})
            print(f"  [SKIP] {key:38s} {res.get('reason') or res.get('error','')[:60]}")

    # 汇总：从 checkpoint 的 done runs 重建完整结果（单一真相源，续跑后依然完整）
    ckpt._flush()
    done_metrics = [
        {"key": k, **v["metrics"]}
        for k, v in ckpt.state["runs"].items()
        if v.get("status") == "done" and isinstance(v.get("metrics"), dict) and "PR-AUC" in v["metrics"]
    ]
    # 老格式 checkpoint（只存 PR-AUC/AUC）没有完整指标时，退化为用本次结果
    full_done = [m for m in done_metrics if "AUC" in m and "ECE" in m]
    result_path = outdir / f"tabular_results_{tag}_{args.split}.csv"
    if full_done:
        sdf = pd.DataFrame(full_done).drop(columns=["key"], errors="ignore")
    elif preds_rows:
        sdf = pd.DataFrame(preds_rows)
    else:
        sdf = pd.DataFrame()
    if len(sdf):
        sdf["dataset"] = tag
        sdf.to_csv(result_path, index=False, float_format="%.6f")
        agg = sdf.groupby("model").agg(
            AUC_mean=("AUC", "mean"), AUC_std=("AUC", "std"),
            PR_AUC_mean=("PR-AUC", "mean"), PR_AUC_std=("PR-AUC", "std"),
            ECE_mean=("ECE", "mean"), n_seeds=("seed", "nunique"),
        ).reset_index()
        agg.to_csv(outdir / f"tabular_summary_{tag}_{args.split}.csv", index=False, float_format="%.6f")
        print(f"\n[汇总] tabular_summary_{tag}_{args.split}.csv")
        print(agg.to_string(index=False))
    print(f"\n[done] 表格轨道完成，checkpoint={ckpt_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
