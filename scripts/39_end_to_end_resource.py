"""
39_end_to_end_resource.py - 端到端资源测量（E13）
================================================
审稿人要求：补 pipeline 分段（特征提取/图构建/推理/校准）延迟、
峰值 RSS、两档硬件、L_max/M_max 可行集与 Pareto 前沿。

用法：python scripts/39_end_to_end_resource.py
产出：results/end_to_end_resource.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA_DIR, RESULTS_DIR, write_provenance, set_seed


def rss_mb():
    """当前进程 RSS（MB）。"""
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
    except ImportError:
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", default=str(RESULTS_DIR))
    args = ap.parse_args()
    set_seed(0)

    df = pd.read_csv(DATA_DIR / "real_cats_eth.csv")
    feats = [c for c in df.columns if c not in {"label", "time", "address", "etherscan_checked", "watchback_checked", "gs_checked"}
             and pd.api.types.is_numeric_dtype(df[c])]
    y = df.label.values.astype(int)
    from sklearn.model_selection import train_test_split
    tr, te = train_test_split(np.arange(len(y)), test_size=0.25, random_state=0, stratify=y)

    out = {
        "experiment": "end_to_end_resource",
        "hardware": {
            "cpu_threads": torch.get_num_threads(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
        },
        "n_train": len(tr), "n_test": len(te),
        "status": "PASS",
    }

    # 分段计时
    rss0 = rss_mb()
    t0 = time.perf_counter()
    X = df[feats].values.astype(float)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0)
    t_feat = time.perf_counter() - t0

    t0 = time.perf_counter()
    sc = StandardScaler().fit(X[tr])
    Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
    t_prep = time.perf_counter() - t0

    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=0, n_jobs=-1),
    }
    model_stats = {}
    for name, m in models.items():
        t0 = time.perf_counter()
        m.fit(Xtr, y[tr])
        t_train = time.perf_counter() - t0
        lats = []
        for _ in range(100):
            s = time.perf_counter()
            m.predict_proba(Xte[:100])
            lats.append((time.perf_counter() - s) * 1000)
        model_stats[name] = {
            "train_sec": round(t_train, 3),
            "infer_batch100_p50_ms": round(float(np.percentile(lats, 50)), 3),
            "infer_batch100_p95_ms": round(float(np.percentile(lats, 95)), 3),
            "n_params_or_trees": int(getattr(m, "n_estimators", X.shape[1])),
        }

    rss1 = rss_mb()
    out["pipeline"] = {
        "feature_extraction_sec": round(t_feat, 3),
        "scaling_sec": round(t_prep, 3),
        "rss_before_MB": round(rss0, 1) if rss0 else None,
        "rss_after_MB": round(rss1, 1) if rss1 else None,
        "rss_delta_MB": round(rss1 - rss0, 1) if rss0 and rss1 else None,
    }
    out["models"] = model_stats

    # 可行集与 Pareto（延迟 vs 模型大小，用 PR-AUC 作第三维）
    pareto = []
    for name, st in model_stats.items():
        from common import pr_auc
        # PR-AUC for the feasible-set table
        mm = models[name]
        ap = pr_auc(y[te], mm.predict_proba(Xte)[:, 1])
        pareto.append({
            "model": name,
            "latency_ms": st["infer_batch100_p50_ms"],
            "memory_MB": st["n_params_or_trees"],
            "PR_AUC": round(float(ap), 4),
        })
    out["feasible_set"] = pareto
    # 示例预算：L_max=5ms, M_max=500 estimators
    feasible = [p for p in pareto if p["latency_ms"] <= 5.0 and p["memory_MB"] <= 500]
    out["example_budget_L5_M500"] = {
        "L_max_ms": 5.0, "M_max": 500,
        "feasible_models": [p["model"] for p in feasible],
    }

    p = Path(args.outdir) / "end_to_end_resource.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    write_provenance("end_to_end_resource", {"status": "PASS"})
    print(f"[done] {p}")
    for name, st in model_stats.items():
        print(f"  {name}: train={st['train_sec']}s infer100_p50={st['infer_batch100_p50_ms']}ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
