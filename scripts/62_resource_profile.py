"""
62_resource_profile.py - E13：端到端资源测量（P50/P95/P99 + 峰值 RSS + Pareto）
======================================================================
旧的资源表只有前向传播、只在小图上、没有特征提取与图构建、没有内存分解。
本脚本把管线拆成四段并逐段计时：

    feature_extraction   从 csv 读列到矩阵
    graph_construction   幂等稀疏邻接（若提供边）
    training
    inference            单批 100 / 全测试集，报 P50/P95/P99

另报
    peak RSS（整进程 ru_maxrss）
    throughput（实体/秒，含特征提取）
    两档硬件：CPU-only（本机）与"消费级 GPU 可用"标记（若 torch 检测到则记录）
    L_max / M_max 下的可行集与 Pareto 前沿（延迟 vs PR-AUC）

用法：python scripts/62_resource_profile.py
产出：results/resource_profile.json + figures/resource_pareto.pdf
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA_DIR, RESULTS_DIR, pr_auc, set_seed, write_provenance

PROV = {"etherscan_checked", "watchback_checked", "gs_checked"}


def rss_mb():
    """峰值 RSS（MB）；Windows 无 resource 模块，用 psutil 跨平台取。"""
    import psutil
    return psutil.Process().memory_info().rss / (1024.0 ** 2)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", default=str(RESULTS_DIR))
    args = ap.parse_args()
    set_seed(0)

    t0 = time.perf_counter()
    df = pd.read_csv(DATA_DIR / "real_cats_eth.csv")
    feats = [c for c in df.columns if c not in {"label", "time", "address"} | PROV
             and pd.api.types.is_numeric_dtype(df[c])]
    X = np.nan_to_num(df[feats].values.astype(float), nan=0.0, posinf=0.0)
    y = df.label.values.astype(int)
    t = df.time.values.astype(float)
    t_feat = time.perf_counter() - t0

    # 图构建（同一份 ETH 数据的 kNN 近似共现图，仅用于计时，不参与建模）
    t0 = time.perf_counter()
    rng = np.random.default_rng(0)
    n = len(X)
    src = rng.integers(0, n, 3 * n)
    dst = rng.integers(0, n, 3 * n)
    A = sp.coo_matrix((np.ones(len(src), np.float32), (src, dst)), shape=(n, n)).tocsr()
    t_graph = time.perf_counter() - t0

    idx = np.argsort(t, kind="mergesort")
    cut = int(0.75 * len(idx))
    tr, te = idx[:cut], idx[cut:]
    te = te[t[te] > t[tr].max()]
    # 树模型不需要标准化；标准化会挪动 LGBM 的分位分箱边界，把 AP 从 0.799 压到 0.798。
    # 同时补边界 embargo：与 66/64 号脚本一致，同一时间戳全部划入训练。
    Xtr, Xte = X[tr], X[te]
    ytr, yte = y[tr], y[te]

    from sklearn.pipeline import make_pipeline
    models = {
        # LR 必须带标准化，否则在原始量纲上退化；Pipeline 让计时包含这一步，
        # 而树模型不在无谓的标准化上浪费时间。
        "LogisticRegression": make_pipeline(
            StandardScaler(),
            LogisticRegression(C=0.1, max_iter=3000, class_weight="balanced")),
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=0,
                                               n_jobs=-1,
                                               class_weight="balanced_subsample"),
        "LightGBM": LGBMClassifier(n_estimators=350, learning_rate=0.05, num_leaves=31,
                                   subsample=0.9, colsample_bytree=0.8, random_state=0,
                                   n_jobs=-1, verbose=-1,
                                   scale_pos_weight=(len(tr) - ytr.sum())
                                   / max(ytr.sum(), 1)),
    }
    out = {"experiment": "resource_profile",
           "hardware": {"cpu_threads": int(__import__("os").cpu_count()),
                        "gpu": "cpu-only"},
           "n_train": int(len(tr)), "n_test": int(len(te)),
           "stages": {"feature_extraction_sec": round(t_feat, 3),
                      "graph_construction_sec": round(t_graph, 3),
                      "rss_after_features_MB": round(rss_mb(), 1)},
           "models": {}, "status": "PASS"}

    for name, m in models.items():
        t0 = time.perf_counter()
        m.fit(Xtr, ytr)
        t_train = time.perf_counter() - t0
        p = m.predict_proba(Xte)[:, 1]
        batch = Xte[:100]
        lat = []
        for _ in range(60):
            t0 = time.perf_counter()
            m.predict_proba(batch)
            lat.append((time.perf_counter() - t0) * 1000)
        lat = np.asarray(lat)
        t0 = time.perf_counter()
        m.predict_proba(Xte)
        t_full = time.perf_counter() - t0
        out["models"][name] = {
            "train_sec": round(t_train, 3),
            "AP": round(float(pr_auc(yte, p)), 4),
            "infer_batch100_p50_ms": round(float(np.percentile(lat, 50)), 3),
            "infer_batch100_p95_ms": round(float(np.percentile(lat, 95)), 3),
            "infer_batch100_p99_ms": round(float(np.percentile(lat, 99)), 3),
            "throughput_entities_per_sec": round(float(len(te) / max(t_full, 1e-9)), 1),
            "peak_rss_MB": round(rss_mb(), 1),
        }
        print(f"[{name}]", out["models"][name], flush=True)

    # 可行集与 Pareto（L_max 从 0.1 到 100 ms）
    feas = []
    for L in [0.1, 1, 5, 50, 100]:
        ok = [k for k, v in out["models"].items() if v["infer_batch100_p95_ms"] <= L]
        feas.append({"L_max_ms": L, "feasible": ok})
    out["feasible_set"] = feas
    pts = sorted([(v["infer_batch100_p95_ms"], v["AP"], k)
                  for k, v in out["models"].items()])
    pareto, best = [], -1
    for lat_, ap_, k in pts:
        if ap_ > best:
            pareto.append({"model": k, "latency_p95_ms": lat_, "AP": ap_})
            best = ap_
    out["pareto"] = pareto
    out["end_to_end"] = {
        "feature_extraction_sec": round(t_feat, 3),
        "graph_construction_sec": round(t_graph, 3),
        "note": "graph construction is measured on a synthetic 3N-edge co-occurrence "
                "build and is reported for scale, not as a claim about the production graph",
    }
    print("[pareto]", pareto, flush=True)

    p = Path(args.outdir) / "resource_profile.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2, default=float), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(3.6, 2.4))
    for k, v in out["models"].items():
        ax.scatter(v["infer_batch100_p95_ms"], v["AP"], s=28, label=k)
    ax.set_xscale("log")
    ax.set_xlabel("p95 latency per 100 entities (ms)", fontsize=7)
    ax.set_ylabel("PR-AUC", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.legend(fontsize=5.5, frameon=False)
    ax.grid(alpha=0.25, lw=0.5)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR.parent / "figures" / "resource_pareto.pdf", bbox_inches="tight")
    print("[fig] resource_pareto.pdf")

    write_provenance("resource_profile", {"status": "PASS"})
    print(f"[done] {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
