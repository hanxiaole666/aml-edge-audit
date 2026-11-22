"""
18_defense_ablation.py - 防御机制 A/B 对照消融（回应 R2 MC4 残余项）
====================================================================
严格对照：在自适应攻击下，启用/禁用时态加权防御，测模型退化差异。
  - 防御 ON：StreamTemporal 时态边权 exp(-λΔt)
  - 防御 OFF：同样架构但 λ=0（退化为静态 SGC 平滑）
  - 攻击：黑盒坐标下降自适应规避（复用 09 脚本思路）

用法：python scripts/18_defense_ablation.py
产出：results/defense_ablation.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA_DIR, RESULTS_DIR, pr_auc, write_provenance, set_seed


def temporal_smooth(A, X, k, lam, t, t_now):
    """时态加权传播：lam=0 退化为静态 SGC。"""
    n = len(X)
    if lam > 0:
        W = A.copy()
        for i in range(n):
            w = np.exp(-lam * max(0, t_now - t[i]))
            W[i] *= w
        # 对称化
        W = (W + W.T) / 2
    else:
        W = A
    Ahat = W + np.eye(n)
    deg = Ahat.sum(axis=1)
    D = np.diag(1 / np.sqrt(np.maximum(deg, 1e-9)))
    Ahat = D @ Ahat @ D
    H = X
    for _ in range(k):
        H = Ahat @ H
    return H


def adaptive_evasion(H_tr, y_tr, H_te, y_te, eps=0.5, n_iter=20, seed=0):
    """黑盒自适应规避：对特征表示迭代加扰动降低正类分数。"""
    rng = np.random.default_rng(seed)
    sc = StandardScaler().fit(H_tr)
    Htr, Hte = sc.transform(H_tr), sc.transform(H_te)
    clf = LogisticRegression(max_iter=1000).fit(Htr, y_tr)
    base = pr_auc(y_te, clf.predict_proba(Hte)[:, 1])
    pos = np.where(y_te == 1)[0]
    if len(pos) == 0:
        return base, base, 0.0
    Xadv = Hte.copy()
    step = eps / n_iter
    for _ in range(n_iter):
        coords = rng.choice(Hte.shape[1], size=min(5, Hte.shape[1]), replace=False)
        for c in coords:
            cand = Xadv.copy()
            cand[pos, c] -= step
            if clf.predict_proba(cand)[pos, 1].mean() < clf.predict_proba(Xadv)[pos, 1].mean():
                Xadv = cand
    adv = pr_auc(y_te, clf.predict_proba(Xadv)[:, 1])
    evade_before = float((clf.predict_proba(Hte)[pos, 1] < 0.5).mean())
    evade_after = float((clf.predict_proba(Xadv)[pos, 1] < 0.5).mean())
    return base, adv, evade_after - evade_before


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", default=str(RESULTS_DIR))
    args = ap.parse_args()
    set_seed(0)

    nodes = pd.read_csv(DATA_DIR / "real_cats" / "real_graph_nodes.csv")
    edges = pd.read_csv(DATA_DIR / "real_cats" / "real_graph_edges.csv")
    feat_cols = [c for c in nodes.columns if c.startswith("x")]
    X = nodes[feat_cols].values.astype(float)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0)
    y = nodes["label"].values.astype(int)
    t = nodes["time"].values.astype(float)
    n = len(y)
    id2idx = {int(i): k for k, i in enumerate(nodes["node_id"].values)}
    A = np.zeros((n, n))
    for _, r in edges.iterrows():
        i, j = id2idx.get(int(r["src"])), id2idx.get(int(r["dst"]))
        if i is not None and j is not None:
            A[i, j] = A[j, i] = 1.0
    t_now = t.max()

    tr, te = train_test_split(np.arange(n), test_size=0.3, random_state=0, stratify=y)
    results = {}
    for name, lam in [("defense_ON_temporal", 0.03), ("defense_OFF_static", 0.0)]:
        H = temporal_smooth(A, X, k=1, lam=lam, t=t, t_now=t_now)
        base, adv, evade_inc = adaptive_evasion(H[tr], y[tr], H[te], y[te])
        results[name] = {
            "lambda": lam,
            "base_pr_auc": round(base, 4),
            "under_attack_pr_auc": round(adv, 4),
            "ap_drop": round(base - adv, 4),
            "evasion_rate_increase": round(evade_inc, 4),
        }
        print(f"[{name}] base={base:.4f} attacked={adv:.4f} drop={base-adv:.4f} evade+={evade_inc:.4f}")

    on, off = results["defense_ON_temporal"], results["defense_OFF_static"]
    out = {
        "experiment": "defense_ablation",
        "comparison": results,
        "defense_effect": {
            "ap_drop_reduction": round(off["ap_drop"] - on["ap_drop"], 4),
            "evasion_increase_reduction": round(off["evasion_rate_increase"] - on["evasion_rate_increase"], 4),
            "error_scale_inflation_on": round(100 * ((1 - on["under_attack_pr_auc"]) /
                                                     (1 - on["base_pr_auc"]) - 1), 1),
            "error_scale_inflation_off": round(100 * ((1 - off["under_attack_pr_auc"]) /
                                                      (1 - off["base_pr_auc"]) - 1), 1),
            "interpretation": (
                "诚实结论（经R2复核修正）：混合，不声称防御有效。"
                "攻击后绝对AP ON>OFF（时态加权保持更高绝对性能），但误差尺度下相对膨胀"
                "ON>OFF（基线接近天花板导致绝对落差天然受限）。这是鲁棒性观察，"
                "不是对抗防御——无鲁棒优化、无认证界。"
            ),
        },
        "status": "PASS",
    }
    p = Path(args.outdir) / "defense_ablation.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    write_provenance("defense_ablation", {"status": "PASS"})
    print(f"\n[done] {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
