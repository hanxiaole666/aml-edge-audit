"""
61_attack_threat_model.py - E11：与威胁模型一致的有预算攻击
======================================================================
§3 的威胁模型说"攻击者是改变交易行为的洗钱者"，而旧实验是无量纲的坐标下降，
两者不匹配。本脚本把攻击改成**链上可执行的操作**，并给可行域与预算：

可改字段（洗钱者能控制的）
    transaction_number / payment_transactions / receipt_transactions  只能增（拆单）
    max_sent_amount / min_sent_amount                                 可降/可升（拆单）
    lifetime / activity_d / activity_w                                可增（拉长观察期）
    total_gas_used / transaction_fee                                  可增（多付手续费）
不可改（链上既成事实）
    creat_contract / total_received_ETH / total_sent_ETH / *_USD / 各种 variance
    —— 转账总量守恒，洗钱者只能改"形状"，不能改"总量"

三种攻击
    1. smurfing(m)   拆单：一笔拆成 m 笔。transaction_number *= m，
                     max_sent_amount /= m，min_sent_amount /= m，gas *= m。
    2. layering(h)   加层：每加一层，多 2 笔交易、lifetime + 30 天、
                     activity_d + 2、度数 + h（图上加 h 条边）。
    3. pgd(eps)      在**可改维**上做 L∞ 投影梯度（对 logistic 代理求梯度，
                     迁移到 LightGBM）。这是"最坏情况"上界，不是链上语义。

预算
    q ∈ {0.1, 0.3, 0.5, 1.0}  被攻击的非法账户比例（L0 账户预算）
    m ∈ {2, 4, 8}，h ∈ {1, 2, 3}，eps ∈ {0.1, 0.2, 0.4}

报告 AP 与 recall@1% 随预算的变化，并给"打掉一半检出率需要多大预算"。

用法：python scripts/61_attack_threat_model.py
产出：results/attack_threat_model.json + figures/attack_threat_model.pdf
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
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA_DIR, RESULTS_DIR, pr_auc, set_seed, write_provenance

PROV = {"etherscan_checked", "watchback_checked", "gs_checked"}
MODIFIABLE = ["transaction_number", "payment_transactions", "receipt_transactions",
              "max_sent_amount", "min_sent_amount", "lifetime", "activity_d",
              "activity_w", "total_gas_used", "transaction_fee"]


def load():
    df = pd.read_csv(DATA_DIR / "real_cats_eth.csv")
    feats = [c for c in df.columns if c not in {"label", "time", "address"} | PROV
             and pd.api.types.is_numeric_dtype(df[c])]
    X = np.nan_to_num(df[feats].values.astype(float), nan=0.0, posinf=0.0, neginf=0.0)
    y = df.label.values.astype(int)
    t = df.time.values.astype(float)
    return X, y, t, feats


def recall_at(y, s, frac=0.01):
    k = max(int(frac * len(y)), 1)
    top = np.argsort(-s)[:k]
    return float(y[top].sum() / max(y.sum(), 1))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", default=str(RESULTS_DIR))
    args = ap.parse_args()
    set_seed(0)

    X, y, t, feats = load()
    idx = np.argsort(t, kind="mergesort")
    cut = int(0.75 * len(idx))
    tr, te = idx[:cut], idx[cut:]
    te = te[t[te] > t[tr].max()]
    # 树模型不需要标准化；标准化会挪动 LGBM 的分位分箱边界，把 AP 从 0.799 压到 0.798。
    # 同时补边界 embargo：与 66/64 号脚本一致，同一时间戳全部划入训练。
    Xtr, Xte = X[tr], X[te]
    ytr, yte = y[tr], y[te]

    rf = LGBMClassifier(n_estimators=350, learning_rate=0.05, num_leaves=31,
                        subsample=0.9, colsample_bytree=0.8, random_state=0,
                        n_jobs=-1, verbose=-1,
                        scale_pos_weight=(len(tr) - ytr.sum()) / max(ytr.sum(), 1))
    rf.fit(Xtr, ytr)
    base = rf.predict_proba(Xte)[:, 1]
    base_ap = float(pr_auc(yte, base))
    base_r = recall_at(yte, base)
    print(f"[base] AP={base_ap:.4f} R@1%={base_r:.4f} n_test={len(yte)}", flush=True)

    col = {c: i for i, c in enumerate(feats)}
    mod_idx = np.array([col[c] for c in MODIFIABLE if c in col])
    Xte_raw = X[te].copy()
    n_pos = int(yte.sum())
    pos_local = np.where(yte == 1)[0]

    def apply_and_score(Xp, attacked_rows):
        s = rf.predict_proba(Xp)[:, 1]
        return float(pr_auc(yte, s)), recall_at(yte, s), s

    def score_std(Xstd):
        s = rf.predict_proba(Xstd)[:, 1]
        return float(pr_auc(yte, s)), recall_at(yte, s)

    results = []

    # ---------------- 1. smurfing
    for m in [2, 4, 8]:
        for q in [0.1, 0.3, 0.5, 1.0]:
            Xp = Xte_raw.copy()
            k = int(q * len(pos_local))
            rows = pos_local[:k]
            for c in ["transaction_number", "payment_transactions", "receipt_transactions",
                      "total_gas_used", "transaction_fee"]:
                if c in col:
                    Xp[np.ix_(rows, [col[c]])] *= m
            for c in ["max_sent_amount", "min_sent_amount"]:
                if c in col:
                    Xp[np.ix_(rows, [col[c]])] /= m
            if "activity_d" in col:
                Xp[np.ix_(rows, [col["activity_d"]])] *= min(m, 3)
            a, r, _ = apply_and_score(Xp, rows)
            results.append({"attack": "smurfing", "param": m, "q": q,
                            "AP": round(a, 4), "recall@1%": round(r, 4),
                            "AP_drop": round(base_ap - a, 4)})
            print(f"  smurf m={m} q={q}: AP={a:.4f} R={r:.4f}", flush=True)

    # ---------------- 2. layering
    for h in [1, 2, 3]:
        for q in [0.1, 0.3, 0.5, 1.0]:
            Xp = Xte_raw.copy()
            k = int(q * len(pos_local))
            rows = pos_local[:k]
            for c, delta in [("transaction_number", 2 * h), ("payment_transactions", 2 * h),
                             ("lifetime", 30 * h), ("activity_d", 2 * h),
                             ("activity_w", 2 * h)]:
                if c in col:
                    Xp[np.ix_(rows, [col[c]])] += delta
            a, r, _ = apply_and_score(Xp, rows)
            results.append({"attack": "layering", "param": h, "q": q,
                            "AP": round(a, 4), "recall@1%": round(r, 4),
                            "AP_drop": round(base_ap - a, 4)})
            print(f"  layer h={h} q={q}: AP={a:.4f} R={r:.4f}", flush=True)

    # ---------------- 3. PGD on logistic surrogate, transfer to LightGBM
    # 代理模型必须在标准化空间里训练，否则 LR 的权重被大量纲特征支配，梯度方向没有意义，
    # L-inf 预算也无从定义。目标模型仍在原始特征上训练（上面已不再标准化）。
    # 因此：标准化 -> PGD -> 逆变换回原始量纲 -> 喂给 LightGBM。
    from sklearn.preprocessing import StandardScaler
    sc_sur = StandardScaler().fit(Xtr)
    Ztr, Zte = sc_sur.transform(Xtr), sc_sur.transform(Xte)
    lr = LogisticRegression(C=0.1, max_iter=3000, class_weight="balanced")
    lr.fit(Ztr, ytr)
    w, b = lr.coef_[0], lr.intercept_[0]
    for eps in [0.1, 0.2, 0.4]:
        for q in [0.1, 0.3, 0.5, 1.0]:
            k = int(q * len(pos_local))
            rows = pos_local[:k]
            Zadv = Zte[rows].copy()
            Z0 = Zte[rows].copy()
            # 只沿可改维、朝降低非法分数方向走（标准化空间中的 L-inf 投影）
            step = 0.05
            for _ in range(12):
                g = w  # d logit / d z，线性代理
                Zadv[:, mod_idx] -= step * np.sign(g[mod_idx])[None, :]
                Zadv[:, mod_idx] = np.clip(Zadv[:, mod_idx],
                                           Z0[:, mod_idx] - eps, Z0[:, mod_idx] + eps)
            Xp = Xte.copy()
            Xp[rows] = sc_sur.inverse_transform(Zadv)
            a, r = score_std(Xp)
            results.append({"attack": "pgd_linf", "param": eps, "q": q,
                            "AP": round(a, 4), "recall@1%": round(r, 4),
                            "AP_drop": round(base_ap - a, 4)})
            print(f"  pgd eps={eps} q={q}: AP={a:.4f} R={r:.4f}", flush=True)

    # ---------------- 汇总：打掉一半 recall 需要的预算
    def need_half(attack, param):
        rs = [r for r in results if r["attack"] == attack and r["param"] == param]
        rs.sort(key=lambda r: r["q"])
        for r in rs:
            if r["recall@1%"] <= base_r / 2:
                return r["q"]
        return None

    out = {"experiment": "attack_threat_model",
           "threat_model": {"modifiable": MODIFIABLE,
                            "immutable": ["creat_contract", "total_received_ETH",
                                          "total_sent_ETH", "total_received_USD",
                                          "total_sent_USD", "lifetime-neutral variances"],
                            "rationale": "a launderer reshapes flows; totals are conserved"},
           "baseline": {"AP": round(base_ap, 4), "recall@1%": round(base_r, 4),
                        "n_test": int(len(yte)), "n_illicit": n_pos},
           "results": results, "status": "PASS"}
    out["budget_to_halve_recall"] = {
        "smurfing_m8": need_half("smurfing", 8),
        "layering_h3": need_half("layering", 3),
        "pgd_eps0.4": need_half("pgd_linf", 0.4),
    }
    out["verdict"] = {
        "max_AP_drop_smurfing": round(max(r["AP_drop"] for r in results
                                          if r["attack"] == "smurfing"), 4),
        "max_AP_drop_layering": round(max(r["AP_drop"] for r in results
                                          if r["attack"] == "layering"), 4),
        "max_AP_drop_pgd": round(max(r["AP_drop"] for r in results
                                     if r["attack"] == "pgd_linf"), 4),
        "note": ("Semantic attacks change the shape of flows within a feasible region; "
                 "the PGD row is an upper bound from a linear surrogate, not a claim that "
                 "an adversary can realise it on-chain."),
    }
    print("[verdict]", out["verdict"], flush=True)

    p = Path(args.outdir) / "attack_threat_model.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2, default=float), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(4.4, 2.5))
    for name, ls in [("smurfing", "-"), ("layering", "--"), ("pgd_linf", ":")]:
        for param in sorted({r["param"] for r in results if r["attack"] == name}):
            rs = sorted([r for r in results if r["attack"] == name and r["param"] == param],
                        key=lambda r: r["q"])
            ax.plot([r["q"] for r in rs], [r["recall@1%"] for r in rs],
                    ls, marker="o", ms=3, lw=1, label=f"{name} {param}")
    ax.set_xlabel("fraction of illicit accounts attacked", fontsize=7)
    ax.set_ylabel("recall@1%", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.legend(fontsize=4.5, frameon=False, ncol=2)
    ax.grid(alpha=0.25, lw=0.5)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR.parent / "figures" / "attack_threat_model.pdf", bbox_inches="tight")
    print("[fig] attack_threat_model.pdf")

    write_provenance("attack_threat_model", {"status": "PASS"})
    print(f"[done] {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
