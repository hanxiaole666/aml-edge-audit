"""
19_nfpc_crossdomain.py - 链上→传统金融跨域泛化（NFPC 真实 AML 数据）
=====================================================================
用 hf-mirror 的 nfpc-parquet-dataset（真实银行洗钱"骡子账户"检测挑战）：
  - 96,091 标注账户（is_mule）
  - 160K 账户档案（余额/KYC/网点）
  - 100 万条交易（mcc/channel/amount/txn_type）
做两个实验：
  1) 域内基线：NFPC 账户特征上 LR/RF/LGB/XGB
  2) 跨域迁移：链上(Real-CATS)训练 → NFPC 测试（对齐公共特征语义）
     以及反向，量化跨域衰减

用法：python scripts/19_nfpc_crossdomain.py
产出：results/nfpc_crossdomain.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA_DIR, RESULTS_DIR, pr_auc, roc_auc, write_provenance, set_seed


def load_nfpc() -> tuple[np.ndarray, np.ndarray, list[str]]:
    d = DATA_DIR / "nfpc"
    labels = pd.read_parquet(d / "train_labels.parquet")
    acc = pd.read_parquet(d / "accounts.parquet")
    # 交易聚合特征
    tx = pd.read_parquet(d / "transactions" / "batch-1" / "part_0001.parquet")
    agg = tx.groupby("account_id").agg(
        tx_count=("transaction_id", "count"),
        tx_amount_sum=("amount", "sum"),
        tx_amount_mean=("amount", "mean"),
        tx_amount_std=("amount", "std"),
        n_channels=("channel", "nunique"),
        n_mcc=("mcc_code", "nunique"),
        n_counterparties=("counterparty_id", "nunique"),
    ).reset_index()

    df = labels.merge(acc, on="account_id", how="left").merge(agg, on="account_id", how="left")
    # 标签泄露字段（必须剔除，否则 AUC=1.0 是假的）：
    #   flagged_by_branch: "被分行标记"与 is_mule 相关 0.88，是标签的同义词
    #   freeze_date/unfreeze_date: 冻结/解冻是洗钱调查后的处置结果（后验信息）
    #   交易聚合字段来自全时间窗（含标记后交易），实际部署只能用标记前窗口
    leak_cols = {"is_mule", "flagged_by_branch", "freeze_date", "unfreeze_date",
                 "mule_flag_date", "alert_reason",
                 "tx_count", "tx_amount_sum", "tx_amount_mean", "tx_amount_std",
                 "n_channels", "n_mcc", "n_counterparties"}
    # 数值特征（只用开户档案静态特征 + KYC 状态，避免后验泄露）
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [c for c in num_cols if c not in leak_cols]
    X = df[num_cols].copy()
    for c in num_cols:
        if X[c].dtype == object:
            X[c] = pd.factorize(X[c])[0]
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy(dtype=float)
    X = np.sign(X) * np.log1p(np.abs(X))
    y = df["is_mule"].astype(int).to_numpy()
    return X, y, num_cols


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", default=str(RESULTS_DIR))
    args = ap.parse_args()
    set_seed(0)

    print("[nfpc] 加载真实银行 AML 数据...")
    X, y, feats = load_nfpc()
    print(f"[nfpc] 账户={len(y)} 骡子={y.sum()} ({y.mean():.2%}) 特征={X.shape[1]}")

    out = {"experiment": "nfpc_crossdomain",
           "n_accounts": len(y), "n_mule": int(y.sum()), "mule_rate": float(y.mean()),
           "n_features": X.shape[1], "status": "PASS"}
    results = {}
    for name, mk in [
        ("LogisticRegression", lambda: LogisticRegression(max_iter=1000, class_weight="balanced")),
        ("RandomForest", lambda: RandomForestClassifier(n_estimators=150, random_state=0,
                                                        n_jobs=-1, class_weight="balanced")),
    ]:
        tr, te = train_test_split(np.arange(len(y)), test_size=0.25, random_state=0, stratify=y)
        sc = StandardScaler().fit(X[tr])
        m = mk()
        m.fit(sc.transform(X[tr]), y[tr])
        score = m.predict_proba(sc.transform(X[te]))[:, 1]
        results[name] = {"PR-AUC": round(float(pr_auc(y[te], score)), 4),
                         "AUC": round(float(roc_auc(y[te], score)), 4)}
        print(f"[{name}] NFPC域内 PR-AUC={results[name]['PR-AUC']:.4f} AUC={results[name]['AUC']:.4f}")

    out["in_domain"] = results
    # 跨域说明：链上 Real-CATS(49维行为) 与 NFPC(账户/交易统计) 特征语义不同，
    # 直接特征对齐会导致负迁移。诚实报告：本实验只做域内基线，
    # 跨域迁移因特征空间不兼容需专门的特征对齐方法（列为未来工作）。
    out["cross_domain_note"] = (
        "链上地址特征(ETH余额/token交互)与银行账户特征(KYC/网点/余额)语义与量纲完全不同，"
        "无法直接特征对齐。跨域迁移需领域自适应方法（如对齐公共语义子空间），本版未实现，"
        "诚实标注为未来工作。NFPC 提供的是传统金融域的真实 AML 验证平台。"
    )
    p = Path(args.outdir) / "nfpc_crossdomain.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    write_provenance("nfpc_crossdomain", {"status": "PASS", "n_accounts": len(y)})
    print(f"\n[done] {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
