"""
10_cross_chain.py - 跨链泛化实验（BTC <-> ETH）
===============================================
用 Real-CATS 的比特币数据（CB 犯罪 + BB 良性）与以太坊数据（CE+BE），
做跨链迁移：在链 A 训练，在链 B 测试，量化泛化衰减。

用法：
  python scripts/10_cross_chain.py
产出：results/cross_chain.json
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA_DIR, RESULTS_DIR, pr_auc, roc_auc, write_provenance, set_seed


def load_chain(chain: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """加载一条链的数据（犯罪+良性合并），返回 X, y, feat_names。"""
    base = DATA_DIR / "real_cats"
    if chain == "ETH":
        pos_f, neg_f = "CE.tsv", "BE.tsv"
    else:
        pos_f, neg_f = "CB.tsv", "BB.tsv"
    frames = []
    for fn, label in [(pos_f, 1), (neg_f, 0)]:
        p = base / fn
        if not p.exists():
            raise FileNotFoundError(f"{fn} 不存在，请先下载")
        df = pd.read_csv(p, sep="\t")
        df["label"] = label
        frames.append(df)
    data = pd.concat(frames, ignore_index=True)
    # 标签来源字段是标签派生元数据，跨链对齐前先剔除（见 54 号审计）
    exclude = {"address", "label", "first_time", "last_time", "time",
               "etherscan_checked", "watchback_checked", "gs_checked"}
    feats = [c for c in data.columns if c not in exclude and
             pd.api.types.is_numeric_dtype(data[c])]
    X = data[feats].values.astype(float)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    X = np.sign(X) * np.log1p(np.abs(X))
    y = data["label"].values.astype(int)
    return X, y, feats


def align_features(X_a, fa, X_b, fb) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """对齐两条链的公共特征列。"""
    common = [c for c in fa if c in fb]
    ia = [fa.index(c) for c in common]
    ib = [fb.index(c) for c in common]
    return X_a[:, ia], X_b[:, ib], common


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", default=str(RESULTS_DIR))
    args = ap.parse_args()
    set_seed(0)

    print("[data] 加载 BTC 与 ETH 真实数据...")
    X_eth, y_eth, f_eth = load_chain("ETH")
    X_btc, y_btc, f_btc = load_chain("BTC")
    print(f"  ETH: {X_eth.shape}, BTC: {X_btc.shape}")
    Xe, Xb, common = align_features(X_eth, f_eth, X_btc, f_btc)
    print(f"  公共特征: {len(common)}")

    results = {}
    for name, mk in [("LogisticRegression", lambda: LogisticRegression(max_iter=1000)),
                     ("RandomForest", lambda: RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1))]:
        # 同链基线（各自内部随机分裂）
        from sklearn.model_selection import train_test_split
        _, Xe_te, _, ye_te = train_test_split(Xe, y_eth, test_size=0.3, random_state=0, stratify=y_eth)
        _, Xb_te, _, yb_te = train_test_split(Xb, y_btc, test_size=0.3, random_state=0, stratify=y_btc)

        # ETH 内部训练/测试
        m = mk(); Xe_tr, _, ye_tr, _ = train_test_split(Xe, y_eth, test_size=0.3, random_state=0, stratify=y_eth)
        m.fit(Xe_tr, ye_tr)
        eth_internal = pr_auc(ye_te, m.predict_proba(Xe_te)[:, 1])

        # ETH -> BTC
        m2 = mk(); m2.fit(Xe, y_eth)
        eth_to_btc = pr_auc(y_btc, m2.predict_proba(Xb)[:, 1])

        # BTC 内部
        m3 = mk(); Xb_tr, _, yb_tr, _ = train_test_split(Xb, y_btc, test_size=0.3, random_state=0, stratify=y_btc)
        m3.fit(Xb_tr, yb_tr)
        btc_internal = pr_auc(yb_te, m3.predict_proba(Xb_te)[:, 1])

        # BTC -> ETH
        m4 = mk(); m4.fit(Xb, y_btc)
        btc_to_eth = pr_auc(y_eth, m4.predict_proba(Xe)[:, 1])

        results[name] = {
            "ETH_internal": eth_internal, "BTC_internal": btc_internal,
            "ETH_to_BTC": eth_to_btc, "BTC_to_ETH": btc_to_eth,
            "ETH_to_BTC_drop": eth_internal - eth_to_btc,
            "BTC_to_ETH_drop": btc_internal - btc_to_eth,
        }
        print(f"[{name}] ETH internal={eth_internal:.4f} ETH→BTC={eth_to_btc:.4f} (drop {eth_internal-eth_to_btc:.4f})")
        print(f"[{name}] BTC internal={btc_internal:.4f} BTC→ETH={btc_to_eth:.4f} (drop {btc_internal-btc_to_eth:.4f})")

    out = {"experiment": "cross_chain", "n_common_features": len(common), "models": results, "status": "PASS"}
    out_path = Path(args.outdir) / "cross_chain.json"
    import json
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    write_provenance("cross_chain", out)
    print(f"\n[done] {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
