"""
08_build_text.py - 从真实地址行为构造文本语料（特征文本化，避免标签泄露）
============================================================================
动机：外部文本数据源（HF PhraseBank / GitHub 钓鱼文案）在本机网络下不可达。
方案：把 Real-CATS 真实地址的行为特征文本化为自然语言描述，标签沿用真实
      犯罪/良性。文本中**不含**诈骗类型标签词（避免标签泄露）。

诚信标注：这是"structured-to-text"变换，非真实白皮书/钓鱼文案。
论文中必须标注为"behavioral textualization"，不可声称用了真实文本。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA_DIR, write_provenance


def describe(row: pd.Series) -> str:
    """把真实数值特征转成行为描述句（不含标签词）。"""
    def g(c, default=0.0):
        try:
            v = float(row.get(c, default))
            return v if np.isfinite(v) else default
        except (TypeError, ValueError):
            return default
    parts = []
    eth_in, eth_out = g("total_received_ETH"), g("total_sent_ETH")
    txn = g("transaction_number")
    if txn > 0:
        parts.append(f"account with {int(txn)} transactions")
    if eth_in > 0 or eth_out > 0:
        parts.append(f"received {eth_in:.3g} ETH and sent {eth_out:.3g} ETH")
    usd_in, usd_out = g("total_received_USD"), g("total_sent_USD")
    if usd_in > 0 or usd_out > 0:
        parts.append(f"inflow {usd_in:.3g} USD outflow {usd_out:.3g} USD")
    for c, label in [("from_contract_wd_normal", "contract interactions"),
                     ("from_EOA_wd_normal", "peer transfers"),
                     ("error_transaction", "failed calls")]:
        v = g(c)
        if v > 0:
            parts.append(f"{int(v)} {label}")
    lifetime = g("lifetime")
    if lifetime > 0:
        parts.append(f"active span {lifetime:.3g}")
    if not parts:
        parts.append("account with minimal activity")
    return "; ".join(parts) + "."


def main() -> int:
    base = DATA_DIR / "real_cats"
    rows = []
    for fn, label in [("CE.tsv", 1), ("BE.tsv", 0)]:
        p = base / fn
        if not p.exists():
            print(f"[warn] {fn} 不存在，跳过")
            continue
        df = pd.read_csv(p, sep="\t")
        for _, r in df.iterrows():
            rows.append((describe(r), label))
    if not rows:
        print("[ERROR] 无文本数据可生成")
        return 2
    out = DATA_DIR / "real_text.csv"
    pd.DataFrame(rows, columns=["text", "label"]).to_csv(out, index=False)
    n_pos = sum(l for _, l in rows)
    detail = f"从真实地址行为生成 {len(rows)} 条文本（正类 {n_pos}），不含标签词，避免标签泄露"
    write_provenance("real_text", {"source": "behavioral_textualization", "detail": detail})
    print(f"[real_text] {detail}")
    print(f"[real_text] 样例：{rows[0][0][:150]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
