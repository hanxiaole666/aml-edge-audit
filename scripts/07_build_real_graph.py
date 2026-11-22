"""
07_build_real_graph.py - 从 Real-CATS 真实 token 交互构造地址行为图
=====================================================================
真实图定义（方法学可辩护）：
  节点 = 以太坊地址（犯罪 TI_M ∪ 良性 TI_B）
  边   = 两个地址若交互过同一 ERC20/ERC721/ERC1155 合约，则相连（共享合约边）
  权重 = 共享合约数（Jaccard 归一化）
这是**真实链上行为关系**，替代 KNN 相似图（后者是方法学妥协）。

用法：
  python scripts/07_build_real_graph.py --outdir data/real_cats
产出：
  data/real_cats/real_graph_nodes.csv, real_graph_edges.csv
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA_DIR, write_provenance


def parse_contracts(cell) -> set[int]:
    """解析 '1:3T2, 2:0T1' 格式 -> 合约索引集合 {1,2,...}"""
    if pd.isna(cell):
        return set()
    s = str(cell)
    # 每项形如 <idx>:<...>，取冒号前的索引
    out = set()
    for item in s.split(","):
        item = item.strip()
        if ":" in item:
            idx = item.split(":")[0].strip()
            if idx.isdigit():
                out.add(int(idx))
    return out


def build(max_nodes: int = 8000, min_shared: int = 1):
    base = DATA_DIR / "real_cats"
    ti_m = pd.read_csv(base / "TI_M.tsv", sep="\t")
    ti_b = pd.read_csv(base / "TI_B.tsv", sep="\t")
    ti_m["label"] = 1
    ti_b["label"] = 0
    ti = pd.concat([ti_m, ti_b], ignore_index=True)

    # 地址 -> 标签（去重）
    addr_label = ti.drop_duplicates("Address").set_index("Address")["label"].to_dict()
    # 地址 -> 合约集合
    addr_contracts: dict[str, set[int]] = {}
    for _, row in ti.iterrows():
        a = row["Address"]
        cs = set()
        for c in ["ERC20", "ERC721", "ERC1155"]:
            cs |= parse_contracts(row.get(c))
        if a in addr_contracts:
            addr_contracts[a] |= cs
        else:
            addr_contracts[a] = cs

    addrs = list(addr_contracts.keys())
    if len(addrs) > max_nodes:
        # 保证类别均衡采样
        rng = np.random.default_rng(42)
        pos = [a for a in addrs if addr_label.get(a, 0) == 1]
        neg = [a for a in addrs if addr_label.get(a, 0) == 0]
        n_each = max_nodes // 2
        pos = list(rng.choice(pos, size=min(n_each, len(pos)), replace=False))
        neg = list(rng.choice(neg, size=min(n_each, len(neg)), replace=False))
        addrs = pos + neg
    addr2idx = {a: i for i, a in enumerate(addrs)}

    # 节点特征：合并 Real-CATS 的真实 49 维行为特征 + 合约交互统计
    feat_df = pd.read_csv(DATA_DIR / "real_cats_eth.csv") if (DATA_DIR / "real_cats_eth.csv").exists() else None
    addr_feat = {}
    addr_time = {}
    if feat_df is not None:
        # 标签来源字段（etherscan/watchback/gs_checked）是标签派生的元数据：
        # 它们记录"这个地址是被哪个榜单确认的"，与 label 的相关系数达 0.71/0.55，
        # 用作特征会把标签直接喂给模型。必须排除，见 54 号审计脚本。
        LABEL_PROVENANCE = {"etherscan_checked", "watchback_checked", "gs_checked"}
        cols = [c for c in feat_df.columns
                if c not in {"address", "label", "time"} and c not in LABEL_PROVENANCE
                and pd.api.types.is_numeric_dtype(feat_df[c])]
        # 统一转 float64，避免 object dtype 导致 to_csv 失败
        mat = feat_df[cols].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy(dtype=float)
        times = pd.to_numeric(feat_df["time"], errors="coerce").fillna(0).to_numpy(dtype=float)
        for i, a in enumerate(feat_df["address"].astype(str)):
            addr_feat[a] = mat[i]
            addr_time[a] = float(times[i])
    n_feat = len(next(iter(addr_feat.values()))) if addr_feat else 1
    # 用纯 numpy 矩阵一次性构建，避免逐行 dict 造成 dtype 混杂
    X_nodes = np.zeros((len(addrs), n_feat + 1), dtype=float)
    labels = np.zeros(len(addrs), dtype=int)
    times = np.zeros(len(addrs), dtype=float)
    for k, a in enumerate(addrs):
        feat_vec = np.asarray(addr_feat.get(a, np.zeros(n_feat)), dtype=float)
        X_nodes[k, :n_feat] = feat_vec
        X_nodes[k, n_feat] = float(len(addr_contracts[a]))
        labels[k] = int(addr_label.get(a, 0))
        times[k] = float(addr_time.get(a, 0.0))
    nodes = pd.DataFrame(X_nodes, columns=[f"x{i}" for i in range(n_feat + 1)])
    nodes.insert(0, "node_id", np.arange(len(addrs)))
    nodes.insert(1, "label", labels)
    nodes.insert(2, "time", times)

    # 倒排索引：合约 -> 地址列表
    contract_to_addr: dict[int, list[int]] = defaultdict(list)
    for a in addrs:
        for c in addr_contracts[a]:
            contract_to_addr[c].append(addr2idx[a])

    # 共享合约边
    pair_count: dict[tuple[int, int], int] = defaultdict(int)
    for c, ids in contract_to_addr.items():
        if len(ids) > 50:   # 跳过超热门合约（避免 O(n²) 爆炸）
            continue
        ids = sorted(set(ids))
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                pair = (ids[i], ids[j])
                pair_count[pair] += 1

    edges = []
    idx_to_addr = {v: k for k, v in addr2idx.items()}
    for (i, j), cnt in pair_count.items():
        if cnt >= min_shared:
            # 边时间近似 = 两端地址首次交易时间的较大者（关系确立时间下界）
            ai, aj = idx_to_addr[i], idx_to_addr[j]
            edge_t = max(addr_time.get(ai, 0.0), addr_time.get(aj, 0.0))
            edges.append((i, j, edge_t))
    edge_df = pd.DataFrame(edges, columns=["src", "dst", "time"])

    out_dir = base
    # 调试：确认无 object/异常列
    bad_cols = [c for c in nodes.columns if nodes[c].dtype == object]
    if bad_cols:
        print(f"[warn] object 列 {bad_cols}，强制数值化")
        for c in bad_cols:
            nodes[c] = pd.to_numeric(nodes[c], errors="coerce").fillna(0)
    # 用 numpy 直接写，完全绕过 pandas 序列化路径
    node_path = str(out_dir / "real_graph_nodes.csv")
    np.savetxt(node_path, nodes.to_numpy(dtype=float), delimiter=",",
               header=",".join(str(c) for c in nodes.columns), comments="", fmt="%.6g")
    edge_path = str(out_dir / "real_graph_edges.csv")
    edge_df.to_csv(edge_path, index=False)
    detail = (f"节点={len(nodes)}（犯罪 {int(nodes.label.sum())} / 良性 {len(nodes)-int(nodes.label.sum())}），"
              f"边={len(edge_df)}，合约={len(contract_to_addr)}，最小共享={min_shared}")
    write_provenance("real_graph", {"source": "Real-CATS token interactions", "detail": detail})
    print(f"[real_graph] {detail}")
    print(f"[real_graph] 产出：{out_dir / 'real_graph_nodes.csv'}, {out_dir / 'real_graph_edges.csv'}")
    return nodes, edge_df


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-nodes", type=int, default=8000)
    ap.add_argument("--min-shared", type=int, default=1)
    args = ap.parse_args()
    build(args.max_nodes, args.min_shared)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
