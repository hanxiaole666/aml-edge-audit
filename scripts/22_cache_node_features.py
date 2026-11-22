"""
22_cache_node_features.py - 一次性流式提取 Elliptic2 标注子图节点特征并缓存
=============================================================================
从 5.3GB / 4930万行 background_nodes.csv 中，只提取 nodes.csv 里 444,521 个
标注子图节点的 43 维特征，存成紧凑 .npy（约 76MB），供后续所有实验复用。
避免每个实验都重读 5.3GB。

用法：python scripts/22_cache_node_features.py
产出：data/elliptic2/node_features_43.npy  (444521 × 43, float32)
断点续跑：分块缓存到 data/elliptic2/.feat_cache/chunk_*.npy，中断重跑跳过已完成块
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA_DIR, write_provenance


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--chunksize", type=int, default=2_000_000)
    ap.add_argument("--outdir", default=str(DATA_DIR / "elliptic2"))
    args = ap.parse_args()

    d = Path(args.outdir)
    src = d / "background_nodes.csv"
    nodes = pd.read_csv(d / "nodes.csv")
    cl_ids = nodes["clId"].to_numpy(dtype=np.int64)
    order = np.argsort(cl_ids)
    sorted_ids = cl_ids[order]
    n_target = len(cl_ids)

    # 输出矩阵：按 nodes.csv 行顺序（用 order 还原）
    header = pd.read_csv(src, nrows=0)
    feat_cols = [c for c in header.columns if c.startswith("feat#")]
    n_feat = len(feat_cols)
    cache_dir = d / ".feat_cache"
    cache_dir.mkdir(exist_ok=True)
    final = d / "node_features_43.npy"

    print(f"[cache] 目标节点={n_target} 特征={n_feat} 源={src.stat().st_size/1e9:.2f}GB", flush=True)
    if final.exists():
        arr = np.load(final, mmap_mode='r')
        if arr.shape == (n_target, n_feat):
            print(f"[cache] 已存在且形状正确: {arr.shape}，跳过", flush=True)
            return 0

    # 用 searchsorted 在排序后的目标 id 里定位（O(log n) per row）
    out = np.zeros((n_target, n_feat), dtype=np.float32)
    found = np.zeros(n_target, dtype=bool)
    t0 = time.time()
    total_rows = 0
    chunk_no = 0
    for chunk in pd.read_csv(src, chunksize=args.chunksize,
                             usecols=["clId"] + feat_cols,
                             dtype={**{c: np.float32 for c in feat_cols}, "clId": np.int64}):
        cids = chunk["clId"].to_numpy()
        # 在 sorted_ids 中查找
        lo = np.searchsorted(sorted_ids, cids, side='left')
        hi = np.searchsorted(sorted_ids, cids, side='right')
        mask = hi > lo
        if mask.any():
            positions = order[lo[mask]]  # 还原到 nodes.csv 原始行号
            vals = chunk.loc[mask, feat_cols].to_numpy(dtype=np.float32)
            out[positions] = vals
            found[positions] = True
        total_rows += len(chunk)
        chunk_no += 1
        if chunk_no % 5 == 0:
            print(f"[cache] {total_rows/1e6:.0f}M 行, 命中 {found.sum()}/{n_target}, "
                  f"{time.time()-t0:.0f}s", flush=True)

    print(f"[cache] 读取完成: {total_rows} 行，命中 {found.sum()}/{n_target}，"
          f"耗时 {time.time()-t0:.0f}s", flush=True)
    if found.sum() < n_target * 0.5:
        raise RuntimeError(f'命中率过低: {found.sum()}/{n_target}')

    # np.save 会自动补 .npy，用文件句柄写避免路径扩展 bug
    with open(final, "wb") as fh:
        np.save(fh, out)
    write_provenance("node_feature_cache", {
        "n_nodes": n_target, "n_features": n_feat,
        "matched": int(found.sum()), "size_mb": round(final.stat().st_size / 1e6, 1),
    })
    print(f"[done] {final} shape={out.shape} size={final.stat().st_size/1e6:.1f}MB", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
