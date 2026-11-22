"""
15_elliptic2_extract.py - 从 Kaggle Elliptic2 的 25.8GB ZIP 中只提取小的标注子图文件
====================================================================================
核心技术：HTTP Range 断点下载 + ZIP local file header 解析
  - Elliptic2 完整 ZIP 25.8GB（background_edges 24GB + background_nodes 1.3GB 是大头）
  - 但标注子图只需 edges.csv(5.1MB) + nodes.csv(3.6MB) + connected_components.csv(0.3MB)
  - 用 Range 只下载这 3 个文件对应的字节区间，绕开 25GB 背景图

已实测：Kaggle 下载 URL 免登录（302 直跳 GCS 签名 URL）、支持 Range（HTTP 206）、
        accept-ranges: bytes。ZIP64 中央目录已解析出各文件精确偏移。

用法：
  python scripts/15_elliptic2_extract.py              # 下载并提取小文件
  python scripts/15_elliptic2_extract.py --force      # 强制重下
断点续跑：每个文件单独落盘 .part，中断后重跑继续（Range 从已下载长度继续）
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
import time
import zipfile
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA_DIR, RESULTS_DIR, write_provenance

URL = 'https://www.kaggle.com/api/v1/datasets/download/ellipticco/elliptic2-data-set'
TOTAL = 25805701153
# 从 ZIP64 中央目录解析出的精确偏移（2026-11-21 实测）
FILES = {
    'connected_components.csv': {'local_off': 25796766861, 'comp': 295701, 'uncomp': 1610070},
    'edges.csv': {'local_off': 25797062636, 'comp': 5086842, 'uncomp': 11356690},
    'nodes.csv': {'local_off': 25802149537, 'comp': 3551015, 'uncomp': 7470115},
}
HEADERS = {'User-Agent': 'Mozilla/5.0'}


def parse_local_header(offset: int) -> dict:
    """读 ZIP local file header（30 字节）拿文件名长度与 extra 长度。"""
    r = requests.get(URL, headers={**HEADERS, 'Range': f'bytes={offset}-{offset+30}'}, timeout=60)
    r.raise_for_status()
    h = r.content[:30]
    if h[:4] != b'PK\x03\x04':
        raise RuntimeError(f'local header 签名错误: {h[:4]} at {offset}')
    nlen = struct.unpack('<H', h[26:28])[0]
    elen = struct.unpack('<H', h[28:30])[0]
    return {'nlen': nlen, 'elen': elen, 'data_offset': offset + 30 + nlen + elen}


def download_range(offset: int, length: int, target: Path, force: bool = False) -> Path:
    """Range 下载指定字节区间，支持断点续传。"""
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_suffix(target.suffix + '.part')
    start_byte = offset
    existing = 0
    if part.exists() and not force:
        existing = part.stat().st_size
        if existing >= length:
            existing = 0  # 已完整则重下
    if existing > 0:
        start_byte = offset + existing
        mode = 'ab'
        print(f'  [resume] {target.name}: 已有 {existing}/{length} bytes，从 offset {start_byte} 继续')
    else:
        mode = 'wb'
        print(f'  [download] {target.name}: {length} bytes from offset {offset}')
    end_byte = offset + length - 1
    if start_byte > end_byte:
        return part
    r = requests.get(URL, headers={**HEADERS, 'Range': f'bytes={start_byte}-{end_byte}'},
                     timeout=300, stream=True)
    if r.status_code != 206:
        raise RuntimeError(f'Range 请求失败: {r.status_code}')
    with open(part, mode) as fh:
        for chunk in r.iter_content(chunk_size=1 << 20):
            if chunk:
                fh.write(chunk)
    if part.stat().st_size < length:
        raise RuntimeError(f'{target.name} 下载不完整: {part.stat().st_size}/{length}')
    import os
    os.replace(part, target)
    return target


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--outdir', default=str(DATA_DIR / 'elliptic2'))
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print('[elliptic2] Kaggle 免登录 + Range 断点提取（只下小文件，跳过 25GB 背景图）')
    downloaded = {}
    for name, meta in FILES.items():
        target = outdir / (name + '.deflate')
        if target.exists() and not args.force:
            print(f'  [skip] {name}: 已存在')
            downloaded[name] = target
            continue
        # 解析 local header 得到数据起始偏移
        lh = parse_local_header(meta['local_off'])
        data_off = lh['data_offset']
        print(f'  {name}: data_offset={data_off} (local_off={meta["local_off"]}, nlen={lh["nlen"]}, elen={lh["elen"]})')
        raw = download_range(data_off, meta['comp'], target, args.force)
        downloaded[name] = raw

    # 解压 raw deflate（ZIP method=8，无 zlib 头）
    extracted = {}
    for name, raw in downloaded.items():
        out = outdir / name
        if out.exists() and not args.force:
            print(f'  [skip] 解压 {name}: 已存在')
            extracted[name] = out
            continue
        data = raw.read_bytes()
        # raw deflate: wbits=-15
        import zlib
        try:
            dec = zlib.decompressobj(-15).decompress(data)
        except Exception as e:
            # 可能带 PK 头残留，尝试跳过
            raise RuntimeError(f'解压 {name} 失败: {e}')
        if len(dec) < FILES[name]['uncomp'] * 0.95:
            raise RuntimeError(f'{name} 解压大小异常: {len(dec)} vs {FILES[name]["uncomp"]}')
        out.write_bytes(dec)
        extracted[name] = out
        print(f'  [extract] {name}: {len(dec)} bytes')

    detail = f'从 25.8GB ZIP 中用 Range 只提取 3 个标注子图文件：{", ".join(extracted.keys())}'
    write_provenance('elliptic2', {'source': 'Kaggle (免登录+Range)', 'detail': detail})
    (RESULTS_DIR / 'elliptic2_extraction.json').write_text(
        json.dumps({'status': 'PASS', 'files': {k: str(v) for k, v in extracted.items()},
                    'detail': detail}, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'\n[done] {detail}')
    for name, p in extracted.items():
        print(f'  {outdir / name}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
