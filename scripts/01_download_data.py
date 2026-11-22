"""
01_download_data.py - 数据获取（自动下载 / 已存在跳过 / 失败降级合成）
=====================================================================
用法：
    python scripts/01_download_data.py --dataset german        # UCI German Credit（真实，自动下载）
    python scripts/01_download_data.py --dataset synthetic      # 合成图+表格（任何环境可跑）
    python scripts/01_download_data.py --dataset elliptic2      # Elliptic2 指引（需申请，不自动下载）
    python scripts/01_download_data.py --dataset all            # german + synthetic
    python scripts/01_download_data.py --dataset german --force # 强制重新下载
    python scripts/01_download_data.py --dataset german --check # 仅检查，不下载

幂等性：再运行一次不会重复下载；--force 才覆盖。
降级：网络失败自动产出合成数据并写 provenance.jsonl，绝不中断主流程。
"""
from __future__ import annotations

import argparse
import io
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    DATA_DIR, ensure_data, download_url, write_provenance, set_seed,
)

UCI_GERMAN_URL = "https://archive.ics.uci.edu/static/public/144/statlog+german+credit+data.zip"
UCI_GERMAN_ALT = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"

# German Credit 的 UCI 列名（statlog 格式，tab 分隔，无表头）
GERMAN_COLS = [
    "checking", "duration", "credit_history", "purpose", "amount", "savings",
    "employment", "installment_rate", "personal_status", "guarantors",
    "residence", "property", "age", "installment_plans", "housing",
    "nb_credits", "job", "nb_people", "phone", "foreign", "label",
]


def _synthetic_german(target: Path) -> Path:
    """降级用：生成与 German Credit 同 schema 的合成信贷数据。"""
    set_seed(42)
    n = 1000
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "checking": rng.integers(1, 5, n),
        "duration": rng.integers(4, 72, n),
        "credit_history": rng.integers(1, 6, n),
        "purpose": rng.integers(1, 11, n),
        "amount": rng.integers(250, 10000, n),
        "savings": rng.integers(1, 6, n),
        "employment": rng.integers(1, 6, n),
        "installment_rate": rng.integers(1, 5, n),
        "personal_status": rng.integers(1, 5, n),
        "guarantors": rng.integers(1, 4, n),
        "residence": rng.integers(1, 5, n),
        "property": rng.integers(1, 5, n),
        "age": rng.integers(18, 75, n),
        "installment_plans": rng.integers(1, 4, n),
        "housing": rng.integers(1, 4, n),
        "nb_credits": rng.integers(1, 5, n),
        "job": rng.integers(1, 5, n),
        "nb_people": rng.integers(1, 3, n),
        "phone": rng.integers(1, 3, n),
        "foreign": rng.integers(1, 3, n),
    })
    # 用逻辑函数生成标签，保证信号真实存在（不是纯随机）
    logit = (-1.2 + 0.03 * df["duration"] - 0.0002 * df["amount"]
             + 0.25 * df["savings"] - 0.2 * df["checking"])
    p = 1 / (1 + np.exp(-logit))
    df["label"] = (rng.random(n) < p).astype(int)
    df.to_csv(target, index=False)
    return target


def _download_german(target: Path) -> Path:
    """真实下载 UCI German Credit，自动解析 zip 或 data 文件。"""
    import requests
    out_csv = target  # data/german_credit.csv
    try:
        resp = requests.get(UCI_GERMAN_URL, timeout=30)
        resp.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            names = [n for n in zf.namelist() if n.endswith("german.data")]
            if not names:
                raise RuntimeError("zip 中未找到 german.data")
            raw_bytes = zf.read(names[0])
    except Exception:  # zip 失败则尝试直链 data
        resp = requests.get(UCI_GERMAN_ALT, timeout=30)
        resp.raise_for_status()
        raw_bytes = resp.content

    text = raw_bytes.decode("utf-8", errors="replace")
    # statlog german.data：tab/空格分隔，21 列，最后一列 1=good 2=bad
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.replace("\t", " ").split()
        if len(parts) != 21:
            continue
        rows.append(parts)
    if len(rows) < 500:
        raise RuntimeError(f"解析出的样本过少：{len(rows)}")
    df = pd.DataFrame(rows, columns=GERMAN_COLS)
    # UCI german.data 混合类型：数值列（duration/amount/age/...）直接转数值，
    # 分类列（checking=A11..A14 等字符串）做标签编码，避免 to_numeric 产生 NaN。
    for c in GERMAN_COLS[:-1]:  # 除 label 外
        converted = pd.to_numeric(df[c], errors="coerce")
        if converted.notna().all():
            df[c] = converted
        else:
            # 分类字符串 -> 稳定标签编码（按字母序，保证可复现）
            cats = sorted(df[c].unique())
            df[c] = df[c].map({v: i for i, v in enumerate(cats)}).astype(int)
    # UCI 原始标签 1=good(良性) 2=bad(坏)，转为 0=good 1=bad（风险正类）
    # 注意：label 列来自字符串解析，需先转数值再比较（否则 "2"==2 为 False）
    df["label"] = (pd.to_numeric(df["label"], errors="coerce") == 2).astype(int)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    return out_csv


def _download_check(target: Path) -> Path:
    """--check 不真正下载，仅报告目标路径；交由 ensure_data 判定。"""
    return target


def prepare_german(force: bool = False, check_only: bool = False) -> dict:
    target = DATA_DIR / "german_credit.csv"

    def _validator(path: Path) -> bool:
        try:
            df = pd.read_csv(path, nrows=5)
            return "label" in df.columns and len(df.columns) >= 5
        except Exception:
            return False

    if check_only:
        exists = target.exists() and target.stat().st_size > 0
        print(f"[check] {target.name}: {'存在 (' + str(target.stat().st_size) + ' bytes)' if exists else '不存在'}")
        return {"dataset": "german", "status": "cached" if exists else "missing"}

    # ensure_data：存在则跳过；不存在下载；下载失败降级合成
    # 用 downloader 包装，check_only 场景不涉及
    def _downloader(t: Path) -> Path:
        return _download_german(t)

    res = ensure_data(
        "german_credit", target, _downloader,
        synthetic=_synthetic_german, validator=_validator, force=force,
    )
    df = pd.read_csv(res.path)
    write_provenance("german_credit", {
        "source": res.source, "detail": res.detail,
        "n": len(df), "pos_rate": float(df["label"].mean()),
    })
    print(f"[german] 来源={res.source} 样本={len(df)} 正类率={df['label'].mean():.3f} 说明={res.detail}")
    return {"dataset": "german", "status": res.source, "n": len(df), "path": str(res.path)}


def prepare_synthetic(force: bool = False) -> dict:
    """合成三件套：表格 + 图（节点/边）+ 文本。任何环境可复现。"""
    set_seed(2026)
    rng = np.random.default_rng(2026)

    # 1) 表格
    tab = DATA_DIR / "synthetic_tabular.csv"
    if not tab.exists() or force:
        n = 800
        f1 = rng.normal(0, 1, n)
        f2 = rng.normal(0, 1, n)
        f3 = rng.normal(0, 1, n)
        logit = -1.0 + 1.5 * f1 - 0.8 * f2 + 0.3 * f3
        y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
        pd.DataFrame({"f1": f1, "f2": f2, "f3": f3, "label": y}).to_csv(tab, index=False)

    # 2) 图：节点含特征+标签+时间，边含时间（支撑流式时间分裂）
    nodes_p = DATA_DIR / "synthetic_graph_nodes.csv"
    edges_p = DATA_DIR / "synthetic_graph_edges.csv"
    if (not nodes_p.exists() or not edges_p.exists() or force):
        n_nodes = 400
        feats = rng.normal(0, 1, (n_nodes, 8))
        # 社区结构：前 200 节点为一类（高风险倾向）
        logit = (-2.0 if True else 0) + np.concatenate([
            rng.normal(1.0, 0.5, n_nodes // 2), rng.normal(-1.0, 0.5, n_nodes - n_nodes // 2)
        ]) + 0.4 * feats[:, 0]
        y = (rng.random(n_nodes) < 1 / (1 + np.exp(-np.clip(logit, -10, 10)))).astype(int)
        times = np.sort(rng.integers(0, 100, n_nodes))
        nd = pd.DataFrame(feats, columns=[f"x{i}" for i in range(8)])
        nd.insert(0, "node_id", range(n_nodes))
        nd["label"] = y
        nd["time"] = times
        nd.to_csv(nodes_p, index=False)

        edges = []
        for i in range(n_nodes):
            for _ in range(rng.integers(2, 6)):
                j = int(rng.integers(0, n_nodes))
                if j != i:
                    edges.append((i, j, int(rng.integers(0, 100))))
        ed = pd.DataFrame(edges, columns=["src", "dst", "time"])
        ed.to_csv(edges_p, index=False)

    # 3) 文本：短句 + 风险标签（供文本轨道冒烟）
    txt_p = DATA_DIR / "synthetic_text.csv"
    if not txt_p.exists() or force:
        risky = ["wire transfer to shell company", "rapid movement of funds", "unusual offshore account",
                 "phishing link requested payment", "mixer service obfuscation"]
        safe = ["payroll payment to employee", "monthly rent payment", "invoice for office supplies",
                "refund to customer account", "subscription renewal"]
        rows = []
        for _ in range(120):
            if rng.random() < 0.25:
                rows.append((" ".join(rng.choice(risky, 2)), 1))
            else:
                rows.append((" ".join(rng.choice(safe, 2)), 0))
        pd.DataFrame(rows, columns=["text", "label"]).to_csv(txt_p, index=False)

    write_provenance("synthetic", {"source": "synthetic", "detail": "确定性生成 seed=2026",
                                   "files": [str(tab.name), str(nodes_p.name), str(edges_p.name), str(txt_p.name)]})
    print(f"[synthetic] 已就绪：{tab.name}, {nodes_p.name}, {edges_p.name}, {txt_p.name}")
    return {"dataset": "synthetic", "status": "synthetic", "path": str(DATA_DIR)}


def elliptic2_guide() -> dict:
    """Elliptic2 需申请，不能自动下载；输出清晰指引。"""
    guide = {
        "dataset": "elliptic2",
        "status": "manual_required",
        "reason": "Elliptic2 需向 Elliptic 申请（填表 1-2 周），不可自动抓取",
        "steps": [
            "1. 访问 http://elliptic.co/elliptic2 填表申请",
            "2. 或使用 Kaggle 镜像（需凭据），下载后解压到 data/elliptic2/",
            "3. 官方代码：https://github.com/MITIBMxGraph/Elliptic2",
            "4. 论文引用：arXiv:2404.19109 (KDD MLF 2024)",
            "5. 落盘后运行：python scripts/01_download_data.py --dataset elliptic2 --check",
        ],
        "expected_layout": "data/elliptic2/{nodes.csv, edges.csv, labels.csv, subgraphs.csv}",
    }
    print("[elliptic2] " + guide["reason"])
    for s in guide["steps"]:
        print("  " + s)
    # 检查是否已手动放置
    p = DATA_DIR / "elliptic2"
    guide["present"] = p.exists() and any(p.iterdir()) if p.exists() else False
    print(f"[elliptic2] 数据目录状态：{'已检测到文件' if guide['present'] else '未检测到（需手动放置）'}")
    write_provenance("elliptic2", {"source": "manual_required", "detail": guide["reason"],
                                   "present": guide["present"]})
    return guide


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="all",
                    choices=["german", "synthetic", "elliptic2", "real_cats", "all"])
    ap.add_argument("--force", action="store_true", help="强制重新下载/生成（覆盖已有）")
    ap.add_argument("--check", action="store_true", help="仅检查数据是否存在，不下载")
    args = ap.parse_args()

    results = []
    if args.dataset in ("german", "all"):
        results.append(prepare_german(force=args.force, check_only=args.check))
    if args.dataset in ("synthetic", "all"):
        if args.check:
            n = len(list(DATA_DIR.glob("synthetic_*.csv")))
            print(f"[check] synthetic 文件数：{n}")
        else:
            results.append(prepare_synthetic(force=args.force))
    if args.dataset == "elliptic2":
        results.append(elliptic2_guide())
    if args.dataset in ("real_cats", "all"):
        out = DATA_DIR / "real_cats_eth.csv"
        if args.check:
            print(f"[check] real_cats_eth.csv: "
                  f"{'存在' if out.exists() else '不存在'}（原始 TSV 在 data/real_cats/）")
        else:
            try:
                results.append(prepare_real_cats(force=args.force))
            except Exception as exc:
                # 真实数据下载失败不阻断其他数据集（论文级结果不可用，但管线仍可跑）
                print(f"[WARN] Real-CATS 获取失败（不影响其他数据集）：{type(exc).__name__}: {exc}")
                write_provenance("real_cats", {"source": "failed",
                                               "detail": f"{type(exc).__name__}: {exc}"})

    print("[done] 数据准备完成：" + ", ".join(r.get("dataset", "?") for r in results))
    return 0


# ===========================================================================
# Real-CATS 真实链上数据（论文级）—— 追加于原脚本之后
# 来源：https://github.com/sjdseu/Real-CATS (CC BY-NC-SA)
# CE.tsv = 12,572 个犯罪以太坊地址（21 类诈骗）；BE.tsv = 16,019 个良性地址
# 合并后做地址级二分类，first_time 支持严格时间分裂
# ===========================================================================
REAL_CATS_BASE = "https://raw.githubusercontent.com/sjdseu/Real-CATS/master/"


def _download_real_cats_file(target: Path, filename: str, timeout: int = 180) -> Path:
    """下载单个 Real-CATS TSV 文件，带超时与重试。"""
    import requests
    target.parent.mkdir(parents=True, exist_ok=True)
    url = REAL_CATS_BASE + filename
    last_exc = None
    for attempt in range(3):
        try:
            resp = requests.get(url, timeout=timeout)
            resp.raise_for_status()
            tmp = target.with_suffix(target.suffix + ".part")
            with open(tmp, "wb") as fh:
                fh.write(resp.content)
            os.replace(tmp, target)
            return target
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
    raise RuntimeError(f"下载 {filename} 失败（3 次重试）：{last_exc}")


def prepare_real_cats(force: bool = False) -> dict:
    """下载并合并 Real-CATS CE(犯罪)+BE(良性) 为统一二分类数据集。

    产出 data/real_cats_eth.csv，列：address, label(0=良性 1=犯罪),
    first_time(时间戳), 以及 49 个数值特征。
    """
    d = DATA_DIR / "real_cats"
    d.mkdir(parents=True, exist_ok=True)
    ce = d / "CE.tsv"
    be = d / "BE.tsv"
    out = DATA_DIR / "real_cats_eth.csv"

    def _validator(p: Path) -> bool:
        try:
            df = pd.read_csv(p, sep="\t", nrows=2)
            return "address" in df.columns and "label" in df.columns
        except Exception:
            return False

    # 用 ensure_data 保障两个原始文件（存在跳过、缺失下载、失败抛错由上层处理）
    for p, fn in [(ce, "CE.tsv"), (be, "BE.tsv")]:
        if not p.exists() or p.stat().st_size < 1000 or force:
            res = ensure_data(f"real_cats_{fn}", p,
                              lambda t, f=fn: _download_real_cats_file(t, f),
                              synthetic=None, validator=_validator, force=force)
            print(f"[real_cats] {fn}: {res.detail}")
        else:
            print(f"[real_cats] {fn}: 已存在，跳过下载")

    if out.exists() and not force:
        df = pd.read_csv(out)
        print(f"[real_cats] 合并文件已存在：{out.name} n={len(df)}（跳过合并）")
        write_provenance("real_cats", {"source": "cached", "detail": "合并文件已存在", "n": len(df)})
        return {"dataset": "real_cats", "status": "cached", "n": len(df), "path": str(out)}

    # 读取并合并
    cdf = pd.read_csv(ce, sep="\t")
    bdf = pd.read_csv(be, sep="\t")
    cdf["label"] = 1  # 犯罪
    bdf["label"] = 0  # 良性
    merged = pd.concat([cdf, bdf], ignore_index=True)

    # 时间列转时间戳（支持严格时间分裂）
    merged["time"] = pd.to_datetime(merged["first_time"], errors="coerce")
    merged["time"] = (merged["time"] - pd.Timestamp("1970-01-01")) // pd.Timedelta("1s")
    # 缺失时间的用中位数填补（保证所有样本可用；记录填补数）
    n_missing_time = int(merged["time"].isna().sum())
    merged["time"] = merged["time"].fillna(merged["time"].median())

    # 特征：只保留数值列，排除地址/标签/原始时间字符串
    drop_cols = {"address", "label", "first_time", "last_time", "time"}
    feat_cols = [c for c in merged.columns
                 if c not in drop_cols and pd.api.types.is_numeric_dtype(merged[c])]
    # 清理 inf / 极端值（log1p 压缩数量级差异）
    X = merged[feat_cols].replace([np.inf, -np.inf], np.nan)
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0)
    X = np.sign(X) * np.log1p(np.abs(X))
    out_df = pd.DataFrame(X, columns=feat_cols)
    out_df.insert(0, "time", merged["time"].values)
    out_df.insert(0, "label", merged["label"].values)
    out_df.insert(0, "address", merged["address"].values)
    out_df.to_csv(out, index=False)

    detail = (f"CE={len(cdf)} 犯罪 + BE={len(bdf)} 良性；{len(feat_cols)} 数值特征；"
              f"时间缺失填补 {n_missing_time} 条")
    write_provenance("real_cats", {"source": "downloaded", "detail": detail,
                                   "n": len(out_df), "n_criminal": len(cdf), "n_benign": len(bdf)})
    print(f"[real_cats] 合并完成：{out.name} n={len(out_df)} pos_rate={out_df['label'].mean():.3f} ({detail})")
    return {"dataset": "real_cats", "status": "downloaded", "n": len(out_df), "path": str(out)}


if __name__ == "__main__":
    raise SystemExit(main())
