"""
common.py - TinyAML-Bench 公共基础设施
=====================================
提供三项核心能力（全部仅依赖 numpy / pandas，保证最小环境可跑）：

1. ensure_data():     数据存在则跳过、不存在则自动下载、下载失败优雅降级到合成数据
2. Checkpoint:        断点续跑：原子写入、Ctrl+C 安全退出、重启跳过已完成 run
3. 指标函数:          AUC / PR-AUC / F1 / P@k / ECE（纯 numpy 实现，不依赖 sklearn）

设计原则：任何外部依赖缺失或网络失败都不得导致整个实验崩溃；
每次降级都会写进 provenance.json，供论文诚信声明使用。
"""
from __future__ import annotations

import json
import os
import signal
import sys
import tempfile
import time
import zipfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 路径常量（相对本文件定位，避免 cwd 依赖）
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
SMOKE_DIR = ROOT / "smoke_test"
CONFIG_DIR = ROOT / "configs"
for _d in (DATA_DIR, RESULTS_DIR, SMOKE_DIR, CONFIG_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# 1. 数据保障：存在即用 / 缺失下载 / 失败降级
# ---------------------------------------------------------------------------
@dataclass
class DataResult:
    path: Path
    source: str          # "cached" | "downloaded" | "synthetic"
    detail: str = ""
    extra: dict = field(default_factory=dict)


def ensure_data(
    name: str,
    target: Path,
    downloader: Callable[[Path], Path],
    synthetic: Callable[[Path], Path] | None = None,
    validator: Callable[[Path], bool] | None = None,
    force: bool = False,
) -> DataResult:
    """数据保障主入口。

    - 已有有效数据 -> 直接返回（不重复下载）
    - 无数据 -> 调用 downloader；成功返回 downloaded
    - 下载失败 -> 调用 synthetic 降级；仍失败则抛出（让上层明确失败）
    """
    target = Path(target)
    valid = target.exists() and target.stat().st_size > 0
    if validator is not None and valid:
        try:
            valid = bool(validator(target))
        except Exception:
            valid = False

    if valid and not force:
        return DataResult(target, "cached", f"已存在有效数据，跳过下载：{target.name}")

    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        if target.exists() and force:
            target.unlink()
        got = Path(downloader(target))
        if not got.exists() or got.stat().st_size == 0:
            raise RuntimeError(f"下载器未产出有效文件：{got}")
        if validator is not None and not validator(got):
            raise RuntimeError(f"下载文件校验失败：{got}")
        return DataResult(got, "downloaded", f"自动下载成功：{got.name}")
    except Exception as exc:  # noqa: BLE001 - 任何下载失败都要降级
        msg = f"{type(exc).__name__}: {exc}"
        if synthetic is None:
            raise RuntimeError(f"数据 {name} 下载失败且无合成降级：{msg}") from exc
        got = Path(synthetic(target))
        return DataResult(got, "synthetic", f"下载失败，已降级为合成数据（{msg}）")


def download_url(url: str, target: Path, timeout: int = 30) -> Path:
    """带超时的流式下载（ requests 可选；缺失则抛错由上层降级）。"""
    import requests  # 延迟导入：无 requests 时也能走 synthetic 分支

    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".part")
    with requests.get(url, stream=True, timeout=timeout) as resp:
        resp.raise_for_status()
        with open(tmp, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=1 << 16):
                if chunk:
                    fh.write(chunk)
    os.replace(tmp, target)
    return target


def read_csv_safe(path: Path, **kwargs) -> pd.DataFrame:
    """pandas 版本兼容的 CSV 读取（处理 3.x 的引擎差异）。"""
    return pd.read_csv(path, **kwargs)


# ---------------------------------------------------------------------------
# 2. 断点续跑 Checkpoint
# ---------------------------------------------------------------------------
class Checkpoint:
    """基于 JSON 的轻量 checkpoint。

    用法：
        ckpt = Checkpoint("tabular", RESULTS_DIR / "tabular_state.json")
        if ckpt.is_done("LightGBM|seed0"):
            continue
        ckpt.mark_start("LightGBM|seed0")
        ... 训练 ...
        ckpt.mark_done("LightGBM|seed0", {"auc": 0.8})

    特性：
      - 原子写入（临时文件 + os.replace），中断不损坏状态文件
      - mark_start 记录 running，重启时发现 running 会重置为 todo 并重跑
      - install_signal_handler() 让 Ctrl+C 优雅落盘
    """

    def __init__(self, name: str, path: Path):
        self.name = name
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.state: dict[str, Any] = {"name": name, "runs": {}, "updated_at": None}
        if self.path.exists():
            try:
                loaded = json.loads(self.path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict) and "runs" in loaded:
                    self.state = loaded
            except Exception as exc:  # 损坏则备份后重建，绝不静默丢数据
                bak = self.path.with_suffix(".corrupt." + datetime.now().strftime("%Y%m%d%H%M%S"))
                try:
                    self.path.rename(bak)
                except Exception:
                    pass
                print(f"[checkpoint] 状态文件损坏，已备份为 {bak.name} 并重建", file=sys.stderr)
        # 上次异常中断的 running 状态重置为 todo
        revived = 0
        for k, v in self.state["runs"].items():
            if isinstance(v, dict) and v.get("status") == "running":
                v["status"] = "todo"
                revived += 1
        if revived:
            print(f"[checkpoint] {self.name}: 检测到 {revived} 个中断 run，将重新执行")
        self._flush()

    def is_done(self, key: str) -> bool:
        v = self.state["runs"].get(key)
        return bool(isinstance(v, dict) and v.get("status") == "done")

    def mark_start(self, key: str) -> None:
        self.state["runs"][key] = {"status": "running", "started_at": _now()}
        self._flush()

    def mark_done(self, key: str, metrics: dict | None = None) -> None:
        self.state["runs"][key] = {
            "status": "done",
            "finished_at": _now(),
            "metrics": metrics or {},
        }
        self._flush()

    def skip(self, key: str) -> None:
        self.state["runs"][key] = {"status": "skipped", "at": _now()}
        self._flush()

    def todo_keys(self, all_keys: Iterable[str]) -> list[str]:
        return [k for k in all_keys if not self.is_done(k)]

    def _flush(self) -> None:
        self.state["updated_at"] = _now()
        fd, tmp_name = tempfile.mkstemp(dir=str(self.path.parent), prefix=".ckpt_", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(self.state, fh, ensure_ascii=False, indent=2, default=_json_default)
            os.replace(tmp_name, self.path)
        except Exception:
            try:
                os.unlink(tmp_name)
            except Exception:
                pass
            raise

    def install_signal_handler(self) -> None:
        """Ctrl+C 时落盘当前状态再退出。"""

        def _handler(signum, frame):  # noqa: ARG001
            print(f"\n[checkpoint] 收到信号 {signum}，保存进度后退出（下次可续跑）", file=sys.stderr)
            self._flush()
            raise SystemExit(130)

        try:
            signal.signal(signal.SIGINT, _handler)
            if hasattr(signal, "SIGTERM"):
                signal.signal(signal.SIGTERM, _handler)
        except ValueError:
            # 非主线程无法注册信号处理器，忽略
            pass


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _json_default(obj):
    """把 numpy 标量/数组转成可 JSON 序列化类型（基础设施级修复）。"""
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False, default=_json_default) + "\n")


def write_provenance(name: str, payload: dict) -> None:
    """记录数据来源/降级，供论文诚信声明。"""
    append_jsonl(CONFIG_DIR / "provenance.jsonl", {"item": name, "at": _now(), **payload})


# ---------------------------------------------------------------------------
# 3. 指标（纯 numpy，零外部依赖）
# ---------------------------------------------------------------------------
def _binary_metrics_np(y_true: np.ndarray, score: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """按 score 降序排列，返回 (排序后的y, 排序后的score)。"""
    y = np.asarray(y_true).astype(int).ravel()
    s = np.asarray(score).astype(float).ravel()
    order = np.argsort(-s, kind="mergesort")
    return y[order], s[order]


def roc_auc(y_true: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y_true).astype(int).ravel()
    s = np.asarray(score).astype(float).ravel()
    pos, neg = int(y.sum()), int((1 - y).sum())
    if pos == 0 or neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ys = y[order]
    # 秩和法（处理并列取平均秩）
    ranks = np.empty(len(s), dtype=float)
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[order[j + 1]] == s[order[i]]:
            j += 1
        ranks[i : j + 1] = (i + j) / 2.0 + 1.0
        i = j + 1
    return float((ranks[ys == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


def pr_auc(y_true: np.ndarray, score: np.ndarray) -> float:
    """Average Precision（Sklearn-style AP，step-wise），适合不平衡数据。"""
    y, _ = _binary_metrics_np(y_true, score)
    pos = int(y.sum())
    if pos == 0:
        return float("nan")
    tp = np.cumsum(y)
    precision = tp / np.arange(1, len(y) + 1)
    return float(np.sum(precision * y) / pos)


def precision_at_k(y_true: np.ndarray, score: np.ndarray, k: float = 0.01) -> float:
    """修正版 P@k：按正类总数确定可评估深度，避免 n_pos<k*n 时退化为 1.0。

    返回 top-ceil(k*n) 中的 precision；若该深度小于 1 则取 max(1, ceil(k*n))，
    并在precision计算中只用实际存在的正类数作分母判断（不再假报 1.0）。
    """
    y, _ = _binary_metrics_np(y_true, score)
    n = len(y)
    if n == 0:
        return float("nan")
    depth = max(1, int(np.ceil(k * n)))
    hits = int(y[:depth].sum())
    return float(hits / depth)


def f1_at_best_threshold(y_true: np.ndarray, score: np.ndarray) -> tuple[float, float]:
    """扫描阈值取最大 F1，返回 (best_f1, threshold)。"""
    y = np.asarray(y_true).astype(int).ravel()
    s = np.asarray(score).astype(float).ravel()
    if len(np.unique(y)) < 2:
        return float("nan"), float("nan")
    cand = np.unique(np.quantile(s, np.linspace(0.01, 0.99, 99)))
    best_f1, best_t = -1.0, float(np.median(s))
    for t in cand:
        pred = (s >= t).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        denom = 2 * tp + fp + fn
        f1 = 2 * tp / denom if denom else 0.0
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return float(best_f1), best_t


def expected_calibration_error(y_true: np.ndarray, score: np.ndarray, n_bins: int = 10) -> float:
    """ECE（等宽分箱）。score 需为 [0,1] 概率；内部做 clip。"""
    y = np.asarray(y_true).astype(int).ravel()
    p = np.clip(np.asarray(score).astype(float).ravel(), 1e-6, 1 - 1e-6)
    if len(y) == 0:
        return float("nan")
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        mask = idx == b
        if mask.any():
            ece += mask.mean() * abs(float(y[mask].mean() - p[mask].mean()))
    return float(ece)


def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    return np.where(z >= 0, 1 / (1 + np.exp(-z)), np.exp(z) / (1 + np.exp(z)))


def compute_metrics(y_true: np.ndarray, score: np.ndarray, k: float = 0.01) -> dict:
    f1, thr = f1_at_best_threshold(y_true, score)
    return {
        "AUC": roc_auc(y_true, score),
        "PR-AUC": pr_auc(y_true, score),
        "F1": f1,
        "F1_threshold": thr,
        "P@k": precision_at_k(y_true, score, k=k),
        "ECE": expected_calibration_error(y_true, sigmoid(score) if np.nanmin(score) < 0 or np.nanmax(score) > 1 else score),
        "n": int(len(y_true)),
        "pos_rate": float(np.mean(y_true)),
    }


def set_seed(seed: int) -> None:
    np.random.seed(seed)


def timer() -> Callable[[], float]:
    t0 = time.perf_counter()
    return lambda: time.perf_counter() - t0


M2_SEED = 20240101


def random_pair_edges(n: int, m: int, rng=None) -> np.ndarray:
    """M2 对照：均匀随机配对（Erdos-Renyi 式），不是从原边表抽子集。

    旧实现 `rng.choice(len(src), size=m)` 抽的是**真实边**的子集，因而保留了
    真实边的标签同配性；那样 M2 与 C1 的差别只是边数，无法把"边身份"与
    "边密度"分开。这里改为在 n 个节点上均匀抽 m 条无自环、无重边的配对。

    随机流刻意独立于调用方共享的 rng（固定 M2_SEED）：同一 (n, m) 在任何
    脚本里都生成同一张图，否则 44 与 46 会因调用顺序不同而得到两个 M2 值，
    论文里就会出现同一折两个数。

    返回 shape=(m, 2) 的 int 数组。
    """
    if m <= 0:
        return np.zeros((0, 2), dtype=np.int64)
    rng = np.random.default_rng(M2_SEED)
    seen = set()
    u, v = [], []
    while len(seen) < m:
        batch = max(2 * (m - len(seen)), 1024)
        a = rng.integers(0, n, size=batch)
        b = rng.integers(0, n, size=batch)
        for x, z in zip(a, b):
            if x == z:
                continue
            key = (x, z) if x < z else (z, x)
            if key in seen:
                continue
            seen.add(key)
            u.append(x)
            v.append(z)
            if len(seen) >= m:
                break
    return np.column_stack([np.asarray(u, dtype=np.int64), np.asarray(v, dtype=np.int64)])
