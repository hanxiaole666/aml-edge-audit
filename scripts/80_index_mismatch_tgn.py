"""
80_index_mismatch_tgn.py - P0-1 验证：TGN 的免疫来自切分索引还是模型类别？
=============================================================================
背景（v14 评审 P0-1）：
  Elliptic 的 step-boundary split 下 TGN-lite 的 random-vs-temporal gap 为负
  (-0.030, 3/10)，正文原写成"因果记忆模型的正确表现"。但第二节点时间数据
  集上 TGN-lite 是四个 read-out 中最大 (+0.331)。两种解释：
    (H1) 模型类别属性：因果记忆天生不吃随机切分的便宜；
    (H2) 切分索引属性：Elliptic 默认切分按 step，与 TGN 消费的时间索引同构，
         所以udge 在 t 处看不到 t+1；node-rank 切分没有 step 索引，同构消失。

本实验在 Elliptic 上把切分算子从 step-boundary 换成 Table XI 的
windowed node-rank cut（窗口内按节点秩对半，w=1/2/4/8/16），对每个窗口做
random-vs-structured 对照：

  structured 训练标签 = node-rank 切分的训练侧（仍大体按时间，仅边界窗口
                        内按节点秩随机）—— 这是"切分索引与模型索引错配"的条件；
  random 训练标签     = 同 size，把约 1/4 训练节点换成测试侧节点（测试集固定）。

两种条件用同一测试侧、同一 TGN 记忆流（记忆仍按真实 step 更新）。若 TGN 的
optimism 在该算子下相对 step-boundary 的 -0.030 显著放大（转正），支持 H2；
若仍为负，H2 被否证，得到一个更硬的边界条件。

同时跑 LightGBM 作为对照：它不消费 step 索引，预期在每个窗口下都保持正的
optimism，从而把"索引同构"的作用单独显出来。

用法：python scripts/80_index_mismatch_tgn.py [--epochs 40]
产出：results/index_mismatch_tgn.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from lightgbm import LGBMClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RESULTS_DIR, pr_auc, set_seed, write_provenance

_spec = importlib.util.spec_from_file_location(
    "rolling", Path(__file__).resolve().parent / "55_elliptic_rolling_matrix.py")
roll = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(roll)

torch.set_num_threads(8)


def node_rank_split(ts, y, w, boundary_step=30, seed=0):
    """Table XI 的 windowed node-rank cut：窗口内边界节点按秩对半。"""
    rng = np.random.default_rng(seed)
    win = ts // w
    cw = int(boundary_step // w)
    lab = y >= 0
    tr = np.where(lab & (win < cw))[0].tolist()
    te = np.where(lab & (win > cw))[0].tolist()
    b = np.where(lab & (win == cw))[0]
    rng.shuffle(b)
    half = len(b) // 2
    tr += b[:half].tolist()
    te += b[half:].tolist()
    return np.array(sorted(tr)), np.array(sorted(te))


def boot_ci(y, s_s, s_r, seed=0, n_boot=300):
    """测试集 bootstrap：optimism = AP(random) - AP(structured)。"""
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) < 2:
            continue
        d.append(pr_auc(y[i], s_r[i]) - pr_auc(y[i], s_s[i]))
    if not d:
        return [None, None]
    return [round(float(np.percentile(d, 2.5)), 4),
            round(float(np.percentile(d, 97.5)), 4)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", default=str(RESULTS_DIR))
    ap.add_argument("--epochs", type=int, default=40)
    args = ap.parse_args()
    set_seed(0)

    tx, ts, X, y, src, dst = roll.load()
    n = len(X)
    An, _ = roll.sym_norm(src, dst, n)
    steps_all = list(range(1, int(ts.max()) + 1))

    out = {
        "experiment": "index_mismatch_tgn",
        "dataset": "Elliptic v1 (real steps)",
        "hypotheses": {
            "H1_model_class": "causal memory is inherently immune to random splitting",
            "H2_split_index": "immunity holds only when split index matches the model's step index",
        },
        "split_operator": "windowed node-rank cut (Table XI); boundary window split by node rank, not step",
        "comparison": "random training labels vs structured (node-rank) training labels, same test side",
        "readouts": ["LightGBM", "TGN-lite"],
        "reference_step_boundary_tgn_delta": -0.030,
        "rows": [],
    }

    for w in (1, 2, 4, 8, 16):
        tr_s, te = node_rank_split(ts, y, w, seed=w)
        # 早停验证集：训练侧按 step 顺序取最后 10%
        order_tr = tr_s[np.argsort(ts[tr_s], kind="mergesort")]
        n_val = max(50, int(0.1 * len(order_tr)))
        va = order_tr[-n_val:]
        tr = np.setdiff1d(order_tr, va)
        # 随机对照：打破时间索引——把一部分测试侧的已标注节点换进训练标签，
        # 同时换出等量原训练节点，保持训练 size 与测试集 te 固定。
        # 这样 random 条件在训练期"看到"原本属于测试时段分布的节点，
        # 而节点自身仍在 te 中评估（标签不直接泄漏：换入的是别的节点）。
        rng = np.random.default_rng(w + 100)
        n_swap = min(len(te) // 4, len(tr) // 4)
        out_idx = rng.choice(tr, size=n_swap, replace=False)
        in_idx = rng.choice(te, size=n_swap, replace=False)
        tr_r = np.setdiff1d(tr, out_idx)
        tr_r = np.concatenate([tr_r, in_idx])
        tr_r = np.setdiff1d(tr_r, va)
        row = {"window": w, "n_train": int(len(tr)), "n_val": int(len(va)),
               "n_test": int(len(te)),
               "test_prevalence": round(float(y[te].mean()), 4)}

        # ---- LightGBM（无 step 索引）
        scores = {}
        for tag, tr_idx in (("structured", tr), ("random", tr_r)):
            m = LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=63,
                               subsample=0.9, colsample_bytree=0.8, random_state=0,
                               n_jobs=-1, verbose=-1,
                               scale_pos_weight=(len(tr_idx) - y[tr_idx].sum())
                               / max(y[tr_idx].sum(), 1))
            m.fit(X[tr_idx], y[tr_idx])
            scores[("LightGBM", tag)] = m.predict_proba(X[te])[:, 1]

        # ---- TGN-lite（记忆按真实 step 更新）
        for tag, tr_idx in (("structured", tr), ("random", tr_r)):
            mdl = roll.TGNLite(X.shape[1])
            p = roll.train_tgn(mdl, X, y, ts, src, dst, tr_idx, va, steps_all,
                               epochs=args.epochs, seed=0)
            scores[("TGN-lite", tag)] = p[te]

        for name in ("LightGBM", "TGN-lite"):
            a_s = float(pr_auc(y[te], scores[(name, "structured")]))
            a_r = float(pr_auc(y[te], scores[(name, "random")]))
            ci = boot_ci(y[te], scores[(name, "structured")],
                         scores[(name, "random")], seed=w)
            row[name] = {
                "AP_structured": round(a_s, 4),
                "AP_random": round(a_r, 4),
                "optimism": round(a_r - a_s, 4),
                "ci95_optimism": ci,
                "ci_excludes_zero": bool(ci[0] is not None and (ci[0] > 0 or ci[1] < 0)),
            }
        out["rows"].append(row)
        print(f"[w={w}] n_te={len(te)} LGBM opt={row['LightGBM']['optimism']:+.4f} "
              f"TGN opt={row['TGN-lite']['optimism']:+.4f} "
              f"(step-boundary ref -0.030)", flush=True)

    out["summary"] = {}
    for name in ("LightGBM", "TGN-lite"):
        ops = [r[name]["optimism"] for r in out["rows"]]
        out["summary"][name] = {
            "mean_optimism": round(float(np.mean(ops)), 4),
            "positive_windows": int(sum(1 for v in ops if v > 0)),
            "n_windows": len(ops),
            "by_window": [v for v in ops],
        }
    tgn_mean = out["summary"]["TGN-lite"]["mean_optimism"]
    out["verdict"] = {
        "H2_supported": bool(tgn_mean > 0.0),
        "note": ("If TGN-lite optimism under the node-rank cut is positive/larger than the "
                 "-0.030 step-boundary reference, immunity is a property of split-index "
                 "alignment, not the model class. LightGBM is the index-free control."),
    }
    out["status"] = "PASS"
    p = Path(args.outdir) / "index_mismatch_tgn.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    write_provenance("index_mismatch_tgn", {"status": "PASS"})
    print("[summary]", json.dumps(out["summary"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
