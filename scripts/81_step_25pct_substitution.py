"""
81_step_25pct_substitution.py - P0 缺失对照格：step-boundary 下的 25% 节点置换
==============================================================================
v15 的 Table XI 在 windowed node-rank cut 下用"25% 训练侧节点置换"量出
TGN-lite optimism +0.075、LightGBM +0.131。参照的 -0.030 来自 step-boundary
下的完整随机切分（100% 干预）。两者干预强度不同，符号翻转混入了强度效应。
本脚本在 step-boundary 切分下做【同样的 25% 节点置换】，把干预强度固定，
只让切分算子变化，从而分离"索引错配"与"干预更强"。

对照设计（与 Table IX 同一切分、与 Table XI 同一干预）：
  structured: tr = t<=c-2, va = t==c-1, embargo t==c, te = c+1..c+4
  substituted: 从 tr 随机换出 1/4，从 te 换入 1/4（测试集 te 固定，va 固定）
  optimism = AP(substituted) - AP(structured)

两个 read-out：LightGBM（索引无关基线）、TGN-lite（时间索引模型）。
用与 Table IX 相同的 5 个非退化 cutoff（22/26/30/34/38）+ 退化 42 供核对。

用法：python scripts/81_step_25pct_substitution.py [--epochs 40]
产出：results/step_25pct_substitution.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch
from lightgbm import LGBMClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RESULTS_DIR, pr_auc, set_seed, write_provenance

_spec = importlib.util.spec_from_file_location(
    "rolling", Path(__file__).resolve().parent / "55_elliptic_rolling_matrix.py")
roll = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(roll)

torch.set_num_threads(8)
CUTOFFS = (22, 26, 30, 34, 38, 42)
TEST_SPAN = 4


def step_split(ts, y, c):
    tr = np.where((y >= 0) & (ts <= c - 2))[0]
    va = np.where((y >= 0) & (ts == c - 1))[0]
    te = np.where((y >= 0) & (ts >= c + 1) & (ts <= c + TEST_SPAN))[0]
    return tr, va, te


def substitute(tr, te, val, seed):
    """把 1/4 训练节点换成测试侧节点；测试集与验证集固定。"""
    rng = np.random.default_rng(seed)
    n_swap = min(len(te) // 4, len(tr) // 4)
    out_idx = rng.choice(tr, size=n_swap, replace=False)
    in_idx = rng.choice(te, size=n_swap, replace=False)
    tr_sub = np.concatenate([np.setdiff1d(tr, out_idx), in_idx])
    tr_sub = np.setdiff1d(tr_sub, val)
    return tr_sub, n_swap


def boot_ci(y, s_t, s_r, seed=0, n_boot=300):
    """测试集 bootstrap：optimism = AP(random) - AP(structured)。"""
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) < 2:
            continue
        d.append(pr_auc(y[i], s_r[i]) - pr_auc(y[i], s_t[i]))
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
    steps_all = list(range(1, int(ts.max()) + 1))

    out = {
        "experiment": "step_25pct_substitution",
        "dataset": "Elliptic v1 (real steps)",
        "purpose": "fix intervention strength at 25% and vary only the split operator; "
                   "the matching node-rank 25% condition is in index_mismatch_tgn.json",
        "split": "step boundary: tr=t<=c-2, val=t==c-1, embargo=t==c, te=c+1..c+4",
        "intervention": "substitute one quarter of training nodes with test-side nodes "
                        "(node identity substitution, not label randomisation); "
                        "test set and validation set held fixed",
        "comparison_reference": {
            "step_full_random": {"LightGBM": 0.038, "TGN-lite": -0.030},
            "noderank_25pct": {"LightGBM": 0.131, "TGN-lite": 0.075},
        },
        "readouts": ["LightGBM", "TGN-lite"],
        "rows": [],
    }

    for c in CUTOFFS:
        tr, va, te = step_split(ts, y, c)
        if len(tr) < 500 or len(va) < 100 or len(te) < 200 or y[tr].sum() < 30:
            print(f"[skip] c={c} tr={len(tr)} va={len(va)} te={len(te)}", flush=True)
            continue
        tr_sub, n_swap = substitute(tr, te, va, seed=c)

        scores = {}
        for name in ("LightGBM", "TGN-lite"):
            for tag, tr_idx in (("structured", tr), ("substituted", tr_sub)):
                if name == "LightGBM":
                    m = LGBMClassifier(
                        n_estimators=300, learning_rate=0.05, num_leaves=63,
                        subsample=0.9, colsample_bytree=0.8, random_state=0,
                        n_jobs=-1, verbose=-1,
                        scale_pos_weight=(len(tr_idx) - y[tr_idx].sum())
                        / max(y[tr_idx].sum(), 1))
                    m.fit(X[tr_idx], y[tr_idx])
                    scores[(name, tag)] = m.predict_proba(X[te])[:, 1]
                else:
                    mdl = roll.TGNLite(X.shape[1])
                    p = roll.train_tgn(mdl, X, y, ts, src, dst, tr_idx, va,
                                       steps_all, epochs=args.epochs, seed=0)
                    scores[(name, tag)] = p[te]

        row = {"cutoff": c, "n_train": int(len(tr)), "n_val": int(len(va)),
               "n_test": int(len(te)), "n_substituted": int(n_swap),
               "degenerate": bool(y[te].mean() < 0.05)}
        for name in ("LightGBM", "TGN-lite"):
            a_s = float(pr_auc(y[te], scores[(name, "structured")]))
            a_r = float(pr_auc(y[te], scores[(name, "substituted")]))
            ci = boot_ci(y[te], scores[(name, "structured")],
                         scores[(name, "substituted")], seed=c)
            row[name] = {
                "AP_structured": round(a_s, 4),
                "AP_substituted": round(a_r, 4),
                "optimism": round(a_r - a_s, 4),
                "ci95_optimism": ci,
            }
        out["rows"].append(row)
        print(f"[c={c}] n_swap={n_swap} LGBM opt={row['LightGBM']['optimism']:+.4f} "
              f"CI={row['LightGBM']['ci95_optimism']} "
              f"TGN opt={row['TGN-lite']['optimism']:+.4f} "
              f"CI={row['TGN-lite']['ci95_optimism']}", flush=True)

    good = [r for r in out["rows"] if not r["degenerate"]]
    out["summary"] = {}
    for name in ("LightGBM", "TGN-lite"):
        ops = [r[name]["optimism"] for r in good]
        out["summary"][name] = {
            "mean_optimism": round(float(np.mean(ops)), 4),
            "positive_cutoffs": int(sum(1 for v in ops if v > 0)),
            "n_cutoffs": len(ops),
        }
    lg = out["summary"]["LightGBM"]["mean_optimism"]
    tg = out["summary"]["TGN-lite"]["mean_optimism"]
    out["architecture_gap"] = {
        "step_25pct": round(lg - tg, 4),
        "step_full_random": round(0.038 - (-0.030), 4),
        "noderank_25pct": round(0.131 - 0.075, 4),
    }
    out["verdict"] = {
        "note": "With intervention strength fixed at 25%, compare step vs node-rank. "
                "TGN's advantage over the tree under each condition is the "
                "architecture contribution; the sign of each model's own optimism "
                "is the split contribution.",
    }
    out["status"] = "PASS"
    p = Path(args.outdir) / "step_25pct_substitution.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    write_provenance("step_25pct_substitution", {"status": "PASS"})
    print("[summary]", json.dumps(out["summary"], ensure_ascii=False))
    print("[architecture_gap]", json.dumps(out["architecture_gap"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
