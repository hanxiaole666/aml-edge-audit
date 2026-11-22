"""
49_elliptic_v1_edge_protocol.py - E4：在 Elliptic（49 个真实时间步）上复验四条件
======================================================================
E2/E3 的证据都来自 Real-CATS 共现图，其边时间由端点 first-seen 时间派生。
审稿人从第一轮起要求在不依赖近似时间的数据上复验。Elliptic v1 提供真实的
交易时间步（1..49），是本项目唯一可获得的大规模真实时间戳 AML 图。

数据：elliptic_txs_features.csv（203,769 交易 × 166 列）
      elliptic_txs_classes.csv（1=illicit, 2=licit, unknown 未标注）
      elliptic_txs_edgelist.csv（234,355 条有向支付边）
    边本身无时间戳，但两端节点有真实时间步，故
        edge_time = max(t_u, t_v)
    是"边可存在的最早时间"的**下界中的上界**（边不可能早于两端出现），
    比 Real-CATS 的地址 first-seen 更严格。论文中如实标注这一点。

四个条件（与 E2 相同定义）：
    C1 keep-all        保留全部边
    C2 past-only       跨切分边仅在 edge_time <= 测试节点时间步时保留
    C3 remove-all      仅保留训练-训练边
    M2 random-matched  与 C2 等量的随机边

切分：按时间步切 5 个 cutoff（30/33/36/39/42），训练=cutoff 之前，
测试=cutoff 之后（跳过 1 个时间步的 embargo）。每条件报 bootstrap 95% CI
与配对 bootstrap。

模型：LogisticRegression(C=0.1) on 2-hop symmetric-normalized smoothing，
特征取 Elliptic 的 93 维 local 特征（不含 72 维 aggregated，后者含邻居
聚合信息，会与边条件混淆）。另报 LightGBM on raw local features 作为
不依赖图的参照。

用法：python scripts/49_elliptic_v1_edge_protocol.py
产出：results/elliptic_v1_edge_protocol.json + figures/elliptic_v1_protocol.pdf
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA_DIR, RESULTS_DIR, pr_auc, write_provenance, set_seed, random_pair_edges


def _ap(y, s):
    order = np.argsort(-s)
    y = y[order]
    prec = np.cumsum(y) / (np.arange(len(y)) + 1)
    return float(np.sum(prec * y) / max(y.sum(), 1))


def boot_ci(y, s, n_boot=400, seed=0):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        vals.append(_ap(y[idx], s[idx]))
    if not vals:
        return [None, None]
    return [round(float(np.percentile(vals, 2.5)), 4),
            round(float(np.percentile(vals, 97.5)), 4)]


def paired_ci(y, sa, sb, n_boot=400, seed=0):
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        d.append(_ap(y[idx], sa[idx]) - _ap(y[idx], sb[idx]))
    if not d:
        return {"diff": None, "ci95": [None, None], "significant": False}
    d = np.array(d)
    lo, hi = np.percentile(d, [2.5, 97.5])
    return {"diff": round(float(d.mean()), 4),
            "ci95": [round(float(lo), 4), round(float(hi), 4)],
            "significant": bool(lo > 0 or hi < 0)}


def load():
    d = DATA_DIR / "elliptic_v1"
    feat = pd.read_csv(d / "elliptic_txs_features.csv", header=None,
                       dtype={0: np.int64, 1: np.int64})
    # 列 0=txId, 1=time step, 2..94=local 93 维, 95..166=aggregated 72 维
    tx = feat.iloc[:, 0].values.astype(np.int64)
    ts = feat.iloc[:, 1].values.astype(np.int64)
    X = feat.iloc[:, 2:95].values.astype(np.float32)
    cls = pd.read_csv(d / "elliptic_txs_classes.csv")
    lab = dict(zip(cls.txId.values.astype(np.int64), cls["class"].values))
    y = np.array([1 if lab.get(int(t)) == "1" else (0 if lab.get(int(t)) == "2" else -1)
                  for t in tx], dtype=np.int64)
    el = pd.read_csv(d / "elliptic_txs_edgelist.csv")
    id2idx = {int(t): i for i, t in enumerate(tx)}
    src = np.array([id2idx[int(a)] for a in el.txId1.values if int(a) in id2idx], dtype=np.int64)
    dst = np.array([id2idx[int(b)] for b in el.txId2.values if int(b) in id2idx], dtype=np.int64)
    # 过滤：两端都在索引内
    keep = np.array([int(a) in id2idx and int(b) in id2idx
                     for a, b in zip(el.txId1.values, el.txId2.values)])
    src = np.array([id2idx[int(a)] for a in el.txId1.values[keep]], dtype=np.int64)
    dst = np.array([id2idx[int(b)] for b in el.txId2.values[keep]], dtype=np.int64)
    return tx, ts, X, y, src, dst


def _sym(src, dst, n):
    """无向化后的稀疏邻接（去重）。"""
    r = np.concatenate([src, dst])
    c = np.concatenate([dst, src])
    A = sp.coo_matrix((np.ones(len(r), dtype=np.float32), (r, c)), shape=(n, n)).tocsr()
    A.data[:] = 1.0
    return A


def build_adj(n, src, dst, ets, ts, tr_set, te_set, condition, rng, target_edges=None):
    if condition == "C1":
        return _sym(src, dst, n)
    if condition == "C2":
        in_tr = np.array([s in tr_set for s in src]) & np.array([d in tr_set for d in dst])
        cross = np.array([(s in tr_set) != (d in tr_set) for s, d in zip(src, dst)])
        # 跨切分边：只有 edge_time <= 测试端时间步才保留
        test_end = np.array([(d if s in tr_set else s) for s, d in zip(src, dst)])
        past = cross & (ets <= ts[test_end])
        keep = in_tr | past
        return _sym(src[keep], dst[keep], n)
    if condition == "C3":
        in_tr = np.array([s in tr_set for s in src]) & np.array([d in tr_set for d in dst])
        return _sym(src[in_tr], dst[in_tr], n)
    if condition == "M2":
        e = random_pair_edges(n, target_edges, rng)
        return _sym(e[:, 0], e[:, 1], n)
    raise ValueError(condition)


def propagate(A, Xs):
    n = A.shape[0]
    Ahat = (A + sp.eye(n, dtype=np.float32, format="csr")).tocsr()
    deg = np.asarray(Ahat.sum(axis=1)).ravel()
    dinv = 1.0 / np.sqrt(np.maximum(deg, 1e-9))
    D = sp.diags(dinv.astype(np.float32))
    P = (D @ Ahat @ D).tocsr()
    return P @ (P @ Xs)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", default=str(RESULTS_DIR))
    args = ap.parse_args()
    set_seed(0)

    tx, ts, X, y, src, dst = load()
    lab_mask = y >= 0
    print(f"[data] tx={len(tx)} edges={len(src)} labelled={lab_mask.sum()} "
          f"illicit={(y == 1).sum()} licit={(y == 0).sum()} steps={ts.min()}..{ts.max()}",
          flush=True)
    # edge time = max of endpoint steps（边不可能早于两端出现）
    ets = np.maximum(ts[src], ts[dst]).astype(np.float32)

    out = {"experiment": "elliptic_v1_edge_protocol",
           "n_tx": int(len(tx)), "n_edges": int(len(src)),
           "n_labelled": int(lab_mask.sum()),
           "illicit": int((y == 1).sum()), "licit": int((y == 0).sum()),
           "edge_time_definition": "max(time_step(u), time_step(v))",
           "model": "LogisticRegression(C=0.1) on 2-hop symmetric-normalized smoothing, 93 local features",
           "cutoffs": [30, 33, 36, 39, 42], "splits": [], "status": "PASS"}

    scores_all = []
    for cutoff in out["cutoffs"]:
        tr = np.where(lab_mask & (ts <= cutoff - 1))[0]          # embargo: 跳过 cutoff 本身
        te = np.where(lab_mask & (ts >= cutoff + 1))[0]
        if len(tr) < 200 or len(te) < 200:
            print(f"  cutoff={cutoff}: skipped (n_tr={len(tr)}, n_te={len(te)})")
            continue
        tr_set, te_set = set(tr.tolist()), set(te.tolist())
        rng = np.random.default_rng(cutoff)
        sc = StandardScaler().fit(X[tr])
        Xs = sc.transform(X).astype(np.float32)
        A_c2 = build_adj(len(tx), src, dst, ets, ts, tr_set, te_set, "C2", rng)
        n_c2 = int(A_c2.nnz / 2)
        conds = {
            "C1": build_adj(len(tx), src, dst, ets, ts, tr_set, te_set, "C1", rng),
            "C2": A_c2,
            "C3": build_adj(len(tx), src, dst, ets, ts, tr_set, te_set, "C3", rng),
            "M2": build_adj(len(tx), src, dst, ets, ts, tr_set, te_set, "M2", rng, target_edges=n_c2),
        }
        rows = {}
        sc_out = {}
        for name, A in conds.items():
            H = propagate(A, Xs)
            clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=0.1)
            clf.fit(H[tr], y[tr])
            s = clf.predict_proba(H[te])[:, 1]
            rows[name] = {"AP": round(float(pr_auc(y[te], s)), 4),
                          "n_edges": int(A.nnz / 2),
                          "CI95": boot_ci(y[te], s, seed=cutoff)}
            sc_out[name] = s
        # 不依赖图的 LightGBM 参照
        from lightgbm import LGBMClassifier
        lgb = LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=31,
                             random_state=0, n_jobs=-1, verbose=-1,
                             scale_pos_weight=(len(tr) - y[tr].sum()) / max(y[tr].sum(), 1))
        lgb.fit(X[tr], y[tr])
        rows["LightGBM-nograph"] = {"AP": round(float(pr_auc(y[te], lgb.predict_proba(X[te])[:, 1])), 4),
                                    "CI95": boot_ci(y[te], lgb.predict_proba(X[te])[:, 1], seed=cutoff)}
        split = {"cutoff": cutoff, "n_train": int(len(tr)), "n_test": int(len(te)),
                 "test_prevalence": round(float(y[te].mean()), 4), "conditions": rows,
                 "paired": {"C2_vs_C3": paired_ci(y[te], sc_out["C2"], sc_out["C3"], seed=cutoff),
                            "C2_vs_M2": paired_ci(y[te], sc_out["C2"], sc_out["M2"], seed=cutoff),
                            "C1_vs_C2": paired_ci(y[te], sc_out["C1"], sc_out["C2"], seed=cutoff)}}
        out["splits"].append(split)
        scores_all.append(split)
        print(f"  cutoff={cutoff} n_te={len(te)} prev={y[te].mean():.3f} "
              + " ".join(f"{k}={v['AP']:.3f}" for k, v in rows.items()), flush=True)

    agg = {}
    for c in ["C1", "C2", "C3", "M2", "LightGBM-nograph"]:
        vals = [s["conditions"][c]["AP"] for s in scores_all if c in s["conditions"]]
        agg[c] = {"mean": round(float(np.mean(vals)), 4),
                  "std": round(float(np.std(vals, ddof=1)), 4) if len(vals) > 1 else 0.0,
                  "n_splits": len(vals)}
    out["aggregate"] = agg
    order_ok = all(s["conditions"]["C1"]["AP"] < s["conditions"]["C2"]["AP"]
                   < s["conditions"]["C3"]["AP"] for s in scores_all
                   if all(k in s["conditions"] for k in ["C1", "C2", "C3"]))
    out["verdict"] = {
        "C1_lt_C2_lt_C3_all_splits": bool(order_ok),
        "C2_gt_M2_all_splits": bool(all(
            s["conditions"]["C2"]["AP"] > s["conditions"]["M2"]["AP"]
            for s in scores_all if "M2" in s["conditions"])),
        "interpretation": ("The Real-CATS ordering replicates on real time steps."
                           if order_ok else
                           "The ordering does not hold on real time steps; the Real-CATS "
                           "result is specific to derived first-seen times."),
    }
    print("\n[aggregate]", agg)
    print("[verdict]", out["verdict"])

    # 图
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figdir = Path(__file__).resolve().parent.parent / "submission_tdsc" / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "font.family": "serif", "figure.dpi": 200})
    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    xs = [s["cutoff"] for s in scores_all]
    for c, mk in [("C1", "o"), ("C2", "s"), ("C3", "^"), ("M2", "x")]:
        ax.plot(xs, [s["conditions"][c]["AP"] for s in scores_all], mk + "-", ms=4, lw=1.0, label=c)
    ax.set_xlabel("Training cutoff (time step)")
    ax.set_ylabel("PR-AUC on future steps")
    ax.legend(fontsize=7, ncol=4)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(figdir / "elliptic_v1_protocol.pdf", bbox_inches="tight")
    plt.close(fig)

    p = Path(args.outdir) / "elliptic_v1_edge_protocol.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    write_provenance("elliptic_v1_edge_protocol", {"status": "PASS", "n_splits": len(scores_all)})
    print(f"[done] {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
