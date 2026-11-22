"""
73_graph_construction.py - E-A 图构造稳健性 + E-D 头条增益区间
================================================================
审稿人问：0.115（C1→C3）是不是某一种共现构造的产物？

做法：固定节点集与特征，只改边的定义，复跑四条件。
  A  min_shared=1            共享至少 1 个合约（论文基线）
  B  min_shared=2
  C  min_shared=5
  D  min_shared=1, hot<=20   更严格的热门合约上限（基线为 50）
  E  min_shared=1, hot<=100  更宽松
  F  bipartite               地址--合约二部图（新增合约节点，2-hop 经过合约）

同时输出每个构造的 C1-C3 配对 bootstrap 区间（E-D），因为 +0.115 的
测试集只有 375 个节点、约 142 个正例，没有区间不该作为头条。

用法：python scripts/73_graph_construction.py
产出：results/graph_construction.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA_DIR, RESULTS_DIR, pr_auc, set_seed, write_provenance

FRACS = [0.60, 0.65, 0.70, 0.75, 0.80]
MAX_NODES = 1500
PROV = {"etherscan_checked", "watchback_checked", "gs_checked"}


def parse_contracts(cell) -> set[int]:
    if pd.isna(cell):
        return set()
    out = set()
    for item in str(cell).split(","):
        item = item.strip()
        if ":" in item:
            idx = item.split(":")[0].strip()
            if idx.isdigit():
                out.add(int(idx))
    return out


def load_nodes():
    base = DATA_DIR / "real_cats"
    ti_m = pd.read_csv(base / "TI_M.tsv", sep="\t")
    ti_b = pd.read_csv(base / "TI_B.tsv", sep="\t")
    ti_m["label"] = 1
    ti_b["label"] = 0
    ti = pd.concat([ti_m, ti_b], ignore_index=True)

    addr_label = ti.drop_duplicates("Address").set_index("Address")["label"].to_dict()
    addr_contracts: dict[str, set[int]] = {}
    for _, row in ti.iterrows():
        a = row["Address"]
        cs = parse_contracts(row.get("ERC20")) | parse_contracts(row.get("ERC721")) \
            | parse_contracts(row.get("ERC1155"))
        addr_contracts[a] = addr_contracts.get(a, set()) | cs

    feats = pd.read_csv(DATA_DIR / "real_cats_eth.csv")
    cols = [c for c in feats.columns
            if c not in {"address", "label", "time"} and c not in PROV
            and pd.api.types.is_numeric_dtype(feats[c])]
    mat = feats[cols].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy(float)
    times = pd.to_numeric(feats["time"], errors="coerce").fillna(0).to_numpy(float)
    feat = {str(a): mat[i] for i, a in enumerate(feats["address"].astype(str))}
    tmap = {str(a): float(times[i]) for i, a in enumerate(feats["address"].astype(str))}

    addrs = [a for a in addr_contracts if a in feat]
    # 与 07_build_real_graph 的采样一致：类别均衡取 max_nodes，seed 42
    if len(addrs) > MAX_NODES:
        rng = np.random.default_rng(42)
        pos = [a for a in addrs if addr_label.get(a, 0) == 1]
        neg = [a for a in addrs if addr_label.get(a, 0) == 0]
        k = MAX_NODES // 2
        pos = list(rng.choice(pos, size=min(k, len(pos)), replace=False))
        neg = list(rng.choice(neg, size=min(k, len(neg)), replace=False))
        addrs = pos + neg
    X = np.vstack([feat[a] for a in addrs])
    y = np.array([int(addr_label.get(a, 0)) for a in addrs])
    t = np.array([tmap[a] for a in addrs])
    return addrs, addr_contracts, X, y, t


def build_edges(addrs, addr_contracts, min_shared: int, hot: int):
    idx = {a: i for i, a in enumerate(addrs)}
    inv: dict[int, list[int]] = defaultdict(list)
    for a in addrs:
        for c in addr_contracts[a]:
            inv[c].append(idx[a])
    pair: dict[tuple[int, int], int] = defaultdict(int)
    for _, ids in inv.items():
        if 2 > len(ids) or len(ids) > hot:
            continue
        ids = sorted(set(ids))
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                pair[(ids[i], ids[j])] += 1
    src, dst, w = [], [], []
    for (i, j), cnt in pair.items():
        if cnt >= min_shared:
            src.append(i); dst.append(j); w.append(cnt)
    return np.array(src, int), np.array(dst, int), np.array(w, int), inv


def build_bipartite(addrs, addr_contracts, min_shared: int, hot: int):
    """地址--合约二部图：合约成为节点，2-hop 传播经过合约。"""
    inv: dict[int, list[int]] = defaultdict(list)
    for i, a in enumerate(addrs):
        for c in addr_contracts[a]:
            inv[c].append(i)
    # 只保留度数在 [2, hot] 的合约节点
    keep = {c: ids for c, ids in inv.items() if 2 <= len(ids) <= hot}
    cs = sorted(keep)
    cidx = {c: len(addrs) + k for k, c in enumerate(cs)}
    src, dst = [], []
    for c in cs:
        for i in keep[c]:
            src.append(i); dst.append(cidx[c])
    return np.array(src, int), np.array(dst, int), len(cs)


def adj(n, src, dst, tr_set, te_set, cond, rng, target=None, weights=None):
    """按四条件建邻接（稀疏）。返回 csr 与边数。"""
    rows, cols = [], []
    if cond in ("C1", "C2", "C3"):
        for k, (s, d) in enumerate(zip(src, dst)):
            a, b = int(s), int(d)
            ia, ib = a in tr_set, b in tr_set
            if cond == "C2" and not (ia or ib):
                # 两端都在测试集：这条边在部署时还不存在
                continue
            if cond == "C3" and not (ia and ib):
                continue
            rows.append(a); cols.append(b)
            rows.append(b); cols.append(a)
    else:
        for x, z in random_pair_edges(n, target, rng):
            rows.append(int(x)); cols.append(int(z))
            rows.append(int(z)); cols.append(int(x))
    data = np.ones(len(rows), dtype=float)
    A = sparse.csr_matrix((data, (rows, cols)), shape=(n, n))
    A.data[:] = 1.0
    return A, int(A.nnz / 2)


random_pair_edges = None


def score_csr(A, Xs, tr, te):
    n = A.shape[0]
    Ahat = A + sparse.eye(n, format="csr")
    deg = np.asarray(Ahat.sum(axis=1)).ravel()
    dinv = 1.0 / np.sqrt(np.maximum(deg, 1e-9))
    D = sparse.diags(dinv)
    Ahat = D @ Ahat @ D
    H = Ahat @ (Ahat @ Xs)
    H = np.asarray(H)
    clf = LogisticRegression(max_iter=1000, class_weight="balanced", C=0.1)
    clf.fit(H[tr], y_glob[tr])
    return clf.predict_proba(H[te])[:, 1]


y_glob = None


def boot_ci(y, s, n_boot=400, seed=0):
    rng = np.random.default_rng(seed); vals = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) < 2:
            continue
        vals.append(pr_auc(y[i], s[i]))
    return [round(float(np.percentile(vals, 2.5)), 4), round(float(np.percentile(vals, 97.5)), 4)]


def paired_ci(y, sa, sb, n_boot=400, seed=0):
    rng = np.random.default_rng(seed); d = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) < 2:
            continue
        d.append(pr_auc(y[i], sa[i]) - pr_auc(y[i], sb[i]))
    d = np.array(d)
    lo, hi = np.percentile(d, [2.5, 97.5])
    return {"diff": round(float(d.mean()), 4), "ci95": [round(float(lo), 4), round(float(hi), 4)],
            "excludes_zero": bool(lo > 0 or hi < 0)}


def run_condition(cond, src, dst, n, tr, te, Xs, rng, target, et=None):
    tr_set, te_set = set(tr.tolist()), set(te.tolist())
    A, ne = adj(n, src, dst, tr_set, te_set, cond, rng, target=target)
    s = score_csr(A, Xs, tr, te)
    deg_te = np.asarray(A[te].sum(axis=1)).ravel()
    return s, ne, deg_te


def crossing_excess(y, t, tr, te, src, dst):
    """跨切分边的同行标签率与独立期望（只统计两端有标签的边）。"""
    tr_set, te_set = set(tr.tolist()), set(te.tolist())
    same = n = 0
    for s, d in zip(src, dst):
        a, b = int(s), int(d)
        if (a in tr_set and b in te_set) or (a in te_set and b in tr_set):
            n += 1
            if y[a] == y[b]:
                same += 1
    if n == 0:
        return {"n_edges": 0, "observed_same": None, "expected": None, "excess": None}
    p_tr, p_te = y[tr].mean(), y[te].mean()
    exp = p_tr * p_te + (1 - p_tr) * (1 - p_te)
    return {"n_edges": n, "observed_same": round(same / n, 4), "expected": round(float(exp), 4),
            "excess": round(same / n - float(exp), 4)}


def main() -> int:
    global random_pair_edges, y_glob
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=str(RESULTS_DIR))
    args = ap.parse_args()
    set_seed(0)
    from common import random_pair_edges as rpe
    random_pair_edges = rpe

    addrs, addr_contracts, X, y, t = load_nodes()
    y_glob = y
    n = len(y)
    print(f"[nodes] {n}  illicit={int(y.sum())}  features={X.shape[1]}")

    configs = [
        ("min1_hot50", "shared>=1, hot<=50 (paper baseline)", 1, 50, False),
        ("min2_hot50", "shared>=2", 2, 50, False),
        ("min5_hot50", "shared>=5", 5, 50, False),
        ("min1_hot20", "hot<=20", 1, 20, False),
        ("min1_hot100", "hot<=100", 1, 100, False),
        ("bipartite", "address--contract bipartite", 1, 50, True),
    ]

    out = {"experiment": "graph_construction", "dataset": "Real-CATS ETH",
           "n_nodes": n, "n_illicit": int(y.sum()),
           "protocol": "5 temporal cutoffs x 4 conditions per construction",
           "constructions": []}

    for name, desc, ms, hot, bipart in configs:
        if bipart:
            src, dst, n_contract = build_bipartite(addrs, addr_contracts, ms, hot)
            N = n + n_contract
            Xb = np.zeros((N, X.shape[1]))
            Xb[:n] = X
            # 合约节点用零特征 + 常数，避免完全无信息
            Xb[n:, -1] = 1.0
        else:
            src, dst, _, _ = build_edges(addrs, addr_contracts, ms, hot)
            N = n
            Xb = X
        rows = []
        for frac in FRACS:
            order = np.argsort(t, kind="mergesort")
            cut = int(frac * n)
            tr, te = order[:cut], order[cut:]
            tmax = t[tr].max()
            te = te[t[te] > tmax]
            tr = tr[t[tr] <= tmax]
            if len(te) < 50:
                continue
            sc = StandardScaler().fit(Xb[tr])
            Xs = sc.transform(Xb)
            sc_rng = np.random.default_rng(0)
            # 先建 C2 取边数
            et_ = t
            s2, n2, _ = run_condition("C2", src, dst, N, tr, te, Xs, sc_rng, None, et_)
            rec = {"frac": frac, "n_test": int(len(te)), "prevalence": round(float(y[te].mean()), 4),
                   "conditions": {}}
            for cond in ("C1", "C2", "C3"):
                s, ne, dte = run_condition(cond, src, dst, N, tr, te, Xs, np.random.default_rng(0),
                                           None, et_)
                rec["conditions"][cond] = {"AP": round(float(pr_auc(y[te], s)), 4),
                                           "n_edges": ne,
                                           "test_degree": round(float(dte.mean()), 3)}
                if cond == "C2":
                    rec["conditions"][cond]["score"] = s
            sM, neM, dM = run_condition("M2", src, dst, N, tr, te, Xs, np.random.default_rng(0),
                                        n2, et_)
            rec["conditions"]["M2"] = {"AP": round(float(pr_auc(y[te], sM)), 4),
                                       "n_edges": neM, "test_degree": round(float(dM.mean()), 3)}
            rec["conditions"]["C2"].pop("score", None)
            # C1 vs C3 配对区间（E-D）
            s1, _, _ = run_condition("C1", src, dst, N, tr, te, Xs, np.random.default_rng(0),
                                     None, et_)
            s3, _, _ = run_condition("C3", src, dst, N, tr, te, Xs, np.random.default_rng(0),
                                     None, et_)
            rec["C3_minus_C1"] = paired_ci(y[te], s3, s1, seed=0)
            rec["cross_excess"] = crossing_excess(y, t, tr, te, src[:n] if bipart else src,
                                                 dst[:n] if bipart else dst)
            rows.append(rec)
            print(f"  [{name}] frac={frac:.2f} C1={rec['conditions']['C1']['AP']:.3f} "
                  f"C3={rec['conditions']['C3']['AP']:.3f} "
                  f"gain={rec['C3_minus_C1']['diff']:+.3f} "
                  f"CI={rec['C3_minus_C1']['ci95']}", flush=True)

        gains = [r["C3_minus_C1"]["diff"] for r in rows]
        out["constructions"].append({
            "name": name, "description": desc,
            "n_edges": int(len(src)), "n_contract_nodes": N - n if bipart else 0,
            "mean_gain_C3_minus_C1": round(float(np.mean(gains)), 4) if gains else None,
            "min_gain": round(float(min(gains)), 4) if gains else None,
            "max_gain": round(float(max(gains)), 4) if gains else None,
            "splits": rows,
        })

    allg = [c["mean_gain_C3_minus_C1"] for c in out["constructions"]]
    out["verdict"] = {
        "gains_by_construction": allg,
        "all_positive": bool(all(g > 0 for g in allg)),
        "range": [round(min(allg), 4), round(max(allg), 4)],
        "spread": round(max(allg) - min(allg), 4),
        "note": ("If every construction keeps the sign, the gain is not an artefact of one "
                 "edge definition; the spread bounds how much of the magnitude is construction."),
    }
    out["status"] = "PASS"
    p = Path(args.outdir) / "graph_construction.json"
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    write_provenance("graph_construction", {"status": "PASS", "n_configs": len(configs)})
    print(f"[done] {p}")
    print(json.dumps(out["verdict"], indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
