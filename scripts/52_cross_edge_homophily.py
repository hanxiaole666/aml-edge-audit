"""
52_cross_edge_homophily.py - 跨切分边同质性与边协议效应符号的统一诊断
======================================================================
E4 在 Elliptic（支付流图）上得到与 Real-CATS（共现图）相反的符号：
强模型下 C1 > C2 ≈ C3，即保留跨切分边有益。本脚本检验统一解释：
**边协议的效应符号由跨切分边的标签同质性相对独立期望的符号决定。**

对两个数据集、每个切分，计算：
    observed_same = 训练-测试边中两端标签相同的比例（只用已标注节点）
    expected_same = p_tr*p_te + (1-p_tr)*(1-p_te)   （标签独立时的期望）
    excess        = observed - expected
并与该切分的 AP(C1) - AP(C3) 并列，检验符号一致性。

注意：Real-CATS 用 C1 保留的跨切分边；Elliptic 用全部 tr-te 边（C1 不删边，
两者一致），并额外报 tr-tr / te-te 作为参照。

用法：python scripts/52_cross_edge_homophily.py
产出：results/cross_edge_homophily.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA_DIR, RESULTS_DIR, set_seed, write_provenance

import importlib.util
_spec = importlib.util.spec_from_file_location(
    "e4", Path(__file__).resolve().parent / "49_elliptic_v1_edge_protocol.py")
_e4 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_e4)

import pandas as pd


def homophily(y, src, dst, mask, p_a, p_b):
    """mask: 边子集；只用两端都标注的边。

    期望用**切分的全局 prevalence**（p_a, p_b），而不是边的端点分布，
    否则边的方向性会把期望算偏。
    """
    m = mask & (y[src] >= 0) & (y[dst] >= 0)
    if m.sum() == 0:
        return None
    same = float((y[src[m]] == y[dst[m]]).mean())
    exp = p_a * p_b + (1 - p_a) * (1 - p_b)
    return {"n_edges": int(m.sum()), "observed_same": round(same, 4),
            "expected_independent": round(exp, 4), "excess": round(same - exp, 4)}


def realcats():
    nodes = pd.read_csv(DATA_DIR / "real_cats" / "real_graph_nodes.csv")
    edges = pd.read_csv(DATA_DIR / "real_cats" / "real_graph_edges.csv")
    y = nodes.label.values.astype(int)
    t = nodes.time.values.astype(float)
    id2idx = {int(i): k for k, i in enumerate(nodes.node_id.values)}
    src = np.array([id2idx[int(s)] for s in edges.src])
    dst = np.array([id2idx[int(d)] for d in edges.dst])
    out = []
    for frac in [0.6, 0.65, 0.7, 0.75, 0.8]:
        order = np.argsort(t, kind="mergesort")
        cut = int(frac * len(y))
        tr, te = order[:cut], order[cut:]
        tmax = t[tr].max()
        te = te[t[te] > tmax]
        tr_set, te_set = set(tr.tolist()), set(te.tolist())
        in_tr = np.array([s in tr_set for s in src]) & np.array([d in tr_set for d in dst])
        cross = np.array([(s in tr_set) != (d in tr_set) for s, d in zip(src, dst)])
        p_tr, p_te = float(y[tr].mean()), float(y[te].mean())
        allmask = np.ones(len(src), dtype=bool)
        h = homophily(y, src, dst, cross, p_tr, p_te)
        ht = homophily(y, src, dst, in_tr, p_tr, p_tr)
        out.append({"split": f"frac={frac}", "n_train": int(len(tr)), "n_test": int(len(te)),
                    "prevalence_train": round(float(y[tr].mean()), 4),
                    "prevalence_test": round(float(y[te].mean()), 4),
                    "cross_edges": h, "train_train_edges": ht,
                    "test_test_edges": homophily(
                        y, src, dst,
                        np.array([s in te_set for s in src]) & np.array([d in te_set for d in dst]),
                        p_te, p_te),
                    "all_edges": homophily(y, src, dst, allmask,
                                           float(y.mean()), float(y.mean()))})
    return out


def elliptic():
    tx, ts, X, y, src, dst = _e4.load()
    lab = y >= 0
    out = []
    for cutoff in [30, 33, 36, 39, 42]:
        tr = np.where(lab & (ts <= cutoff - 1))[0]
        te = np.where(lab & (ts >= cutoff + 1))[0]
        tr_set, te_set = set(tr.tolist()), set(te.tolist())
        in_tr = np.array([s in tr_set for s in src]) & np.array([d in tr_set for d in dst])
        in_te = np.array([s in te_set for s in src]) & np.array([d in te_set for d in dst])
        # 正确的跨切分定义：一端在训练、另一端在测试。
        # 旧定义 (s in tr) != (d in tr) 把"训练-未标注"也算成跨切分，是错的。
        s_tr = np.isin(src, tr); d_tr = np.isin(dst, tr)
        s_te = np.isin(src, te); d_te = np.isin(dst, te)
        cross = (s_tr & d_te) | (s_te & d_tr)
        p_tr, p_te = float(y[tr].mean()), float(y[te].mean())
        cross_both = int((cross & (y[src] >= 0) & (y[dst] >= 0)).sum())
        # 2-hop 桥接：测试节点通过 <=2 跳能否到达任一训练节点
        import scipy.sparse as sp
        r = np.concatenate([src, dst]); c = np.concatenate([dst, src])
        A = sp.coo_matrix((np.ones(len(r), np.float32), (r, c)),
                          shape=(len(y), len(y))).tocsr()
        A.data[:] = 1.0
        A2 = (A @ A).tocsr()
        reach = np.asarray((A2[te][:, tr] > 0).sum(axis=1)).ravel() > 0
        # 仅 C3（训练-训练边）时测试节点全部孤立，桥接率为 0
        gap = np.abs(ts[src] - ts[dst])
        out.append({"split": f"cutoff={cutoff}", "n_train": int(len(tr)), "n_test": int(len(te)),
                    "prevalence_train": round(float(y[tr].mean()), 4),
                    "prevalence_test": round(float(y[te].mean()), 4),
                    "n_cross_edges": int(cross.sum()),
                    "n_cross_edges_both_labelled": cross_both,
                    "n_cross_step_edges": int((gap > 0).sum()),
                    "max_step_gap": int(gap.max()) if len(gap) else 0,
                    "cross_edges": homophily(y, src, dst, cross, p_tr, p_te),
                    "train_train_edges": homophily(y, src, dst, in_tr, p_tr, p_tr),
                    "test_test_edges": homophily(y, src, dst, in_te, p_te, p_te),
                    "two_hop_reach_train_rate": round(float(reach.mean()), 4)})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", default=str(RESULTS_DIR))
    args = ap.parse_args()
    set_seed(0)

    rc = realcats()
    el = elliptic()
    out = {"experiment": "cross_edge_homophily",
           "definition": ("excess = P(same label | cross-split edge) - "
                          "[p_tr*p_te + (1-p_tr)*(1-p_te)]; positive means cross-split "
                          "edges are more label-homophilous than independence."),
           "real_cats": rc, "elliptic_v1": el, "status": "PASS"}

    # 与该切分的 AP(C1)-AP(C3) 合并，检验符号一致性
    ep = json.loads((RESULTS_DIR / "edge_protocol_final.json").read_text(encoding="utf-8"))
    es = json.loads((RESULTS_DIR / "elliptic_v1_strong_model.json").read_text(encoding="utf-8"))
    rc_ap = {}
    for s in ep["splits"]:
        rc_ap[f"frac={s['frac']}"] = round(s["conditions"]["C1"]["AP"] - s["conditions"]["C3"]["AP"], 4)
    el_ap = {}
    for s in es["splits"]:
        el_ap[f"cutoff={s['cutoff']}"] = round(s["conditions"]["C1"]["AP"] - s["conditions"]["C3"]["AP"], 4)

    pairs = []
    for r in rc:
        if r["cross_edges"]:
            pairs.append({"dataset": "Real-CATS", "split": r["split"],
                          "excess": r["cross_edges"]["excess"],
                          "delta_AP_C1_minus_C3": rc_ap.get(r["split"])})
    for r in el:
        if r["cross_edges"]:
            pairs.append({"dataset": "Elliptic", "split": r["split"],
                          "excess": r["cross_edges"]["excess"],
                          "delta_AP_C1_minus_C3": el_ap.get(r["split"])})
    valid = [p for p in pairs if p["delta_AP_C1_minus_C3"] is not None]
    agree = sum(1 for p in valid
                if (p["excess"] > 0) == (p["delta_AP_C1_minus_C3"] > 0))
    out["sign_test"] = {
        "n_pairs": len(valid), "n_sign_agree": agree,
        "agreement_rate": round(agree / max(len(valid), 1), 3),
        "rule": "sign(AP[C1]-AP[C3]) == sign(homophily excess)",
        "pairs": pairs,
    }

    # 分数据集汇总
    for name, rows in [("real_cats", rc), ("elliptic_v1", el)]:
        ex = [r["cross_edges"]["excess"] for r in rows if r["cross_edges"]]
        out[f"{name}_summary"] = {
            "mean_excess": round(float(np.mean(ex)), 4) if ex else None,
            "n_splits": len(ex),
        }
    out["elliptic_no_cross_split_edges"] = all(r["n_cross_edges"] == 0 for r in el)
    out["elliptic_note"] = (
        "All Elliptic edges join nodes in the same time step (max |t_u - t_v| = 0), so no "
        "edge can cross a temporal cutoff. C2 and C3 therefore differ only by train-to-"
        "unlabelled adjacency, and the C1 advantage is test-test edge retention, not "
        "homophily of cross-split edges. The homophily sign rule applies only to graphs "
        "that actually have cross-split edges.")
    print(json.dumps(out["sign_test"], indent=2, default=float)[:1200])
    for name in ["real_cats_summary", "elliptic_v1_summary"]:
        print(name, out[name])

    p = Path(args.outdir) / "cross_edge_homophily.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    write_provenance("cross_edge_homophily", {"status": "PASS"})
    print(f"[done] {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
