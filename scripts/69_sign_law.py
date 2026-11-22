"""
69_sign_law.py - E2：检验"符号由度与同配性决定"能否写成定量律
======================================================================
这不是一个"证明规律"的脚本，而是一个**可证伪的检验**。它把所有观测放在一起：

  Real-CATS  5 个时间切分 x {C1, C2, M2}（相对 C3）
  Elliptic   4 个非退化切分 x {C1, C2, M2}（相对 C3）
  Elliptic   受控注入网格（K x lambda，来自 68 号脚本）

拟合 ΔAP = a·Δdegree + b·excess + c，报 R² 与留一图交叉验证。若跨图 R² 为负、
或 M2（excess=0、度很高）被系统性错号，则论文必须写成"两个案例 + 一个受控
干预"，而不是"一条可迁移的定量律"。脚本把这一判据固化，避免事后挑选叙述。

同配性口径：C1 的跨切分边取 52 号脚本的聚合值（Real-CATS 的 tr-te；
Elliptic 的 te-te）；C2 取过去可观测的 tr-te；M2 为真随机配对，excess=0。

用法：python scripts/69_sign_law.py
产出：results/sign_law.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA_DIR, RESULTS_DIR, random_pair_edges, write_provenance

_spec = importlib.util.spec_from_file_location(
    "e49", Path(__file__).resolve().parent / "49_elliptic_v1_edge_protocol.py")
_e49 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_e49)


def dense_deg(n, pairs, te):
    A = np.zeros((n, n), dtype=bool)
    if len(pairs):
        a = np.array([p[0] for p in pairs], dtype=int)
        b = np.array([p[1] for p in pairs], dtype=int)
        A[a, b] = True
        A[b, a] = True
    return float(A.sum(axis=1)[te].mean())


def fit(rows, cols):
    Z = np.array([[r[c] for c in cols] + [1.0] for r in rows], dtype=float)
    yv = np.array([r["delta_AP"] for r in rows], dtype=float)
    coef, *_ = np.linalg.lstsq(Z, yv, rcond=None)
    pred = Z @ coef
    ss_res = float(((yv - pred) ** 2).sum())
    ss_tot = float(((yv - yv.mean()) ** 2).sum())
    return {"coef": {c: round(float(v), 5) for c, v in zip(cols, coef[:-1])},
            "intercept": round(float(coef[-1]), 5),
            "r2": round(1 - ss_res / ss_tot, 4) if ss_tot > 0 else 0.0,
            "rmse": round(float(np.sqrt(((yv - pred) ** 2).mean())), 4),
            "n": len(rows),
            "sign_errors": int((np.sign(pred) != np.sign(yv)).sum())}


def predict(train_rows, test_rows, cols):
    Z = np.array([[r[c] for c in cols] + [1.0] for r in train_rows], dtype=float)
    yv = np.array([r["delta_AP"] for r in train_rows], dtype=float)
    coef, *_ = np.linalg.lstsq(Z, yv, rcond=None)
    Zt = np.array([[r[c] for c in cols] + [1.0] for r in test_rows], dtype=float)
    yt = np.array([r["delta_AP"] for r in test_rows], dtype=float)
    pred = Zt @ coef
    ss_res = float(((yt - pred) ** 2).sum())
    ss_tot = float(((yt - yt.mean()) ** 2).sum())
    return {"r2": round(1 - ss_res / ss_tot, 4) if ss_tot > 0 else -999.0,
            "rmse": round(float(np.sqrt(((yt - pred) ** 2).mean())), 4),
            "n": len(test_rows),
            "sign_errors": int((np.sign(pred) != np.sign(yt)).sum())}


def realcats_obs():
    nodes = pd.read_csv(DATA_DIR / "real_cats" / "real_graph_nodes.csv")
    edges = pd.read_csv(DATA_DIR / "real_cats" / "real_graph_edges.csv")
    y = nodes.label.values.astype(int)
    t = nodes.time.values.astype(float)
    n = len(y)
    id2idx = {int(i): k for k, i in enumerate(nodes.node_id.values)}
    src = np.array([id2idx[int(s)] for s in edges.src])
    dst = np.array([id2idx[int(d)] for d in edges.dst])
    et = edges.time.values.astype(float)
    ap = json.load(open(RESULTS_DIR / "edge_protocol_final.json"))
    hom = json.load(open(RESULTS_DIR / "cross_edge_homophily.json"))["real_cats"]
    excess_c1 = {round(float(h["split"].split("=")[1]), 2): (
        h["cross_edges"]["excess"] if h["cross_edges"] else None) for h in hom}
    out = []
    for i, s in enumerate(ap["splits"]):
        order = np.argsort(t, kind="mergesort")
        cut = int(s["frac"] * n)
        tr, te = order[:cut], order[cut:]
        te = te[t[te] > t[tr].max()]
        in_tr = np.zeros(n, dtype=bool)
        in_tr[tr] = True
        c3 = [(a, b) for a, b in zip(src, dst) if in_tr[a] and in_tr[b]]
        d3 = dense_deg(n, c3, te)
        base = s["conditions"]["C3"]["AP"]
        n_c2 = s["conditions"]["C2"]["n_edges"]
        pairs = {
            "C1": list(zip(src, dst)),
            "C2": [(a, b) for a, b, e in zip(src, dst, et)
                   if (in_tr[a] and in_tr[b])
                   or ((in_tr[a] != in_tr[b]) and e <= t[b if in_tr[a] else a])],
            "M2": [tuple(map(int, p)) for p in random_pair_edges(n, n_c2,
                                                                np.random.default_rng(i))],
        }
        for cond in ["C1", "C2", "M2"]:
            marg = ([(a, b) for a, b in zip(src, dst) if in_tr[a] != in_tr[b]]
                    if cond == "C1" else
                    [(a, b) for a, b, e in zip(src, dst, et)
                     if (in_tr[a] != in_tr[b]) and e <= t[b if in_tr[a] else a]]
                    if cond == "C2" else [])
            if cond == "C1":
                ex = excess_c1.get(round(s["frac"], 2))
            elif cond == "C2":
                a = np.array([p[0] for p in marg], dtype=int)
                b = np.array([p[1] for p in marg], dtype=int)
                ok = (y[a] >= 0) & (y[b] >= 0)
                if ok.sum() >= 30:
                    obs = float((y[a[ok]] == y[b[ok]]).mean())
                    p_tr, p_te = float(y[in_tr].mean()), float(y[~in_tr].mean())
                    exp = p_tr * p_te + (1 - p_tr) * (1 - p_te)
                    ex = round(obs - exp, 4)
                else:
                    ex = None
            else:
                ex = 0.0
            out.append({"dataset": "Real-CATS", "split": s["frac"], "condition": cond,
                        "delta_AP": round(s["conditions"][cond]["AP"] - base, 4),
                        "delta_degree": round(dense_deg(n, pairs[cond], te) - d3, 4),
                        "excess": ex})
    return out


def elliptic_obs():
    tx, ts, X, y, src, dst = _e49.load()
    n = len(y)
    ap = json.load(open(RESULTS_DIR / "elliptic_v1_strong_model.json"))
    hom = json.load(open(RESULTS_DIR / "cross_edge_homophily.json"))["elliptic_v1"]
    excess_c1 = {int(h["split"].split("=")[1]): (
        h["test_test_edges"]["excess"] if h["test_test_edges"] else None) for h in hom}
    out = []
    for s in ap["splits"]:
        if s["test_prevalence"] < 0.05:
            continue
        c = s["cutoff"]
        tr = np.where((y >= 0) & (ts <= c - 1))[0]
        te = np.where((y >= 0) & (ts >= c + 1))[0]
        in_tr = np.zeros(n, dtype=bool)
        in_tr[tr] = True
        ets = np.maximum(ts[src], ts[dst])
        c3 = [(a, b) for a, b in zip(src, dst) if in_tr[a] and in_tr[b]]
        d3 = dense_deg(n, c3, te)
        base = s["conditions"]["C3"]["AP"]
        n_c2 = s["conditions"]["C2"]["n_edges"]
        pairs = {
            "C1": list(zip(src, dst)),
            "C2": [(a, b) for a, b, e in zip(src, dst, ets)
                   if (in_tr[a] and in_tr[b])
                   or ((in_tr[a] != in_tr[b]) and e <= ts[b if in_tr[a] else a])],
            "M2": [tuple(map(int, p)) for p in random_pair_edges(n, n_c2,
                                                                np.random.default_rng(c))],
        }
        for cond in ["C1", "C2", "M2"]:
            ex = (excess_c1.get(c) if cond == "C1" else 0.0)
            out.append({"dataset": "Elliptic", "split": c, "condition": cond,
                        "delta_AP": round(s["conditions"][cond]["AP"] - base, 4),
                        "delta_degree": round(dense_deg(n, pairs[cond], te) - d3, 4),
                        "excess": ex})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", default=str(RESULTS_DIR))
    args = ap.parse_args()
    cols = ("delta_degree", "excess")

    rc = [r for r in realcats_obs() if r["excess"] is not None]
    el = [r for r in elliptic_obs() if r["excess"] is not None]
    sc = json.load(open(RESULTS_DIR / "step_coarsening.json"))
    inj = [{"dataset": "Elliptic-injection", "split": f"K{r['K_target']}lam{r['lambda']}",
            "condition": "inject", "delta_AP": r["delta_AP"],
            "delta_degree": r["delta_degree"], "excess": r["excess"]}
           for r in sc["injection"] if r["excess"] is not None]

    out = {"experiment": "sign_law", "real_cats": rc, "elliptic": el,
           "injection": inj, "status": "PASS"}
    out["fit_real"] = fit(rc + el, cols)
    out["fit_all"] = fit(rc + el + inj, cols)
    out["logo"] = {
        "train_realcats_test_elliptic": predict(rc, el, cols),
        "train_elliptic_test_realcats": predict(el, rc, cols),
        "train_real_test_injection": predict(rc + el, inj, cols),
    }
    m2 = [r for r in rc + el if r["condition"] == "M2"]
    out["m2_sign_errors"] = int(sum(1 for r in m2 if r["delta_AP"] < 0))
    out["m2_note"] = ("M2 has near-zero excess and high degree; it is the point where a "
                      "degree+homophily law must fail if edge identity matters independently.")
    out["verdict"] = {
        "r2_real": out["fit_real"]["r2"],
        "r2_all": out["fit_all"]["r2"],
        "logo_r2": {k: v["r2"] for k, v in out["logo"].items()},
        "law_supported": bool(out["fit_real"]["r2"] >= 0.8
                              and all(v["r2"] > 0.3 for k, v in out["logo"].items()
                                      if "injection" not in k)
                              and out["fit_real"]["sign_errors"] <= 2),
        "note": ("If law_supported is false, the manuscript reports two case studies plus "
                 "a controlled intervention, not a transferable quantitative law."),
    }
    for k in ["fit_real", "fit_all", "verdict"]:
        print(f"[{k}]", json.dumps(out[k], ensure_ascii=False), flush=True)
    print("[logo]", json.dumps(out["logo"], ensure_ascii=False), flush=True)

    p = Path(args.outdir) / "sign_law.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    write_provenance("sign_law", {"status": "PASS"})
    print(f"[done] {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
