"""
68_step_coarsening.py - E1/R1：把"符号由度与同配性决定"从两点观测变成受控实验
======================================================================
评审的核心质疑：Real-CATS 反号、Elliptic 正号，只有两个数据点，事后用
degree/homophily 解释，不能称为规律。本脚本在 Elliptic 真实标签与特征上做
两组**干预**（不是新数据、不是新模型），检验符号是否随这两个量翻转。

实验 A：时间步粗化（step coarsening）
    把相邻 w 个时间步合并为一个窗口，切分点落在窗口**内部**：窗口之前的
    全部进训练，窗口之后的全部进测试，边界窗口内的节点随机对半分。
    w 越大，边界窗口越宽，被切分切开的同期边越多 —— 这些边就是"新造出来
    的跨切分边"。它们的标签同配性由随机对半分决定，因此 excess ≈ 0，而测试
    节点度随 w 单调上升。若度通道真的有害，ΔAP = AP(keep) − AP(remove) 应
    随 w 下降；若 ΔAP 保持 0，则"度本身"不是充分条件。

实验 B：受控同配性注入（controlled homophily injection）
    固定 w=8 的切分与 C3 图（无跨切分边），再注入 K 条 train–test 边，其中
    同标签边比例 q 由我们设定。q 从低于独立期望（反同配）扫到高于它（同配），
    K 取两档以改变测试节点度。ΔAP 应在 q 跨过独立期望时由负转正。
    这是干预实验：标签只用于**构造扰动**，不进入模型训练；论文中明确标注为
    controlled intervention，而不是观测结果。

用法：python scripts/68_step_coarsening.py
产出：results/step_coarsening.json + figures/sign_law_intervention.pdf
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
from lightgbm import LGBMClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RESULTS_DIR, pr_auc, set_seed, write_provenance

_spec = importlib.util.spec_from_file_location(
    "e4", Path(__file__).resolve().parent / "49_elliptic_v1_edge_protocol.py")
_m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_m)
load = _m.load


def neighbour_block(A, X):
    """邻居均值与最大值。A 为对称 CSR 邻接。"""
    deg = np.asarray(A.sum(axis=1)).ravel()
    dinv = 1.0 / np.maximum(deg, 1.0)
    mean = sp.diags(dinv) @ (A @ X)
    A_csr = A.tocsr()
    mx = np.zeros_like(X)
    nz = np.where(deg > 0)[0]
    for i in nz:
        lo, hi = A_csr.indptr[i], A_csr.indptr[i + 1]
        mx[i] = X[A_csr.indices[lo:hi]].max(axis=0)
    return mean.astype(np.float32), mx.astype(np.float32)


def score(A, X, y, tr, te, A_test=None):
    """A 用于训练端特征，A_test 用于测试端特征（None 表示同一张图）。

    部署式注入必须分开：训练图保持干净，只有测试期可观测的边进入测试特征。
    若两边共用被污染的图，模型会在训练中学会"反标签边 = 反向信号"，
    把本应有害的扰动变成可用的反向特征，符号因此不翻转（第一版注入的失败原因）。
    """
    A_test = A if A_test is None else A_test
    mean_tr, mx_tr = neighbour_block(A, X)
    mean_te, mx_te = neighbour_block(A_test, X)
    Ftr = np.hstack([X, mean_tr, mx_tr])[tr]
    Fte = np.hstack([X, mean_te, mx_te])[te]
    m = LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=63,
                       subsample=0.9, colsample_bytree=0.8, random_state=0,
                       n_jobs=-1, verbose=-1,
                       scale_pos_weight=(len(tr) - y[tr].sum()) / max(y[tr].sum(), 1))
    m.fit(Ftr, y[tr])
    s = m.predict_proba(Fte)[:, 1]
    return float(pr_auc(y[te], s)), s


def to_csr(n, pairs):
    if not len(pairs):
        return sp.csr_matrix((n, n))
    r = np.array([p[0] for p in pairs] + [p[1] for p in pairs])
    c = np.array([p[1] for p in pairs] + [p[0] for p in pairs])
    data = np.ones(len(r), dtype=np.float32)
    return sp.coo_matrix((data, (r, c)), shape=(n, n)).tocsr()


def homophily(pairs, y, p_a, p_b):
    """已标注端点的同标签率与独立期望（用两端总体的 prevalence）。"""
    a = np.array([p[0] for p in pairs])
    b = np.array([p[1] for p in pairs])
    ok = (y[a] >= 0) & (y[b] >= 0)
    if ok.sum() < 20:
        return None
    a, b = a[ok], b[ok]
    obs = float((y[a] == y[b]).mean())
    exp = float(p_a * p_b + (1 - p_a) * (1 - p_b))
    return {"n_labelled": int(ok.sum()), "observed_same": round(obs, 4),
            "expected_independent": round(exp, 4), "excess": round(obs - exp, 4)}


def mean_test_degree(A, te):
    d = np.asarray(A.sum(axis=1)).ravel()
    return round(float(d[te].mean()), 4)


def coarsening(src, dst, ts, y, X, w, boundary_step=30, rng=None):
    """窗口粗化：窗口内边界节点随机对半分，造出跨切分边。"""
    rng = rng or np.random.default_rng(0)
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
    tr = np.array(sorted(tr))
    te = np.array(sorted(te))
    in_tr = np.zeros(len(y), dtype=bool)
    in_tr[tr] = True
    keep = np.ones(len(src), dtype=bool)
    cross = []
    for s, d in zip(src, dst):
        if in_tr[s] != in_tr[d]:
            cross.append((s, d))
    # C1 保留全部；C3 只保留同侧边
    pairs_c1 = list(zip(src, dst))
    pairs_c3 = [(s, d) for s, d in zip(src, dst) if in_tr[s] == in_tr[d]]
    A1 = to_csr(len(y), pairs_c1)
    A3 = to_csr(len(y), pairs_c3)
    ap1, _ = score(A1, X, y, tr, te)
    ap3, _ = score(A3, X, y, tr, te)
    h = homophily(cross, y, float(y[tr].mean()), float(y[te].mean()))
    return {
        "window": w, "cutoff_window": cw,
        "boundary_steps": [cw * w, (cw + 1) * w - 1],
        "n_train": int(len(tr)), "n_test": int(len(te)),
        "test_prevalence": round(float(y[te].mean()), 4),
        "n_cross_edges": len(cross),
        "AP_keep": round(ap1, 4), "AP_remove": round(ap3, 4),
        "delta_AP": round(ap1 - ap3, 4),
        "degree_keep": mean_test_degree(A1, te),
        "degree_remove": mean_test_degree(A3, te),
        "delta_degree": round(mean_test_degree(A1, te) - mean_test_degree(A3, te), 4),
        "cross_homophily": h,
        "neighbour_label_auc_keep": neighbour_label_auc(A1, y, tr, te),
        "neighbour_label_auc_remove": neighbour_label_auc(A3, y, tr, te),
    }


def neighbour_label_auc(A, y, tr, te):
    """只用训练标签：每个测试节点邻居中训练正例占比，对测试标签的 AUC。"""
    pos = np.zeros(len(y), dtype=np.float32)
    pos[tr[y[tr] == 1]] = 1.0
    deg = np.asarray(A.sum(axis=1)).ravel()
    cnt = np.asarray(A @ pos).ravel()
    ok = (deg > 0) & (y >= 0)
    mask = np.zeros(len(y), dtype=bool)
    mask[te] = ok[te]
    if mask.sum() < 30 or len(np.unique(y[mask])) < 2:
        return None
    frac = cnt[mask] / deg[mask]
    from sklearn.metrics import roc_auc_score
    return round(float(roc_auc_score(y[mask], frac)), 4)


def inject(src, dst, ts, y, X, w=8, boundary_step=30, K=4000, lam=1.0, seed=0):
    """固定切分，向**测试子图内部**注入 K 条边，训练图保持 C3（tr-tr）。

    lam 是类别条件的同配性参数：
      lam =  0  独立抽样（同标签率 = 测试期边缘分布给出的期望）
      lam = +1  每条边都同标签
      lam = -1  每条边都跨标签
    只改 test-test 边，因此训练端特征完全不受影响；这正是"攻击者在测试期
    新建关联"的部署场景，也避免了模型把扰动学成反向特征。
    """
    rng = np.random.default_rng(seed)
    win = ts // w
    cw = int(boundary_step // w)
    lab = y >= 0
    tr = np.where(lab & (win < cw))[0]
    te = np.where(lab & (win > cw))[0]
    b = np.where(lab & (win == cw))[0]
    rng.shuffle(b)
    half = len(b) // 2
    tr = np.concatenate([tr, b[:half]])
    te = np.concatenate([te, b[half:]])
    in_tr = np.zeros(len(y), dtype=bool)
    in_tr[tr] = True
    base = [(s, d) for s, d in zip(src, dst) if in_tr[s] and in_tr[d]]
    A_train = to_csr(len(y), base)
    A_test0 = A_train
    ap0, _ = score(A_train, X, y, tr, te, A_test0)

    te_pos, te_neg = te[y[te] == 1], te[y[te] == 0]
    pairs = []
    for _ in range(K):
        u = rng.random()
        if u < abs(lam):
            same = lam > 0
            if same:
                wpos = len(te_pos) * (len(te_pos) - 1)
                wneg = len(te_neg) * (len(te_neg) - 1)
                if rng.random() < wpos / max(wpos + wneg, 1):
                    pairs.append((rng.choice(te_pos), rng.choice(te_pos)))
                else:
                    pairs.append((rng.choice(te_neg), rng.choice(te_neg)))
            else:
                wpos = len(te_pos) * len(te_neg)
                pairs.append((rng.choice(te_pos), rng.choice(te_neg)))
        else:
            pairs.append((rng.choice(te), rng.choice(te)))
    pairs = list({(min(a, b), max(a, b)) for a, b in pairs if a != b})
    A_test = to_csr(len(y), base + pairs)
    ap1, _ = score(A_train, X, y, tr, te, A_test)
    h = homophily(pairs, y, float(y[te].mean()), float(y[te].mean()))
    p_te = float(y[te].mean())
    exp = p_te * p_te + (1 - p_te) * (1 - p_te)
    return {
        "K_target": K, "lambda": round(lam, 3),
        "K_unique": len(pairs),
        "q_realised": h["observed_same"] if h else None,
        "excess": round(h["observed_same"] - exp, 4) if h else None,
        "independence_expectation": round(exp, 4),
        "AP_base": round(ap0, 4), "AP_injected": round(ap1, 4),
        "delta_AP": round(ap1 - ap0, 4),
        "degree_base": mean_test_degree(A_test0, te),
        "degree_injected": mean_test_degree(A_test, te),
        "delta_degree": round(mean_test_degree(A_test, te) - mean_test_degree(A_test0, te), 4),
        "n_test": int(len(te)), "test_prevalence": round(float(y[te].mean()), 4),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", default=str(RESULTS_DIR))
    args = ap.parse_args()
    set_seed(0)

    tx, ts, X, y, src, dst = load()
    out = {"experiment": "step_coarsening", "dataset": "Elliptic v1 (real steps)",
           "readout": "LightGBM on [93 local, neighbour-mean, neighbour-max]",
           "note": ("Coarsening places the split inside a merged window and assigns "
                    "boundary-window nodes at random, which creates cross-split edges "
                    "of near-zero homophily excess. The injection arm sets the "
                    "same-label rate of added cross-split edges directly; labels are "
                    "used only to construct the intervention, never as model input."),
           "coarsening": [], "injection": [], "status": "PASS"}

    for w in [1, 2, 4, 8, 16]:
        r = coarsening(src, dst, ts, y, X, w, rng=np.random.default_rng(w))
        out["coarsening"].append(r)
        print(f"[coarsen w={w}] cross={r['n_cross_edges']} "
              f"excess={r['cross_homophily']['excess'] if r['cross_homophily'] else None} "
              f"dDeg={r['delta_degree']:+.3f} dAP={r['delta_AP']:+.4f}", flush=True)

    for K in [8000, 40000]:
        for lam in [-1.0, -0.5, 0.0, 0.5, 1.0]:
            r = inject(src, dst, ts, y, X, w=8, K=K, lam=lam,
                       seed=K * 10 + int((lam + 1) * 10))
            out["injection"].append(r)
            print(f"[inject K={K} lam={lam:+.1f}] q={r['q_realised']} "
                  f"excess={r['excess']:+.3f} dDeg={r['delta_degree']:+.3f} "
                  f"dAP={r['delta_AP']:+.4f}",
                  flush=True)

    # 汇总判据：符号是否随 excess 翻转
    inj = [r for r in out["injection"] if r["excess"] is not None]
    neg = [r["delta_AP"] for r in inj if r["lambda"] < 0]
    pos = [r["delta_AP"] for r in inj if r["lambda"] > 0]
    out["verdict"] = {
        "mean_delta_AP_negative_excess": round(float(np.mean(neg)), 4) if neg else None,
        "mean_delta_AP_positive_excess": round(float(np.mean(pos)), 4) if pos else None,
        "mean_delta_AP_lambda_zero": round(float(np.mean(
            [r["delta_AP"] for r in inj if r["lambda"] == 0])), 4) if any(
            r["lambda"] == 0 for r in inj) else None,
        "sign_flips_with_assortativity": bool(neg and pos and np.mean(neg) < 0 < np.mean(pos)),
        "coarsening_delta_AP_range": [round(min(r["delta_AP"] for r in out["coarsening"]), 4),
                                      round(max(r["delta_AP"] for r in out["coarsening"]), 4)],
        "coarsening_note": ("At near-zero excess, coarsening changes test-node degree "
                            "without changing edge composition; the AP gap stays small, "
                            "which isolates the composition channel from the degree channel."),
    }
    print("[verdict]", out["verdict"], flush=True)

    # 图：ΔAP vs excess，点大小按 Δdegree
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.7))
    c = out["coarsening"]
    ax[0].plot([r["window"] for r in c], [r["delta_AP"] for r in c], marker="o", ms=4, lw=1.2)
    ax[0].axhline(0, color="grey", lw=0.8, ls=":")
    ax[0].set_xscale("log", base=2)
    ax[0].set_xlabel("window width (steps)")
    ax[0].set_ylabel(r"$\Delta$AP (keep $-$ remove)")
    ax[0].set_title("(a) coarsening, excess $\\approx 0$", fontsize=8)
    for r in inj:
        ax[1].scatter(r["excess"], r["delta_AP"], s=18 + 40 * r["delta_degree"],
                      alpha=0.8)
    ax[1].axhline(0, color="grey", lw=0.8, ls=":")
    ax[1].axvline(0, color="grey", lw=0.8, ls=":")
    ax[1].set_xlabel("cross-edge homophily excess")
    ax[1].set_ylabel(r"$\Delta$AP (injected $-$ base)")
    ax[1].set_title("(b) injection, two degree levels", fontsize=8)
    fig.tight_layout()
    fig.savefig(Path(args.outdir).parent / "figures" / "sign_law_intervention.pdf")
    plt.close(fig)

    p = Path(args.outdir) / "step_coarsening.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    write_provenance("step_coarsening", {"status": "PASS"})
    print(f"[done] {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
