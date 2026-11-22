"""
34_embedding_visualization.py - 表示空间可视化（回应审稿人 m6）
==============================================================
生成两张图：
  Fig A: Elliptic2 子图表示（SGC 池化）的 t-SNE，按 suspicious/licit 着色
  Fig B: Real-CATS 真实地址行为特征的 PCA，按 label 着色
并计算类别分离度指标（silhouette score），量化表示是否可分辨。

用法：python scripts/34_embedding_visualization.py
产出：results/embedding_analysis.json + figures/embedding_*.pdf
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA_DIR, RESULTS_DIR, write_provenance


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", default=str(RESULTS_DIR))
    args = ap.parse_args()
    figdir = Path(__file__).resolve().parent.parent / "submission_tdsc" / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 8, "font.family": "serif", "figure.dpi": 200})
    out = {"experiment": "embedding_visualization", "status": "PASS"}

    # A. Elliptic2 子图表示
    print("[embed] Elliptic2 subgraph embeddings...", flush=True)
    cache = DATA_DIR / "elliptic2" / "elliptic2_subgraph_embeddings.npy"
    label_cache = DATA_DIR / "elliptic2" / "elliptic2_subgraph_labels.npy"
    X = None
    if cache.exists() and label_cache.exists():
        X = np.load(cache); y = np.load(label_cache)
    else:
        # 从 20 脚本的表示重构（采样以控制 t-SNE 耗时）
        nodes = pd.read_csv(DATA_DIR / "elliptic2" / "nodes.csv")
        comps = pd.read_csv(DATA_DIR / "elliptic2" / "connected_components.csv")
        Xn = np.load(DATA_DIR / "elliptic2" / "node_features_43.npy")
        cl2idx = {int(c): i for i, c in enumerate(nodes.clId.values)}
        cc_nodes = {}
        for cl, cc in zip(nodes.clId.values, nodes.ccId.values):
            cc_nodes.setdefault(int(cc), []).append(int(cl))
        rng = np.random.default_rng(0)
        rows, labels = [], []
        for _, r in comps.iterrows():
            cc = int(r.ccId); cls = cc_nodes.get(cc, [])
            if not cls: continue
            M = Xn[[cl2idx[c] for c in cls if c in cl2idx]]
            if M.size == 0: continue
            vec = np.concatenate([np.nanmean(M, axis=0), np.nanmax(M, axis=0),
                                  np.nanstd(M, axis=0), [len(cls)]])
            rows.append(np.nan_to_num(vec, nan=0.0, posinf=0.0))
            labels.append(1 if r.ccLabel == "suspicious" else 0)
        X = np.vstack(rows); y = np.array(labels)
        np.save(cache, X); np.save(label_cache, y)
    # 采样：保留全部少数类 + 等量多数类
    pos = np.where(y == 1)[0]; neg = np.where(y == 0)[0]
    rng = np.random.default_rng(0)
    sel = np.concatenate([pos, rng.choice(neg, size=min(len(neg), len(pos) * 3), replace=False)])
    Xs = StandardScaler().fit_transform(X[sel][:, :43])
    Z = TSNE(n_components=2, random_state=0, perplexity=30).fit_transform(Xs)
    sil = float(silhouette_score(Xs, y[sel]))
    out["elliptic2"] = {"n_sampled": len(sel), "n_suspicious": int((y[sel] == 1).sum()),
                        "silhouette": round(sil, 4)}

    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    for lab, color, name in [(0, "#b0b0b0", "licit"), (1, "#c00000", "suspicious")]:
        m = y[sel] == lab
        ax.scatter(Z[m, 0], Z[m, 1], s=4, c=color, alpha=0.5, label=name, edgecolors="none")
    ax.set_title(f"Elliptic2 subgraphs (sil={sil:.3f})")
    ax.legend(fontsize=7, markerscale=2)
    ax.axis("off")
    fig.tight_layout(); fig.savefig(figdir / "embedding_elliptic2.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Elliptic2: n={len(sel)} silhouette={sil:.4f}", flush=True)

    # B. Real-CATS 地址行为特征 PCA
    print("[embed] Real-CATS address features...", flush=True)
    df = pd.read_csv(DATA_DIR / "real_cats_eth.csv")
    feats = [c for c in df.columns if c not in {"label", "time", "address", "etherscan_checked", "watchback_checked", "gs_checked"}
             and pd.api.types.is_numeric_dtype(df[c])]
    Xr = df[feats].values.astype(float)
    Xr = np.nan_to_num(Xr, nan=0.0, posinf=0.0)
    yr = df.label.values.astype(int)
    pos = np.where(yr == 1)[0]; neg = np.where(yr == 0)[0]
    sel = np.concatenate([rng.choice(pos, 2000, replace=False),
                          rng.choice(neg, 2000, replace=False)])
    Xr_s = StandardScaler().fit_transform(Xr[sel][:, :49])
    pca = PCA(n_components=2, random_state=0)
    Zr = pca.fit_transform(Xr_s)
    sil_r = float(silhouette_score(Xr_s, yr[sel]))
    out["realcats"] = {"n_sampled": len(sel), "pca_explained": [round(float(v), 4) for v in pca.explained_variance_ratio_],
                       "silhouette": round(sil_r, 4)}

    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    for lab, color, name in [(0, "#b0b0b0", "benign"), (1, "#c00000", "criminal")]:
        m = yr[sel] == lab
        ax.scatter(Zr[m, 0], Zr[m, 1], s=4, c=color, alpha=0.4, label=name, edgecolors="none")
    ax.set_title(f"Real-CATS addresses (sil={sil_r:.3f})")
    ax.legend(fontsize=7, markerscale=2)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    fig.tight_layout(); fig.savefig(figdir / "embedding_realcats.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Real-CATS: n={len(sel)} silhouette={sil_r:.4f}", flush=True)

    p = Path(args.outdir) / "embedding_analysis.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    write_provenance("embedding_visualization", {"status": "PASS"})
    print(f"[done] {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
