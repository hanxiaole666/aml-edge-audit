"""
09_adaptive_audit.py - 自适应对抗攻击 + 标签质量审计
=====================================================
补完两个缺口：
  1) 自适应攻击（此前只有固定随机扰动）：黑盒坐标下降式迭代攻击，
     攻击者仅靠查询分数，迭代找最大退化方向（更贴近真实威胁模型）。
  2) 标签质量审计：标签来源偏差、时间分布漂移、类别不平衡、
     特征-标签相关性审查（诚实报告数据局限）。

用法：
  python scripts/09_adaptive_audit.py --data data/real_cats_eth.csv
断点续跑：checkpoint=results/_checkpoints/adaptive_audit.json
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    RESULTS_DIR, Checkpoint, pr_auc, roc_auc, write_provenance, set_seed,
)


def load_data(path: str):
    df = pd.read_csv(path)
    exclude = {"label", "time", "address", "etherscan_checked", "watchback_checked", "gs_checked"}
    feats = [c for c in df.columns if c not in exclude and
             pd.api.types.is_numeric_dtype(df[c])]
    X = df[feats].values.astype(float)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = df["label"].values.astype(int)
    t = df["time"].values.astype(float) if "time" in df.columns else np.zeros(len(y))
    return df, feats, X, y, t


def adaptive_attack(X_te: np.ndarray, y_te: np.ndarray, model,
                    eps: float = 0.5, n_iter: int = 40, n_coord: int = 10,
                    seed: int = 0) -> dict:
    """黑盒自适应攻击：坐标下降式迭代，仅用模型分数查询。

    每轮随机选 n_coord 个特征维度，对每个维度试探 +δ/-δ，
    选择使"目标类置信度下降最多"的方向（降低正类分数 = 让犯罪者被判良性）。
    预算约束：||Δ||_inf <= eps（标准化空间）。
    """
    rng = np.random.default_rng(seed)
    d = X_te.shape[1]
    base_score = model.predict_proba(X_te)[:, 1]
    base_ap = pr_auc(y_te, base_score)
    X_adv = X_te.copy()
    delta = np.zeros_like(X_te)
    step = eps / n_iter
    # 只攻击真实正类（犯罪者规避检测）
    pos_idx = np.where(y_te == 1)[0]
    if len(pos_idx) == 0:
        return {"base_ap": base_ap, "status": "SKIP", "reason": "无正类样本"}
    for it in range(n_iter):
        coords = rng.choice(d, size=min(n_coord, d), replace=False)
        cur_score = model.predict_proba(X_adv)[:, 1]
        for c in coords:
            # 试探正负方向
            cand_plus = X_adv.copy(); cand_plus[pos_idx, c] += step
            cand_minus = X_adv.copy(); cand_minus[pos_idx, c] -= step
            # 预算裁剪
            cand_plus[pos_idx, c] = np.clip(cand_plus[pos_idx, c], X_te[pos_idx, c] - eps, X_te[pos_idx, c] + eps)
            cand_minus[pos_idx, c] = np.clip(cand_minus[pos_idx, c], X_te[pos_idx, c] - eps, X_te[pos_idx, c] + eps)
            s_plus = model.predict_proba(cand_plus)[pos_idx, 1].mean()
            s_minus = model.predict_proba(cand_minus)[pos_idx, 1].mean()
            if s_minus <= s_plus:  # 选让正类分数更低的方向
                X_adv[pos_idx, c] = cand_minus[pos_idx, c]
            else:
                X_adv[pos_idx, c] = cand_plus[pos_idx, c]
    adv_score = model.predict_proba(X_adv)[:, 1]
    adv_ap = pr_auc(y_te, adv_score)
    # 攻击成功率：正类中被压到阈值以下的比例（阈值取 base 最优F1阈值近似 0.5）
    evade_rate = float((adv_score[pos_idx] < 0.5).mean())
    base_evade = float((base_score[pos_idx] < 0.5).mean())
    return {
        "base_ap": base_ap, "adaptive_ap": adv_ap,
        "ap_drop": base_ap - adv_ap,
        "evasion_rate_before": base_evade, "evasion_rate_after": evade_rate,
        "evasion_increase": evade_rate - base_evade,
        "eps": eps, "n_iter": n_iter, "n_coords": n_coord,
        "status": "PASS",
    }


def run_adaptive(data_path: str) -> dict:
    set_seed(0)
    _, feats, X, y, _ = load_data(data_path)
    tr, te = train_test_split(np.arange(len(y)), test_size=0.3, random_state=0, stratify=y)
    sc = StandardScaler().fit(X[tr])
    Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
    results = {}
    for name, m in [("LogisticRegression", LogisticRegression(max_iter=1000)),
                    ("RandomForest", RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1))]:
        m.fit(Xtr, y[tr])
        results[name] = {}
        for eps in [0.1, 0.5]:
            results[name][f"eps{eps}"] = adaptive_attack(Xte, y[te], m, eps=eps, n_iter=30, n_coord=8)
    return {"experiment": "adaptive_attack", "models": results, "status": "PASS"}


def run_label_audit(data_path: str) -> dict:
    """标签质量审计：来源、时间偏差、不平衡、特征-标签相关性。"""
    df, feats, X, y, t = load_data(data_path)
    n = len(y)
    pos = int(y.sum()); neg = n - pos
    out = {
        "experiment": "label_quality_audit",
        "n_total": n, "n_positive": pos, "n_negative": neg,
        "positive_rate": pos / n,
        "imbalance_ratio": neg / max(pos, 1),
    }
    # 时间分布偏差：正类 vs 负类的时间分布是否一致（KS 统计量近似）
    if t.max() > 0:
        from scipy.stats import ks_2samp
        tpos, tneg = t[y == 1], t[y == 0]
        ks = ks_2samp(tpos, tneg)
        out["time_distribution"] = {
            "positive_mean": float(tpos.mean()), "negative_mean": float(tneg.mean()),
            "ks_statistic": float(ks.statistic), "ks_pvalue": float(ks.pvalue),
            "temporal_label_bias": bool(ks.pvalue < 0.05),
        }
    # 特征-标签相关性（点二列相关），找最强单特征
    corrs = []
    for i, f in enumerate(feats):
        if np.std(X[:, i]) > 0:
            r = np.corrcoef(X[:, i], y)[0, 1]
            if np.isfinite(r):
                corrs.append((abs(float(r)), f, float(r)))
    corrs.sort(reverse=True)
    out["top_label_correlated_features"] = [
        {"feature": f, "corr": round(r, 4)} for _, f, r in corrs[:10]
    ]
    # 单一特征过强 = 潜在标签泄漏信号
    out["suspicious_single_feature"] = bool(corrs[0][0] > 0.9) if corrs else False
    # 数据来源声明
    out["source_notes"] = [
        "Real-CATS 犯罪地址：来自公开报道/执法公告（Phishing Scam 等 21 类）",
        "Real-CATS 良性地址：交易所客户，可能存在选择偏差（非随机抽样）",
        "时间缺失样本用中位数填补（见 provenance）",
        "标签质量局限：犯罪标签依赖第三方报告，存在报告偏差；良性标签不保证绝对干净",
    ]
    out["status"] = "PASS"
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data/real_cats_eth.csv")
    ap.add_argument("--outdir", default=str(RESULTS_DIR))
    ap.add_argument("--fresh", action="store_true")
    args = ap.parse_args()
    data_path = Path(args.data)
    outdir = Path(args.outdir)
    ckpt = Checkpoint("adaptive_audit", outdir / "_checkpoints" / "adaptive_audit.json")
    ckpt.install_signal_handler()

    tasks = {
        "adaptive": lambda: run_adaptive(str(data_path)),
        "label_audit": lambda: run_label_audit(str(data_path)),
    }
    output = {}
    for key, fn in tasks.items():
        if ckpt.is_done(key) and not args.fresh:
            print(f"  [SKIP] {key}: 已完成"); continue
        ckpt.mark_start(key)
        try:
            res = fn()
        except Exception as exc:
            res = {"status": "FAIL", "error": f"{type(exc).__name__}: {exc}"}
        output[key] = res
        ckpt.mark_done(key, {"status": res.get("status", "?")})
        write_provenance("adaptive_audit", {"key": key, "result": res})
        print(f"  [{res.get('status','?'):4s}] {key}")
    import json
    out_file = outdir / "adaptive_audit.json"
    out_file.write_text(json.dumps(output, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    print(f"\n[done] {out_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
