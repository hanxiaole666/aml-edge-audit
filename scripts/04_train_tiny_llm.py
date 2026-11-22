"""
04_train_tiny_llm.py - 文本轨道：TF-IDF 基线（必跑）+ 可选 Tiny-LLM 增强
=============================================================================
默认模式（tfidf）：TF-IDF + LogisticRegression，纯 sklearn，任何环境秒级跑通。
增强模式（--model qwen）：尝试 Qwen3-0.6B / bert-tiny（需 transformers + 模型下载），
                          缺依赖或下载失败自动降级为 tfidf，绝不中断主流程。

关键设计（对应评审陷阱）：
  - thinking 模式：如用 Qwen3.5，统一 enable_thinking=False（此处仅记录开关）
  - 断点续跑：每个 (model, seed) 独立 run，checkpoint=results/_checkpoints/text.json
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    RESULTS_DIR, Checkpoint, compute_metrics, set_seed, write_provenance,
)


def run_tfidf(texts: list[str], labels: np.ndarray, seed: int, quick: bool) -> dict:
    set_seed(seed)
    from sklearn.model_selection import train_test_split
    tr, te = train_test_split(np.arange(len(texts)), test_size=0.3,
                              random_state=seed, stratify=labels)
    vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), sublinear_tf=True)
    X = vec.fit_transform([texts[i] for i in tr])
    Xte = vec.transform([texts[i] for i in te])
    clf = LogisticRegression(max_iter=1000, C=2.0)
    t0 = time.perf_counter()
    clf.fit(X, labels[tr])
    score = clf.predict_proba(Xte)[:, 1]
    met = compute_metrics(labels[te], score)
    met.update({"model": "text_TFIDF-LR", "seed": seed, "status": "done",
                "fit_seconds": round(time.perf_counter() - t0, 3)})
    return met


def run_tiny_llm(texts: list[str], labels: np.ndarray, seed: int,
                 model_id: str, quick: bool) -> dict:
    """可选增强：真实小 LLM 微调。缺依赖/下载失败抛异常，由上层降级。"""
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from sklearn.model_selection import train_test_split

    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id, num_labels=2, trust_remote_code=True,
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token or tok.unk_token
    tr, te = train_test_split(np.arange(len(texts)), test_size=0.3,
                              random_state=seed, stratify=labels)
    enc = tok([texts[i] for i in tr], truncation=True, padding=True, max_length=128,
              return_tensors="pt")
    y_tr = torch.tensor(labels[tr], dtype=torch.long)
    optim = torch.optim.AdamW(model.parameters(), lr=2e-5)
    model.train()
    steps = 50 if quick else 200
    for _ in range(steps):
        optim.zero_grad()
        out = model(**enc, labels=y_tr)
        out.loss.backward()
        optim.step()
    model.eval()
    enc_te = tok([texts[i] for i in te], truncation=True, padding=True, max_length=128,
                 return_tensors="pt")
    with torch.no_grad():
        logits = model(**enc_te).logits
        score = torch.softmax(logits, dim=-1)[:, 1].numpy()
    met = compute_metrics(labels[te], score)
    met.update({"model": f"text_{model_id}", "seed": seed, "status": "done",
                "thinking": "disabled", "note": "真实小 LLM 微调"})
    return met


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data/synthetic_text.csv")
    ap.add_argument("--outdir", default=str(RESULTS_DIR))
    ap.add_argument("--tag", default="", help="数据集标签，区分 checkpoint/结果文件")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--model", default="tfidf", choices=["tfidf", "qwen", "bert-tiny"],
                    help="tfidf=默认轻量基线；qwen/bert-tiny=可选真实小LLM（失败自动降级）")
    ap.add_argument("--model-id", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--fresh", action="store_true")
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    texts = df["text"].astype(str).tolist()
    labels = df["label"].astype(int).values
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    data_tag = args.tag or Path(args.data).stem
    ckpt = Checkpoint(f"text_{data_tag}", outdir / "_checkpoints" / f"text_{data_tag}.json")
    ckpt.install_signal_handler()
    print(f"[data] {args.data}: n={len(df)} pos_rate={labels.mean():.3f}")

    all_keys = [f"{args.model}|seed{s}" for s in range(args.seeds)]
    todo = all_keys if args.fresh else ckpt.todo_keys(all_keys)
    print(f"[checkpoint] 总 run={len(all_keys)} 待跑={len(todo)} 已完成={len(all_keys)-len(todo)}")

    rows = []
    for key in todo:
        seed = int(key.split("|seed")[1])
        ckpt.mark_start(key)
        used_fallback = False
        try:
            if args.model == "tfidf":
                res = run_tfidf(texts, labels, seed, args.quick)
            else:
                mid = "prajjwal1/bert-tiny" if args.model == "bert-tiny" else args.model_id
                res = run_tiny_llm(texts, labels, seed, mid, args.quick)
        except Exception as exc:
            msg = f"{type(exc).__name__}: {exc}"
            print(f"  [FALLBACK] {key}: 小LLM不可用({msg[:70]})，降级为 TF-IDF")
            write_provenance("text_llm_fallback", {"key": key, "reason": msg})
            used_fallback = True
            try:
                res = run_tfidf(texts, labels, seed, args.quick)
                res["model"] = f"text_TFIDF-LR(fallback-from-{args.model})"
            except Exception as exc2:
                res = {"status": "failed", "error": f"{type(exc2).__name__}: {exc2}"}

        if res.get("status") == "done":
            ckpt.mark_done(key, {"PR-AUC": round(res["PR-AUC"], 4)})
            rows.append(res)
            print(f"  [OK] {key:30s} PR-AUC={res['PR-AUC']:.4f} AUC={res['AUC']:.4f}"
                  + (" [降级]" if used_fallback else ""))
        else:
            ckpt.skip(key)
            print(f"  [SKIP] {key:30s} {res.get('error','')[:60]}")

    if rows:
        rdf = pd.DataFrame(rows)
        rdf["dataset"] = data_tag
        rdf.to_csv(outdir / f"text_results_{data_tag}.csv", index=False, float_format="%.6f")
        print("\n[汇总]")
        print(rdf.groupby("model").agg(PR_AUC_mean=("PR-AUC", "mean"),
                                       AUC_mean=("AUC", "mean"), n=("seed", "nunique")).to_string())
    print(f"\n[done] 文本轨道完成，checkpoint={outdir / '_checkpoints' / 'text.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
