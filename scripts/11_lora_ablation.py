"""
11_lora_ablation.py - 真实小语言模型微调 + LoRA rank 消融
=========================================================
用 hf-mirror.com 下载 bert-tiny（4.4M 参数），在真实行为文本上微调，
并做 LoRA rank {4,8,16} 消融（补完增补实验清单 #20 的 LoRA rank 缺口）。

诚信标注：bert-tiny 是极小模型，用于管线验证；论文主线仍是 Qwen3-0.6B/1.7B，
         本实验证明"LoRA rank 消融协议可运行、结果可复现"。

用法：
  HF_ENDPOINT=https://hf-mirror.com python scripts/11_lora_ablation.py --data data/real_text.csv
断点续跑：checkpoint=results/_checkpoints/lora_ablation.json
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    RESULTS_DIR, Checkpoint, pr_auc, roc_auc, write_provenance, set_seed,
)

MODEL_ID = "prajjwal1/bert-tiny"


def run_lora(texts, labels, rank: int, seed: int, epochs: int = 2, max_len: int = 64):
    """LoRA 微调（用 peft）并返回测试集 PR-AUC。"""
    from peft import LoraConfig, get_peft_model
    set_seed(seed)
    torch.manual_seed(seed)
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID, num_labels=2)
    if rank > 0:
        lcfg = LoraConfig(
            r=rank, lora_alpha=2 * rank,
            target_modules=["query", "value"],
            lora_dropout=0.1, task_type="SEQ_CLS",
        )
        model = get_peft_model(model, lcfg)
    tr, te = train_test_split(np.arange(len(texts)), test_size=0.2,
                              random_state=seed, stratify=labels)
    # 小规模子集（CPU 可承受）：每轮 800 训练 / 400 测试
    if len(tr) > 800:
        rng = np.random.default_rng(seed)
        tr = rng.choice(tr, 800, replace=False)
        te = rng.choice(te, 400, replace=False)

    def encode(idx):
        return tok([texts[i] for i in idx], truncation=True, padding="max_length",
                   max_length=max_len, return_tensors="pt")
    enc_tr = encode(tr)
    y_tr = torch.tensor(labels[tr], dtype=torch.long)
    enc_te = encode(te)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=3e-4)
    model.train()
    t0 = time.perf_counter()
    for _ in range(epochs):
        opt.zero_grad()
        out = model(**enc_tr, labels=y_tr)
        out.loss.backward()
        opt.step()
    train_sec = time.perf_counter() - t0

    model.eval()
    with torch.no_grad():
        logits = model(**enc_te).logits
        probs = torch.softmax(logits, dim=-1)[:, 1].numpy()
    return {
        "rank": rank, "seed": seed,
        "PR-AUC": float(pr_auc(labels[te], probs)),
        "AUC": float(roc_auc(labels[te], probs)),
        "train_seconds": round(train_sec, 2),
        "trainable_params": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
        "status": "done",
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data/real_text.csv")
    ap.add_argument("--outdir", default=str(RESULTS_DIR))
    ap.add_argument("--ranks", type=int, nargs="*", default=[0, 4, 8, 16])
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--fresh", action="store_true")
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    texts = df["text"].astype(str).tolist()
    labels = df["label"].values.astype(int)
    outdir = Path(args.outdir)
    ckpt = Checkpoint("lora", outdir / "_checkpoints" / "lora_ablation.json")
    ckpt.install_signal_handler()
    print(f"[data] {args.data}: n={len(df)} pos_rate={labels.mean():.3f}")
    print(f"[model] {MODEL_ID} via {os.environ.get('HF_ENDPOINT')}")

    all_keys = [f"rank{r}|seed{s}" for r in args.ranks for s in range(args.seeds)]
    todo = all_keys if args.fresh else ckpt.todo_keys(all_keys)
    print(f"[checkpoint] 总 run={len(all_keys)} 待跑={len(todo)} 已完成={len(all_keys)-len(todo)}")

    rows = []
    for key in todo:
        rank = int(key.split("rank")[1].split("|")[0])
        seed = int(key.split("seed")[1])
        ckpt.mark_start(key)
        try:
            res = run_lora(texts, labels, rank, seed)
        except Exception as exc:
            res = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
        if res.get("status") == "done":
            ckpt.mark_done(key, res)
            rows.append(res)
            write_provenance("lora_ablation", {"key": key, "result": res})
            print(f"  [OK] {key:20s} PR-AUC={res['PR-AUC']:.4f} params={res['trainable_params']} time={res['train_seconds']}s")
        else:
            ckpt.skip(key)
            print(f"  [SKIP] {key:20s} {res.get('error','')[:60]}")

    # 从 checkpoint 重建完整汇总
    ckpt._flush()
    full = [{"key": k, **v["metrics"]} for k, v in ckpt.state["runs"].items()
            if v.get("status") == "done" and "PR-AUC" in v.get("metrics", {})]
    if full:
        sdf = pd.DataFrame(full)
        sdf.to_csv(outdir / "lora_ablation_results.csv", index=False, float_format="%.6f")
        print("\n[汇总]")
        print(sdf.groupby("rank").agg(PR_AUC_mean=("PR-AUC", "mean"),
                                      params=("trainable_params", "first"),
                                      sec=("train_seconds", "mean")).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
