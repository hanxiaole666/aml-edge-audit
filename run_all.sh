#!/usr/bin/env bash
# run_all.sh — TinyAML-Bench 一键全流程（自动下载 + 三轨实验 + 统一评估）
# ============================================================================
# 特性：
#   1. 数据自动下载：无数据自动下载，已有数据跳过（01_download_data.py 负责）
#   2. 断点续跑：每个子脚本用 checkpoint，中断后重跑自动跳过已完成 run
#   3. 优雅降级：依赖缺失/网络失败自动降级，不中断主流程
#
# 用法：
#   bash run_all.sh                  # 完整模式（默认 5 种子）
#   bash run_all.sh --quick          # 冒烟模式（2 种子，秒级）
#   bash run_all.sh --fresh          # 忽略 checkpoint 全部重跑
#   bash run_all.sh --text-model qwen  # 尝试真实小 LLM（失败自动降级 TF-IDF）
# ============================================================================
set -uo pipefail
cd "$(dirname "$0")"

QUICK=""
FRESH=""
TEXT_MODEL="tfidf"
SEEDS=5
K_VALUES="1 2"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --quick) QUICK="--quick"; SEEDS=2; K_VALUES="1"; shift ;;
    --fresh) FRESH="--fresh"; shift ;;
    --text-model) TEXT_MODEL="$2"; shift 2 ;;
    --seeds) SEEDS="$2"; shift 2 ;;
    *) echo "未知参数: $1"; shift ;;
  esac
done

COMMON_ARGS=""
[[ -n "$QUICK" ]] && COMMON_ARGS="$COMMON_ARGS --quick"
[[ -n "$FRESH" ]] && COMMON_ARGS="$COMMON_ARGS --fresh"

echo "=============================================="
echo " TinyAML-Bench 全流程  quick=${QUICK:-no}  fresh=${FRESH:-no}  seeds=$SEEDS"
echo "=============================================="

echo ""
echo "== [1/5] 数据准备（有数据跳过，无数据自动下载/降级）=="
# --fresh 只清空训练 checkpoint，不影响数据缓存（数据未变无需重下）。
# 若确需强制重下数据，单独运行：python scripts/01_download_data.py --dataset german --force
python scripts/01_download_data.py --dataset all || { echo "[FATAL] 数据准备失败"; exit 1; }

echo ""
echo "== [2/5] 表格轨道（LR/RF/LightGBM/XGBoost x $SEEDS 种子）=="
python scripts/02_train_tabular.py --data data/german_credit.csv --label label \
  --seeds "$SEEDS" $COMMON_ARGS || echo "[WARN] 表格轨道部分失败，继续"

echo ""
echo "== [3/5] 图轨道（SGC/GCN-lite/SAGE-lite/StreamTemporal x random/temporal）=="
python scripts/03_train_gnn.py --seeds "$SEEDS" --k $K_VALUES $COMMON_ARGS \
  || echo "[WARN] 图轨道部分失败，继续"

echo ""
echo "== [4/5] 文本轨道（model=$TEXT_MODEL，失败自动降级 TF-IDF）=="
python scripts/04_train_tiny_llm.py --model "$TEXT_MODEL" --seeds "$SEEDS" $COMMON_ARGS \
  || echo "[WARN] 文本轨道部分失败，继续"

echo ""
echo "== [5/5] 统一评估（汇总 + 泄露量化 + 论文表格）=="
python scripts/05_evaluate.py --leakage || { echo "[FATAL] 评估失败"; exit 1; }

echo ""
echo "=============================================="
echo " ALL DONE - 结果见 results/final_summary.csv"
echo " 论文表格: results/paper_tables.md"
echo " 泄露量化: results/leakage_quantification.csv"
echo " 数据来源: configs/provenance.jsonl"
echo "=============================================="
