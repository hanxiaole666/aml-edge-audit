# TinyAML-Bench 可复现环境
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MPLBACKEND=Agg

WORKDIR /app

# 系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl git \
    && rm -rf /var/lib/apt/lists/*

# Python 依赖（锁定requirements，先装核心保证可跑）
COPY requirements.txt /app/
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 代码
COPY scripts/ /app/scripts/
COPY run_all.sh /app/
COPY data/ /app/data/
RUN chmod +x /app/run_all.sh

# 冒烟自检：数据自动下载/跳过 + 全流程 quick
RUN bash run_all.sh --quick --seeds 2

# 默认：完整复核（5 种子）；可用 --quick 覆盖
CMD ["bash", "run_all.sh", "--seeds", "5"]
