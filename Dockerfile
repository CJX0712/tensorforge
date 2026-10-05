FROM python:3.12-slim

WORKDIR /app

# 运行时依赖（纯 numpy，离线可跑）
COPY requirements.txt requirements.lock.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# 安全门禁 + 冒烟 + 确定性校验
RUN pip install --no-cache-dir ruff pytest && \
    ruff check . && ruff format --check . && \
    python -m pytest -q -W ignore::UserWarning && \
    python examples/run_demo.py

CMD ["python", "examples/run_demo.py"]
