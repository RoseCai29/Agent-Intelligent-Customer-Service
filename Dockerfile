FROM python:3.11-slim

WORKDIR /app

# 依赖层（缓存友好）
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 应用层
COPY app ./app
COPY config.yaml .
COPY data ./data

# 数据目录（SQLite 落盘）
ENV KEFU_STORE_PATH=data/kefu.db
RUN mkdir -p data

EXPOSE 8000

CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8000"]