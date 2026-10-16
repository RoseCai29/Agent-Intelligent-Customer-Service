FROM python:3.11-slim

WORKDIR /app

# 依赖层（缓存友好）
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 应用层
COPY app ./app
COPY config.yaml .
# 只拷 wiki 源文档进镜像；运行时数据库（*.db*）在构建层生成/挂载，不固化进镜像层
COPY data/wiki ./data/wiki

# 数据目录（SQLite 落盘：运行时用环境变量注入，而非写死的死键）
ENV KEFU_SERVER_STORE_PATH=/app/data/kefu.db
RUN mkdir -p /app/data

EXPOSE 8000

# 启动：--factory 调用 make_app()，否则 Runtime 永不构建、所有端点全 503
CMD ["uvicorn", "app.server:make_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]

# 健康检查：容器启动后 /health 就绪才算 healthy，供 compose depends_on 依赖
HEALTHCHECK --interval=10s --timeout=5s --start-period=15s --retries=3 \
  CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)" || exit 1