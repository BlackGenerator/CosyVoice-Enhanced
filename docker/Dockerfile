# 使用轻量级 Python 基础镜像（无 CUDA）
FROM python:3.10-slim

# 设置工作目录
WORKDIR /workspace/CosyVoice

# 安装系统依赖（用于 librosa, pyworld 等）
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        gcc \
        g++ \
        libsndfile1-dev \
        libsox-dev \
        ffmpeg \
        git \
        && rm -rf /var/lib/apt/lists/*

# 升级 pip
RUN pip install --no-cache-dir --upgrade pip

# 复制并安装 Python 依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制源代码
COPY . .

# 创建日志和临时目录
RUN mkdir -p logs tmp

# 默认使用 CPU 推理
ENV OMP_NUM_THREADS=4
ENV PYTHONPATH=/workspace/CosyVoice

# 暴露端口（API 和 WebUI）
EXPOSE 9996 8000

# 默认启动命令由 docker-compose 覆盖
CMD ["python", "-c", "print('Use docker-compose to run api or webui')"]