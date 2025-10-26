FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

# Use a single, consistent environment name
ARG VENV_NAME="cosyvoice_vllm"
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

SHELL ["/bin/bash", "--login", "-c"]

RUN apt-get update -y --fix-missing && \
    apt-get install -y git build-essential curl wget ffmpeg unzip git-lfs sox libsox-dev && \
    apt-get clean && \
    git lfs install

# --- Conda Installation ---
# RUN wget --quiet https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O ~/miniforge.sh && \
#     /bin/bash ~/miniforge.sh -b -p /opt/conda && \
#     rm ~/miniforge.sh
# --- Conda Installation ---
# 从构建上下文复制预先下载好的安装脚本
COPY docker/Miniforge3-Linux-x86_64.sh /tmp/miniforge.sh

# 执行安装脚本并随后清理
RUN /bin/bash /tmp/miniforge.sh -b -p /opt/conda && \
    rm /tmp/miniforge.sh

ENV PATH /opt/conda/bin:$PATH

# --- Conda Configuration ---
RUN conda config --add channels conda-forge && \
    conda config --set channel_priority strict

# --- Python Environment Setup ---
# Create the environment that will actually be used
RUN conda create -y -n ${VENV_NAME} python=3.10
# Set up the shell to auto-activate this environment
RUN echo "source /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate ${VENV_NAME}" >> ~/.bashrc

# --- Project Checkout ---
WORKDIR /workspace
ENV PYTHONPATH="${PYTHONPATH}:/workspace/CosyVoice:/workspace/CosyVoice/third_party/Matcha-TTS"
RUN git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git

WORKDIR /workspace/CosyVoice

# --- Python Dependencies Installation (Optimized for Caching) ---
RUN echo 'numpy<2.0' > constraints.txt && \
    grep -v '^numpy' requirements.txt | grep -v '^transformers' | grep -v '^vllm' > requirements_filtered.txt

RUN conda run -n ${VENV_NAME} conda install -y -c conda-forge pynini==2.1.5 && \
    conda run -n ${VENV_NAME} pip install \
        -r requirements_filtered.txt \
        -c constraints.txt && \
    conda run -n ${VENV_NAME} pip install \
        pyloudnorm flask waitress transformers==4.53.3 vllm==0.9.0 \
        -c constraints.txt


# STEP 5: Upgrade onnxruntime
RUN conda run -n ${VENV_NAME} pip uninstall -y onnxruntime onnxruntime-gpu && \
    conda run -n ${VENV_NAME} pip install \
        onnxruntime-gpu \
        --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/

RUN rm constraints.txt requirements_filtered.txt

# --- Application Setup ---
COPY api/ /workspace/CosyVoice/api/
COPY utils/ /workspace/CosyVoice/utils/

# --- Expose Port & Health Check ---
EXPOSE 9996
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:9996/health || exit 1

