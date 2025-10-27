FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

# Use a single, consistent environment name
ARG VENV_NAME="cosyvoice_vllm"
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

SHELL ["/bin/bash", "--login", "-c"]

# Install system dependencies
RUN apt-get update -y --fix-missing && \
    apt-get install -y git build-essential curl wget ffmpeg unzip git-lfs sox libsox-dev && \
    apt-get clean && \
    git lfs install

# --- Install Miniforge (Conda base) ---
COPY docker/Miniforge3-Linux-x86_64.sh /tmp/miniforge.sh
RUN /bin/bash /tmp/miniforge.sh -b -p /opt/conda && \
    rm /tmp/miniforge.sh

ENV PATH=/opt/conda/bin:$PATH

# --- Install Mamba for faster environment management ---
RUN conda install -y -c conda-forge mamba

# --- Conda configuration (applies to Mamba too) ---
RUN conda config --add channels conda-forge && \
    conda config --set channel_priority strict

# --- Create Python environment using MAMBA (fast!) ---
RUN mamba create -y -n ${VENV_NAME} python=3.10

# --- Auto-activate environment in bash shells ---
RUN echo "source /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate ${VENV_NAME}" >> ~/.bashrc

# --- Clone project ---
WORKDIR /workspace
ENV PYTHONPATH="${PYTHONPATH}:/workspace/CosyVoice:/workspace/CosyVoice/third_party/Matcha-TTS"
RUN git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git

WORKDIR /workspace/CosyVoice

# --- Prepare filtered requirements ---
RUN echo 'numpy<2.0' > constraints.txt && \
    grep -v '^numpy' requirements.txt | grep -v '^transformers' | grep -v '^vllm' > requirements_filtered.txt

# --- Install conda packages with MAMBA ---
RUN mamba install -y -n ${VENV_NAME} -c conda-forge pynini==2.1.5

# --- Install PyPI packages with pip via conda run (safe in Docker) ---
RUN conda run -n ${VENV_NAME} pip install -r requirements_filtered.txt -c constraints.txt && \
    conda run -n ${VENV_NAME} pip install pyloudnorm flask waitress transformers==4.53.3 vllm==0.9.0 -c constraints.txt

# --- Upgrade onnxruntime to CUDA 12 compatible version ---
RUN conda run -n ${VENV_NAME} pip uninstall -y onnxruntime onnxruntime-gpu && \
    conda run -n ${VENV_NAME} pip install onnxruntime-gpu \
        --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/

# Install WebUI dependencies
RUN pip install "gradio==5.4.0"

# --- Cleanup ---
RUN rm -f constraints.txt requirements_filtered.txt

# --- Copy application code ---
COPY api/ /workspace/CosyVoice/api/
COPY utils/ /workspace/CosyVoice/utils/

# --- Expose and health check ---
EXPOSE 9996
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:9996/health || exit 1