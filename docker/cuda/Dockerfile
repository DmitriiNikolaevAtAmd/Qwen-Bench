ARG BASE_IMAGE=hiyouga/pytorch:th2.6.0-cu124-flashattn2.7.4-cxx11abi0-devel
FROM ${BASE_IMAGE}

RUN apt-get update && apt-get install -y \
    git \
    git-lfs \
    neovim \
    fish \
    pybind11-dev \
    && git lfs install \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1
ENV PYTHONHASHSEED=42
ENV PYTHONWARNINGS="ignore::UserWarning,ignore::FutureWarning,ignore::DeprecationWarning"
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ENV TOKENIZERS_PARALLELISM=false
ENV TRANSFORMERS_VERBOSITY=error
ENV HF_HUB_DISABLE_PROGRESS_BARS=1

ENV NCCL_DEBUG=ERROR
ENV GLOO_LOG_LEVEL=ERROR
ENV NCCL_NET_GDR_LEVEL=PHB
ENV NCCL_IB_DISABLE=0
ENV TORCH_CPP_LOG_LEVEL=ERROR
ENV TORCH_NCCL_ASYNC_ERROR_HANDLING=0

ENV VLLM_WORKER_MULTIPROC_METHOD=spawn
ENV MAX_JOBS=16
ENV FLASH_ATTENTION_FORCE_BUILD=TRUE
ENV PIP_ROOT_USER_ACTION=ignore

RUN mkdir -p /workspace/code
WORKDIR /workspace/code

COPY requirements/cuda/requirements.txt /workspace/code/requirements/cuda/
RUN pip install --no-cache-dir --upgrade pip packaging wheel setuptools && \
    pip install --no-cache-dir -r requirements/cuda/requirements.txt

# Ensure numpy/pandas stay compatible after all installs (Python 3.10 caps numpy <2.3)
RUN pip install --no-cache-dir "numpy>=2.0.0,<2.3" "pandas>=2.0.0"

# Megatron-LM (megatron.training is not pip-installable, need full repo on PYTHONPATH)
RUN pip install --no-cache-dir pybind11 && \
    git clone --depth 1 https://github.com/NVIDIA/Megatron-LM.git /opt/megatron && \
    cd /opt/megatron/megatron/core/datasets && make
ENV PYTHONPATH="/opt/megatron"

COPY . /workspace/code/

EXPOSE 8000

SHELL ["/bin/fish", "-c"]
CMD ["/bin/fish"]
