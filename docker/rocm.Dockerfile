ARG BASE_IMAGE=rocm/pytorch:rocm7.2_ubuntu22.04_py3.10_pytorch_release_2.9.1
FROM ${BASE_IMAGE}

RUN apt-get update && apt-get install -y \
    git \
    neovim \
    fish \
    && rm -rf /var/lib/apt/lists/*

# ENV PYTHONUNBUFFERED=1
# ENV PYTHONHASHSEED=42
# ENV PYTHONWARNINGS="ignore::UserWarning,ignore::FutureWarning,ignore::DeprecationWarning"

# ENV HSA_NO_SCRATCH_RECLAIM=1
# ENV HSA_ENABLE_SDMA=1
# ENV HSA_FORCE_FINE_GRAIN_PCIE=1

# ENV TOKENIZERS_PARALLELISM=false
# ENV TRANSFORMERS_VERBOSITY=error
# ENV HF_HUB_DISABLE_PROGRESS_BARS=1

# ENV RCCL_DEBUG=ERROR
# ENV NCCL_DEBUG=ERROR
# ENV GLOO_LOG_LEVEL=ERROR
# ENV NCCL_NET_GDR_LEVEL=PHB
# ENV NCCL_IB_DISABLE=0
# ENV RCCL_MSCCL_ENABLE=0

# ENV VLLM_WORKER_MULTIPROC_METHOD=spawn
# ENV MAX_JOBS=16
# ENV FLASH_ATTENTION_FORCE_BUILD=TRUE
# ENV PIP_ROOT_USER_ACTION=ignore

RUN mkdir -p /workspace/code
WORKDIR /workspace/code

COPY reqs/rocm-requirements.txt /workspace/code/reqs/
RUN pip install --no-cache-dir --upgrade pip packaging wheel setuptools && \
    pip install --no-cache-dir -r reqs/rocm-requirements.txt

# Ensure numpy/pandas stay compatible after all installs (Python 3.10 caps numpy <2.3)
RUN pip install --no-cache-dir "numpy>=2.0.0,<2.3" "pandas>=2.0.0"

COPY . /workspace/code/

SHELL ["/bin/fish", "-c"]
CMD ["/bin/fish"]
