ARG BASE_IMAGE=rocm/pytorch:rocm6.4.1_ubuntu22.04_py3.10_pytorch_release_2.6.0
FROM ${BASE_IMAGE}

RUN apt-get update && apt-get install -y \
    git \
    neovim \
    fish \
    && rm -rf /var/lib/apt/lists/*

# ENV PYTHONUNBUFFERED=1
# ENV PYTHONHASHSEED=42
# ENV PYTHONWARNINGS="ignore::UserWarning,ignore::FutureWarning,ignore::DeprecationWarning"
# ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

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

# ARG PYTORCH_INDEX=https://download.pytorch.org/whl/rocm6.3

RUN mkdir -p /workspace/code
WORKDIR /workspace/code

COPY reqs/rocm-requirements.txt /workspace/code/reqs/
RUN pip install --no-cache-dir --upgrade pip packaging wheel setuptools && \
    pip install --no-cache-dir --upgrade-strategy only-if-needed -r reqs/rocm-requirements.txt

# RUN pip uninstall -y torch torchvision torchaudio && \
#     pip install --no-cache-dir --pre torch torchvision torchaudio --index-url "${PYTORCH_INDEX}" && \
#     pip install --no-cache-dir llamafactory[metrics,deepspeed]

COPY . /workspace/code/

SHELL ["/bin/fish", "-c"]
CMD ["/bin/fish"]
