FROM rocm/primus:v25.11

RUN apt-get update && apt-get install -y \
    git \
    neovim \
    ranger \
    zip \
    tree \
    fish \
    iproute2 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1
ENV PYTHONHASHSEED=42
ENV PYTHONWARNINGS="ignore::UserWarning,ignore::FutureWarning,ignore::DeprecationWarning"
ENV PYTORCH_ALLOC_CONF=expandable_segments:True
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ENV HSA_NO_SCRATCH_RECLAIM=1
ENV HSA_ENABLE_SDMA=1
ENV HSA_FORCE_FINE_GRAIN_PCIE=1

ENV TOKENIZERS_PARALLELISM=false
ENV TRANSFORMERS_VERBOSITY=error
ENV HF_HUB_DISABLE_PROGRESS_BARS=1

ENV RCCL_DEBUG=ERROR
ENV NCCL_DEBUG=ERROR
ENV GLOO_LOG_LEVEL=ERROR
ENV NCCL_NET_GDR_LEVEL=PHB
ENV NCCL_IB_DISABLE=0
ENV RCCL_MSCCL_ENABLE=0

ENV TRACELENS_ENABLED=1
ENV PROFILE_ITERS="7,9"

RUN mkdir -p /workspace/code
WORKDIR /workspace/code

COPY amd-requirements.txt /workspace/code/
RUN pip install --no-cache-dir -r amd-requirements.txt

COPY . /workspace/code/

SHELL ["/bin/fish", "-c"]
CMD ["/bin/fish"]
