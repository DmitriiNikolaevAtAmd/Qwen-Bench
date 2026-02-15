FROM nvcr.io/nvidia/nemo:25.04

# Remove APEX to disable gradient_accumulation_fusion (no AMD equivalent)
RUN pip uninstall -y apex || true

RUN apt-get update && apt-get install -y \
    git \
    neovim \
    ranger \
    zip \
    tmux \
    tree \
    fish \
    && rm -rf /var/lib/apt/lists/*

ENV USE_APEX=NO
ENV TRANSFORMERS_NO_APEX=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONHASHSEED=42
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ENV NCCL_DEBUG=ERROR
ENV NCCL_IB_DISABLE=0
ENV NCCL_NET_GDR_LEVEL=PHB

RUN mkdir -p /workspace/code
WORKDIR /workspace/code
COPY nvd-requirements.txt /workspace/code/
RUN pip install --no-cache-dir --upgrade-strategy only-if-needed -r nvd-requirements.txt

SHELL ["/usr/bin/fish", "-c"]
CMD ["/usr/bin/fish"]
