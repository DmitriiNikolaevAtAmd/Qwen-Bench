"""Tokenizer resolution and HuggingFace authentication helpers."""
import os
from pathlib import Path

from omegaconf import DictConfig


def get_tokenizer_path(cfg: DictConfig) -> str:
    """Resolve tokenizer: prefer local cache, fallback to HuggingFace."""
    data_dir = Path(cfg.paths.data_dir)
    local = data_dir / "tokenizers" / cfg.model.name
    if local.is_dir():
        return str(local)
    return cfg.model.tokenizer_path


def ensure_hf_token(repo_id: str) -> None:
    """Ensure HF_TOKEN is set for gated repos."""
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if token:
        os.environ.setdefault("HF_TOKEN", token)
        os.environ.setdefault("HUGGINGFACE_HUB_TOKEN", token)
    elif repo_id.startswith("meta-llama/"):
        raise RuntimeError(
            f"HF_TOKEN required for gated repo: {repo_id}. "
            "Set HF_TOKEN in the environment or in config.env."
        )
