"""Purge stage: remove outputs, caches, and optionally data."""
import logging
import shutil
from pathlib import Path

from omegaconf import DictConfig

log = logging.getLogger(__name__)


def run(cfg: DictConfig, *, with_data: bool = False) -> None:
    """Remove output files and HF caches.

    Args:
        cfg: Hydra config with paths.
        with_data: If True, also remove the data directory.
    """
    output_dir = Path(cfg.paths.output_dir)
    hf_home = Path(cfg.paths.hf_home)
    hf_datasets_cache = Path(cfg.paths.hf_datasets_cache)

    # Remove output files
    if output_dir.exists():
        log.info("Removing output files: %s", output_dir)
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Remove HF caches
    for cache_dir in (hf_home, hf_datasets_cache):
        if cache_dir.exists():
            log.info("Removing cache: %s", cache_dir)
            shutil.rmtree(cache_dir)

    # Optionally remove data
    if with_data:
        data_dir = Path(cfg.paths.data_dir)
        if data_dir.exists():
            log.info("Removing data: %s", data_dir)
            shutil.rmtree(data_dir)
        data_dir.mkdir(parents=True, exist_ok=True)

    log.info("Purge complete")
