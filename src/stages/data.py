"""Data pipeline stage: load -> split -> store.

Orchestrates the three data sub-steps using values from the Hydra config.
"""
import logging
from pathlib import Path

from omegaconf import DictConfig

log = logging.getLogger(__name__)


def run(cfg: DictConfig) -> None:
    """Run the full data pipeline: load, split, store."""
    from src.data.load import load_pseudo_camera
    from src.data.split import split_shards
    from src.data.store import store_metadata

    data_dir = str(cfg.paths.data_dir)
    samples = int(cfg.data.samples)
    train_split = float(cfg.data.train_split)
    seed = int(cfg.seed)

    Path(data_dir).mkdir(parents=True, exist_ok=True)
    Path(str(cfg.paths.hf_home)).mkdir(parents=True, exist_ok=True)

    raw_jsonl = f"{data_dir}/pseudo-camera-raw.jsonl"
    wds_dir = f"{data_dir}/webdataset"

    # Step 1: Load image-caption pairs
    log.info("Step 1/3: Loading pseudo-camera-10k images + captions")
    load_pseudo_camera(num_samples=samples, output_file=raw_jsonl)

    # Step 2: Split into WebDataset shards
    log.info("Step 2/3: Splitting into WebDataset shards")
    split_shards(
        input_file=raw_jsonl,
        output_dir=wds_dir,
        max_samples=samples,
        train_split=train_split,
        max_per_shard=1000,
        seed=seed,
    )

    # Step 3: Store Megatron-Energon metadata
    log.info("Step 3/3: Storing Megatron-Energon metadata")
    store_metadata(input_dir=wds_dir)

    log.info("Data pipeline complete")
