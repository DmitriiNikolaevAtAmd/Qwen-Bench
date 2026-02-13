#!/usr/bin/env python3
"""
Qwen-Bench CLI -- Hydra-based entrypoint for all benchmark stages.

Usage (inside container):
    python -m src stage=data
    python -m src stage=train
    python -m src stage=all
    python -m src stage=train training.learning_rate=1e-3
    python -m src --multirun training.learning_rate=1e-4,3e-4,1e-3

Usage (from host via Docker):
    ./scripts/run.sh stage=data
    ./scripts/run.sh stage=train training=full
"""
import logging
import os
import time

import hydra
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)

VALID_STAGES = ("data", "train", "wrap", "purge", "all")


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    stage = cfg.stage
    if stage not in VALID_STAGES:
        raise ValueError(
            f"Unknown stage: '{stage}'. Must be one of {VALID_STAGES}"
        )

    log.info("Qwen-Bench stage=%s", stage)
    log.info("Resolved config:\n%s", OmegaConf.to_yaml(cfg))

    # Export HF environment variables from config
    os.environ["HF_HOME"] = str(cfg.paths.hf_home)
    os.environ["HF_DATASETS_CACHE"] = str(cfg.paths.hf_datasets_cache)

    # Lazy imports to avoid loading all stage deps at startup
    from src.stages import data as data_stage
    from src.stages import purge as purge_stage
    from src.stages import train as train_stage
    from src.stages import wrap as wrap_stage

    stages = {
        "data": data_stage.run,
        "train": train_stage.run,
        "wrap": wrap_stage.run,
        "purge": purge_stage.run,
    }

    if stage == "all":
        pipeline = ("data", "train", "wrap")
        for name in pipeline:
            log.info("=== Pipeline stage: %s ===", name)
            t0 = time.time()
            stages[name](cfg)
            elapsed = time.time() - t0
            log.info("=== %s completed in %.1fs ===", name, elapsed)
    else:
        t0 = time.time()
        stages[stage](cfg)
        elapsed = time.time() - t0
        log.info("Stage '%s' completed in %.1fs", stage, elapsed)


if __name__ == "__main__":
    main()
