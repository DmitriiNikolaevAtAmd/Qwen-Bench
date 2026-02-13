import logging
import os
import time

import hydra
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)

VALID_STAGES = ("data", "train", "wrap", "purge", "all")


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    stage = cfg.stage
    if stage not in VALID_STAGES:
        raise ValueError(
            f"Unknown stage: '{stage}'. Must be one of {VALID_STAGES}"
        )

    log.info("Qwen-Bench stage=%s", stage)
    log.info("Resolved config:\n%s", OmegaConf.to_yaml(cfg))

    os.environ["HF_HOME"] = str(cfg.paths.hf_home)
    os.environ["HF_DATASETS_CACHE"] = str(cfg.paths.hf_datasets_cache)

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
        for name in ("data", "train", "wrap"):
            log.info("=== Pipeline stage: %s ===", name)
            t0 = time.time()
            stages[name](cfg)
            log.info("=== %s completed in %.1fs ===", name, time.time() - t0)
    else:
        t0 = time.time()
        stages[stage](cfg)
        log.info("Stage '%s' completed in %.1fs", stage, time.time() - t0)


if __name__ == "__main__":
    main()
