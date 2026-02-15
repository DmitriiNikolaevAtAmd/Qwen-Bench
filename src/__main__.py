import hydra
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    from src.core.main import run
    run(cfg)


if __name__ == "__main__":
    main()
