from pathlib import Path

from omegaconf import DictConfig
from rich.text import Text

from src import console


def _step(cfg: DictConfig, n: int, title: str, detail: str = "") -> None:
    c = cfg.theme.colors
    label = Text.assemble(
        (f" {n}. ", f"bold {c.accent} on {c.data}"),
        (" ", ""),
        (title, f"bold {c.accent}"),
    )
    if detail:
        label.append(f"  {detail}", style="dim")
    console.print(label)
    console.print()


def run(cfg: DictConfig) -> None:
    from src.data.load import load_pseudo_camera
    from src.data.split import split_shards
    from src.data.store import store_metadata

    data_dir = str(cfg.paths.data_dir)
    samples = int(cfg.data.samples)
    train_split = float(cfg.data.train_split)
    seed = int(cfg.seed)
    syntax_theme = str(cfg.theme.syntax)

    Path(data_dir).mkdir(parents=True, exist_ok=True)
    Path(str(cfg.paths.hf_home)).mkdir(parents=True, exist_ok=True)

    raw_jsonl = f"{data_dir}/pseudo-camera-raw.jsonl"
    wds_dir = f"{data_dir}/webdataset"

    _step(cfg, 1, "Load", "pseudo-camera-10k (images + captions)")
    load_pseudo_camera(num_samples=samples, output_file=raw_jsonl)
    console.print()

    _step(cfg, 2, "Split", "into WebDataset shards")
    split_shards(
        input_file=raw_jsonl,
        output_dir=wds_dir,
        max_samples=samples,
        train_split=train_split,
        max_per_shard=1000,
        seed=seed,
    )
    console.print()

    _step(cfg, 3, "Store", "Megatron-Energon metadata")
    store_metadata(input_dir=wds_dir, syntax_theme=syntax_theme)
