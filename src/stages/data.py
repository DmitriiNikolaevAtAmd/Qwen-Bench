from pathlib import Path

from omegaconf import DictConfig

from src import console


def run(cfg: DictConfig) -> None:
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

    console.print()
    console.print("[bold cyan]1.[/bold cyan] [bold]Load[/bold] pseudo-camera-10k (images + captions)")
    console.print()
    load_pseudo_camera(num_samples=samples, output_file=raw_jsonl)

    console.print()
    console.print("[bold cyan]2.[/bold cyan] [bold]Split[/bold] into WebDataset shards")
    console.print()
    split_shards(
        input_file=raw_jsonl,
        output_dir=wds_dir,
        max_samples=samples,
        train_split=train_split,
        max_per_shard=1000,
        seed=seed,
    )

    console.print()
    console.print("[bold cyan]3.[/bold cyan] [bold]Store[/bold] Megatron-Energon metadata")
    console.print()
    store_metadata(input_dir=wds_dir)
