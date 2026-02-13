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

    console.print("[bold]Step 1/3:[/bold] Loading pseudo-camera-10k images + captions")
    load_pseudo_camera(num_samples=samples, output_file=raw_jsonl)

    console.print("[bold]Step 2/3:[/bold] Splitting into WebDataset shards")
    split_shards(
        input_file=raw_jsonl,
        output_dir=wds_dir,
        max_samples=samples,
        train_split=train_split,
        max_per_shard=1000,
        seed=seed,
    )

    console.print("[bold]Step 3/3:[/bold] Storing Megatron-Energon metadata")
    store_metadata(input_dir=wds_dir)

    console.print("[bold green]Data pipeline complete[/bold green]")
