"""Eval stage: extract metrics from training logs and build dashboard.

1. Scan output_dir for Megatron training logs.
2. Extract metrics (loss, lr, step time, grad norm, memory) via regex.
3. Save per-run JSON benchmark files.
4. Generate a 2x3 comparison plot (compare.png).
5. Print a summary table.
"""
from pathlib import Path

from omegaconf import DictConfig
from rich.panel import Panel
from rich.table import Table

from src import console
from src.eval.compare import create_comparison_plot, load_benchmarks, print_summary
from src.eval.extract import extract_from_log


# ---------------------------------------------------------------------------
# Rich output helpers
# ---------------------------------------------------------------------------

def _step(cfg: DictConfig, n: int, title: str, detail: str = "") -> None:
    c = cfg.theme.colors
    console.print(
        f"[{c.eval}]{n}.[/{c.eval}] [bold]{title}[/bold]  [dim]{detail}[/dim]"
    )
    console.print()


def _kv(cfg: DictConfig, key: str, val: str) -> None:
    c = cfg.theme.colors
    console.print(f"  [{c.success}]{key}[/{c.success}]  {val}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(cfg: DictConfig) -> None:
    """Run the eval stage."""
    c = cfg.theme.colors
    m = cfg.model
    t = cfg.training
    output_dir = Path(cfg.paths.output_dir)

    # -- 1. Find training log(s) ----------------------------------------------
    _step(cfg, 1, "Scan", f"Looking for training logs in {output_dir}")

    log_files = sorted(output_dir.glob("**/*.log")) + sorted(output_dir.glob("**/stdout*"))
    # Also look for logs redirected to .txt
    log_files += sorted(output_dir.glob("**/*.txt"))

    _kv(cfg, "log files", str(len(log_files)))
    console.print()

    # -- 2. Extract metrics from each log -------------------------------------
    _step(cfg, 2, "Extract", "Parsing Megatron log output")

    json_files_created = []
    for log_file in log_files:
        cb = extract_from_log(
            log_file=str(log_file),
            output_dir=str(output_dir),
            model_name=m.get("name", m.get("display_name", "model")),
            platform="auto",
            framework="megatron",
            dataset=t.get("dataset", "benchmark"),
            num_gpus=getattr(t.parallel, "data", 1) if hasattr(t, "parallel") else 1,
            global_batch_size=t.get("global_batch_size"),
            sequence_length=t.get("seq_length"),
        )

        if cb.step_times:
            filepath = cb.save(max_steps=t.get("train_iters"))
            json_files_created.append(filepath)
            _kv(cfg, "saved", str(filepath))

    if not json_files_created:
        # No logs found -- check if JSON benchmarks already exist
        existing = list(output_dir.glob("train_*.json"))
        if not existing:
            console.print(f"  [{c.warn}]No training logs or benchmark files found.[/{c.warn}]")
            console.print()
            return

    _kv(cfg, "benchmarks", str(len(json_files_created)))
    console.print()

    # -- 3. Load all benchmark JSONs ------------------------------------------
    _step(cfg, 3, "Load", f"Reading benchmark files from {output_dir}")

    benchmarks = load_benchmarks(str(output_dir))
    _kv(cfg, "loaded", str(len(benchmarks)))
    console.print()

    if not benchmarks:
        console.print(f"  [{c.warn}]No benchmark data to compare.[/{c.warn}]")
        console.print()
        return

    # -- 4. Generate comparison plot ------------------------------------------
    _step(cfg, 4, "Plot", "Building 2x3 dashboard")

    plot_path = output_dir / "compare.png"
    create_comparison_plot(benchmarks, str(plot_path))
    _kv(cfg, "plot", str(plot_path))
    console.print()

    # -- 5. Summary -----------------------------------------------------------
    _step(cfg, 5, "Summary", "Performance comparison")

    summary_text = print_summary(benchmarks)

    info = Table.grid(padding=(0, 2))
    info.add_column(style="dim")
    info.add_column()
    for line in summary_text.strip().split("\n"):
        if line.startswith("-"):
            continue
        parts = line.split(None, 1)
        if len(parts) == 2:
            info.add_row(parts[0], parts[1])
        elif parts:
            info.add_row("", parts[0])

    console.print(Panel(
        summary_text,
        title=f"[{c.eval}]Benchmark Results[/{c.eval}]",
        border_style="dim",
        padding=(1, 2),
    ))
    console.print()
