"""Eval stage: extract metrics from training logs and build dashboard.

Supports two workflows:
  a) Post-training (inside container): extract from logs -> save JSON -> plot
  b) Local comparison: existing JSONs in output subdirs -> plot master compare.png

Steps:
1. Check for existing benchmark JSONs (recursive scan).
2. If none found, extract from training logs and create them.
3. Load all benchmark JSONs.
4. Generate a 2x3 comparison plot (compare.png).
5. Print a summary table.
"""
import torch
from pathlib import Path

from omegaconf import DictConfig
from rich.panel import Panel

from src import console
from src.eval.compare import create_comparison_plot, load_benchmarks, print_summary
from src.eval.extract import extract_from_log, save_benchmark


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

    # -- 1. Scan for existing benchmark JSONs ---------------------------------
    _step(cfg, 1, "Scan", f"Looking for benchmarks in {output_dir}")

    existing_jsons = sorted(output_dir.glob("**/train_*.json"))
    _kv(cfg, "json files", str(len(existing_jsons)))

    # -- 2. Extract from logs (only if no JSONs found) ------------------------
    if not existing_jsons:
        log_files = sorted(output_dir.glob("**/*.log")) + sorted(output_dir.glob("**/stdout*"))
        log_files += sorted(output_dir.glob("**/*.txt"))
        _kv(cfg, "log files", str(len(log_files)))
        console.print()

        _step(cfg, 2, "Extract", "Parsing Megatron log output")

        is_rocm = hasattr(torch.version, "hip") and torch.version.hip is not None
        platform = "rocm" if is_rocm else "cuda"

        for log_file in log_files:
            data = extract_from_log(str(log_file))
            if data.get("step_times"):
                filepath = save_benchmark(
                    data,
                    output_dir=str(output_dir),
                    platform=platform,
                    framework="megatron",
                    model=m.get("name", "model"),
                    dataset=t.get("dataset", "benchmark"),
                )
                _kv(cfg, "saved", str(filepath))

    console.print()

    # -- 3. Load all benchmark JSONs (recursive) ------------------------------
    _step(cfg, 3, "Load", f"Reading benchmark files from {output_dir}")

    benchmarks = load_benchmarks(str(output_dir))
    _kv(cfg, "loaded", str(len(benchmarks)))
    console.print()

    if not benchmarks:
        console.print(f"  [{c.warn}]No benchmark data found.[/{c.warn}]")
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

    console.print(Panel(
        summary_text,
        title=f"[{c.eval}]Benchmark Results[/{c.eval}]",
        border_style="dim",
        padding=(1, 2),
    ))
    console.print()
