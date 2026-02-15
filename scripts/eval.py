#!/usr/bin/env python3
"""Standalone eval script -- extract metrics from logs, build compare.png.

Runs locally without Docker, Hydra, Rich, or PyTorch.
Requires: matplotlib, numpy, seaborn, pyyaml.

Workflow:
  1. Scan output/ for training_*.log files
  2. Extract time series (step_times, loss, LR, grad_norm, memory) from each
  3. Save as abbreviated JSON (train_{platform}_{fw}_{model}_{ds}.json)
  4. Load all JSONs, compute stats from arrays + config.yaml
  5. Generate comparison dashboard (compare.png)

Usage:
    python scripts/eval.py [output_dir]
    make eval
"""
import importlib.util
import re
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent

# Load src/eval modules directly (bypass src/__init__.py which needs rich/hydra)
_EXTRACT = _ROOT / "src" / "eval" / "extract.py"
_COMPARE = _ROOT / "src" / "eval" / "compare.py"

spec_e = importlib.util.spec_from_file_location("extract", _EXTRACT)
extract = importlib.util.module_from_spec(spec_e)
spec_e.loader.exec_module(extract)

spec_c = importlib.util.spec_from_file_location("compare", _COMPARE)
compare = importlib.util.module_from_spec(spec_c)
spec_c.loader.exec_module(compare)


def _parse_log_filename(stem: str) -> dict:
    """Parse training_{platform}_{framework}_{model}_{dataset} from log stem."""
    # training_cuda_megatron_qwen_pseudo_camera  or  training_cuda_mega_qwen_pc
    parts = stem.split("_", 1)  # ["training", "cuda_megatron_qwen_pseudo_camera"]
    if len(parts) < 2:
        return {}
    rest = parts[1].split("_")
    if len(rest) < 3:
        return {}
    return {
        "platform": rest[0],
        "framework": rest[1],
        "model": rest[2],
        "dataset": "_".join(rest[3:]) if len(rest) > 3 else "benchmark",
    }


def main() -> None:
    output_dir = sys.argv[1] if len(sys.argv) > 1 else str(_ROOT / "output")
    output_path = Path(output_dir)

    if not output_path.exists():
        print(f"Error: {output_path} does not exist")
        sys.exit(1)

    # -- 1. Extract from logs --------------------------------------------------
    log_files = sorted(output_path.glob("**/training_*.log"))
    if log_files:
        print(f"Extracting from {len(log_files)} log(s):")
        for log_file in log_files:
            meta = _parse_log_filename(log_file.stem)
            if not meta:
                print(f"  skip  {log_file.name} (cannot parse filename)")
                continue

            data = extract.extract_from_log(str(log_file))
            if not data.get("step_times"):
                print(f"  skip  {log_file.name} (no step data)")
                continue

            n_mem = len(data.get("memory_allocated", []))
            filepath = extract.save_benchmark(
                data,
                output_dir=str(log_file.parent),
                platform=meta["platform"],
                framework=meta["framework"],
                model=meta["model"],
                dataset=meta["dataset"],
            )
            print(f"  saved {filepath.name}  ({len(data['step_times'])} steps, {n_mem} mem samples)")
        print()

    # -- 2. Load benchmarks ----------------------------------------------------
    benchmarks = compare.load_benchmarks(str(output_path))
    if not benchmarks:
        print(f"No benchmark files (train_*.json) found in {output_path}")
        sys.exit(1)

    print(f"Loaded {len(benchmarks)} benchmark(s):")
    for key in sorted(benchmarks):
        d = benchmarks[key]
        plat = {"cuda": "NVIDIA", "rocm": "AMD"}.get(d.get("_platform", ""), d.get("_platform", ""))
        fw = compare.FRAMEWORK_DISPLAY.get(d.get("_framework", ""), d.get("_framework", ""))
        mdl = d.get("_model", "unknown").capitalize()
        tps = d.get("performance_metrics", {}).get("tokens_per_second_per_gpu")
        mem = d.get("memory_metrics", {})
        alloc = mem.get("avg_memory_allocated_gb")
        resv = mem.get("avg_memory_reserved_gb")
        parts = [f"{plat} {fw} {mdl}"]
        if tps:
            parts.append(f"{tps:,.0f} tok/s/GPU")
        if alloc:
            parts.append(f"mem {alloc:.1f}/{resv:.1f} GB" if resv else f"mem {alloc:.1f} GB")
        print(f"  {' — '.join(parts)}")

    # -- 3. Generate comparison plot -------------------------------------------
    plot_path = output_path / "compare.png"
    compare.create_comparison_plot(benchmarks, str(plot_path))
    print(f"\nSaved: {plot_path}")

    # -- 4. Print summary ------------------------------------------------------
    summary = compare.print_summary(benchmarks)
    print(f"\n{summary}")


if __name__ == "__main__":
    main()
