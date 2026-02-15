#!/usr/bin/env python3
"""Standalone eval script -- load benchmark JSONs and build compare.png.

Runs locally without Docker, Hydra, Rich, or PyTorch.
Only requires: matplotlib, numpy (seaborn optional).

Usage:
    python scripts/eval.py [output_dir]
    make eval
"""
import importlib.util
import sys
from pathlib import Path

# Load src/eval/compare.py directly (bypass src/__init__.py which needs rich/hydra)
_ROOT = Path(__file__).resolve().parent.parent
_COMPARE = _ROOT / "src" / "eval" / "compare.py"

spec = importlib.util.spec_from_file_location("compare", _COMPARE)
compare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compare)


def main() -> None:
    output_dir = sys.argv[1] if len(sys.argv) > 1 else str(_ROOT / "output")
    output_path = Path(output_dir)

    if not output_path.exists():
        print(f"Error: {output_path} does not exist")
        sys.exit(1)

    # 1. Load benchmarks (recursive scan for train_*.json)
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
        tps = d.get("performance_metrics", {}).get("tokens_per_second_per_gpu", 0)
        print(f"  {plat} {fw} {mdl}  --  {tps:,.0f} tok/s/GPU")

    # 2. Generate comparison plot
    plot_path = output_path / "compare.png"
    compare.create_comparison_plot(benchmarks, str(plot_path))
    print(f"\nSaved: {plot_path}")

    # 3. Print summary
    summary = compare.print_summary(benchmarks)
    print(f"\n{summary}")


if __name__ == "__main__":
    main()
