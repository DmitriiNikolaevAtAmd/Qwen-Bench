#!/usr/bin/env python3
"""
Validate that all expected training outputs exist before running comparisons.

Outputs follow the naming convention:
    train_{platform}_{framework}_{model}_{dataset}.json

Usage:
    python3 validate_outputs.py [--platform nvd|amd|all] [--results-dir ./output]
    
Exit codes:
    0 - All expected outputs are present
    1 - Some outputs are missing
"""

import argparse
import os
import sys
from pathlib import Path


DATASETS = ["bc", "c4"]

# Expected (framework, model) pairs for each platform
# Only includes frameworks that are actually run by run_nvd.sh and run_amd.sh
NVD_PAIRS = [
    ("nemo", "llama"),
    ("nemo", "qwen"),
]

AMD_PAIRS = [
    ("prim", "llama"),
    ("prim", "qwen"),
]

# Full list of all available frameworks (for reference)
# NVD_ALL = ["deep", "fsdp", "mega", "nemo", "tran"]
# AMD_ALL = ["deep", "fsdp", "mega", "nemo", "prim", "tran"]


def get_expected_files(platform: str, datasets: list[str]) -> list[str]:
    """Generate expected filenames for the given platform and datasets."""
    if platform == "nvd":
        pairs = NVD_PAIRS
    elif platform == "amd":
        pairs = AMD_PAIRS
    elif platform == "all":
        pairs = [("nvd", fw, model) for fw, model in NVD_PAIRS] + \
                [("amd", fw, model) for fw, model in AMD_PAIRS]
    else:
        raise ValueError(f"Unknown platform: {platform}")
    
    expected = []
    for item in pairs:
        if len(item) == 3:
            plat, fw, model = item
        else:
            fw, model = item
            plat = platform
        for ds in datasets:
            expected.append(f"train_{plat}_{fw}_{model}_{ds}.json")
    
    return sorted(expected)


def validate_outputs(results_dir: str, platform: str, datasets: list[str]) -> tuple[list[str], list[str]]:
    """
    Check which expected outputs exist and which are missing.
    
    Returns:
        Tuple of (found_files, missing_files)
    """
    results_path = Path(results_dir)
    expected = get_expected_files(platform, datasets)
    
    found = []
    missing = []
    
    for filename in expected:
        filepath = results_path / filename
        if filepath.exists():
            found.append(filename)
        else:
            missing.append(filename)
    
    return found, missing


def main():
    parser = argparse.ArgumentParser(
        description='Validate that all expected training outputs exist'
    )
    default_dir = os.environ.get('OUTPUT_DIR', './output')
    parser.add_argument('--results-dir', default=default_dir,
                       help='Directory containing benchmark JSON files (default: ./output)')
    parser.add_argument('--platform', choices=['nvd', 'amd', 'all'], default='all',
                       help='Platform to validate (default: all)')
    parser.add_argument('--dataset', nargs='*', default=None,
                       help='Datasets to validate (default: all known: bc c4)')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Only output missing files, no success messages')
    
    args = parser.parse_args()
    
    datasets = args.dataset if args.dataset else DATASETS
    
    results_path = Path(args.results_dir)
    if not results_path.exists():
        print(f"[x] Output directory does not exist: {args.results_dir}")
        return 1
    
    found, missing = validate_outputs(args.results_dir, args.platform, datasets)
    
    if not args.quiet:
        print("=" * 70)
        print(f"TRAINING OUTPUT VALIDATION - Platform: {args.platform.upper()}")
        print("=" * 70)
        print(f"\nResults directory: {args.results_dir}")
        print(f"Datasets: {', '.join(datasets)}")
        print(f"\nFound {len(found)}/{len(found) + len(missing)} expected outputs:")
    
    if found and not args.quiet:
        for f in found:
            print(f"  [+] {f}")
    
    if missing:
        if not args.quiet:
            print(f"\nMissing {len(missing)} outputs:")
        for f in missing:
            print(f"  [x] {f}")
        
        if not args.quiet:
            print("\n" + "-" * 70)
            print("VALIDATION FAILED")
            print("-" * 70)
            print("\nRun the corresponding training scripts to generate missing outputs:")
            
            for f in missing:
                # train_amd_prim_llama_bc.json -> train/train_amd_prim_llama.sh
                stem = f.replace(".json", "")
                # Remove dataset suffix to get the script name
                parts = stem.rsplit("_", 1)
                script_base = parts[0]
                print(f"  ./train/{script_base}.sh  ->  output/{f}")
        
        return 1
    
    if not args.quiet:
        print("\n" + "-" * 70)
        print("VALIDATION PASSED - All expected outputs are present")
        print("-" * 70)
        print("\nYou can now run the comparison scripts:")
        if args.platform in ("nvd", "all"):
            print("  ./evaluate/compare_nvd.sh")
        if args.platform in ("amd", "all"):
            print("  ./evaluate/compare_amd.sh")
        if args.platform == "all":
            print("  ./evaluate/compare_all.sh")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
