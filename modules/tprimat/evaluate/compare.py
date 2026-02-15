#!/usr/bin/env python3
"""
Universal GPU Benchmark Comparison Script

Automatically discovers and compares any available benchmark results with:
- Dynamic platform/framework/model detection
- Automatic color and style assignment
- Performance metrics (throughput, step time, memory)
- Visual plots and analysis

Usage:
    python3 compare.py [--results-dir ./output]
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import seaborn as sns

# ── Seaborn theme — clean white journal style ────────────────────────────────
sns.set_theme(
    style='whitegrid',
    font_scale=1.2,
    rc={
        'figure.facecolor': 'white',
        'figure.dpi': 150,
        'axes.facecolor': 'white',
        'axes.edgecolor': '#333333',
        'axes.linewidth': 0.5,
        'axes.titlepad': 8,
        'axes.labelpad': 6,
        'grid.color': '#E0E0E0',
        'grid.linewidth': 0.3,
        'text.color': '#222222',
        'font.family': 'sans-serif',
        'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
        'font.weight': 'extra bold',
        'axes.titleweight': 'extra bold',
        'axes.labelweight': 'extra bold',
        'legend.frameon': True,
        'legend.framealpha': 0.95,
        'legend.edgecolor': '#CCCCCC',
        'legend.fancybox': False,
        'legend.fontsize': 13,
        'lines.linewidth': 0.8,
        'lines.markersize': 1.2,
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'xtick.major.size': 3,
        'ytick.major.size': 3,
        'xtick.major.width': 0.4,
        'ytick.major.width': 0.4,
    },
)

# ── Color palettes — ordered dark → light (darker = bigger model)
# NVIDIA: warm greens / sage (richer, less blue-tinted)
# AMD:    reds (deep → medium → soft → light)
# All rendered with _ALPHA transparency for a pastel feel
PLATFORM_COLORS = {
    'nvidia': ['#2D8B57', '#4CAF6E', '#73C68A', '#9AD8A8', '#BFE8CA'],
    'amd': ['#C0392B', '#E05545', '#E8836F', '#F0A898', '#F5C4B8'],
    'unknown': ['#555555', '#808080', '#A0A0A0', '#C0C0C0', '#DDDDDD'],
}
_ALPHA = 0.75  # global transparency for softer pastel look

# ── Theme constants ──────────────────────────────────────────────────────────
_TITLE_COLOR = '#222222'
_ANNOTATION_COLOR = '#444444'
_PANEL_LABEL_COLOR = '#555555'

FRAMEWORK_DISPLAY = {
    'nemo': 'NeMo',
    'mega': 'Megatron',
    'prim': 'Primus',
    'deep': 'DeepSpeed',
    'fsdp': 'FSDP',
    'tran': 'Transformers',
}

MARKERS = ['o', 's', '^', 'D', 'v', 'p', 'h', '*']
LINESTYLES = ['-', '-', '-', '-']  # all solid — distinguish by color + marker


def load_benchmarks(results_dir: str) -> Dict[str, Dict]:
    """Load all benchmark JSON files from the results directory.
    
    Args:
        results_dir: Directory containing benchmark JSON files
    
    Returns:
        Dict mapping unique keys to benchmark data
    """
    results_path = Path(results_dir)
    benchmarks = {}
    
    for json_file in sorted(results_path.glob("*.json")):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            filename = json_file.stem
            parts = filename.split('_')
            
            # Parse filename: train_{platform}_{framework}_{model}[_{dataset}]
            platform = 'unknown'
            framework = 'unknown'
            model = 'unknown'
            dataset = None
            
            if len(parts) >= 4:
                platform_code = parts[1]
                platform = {'nvd': 'nvidia', 'amd': 'amd'}.get(platform_code, platform_code)
                framework = parts[2]
                model = parts[3]
                if len(parts) >= 5:
                    dataset = parts[4]
            
            # Fallback to JSON content
            if dataset is None:
                dataset = data.get('dataset')
            if platform == 'unknown':
                platform = data.get('platform', 'unknown')
                if platform == 'nvd':
                    platform = 'nvidia'
            
            # Build unique key
            key = f"{platform}-{framework}-{model}"
            if dataset:
                key = f"{key}-{dataset}"
            
            # Store parsed metadata
            data['_platform'] = platform
            data['_framework'] = framework
            data['_model'] = model
            data['_dataset'] = dataset
            data['_key'] = key
            
            benchmarks[key] = data
            
            ds_info = f" ({dataset})" if dataset else ""
            fw_display = FRAMEWORK_DISPLAY.get(framework, framework.upper())
            print(f"[+] Loaded: {platform.upper()} {fw_display} {model.upper()}{ds_info}")
            
        except Exception as e:
            print(f"[!] Error loading {json_file}: {e}")
    
    return benchmarks


def generate_styles(benchmarks: Dict[str, Dict]) -> Dict[str, Dict]:
    """Generate unique styles for each benchmark.
    
    Colors are assigned dark → light within each platform,
    sorted by model size (params) descending so darker = bigger model.
    """
    styles = {}
    
    # Group by platform to assign colors
    platform_counts = {}
    for key, data in benchmarks.items():
        platform = data.get('_platform', 'unknown')
        platform_counts.setdefault(platform, []).append(key)
    
    for platform, keys in platform_counts.items():
        colors = PLATFORM_COLORS.get(platform, PLATFORM_COLORS['unknown'])
        
        # Sort by model size (params) descending, then by name for stability
        def _sort_key(k):
            params = benchmarks[k].get('model_info', {}).get('total_params', 0)
            return (-params, k)
        
        sorted_keys = sorted(keys, key=_sort_key)
        
        for i, key in enumerate(sorted_keys):
            data = benchmarks[key]
            framework = data.get('_framework', 'unknown')
            model = data.get('_model', 'unknown')
            fw_display = FRAMEWORK_DISPLAY.get(framework, framework.upper())
            platform_display = platform.upper()
            
            label = f"{platform_display} {fw_display} {model.capitalize()}"
            
            styles[key] = {
                'color': colors[i % len(colors)],
                'marker': MARKERS[i % len(MARKERS)],
                'linestyle': LINESTYLES[i % len(LINESTYLES)],
                'label': label,
            }
    
    return styles


def _sparse_markevery(n_points: int, target: int = 20) -> int:
    """Return markevery interval so ~target markers appear on a line."""
    return max(1, n_points // target)


def _style_bar_ax(ax, title: str):
    """Apply bar-chart styling to an axes."""
    ax.set_title(title, fontsize=20, color=_TITLE_COLOR, pad=6)
    ax.grid(axis='y', linewidth=0.3)
    ax.grid(axis='x', visible=False)


def _style_line_ax(ax, title: str, xlabel: str = 'Step', ylabel: str = ''):
    """Apply line-chart styling to an axes."""
    ax.set_title(title, fontsize=20, color=_TITLE_COLOR, pad=6)
    ax.set_xlabel(xlabel, fontsize=17, color=_ANNOTATION_COLOR)
    ax.set_ylabel(ylabel, fontsize=17, color=_ANNOTATION_COLOR)
    ax.grid(linewidth=0.3)
    ax.legend(fontsize=12, loc='best', handlelength=2.2, borderpad=0.4,
              labelspacing=0.3, handletextpad=0.5)


def create_comparison_plot(
    benchmarks: Dict[str, Dict],
    output_file: str,
):
    """Create visual comparison plot for all benchmarks."""

    if not benchmarks:
        print("[!] No benchmark data to plot")
        return None

    styles = generate_styles(benchmarks)
    # NVIDIA first, then AMD, then others — within each platform sort alphabetically
    _platform_order = {'nvidia': 0, 'amd': 1}
    ordered_keys = sorted(benchmarks.keys(),
                          key=lambda k: (_platform_order.get(benchmarks[k]['_platform'], 9), k))

    # ── Determine title ──────────────────────────────────────────────────
    platforms = set(d.get('_platform', 'unknown') for d in benchmarks.values())
    if 'nvidia' in platforms and 'amd' in platforms:
        title = 'NVIDIA H100 vs AMD MI300X  —  Training Benchmark Comparison'
    elif 'nvidia' in platforms:
        title = 'NVIDIA H100  —  Training Benchmark Results'
    elif 'amd' in platforms:
        title = 'AMD MI300X  —  Training Benchmark Results'
    else:
        title = 'Training Benchmark Results'

    fig, axes = plt.subplots(2, 3, figsize=(24, 14))
    fig.suptitle('NVIDIA H100 vs AMD MI300X', fontsize=24, fontweight='black',
                 color=_TITLE_COLOR, y=0.98)
    axes = axes.flatten()
    panel_labels = []  # no panel labels

    # ── Panel (a): Throughput bar chart ──────────────────────────────────
    ax = axes[0]
    labels, values, colors = [], [], []
    for key in ordered_keys:
        tps = benchmarks[key].get('performance_metrics', {}).get('tokens_per_second_per_gpu')
        if tps:
            labels.append(styles[key]['label'])
            values.append(tps)
            colors.append(styles[key]['color'])
    if values:
        bars = ax.bar(range(len(values)), values, color=colors, width=0.7,
                      edgecolor='white', linewidth=0.4, alpha=_ALPHA)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=40, ha='right', fontsize=13)
        ax.set_ylabel('Tokens / s / GPU', fontsize=17, color=_ANNOTATION_COLOR)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 40,
                    f'{val:,.0f}', ha='center', va='bottom', fontsize=14,
                    fontweight='bold', color=_ANNOTATION_COLOR)
    _style_bar_ax(ax, 'Per-GPU Throughput')

    # ── Panel (b): Memory usage (single overlapping bar) ────────────────
    ax = axes[1]
    mem_labels, mem_alloc, mem_resv, mem_colors = [], [], [], []
    for key in ordered_keys:
        mem = benchmarks[key].get('memory_metrics', {})
        alloc = mem.get('avg_memory_allocated_gb')
        resv = mem.get('avg_memory_reserved_gb')
        if (alloc and alloc != 'N/A') or (resv and resv != 'N/A'):
            mem_labels.append(styles[key]['label'])
            mem_alloc.append(float(alloc) if alloc and alloc != 'N/A' else 0)
            mem_resv.append(float(resv) if resv and resv != 'N/A' else 0)
            mem_colors.append(styles[key]['color'])
    if mem_labels:
        n = len(mem_labels)
        bw = 0.7
        x = np.arange(n)
        has_a, has_r = any(v > 0 for v in mem_alloc), any(v > 0 for v in mem_resv)
        if has_a and has_r:
            # Draw reserved (full height) first as lighter background
            bars_r = ax.bar(x, mem_resv, bw, color=mem_colors, alpha=_ALPHA * 0.35,
                            edgecolor='#999999', linewidth=0.3, hatch='///', label='Reserved')
            # Draw allocated on top as solid foreground (same position, shorter)
            bars_a = ax.bar(x, mem_alloc, bw, color=mem_colors,
                            edgecolor='white', linewidth=0.4, alpha=_ALPHA, label='Allocated')
            # Annotate allocated value inside the bar
            for bar, val in zip(bars_a, mem_alloc):
                if val > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() - 1.5,
                            f'{val:.1f}', ha='center', va='top', fontsize=14,
                            fontweight='bold', color='white')
            # Annotate reserved value above the bar
            for bar, val in zip(bars_r, mem_resv):
                if val > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.3,
                            f'{val:.1f}', ha='center', va='bottom', fontsize=14,
                            fontweight='bold', color='#888888')
            ax.legend(fontsize=12, loc='upper right', framealpha=0.9)
        elif has_r:
            ax.bar(x, mem_resv, bw, color=mem_colors, alpha=_ALPHA * 0.5,
                   edgecolor='#999999', linewidth=0.3, hatch='///')
        else:
            ax.bar(x, mem_alloc, bw, color=mem_colors, alpha=_ALPHA,
                   edgecolor='white', linewidth=0.4)
        ax.set_xticks(x)
        ax.set_xticklabels(mem_labels, rotation=40, ha='right', fontsize=12)
        ax.set_ylabel('Memory (GB)', fontsize=17, color=_ANNOTATION_COLOR)
    _style_bar_ax(ax, 'GPU Memory Usage')

    # ── Panel (c): Training loss ─────────────────────────────────────────
    ax = axes[2]
    for key in ordered_keys:
        loss = benchmarks[key].get('loss_values', [])
        if loss:
            s = styles[key]
            ax.plot(range(len(loss)), loss, color=s['color'], marker='.',
                    markersize=1.5, alpha=_ALPHA, label=s['label'])
    ax.set_ylim(bottom=0)
    _style_line_ax(ax, 'Training Loss over Time', ylabel='Loss')

    # ── Panel (d): Learning rate ─────────────────────────────────────────
    ax = axes[3]
    for key in ordered_keys:
        lr = benchmarks[key].get('learning_rates', [])
        if lr:
            s = styles[key]
            ax.plot(range(len(lr)), lr, color=s['color'], marker='.',
                    markersize=1.5, alpha=_ALPHA, label=s['label'])
    ax.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
    ax.set_ylim(bottom=0)
    _style_line_ax(ax, 'Learning Rate over Time', ylabel='Learning Rate')

    # ── Panel (e): Step duration (zoomed, peaks annotated) ──────────────
    ax = axes[4]
    _e_series = []
    for key in ordered_keys:
        times = benchmarks[key].get('step_times', [])
        if times:
            s = styles[key]
            ax.plot(range(len(times)), times, color=s['color'], marker='.',
                    markersize=1.5, alpha=_ALPHA, label=s['label'])
            _e_series.append((times, s['color']))
    if _e_series:
        ax.set_ylim(bottom=0, top=5.1)
    else:
        ax.set_ylim(bottom=0)
    _style_line_ax(ax, 'Step Duration over Time', ylabel='Time (s)')

    # ── Panel (f): Gradient norm (zoomed, peaks annotated) ────────────
    ax = axes[5]
    _f_series = []
    for key in ordered_keys:
        gn = benchmarks[key].get('grad_norms', [])
        if gn:
            s = styles[key]
            ax.plot(range(len(gn)), gn, color=s['color'], marker='.',
                    markersize=1.5, alpha=_ALPHA, label=s['label'])
            _f_series.append((gn, s['color']))
    if _f_series:
        ax.set_ylim(bottom=0, top=20.1)
    else:
        ax.set_ylim(bottom=0)
    _style_line_ax(ax, 'Gradient Norm over Time', ylabel='Grad Norm')

    # ── Finishing touches ────────────────────────────────────────────────
    for i, ax in enumerate(axes):
        sns.despine(ax=ax)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_file, dpi=600, bbox_inches='tight', facecolor='white')
    print(f"  + Plot saved to: {output_file}")

    return fig


def print_summary(benchmarks: Dict[str, Dict]):
    """Print performance summary for all benchmarks."""
    
    print("\n" + "=" * 100)
    print("PERFORMANCE SUMMARY")
    print("=" * 100)
    
    print(f"\n{'Configuration':<40} {'Tokens/s/GPU':>14} {'Step Time':>12} {'Final Loss':>12}")
    print("-" * 100)
    
    for key in sorted(benchmarks.keys()):
        data = benchmarks[key]
        perf = data.get('performance_metrics', {})
        tps = perf.get('tokens_per_second_per_gpu', 0)
        step_time = perf.get('avg_step_time_seconds', 0)
        loss = data.get('loss_values', [])
        final_loss = loss[-1] if loss else 0
        
        platform = data.get('_platform', 'unknown').upper()
        framework = FRAMEWORK_DISPLAY.get(data.get('_framework', ''), data.get('_framework', 'unknown'))
        model = data.get('_model', 'unknown').capitalize()
        
        label = f"{platform} {framework} {model}"
        print(f"{label:<40} {tps:>14,.0f} {step_time:>12.3f}s {final_loss:>12.4f}")
    
    # Best performers
    print("\n" + "-" * 100)
    print("BEST PERFORMERS")
    print("-" * 100)
    
    # Best throughput
    best_tps = max(benchmarks.items(),
                   key=lambda x: x[1].get('performance_metrics', {}).get('tokens_per_second_per_gpu', 0))
    tps_val = best_tps[1].get('performance_metrics', {}).get('tokens_per_second_per_gpu', 0)
    print(f"\n  Highest Throughput: {best_tps[0]}")
    print(f"    {tps_val:,.0f} tokens/s/GPU")
    
    # Fastest step
    best_step = min(benchmarks.items(),
                    key=lambda x: x[1].get('performance_metrics', {}).get('avg_step_time_seconds', float('inf')))
    step_val = best_step[1].get('performance_metrics', {}).get('avg_step_time_seconds', 0)
    print(f"\n  Fastest Step Time: {best_step[0]}")
    print(f"    {step_val:.3f}s per step")
    
    # Cross-platform comparison if both exist
    platforms = set(d.get('_platform') for d in benchmarks.values())
    if 'nvidia' in platforms and 'amd' in platforms:
        print("\n" + "-" * 100)
        print("CROSS-PLATFORM COMPARISON")
        print("-" * 100)
        
        nvidia_best = max(
            (d for d in benchmarks.values() if d.get('_platform') == 'nvidia'),
            key=lambda x: x.get('performance_metrics', {}).get('tokens_per_second_per_gpu', 0),
            default=None
        )
        amd_best = max(
            (d for d in benchmarks.values() if d.get('_platform') == 'amd'),
            key=lambda x: x.get('performance_metrics', {}).get('tokens_per_second_per_gpu', 0),
            default=None
        )
        
        if nvidia_best and amd_best:
            nvidia_tps = nvidia_best.get('performance_metrics', {}).get('tokens_per_second_per_gpu', 0)
            amd_tps = amd_best.get('performance_metrics', {}).get('tokens_per_second_per_gpu', 0)
            
            if nvidia_tps > 0 and amd_tps > 0:
                ratio = amd_tps / nvidia_tps
                winner = "AMD" if ratio > 1 else "NVIDIA"
                ratio_display = ratio if ratio > 1 else 1 / ratio
                
                print(f"\n  NVIDIA best: {nvidia_tps:,.0f} tokens/s/GPU")
                print(f"  AMD best:    {amd_tps:,.0f} tokens/s/GPU")
                print(f"  -> {winner} is {ratio_display:.2f}x faster")
    
    print("\n" + "=" * 100 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Universal GPU benchmark comparison - auto-discovers all results'
    )
    default_dir = os.environ.get('OUTPUT_DIR', './output')
    parser.add_argument('--results-dir', default=default_dir,
                       help='Directory containing benchmark JSON files (default: OUTPUT_DIR or ./output)')
    parser.add_argument('--output', default='compare.png',
                       help='Output filename for the plot (default: compare.png)')
    
    args = parser.parse_args()
    
    print("=" * 100)
    print("UNIVERSAL BENCHMARK COMPARISON")
    print("=" * 100)
    print(f"\nScanning: {args.results_dir}")
    
    results_path = Path(args.results_dir)
    
    benchmarks = load_benchmarks(args.results_dir)
    
    if not benchmarks:
        print("\n  x No benchmarks found")
        print(f"\nExpected files in {args.results_dir}/:")
        print("  Format: train_{platform}_{framework}_{model}[_{dataset}].json")
        print("  Examples:")
        print("    train_nvd_nemo_llama_bc.json")
        print("    train_amd_prim_qwen_c4.json")
        return 1
    
    print(f"\n  Found {len(benchmarks)} benchmark(s)")
    
    output_file = str(results_path / os.path.basename(args.output))
    
    print(f"\nGenerating plot: {output_file}")
    try:
        fig = create_comparison_plot(benchmarks, output_file)
        if fig:
            plt.close(fig)
    except Exception as e:
        print(f"[!] Plot generation failed: {e}")
        import traceback
        traceback.print_exc()
    
    print_summary(benchmarks)
    
    print("\n" + "=" * 100)
    print("GENERATED PLOT")
    print("=" * 100)
    print(f"  + {output_file}")
    print()
    
    return 0


if __name__ == "__main__":
    exit(main())
