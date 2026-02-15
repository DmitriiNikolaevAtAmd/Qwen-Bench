"""Training benchmark comparison dashboard.

Loads JSON benchmark files from the output directory, generates a 2x3
matplotlib figure (throughput, memory, loss, lr, step time, grad norm),
and prints a summary table.

Adapted from the tprimat project's evaluate/compare.py.
"""
import json
import logging
from pathlib import Path
from typing import Any, Optional

logging.getLogger("matplotlib").setLevel(logging.WARNING)

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


# -- Seaborn theme -- clean white journal style --------------------------------

sns.set_theme(
    style="whitegrid",
    font_scale=1.2,
    rc={
        "figure.facecolor": "white",
        "figure.dpi": 150,
        "axes.facecolor": "white",
        "axes.edgecolor": "#333333",
        "axes.linewidth": 0.5,
        "axes.titlepad": 8,
        "axes.labelpad": 6,
        "grid.color": "#E0E0E0",
        "grid.linewidth": 0.3,
        "text.color": "#222222",
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.weight": "extra bold",
        "axes.titleweight": "extra bold",
        "axes.labelweight": "extra bold",
        "legend.frameon": True,
        "legend.framealpha": 0.95,
        "legend.edgecolor": "#CCCCCC",
        "legend.fancybox": False,
        "legend.fontsize": 13,
        "lines.linewidth": 0.8,
        "lines.markersize": 1.2,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.major.width": 0.4,
        "ytick.major.width": 0.4,
    },
)

PLATFORM_COLORS = {
    "cuda": ["#2D8B57", "#4CAF6E", "#73C68A", "#9AD8A8", "#BFE8CA"],
    "rocm": ["#C0392B", "#E05545", "#E8836F", "#F0A898", "#F5C4B8"],
    "unknown": ["#555555", "#808080", "#A0A0A0", "#C0C0C0", "#DDDDDD"],
}
_ALPHA = 0.75

_TITLE_COLOR = "#222222"
_ANNOTATION_COLOR = "#444444"

FRAMEWORK_DISPLAY = {
    "megatron": "Megatron",
    "nemo": "NeMo",
    "prim": "Primus",
    "deep": "DeepSpeed",
    "fsdp": "FSDP",
    "tran": "Transformers",
}

MARKERS = ["o", "s", "^", "D", "v", "p", "h", "*"]


# -- Load benchmarks ----------------------------------------------------------


def _load_config(json_path: Path) -> dict:
    """Try to load config.yaml from the same directory as a JSON file."""
    try:
        import yaml
    except ImportError:
        return {}
    cfg_path = json_path.parent / "config.yaml"
    if not cfg_path.exists():
        return {}
    try:
        with open(cfg_path) as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def _compute_stats(data: dict, cfg: dict) -> None:
    """Compute performance_metrics and memory_metrics from time-series arrays."""
    step_times = data.get("step_times", [])
    if len(step_times) > 1:
        steady = step_times[1:]  # skip warmup
    else:
        steady = step_times

    avg_step = sum(steady) / len(steady) if steady else 0

    # Throughput: need global_batch_size, seq_length, num_gpus from config
    training = cfg.get("training", {})
    gbs = training.get("global_batch_size") or data.get("training_config", {}).get("global_batch_size")
    seq = training.get("seq_length") or data.get("training_config", {}).get("sequence_length")
    ngpu = training.get("parallel", {}).get("data") or data.get("training_config", {}).get("num_gpus") or 1

    tps = None
    tps_gpu = None
    if gbs and seq and avg_step > 0:
        tps = gbs * seq / avg_step
        tps_gpu = tps / ngpu

    data["performance_metrics"] = {
        "total_steps": len(step_times),
        "total_time_seconds": round(sum(step_times), 2),
        "avg_step_time_seconds": round(avg_step, 5),
        "tokens_per_second": round(tps, 2) if tps else None,
        "tokens_per_second_per_gpu": round(tps_gpu, 2) if tps_gpu else None,
    }

    # Memory metrics from arrays
    mem_alloc = data.get("memory_allocated", [])
    mem_resv = data.get("memory_reserved", [])
    if mem_alloc or mem_resv:
        mm: dict[str, Any] = {}
        if mem_alloc:
            mm["peak_memory_allocated_gb"] = max(mem_alloc)
            mm["avg_memory_allocated_gb"] = round(sum(mem_alloc) / len(mem_alloc), 2)
        if mem_resv:
            mm["peak_memory_reserved_gb"] = max(mem_resv)
            mm["avg_memory_reserved_gb"] = round(sum(mem_resv) / len(mem_resv), 2)
        data["memory_metrics"] = mm


# Reverse abbreviation maps for display
_FRAMEWORK_FULL = {
    "mega": "megatron",
    "nemo": "nemo",
    "prim": "primus",
    "deep": "deepspeed",
    "tran": "transformers",
    "fsdp": "fsdp",
    "megatron": "megatron",
}


def load_benchmarks(results_dir: str) -> dict[str, dict]:
    """Discover and load all ``train_*.json`` benchmark files.

    Filename convention: train_{platform}_{framework}_{model}_{dataset}.json
    Metadata is parsed from the filename; stats are computed from arrays.
    """
    results_path = Path(results_dir)
    benchmarks: dict[str, dict] = {}

    for json_file in sorted(results_path.glob("**/train_*.json")):
        try:
            with open(json_file, "r") as f:
                data = json.load(f)

            parts = json_file.stem.split("_")

            # train_{platform}_{framework}_{model}_{dataset}
            if len(parts) >= 5:
                platform_raw = parts[1]
                framework = parts[2]
                model = parts[3]
                dataset = "_".join(parts[4:])
            elif len(parts) >= 4:
                platform_raw = parts[1]
                framework = parts[2]
                model = parts[3]
                dataset = None
            elif len(parts) >= 3:
                platform_raw = data.get("platform", parts[1])
                framework = parts[1]
                model = parts[2]
                dataset = None
            else:
                continue

            # Normalise platform
            platform = {"nvd": "cuda", "nvidia": "cuda", "amd": "rocm"}.get(platform_raw, platform_raw)
            # Resolve framework abbreviation
            framework_full = _FRAMEWORK_FULL.get(framework, framework)

            data["_platform"] = platform
            data["_framework"] = framework_full
            data["_model"] = model
            data["_dataset"] = dataset or data.get("dataset")

            key = f"{platform}-{framework_full}-{model}"
            if data["_dataset"]:
                key = f"{key}-{data['_dataset']}"
            data["_key"] = key

            # Compute stats from arrays + config
            if "performance_metrics" not in data:
                cfg = _load_config(json_file)
                _compute_stats(data, cfg)

            benchmarks[key] = data
        except Exception:
            pass

    return benchmarks


# -- Style generation ----------------------------------------------------------


def _generate_styles(benchmarks: dict[str, dict]) -> dict[str, dict]:
    """Assign color/marker/label to each benchmark entry."""
    styles: dict[str, dict] = {}
    platform_groups: dict[str, list[str]] = {}

    for key, data in benchmarks.items():
        plat = data.get("_platform", "unknown")
        platform_groups.setdefault(plat, []).append(key)

    for plat, keys in platform_groups.items():
        colors = PLATFORM_COLORS.get(plat, PLATFORM_COLORS["unknown"])

        def _sort(k: str) -> tuple:
            p = benchmarks[k].get("model_info", {}).get("total_params", 0) or 0
            return (-p, k)

        sorted_keys = sorted(keys, key=_sort)
        for i, key in enumerate(sorted_keys):
            d = benchmarks[key]
            fw = d.get("_framework", "unknown")
            mdl = d.get("_model", "unknown")
            fw_disp = FRAMEWORK_DISPLAY.get(fw, fw.upper())
            plat_disp = {"cuda": "NVIDIA", "rocm": "AMD"}.get(plat, plat.upper())
            label = f"{plat_disp} {fw_disp} {mdl.capitalize()}"
            styles[key] = {
                "color": colors[i % len(colors)],
                "marker": MARKERS[i % len(MARKERS)],
                "label": label,
            }

    return styles


# -- Plotting helpers ----------------------------------------------------------


def _style_bar(ax: plt.Axes, title: str) -> None:
    ax.set_title(title, fontsize=20, color=_TITLE_COLOR, pad=6)
    ax.grid(axis="y", linewidth=0.3)
    ax.grid(axis="x", visible=False)


def _style_line(ax: plt.Axes, title: str, xlabel: str = "Step", ylabel: str = "") -> None:
    ax.set_title(title, fontsize=20, color=_TITLE_COLOR, pad=6)
    ax.set_xlabel(xlabel, fontsize=17, color=_ANNOTATION_COLOR)
    ax.set_ylabel(ylabel, fontsize=17, color=_ANNOTATION_COLOR)
    ax.grid(linewidth=0.3)
    ax.legend(
        fontsize=12, loc="best", handlelength=2.2,
        borderpad=0.4, labelspacing=0.3, handletextpad=0.5,
    )


# -- Main plot -----------------------------------------------------------------


def create_comparison_plot(
    benchmarks: dict[str, dict],
    output_file: str,
) -> Optional[plt.Figure]:
    """Create the 2x3 benchmark dashboard and save to *output_file*."""
    if not benchmarks:
        return None

    styles = _generate_styles(benchmarks)
    _order = {"cuda": 0, "rocm": 1}
    ordered = sorted(
        benchmarks.keys(),
        key=lambda k: (_order.get(benchmarks[k]["_platform"], 9), k),
    )

    fig, axes = plt.subplots(2, 3, figsize=(24, 14))

    # Build a title from discovered platforms
    platforms = {d.get("_platform", "unknown") for d in benchmarks.values()}
    if "cuda" in platforms and "rocm" in platforms:
        title = "NVIDIA vs AMD Benchmark Results"
    elif "cuda" in platforms:
        title = "NVIDIA Benchmark Results"
    elif "rocm" in platforms:
        title = "AMD Benchmark Results"
    else:
        title = "Benchmark Results"

    fig.suptitle(title, fontsize=24, fontweight="black", color=_TITLE_COLOR, y=0.98)
    axes = axes.flatten()

    # -- (a) Per-GPU throughput bar chart --------------------------------------
    ax = axes[0]
    labels, values, colors = [], [], []
    for key in ordered:
        tps = benchmarks[key].get("performance_metrics", {}).get("tokens_per_second_per_gpu")
        if tps:
            labels.append(styles[key]["label"])
            values.append(tps)
            colors.append(styles[key]["color"])
    if values:
        bars = ax.bar(
            range(len(values)), values, color=colors,
            width=0.7, edgecolor="white", linewidth=0.4, alpha=_ALPHA,
        )
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=13)
        ax.set_ylabel("Tokens/s/GPU", fontsize=17, color=_ANNOTATION_COLOR)
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 40,
                f"{val:,.0f}", ha="center", va="bottom", fontsize=14,
                fontweight="bold", color=_ANNOTATION_COLOR,
            )
    _style_bar(ax, "Average Per-GPU Throughput")

    # -- (b) Memory bar chart --------------------------------------------------
    ax = axes[1]
    mem_labels, mem_alloc, mem_resv, mem_colors = [], [], [], []
    for key in ordered:
        mem = benchmarks[key].get("memory_metrics", {})
        alloc = mem.get("avg_memory_allocated_gb")
        resv = mem.get("avg_memory_reserved_gb")
        if alloc or resv:
            mem_labels.append(styles[key]["label"])
            mem_alloc.append(float(alloc) if alloc else 0)
            mem_resv.append(float(resv) if resv else 0)
            mem_colors.append(styles[key]["color"])
    if mem_labels:
        x = np.arange(len(mem_labels))
        bw = 0.7
        has_a = any(v > 0 for v in mem_alloc)
        has_r = any(v > 0 for v in mem_resv)
        if has_a and has_r:
            bars_r = ax.bar(
                x, mem_resv, bw, color=mem_colors,
                alpha=_ALPHA * 0.35, edgecolor="#999999",
                linewidth=0.3, hatch="///", label="System",
            )
            bars_a = ax.bar(
                x, mem_alloc, bw, color=mem_colors,
                edgecolor="white", linewidth=0.4, alpha=_ALPHA, label="PyTorch",
            )
            for bar, val in zip(bars_r, mem_resv):
                if val > 0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 0.5,
                        f"{val:.1f}", ha="center", va="bottom", fontsize=14,
                        fontweight="bold", color=_ANNOTATION_COLOR,
                    )
            for bar, val in zip(bars_a, mem_alloc):
                if val > 0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.0, bar.get_height() - 1.5,
                        f"{val:.1f}", ha="center", va="top", fontsize=14,
                        fontweight="bold", color="white",
                    )
            ax.legend(fontsize=12, loc="upper right", framealpha=0.9)
        elif has_a:
            ax.bar(x, mem_alloc, bw, color=mem_colors, alpha=_ALPHA, edgecolor="white", linewidth=0.4)
        elif has_r:
            ax.bar(x, mem_resv, bw, color=mem_colors, alpha=_ALPHA * 0.5, edgecolor="#999999", hatch="///")
        ax.set_xticks(x)
        ax.set_xticklabels(mem_labels, rotation=40, ha="right", fontsize=12)
        ax.set_ylabel("Memory (GB)", fontsize=17, color=_ANNOTATION_COLOR)
    _style_bar(ax, "GPU Memory Usage")

    # -- (c) Training loss -----------------------------------------------------
    ax = axes[2]
    for key in ordered:
        loss = benchmarks[key].get("loss_values", [])
        if loss:
            s = styles[key]
            ax.plot(range(len(loss)), loss, color=s["color"], marker=".", markersize=1.5, alpha=_ALPHA, label=s["label"])
    ax.set_ylim(bottom=0)
    _style_line(ax, "Training Loss over Time", ylabel="Loss")

    # -- (d) Learning rate -----------------------------------------------------
    ax = axes[3]
    for key in ordered:
        lr = benchmarks[key].get("learning_rates", [])
        if lr:
            s = styles[key]
            ax.plot(range(len(lr)), lr, color=s["color"], marker=".", markersize=1.5, alpha=_ALPHA, label=s["label"])
    ax.ticklabel_format(axis="y", style="scientific", scilimits=(0, 0))
    ax.set_ylim(bottom=0)
    _style_line(ax, "Learning Rate over Time", ylabel="Learning Rate")

    # -- (e) Step duration -----------------------------------------------------
    ax = axes[4]
    for key in ordered:
        times = benchmarks[key].get("step_times", [])
        if times:
            s = styles[key]
            ax.plot(range(len(times)), times, color=s["color"], marker=".", markersize=1.5, alpha=_ALPHA, label=s["label"])
    ax.set_ylim(bottom=0)
    _style_line(ax, "Step Duration over Time", ylabel="Time (secs)")

    # -- (f) Gradient norm -----------------------------------------------------
    ax = axes[5]
    for key in ordered:
        gn = benchmarks[key].get("grad_norms", [])
        if gn:
            s = styles[key]
            ax.plot(range(len(gn)), gn, color=s["color"], marker=".", markersize=1.5, alpha=_ALPHA, label=s["label"])
    ax.set_ylim(bottom=0)
    _style_line(ax, "Gradient Norm over Time", ylabel="Grad Norm")

    # -- Finish ----------------------------------------------------------------
    for ax in axes.flatten():
        sns.despine(ax=ax)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_file, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    return fig


# -- Summary -------------------------------------------------------------------


def print_summary(benchmarks: dict[str, dict]) -> str:
    """Build a plain-text performance summary. Returns the summary string."""
    lines: list[str] = []
    header = f"{'Configuration':<40} {'Tokens/s/GPU':>14} {'Step Time':>12} {'Final Loss':>12}"
    sep = "-" * 80

    lines.append(sep)
    lines.append(header)
    lines.append(sep)

    for key in sorted(benchmarks.keys()):
        data = benchmarks[key]
        perf = data.get("performance_metrics", {})
        tps = perf.get("tokens_per_second_per_gpu", 0) or 0
        step_time = perf.get("avg_step_time_seconds", 0) or 0
        loss = data.get("loss_values", [])
        final_loss = loss[-1] if loss else 0

        plat_raw = data.get("_platform", "unknown")
        plat_disp = {"cuda": "NVIDIA", "rocm": "AMD"}.get(plat_raw, plat_raw.upper())
        fw = FRAMEWORK_DISPLAY.get(data.get("_framework", ""), data.get("_framework", "unknown"))
        model = data.get("_model", "unknown").capitalize()

        label = f"{plat_disp} {fw} {model}"
        lines.append(f"{label:<40} {tps:>14,.0f} {step_time:>12.3f}s {final_loss:>12.4f}")

    lines.append(sep)
    return "\n".join(lines)
