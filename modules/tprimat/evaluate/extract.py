#!/usr/bin/env python3
"""
Extract benchmark metrics from Primus training logs.

Usage:
    python3 extract.py \
        --log-file primus_training.log \
        --model-name llama \
        --output output/train_amd_prim_llama.json \
        --num-gpus 8 \
        --global-batch-size 128 \
        --sequence-length 2048
"""
import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import (
    round_floats,
    detect_gpu_info,
    extract_step_times_from_log,
    extract_param_count_from_log,
    extract_memory_from_log,
    extract_system_memory_from_log,
    parse_memory_log,
    get_parallelism_config,
    print_summary
)


def extract_metrics_from_log(log_file, num_gpus, global_batch_size, seq_length, micro_batch_size=1, tensor_parallel_size=1, pipeline_parallel_size=1, parallel_strategy=None, model_name=None):
    """Extract comprehensive metrics from Primus training log."""
    
    print(f"Analyzing log file: {log_file}")
    
    # Extract parameter count from log
    param_count = extract_param_count_from_log(log_file)
    if param_count:
        print(f"  + Model parameters: {param_count:,}")
    else:
        print(f"  ! Could not extract parameter count from log")
    
    # Extract step times, tokens per GPU, loss values, learning rates, and grad norms
    step_times, tokens_per_gpu_values, loss_values, learning_rates, grad_norms = extract_step_times_from_log(log_file)
    
    if not step_times:
        print("[!] No timing data found in log file")
        print("     Log file might use different format.")
        print("     Check the log manually and adjust regex patterns.")
        print("\n     Expected format: 'elapsed time per iteration (ms): X/Y'")
        return None
    
    print(f"  + Found {len(step_times)} step timing entries")
    if len(step_times) >= 2:
        jit_step = step_times[0]
        steady_steps = step_times[1:]
        steady_avg = sum(steady_steps) / len(steady_steps)
        print(f"    Iter 1 (warmup/JIT): {jit_step:.3f}s")
        print(f"    Iter 2+ (steady):    {steady_avg:.3f}s")
    
    if tokens_per_gpu_values:
        print(f"  + Found {len(tokens_per_gpu_values)} tokens/GPU entries (using Primus native metrics)")
    
    if loss_values:
        print(f"  + Found {len(loss_values)} loss values")
    
    if grad_norms:
        print(f"  + Found {len(grad_norms)} grad norm values")
    
    # Extract per-iteration memory from training log (PyTorch + System)
    memory_values = extract_memory_from_log(log_file)
    system_memory_values = extract_system_memory_from_log(log_file)
    
    if memory_values:
        print(f"  + Found {len(memory_values)} PyTorch memory entries (from training log)")
        if len(memory_values) >= 2:
            warmup_mem = memory_values[0]
            steady_mem = sum(memory_values[1:]) / len(memory_values[1:])
            print(f"    Iter 1 (warmup/JIT): {warmup_mem:.2f} GB")
            print(f"    Steady state (avg):  {steady_mem:.2f} GB")
    
    if system_memory_values:
        print(f"  + Found {len(system_memory_values)} system memory entries (from training log)")
        if len(system_memory_values) >= 2:
            warmup_sys = system_memory_values[0]
            steady_sys = sum(system_memory_values[1:]) / len(system_memory_values[1:])
            print(f"    Iter 1 (warmup/JIT): {warmup_sys:.2f} GB")
            print(f"    Steady state (avg):  {steady_sys:.2f} GB")
    
    # Auto-detect GPU info
    gpu_info = detect_gpu_info()
    
    # Detect platform from GPU info
    # Use software_stack as the primary indicator
    software_stack = gpu_info.get("software_stack", "rocm")
    
    # Also check device name for additional validation
    device_name_lower = gpu_info.get("device_name", "").lower()
    is_amd = "amd" in device_name_lower or "mi" in device_name_lower or software_stack == "rocm"
    is_nvidia = "nvidia" in device_name_lower or "h100" in device_name_lower or "a100" in device_name_lower or software_stack == "cuda"
    
    # Set platform based on software stack (more reliable than device name)
    if software_stack == "rocm":
        platform = "amd"
    elif software_stack == "cuda":
        platform = "nvd"
    elif is_amd:
        platform = "amd"
    elif is_nvidia:
        platform = "nvd"
    else:
        # Default to amd for Primus logs (Primus is AMD-focused)
        platform = "amd"
    
    # Calculate metrics (skip first step as warmup)
    step_times_no_warmup = step_times[1:] if len(step_times) > 1 else step_times
    avg_step_time = sum(step_times_no_warmup) / len(step_times_no_warmup)
    min_step_time = min(step_times_no_warmup)
    max_step_time = max(step_times_no_warmup)
    
    # Calculate token-based throughput
    # Always calculate from batch size and step time for consistency with NeMo/NVIDIA
    # Note: Primus native metrics report per-microbatch throughput, not per-training-step
    tokens_per_step = global_batch_size * seq_length
    tokens_per_second = tokens_per_step / avg_step_time
    tokens_per_second_per_gpu = tokens_per_second / num_gpus
    print(f"  * Calculated tokens/s/GPU: {tokens_per_second_per_gpu:.1f} (consistent with NeMo)")
    
    steps_per_second = 1.0 / avg_step_time
    
    if not parallel_strategy:
        parallel_strategy = os.environ.get('PARALLEL', None)
    
    parallelism_config = get_parallelism_config(
        parallel_strategy or "unknown",
        model_name or "llama",
        platform
    )
    
    # Calculate throughput per GPU core (for cross-platform comparison)
    throughput_per_gpu_core = 0
    if gpu_info.get("gpu_cores", 0) > 0:
        throughput_per_gpu_core = steps_per_second / gpu_info["gpu_cores"]
    
    # Calculate data_parallel_size and gradient_accumulation from batch sizes
    data_parallel_size = num_gpus // (tensor_parallel_size * pipeline_parallel_size)
    gradient_accumulation_steps = global_batch_size // (micro_batch_size * data_parallel_size)
    
    results = {
        "platform": platform,
        "gpu_info": gpu_info,
        "timestamp": datetime.now().isoformat(),
        "model_info": {
            "total_params": param_count,
            "trainable_params": param_count,  # Megatron logs don't distinguish; assume all trainable
        },
        "parallelism_config": {
            "strategy_name": parallel_strategy or parallelism_config.get("strategy", "unknown"),
            "tensor_model_parallel_size": tensor_parallel_size,
            "pipeline_model_parallel_size": pipeline_parallel_size,
            "data_parallel_size": data_parallel_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
        },
        "training_config": {
            "max_steps": len(step_times),
            "global_batch_size": global_batch_size,
            "micro_batch_size": micro_batch_size,
            "sequence_length": seq_length,
            "num_gpus": num_gpus,
            "parallel_strategy": parallel_strategy or "unknown",
        },
        "performance_metrics": {
            "total_steps": len(step_times),
            "total_time_seconds": sum(step_times),
            "avg_step_time_seconds": avg_step_time,
            "min_step_time_seconds": min_step_time,
            "max_step_time_seconds": max_step_time,
            "steps_per_second": steps_per_second,
            "tokens_per_second": tokens_per_second,
            "tokens_per_second_per_gpu": tokens_per_second_per_gpu,
            "throughput_per_gpu_core": throughput_per_gpu_core,
        },
        "step_times": step_times,
        "loss_values": loss_values if loss_values else [],
        "learning_rates": learning_rates if learning_rates else [],
        "grad_norms": grad_norms if grad_norms else [],
    }
    
    # Add memory metrics if available (per-step tracking from training log).
    # PyTorch: mem-alloc-GB / memory (GB) | allocated → torch.cuda.memory_allocated()
    if memory_values:
        steady_values = memory_values[1:] if len(memory_values) > 1 else memory_values
        results["memory_metrics"] = {
            "peak_memory_allocated_gb": max(memory_values),
            "avg_memory_allocated_gb": sum(steady_values) / len(steady_values),
            "min_memory_allocated_gb": min(memory_values),
            "warmup_memory_allocated_gb": memory_values[0],
            "steady_state_memory_allocated_gb": sum(steady_values) / len(steady_values),
            "measurement_method": "training_log_mem_alloc",
        }
        # Include per-step memory values for detailed analysis
        results["memory_values"] = memory_values
    
    # System: mem-reserved-GB / hip mem usage → total VRAM, sampled per iteration
    if system_memory_values:
        steady_sys = system_memory_values[1:] if len(system_memory_values) > 1 else system_memory_values
        results.setdefault("memory_metrics", {}).update({
            "peak_memory_reserved_gb": max(system_memory_values),
            "avg_memory_reserved_gb": sum(steady_sys) / len(steady_sys),
            "min_memory_reserved_gb": min(system_memory_values),
            "reserved_measurement_method": "training_log_per_iteration",
        })
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Extract benchmark metrics from Primus training logs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # With model name (auto-generates filename)
  python3 extract.py \\
      --log-file primus_training.log \\
      --model-name llama \\
      --num-gpus 8 \\
      --global-batch-size 128 \\
      --sequence-length 2048
  
  # With explicit output path
  python3 extract.py \\
      --log-file primus_training.log \\
      --output output/train_amd_prim_llama.json \\
      --num-gpus 8 \\
      --global-batch-size 128 \\
      --sequence-length 2048
        """
    )
    
    parser.add_argument('--log-file', required=True, 
                       help='Prim/Mega training log file')
    parser.add_argument('--output', 
                       help='Output JSON file path (if not specified, uses output/train_<framework>_<model>.json)')
    parser.add_argument('--model-name', 
                       help='Model name (e.g., llama, qwen) - used in auto-generated filename')
    parser.add_argument('--dataset',
                       help='Dataset name (e.g., bc, c4) - included in output filename and JSON')
    parser.add_argument('--num-gpus', type=int, required=True,
                       help='Number of GPUs used in training')
    parser.add_argument('--global-batch-size', type=int, required=True,
                       help='Global batch size (total across all GPUs)')
    parser.add_argument('--sequence-length', type=int, default=2048,
                       help='Sequence length (default: 2048)')
    parser.add_argument('--micro-batch-size', type=int, default=1,
                       help='Micro batch size per GPU (default: 1)')
    parser.add_argument('--tensor-parallel-size', type=int, default=1,
                       help='Tensor parallel size (default: 1)')
    parser.add_argument('--pipeline-parallel-size', type=int, default=1,
                       help='Pipeline parallel size (default: 1)')
    parser.add_argument('--parallel-strategy', 
                       help='Parallelism strategy name (e.g., balanced, minimal_communication)')
    parser.add_argument('--memory-log',
                       help='rocm-smi/nvidia-smi memory log file to parse')
    parser.add_argument('--peak-memory-gb', type=float,
                       help='Peak GPU memory usage in GB (deprecated, use --memory-log)')
    parser.add_argument('--memory-values-file',
                       help='JSON file with memory_values array (deprecated, use --memory-log)')
    parser.add_argument('--verbose', action='store_true',
                       help='Print verbose output')
    
    args = parser.parse_args()
    
    # Check if log file exists, fallback to backup logs if needed
    log_file = Path(args.log_file)
    if not log_file.exists():
        print(f"[!] Primary log file not found: {log_file}")
        
        # Try to find backup logs in the same directory
        log_dir = log_file.parent
        # Look for backup logs with new naming (primus_training_MODEL.log) and old naming (primus_training_MODEL_*.log)
        log_pattern = f"primus_training_{args.model_name}*.log" if args.model_name else "primus_training*.log"
        
        backup_logs = sorted(log_dir.glob(log_pattern), key=lambda p: p.stat().st_mtime, reverse=True)
        
        if backup_logs:
            log_file = backup_logs[0]
            print(f"[+] Found backup log: {log_file}")
            print(f"  Using most recent backup from: {datetime.fromtimestamp(log_file.stat().st_mtime)}")
            print()
        else:
            print(f"  x No backup logs found matching pattern: {log_pattern}")
            print(f"   Searched in: {log_dir}")
            return 1
    
    # Extract metrics
    results = extract_metrics_from_log(
        str(log_file),
        args.num_gpus,
        args.global_batch_size,
        args.sequence_length,
        args.micro_batch_size,
        args.tensor_parallel_size,
        args.pipeline_parallel_size,
        args.parallel_strategy,
        args.model_name
    )
    
    # Memory source priority:
    #   PyTorch (allocated): per-iteration from training log (mem-alloc-GB, etc.)
    #   System (reserved):   per-iteration from training log (mem-reserved-GB, hip mem usage)
    #                        → fallback to rocm-smi/nvidia-smi log (sampled every N seconds)
    #
    # Per-iteration sampling from the training log is preferred because it captures
    # memory at every iteration boundary, avoiding the sampling mismatch that occurs
    # when rocm-smi is polled at a different rate (every 2s) than iterations run.
    has_training_log_memory = results and results.get("memory_values")
    has_system_from_log = results and results.get("memory_metrics", {}).get("reserved_measurement_method") == "training_log_per_iteration"
    
    if results and args.memory_log:
        num_steps = len(results.get('step_times', []))
        mem_data = parse_memory_log(args.memory_log, num_steps=num_steps)
        if mem_data:
            if has_system_from_log:
                # Per-iteration system memory already extracted from training log — prefer it.
                # Keep smi data only as a diagnostic reference.
                raw_samples = mem_data.get('raw_samples', len(mem_data['memory_values']))
                print(f"  + System memory: using per-iteration training log (preferred)")
                print(f"    SMI fallback ({raw_samples} samples): peak={mem_data['peak_memory_gb']:.2f} GB, avg={mem_data['avg_memory_gb']:.2f} GB")
            else:
                # No per-iteration system memory from training log; use smi data.
                results.setdefault("memory_metrics", {}).update({
                    "peak_memory_reserved_gb": mem_data['peak_memory_gb'],
                    "avg_memory_reserved_gb": mem_data['avg_memory_gb'],
                    "min_memory_reserved_gb": mem_data['min_memory_gb'],
                    "reserved_measurement_method": "smi_periodic",
                })
                raw_samples = mem_data.get('raw_samples', len(mem_data['memory_values']))
                print(f"  + System memory (smi): {raw_samples} samples → {len(mem_data['memory_values'])} values")
                print(f"    Peak: {mem_data['peak_memory_gb']:.2f} GB, Avg: {mem_data['avg_memory_gb']:.2f} GB")
            if not has_training_log_memory:
                # No training log PyTorch memory available; fill allocated fields
                # from smi data so downstream code has something, but mark the source.
                results.setdefault("memory_metrics", {}).update({
                    "peak_memory_allocated_gb": mem_data['peak_memory_gb'],
                    "avg_memory_allocated_gb": mem_data['avg_memory_gb'],
                    "min_memory_allocated_gb": mem_data['min_memory_gb'],
                    "measurement_method": "smi_total_vram",
                })
                results["memory_values"] = mem_data['memory_values']
                print(f"    WARNING: no PyTorch memory from log; using smi for both metrics.")
        else:
            print(f"  ! Warning: Could not parse memory log: {args.memory_log}")
    elif results and (has_training_log_memory or has_system_from_log):
        parts = []
        if has_training_log_memory:
            parts.append("PyTorch")
        if has_system_from_log:
            parts.append("System")
        print(f"  + Using per-iteration training log memory ({' + '.join(parts)})")
    # Legacy: Load from intermediate JSON file
    elif results and args.memory_values_file:
        try:
            with open(args.memory_values_file, 'r') as f:
                mem_data = json.load(f)
            results["memory_metrics"] = {
                "peak_memory_allocated_gb": mem_data.get('peak_memory_gb', 0),
                "avg_memory_allocated_gb": mem_data.get('avg_memory_gb', 0),
                "min_memory_allocated_gb": mem_data.get('min_memory_gb', 0),
            }
            if 'memory_values' in mem_data:
                results["memory_values"] = mem_data['memory_values']
        except (json.JSONDecodeError, IOError) as e:
            print(f"  ! Warning: Could not load memory values file: {e}")
    # Legacy: Use peak-only value
    elif results and args.peak_memory_gb:
        results["memory_metrics"] = {
            "peak_memory_allocated_gb": args.peak_memory_gb,
            "avg_memory_allocated_gb": args.peak_memory_gb,
            "min_memory_allocated_gb": args.peak_memory_gb,
        }
    
    if results:
        # Add dataset to results when provided
        if args.dataset:
            results["dataset"] = args.dataset
        # Determine output path
        if args.output:
            output_path = Path(args.output).resolve()
        else:
            # Auto-generate filename with model name and optional dataset
            software_stack = results['gpu_info'].get('software_stack', 'unknown')
            model_suffix = f"_{args.model_name}" if args.model_name else ""
            dataset_suffix = f"_{args.dataset}" if args.dataset else ""
            output_path = Path(f"./output/train_{software_stack}{model_suffix}{dataset_suffix}.json").resolve()
        
        # Save results (round all floats to 5 decimal places)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Round all float values to 5 decimal places
        results_rounded = round_floats(results, precision=5)
        
        with open(output_path, 'w') as f:
            json.dump(results_rounded, f, indent=2)
        
        print(f"  + Metrics saved to: {output_path}")
        
        # Print summary
        print_summary(results)
        
        # Print next steps
        print("Next Steps:")
        print("  1. Run on NVIDIA GPU and collect results")
        print("  2. Compare with: python3 compare.py")
        print()
        
    else:
        print("  x Failed to extract metrics from log file")
        print("\nTroubleshooting:")
        print("  1. Check log file format")
        print("  2. Verify log contains timing information")
        print("  3. Try opening the log and looking for step times manually")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
