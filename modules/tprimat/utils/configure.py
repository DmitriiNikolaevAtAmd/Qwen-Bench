#!/usr/bin/env python3
"""
Apply env-based overrides to a TPrimat config YAML and write the result.
Static defaults live in the YAML; this module only sets values from environment.
Usage: python3 -m utils.configure <input_yaml> <output_yaml> [--profiling]
   or: python3 utils/configure.py <input_yaml> <output_yaml> [--profiling]
"""
import argparse
import os
import sys

import yaml


def purge_keys(d, keys):
    """Recursively remove keys from dict (and nested dicts/lists)."""
    if isinstance(d, dict):
        for k in list(keys):
            d.pop(k, None)
        for v in d.values():
            purge_keys(v, keys)
    elif isinstance(d, list):
        for item in d:
            purge_keys(item, keys)


def configure(config: dict, *, profiling: bool = False) -> None:
    """Apply env-based overrides to config in place."""
    tp = int(os.environ["TP"])
    pp = int(os.environ["PP"])
    gbs = int(os.environ["GBS"])
    mbs = int(os.environ["MBS"])
    seq_len = int(os.environ["SL"])
    grad_accum = int(os.environ["GA"])
    train_iters = int(os.environ["TRAIN_ITERS"])
    warmup_steps = int(os.environ["WARMUP_STEPS"])
    beta1 = float(os.environ.get("BETA1", "0.9"))
    beta2 = float(os.environ.get("BETA2", "0.95"))

    config["tensor_model_parallel_size"] = tp
    config["pipeline_model_parallel_size"] = pp
    config["sequence_parallel"] = tp > 1
    config["global_batch_size"] = gbs
    config["micro_batch_size"] = mbs
    config["seq_length"] = seq_len
    config["encoder_seq_length"] = seq_len
    config["gradient_accumulation_steps"] = grad_accum
    config["train_iters"] = train_iters
    config["lr_decay_iters"] = train_iters
    config["lr_warmup_iters"] = warmup_steps
    config["eval_interval"] = train_iters + 1
    config["seed"] = int(os.environ.get("SEED", "42"))
    config["adam_beta1"] = beta1
    config["adam_beta2"] = beta2
    config["data_path"] = os.environ["DATA_PREFIX"]
    config["tokenizer_model"] = os.environ["TOKENIZER_MODEL"]

    if "modules" in config and "pre_trainer" in config["modules"]:
        overrides = config["modules"]["pre_trainer"].get("overrides", {})
        overrides["init_method_std"] = 0.02
        overrides["global_batch_size"] = gbs
        overrides["micro_batch_size"] = mbs
        overrides["seq_length"] = seq_len
        overrides["lr_warmup_iters"] = warmup_steps
        overrides["train_iters"] = train_iters
        # MOCK_DATA env controls whether to use synthetic or real data.
        # Default is "true" (matches YAML) so existing mock-data runs keep working.
        # Set MOCK_DATA=false in config.env to use real --data_path + --split from CLI.
        mock_data = os.environ.get("MOCK_DATA", "true").lower() == "true"
        overrides["mock_data"] = mock_data
        config["modules"]["pre_trainer"]["overrides"] = overrides

    if profiling:
        profiling_enabled = os.environ.get("PROFILING", "false").lower() == "true"
        config["profile"] = profiling_enabled
        config["use_pytorch_profiler"] = profiling_enabled
        config["torch_profiler_with_stack"] = profiling_enabled
        config["torch_profiler_record_shapes"] = profiling_enabled
        if profiling_enabled:
            config["torch_profiler_wait"] = int(os.environ.get("PROFILE_WAIT", "5"))
            config["torch_profiler_warmup"] = int(os.environ.get("PROFILE_WARMUP", "1"))
            config["torch_profiler_active"] = int(os.environ.get("PROFILE_ACTIVE", "2"))
            config["torch_profiler_repeat"] = int(os.environ.get("PROFILE_REPEAT", "1"))
            # Ensure profile window is within train_iters to avoid hang after training (Primus default is 10–12)
            profile_step_start = int(os.environ.get("PROFILE_STEP_START", "1"))
            profile_step_end = int(os.environ.get("PROFILE_STEP_END", str(min(5, train_iters))))
            profile_step_end = min(profile_step_end, train_iters)
            profile_step_start = min(profile_step_start, profile_step_end)
            config["profile_step_start"] = profile_step_start
            config["profile_step_end"] = profile_step_end
            profile_dir = os.path.join(os.environ.get("OUTPUT_DIR", "./output"), "profiles")
            os.makedirs(profile_dir, exist_ok=True)
            config["torch_profiler_trace_dir"] = profile_dir
            if "modules" in config and "pre_trainer" in config["modules"]:
                overrides = config["modules"]["pre_trainer"].get("overrides", {})
                overrides["profile_step_start"] = profile_step_start
                overrides["profile_step_end"] = profile_step_end
                config["modules"]["pre_trainer"]["overrides"] = overrides

    purge_keys(config, ["use_fused_rmsnorm", "blend_per_split"])


def main():
    parser = argparse.ArgumentParser(description="Apply env overrides to TPrimat config YAML")
    parser.add_argument("input_yaml", help="Path to base config YAML")
    parser.add_argument("output_yaml", help="Path to write patched config")
    parser.add_argument("--profiling", action="store_true", help="Apply PROFILING/PROFILE_* env")
    args = parser.parse_args()

    with open(args.input_yaml) as f:
        config = yaml.safe_load(f)

    configure(config, profiling=args.profiling)

    with open(args.output_yaml, "w") as f:
        yaml.dump(config, f)

    print(f"[TPrimat] Patched config written to: {args.output_yaml}")


if __name__ == "__main__":
    main()
    sys.exit(0)
