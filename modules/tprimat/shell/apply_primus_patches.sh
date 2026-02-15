#!/bin/bash
# Apply Primus patches (use_fused_rmsnorm removal, pre_trainer extra_args, pyc purge).
# Expects PRIMUS_PATH to be set. Run from repo root or set TPRIMAT_PATH.
set -e
TPRIMAT_PATH="${TPRIMAT_PATH:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PRIMUS_PATH="${PRIMUS_PATH:?PRIMUS_PATH must be set}"

# 1. Remove --use_fused_rmsnorm from ALL Primus shell scripts
find "$PRIMUS_PATH" -name "*.sh" -exec sed -i '/use_fused_rmsnorm/d' {} + 2>/dev/null || true
echo "[TPrimat] Purged use_fused_rmsnorm from all Primus .sh scripts"

# 2. Remove use_fused_rmsnorm from ALL Primus YAML configs
python3 -c "
import pathlib, yaml
count = 0
for p in pathlib.Path('$PRIMUS_PATH').rglob('*.yaml'):
    try:
        txt = p.read_text()
        if 'use_fused_rmsnorm' not in txt:
            continue
        d = yaml.safe_load(txt)
        def purge(d):
            if isinstance(d, dict):
                d.pop('use_fused_rmsnorm', None)
                for v in d.values(): purge(v)
            elif isinstance(d, list):
                for v in d: purge(v)
        purge(d)
        yaml.dump(d, open(str(p), 'w'), default_flow_style=False)
        count += 1
    except Exception:
        pass
print(f'[TPrimat] Purged use_fused_rmsnorm from {count} YAML files')
" 2>&1

# 3. Patch pre_trainer.py to not crash on unknown extra_args
PRE_TRAINER="$PRIMUS_PATH/primus/modules/trainer/megatron/pre_trainer.py"
if [ -f "$PRE_TRAINER" ]; then
  python3 -c "
import pathlib
p = pathlib.Path('$PRE_TRAINER')
src = p.read_text()
if 'if extra_args:' in src and 'if False:' not in src:
    src = src.replace('if extra_args:', 'if False:  # patched: extra_args check disabled', 1)
    p.write_text(src)
    print('[TPrimat] Patched pre_trainer.py: disabled extra_args check')
elif 'if False:' in src:
    print('[TPrimat] pre_trainer.py already patched')
else:
    print('[TPrimat] pre_trainer.py: if extra_args: not found')
"
else
  echo "[TPrimat] pre_trainer.py not found at $PRE_TRAINER" >&2
fi

# 4. Patch trainer.py: when using real data (not mock), Primus builds blend_per_split from
#    data_path which conflicts with --split on the CLI. Fix: clear split when blend_per_split exists.
TRAINER="$PRIMUS_PATH/primus/modules/trainer/megatron/trainer.py"
if [ -f "$TRAINER" ]; then
  python3 << PYEOF
import pathlib
p = pathlib.Path("$TRAINER")
src = p.read_text()
marker = "# [TPrimat] patched: split/blend_per_split fix"
if marker in src:
    print("[TPrimat] trainer.py: split/blend_per_split patch already applied")
else:
    target = "return GPTDatasetConfig("
    if target in src:
        fix_lines = [
            "        " + marker,
            "        # When data_path is set (real data), Primus builds blend_per_split internally.",
            "        # The CLI also passes --split which conflicts. Clear split to avoid the assertion.",
            "        if getattr(args, 'blend_per_split', None) is not None:",
            "            args.split = None",
        ]
        fix = "\n".join(fix_lines) + "\n        "
        src = src.replace(target, fix + target, 1)
        p.write_text(src)
        print("[TPrimat] Patched trainer.py: clear split when blend_per_split is set")
    else:
        print("[TPrimat] trainer.py: could not find GPTDatasetConfig injection point")
PYEOF
else
  echo "[TPrimat] trainer.py not found at $TRAINER" >&2
fi

# 5. Make master_port configurable to avoid EADDRINUSE (port 1234 already in use)
find "$PRIMUS_PATH" -name "*.sh" -type f 2>/dev/null | while read -r f; do
  if grep -q "master_port.*1234\|1234.*master_port" "$f" 2>/dev/null; then
    sed -i 's/--master_port 1234/--master_port ${MASTER_PORT:-1234}/g' "$f"
    echo "[TPrimat] Patched master_port in $f"
  fi
done
echo "[TPrimat] Master port configurable via MASTER_PORT (default 1234)"

# 6. Purge .pyc caches so patched code is used
find "$PRIMUS_PATH" -name "*.pyc" -delete 2>/dev/null || true
find "$PRIMUS_PATH" -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
echo "[TPrimat] Purged all .pyc caches from Primus"
