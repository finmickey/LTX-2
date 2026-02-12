#!/usr/bin/env bash
# Hyperparameter sweep: lr x kl_beta grid
# Waits for current optimized_run1 to finish, then runs 8 experiments sequentially.
# Skip: lr=5e-5 kl_beta=0.5 (already running as optimized_run1)

set -euo pipefail

LOGFILE="/home/user/LTX-2/outputs/sweep_runner.log"
BASE_DIR="/home/user/LTX-2"

log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') | $*" | tee -a "$LOGFILE"
}

cd "$BASE_DIR"

# --- Phase 1: Wait for optimized_run1 to finish ---
log "Sweep runner started. Waiting for optimized_run1 to finish..."

while true; do
    if pgrep -f "rl_optimized_run1.yaml" > /dev/null 2>&1; then
        sleep 30
    else
        log "optimized_run1 process no longer running."
        break
    fi
done

# Give GPUs a moment to release memory
sleep 10

# Verify GPUs are free
log "GPU memory status:"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tee -a "$LOGFILE"

# --- Phase 2: Generate configs and run experiments ---
declare -a LRS=("1e-4" "5e-5" "1e-5")
declare -a KL_BETAS=("0.1" "0.5" "1.0")

# Python-friendly lr values for YAML
declare -A LR_YAML=(
    ["1e-4"]="1.0e-04"
    ["5e-5"]="5.0e-05"
    ["1e-5"]="1.0e-05"
)

EXPERIMENT_COUNT=0
TOTAL_EXPERIMENTS=8

for lr in "${LRS[@]}"; do
    for kl in "${KL_BETAS[@]}"; do
        # Skip the already-running combination
        if [[ "$lr" == "5e-5" && "$kl" == "0.5" ]]; then
            log "Skipping lr=${lr} kl_beta=${kl} (already ran as optimized_run1)"
            continue
        fi

        EXPERIMENT_COUNT=$((EXPERIMENT_COUNT + 1))
        RUN_NAME="rl_sweep_lr${lr}_kl${kl}"
        CONFIG_FILE="configs/${RUN_NAME}.yaml"
        OUTPUT_DIR="outputs/${RUN_NAME}"

        log "=== Experiment ${EXPERIMENT_COUNT}/${TOTAL_EXPERIMENTS}: ${RUN_NAME} (lr=${lr}, kl_beta=${kl}) ==="

        # Generate config from base
        python3 -c "
import yaml, copy

with open('configs/rl_optimized_run1.yaml') as f:
    cfg = yaml.safe_load(f)

cfg['optimization']['learning_rate'] = float('${LR_YAML[$lr]}')
cfg['optimization']['steps'] = 3000
cfg['rl']['kl_beta'] = float('${kl}')
cfg['output_dir'] = '${OUTPUT_DIR}'
cfg['wandb']['tags'] = ['ltx2', 'rl', 'nft', 'videoscore', 'sweep', 'lr${lr}', 'kl${kl}']

with open('${CONFIG_FILE}', 'w') as f:
    yaml.dump(cfg, f, default_flow_style=False, sort_keys=True)

print('Config written: ${CONFIG_FILE}')
"
        log "Config created: ${CONFIG_FILE}"

        # Run the experiment
        log "Starting training for ${RUN_NAME}..."
        uv run --no-sync accelerate launch --num_processes 8 \
            packages/ltx-trainer/scripts/rl_train.py "${CONFIG_FILE}" \
            2>&1 | tee -a "${LOGFILE}" || {
            log "ERROR: ${RUN_NAME} failed with exit code $?"
            # Wait a bit and check GPUs before continuing
            sleep 10
            nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tee -a "$LOGFILE"
        }

        log "Finished ${RUN_NAME}."

        # Brief pause between experiments to let GPU memory settle
        sleep 10

        # Check GPU memory
        log "GPU memory after ${RUN_NAME}:"
        nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tee -a "$LOGFILE"
    done
done

log "=== All ${TOTAL_EXPERIMENTS} sweep experiments completed! ==="
