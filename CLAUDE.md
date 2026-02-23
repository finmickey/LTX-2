# LTX-2 RL Training

## Running Training

Always launch training in a **detached tmux session**:
```bash
mkdir -p outputs/<run_name>
tmux new-session -d -s <session_name> \
  "uv run --no-sync accelerate launch --num_processes 8 \
  packages/ltx-trainer/scripts/rl_train.py configs/<config_name>.yaml \
  2>&1 | tee outputs/<run_name>/run.log"
```

Always use `uv run --no-sync` (pypi.nvidia.com is flaky, skip sync).

## Config Files

- Location: `configs/` (e.g. `configs/rl_videoscore_multistep_run2.yaml`)
- Naming convention: `rl_<reward>_<variant>_run<N>.yaml`
- `output_dir` in the config controls where outputs go (e.g. `outputs/rl_videoscore_multistep_run2`)
- **IMPORTANT:** Always use a NEW unique run name (increment `run<N>`) for each launch, even when resuming from a checkpoint. This prevents overwriting validation videos, checkpoints, and logs from previous runs.

## Logs and Outputs

- Training log: `outputs/<run_name>/training.log`
- Saved config: `outputs/<run_name>/training_config.yaml`
- Validation videos: `outputs/<run_name>/samples/step_NNNNN_<nickname>.mp4`
- Checkpoints: `outputs/<run_name>/checkpoints/rl_lora_weights_step_NNNNN.safetensors`
- W&B: project `ltx2-rl-videoscore`

## Precomputed Embeddings (for large prompt sets)

For 100k+ prompts, precompute embeddings to avoid OOM at startup:
```bash
uv run --no-sync python packages/ltx-trainer/scripts/precompute_rl_embeddings.py \
    packages/ltx-trainer/data/rl_prompts_1000.txt \
    --output-dir embeddings/rl_1000 \
    --model-path models/ltx-2-19b-dev.safetensors \
    --text-encoder-path models/gemma-3-12b-it-qat-q4_0-unquantized
```

Then add to the training config YAML:
```yaml
rl:
  precomputed_embeddings_dir: embeddings/rl_1000
```

Omit `precomputed_embeddings_dir` to use the old behavior (encode all at startup).

## Stopping a Run

Kill the accelerate launcher process (parent of the 8 GPU workers):
```bash
ps aux | grep accelerate | grep -v grep
kill <accelerate_pid>
```
After killing, verify GPUs are free before restarting:
```bash
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
```
If stale processes hold GPU memory, `kill -9` them directly.

## Syntax Checking

Ruff is broken (pyproject.toml `target-version` issue). Use:
```bash
python3 -c "import ast; ast.parse(open('<file>').read()); print('OK')"
```
