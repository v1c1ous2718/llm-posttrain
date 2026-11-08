#!/usr/bin/env bash
# Pre-registered LLM experiment. Each step is a separate process so vLLM / training memory is fully released.
set -u
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=runs/pipeline.log
run() { echo "== $(date +%H:%M) $*" | tee -a $LOG; "$@" 2>&1 | grep -E "^\[|^sft|Traceback|Error" | tee -a $LOG; }

# 1. diagnostics on every SFT checkpoint (base already in runs/eval_base.json)
for s in 0025 0050 0100 0200 0438; do
  [ -f runs/eval_sft_$s.json ] || run uv run python -m pt.evaluate ckpt/sft_$s runs/eval_sft_$s.json
done

# 2. GRPO from four starting points, 100 steps each
STEPS=100
declare -A START=( [grpo_from_base]=Qwen/Qwen2.5-0.5B [grpo_from_sft0025]=ckpt/sft_0025
                   [grpo_from_sft0100]=ckpt/sft_0100 [grpo_from_sft0438]=ckpt/sft_0438 )
for name in grpo_from_base grpo_from_sft0025 grpo_from_sft0100 grpo_from_sft0438; do
  [ -d ckpt/$name ] || run uv run python -m pt.rl grpo ${START[$name]} $name $STEPS
done

# 3. PPO from the same start as grpo_from_sft0438
[ -d ckpt/ppo_from_sft0438 ] || run uv run python -m pt.rl ppo ckpt/sft_0438 ppo_from_sft0438 $STEPS

# 4. final evaluation of every RL policy
for name in grpo_from_base grpo_from_sft0025 grpo_from_sft0100 grpo_from_sft0438 ppo_from_sft0438; do
  [ -f runs/eval_$name.json ] || run uv run python -m pt.evaluate ckpt/$name runs/eval_$name.json
done
echo "== $(date +%H:%M) done" | tee -a $LOG
