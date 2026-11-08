#!/usr/bin/env bash
# Follow-up added after the pre-registered runs: lr 1e-6 barely moved the policy in 100 steps (KL ~0.001),
# so rerun both algorithms from the same start with lr 5e-6. Then run the diffusion experiment.
set -u
cd "$(dirname "$0")/.."
until grep -q "== .* done" runs/pipeline.log; do sleep 30; done
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=runs/pipeline.log
run() { echo "== $(date +%H:%M) $*" | tee -a $LOG; "$@" 2>&1 | grep --line-buffered -E "^\[|Traceback|Error" | tee -a $LOG; }
[ -d ckpt/grpo_from_sft0438_lr5e-6 ] || run uv run python -m pt.rl grpo ckpt/sft_0438 grpo_from_sft0438_lr5e-6 100 5e-6
[ -d ckpt/ppo_from_sft0438_lr5e-6 ]  || run uv run python -m pt.rl ppo  ckpt/sft_0438 ppo_from_sft0438_lr5e-6 100 5e-6
for n in grpo_from_sft0438_lr5e-6 ppo_from_sft0438_lr5e-6; do
  [ -f runs/eval_$n.json ] || run uv run python -m pt.evaluate ckpt/$n runs/eval_$n.json
done
echo "== $(date +%H:%M) followup done" | tee -a $LOG
cd ~/diffusion-custom && echo "== $(date +%H:%M) diffusion start" | tee -a ~/llm-posttrain/$LOG
uv run python scripts/run_all.py 2>&1 | grep --line-buffered -vE "it/s\]|Warning|warn" | tee -a runs_diffusion.log | tee -a ~/llm-posttrain/$LOG
echo "== $(date +%H:%M) diffusion done" | tee -a ~/llm-posttrain/$LOG
