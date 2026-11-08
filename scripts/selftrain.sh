#!/usr/bin/env bash
# Self-training (ReST-EM) rounds; each step runs in its own process so vLLM and training never share the GPU.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for r in 1 2 3; do
  [ -f runs/selftrain/round${r}_gen.json ]   || uv run python -m pt.selftrain gen "$r"
  [ -d ckpt/selftrain_r${r} ]                || uv run python -m pt.selftrain train "$r"
  [ -f runs/eval_selftrain_r${r}.json ]      || uv run python -m pt.evaluate ckpt/selftrain_r${r} runs/eval_selftrain_r${r}.json
done
