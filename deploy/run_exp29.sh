#!/usr/bin/env bash
# Experiment 29 on a GPU VM (as trainer). Survives an SSH disconnect:
#   nohup bash deploy/run_exp29.sh > exp29.log 2>&1 &          # quick test grid
#   FULL=1 nohup bash deploy/run_exp29.sh > exp29.log 2>&1 &   # full grid (~4x longer)
# Results: experiments/results/exp29_temperature_*.json (written after every variant).
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_HUB_OFFLINE=1  # .venv/bin/python, not `uv run`: it would re-sync torch from slow PyPI
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv
if [ "${FULL:-0}" = 1 ]; then
  GRID=()  # script defaults: N 8/16/32, T 0.7-3, 2 seeds, pool 5 and 10 per quote
else
  # Quick: beam baseline + 16 sampled paths at 5 temperatures, 2 seeds, pool 5 (as exp. 25-28).
  GRID=(--samples 16 --temperatures 0.7 1.0 1.5 2.0 3.0 --seeds 0 1 --per-quote 5)
fi
# Validation (ITMO) on both models; the beam baseline uses the IDF weight tuned for each (exp. 26/28).
.venv/bin/python experiments/run_temperature_eval.py --model Qwen/Qwen2.5-1.5B-Instruct --beta 0.3 "${GRID[@]}" "$@"
.venv/bin/python experiments/run_temperature_eval.py --model Qwen/Qwen2.5-3B-Instruct --beta 0 "${GRID[@]}" "$@"
echo done
