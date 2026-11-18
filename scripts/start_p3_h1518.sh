#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd "$(dirname "$0")/.." && pwd)"
cd "$repo_dir"
exec bash scripts/start_p3_forcepolicy_loop.sh \
  --graph-version 5 \
  --source /home/dm/DM_das/bottle_grasp_h1214/outputs/state_graph/P3_manual_lift_H1214 \
  --out "$repo_dir/outputs/state_graph/P3_manual_lift_H1518" \
  --overrides configs/p3_v5_manual_lift_h1518.json \
  --director-mode fixed_priority --extra-trials 100 "$@"
