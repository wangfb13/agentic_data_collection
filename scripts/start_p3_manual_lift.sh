#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
exec bash scripts/start_p3_forcepolicy_loop.sh --graph-version 5 --director-mode llm --allow-agent \
  --overrides configs/p3_v5_manual_lift.json --out outputs/state_graph/campaign_white_rh_P3_manual_lift_v5 "$@"
