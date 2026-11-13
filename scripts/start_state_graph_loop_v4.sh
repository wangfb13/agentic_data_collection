#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
OUT="${STATE_GRAPH_CAMPAIGN_OUT:-outputs/state_graph/campaign_white_a_v4}"
if [[ ! -f "$OUT/state.json" ]]; then
  bash run_on_5090.sh state-graph-loop init --config config.state_graph.campaign_v4.json --out "$OUT"
fi
exec bash run_on_5090.sh state-graph-loop run --out "$OUT" "$@"
