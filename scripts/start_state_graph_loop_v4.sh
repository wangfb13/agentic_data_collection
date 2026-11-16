#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source_campaign="${STATE_GRAPH_SEED_CAMPAIGN:-outputs/state_graph/campaign_white_a_v4}"
mode=
args=()
while (( $# )); do
  case "$1" in
    --ready)
      mode="${2:?--ready requires fixed5 or forcepolicy}"
      shift 2 ;;
    *) args+=("$1"); shift ;;
  esac
done
case "$mode" in
  fixed5|forcepolicy)
    suffix="$mode"
    if [[ "$mode" == fixed5 ]]; then suffix=fixed5_device_range; fi
    OUT="${STATE_GRAPH_CAMPAIGN_OUT:-outputs/state_graph/campaign_white_a_v4_${suffix}}"
    python=".venv/bin/python"
    if [[ -f /.dockerenv || -f /run/.containerenv ]]; then python=".venv_robot/bin/python"; fi
    "$python" -m skills.contact_position_a.ready_campaign --source "$source_campaign" --mode "$mode" --out "$OUT"
    ;;
  "")
    OUT="${STATE_GRAPH_CAMPAIGN_OUT:-outputs/state_graph/campaign_white_a_v4}"
    if [[ ! -f "$OUT/state.json" ]]; then
      bash run_on_5090.sh state-graph-loop init --config config.state_graph.campaign_v4.json --out "$OUT"
    fi ;;
  *) echo "Unknown Ready mode '$mode'. Choose fixed5 or forcepolicy." >&2; exit 2 ;;
esac
exec bash run_on_5090.sh state-graph-loop run --out "$OUT" "${args[@]}"
