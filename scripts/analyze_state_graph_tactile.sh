#!/usr/bin/env bash
# Saved-data inference and graph analysis only. Never starts hardware services.
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
variant="${1:-mae}"
case "$variant" in mae|force) ;; *) echo 'Usage: bash scripts/analyze_state_graph_tactile.sh [mae|force]' >&2; exit 2;; esac
if (( $# > 1 )); then echo 'Expected one optional encoder variant' >&2; exit 2; fi
cd "$ROOT_DIR"
config="config.state_graph.symbolic_v3_tac_${variant}_20261112.json"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}" OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"
if [[ -e /.dockerenv || -e /run/.containerenv ]]; then
  "$ROOT_DIR/.venv_robot/bin/python" -m bottle_scan.state_graph.prepare_tactile --config "$config" --device "${TACTILE_ENCODER_DEVICE:-cuda}"
else
  docker exec -e OMP_NUM_THREADS -e OPENBLAS_NUM_THREADS -w "$ROOT_DIR" "${BOTTLE_ROS_CONTAINER:-daimon_ros1_das}" \
    "$ROOT_DIR/.venv_robot/bin/python" -m bottle_scan.state_graph.prepare_tactile --config "$config" --device "${TACTILE_ENCODER_DEVICE:-cuda}"
  # The existing Torch environment runs as container root. Return only this
  # task's generated files to the host user before the host-side analysis.
  docker exec "${BOTTLE_ROS_CONTAINER:-daimon_ros1_das}" chown -R "$(id -u):$(id -g)" \
    "$ROOT_DIR/outputs/state_graph/20261112_symbolic_v3_tac_${variant}" \
    "$ROOT_DIR/outputs/tactile_encoder_v3_20261112/cache"
fi
exec bash run_on_5090.sh state-graph analyze --config "$config"
