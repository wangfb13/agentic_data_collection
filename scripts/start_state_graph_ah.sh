#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export WHITE_PLACEMENT_X5=0318_0347
exec bash run_on_5090.sh python -m skills.contact_position_a.campaign_ah "$@"
