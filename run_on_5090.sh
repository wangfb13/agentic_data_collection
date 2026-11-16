#!/usr/bin/env bash
# Run on the 5090 host or directly inside its existing ROS container. No server start,
# controller reset, homing, model inference, or actuator command is implicit.
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONTAINER="${BOTTLE_ROS_CONTAINER:-daimon_ros1_das}"
COMMAND="${1:-doctor}"
case "$COMMAND" in
  state-graph) SCRIPT_ARGS=(state_graph.py) ;;
  fixed5-once) SCRIPT_ARGS=(-m skills.contact_position_a.fixed5_once) ;;
  state-graph-loop) SCRIPT_ARGS=(state_graph_campaign.py) ;;
  a-axis) SCRIPT_ARGS=(a_axis_task.py) ;;
  agentic) SCRIPT_ARGS=(agentic.py) ;;
  rsi) SCRIPT_ARGS=(rsi.py) ;;
  skill-loop) SCRIPT_ARGS=(skill_loop.py) ;;
  primary-skill-export) SCRIPT_ARGS=(primary_skill_export.py) ;;
  skill-label-review) SCRIPT_ARGS=(skill_label_review.py) ;;
  skill-offline) SCRIPT_ARGS=(skill_offline.py) ;;
  skill-collect) SCRIPT_ARGS=(skill_collect.py) ;;
  skill-extension) SCRIPT_ARGS=(skill_extension.py) ;;
  vision-stage1) SCRIPT_ARGS=(vision_stage1.py) ;;
  skill-feedback) SCRIPT_ARGS=(skill_feedback.py) ;;
  e1e2) SCRIPT_ARGS=(e1e2.py) ;;
  ambiguity) SCRIPT_ARGS=(ambiguity.py) ;;
  analyze_ambiguity|analyze-ambiguity) SCRIPT_ARGS=(analyze_ambiguity.py) ;;
  analyze_e1e2|analyze-e1e2) SCRIPT_ARGS=(analyze_e1e2.py) ;;
  setup) SCRIPT_ARGS=(wizard.py setup) ;;
  run) SCRIPT_ARGS=(wizard.py run) ;;
  doctor) SCRIPT_ARGS=(robot_doctor.py) ;;
  record-pose) SCRIPT_ARGS=(record_pose.py) ;;
  move-tcp) SCRIPT_ARGS=(move_tcp.py) ;;
  prepare) SCRIPT_ARGS=(prepare.py) ;;
  contact) SCRIPT_ARGS=(collect_contact.py) ;;
  visual) SCRIPT_ARGS=(collect_visual.py) ;;
  analyze) SCRIPT_ARGS=(analyze.py) ;;
  stop-test) SCRIPT_ARGS=(stop_test.py) ;;
  validate-force) SCRIPT_ARGS=(validate_force.py) ;;
  opening-repeat) SCRIPT_ARGS=(opening_repeat.py) ;;
  record-stress) SCRIPT_ARGS=(record_stress.py) ;;
  register-report) SCRIPT_ARGS=(register_validation.py) ;;
  python) SCRIPT_ARGS=() ;;
  *) echo 'Usage: bash run_on_5090.sh a-axis|agentic|rsi|skill-loop|primary-skill-export|skill-label-review|skill-offline|skill-collect|skill-extension|skill-feedback|vision-stage1|e1e2|ambiguity|analyze_ambiguity|analyze_e1e2|setup|run|doctor|record-pose|move-tcp|prepare|stop-test|validate-force|opening-repeat|contact|visual|analyze|record-stress|register-report|python [arguments]' >&2; exit 2 ;;
esac
if (( $# )); then shift; fi

# This standard-library portal delegates real execution to the existing V4
# scheduler. Help/review/summary never need Docker or robot imports.
if [[ "$COMMAND" == "fixed5-once" || "$COMMAND" == "state-graph-loop" || "$COMMAND" == "state-graph" || "$COMMAND" == "a-axis" || "$COMMAND" == "primary-skill-export" || "$COMMAND" == "skill-label-review" ]]; then
  cd "$ROOT_DIR"
  a_axis_python="$ROOT_DIR/.venv/bin/python"
  if [[ -e /.dockerenv || -e /run/.containerenv ]]; then a_axis_python="$ROOT_DIR/.venv_robot/bin/python"; fi
  if [[ ! -x "$a_axis_python" ]]; then a_axis_python=python3; fi
  exec "$a_axis_python" "${SCRIPT_ARGS[@]}" "$@"
fi

# Offline visual positioning plans/tests use the host venv directly. Neither
# Docker nor ROS setup is necessary for help, saved-image previews or profiles.
if [[ "$COMMAND" == "vision-stage1" || "$COMMAND" == "skill-loop" ]]; then
  vision_real=0
  vision_help=0
  for vision_arg in "$@"; do
    if [[ "$vision_arg" == "--real" ]]; then vision_real=1; fi
    if [[ "$vision_arg" == "--help" || "$vision_arg" == "-h" ]]; then vision_help=1; fi
  done
  if [[ "$vision_real" == 0 || "$vision_help" == 1 ]]; then
    cd "$ROOT_DIR"
    vision_python="$ROOT_DIR/.venv/bin/python"
    if [[ -e /.dockerenv || -e /run/.containerenv ]]; then vision_python="$ROOT_DIR/.venv_robot/bin/python"; fi
    if [[ ! -x "$vision_python" ]]; then echo "Missing offline Python: $vision_python" >&2; exit 2; fi
    exec "$vision_python" "${SCRIPT_ARGS[@]}" "$@"
  fi
fi

if [[ -e /.dockerenv || -e /run/.containerenv ]]; then
  cd "$ROOT_DIR"
  python_bin="$ROOT_DIR/.venv_robot/bin/python"
  if [[ ! -x "$python_bin" ]]; then
    echo "Missing project .venv_robot; see OPERATOR_GUIDE_zh.md" >&2
    exit 2
  fi
  # New independent analysis/help need no ROS setup. Catkin setup scripts can
  # interpret an inherited --help themselves; leave all legacy routes unchanged.
  if [[ "$COMMAND" == "analyze_e1e2" || "$COMMAND" == "analyze-e1e2" || "$COMMAND" == "analyze_ambiguity" || "$COMMAND" == "analyze-ambiguity" ]]; then
    exec "$python_bin" "${SCRIPT_ARGS[@]}" "$@"
  fi
  if [[ "$COMMAND" == "ambiguity" ]]; then
    ambiguity_real=0
    for ambiguity_arg in "$@"; do
      if [[ "$ambiguity_arg" == "--real" ]]; then ambiguity_real=1; fi
      if [[ "$ambiguity_arg" == "--help" || "$ambiguity_arg" == "-h" ]]; then
        exec "$python_bin" "${SCRIPT_ARGS[@]}" "$@"
      fi
    done
    if [[ "$ambiguity_real" == 0 ]]; then
      exec "$python_bin" "${SCRIPT_ARGS[@]}" "$@"
    fi
  fi
  if [[ "$COMMAND" == "agentic" || "$COMMAND" == "rsi" || "$COMMAND" == "skill-collect" || "$COMMAND" == "skill-extension" || "$COMMAND" == "skill-offline" || "$COMMAND" == "skill-feedback" || "$COMMAND" == "skill-loop" ]]; then
    agentic_real=0
    for agentic_arg in "$@"; do
      if [[ "$agentic_arg" == "--real" ]]; then agentic_real=1; fi
      if [[ "$agentic_arg" == "--help" || "$agentic_arg" == "-h" ]]; then
        exec "$python_bin" "${SCRIPT_ARGS[@]}" "$@"
      fi
    done
    if [[ "$agentic_real" == 0 ]]; then
      exec "$python_bin" "${SCRIPT_ARGS[@]}" "$@"
    fi
  fi
  if [[ "$COMMAND" == "e1e2" ]]; then
    for e1e2_arg in "$@"; do
      if [[ "$e1e2_arg" == "--help" || "$e1e2_arg" == "-h" ]]; then
        exec "$python_bin" "${SCRIPT_ARGS[@]}" "$@"
      fi
    done
  fi
  # Set defaults before sourcing ROS, which otherwise supplies localhost:11311.
  export ROS_MASTER_URI="${ROS_MASTER_URI:-http://192.168.9.77:11311}"
  export ROS_IP="${ROS_IP:-192.168.9.57}"
  # ROS setup scripts are not compatible with nounset.
  set +u
  source /opt/ros/noetic/setup.bash
  source /root/msg_ws/devel/setup.bash
  set -u
  unset ROS_HOSTNAME
  exec "$python_bin" "${SCRIPT_ARGS[@]}" "$@"
fi

if ! command -v docker >/dev/null 2>&1; then
  echo 'docker not found on this host; run in the 5090 ROS container or on its Docker-enabled host. See OPERATOR_GUIDE_zh.md.' >&2
  exit 2
fi
DOCKER_IO=(-i)
if [[ -t 0 && -t 1 ]]; then DOCKER_IO+=(-t); fi
exec docker exec "${DOCKER_IO[@]}" \
  -e ROS_MASTER_URI="${ROS_MASTER_URI:-http://192.168.9.77:11311}" \
  -e ROS_IP="${ROS_IP:-192.168.9.57}" \
  -w "$ROOT_DIR" "$CONTAINER" bash "$ROOT_DIR/run_on_5090.sh" "$COMMAND" "$@"
