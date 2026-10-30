#!/usr/bin/env bash
# Run on the 5090 host or directly inside its existing ROS container. No server start,
# controller reset, homing, model inference, or actuator command is implicit.
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONTAINER="${BOTTLE_ROS_CONTAINER:-daimon_ros1_das}"
COMMAND="${1:-doctor}"
case "$COMMAND" in
  e1e2) SCRIPT_ARGS=(e1e2.py) ;;
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
  *) echo 'Usage: bash run_on_5090.sh e1e2|analyze_e1e2|setup|run|doctor|record-pose|move-tcp|prepare|stop-test|validate-force|opening-repeat|contact|visual|analyze|record-stress|register-report|python [arguments]' >&2; exit 2 ;;
esac
if (( $# )); then shift; fi

if [[ -e /.dockerenv || -e /run/.containerenv ]]; then
  cd "$ROOT_DIR"
  python_bin="$ROOT_DIR/.venv_robot/bin/python"
  if [[ ! -x "$python_bin" ]]; then
    echo "Missing project .venv_robot; see OPERATOR_GUIDE_zh.md" >&2
    exit 2
  fi
  # New independent analysis/help need no ROS setup. Catkin setup scripts can
  # interpret an inherited --help themselves; leave all legacy routes unchanged.
  if [[ "$COMMAND" == "analyze_e1e2" || "$COMMAND" == "analyze-e1e2" ]]; then
    exec "$python_bin" "${SCRIPT_ARGS[@]}" "$@"
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
