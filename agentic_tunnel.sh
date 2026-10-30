#!/usr/bin/env bash
# Run on the laptop, not inside the ROS container. No robot commands.
set -euo pipefail
host="${1:-6f5090}"
proxy_port="${AGENTIC_LOCAL_PROXY_PORT:-7890}"
echo "保持此终端打开：5090 代理 18082 -> 本机 ${proxy_port}；登录回调 8766 -> 5090。"
exec ssh -NT -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 \
  -o ServerAliveCountMax=3 \
  -R "127.0.0.1:18082:127.0.0.1:${proxy_port}" \
  -L 127.0.0.1:8766:127.0.0.1:8766 "$host"
