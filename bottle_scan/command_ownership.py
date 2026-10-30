"""Read ROS master publisher registrations without creating any ROS publisher.

This is a point-in-time exclusion check, not a distributed command lease. Call it
before connecting a controller and again immediately before submitting a motion.
Only exact node names obtained from this process's ``rospy.get_name()`` may be
excluded by callers; a similarly named second experiment is still a conflict.
The optional deployment policy classifies one verified downstream output only;
it never exempts a node globally or hides publisher registrations.
"""
from __future__ import annotations

import math
import os
import sys
import time
from urllib.parse import urlsplit
import xmlrpc.client


# Verified from the existing Spark adapter, not inferred from topic-name patterns:
# src/lerobot/robots/spark3_pose_gripper/config_spark3_pose_gripper.py:31,33
# src/lerobot/robots/spark3_pose_gripper/schema.py:189,197.
REQUIRED_COMMAND_TOPICS = (
    "/wbc_goal_pose",          # shared left/right Cartesian command, arm mask
    "/joint_control",          # deployed hardware interface joint command input
    "/joint_commands",         # retain visibility and guard unknown publishers on output
    "/left_gripper_command",
    "/right_gripper_command",
)


_INFRASTRUCTURE_POLICIES = ("strict", "daimon_public_ros_v1")
_DAIMON_NODE = "/daimon_hw_interface_node"
_DAIMON_OUTPUT = "/joint_commands"
_DAIMON_STATE_CONSUMER = "/wbc_state_publisher"
_DAIMON_PUBLISHED_TOPICS = (
    _DAIMON_OUTPUT, "/joint_states", "/left_gripper_state", "/right_gripper_state",
)
_DAIMON_INPUT_TOPICS = (
    "/wbc_goal_pose", "/joint_control", "/left_gripper_command", "/right_gripper_command",
)


class CommandOwnershipError(RuntimeError):
    def __init__(self, report):
        self.report = report
        super().__init__("命令发布者检查拒绝启动：\n" + format_ownership_report(report))


def _absolute_name(value):
    return (isinstance(value, str) and value.startswith("/") and len(value) > 1
            and not value.endswith("/") and "//" not in value
            and not any(ch.isspace() for ch in value))


def _settings(cfg):
    return cfg.get("hardware", {}).get("public_ros", {}).get("command_ownership", {})


def validate_ownership_config(cfg):
    """Validate this section only; physical calibration is unrelated to this read."""
    errors = []
    try:
        options = _settings(cfg)
        topics = options.get("topics")
        if not isinstance(topics, list) or not topics or not all(_absolute_name(t) for t in topics):
            errors.append("hardware.public_ros.command_ownership.topics must be absolute ROS topic names")
        elif len(set(topics)) != len(topics):
            errors.append("hardware.public_ros.command_ownership.topics must not contain duplicates")
        if not isinstance(topics, list) or any(t not in topics for t in REQUIRED_COMMAND_TOPICS):
            errors.append("command_ownership.topics must include all verified command topics: "
                          + ", ".join(REQUIRED_COMMAND_TOPICS))
        policy = options.get("infrastructure_output_policy", "strict")
        if policy not in _INFRASTRUCTURE_POLICIES:
            errors.append("hardware.public_ros.command_ownership.infrastructure_output_policy must be "
                          + " or ".join(_INFRASTRUCTURE_POLICIES))
        timeout = options.get("timeout_s")
        if isinstance(timeout, bool) or not math.isfinite(float(timeout)) or float(timeout) <= 0:
            errors.append("hardware.public_ros.command_ownership.timeout_s must be positive and finite")
    except (TypeError, ValueError, AttributeError):
        errors.append("hardware.public_ros.command_ownership must specify topics and a finite positive timeout_s")
    return errors


class _TimeoutTransport(xmlrpc.client.Transport):
    def __init__(self, timeout):
        super().__init__()
        self.timeout = timeout

    def make_connection(self, host):
        connection = super().make_connection(host)
        connection.timeout = self.timeout
        return connection


class _TimeoutSafeTransport(xmlrpc.client.SafeTransport):
    def __init__(self, timeout):
        super().__init__()
        self.timeout = timeout

    def make_connection(self, host):
        connection = super().make_connection(host)
        connection.timeout = self.timeout
        return connection


def _get_system_state(timeout_s):
    uri = os.environ.get("ROS_MASTER_URI", "")
    parsed = urlsplit(uri)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise RuntimeError("ROS_MASTER_URI is unset or is not an HTTP(S) ROS master URL")
    transport = (_TimeoutSafeTransport if parsed.scheme == "https" else _TimeoutTransport)(timeout_s)
    with xmlrpc.client.ServerProxy(uri, transport=transport, allow_none=False) as master:
        # This caller id is an RPC argument only. No node/publisher is registered.
        response = master.getSystemState("/bottle_command_ownership_reader")
    if not isinstance(response, (list, tuple)) or len(response) != 3 or response[0] != 1:
        raise RuntimeError("ROS master getSystemState failed: " + repr(response))
    return response[2]


def current_ros_node_names():
    """Exact current-process identity, without importing or initializing rospy."""
    rospy = sys.modules.get("rospy")
    if rospy is None or not rospy.core.is_initialized():
        return ()
    name = rospy.get_name()
    if not _absolute_name(name) or name == "/unnamed":
        raise RuntimeError("Initialized rospy did not provide a valid exact node name")
    return (name,)


def _registration_map(entries, kind):
    """Strictly parse one part of the same master snapshot; merge duplicates."""
    if not isinstance(entries, (list, tuple)):
        raise ValueError("Malformed ROS master " + kind + " list")
    result = {}
    for entry in entries:
        if (not isinstance(entry, (list, tuple)) or len(entry) != 2
                or not _absolute_name(entry[0]) or not isinstance(entry[1], (list, tuple))
                or not all(_absolute_name(node) for node in entry[1])):
            raise ValueError("Malformed ROS master " + kind + " entry: " + repr(entry))
        result.setdefault(entry[0], set()).update(entry[1])
    return result


def _infrastructure_roles(policy, publishers, subscribers):
    """Classify the audited route only if every required live role still holds."""
    evidence = {"policy": policy, "evaluated": policy != "strict", "verified": False,
                "node": _DAIMON_NODE, "output_topic": _DAIMON_OUTPUT,
                "checks": [], "failure_reasons": []}
    if policy == "strict":
        return evidence
    checks = evidence["checks"]
    for topic in _DAIMON_PUBLISHED_TOPICS:
        actual = sorted(publishers.get(topic, ()))
        checks.append({"role": "publishes", "topic": topic, "node": _DAIMON_NODE,
                       "actual_nodes": actual, "passed": _DAIMON_NODE in actual})
    for topic in _DAIMON_INPUT_TOPICS:
        actual = sorted(subscribers.get(topic, ()))
        checks.append({"role": "subscribes", "topic": topic, "node": _DAIMON_NODE,
                       "actual_nodes": actual, "passed": _DAIMON_NODE in actual})
        actual_publishers = sorted(publishers.get(topic, ()))
        checks.append({"role": "does_not_publish_input", "topic": topic, "node": _DAIMON_NODE,
                       "actual_nodes": actual_publishers, "passed": _DAIMON_NODE not in actual_publishers})
    actual = sorted(subscribers.get(_DAIMON_OUTPUT, ()))
    checks.append({"role": "output_has_exact_consumer", "topic": _DAIMON_OUTPUT,
                   "expected_nodes": [_DAIMON_STATE_CONSUMER], "actual_nodes": actual,
                   "passed": actual == [_DAIMON_STATE_CONSUMER]})
    # The exact-consumer check also forbids the hardware node subscribing to its
    # own output. Keep this fact explicit in the saved classification evidence.
    checks.append({"role": "does_not_subscribe_output", "topic": _DAIMON_OUTPUT,
                   "node": _DAIMON_NODE, "actual_nodes": actual, "passed": _DAIMON_NODE not in actual})
    evidence["failure_reasons"] = [check["role"] + " " + check["topic"]
                                    for check in checks if not check["passed"]]
    evidence["verified"] = not evidence["failure_reasons"]
    return evidence


def inspect_all_commands(cfg, own_nodes=(), *, system_state_getter=None):
    """Return a JSON-ready report; any configuration/query error sets allowed=False.

    ``system_state_getter`` is an offline-test seam returning the ROS system-state
    payload ``[publishers, subscribers, services]``, not the XMLRPC status envelope.
    Real execution uses only the read-only master ``getSystemState`` API.
    """
    errors = validate_ownership_config(cfg)
    try:
        own = tuple(own_nodes)
        if isinstance(own_nodes, str) or not all(_absolute_name(n) for n in own):
            raise ValueError("own_nodes must contain exact absolute current-process node names")
    except (TypeError, ValueError) as exc:
        errors.append(str(exc))
        own = ()
    options = _settings(cfg) if not errors else {}
    topics = options.get("topics", list(REQUIRED_COMMAND_TOPICS))
    report = {
        "schema": "bottle_command_ownership_v2", "read_only": True,
        "created_unix_ns": time.time_ns(), "query": "ROS master getSystemState",
        "own_nodes": sorted(set(own)), "queried": False, "allowed": False,
        "topics": {topic: {"publishers": [], "subscribers": [], "external_publishers": [],
                           "infrastructure_output_publishers": [], "conflicting_publishers": []}
                   for topic in topics},
        "infrastructure_output_policy": options.get("infrastructure_output_policy", "strict"),
        "infrastructure_role_check": None,
        "conflicts": {}, "errors": errors,
        "scope": "both arms (shared Cartesian and joint topics) and both grippers",
        "limitation": "point-in-time master registrations; not an exclusive command lease",
    }
    if errors:
        return report
    try:
        getter = system_state_getter or (lambda: _get_system_state(float(options["timeout_s"])))
        state = getter()
        if not isinstance(state, (list, tuple)) or len(state) != 3:
            raise ValueError("Malformed ROS master system state")
        publishers = _registration_map(state[0], "publisher")
        subscribers = _registration_map(state[1], "subscriber")
        report["queried"] = True
        roles = _infrastructure_roles(report["infrastructure_output_policy"], publishers, subscribers)
        report["infrastructure_role_check"] = roles
        if roles["evaluated"] and not roles["verified"]:
            report["errors"].append("基础设施输出角色核对失败（不豁免发布者）：" +
                                    "; ".join(roles["failure_reasons"]))
        for topic in topics:
            all_nodes = sorted(publishers.get(topic, ()))
            external = [node for node in all_nodes if node not in own]
            infrastructure = ([_DAIMON_NODE] if roles["verified"] and topic == _DAIMON_OUTPUT
                              and _DAIMON_NODE in all_nodes else [])
            conflicting = [node for node in external if node not in infrastructure]
            report["topics"][topic] = {
                "publishers": all_nodes, "subscribers": sorted(subscribers.get(topic, ())),
                "external_publishers": external, "infrastructure_output_publishers": infrastructure,
                "conflicting_publishers": conflicting,
            }
            if conflicting:
                report["conflicts"][topic] = conflicting
        report["allowed"] = not report["conflicts"] and not report["errors"]
    except Exception as exc:
        report["errors"].append(type(exc).__name__ + ": " + str(exc))
    return report


def format_ownership_report(report):
    lines = ["命令话题发布者（双臂与双夹爪）："]
    for topic, item in report["topics"].items():
        nodes = ", ".join(item["publishers"]) if report["queried"] else "未能核对"
        lines.append("  " + topic + ": " + (nodes or "无发布者"))
        if item.get("infrastructure_output_publishers"):
            lines.append("    已核对系统输出：" + ", ".join(item["infrastructure_output_publishers"]) +
                         " → " + ", ".join(item["subscribers"]) + "（不属于竞争控制器，无需停止该系统节点）")
    roles = report.get("infrastructure_role_check")
    if roles and roles["verified"]:
        lines.append("系统输出分类依据：同次 ROS 图确认硬件节点订阅 WBC/关节/双夹爪输入，"
                     "发布关节/双夹爪状态；/joint_commands 仅由 /wbc_state_publisher 订阅。")
    if report["own_nodes"]:
        lines.append("仅排除本进程准确节点名：" + ", ".join(report["own_nodes"]))
    for topic, nodes in report["conflicts"].items():
        lines.append("冲突 " + topic + ": " + ", ".join(nodes))
    lines.extend("错误：" + error for error in report["errors"])
    lines.append("检查结果：" + ("通过" if report["allowed"] else "拒绝"))
    lines.append("这是只读来源检查，不创建命令发布者；无发布者不代表错误，实机程序连接后才注册本程序发布者。")
    return "\n".join(lines)


def guard_all_commands(cfg, own_nodes=(), *, system_state_getter=None):
    """Return the successful report, or fail closed with the complete report."""
    report = inspect_all_commands(cfg, own_nodes, system_state_getter=system_state_getter)
    if not report["allowed"]:
        raise CommandOwnershipError(report)
    return report
