"""Command exclusions use fake XMLRPC state; tests never contact a ROS master."""
import contextlib
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from bottle_scan import command_ownership as ownership


def config():
    return {"hardware": {"public_ros": {"command_ownership": {
        "topics": list(ownership.REQUIRED_COMMAND_TOPICS), "timeout_s": 3.}}}}


def inspect(publishers=(), own_nodes=(), cfg=None):
    return ownership.inspect_all_commands(cfg or config(), own_nodes,
        system_state_getter=lambda: [list(publishers), [], []])


class OwnershipTests(unittest.TestCase):
    def test_empty_commands_report_all_verified_topics(self):
        report = inspect()
        self.assertTrue(report["allowed"])
        self.assertTrue(report["read_only"])
        self.assertEqual(set(report["topics"]), set(ownership.REQUIRED_COMMAND_TOPICS))

    def test_each_arm_and_gripper_topic_is_independently_guarded(self):
        for topic in ownership.REQUIRED_COMMAND_TOPICS:
            with self.subTest(topic=topic):
                report = inspect([(topic, ["/teleop"])])
                self.assertFalse(report["allowed"])
                self.assertEqual(report["conflicts"], {topic: ["/teleop"]})

    def test_all_publishers_are_listed_even_duplicate_master_entries(self):
        report = inspect([("/wbc_goal_pose", ["/b", "/a"]), ("/wbc_goal_pose", ["/c", "/a"])])
        self.assertEqual(report["topics"]["/wbc_goal_pose"]["publishers"], ["/a", "/b", "/c"])

    def test_exact_own_name_only_and_no_prefix_allowlist(self):
        report = inspect([("/wbc_goal_pose", ["/bottle_arm_observer_123", "/bottle_arm_observer_456"])],
                         own_nodes=("/bottle_arm_observer_123",))
        self.assertEqual(report["conflicts"]["/wbc_goal_pose"], ["/bottle_arm_observer_456"])
        self.assertEqual(len(report["topics"]["/wbc_goal_pose"]["publishers"]), 2)

    def test_only_current_program_publisher_is_permitted(self):
        report = inspect([("/wbc_goal_pose", ["/this_exact_node"])], ("/this_exact_node",))
        self.assertTrue(report["allowed"])
        self.assertEqual(report["topics"]["/wbc_goal_pose"]["publishers"], ["/this_exact_node"])

    def test_state_and_unrelated_publishers_do_not_conflict(self):
        report = inspect([("/wbc_state_pose", ["/robot"]), ("/joint_states", ["/robot"]),
                          ("/camera/left/command_metadata", ["/camera"]), ("/rosout", ["/teleop"])])
        self.assertTrue(report["allowed"])

    def test_additional_explicit_verified_command_topic_is_guarded(self):
        cfg = config()
        cfg["hardware"]["public_ros"]["command_ownership"]["topics"].append("/deployment_extra_command")
        report = inspect([("/deployment_extra_command", ["/extra"])], cfg=cfg)
        self.assertEqual(report["conflicts"], {"/deployment_extra_command": ["/extra"]})

    def test_missing_topic_cannot_disable_protection(self):
        cfg = config()
        cfg["hardware"]["public_ros"]["command_ownership"]["topics"].pop()
        queried = []
        report = ownership.inspect_all_commands(cfg, system_state_getter=lambda: queried.append(True))
        self.assertFalse(report["allowed"])
        self.assertFalse(queried)
        self.assertIn("include all", report["errors"][0])

    def test_invalid_configuration_fails_closed(self):
        for patch_value in ({"topics": []}, {"topics": ["relative"]}, {"topics": None},
                            {"timeout_s": 0}, {"timeout_s": True}, {"timeout_s": "TODO"},
                            {"timeout_s": float("inf")}, {"timeout_s": None}):
            with self.subTest(value=patch_value):
                cfg = config()
                cfg["hardware"]["public_ros"]["command_ownership"].update(patch_value)
                self.assertFalse(ownership.inspect_all_commands(cfg)["allowed"])
        self.assertFalse(ownership.inspect_all_commands({})["allowed"])

    def test_duplicate_topic_is_rejected(self):
        cfg = config()
        cfg["hardware"]["public_ros"]["command_ownership"]["topics"].append("/wbc_goal_pose")
        self.assertIn("duplicates", ownership.validate_ownership_config(cfg)[0])

    def test_query_failure_raises_with_actionable_report(self):
        def unavailable():
            raise TimeoutError("master unavailable")
        with self.assertRaises(ownership.CommandOwnershipError) as cm:
            ownership.guard_all_commands(config(), system_state_getter=unavailable)
        self.assertFalse(cm.exception.report["queried"])
        self.assertIn("master unavailable", str(cm.exception))
        self.assertIn("/left_gripper_command", str(cm.exception))

    def test_malformed_master_payload_never_looks_like_no_publishers(self):
        for state in (None, [], [[], []], [["bad"], [], []],
                      [[["/wbc_goal_pose", None]], [], []],
                      [[["/wbc_goal_pose", ["relative_node"]]], [], []]):
            with self.subTest(state=state):
                report = ownership.inspect_all_commands(config(), system_state_getter=lambda: state)
                self.assertFalse(report["allowed"])
                self.assertTrue(report["errors"])

    def test_guard_error_prints_all_topics_and_conflicting_names(self):
        with self.assertRaises(ownership.CommandOwnershipError) as cm:
            ownership.guard_all_commands(config(), system_state_getter=lambda:
                [[["/right_gripper_command", ["/force_controller"]]], [], []])
        for text in (*ownership.REQUIRED_COMMAND_TOPICS, "/force_controller"):
            self.assertIn(text, str(cm.exception))

    def test_master_call_has_no_other_methods_and_uses_timeout(self):
        calls = []
        class Master:
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def getSystemState(self, caller):
                calls.append(("getSystemState", caller))
                return [1, "ok", [[], [], []]]
        def server(uri, **kwargs):
            calls.append((uri, kwargs["transport"].timeout))
            return Master()
        with patch.dict("os.environ", {"ROS_MASTER_URI": "http://fake.invalid:11311"}), \
                patch.object(ownership.xmlrpc.client, "ServerProxy", side_effect=server):
            report = ownership.guard_all_commands(config())
        self.assertTrue(report["allowed"])
        self.assertEqual(calls, [("http://fake.invalid:11311", 3.),
                                ("getSystemState", "/bottle_command_ownership_reader")])

    def test_failed_master_status_is_refused(self):
        class Master:
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def getSystemState(self, caller): return [-1, "failure", [[], [], []]]
        with patch.dict("os.environ", {"ROS_MASTER_URI": "http://fake.invalid:11311"}), \
                patch.object(ownership.xmlrpc.client, "ServerProxy", return_value=Master()):
            self.assertFalse(ownership.inspect_all_commands(config())["allowed"])

    def test_missing_master_uri_is_not_defaulted_to_another_robot(self):
        with patch.dict("os.environ", {}, clear=True):
            report = ownership.inspect_all_commands(config())
        self.assertFalse(report["allowed"])
        self.assertIn("ROS_MASTER_URI", report["errors"][0])

    def test_current_node_uses_initialized_rospy_exact_identity(self):
        fake = SimpleNamespace(core=SimpleNamespace(is_initialized=lambda: True),
                               get_name=lambda: "/bottle_arm_observer_123")
        with patch.dict(sys.modules, {"rospy": fake}):
            self.assertEqual(ownership.current_ros_node_names(), ("/bottle_arm_observer_123",))
        fake.core.is_initialized = lambda: False
        with patch.dict(sys.modules, {"rospy": fake}):
            self.assertEqual(ownership.current_ros_node_names(), ())

    def test_string_not_accepted_as_own_node_sequence(self):
        self.assertFalse(inspect(own_nodes="/bottle_node")["allowed"])

    def test_doctor_lists_conflicts_and_returns_failure_without_actuation(self):
        import robot_doctor
        class Arm:
            def __init__(self, **kwargs): pass
            def connect(self): pass
            def snapshot(self): return {"read_only": True}
            def close(self): pass
        class Grip(Arm):
            def snapshot(self): return {"fresh": True}
        report = inspect([("/joint_commands", ["/teleop"])])
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(robot_doctor, "RosArmObserver", Arm), \
                patch.object(robot_doctor, "RosNormalizedGripperObserver", Grip), \
                patch.object(robot_doctor, "load_config", return_value=config()), \
                patch.object(robot_doctor, "inspect_all_commands", return_value=report), \
                contextlib.redirect_stdout(io.StringIO()) as output:
            path = Path(tmp) / "doctor.json"
            result = robot_doctor.main(["--no-dd", "--no-wrench", "--no-cameras", "--output", str(path)])
            saved = json.loads(path.read_text())
        self.assertEqual(result, 1)
        self.assertFalse(saved["read_only_checks_passed"])
        self.assertEqual(saved["checks"]["command_ownership"]["conflicts"], {"/joint_commands": ["/teleop"]})
        self.assertIn("/teleop", output.getvalue())


class InfrastructureRouteTests(unittest.TestCase):
    """Observed public ROS topology, supplied entirely offline."""
    node = "/daimon_hw_interface_node"
    state_consumer = "/wbc_state_publisher"

    def graph(self):
        return [
            [[topic, [self.node]] for topic in
             ("/joint_commands", "/joint_states", "/left_gripper_state", "/right_gripper_state")],
            [[topic, [self.node]] for topic in
             ("/wbc_goal_pose", "/joint_control", "/left_gripper_command", "/right_gripper_command")]
            + [["/joint_commands", [self.state_consumer]]],
            [],
        ]

    def inspect_graph(self, state=None, policy="daimon_public_ros_v1", own_nodes=()):
        cfg = config()
        cfg["hardware"]["public_ros"]["command_ownership"]["infrastructure_output_policy"] = policy
        state = self.graph() if state is None else state
        return ownership.inspect_all_commands(cfg, own_nodes, system_state_getter=lambda: state)

    def test_exact_live_roles_classify_only_known_system_output(self):
        report = self.inspect_graph()
        self.assertTrue(report["allowed"], report)
        item = report["topics"]["/joint_commands"]
        self.assertEqual(item["publishers"], [self.node])
        self.assertEqual(item["external_publishers"], [self.node])
        self.assertEqual(item["infrastructure_output_publishers"], [self.node])
        self.assertEqual(item["conflicting_publishers"], [])
        self.assertEqual(item["subscribers"], [self.state_consumer])
        evidence = report["infrastructure_role_check"]
        self.assertTrue(evidence["verified"])
        self.assertTrue(all(check["passed"] for check in evidence["checks"]))
        # Entire audit evidence remains serializable and visible in doctor output.
        json.dumps(report)
        formatted = ownership.format_ownership_report(report)
        self.assertIn("已核对系统输出", formatted)
        self.assertIn("无需停止该系统节点", formatted)
        self.assertIn("无发布者不代表错误", formatted)
        self.assertIn("同次 ROS 图", formatted)

    def test_default_strict_policy_does_not_exempt_infrastructure(self):
        state = self.graph()
        report = ownership.inspect_all_commands(config(), system_state_getter=lambda: state)
        self.assertFalse(report["allowed"])
        self.assertEqual(report["conflicts"], {"/joint_commands": [self.node]})
        self.assertFalse(report["infrastructure_role_check"]["evaluated"])

    def test_unknown_publisher_on_output_is_still_a_conflict(self):
        state = self.graph()
        state[0][0][1].append("/unknown_controller")
        report = self.inspect_graph(state)
        self.assertFalse(report["allowed"])
        self.assertTrue(report["infrastructure_role_check"]["verified"])
        self.assertEqual(report["conflicts"], {"/joint_commands": ["/unknown_controller"]})
        self.assertEqual(report["topics"]["/joint_commands"]["publishers"],
                         [self.node, "/unknown_controller"])

    def test_known_node_publishing_any_input_is_not_exempt(self):
        for topic in ("/wbc_goal_pose", "/joint_control", "/left_gripper_command", "/right_gripper_command"):
            with self.subTest(topic=topic):
                state = self.graph()
                state[0].append([topic, [self.node]])
                report = self.inspect_graph(state)
                self.assertFalse(report["allowed"])
                self.assertEqual(report["conflicts"][topic], [self.node])
                self.assertFalse(report["infrastructure_role_check"]["verified"])
                self.assertEqual(report["topics"]["/joint_commands"]["infrastructure_output_publishers"], [])

    def test_every_required_published_and_subscribed_role_must_exist(self):
        for group_index in (0, 1):
            for entry_index in range(len(self.graph()[group_index])):
                with self.subTest(group=group_index, entry=entry_index):
                    state = self.graph()
                    state[group_index].pop(entry_index)
                    report = self.inspect_graph(state)
                    self.assertFalse(report["allowed"])
                    self.assertTrue(report["errors"])
                    self.assertFalse(report["infrastructure_role_check"]["verified"])

    def test_output_consumer_must_be_exact_and_cannot_be_hardware_node(self):
        for consumers in ([], ["/unexpected_driver"], [self.node],
                          [self.state_consumer, "/observer"], [self.state_consumer, self.node]):
            with self.subTest(consumers=consumers):
                state = self.graph()
                state[1][-1][1] = consumers
                report = self.inspect_graph(state)
                self.assertFalse(report["allowed"])
                self.assertFalse(report["infrastructure_role_check"]["verified"])
                self.assertEqual(report["conflicts"]["/joint_commands"], [self.node])

    def test_joint_control_input_foreign_publisher_is_guarded(self):
        state = self.graph()
        state[0].append(["/joint_control", ["/joint_teleop"]])
        report = self.inspect_graph(state)
        self.assertFalse(report["allowed"])
        self.assertTrue(report["infrastructure_role_check"]["verified"])
        self.assertEqual(report["conflicts"], {"/joint_control": ["/joint_teleop"]})

    def test_own_publisher_on_input_and_system_output_can_coexist(self):
        state = self.graph()
        state[0].append(["/wbc_goal_pose", ["/bottle_current"]])
        report = self.inspect_graph(state, own_nodes=("/bottle_current",))
        self.assertTrue(report["allowed"])
        self.assertEqual(report["topics"]["/wbc_goal_pose"]["publishers"], ["/bottle_current"])

    def test_unknown_policy_is_rejected_before_reading_ros(self):
        for value in ("allow_any", "daimon_public_ros_v2", None, True, [], {}):
            with self.subTest(policy=value):
                cfg = config()
                cfg["hardware"]["public_ros"]["command_ownership"]["infrastructure_output_policy"] = value
                with patch.object(ownership, "_get_system_state") as query:
                    report = ownership.inspect_all_commands(cfg)
                self.assertFalse(report["allowed"])
                self.assertIn("infrastructure_output_policy", report["errors"][0])
                query.assert_not_called()

    def test_malformed_subscriber_payload_fails_closed_even_with_strict_policy(self):
        for subscribers in (None, {}, ["bad"], [["/joint_commands", None]],
                            [["/joint_commands", ["relative_node"]]], [["relative_topic", [self.node]]]):
            for policy in ("strict", "daimon_public_ros_v1"):
                with self.subTest(subscribers=subscribers, policy=policy):
                    state = self.graph()
                    state[1] = subscribers
                    report = self.inspect_graph(state, policy)
                    self.assertFalse(report["allowed"])
                    self.assertTrue(report["errors"])
                    self.assertFalse(report["queried"])

    def test_roles_and_conflicts_use_one_snapshot_only(self):
        cfg = config()
        cfg["hardware"]["public_ros"]["command_ownership"]["infrastructure_output_policy"] = "daimon_public_ros_v1"
        with patch.object(ownership, "_get_system_state", return_value=self.graph()) as query:
            report = ownership.inspect_all_commands(cfg)
        self.assertTrue(report["allowed"])
        query.assert_called_once_with(3.)


if __name__ == "__main__":
    unittest.main()
