"""Operator TCP helper tests: fake ROS only, never a real robot connection."""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest

import numpy as np

import move_tcp as cli
from bottle_scan.geometry import rotation
from bottle_scan.command_ownership import REQUIRED_COMMAND_TOPICS, guard_all_commands, inspect_all_commands


def ownership_config(_=None):
    return {"hardware": {"public_ros": {"command_ownership": {
        "topics": list(REQUIRED_COMMAND_TOPICS), "timeout_s": 1.}}}}


def empty_ownership(cfg=None, own_nodes=()):
    return guard_all_commands(cfg or ownership_config(), own_nodes,
                              system_state_getter=lambda: [[], [], []])


def state(pose=None, side="right"):
    pose = np.eye(4) if pose is None else pose
    return {"selected_arm":side, "translation_unit":"mm", "base_frame":"base_link",
            "arms": {side:{"base_frame":"base_link", "tcp_frame":side+"_gripper_base_link",
                           "T_base_tcp_mm":pose.tolist()}}}


class FakePublisher:
    def __init__(self, connected=True):
        self.connected = connected

    def get_num_connections(self):
        return int(self.connected)


class FakeArm:
    def __init__(self, arm_side="right", *, never_finish=False):
        self.side = arm_side
        self.pose = np.eye(4)
        self.connects = self.closes = self.moves = self.holds = self.drains = 0
        self.fault = None
        self._publisher = FakePublisher()
        self.target = None
        self.never_finish = never_finish
        self.fail_after_move = False
        self.wrong_completion = False
        self.snapshots = 0

    def connect(self):
        self.connects += 1
        return self

    def snapshot(self):
        self.snapshots += 1
        if self.target is not None and self.fail_after_move:
            raise RuntimeError("stale feedback")
        if self.target is not None and not self.never_finish and not self.wrong_completion:
            self.pose = self.target.copy()
        return state(self.pose, self.side)

    def move_linear(self, pose, speed, angular_speed):
        self.moves += 1
        self.target = pose.copy()
        self.speeds = (speed, angular_speed)

    def motion_done(self):
        return not self.never_finish

    def stop_arm(self):
        self.holds += 1
        self.target = None

    def drain_samples(self):
        self.drains += 1
        return [object()]

    def close(self):
        self.closes += 1


class MoveTcpTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)

    def args(self, *extra):
        return cli.validate_args(cli.parser().parse_args(["--log-dir", str(self.path / "logs"), *extra]))

    def main(self, args, arm=None, input_fn=lambda _: "MOVE", ownership_guard=empty_ownership,
             ownership_inspector=empty_ownership):
        arm = arm or FakeArm()
        factories = []
        def observer(**kwargs):
            factories.append("observer")
            return arm
        def controller(**kwargs):
            factories.append("controller")
            return arm
        output = []
        status = cli.main(["--log-dir", str(self.path/"logs"), *args], observer_factory=observer,
                          controller_factory=controller, input_fn=input_fn, output=output.append,
                          ownership_guard=ownership_guard, ownership_inspector=ownership_inspector,
                          config_loader=ownership_config)
        return status, arm, factories, output

    def records(self):
        return [json.loads(p.read_text()) for p in sorted((self.path / "logs").glob("*.json"))]

    def test_base_translation_preserves_rotation(self):
        pose = np.eye(4)
        pose[:3, :3] = rotation([0, 0, 1], 90)
        args = self.args("--delta-mm", "5", "0", "0")
        _, target = cli.build_target(args, state(pose))
        np.testing.assert_allclose(target[:3, 3], [5, 0, 0])
        np.testing.assert_allclose(target[:3, :3], pose[:3, :3])

    def test_tcp_translation_rotates_delta_from_current_tcp(self):
        pose = np.eye(4)
        pose[:3, :3] = rotation([0, 0, 1], 90)
        args = self.args("--direction", "x+", "--frame", "tcp")
        _, target = cli.build_target(args, state(pose))
        np.testing.assert_allclose(target[:3, 3], [0, 5, 0], atol=1e-9)

    def test_each_direction_and_negative_sign(self):
        for index, axis in enumerate("xyz"):
            for sign, factor in (("+", 1), ("-", -1)):
                args = self.args("--direction", axis+sign, "--distance-mm", "2")
                _, target = cli.build_target(args, state())
                expected = np.zeros(3)
                expected[index] = 2*factor
                np.testing.assert_allclose(target[:3, 3], expected)

    def test_absolute_xyz_keeps_current_orientation(self):
        pose = np.eye(4)
        pose[:3, :3] = rotation([1, 0, 0], 30)
        _, target = cli.build_target(self.args("--xyz-mm", "10", "20", "30"), state(pose))
        np.testing.assert_allclose(target[:3, 3], [10, 20, 30])
        np.testing.assert_allclose(target[:3, :3], pose[:3, :3])

    def test_absolute_quaternion_xyzw(self):
        angle = np.deg2rad(10.) / 2
        args = self.args("--xyz-mm", "0", "0", "0", "--quaternion-xyzw",
                         "0", "0", str(np.sin(angle)), str(np.cos(angle)))
        _, target = cli.build_target(args, state())
        np.testing.assert_allclose(target[:3, :3], rotation([0,0,1], 10), atol=1e-8)

    def test_invalid_target_and_numeric_options(self):
        invalid = [ ["--xyz-mm", "nan", "0", "0"],
                    ["--delta-mm", "inf", "0", "0"],
                    ["--delta-mm", "1", "0", "0", "--quaternion-xyzw", "0", "0", "0", "1"],
                    ["--xyz-mm", "0", "0", "0", "--quaternion-xyzw", "0", "0", "0", "0"],
                    ["--xyz-mm", "0", "0", "0", "--frame", "tcp"] ]
        for args in invalid:
            with self.subTest(args=args), self.assertRaises(ValueError):
                self.args(*args)
        for flag in ("--speed-mm-s", "--angular-speed-deg-s", "--max-distance-mm", "--max-angle-deg", "--timeout-s", "--distance-mm", "--pose-tolerance-mm", "--pose-tolerance-deg"):
            for value in ("0", "-1", "nan", "inf"):
                with self.subTest(flag=flag, value=value), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    self.args(flag, value)

    def test_limits_reject_large_translation_and_rotation(self):
        with self.assertRaisesRegex(ValueError, "单次限值"):
            cli.build_target(self.args("--xyz-mm", "101", "0", "0"), state())
        args = self.args("--xyz-mm", "0", "0", "0", "--quaternion-xyzw", "0", "0", "1", "0")
        with self.assertRaisesRegex(ValueError, "单次限值"):
            cli.build_target(args, state())

    def pose_file(self, data):
        path = self.path / "pose.json"
        path.write_text(json.dumps(data))
        return path

    def test_record_pose_file_accepted(self):
        target = np.eye(4)
        target[2, 3] = 8
        path = self.pose_file({"arm":"right", "label":"free", "T_base_tcp_mm":target.tolist(), "pose_source":state(target)})
        _, actual = cli.build_target(self.args("--pose-file", str(path)), state())
        np.testing.assert_allclose(actual, target)

    def test_pose_file_wrong_arm_frame_units_and_inconsistent_source_rejected(self):
        base = {"arm":"right", "T_base_tcp_mm":np.eye(4).tolist(), "translation_unit":"mm",
                "base_frame":"base_link", "tcp_frame":"right_gripper_base_link"}
        for key, value in (("arm", "left"), ("base_frame", "world"), ("tcp_frame", "tip"), ("translation_unit", "m")):
            data = dict(base, **{key:value})
            with self.subTest(key=key), self.assertRaises(ValueError):
                cli.load_pose(self.pose_file(data), "right", state())
        target = np.eye(4)
        target[0,3] = 10
        data = {"arm":"right", "T_base_tcp_mm":target.tolist(), "pose_source":state()}
        with self.assertRaisesRegex(ValueError, "不一致"):
            cli.load_pose(self.pose_file(data), "right", state())

    def test_preview_only_uses_observer_no_move_or_confirmation(self):
        def forbidden_input(_):
            self.fail("preview asked for confirmation")
        status, arm, factories, _ = self.main(["--direction", "z+"], input_fn=forbidden_input)
        self.assertEqual(status, 0)
        self.assertEqual(factories, ["observer"])
        self.assertEqual((arm.moves, arm.holds, arm.closes), (0, 0, 1))
        self.assertEqual(self.records()[0]["result"], "preview_only")

    def test_idle_operator_cancel_does_not_publish_hold(self):
        status, arm, factories, _ = self.main(["--real", "--direction", "z+"], input_fn=lambda _:"cancel")
        self.assertEqual(status, 0)
        self.assertEqual(factories, ["controller"])
        self.assertEqual((arm.moves, arm.holds), (0, 0))
        self.assertEqual(self.records()[0]["result"], "operator_cancelled")

    def test_real_requires_exactly_one_concrete_confirmation_and_measured_completion(self):
        prompts = []
        def confirm(prompt):
            prompts.append(prompt)
            self.assertEqual(len(self.records()), 1)  # Log exists before action/confirmation.
            return "MOVE"
        status, arm, _, _ = self.main(["--real", "--direction", "z+"], input_fn=confirm)
        self.assertEqual(status, 0)
        self.assertEqual(len(prompts), 1)
        self.assertEqual((arm.moves, arm.holds), (1, 0))
        record = self.records()[0]
        self.assertEqual(record["result"], "reached_by_measured_feedback")
        self.assertEqual(record["end_measured_mm"][2][3], 5)

    def test_changed_start_after_confirmation_is_rejected_without_move(self):
        arm = FakeArm()
        def confirm(_):
            arm.pose[0,3] = 2
            return "MOVE"
        status, arm, _, _ = self.main(["--real", "--direction", "z+"], arm, confirm)
        self.assertEqual(status, 2)
        self.assertEqual((arm.moves, arm.holds), (0,0))
        self.assertIn("起点变化", self.records()[0]["error"])
        self.assertEqual(self.records()[0]["target_mm"][0][3], 0)

    def test_timeout_requests_hold_for_submitted_motion(self):
        arm = FakeArm(never_finish=True)
        status, arm, _, _ = self.main(["--real", "--direction", "z+", "--timeout-s", ".01"], arm)
        self.assertEqual(status, 2)
        self.assertEqual((arm.moves, arm.holds), (1,1))
        self.assertIn("超时", self.records()[0]["error"])

    def test_no_subscriber_timeout_does_not_issue_move_or_hold(self):
        arm = FakeArm()
        arm._publisher.connected = False
        status, arm, _, _ = self.main(["--real", "--direction", "z+", "--timeout-s", ".01"], arm)
        self.assertEqual(status, 2)
        self.assertEqual((arm.moves, arm.holds), (0,0))
        self.assertIn("无订阅者", self.records()[0]["error"])

    def test_false_completion_is_rejected_and_held(self):
        arm = FakeArm()
        arm.wrong_completion = True
        status, arm, _, _ = self.main(["--real", "--direction", "z+"], arm)
        self.assertEqual(status, 2)
        self.assertEqual((arm.moves, arm.holds), (1,1))
        self.assertIn("尚未达到", self.records()[0]["error"])

    def test_measured_residual_uses_same_configured_tolerance_as_controller(self):
        class ResidualArm(FakeArm):
            def snapshot(self):
                super().snapshot()
                measured = self.pose.copy()
                if self.target is not None:
                    measured[0,3] += 1.57672
                    measured[:3,:3] = measured[:3,:3] @ rotation([0,0,1], .15736)
                return state(measured, self.side)
        for tolerance,expected in [(2.,0),(1.,2)]:
            with self.subTest(tolerance=tolerance):
                arm=ResidualArm();options={}
                def factory(**kwargs):
                    options.update(kwargs)
                    return arm
                result=cli.main(['--real','--direction','z+','--pose-tolerance-mm',str(tolerance),
                                 '--log-dir',str(self.path/'logs')],controller_factory=factory,
                                input_fn=lambda _: 'MOVE',output=lambda _:None,
                                ownership_guard=empty_ownership,config_loader=ownership_config)
                self.assertEqual(result,expected)
                self.assertEqual(options['pose_tolerance_mm'],tolerance)
                self.assertEqual(options['pose_tolerance_deg'],1.)
                self.assertEqual(arm.holds,0 if expected==0 else 1)
        self.assertEqual(self.args().pose_tolerance_mm,2.)

    def test_stale_measurements_during_motion_attempt_hold(self):
        arm = FakeArm()
        arm.fail_after_move = True
        status, arm, _, _ = self.main(["--real", "--direction", "z+"], arm)
        self.assertEqual(status, 2)
        self.assertEqual((arm.moves, arm.holds), (1,1))

    def test_drain_remains_active_through_long_operator_wait(self):
        arm = FakeArm()
        def confirm(_):
            begin = arm.drains
            time.sleep(.06)
            self.assertGreater(arm.drains, begin + 3)
            return "cancel"
        status, arm, _, _ = self.main(["--real", "--direction", "z+"], arm, confirm)
        self.assertEqual(status, 0)
        self.assertGreater(self.records()[0]["drained_source_events_total"], 3)

    def test_interrupt_in_idle_menu_never_holds(self):
        def interrupt(_):
            raise KeyboardInterrupt()
        status, arm, _, _ = self.main(["--real"], input_fn=interrupt)
        self.assertEqual(status, 130)
        self.assertEqual((arm.moves, arm.holds), (0,0))

    def test_cancellation_during_motion_holds(self):
        args = self.args("--real", "--direction", "z+")
        arm = FakeArm(never_finish=True)
        cancel = threading.Event()
        original = arm.motion_done
        def done():
            cancel.set()
            return original()
        arm.motion_done = done
        session = cli.MoveSession(args, arm, input_fn=lambda _:"MOVE", output=lambda _:None, cancel=cancel,
                                  ownership_check=empty_ownership)
        session.drain.start()
        try:
            with self.assertRaises(cli.MoveCancelled):
                session.attempt(args)
        finally:
            session.drain.close()
        self.assertEqual((arm.moves, arm.holds), (1,1))

    def test_interactive_direction_default_5mm_and_refresh_quit(self):
        values = iter(["r", "b:z+", "", "MOVE", "q"])
        status, arm, _, _ = self.main(["--real"], input_fn=lambda _:next(values))
        self.assertEqual(status, 0)
        self.assertEqual((arm.moves, arm.holds), (1,0))
        self.assertEqual(arm.pose[2,3], 5)

    def test_log_path_failure_prevents_any_command(self):
        (self.path/"logs").write_text("existing file")
        status, arm, _, _ = self.main(["--real", "--direction", "z+"])
        self.assertEqual(status, 2)
        self.assertEqual((arm.moves, arm.holds), (0,0))

    def test_log_files_do_not_overwrite_previous_attempt(self):
        for _ in range(2):
            self.assertEqual(self.main(["--direction", "z+"])[0], 0)
        self.assertEqual(len(self.records()), 2)

    def test_command_conflict_rejects_real_before_connect_and_any_motion(self):
        def conflict(cfg, own_nodes=()):
            return guard_all_commands(cfg, own_nodes, system_state_getter=lambda:
                [[["/left_gripper_command", ["/other_controller"]]], [], []])
        status, arm, _, output = self.main(["--real", "--direction", "z+"], ownership_guard=conflict)
        self.assertEqual(status, 2)
        self.assertEqual((arm.connects, arm.moves, arm.holds), (0, 0, 0))
        self.assertTrue(any("/other_controller" in line for line in output))

    def test_rechecks_publishers_after_human_wait(self):
        calls = []
        def check(cfg, own_nodes=()):
            calls.append(own_nodes)
            entries = [] if len(calls) == 1 else [["/joint_commands", ["/new_teleop"]]]
            return guard_all_commands(cfg, own_nodes,
                                      system_state_getter=lambda: [entries, [], []])
        status, arm, _, output = self.main(["--real", "--direction", "z+"], ownership_guard=check)
        self.assertEqual(status, 2)
        self.assertEqual(len(calls), 2)
        self.assertEqual((arm.connects, arm.moves, arm.holds), (1, 0, 0))
        self.assertIn("/new_teleop", self.records()[0]["error"])

    def test_preview_lists_conflicts_but_does_not_block_observation(self):
        def conflict(cfg, own_nodes=()):
            return inspect_all_commands(cfg, own_nodes, system_state_getter=lambda:
                [[["/wbc_goal_pose", ["/teleop"]]], [], []])
        status, arm, _, output = self.main(["--direction", "z+"], ownership_inspector=conflict)
        self.assertEqual(status, 0)
        self.assertEqual((arm.connects, arm.moves, arm.holds), (1, 0, 0))
        self.assertTrue(any("/teleop" in line for line in output))

    def test_success_records_before_move_ownership(self):
        status, arm, _, _ = self.main(["--real", "--direction", "z+"])
        self.assertEqual(status, 0)
        self.assertTrue(self.records()[0]["command_ownership_before_move"]["allowed"])

    def test_direct_real_session_without_guard_fails_closed(self):
        args = self.args("--real", "--direction", "z+")
        arm = FakeArm()
        session = cli.MoveSession(args, arm, input_fn=lambda _: "MOVE", output=lambda _: None)
        with self.assertRaisesRegex(RuntimeError, "未提供命令发布者检查"):
            session.attempt(args)
        self.assertEqual((arm.moves, arm.holds), (0, 0))


if __name__ == "__main__":
    unittest.main()
