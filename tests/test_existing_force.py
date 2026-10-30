import copy
import json
from pathlib import Path
import tempfile
import unittest
import io
import signal
from unittest.mock import patch

from bottle_scan.existing_force import (CLI_OVERRIDES, ExistingForceController,
    HostSupervisor, HostTransport, build_existing_argv, find_source_workers, validate_existing_settings)
from tests.test_force_control import force_config


def existing_config():
    cfg = force_config()
    cfg["force"].update(controller="existing", unit="sdk", existing={
        "host_executor_argv": ["fake_host_supervisor"], "source_path": "/host/original.py",
        "source_sha256": "a"*64, "calibration_path": "/host/calibration.json",
        "calibration_sha256": "b"*64, "host_python": "python3",
        "report_dir": "/tmp/unused_fake_force_evidence", "startup_timeout_s": 2.,
        "shutdown_timeout_s": 1., "poll_s": .01, "ros_node_hosts": ["192.168.9.57"],
        "exit_stable_s": .01, "exit_velocity_tolerance_mm_s": 1.,
        "ros_master_uri": "http://192.168.9.77:11311",
        "evaluation_probe_target": .6,
        "control_limits": {"signal": "SDK_1.2.11_saved_reference_39010", "max": 4., "contact_min": .2, "target_tolerance": .2},
        "cli_overrides": dict(position_min=300, position_max=1000, settle_time=1,
            tactile_timeout=2, position_timeout=2, max_run_time=60, max_pair_skew=.5,
            startup_timeout=20, step=1)})
    cfg["hardware"]["gripper_width_calibration"] = [{"normalized": .3, "width_mm": 20.}, {"normalized": 1., "width_mm": 90.}]
    return cfg


class Gripper:
    def __init__(self): self.events = []; self.fresh = True; self.sequence=0
    def snapshot(self):
        self.sequence+=1
        return {"fresh": self.fresh, "width_mm": 36., "normalized": .4,
                "velocity_mm_s":0., "source_sequence":self.sequence, "min_width_mm": 30.}
    def suspend_publication(self): self.events.append("suspend")
    def resume_publication(self, **kwargs): self.events.append("resume")
    def hold_gripper(self): self.events.append("hold")


class FakeTransport:
    def __init__(self, cfg): self.calls = []; self.stop_fails = False; self.closed = False
    def request(self, op, **payload):
        self.calls.append((op, payload))
        if op == "start": return {"started": True, "csv_path": payload["csv_path"]}
        if op == "status":
            return {"running": True, "workers_gone": False, "owned_nodes": ["/actual_verified_worker_123"],
                    "workers": [{"kind": "ros_tag"}, {"kind": "tactile_tag"}]}
        if op == "stop":
            return {"running": self.stop_fails, "workers_gone": not self.stop_fails, "owned_nodes": []}
        raise AssertionError(op)
    def close(self): self.closed = True


class ExistingForceTests(unittest.TestCase):
    def test_original_full_cli_argv_not_pure_function_substitute(self):
        cfg = existing_config()
        argv = build_existing_argv(cfg, 2., "/tmp/new.csv")
        self.assertEqual(argv[:2], ["python3", "/host/original.py"])
        self.assertIn("--execute", argv)
        self.assertEqual(argv[argv.index("--target-sdk-force")+1], "2.0")
        self.assertIn("--calibration", argv)
        self.assertNotIn("--internal-worker", argv)

    def test_sum_target_and_band_convert_explicitly_to_existing_mean(self):
        cfg = existing_config()
        cfg["force"]["target_definition"] = "sum"
        argv = build_existing_argv(cfg, 4., "new.csv")
        self.assertEqual(argv[argv.index("--target-sdk-force")+1], "2.0")
        self.assertEqual(float(argv[argv.index("--deadband")+1]), .1)

    def test_existing_source_limits_are_explicit_not_39020_guard_aliases(self):
        cfg = existing_config()
        cfg["force"]["existing"]["control_limits"].update(max=9., contact_min=.1, target_tolerance=.05)
        argv=build_existing_argv(cfg,2.,"new.csv")
        self.assertEqual(float(argv[argv.index("--max-force")+1]),9.)
        self.assertEqual(float(argv[argv.index("--contact-min-force")+1]),.1)
        self.assertEqual(float(argv[argv.index("--deadband")+1]),.05)
        del cfg["force"]["existing"]["control_limits"]
        with self.assertRaisesRegex(ValueError,"control_limits"):
            build_existing_argv(cfg,2.,"new.csv")

    def test_N_requires_source_specific_conversion_and_no_assumed_factor(self):
        cfg = existing_config()
        cfg["force"]["unit"] = "N"
        with self.assertRaisesRegex(ValueError, "n_per_gf"):
            build_existing_argv(cfg, 2., "new.csv")
        cfg["force"]["existing"].update(n_per_gf=.01, n_conversion_evidence="measured fixture")
        argv = build_existing_argv(cfg, 2., "new.csv")
        self.assertEqual(argv[argv.index("--target-force")+1], "200.0")

    def test_unavailable_executor_and_per_finger_definition_fail_closed(self):
        cfg = existing_config()
        cfg["force"]["existing"]["host_executor_argv"] = ["TODO"]
        with self.assertRaises(ValueError): validate_existing_settings(cfg)
        cfg = existing_config()
        cfg["force"]["target_definition"] = "per_finger"
        with self.assertRaisesRegex(ValueError, "mean"):
            validate_existing_settings(cfg)

    def controller(self, force=(1., 1.)):
        cfg, grip, transports, checks = existing_config(), Gripper(), [], []
        def factory(cfg):
            transport = FakeTransport(cfg)
            transports.append(transport)
            return transport
        controller = ExistingForceController(cfg, grip, lambda: {"normal": force},
            ownership_check=lambda own: checks.append(own), transport_factory=factory)
        self.addCleanup(controller.close)
        return controller, grip, transports, checks

    def test_handoff_suspends_before_start_and_reacquires_only_after_proven_exit(self):
        controller, grip, transports, checks = self.controller()
        controller.set_target(2.)
        self.assertEqual(grip.events, ["suspend"])
        self.assertEqual(controller.check(), ("/actual_verified_worker_123",))
        self.assertEqual(checks[-1], ("/actual_verified_worker_123",))
        controller.exit_force_mode()
        self.assertEqual(grip.events, ["suspend", "resume", "hold"])
        self.assertTrue(transports[0].closed)
        self.assertFalse(controller.active)

    def test_existing_cannot_bypass_runtime_geometric_width_floor(self):
        controller, grip, transports, _ = self.controller()
        controller.set_target(.6)
        effective = transports[0].calls[0][1]["cfg"]
        self.assertAlmostEqual(effective["force"]["existing"]["cli_overrides"]["position_min"], 400.)
        self.assertEqual(controller.cfg["force"]["existing"]["cli_overrides"]["position_min"], 300)

    def test_missing_width_floor_refuses_before_publisher_handoff(self):
        controller, grip, transports, _ = self.controller()
        snapshot = grip.snapshot
        def missing():
            result = snapshot()
            result.pop("min_width_mm")
            return result
        grip.snapshot = missing
        with self.assertRaisesRegex(RuntimeError, "geometric width floor"):
            controller.set_target(.6)
        self.assertFalse(grip.events)
        self.assertFalse(transports)

    def test_target_change_exits_complete_old_cli_before_new_start(self):
        controller, grip, transports, _ = self.controller()
        controller.set_target(.6)
        controller.set_target(2.)
        self.assertEqual(len(transports), 2)
        self.assertTrue(transports[0].closed)
        self.assertEqual(grip.events, ["suspend", "resume", "hold", "suspend"])
        self.assertEqual(transports[1].calls[0][1]["target"], 2.)

    def test_failed_worker_exit_never_reacquires_or_releases(self):
        controller, grip, transports, _ = self.controller()
        controller.set_target(2.)
        transports[0].stop_fails = True
        with self.assertRaisesRegex(RuntimeError, "workers exit"):
            controller.exit_force_mode()
        self.assertEqual(grip.events, ["suspend"])
        self.assertIsNotNone(controller.fault)

    def test_stale_feedback_and_missing_contact_prevent_start(self):
        controller, grip, transports, _ = self.controller()
        grip.fresh = False
        with self.assertRaises(RuntimeError): controller.set_target(2.)
        self.assertFalse(transports)
        controller, grip, transports, _ = self.controller(force=(0., 1.))
        with self.assertRaisesRegex(RuntimeError, "bilateral"):
            controller.set_target(2.)
        self.assertFalse(transports)

    def test_cleanup_has_no_implicit_actuation(self):
        controller, grip, transports, _ = self.controller()
        controller.set_target(2.)
        controller.close()
        self.assertEqual(grip.events, ["suspend"])
        self.assertTrue(transports[0].closed)

    def test_transport_rejects_different_ros_master_before_host_execution(self):
        with patch.dict("os.environ", {"ROS_MASTER_URI":"http://other-master:11311"}), \
                self.assertRaisesRegex(RuntimeError,"same explicit ROS_MASTER_URI"):
            HostTransport(existing_config(),popen=lambda *a,**k:self.fail("must not start host"))

    def test_proc_match_requires_exact_source_and_worker_role(self):
        source = "print('never executed')"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for pid, program, text in ((101, b"python3", source), (102, b"python3", "different"),
                                       (103, b"docker", source)):
                directory = root / str(pid)
                directory.mkdir()
                (directory / "cmdline").write_bytes(b"\0".join([program, b"-c", text.encode(), b"--internal-worker", b"ros_tag", b""]))
                (directory / "status").write_text("NSpid:\t%d\t7\n" % pid)
                (directory / "stat").write_text(str(pid) + " (python3) " + " ".join(["0"]*19+["1234"]))
            matches = find_source_workers(source, root)
        self.assertEqual(matches, [{"host_pid": 101, "namespace_pids": [101, 7], "kind": "ros_tag", "start_ticks": 1234}])

    def test_supervisor_waits_for_actual_workers_after_parent_exit(self):
        events, now = [], [0.]
        class Process:
            def poll(self): return None
            def send_signal(self, sig): events.append(("signal", sig))
            def wait(self, **kwargs): events.append(("wait", kwargs))
        supervisor = HostSupervisor(monotonic=lambda: now[0], sleep=lambda dt: now.__setitem__(0, now[0]+dt))
        supervisor.options = {"shutdown_timeout_s": 1., "poll_s": .1}
        supervisor.process, supervisor.log = Process(), io.StringIO()
        states = iter([{"running": False, "workers_gone": False},
                       {"running": False, "workers_gone": True}])
        supervisor.status = lambda: next(states)
        self.assertTrue(supervisor.stop()["workers_gone"])
        self.assertEqual(events[0], ("signal", signal.SIGINT))
        self.assertAlmostEqual(now[0], .1)
        self.assertIsNone(supervisor.process)

    def test_supervisor_timeout_retains_handle_and_never_claims_exit(self):
        now = [0.]
        class Process:
            def poll(self): return None
            def send_signal(self, sig): pass
        supervisor = HostSupervisor(monotonic=lambda: now[0], sleep=lambda dt: now.__setitem__(0, now[0]+dt))
        supervisor.options = {"shutdown_timeout_s": .2, "poll_s": .1}
        process = supervisor.process = Process()
        supervisor.log = io.StringIO()
        supervisor.status = lambda: {"running": False, "workers_gone": False}
        with self.assertRaisesRegex(RuntimeError, "must NOT"):
            supervisor.stop()
        self.assertIs(supervisor.process, process)
        self.assertFalse(supervisor.log.closed)

    def test_changed_original_source_is_blocked_before_process_creation(self):
        cfg = existing_config()
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "original.py"
            source.write_text("raise AssertionError('must never execute')")
            cfg["force"]["existing"]["source_path"] = str(source)
            supervisor = HostSupervisor(popen=lambda *a, **k: self.fail("must not spawn"))
            with patch.object(Path, "exists", return_value=False), self.assertRaisesRegex(RuntimeError, "checksum"):
                supervisor.start(cfg, 2., Path(tmp) / "new.csv")


if __name__ == "__main__": unittest.main()
