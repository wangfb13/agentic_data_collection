"""CLI/session regressions. Plans and device seams are mocked; no hardware I/O."""
from __future__ import annotations

import builtins
import contextlib
import copy
import io
import sys
import types
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

import prepare
from bottle_scan import cli
from bottle_scan.config import demo_config
from bottle_scan.session import Session


class FakeBackend:
    simulated = True

    def __init__(self):
        self.calls = []
        self.now_ns = 1000

    def drain_samples(self):
        return []

    def step(self, dt):
        self.now_ns += 1

    def monotonic_ns(self):
        return self.now_ns

    def close(self):
        self.calls.append("close")

    def stop_arm(self):
        self.calls.append("stop_arm")

    def hold_gripper(self):
        self.calls.append("hold_gripper")


class FakeRunner:
    fail_initial_check = False

    def __init__(self, session, backend, **kwargs):
        self.session = session
        self.backend = backend
        self.airborne = False
        self.meta = None
        self.stage = "IDLE"
        self.actuation_started = False

    def check_bottle(self, reason):
        self.actuation_started = True
        self.stage = "CHECK"
        self.backend.calls.append("check:" + reason)
        if reason == "run_start" and self.fail_initial_check:
            raise RuntimeError("synthetic initial CHECK failure")

    def ask(self, text):
        return ""


class CliTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.cfg = demo_config()
        self.item = {"trial_id": "trial_0001", "bottle": "light", "kind": "grid", "valid": False,
            "safety_reasons": ["synthetic_rejection"], "delta_acl_mm": [0, 0, 0],
            "T_target": self.cfg["T0"], "T_pre": self.cfg["T0"]}

    def session(self, *, simulated=True, cfg=None, name="session"):
        cfg = copy.deepcopy(self.cfg if cfg is None else cfg)
        with patch("bottle_scan.session.code_version", return_value={"git_commit": None, "source_sha256": "unit-test"}):
            return Session.create(self.root/name, cfg, np.asarray(cfg["T0"]), simulated=simulated)

    @contextlib.contextmanager
    def forbid_device_imports(self):
        original = builtins.__import__
        def guarded(name, *args, **kwargs):
            if name in ("backend", "bottle_scan.backend", "runtime", "bottle_scan.runtime"):
                raise AssertionError("Device/runtime import during offline or rejected invocation: " + name)
            return original(name, *args, **kwargs)
        with patch("builtins.__import__", side_effect=guarded):
            yield

    @contextlib.contextmanager
    def fake_devices(self, backend=None, runner=FakeRunner):
        backend_module = types.ModuleType("bottle_scan.backend")
        factory = Mock(return_value=backend or FakeBackend())
        backend_module.SimBackend = factory
        backend_module.load_hardware = Mock(side_effect=AssertionError("hardware must not load in this unit test"))
        runtime_module = types.ModuleType("bottle_scan.runtime")
        runtime_module.Runner = runner
        with patch.dict(sys.modules, {"bottle_scan.backend": backend_module, "bottle_scan.runtime": runtime_module}):
            yield factory

    def test_default_dry_run_and_prepare_do_not_import_backend(self):
        with self.forbid_device_imports(), contextlib.redirect_stdout(io.StringIO()), \
                patch.object(cli, "plan_for", return_value=[self.item]), \
                patch.object(cli, "dry_report", return_value={}):
            self.assertEqual(cli.main("contact", ["--demo", "--output", str(self.root/"report")]), 0)
            self.assertEqual(prepare.main(["--dry-run", "--demo", "--session", str(self.root/"not_created")]), 0)
        self.assertFalse((self.root/"not_created").exists())

    def test_real_todo_gate_happens_before_backend_import(self):
        with self.forbid_device_imports(), contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(ValueError, "Real mode refused"):
                cli.main("contact", ["--real"])
            with self.assertRaisesRegex(ValueError, "TODO gate"):
                prepare.main(["--real", "--single-step"])

    def test_demo_flags_cannot_enable_real_mode(self):
        with self.forbid_device_imports():
            for argv in (["--real", "--demo"],):
                with self.assertRaises(ValueError):
                    cli.main("contact", argv)
                with self.assertRaises(ValueError):
                    prepare.main(argv)
            with self.assertRaises(ValueError):
                prepare.main(["--real", "--small-demo"])

    def test_session_synthetic_real_separation_before_backend_import(self):
        synthetic = self.session()
        real_cfg = demo_config()
        real_cfg["demo"] = False
        real = self.session(simulated=False, cfg=real_cfg, name="real")
        with self.forbid_device_imports(), patch.object(cli, "plan_for", return_value=[self.item]), \
                patch.object(cli, "dry_report", return_value={}):
            for mode, session in (("--real", synthetic), ("--simulate", real)):
                with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, "Cannot mix"):
                    cli.main("contact", [mode, "--session", str(session.path)])

    def test_trial_id_namespaces_separate_commission_contact_and_visual(self):
        raw = [dict(self.item, trial_id="trial_0001")]
        with patch.object(cli, "commissioning_trials", return_value=copy.deepcopy(raw)), \
                patch.object(cli, "contact_trials", return_value=copy.deepcopy(raw)), \
                patch.object(cli, "visual_points", return_value=[{"point_id": "point_0001"}]):
            commission = cli.plan_for("contact", self.cfg, self.cfg["T0"], True)
            normal = cli.plan_for("contact", self.cfg, self.cfg["T0"])
            visual = cli.plan_for("visual", self.cfg, self.cfg["T0"])
        self.assertEqual(commission[0]["trial_id"], "commission_0001")
        self.assertEqual(normal[0]["trial_id"], "trial_0001")
        self.assertEqual(visual[0]["point_id"], "point_0001")

    def test_plan_mismatch_is_rejected_before_backend_creation(self):
        session = self.session()
        session.lock_plan("contact", [self.item])
        changed = dict(self.item, delta_acl_mm=[1, 0, 0])
        with patch.object(cli, "plan_for", return_value=[changed]), patch.object(cli, "dry_report", return_value={}), \
                self.fake_devices() as factory:
            with self.assertRaisesRegex(ValueError, "Plan changed"):
                cli.main("contact", ["--simulate", "--session", str(session.path)])
            factory.assert_not_called()

    def test_resume_skips_completed_without_overwriting_and_locks_full_plan(self):
        session = self.session()
        session.start_trial("trial_0001")
        session.complete({"trial_id": "trial_0001", "status": "completed", "task": "contact", "bottle": "light"})
        meta_path = session.path/"trials"/"trial_0001"/"meta.json"
        original = meta_path.read_bytes()
        plan = [self.item, dict(self.item, trial_id="trial_0002")]
        backend = FakeBackend()
        with patch.object(cli, "plan_for", return_value=plan), patch.object(cli, "dry_report", return_value={}), \
                self.fake_devices(backend):
            with self.assertRaisesRegex(ValueError, "--resume"):
                cli.main("contact", ["--simulate", "--session", str(session.path)])
            self.assertEqual(cli.main("contact", ["--simulate", "--session", str(session.path), "--resume", "--limit", "1"]), 0)
        refreshed = Session(session.path)
        self.assertEqual(refreshed.progress["completed"], ["trial_0001", "trial_0002"])
        self.assertEqual(meta_path.read_bytes(), original)
        self.assertEqual(len(__import__("json").loads((session.path/"plans"/"contact.json").read_text())), 2)
        self.assertEqual(backend.calls[-1], "close")

    def test_active_attempt_archived_completed_never_reused(self):
        session = self.session()
        first = session.start_trial("trial_0001")
        (first/"evidence.txt").write_text("preserve partial data")
        session.start_trial("trial_0001")
        evidence = list((session.path/"interrupted").glob("trial_0001_*/evidence.txt"))
        self.assertEqual(len(evidence), 1)
        self.assertEqual(evidence[0].read_text(), "preserve partial data")
        session.complete({"trial_id": "trial_0001", "status": "completed"})
        with self.assertRaisesRegex(ValueError, "overwrite"):
            session.start_trial("trial_0001")

    def test_no_pending_returns_without_backend_or_confirmation(self):
        session = self.session()
        session.start_trial("trial_0001")
        session.complete({"trial_id":"trial_0001", "status":"completed"})
        with patch.object(cli, "plan_for", return_value=[self.item]), patch.object(cli, "dry_report", return_value={}), \
                self.forbid_device_imports(), patch("builtins.input", side_effect=AssertionError("no pending must not prompt")), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(cli.main("contact", ["--simulate", "--session", str(session.path)]), 0)

    def test_initial_check_exception_stops_arm_before_close(self):
        session = self.session()
        backend = FakeBackend()
        class FailingRunner(FakeRunner):
            fail_initial_check = True
        with patch.object(cli, "plan_for", return_value=[self.item]), patch.object(cli, "dry_report", return_value={}), \
                self.fake_devices(backend, FailingRunner):
            with self.assertRaisesRegex(RuntimeError, "initial CHECK"):
                cli.main("contact", ["--simulate", "--session", str(session.path)])
        self.assertIn("stop_arm", backend.calls)
        self.assertLess(backend.calls.index("stop_arm"), backend.calls.index("close"))

    def test_heavy_block_starting_with_skip_still_prompts_and_checks_before_trial(self):
        session = self.session()
        # This models an earlier operator confirmation, not a completed row.
        session.progress["last_bottle"] = "light"
        session.save_progress()
        plan = [dict(self.item, bottle="heavy"),
                dict(self.item, trial_id="trial_0002", bottle="heavy", valid=True, safety_reasons=[])]
        backend = FakeBackend()
        test = self

        class ConfirmingRunner(FakeRunner):
            def ask(self, text):
                test.assertIn("heavy", text)
                self.backend.calls.append("ask:heavy")
                return ""

            def check_bottle(self, reason):
                # Neither the skipped heavy row nor the prompt is sufficient
                # to change the durable record of the installed bottle.
                test.assertEqual(Session(self.session.path).progress["last_bottle"], "light")
                super().check_bottle(reason)

            def contact_trial(self, item, **kwargs):
                test.assertEqual(Session(self.session.path).progress["last_bottle"], "heavy")
                self.backend.calls.append("contact:heavy")
                self.session.start_trial(item["trial_id"])
                meta = {"trial_id": item["trial_id"], "bottle": item["bottle"], "task": "contact", "status": "completed"}
                self.session.complete(meta)
                return meta

        with patch.object(cli, "plan_for", return_value=plan), patch.object(cli, "dry_report", return_value={}), \
                self.fake_devices(backend, ConfirmingRunner), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(cli.main("contact", ["--simulate", "--session", str(session.path), "--limit", "1"]), 0)
        self.assertEqual(backend.calls, ["check:run_start", "ask:heavy", "check:bottle_switch", "contact:heavy", "close"])
        refreshed = Session(session.path)
        self.assertEqual(refreshed.progress["last_bottle"], "heavy")
        self.assertEqual(refreshed.progress["completed"], ["trial_0001", "trial_0002"])

    def test_failed_heavy_check_does_not_claim_bottle_changed(self):
        session = self.session()
        session.progress["last_bottle"] = "light"
        session.save_progress()
        plan = [dict(self.item, bottle="heavy"),
                dict(self.item, trial_id="trial_0002", bottle="heavy", valid=True, safety_reasons=[])]
        backend = FakeBackend()

        class RejectingRunner(FakeRunner):
            def ask(self, text):
                self.backend.calls.append("ask:heavy")
                return ""

            def check_bottle(self, reason):
                super().check_bottle(reason)
                if reason == "bottle_switch":
                    raise RuntimeError("heavy bottle position check failed")

            def contact_trial(self, item, **kwargs):
                raise AssertionError("A failed bottle check must not start a trial")

        with patch.object(cli, "plan_for", return_value=plan), patch.object(cli, "dry_report", return_value={}), \
                self.fake_devices(backend, RejectingRunner):
            with self.assertRaisesRegex(RuntimeError, "heavy bottle position"):
                cli.main("contact", ["--simulate", "--session", str(session.path)])
        refreshed = Session(session.path)
        self.assertEqual(refreshed.progress["last_bottle"], "light")
        self.assertEqual(refreshed.progress["completed"], ["trial_0001"])
        self.assertIn("ask:heavy", backend.calls)
        self.assertEqual(backend.calls[-2:], ["stop_arm", "close"])


if __name__ == "__main__":
    unittest.main()
