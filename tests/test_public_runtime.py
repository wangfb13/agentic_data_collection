"""Offline regression coverage for the explicitly selected public ROS runtime."""
import tempfile
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import h5py
import numpy as np

from bottle_scan.backend import Sample, SimBackend
from bottle_scan.config import demo_config
from bottle_scan.runtime import Acquisition, Runner, SafetyFault


class PublicRuntimeHealthTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.cfg = demo_config()
        self.cfg["hardware"]["profile"] = "public_ros"
        self.session = SimpleNamespace(cfg=self.cfg, T0=np.array(self.cfg["T0"]),
            path=Path(self.directory.name),meta={"references":[]},
            progress={"requires_operator_recovery":False})
        self.backend = SimBackend(self.cfg,self.session.T0)
        self.runner = Runner(self.session,self.backend,single_step=False,unattended=True)

    def test_only_explicit_missing_public_current_is_allowed(self):
        original = self.backend.snapshot()
        allowed = dict(original,current_A=float("nan"),current_A_available=False)
        with patch.object(self.backend,"snapshot",return_value=allowed):
            self.assertIs(self.runner._healthy_snapshot(),allowed)
        for marker in (True,None,"false",0):
            invalid = dict(allowed,current_A_available=marker)
            with self.subTest(marker=marker),patch.object(self.backend,"snapshot",return_value=invalid):
                with self.assertRaisesRegex(SafetyFault,"current_A"):
                    self.runner._healthy_snapshot()
        missing_marker = dict(allowed)
        del missing_marker["current_A_available"]
        with patch.object(self.backend,"snapshot",return_value=missing_marker):
            with self.assertRaisesRegex(SafetyFault,"current_A"):
                self.runner._healthy_snapshot()

    def test_missing_current_exception_does_not_hide_bad_width_velocity_or_infinity(self):
        original = dict(self.backend.snapshot(),current_A=float("nan"),current_A_available=False)
        for field in ("width_mm","velocity_mm_s"):
            for value in (float("nan"),float("inf")):
                invalid = dict(original,**{field:value})
                with self.subTest(field=field,value=value),patch.object(self.backend,"snapshot",return_value=invalid):
                    with self.assertRaisesRegex(SafetyFault,field):
                        self.runner._healthy_snapshot()
        invalid = dict(original,current_A=float("inf"))
        with patch.object(self.backend,"snapshot",return_value=invalid):
            with self.assertRaisesRegex(SafetyFault,"current_A"):
                self.runner._healthy_snapshot()

    def test_strict_profile_still_rejects_missing_current(self):
        self.cfg["hardware"]["profile"] = "strict"
        snapshot = dict(self.backend.snapshot(),current_A=float("nan"),current_A_available=False)
        with patch.object(self.backend,"snapshot",return_value=snapshot):
            with self.assertRaisesRegex(SafetyFault,"current_A"):
                self.runner._healthy_snapshot()


class FakeRecoveryBackend:
    simulated = False

    def __init__(self):
        self.now = 10_000_000_000
        self.calls = []
        self.open_width = 90.
        self.feedback_patch = {}

    def monotonic_ns(self):
        return self.now

    def sleep(self, seconds):
        self.now += int(seconds*1e9)

    def stop_arm(self):
        self.calls.append("stop_arm")

    def emergency_open_pre_lift(self, width, speed):
        self.calls.append(("emergency_open",width,speed))
        self.open_width = width

    def emergency_gripper_snapshot(self):
        self.calls.append("gripper_only_snapshot")
        return {"gripper_timestamp_ns":self.now,"width_mm":self.open_width,"gripper_fault":False,
                **self.feedback_patch}

    def snapshot(self):
        raise AssertionError("Broken DD/full health snapshot must not gate pre-lift release")

    def exit_force_mode(self):
        raise AssertionError("Public recovery must call its special entry point")

    def command_open(self,*args):
        raise AssertionError("Ordinary opening is fault-latched and must not be called")


class PublicRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        cfg = demo_config()
        cfg["hardware"]["profile"] = "public_ros"
        self.session = SimpleNamespace(cfg=cfg,T0=np.array(cfg["T0"]),path=Path(self.directory.name),
            meta={"references":[]},progress={"requires_operator_recovery":False})
        self.backend = FakeRecoveryBackend()
        self.runner = Runner(self.session,self.backend,single_step=False,unattended=False)
        self.sleep_patch = patch("bottle_scan.runtime.time.sleep",side_effect=self.backend.sleep)
        self.sleep_patch.start()
        self.addCleanup(self.sleep_patch.stop)

    def test_already_stopped_pre_lift_release_uses_only_special_methods(self):
        self.backend.stop_arm()  # The caller completed its stop before setting already_stopped.
        result = self.runner.emergency_release_pre_lift(already_stopped=True)
        self.assertEqual(self.backend.calls,["stop_arm",("emergency_open",90.,2.),"gripper_only_snapshot"])
        self.assertEqual(result["width_mm"],90.)

    def test_stop_precedes_special_release_and_failed_stop_prevents_opening(self):
        self.runner.emergency_release_pre_lift()
        self.assertEqual(self.backend.calls[:2],["stop_arm",("emergency_open",90.,2.)])
        self.backend.calls.clear()
        with patch.object(self.backend,"stop_arm",side_effect=OSError("stop failed")):
            with self.assertRaisesRegex(OSError,"stop failed"):
                self.runner.emergency_release_pre_lift()
        self.assertEqual(self.backend.calls,[])

    def test_airborne_or_persisted_recovery_lock_never_calls_special_release(self):
        for airborne,recovery in ((True,False),(False,True),(True,True)):
            self.runner.airborne = airborne
            self.session.progress["requires_operator_recovery"] = recovery
            for stopped in (False,True):
                with self.subTest(airborne=airborne,recovery=recovery,stopped=stopped):
                    with self.assertRaisesRegex(SafetyFault,"forbidden"):
                        self.runner.emergency_release_pre_lift(already_stopped=stopped)
                    self.assertEqual(self.backend.calls,[])

    def test_special_release_still_rejects_stale_or_invalid_gripper_feedback(self):
        for override in ({"gripper_timestamp_ns":self.backend.now-1_000_000_000},
                         {"width_mm":float("nan")},{"gripper_fault":True}):
            self.backend.feedback_patch = override
            with self.subTest(override=override),self.assertRaises(SafetyFault):
                self.runner.emergency_release_pre_lift(already_stopped=True)
        self.assertNotIn("stop_arm",self.backend.calls)


class FakeThread:
    def __init__(self, calls, **kwargs):
        self.calls = calls

    def start(self):
        self.calls.append("thread_start")

    def join(self, timeout):
        self.calls.append("thread_join")

    def is_alive(self):
        return False


class HookBackend:
    simulated = False

    def __init__(self):
        self.calls = []
        self.pending = []
        self.cfg = demo_config()

    def begin_acquisition(self):
        self.calls.append("begin")
        self.pending.append(Sample("sensor",np.array([1.],np.float32),100))

    def end_acquisition(self):
        self.calls.append("end")
        self.pending.append(Sample("sensor",np.array([2.],np.float32),200))

    def drain_samples(self):
        self.calls.append("drain")
        result,self.pending = self.pending,[]
        return result


class PublicAcquisitionTests(unittest.TestCase):
    def test_begin_before_worker_or_first_frame_and_end_before_final_flush(self):
        backend = HookBackend()
        with tempfile.TemporaryDirectory() as directory:
            with patch("bottle_scan.runtime.threading.Thread",side_effect=lambda **kwargs:FakeThread(backend.calls,**kwargs)):
                acquisition = Acquisition(backend,directory)
            self.assertEqual(backend.calls,["begin","thread_start"])
            acquisition.pump()
            acquisition.close()
            self.assertEqual(backend.calls,["begin","thread_start","drain","thread_join","end","drain"])
            with h5py.File(Path(directory)/"streams.h5","r") as h5:
                np.testing.assert_array_equal(h5["streams/sensor/data"][:],[[1.],[2.]])
                np.testing.assert_array_equal(h5["streams/sensor/monotonic_ns"][:],[100,200])
                self.assertEqual(h5["streams/sensor"].attrs["committed_count"],2)

    def test_simulated_backend_without_hooks_keeps_existing_pump_behavior(self):
        pending = [Sample("synthetic",np.array([7.],np.float32),100)]
        def drain():
            result = pending[:]
            pending.clear()
            return result
        backend = SimpleNamespace(simulated=True,cfg=demo_config(),drain_samples=Mock(side_effect=drain))
        with tempfile.TemporaryDirectory() as directory:
            acquisition = Acquisition(backend,directory)
            self.assertIsNone(acquisition.thread)
            acquisition.pump()
            acquisition.close()
            self.assertEqual(backend.drain_samples.call_count,2)
            with h5py.File(Path(directory)/"streams.h5","r") as h5:
                self.assertEqual(h5["streams/synthetic"].attrs["committed_count"],1)
                self.assertEqual(float(h5["streams/synthetic/data"][0,0]),7.)

    def test_source_drain_failure_cannot_leave_successful_manifest(self):
        backend = SimpleNamespace(simulated=True,cfg=demo_config(),drain_samples=Mock(side_effect=RuntimeError("source queue overflow")))
        with tempfile.TemporaryDirectory() as directory:
            acquisition=Acquisition(backend,directory)
            with self.assertRaisesRegex(RuntimeError,"source queue overflow"):acquisition.pump()
            with self.assertRaisesRegex(RuntimeError,"source queue overflow"):acquisition.close()
            manifest=json.loads((Path(directory)/"recording_status.json").read_text())
            self.assertFalse(manifest["complete"])
            self.assertIn("source queue overflow",manifest["error"])

    def test_begin_failure_closes_writer_and_marks_incomplete(self):
        backend=SimpleNamespace(simulated=True,cfg=demo_config(),begin_acquisition=Mock(side_effect=RuntimeError("source startup failed")))
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError,"source startup failed"):Acquisition(backend,directory)
            manifest=json.loads((Path(directory)/"recording_status.json").read_text())
            self.assertFalse(manifest["complete"])

    def test_camera_uses_source_association_without_overwriting_it_at_disk_drain(self):
        source={"T_base_leftTCP":np.eye(4).tolist(),"T_base_rightTCP":np.eye(4).tolist(),
                "B_base_mm":np.eye(4).tolist(),"pose_sources":{"left":{"received_monotonic_ns":100}},
                "image_host_monotonic_ns":100}
        pending=[Sample("camera/left_wrist",np.zeros((8,8,3),np.uint8),100,metadata=source)]
        def drain():
            samples=pending[:];pending.clear();return samples
        backend=SimpleNamespace(simulated=True,cfg=demo_config(),drain_samples=drain,
                                recording_session=object(),snapshot=Mock(side_effect=RuntimeError("newer pose must not replace source association")))
        with tempfile.TemporaryDirectory() as directory:
            acquisition=Acquisition(backend,directory);acquisition.pump();acquisition.close()
            backend.snapshot.assert_not_called()
            import csv
            with (Path(directory)/"images/timestamps.csv").open() as stream:rows=list(csv.DictReader(stream))
            self.assertEqual(json.loads(rows[0]["metadata_json"])["image_host_monotonic_ns"],100)


if __name__ == "__main__":
    unittest.main()
