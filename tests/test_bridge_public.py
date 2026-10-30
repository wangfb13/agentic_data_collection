"""Public bridge integration with fake components only; no ROS/device connect."""
import copy
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

import h5py
import numpy as np

from bottle_scan.backend import Sample, load_hardware
from bottle_scan.config import demo_config
from bottle_scan.recording import H5Writer
from bridge_ros_public import PublicRosBackend, CONTROLLER_STATE_CODES
from bottle_scan.command_ownership import inspect_all_commands, REQUIRED_COMMAND_TOPICS


def config():
    cfg = demo_config()
    cfg["demo"] = False
    cfg["hardware"].update(profile="public_ros", backend="public_ros", bridge_module="bridge_ros_public",
        arm_side="right", sensor_serials={"left":"L", "right":"R"},
        gripper_width_calibration=[{"normalized":.3,"width_mm":20},{"normalized":1.,"width_mm":90}],
        public_ros={"software_control_acknowledged":True,"processed_dd_accepted":True,
            "raw_normalized_min":.3,"force_deadband_N":.1,"queue_limit":50,
            "sensors":{"serials":{"left":"L","right":"R"}}})
    cfg["tactile"].update(dd_shape=[96,128,3],dd_dtype="float32")
    return cfg


class FakeComponent:
    def __init__(self, snapshot):
        self.value = snapshot
        self.events = []
        self.calls = []
        self.drain_error = None

    def snapshot(self):
        return self.value

    def drain_samples(self):
        if self.drain_error:
            raise RuntimeError(self.drain_error)
        events, self.events = self.events, []
        return events

    def set_context(self, **kwargs):
        self.calls.append(("context",kwargs))

    def connect(self):
        self.calls.append(("connect",))

    def close(self):
        self.calls.append(("close",))


class FakeGripper(FakeComponent):
    def set_force_target(self, target):
        self.calls.append(("force",target))
        self.value.update(force_target_N=target,controller_state="force")

    def hold_gripper(self):
        self.calls.append(("hold",))
        self.value.update(force_target_N=0.,controller_state="hold")

    def exit_force_mode(self):
        self.hold_gripper()

    def command_open(self, width, speed):
        self.calls.append(("open",width,speed))
        self.value.update(force_target_N=0.,controller_state="open")

    def command_close(self, width, speed):
        self.calls.append(("close_target",width,speed))
        self.value.update(force_target_N=0.,controller_state="close")


class BridgeTests(unittest.TestCase):
    def setUp(self):
        self.now = 10_000_000_000
        dd = [np.zeros((96,128,3),np.float32),np.ones((96,128,3),np.float32)]
        self.sensors = FakeComponent({"timestamp_ns":self.now-40,
            "source_timestamp_ns":{"dd_left":self.now-30,"dd_right":self.now-40,
                                   "wrench_left":self.now-20,"wrench_right":self.now-10},
            "source_sequences":{"dd_left":10,"dd_right":11,"wrench_left":12,"wrench_right":15},
            "normal_N":np.array([1.,2.]),"wrench":np.zeros((2,6)),"dd":dd})
        self.arm = FakeComponent({"arms":{"right":{"T_base_tcp_mm":np.eye(4),
                                                      "received_monotonic_ns":self.now-50}}})
        self.gripper = FakeGripper({"fresh":True,"valid":True,"velocity_available":True,
            "monotonic_ns":self.now-60,"width_mm":50.,"velocity_mm_s":0.,"current_available":False,
            "force_target_N":0.,"controller_state":"idle","fault":None})
        def ownership(cfg, own_nodes=()):
            return inspect_all_commands({"hardware":{"public_ros":{"command_ownership":{
                "topics":list(REQUIRED_COMMAND_TOPICS),"timeout_s":1.}}}}, own_nodes,
                system_state_getter=lambda:[[],[],[]])
        self.backend = PublicRosBackend(config(),arm=self.arm,gripper=self.gripper,sensors=self.sensors,connect=False,
                                        ownership_guard=ownership)
        self.backend.monotonic_ns = lambda: self.now
        # Most tests exercise already-initialized acquisition. Initialization
        # and its no-motion guard are tested separately below.
        self.backend._zero_ready = self.backend._ever_zero_ready = True

    def tearDown(self):
        self.backend.close()

    def test_keyframe_samples_associate_both_arms_at_actual_camera_source_stamps(self):
        self.arm.value["arms"]["left"]=copy.deepcopy(self.arm.value["arms"]["right"])
        self.backend.bind_session(types.SimpleNamespace(cfg=self.backend.cfg,
            meta={"B":np.eye(4).tolist(),"camera_roles":{"left_wrist":"fixed_external_view_on_parked_arm","right_wrist":"eye_in_hand"}}))
        samples={name:Sample("camera/"+name,np.zeros((8,9,3),np.uint8),self.now-offset,
                            metadata={"timestamp_kind":"host_receive_monotonic_only"})
                 for name,offset in (("left_wrist",100),("right_wrist",200))}
        self.sensors.image_samples=Mock(return_value=samples)
        result=self.backend.image_samples()
        self.assertEqual(result["left_wrist"].monotonic_ns,self.now-100)
        self.assertEqual(result["right_wrist"].metadata["image_host_monotonic_ns"],self.now-200)
        self.assertIn("T_base_leftTCP",result["right_wrist"].metadata)
        self.assertIn("T_base_rightTCP",result["right_wrist"].metadata)
        self.assertEqual(result["right_wrist"].metadata["B_base_mm"],np.eye(4).tolist())
        self.assertEqual(self.gripper.calls,[])

    def test_constructor_has_zero_actions_and_snapshot_uses_oldest_actual_source(self):
        for component in (self.arm,self.gripper,self.sensors):
            self.assertEqual(component.calls,[])
        result = self.backend.snapshot()
        self.assertIsNone(result["fault"])
        self.assertEqual(result["timestamp_ns"],self.now-60)
        self.assertTrue(np.isnan(result["current_A"]))
        self.assertIs(result["current_A_available"],False)
        self.assertIn("finite_difference",result["velocity_source"])
        np.testing.assert_array_equal(result["normal_N"],[1,2])
        self.assertEqual(self.backend.drain_samples(),[])
        self.backend.snapshot()
        self.assertEqual(self.backend.drain_samples(),[])  # Polling creates no frames.

    def test_uninitialized_connection_refuses_every_normal_actuator_entry(self):
        self.backend._zero_ready = self.backend._ever_zero_ready = False
        self.assertIn("zero_not_ready",self.backend.snapshot()["fault"])
        actions=[lambda:self.backend.move_linear(np.eye(4),1,1),
                 lambda:self.backend.command_open(60,1),lambda:self.backend.command_close(30,1),
                 lambda:self.backend.set_force_target(1),self.backend.hold_gripper,self.backend.exit_force_mode,
                 lambda:self.backend.emergency_open_pre_lift(60,1)]
        for action in actions:
            with self.assertRaises(RuntimeError):action()
        for component in (self.arm,self.gripper,self.sensors):self.assertEqual(component.calls,[])

    def test_readonly_observation_has_pose_width_and_raw_force_without_zero(self):
        self.backend._zero_ready = False
        raw={**self.sensors.value,"wrench_raw":np.ones((2,6)),"wrench_unit":"unknown_sdk_raw"}
        self.sensors.observation_snapshot=lambda:raw
        value=self.backend.observation_snapshot()
        self.assertIsNone(value["fault"])
        self.assertFalse(value["zero_ready"])
        self.assertNotIn("normal_N",value)
        self.assertNotIn("wrench",value)
        self.assertEqual(value["wrench_unit"],"unknown_sdk_raw")
        for component in (self.arm,self.gripper,self.sensors):self.assertEqual(component.calls,[])

    def test_zero_failure_records_attempt_and_never_leaves_motion_enabled(self):
        from bottle_scan.session import Session
        with tempfile.TemporaryDirectory() as directory:
            session=Session.create(Path(directory)/"session",config(),np.eye(4),simulated=True)
            self.sensors.last_zero_attempt=None
            def reject(**kwargs):
                self.assertIsNone(kwargs["reference"])
                raise RuntimeError("contact imprint")
            self.sensors.zero_unloaded=reject
            with self.assertRaisesRegex(RuntimeError,"contact imprint"):
                self.backend.prepare_zero(session,confirm=lambda _:"EMPTY")
            self.assertFalse(self.backend._zero_ready)
            self.assertEqual(session.meta["zero_runs"][-1]["status"],"failed")
            for component in (self.arm,self.gripper,self.sensors):self.assertEqual(component.calls,[])

    def test_zero_uses_session_reference_and_saves_each_run_with_previous_report(self):
        from bottle_scan.session import Session
        with tempfile.TemporaryDirectory() as directory:
            session=Session.create(Path(directory)/"session",config(),np.eye(4),simulated=True)
            reference_path=session.path/"references"/"mean.npz"
            reference={side+"_dd":np.zeros((96,128,3)) for side in ("left","right")}
            np.savez(reference_path,**reference)
            session.meta["references"].append({"path":"references/mean.npz"})
            calls=[]
            def zero(**kwargs):
                calls.append(kwargs)
                for side in ("left","right"):
                    np.testing.assert_array_equal(kwargs["reference"][side+"_dd"],reference[side+"_dd"])
                return {"status":"ready","fingers":{},"warnings":[]}
            self.sensors.zero_unloaded=zero
            self.backend.prepare_zero(session)
            self.backend.prepare_zero(session)
            self.assertIsNone(calls[0]["previous"])
            self.assertEqual(calls[1]["previous"],session.meta["zero_runs"][0])
            self.assertEqual(len(session.meta["zero_runs"]),2)
            self.assertEqual(len(session.meta["zero_runs"][0]["unloaded_reference"]["sha256"]),64)
            self.assertTrue(self.backend._zero_ready)
            saved=json.loads((session.path/"session_meta.json").read_text())
            self.assertEqual(saved["zero_runs"],session.meta["zero_runs"])

    def test_one_future_or_stale_source_cannot_be_hidden_by_min_timestamp(self):
        for stamp in (self.now+1,self.now-251_000_000):
            self.sensors.value["source_timestamp_ns"]["wrench_right"] = stamp
            result = self.backend.snapshot()
            self.assertIn("wrench_right",result["fault"])
        self.sensors.value["source_timestamp_ns"]["wrench_right"] = self.now
        self.gripper.value["velocity_available"] = False
        self.assertIn("at least two",self.backend.snapshot()["fault"])

    def test_force_reader_keeps_independent_finger_ids_and_receive_times(self):
        force = self.backend._force_reader()
        self.assertEqual(force["source_sequence"],[12,15])
        self.assertEqual(force["monotonic_ns"],[self.now-20,self.now-10])
        np.testing.assert_array_equal(force["normal_N"],[1,2])
        self.assertEqual(self.backend._force_reader()["source_sequence"],[12,15])

    def event(self, stream, stamp=None, data=None, metadata=None):
        return Sample(stream,np.array([1,2],np.uint16) if data is None else data,
                      self.now if stamp is None else stamp,123.25,metadata or {"fid":23})

    def test_idle_drain_and_begin_end_never_claim_idle_data_as_trial(self):
        self.sensors.events = [self.event("dd_encoded_left")]
        self.backend._dispatch_once()
        self.assertEqual(self.backend._idle_discarded,1)
        self.sensors.events = [self.event("dd_encoded_right")]
        self.backend.begin_acquisition()
        self.assertEqual(self.backend._idle_discarded,2)
        self.assertEqual(self.backend.drain_samples(),[])
        active = self.event("gripper",data=np.array([50.,0.,np.nan]))
        self.gripper.events = [active]
        self.backend.end_acquisition()
        samples = self.backend.drain_samples()
        self.assertEqual(len(samples),1)
        self.assertIs(samples[0].data,active.data)
        self.sensors.events = [self.event("dd_encoded_left")]
        self.backend._dispatch_once()
        self.assertEqual(self.backend._idle_discarded,3)
        self.assertEqual(self.backend.drain_samples(),[])

    def test_source_payload_timestamp_and_stage_history_are_preserved(self):
        self.backend.begin_acquisition()
        self.backend.set_trial({"trial_id":"trial_0003"})
        self.backend.set_stage("PROBE")
        first = self.event("dd_encoded_left",metadata={"fid":91,"payload_kind":"encoded_uint16"})
        self.now += 1_000_000
        self.backend.set_stage("SQUEEZE")
        second = self.event("dd_encoded_right",metadata={"fid":92})
        self.sensors.events = [second,first]
        self.backend._dispatch_once()
        samples = self.backend.drain_samples()
        self.assertEqual([x.metadata["stage"] for x in samples],["PROBE","SQUEEZE"])
        self.assertEqual([x.metadata["fid"] for x in samples],[91,92])
        self.assertEqual([x.metadata["trial_id"] for x in samples],["trial_0003"]*2)
        self.assertEqual(samples[0].monotonic_ns,first.monotonic_ns)
        self.assertEqual(samples[0].device_timestamp,123.25)
        self.assertIs(samples[0].data,first.data)
        self.assertEqual(samples[0].data.dtype,np.uint16)
        self.assertEqual(first.metadata,{"fid":91,"payload_kind":"encoded_uint16"})
        self.backend._dispatch_once()
        self.assertEqual(self.backend.drain_samples(),[])

    def test_control_state_is_numeric_fixed_dtype_and_h5_writable_with_current_nan(self):
        self.backend.begin_acquisition()
        self.backend.set_trial({"trial_id":"trial_0001"})
        self.backend.set_stage("SQUEEZE")
        self.backend.set_force_target(2.)
        self.now += 1
        self.backend.hold_gripper()
        source = self.event("gripper",data=np.array([50.,0.,np.nan],np.float64),
                            metadata={"current_available":False,"velocity_kind":"finite_difference"})
        self.gripper.events = [source]
        self.backend._dispatch_once()
        samples = self.backend.drain_samples()
        codes = [s for s in samples if s.stream=="controller_state"]
        self.assertEqual([int(s.data) for s in codes],
                         [CONTROLLER_STATE_CODES["force"],CONTROLLER_STATE_CODES["hold"]])
        self.assertTrue(all(s.data.dtype==np.int16 for s in codes))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"streams.h5"
            with H5Writer(path) as writer:
                for sample in samples:
                    writer.append(sample.stream,sample.data,sample.monotonic_ns,sample.device_timestamp,sample.metadata)
            with h5py.File(path,"r") as h5:
                self.assertEqual(h5["streams/controller_state/data"].dtype,np.int16)
                self.assertEqual(h5["streams/controller_state"].attrs["committed_count"],2)
                self.assertTrue(np.isnan(h5["streams/gripper/data"][0,2]))
                metadata = json.loads(h5["streams/gripper/metadata_json"][0])
                self.assertIs(metadata["current_available"],False)

    def test_queue_overflow_retains_complete_final_batch_then_surfaces_fault(self):
        self.backend._queue_limit = 2
        self.backend.begin_acquisition()
        originals = [self.event("dd_left",self.now+i,np.array([i],np.float32)) for i in range(4)]
        self.sensors.events = originals.copy()
        self.backend._dispatch_once()
        self.assertIn("overflow",self.backend.snapshot()["fault"])
        final = self.backend.drain_samples()
        self.assertEqual(len(final),4)
        self.assertEqual([int(x.data[0]) for x in final],[0,1,2,3])
        self.sensors.events = [self.event("dd_left")]
        self.backend._dispatch_once()
        self.assertEqual(len(self.sensors.events),1)  # No endless over-capacity drains.
        with self.assertRaisesRegex(RuntimeError,"overflow"):
            self.backend.drain_samples()

    def test_component_failure_retains_events_already_drained_from_earlier_component(self):
        self.backend.begin_acquisition()
        self.sensors.events = [self.event("dd_left")]
        self.arm.drain_error = "fake_source_failure"
        self.backend._dispatch_once()
        self.assertEqual(len(self.backend.drain_samples()),1)
        with self.assertRaisesRegex(RuntimeError,"fake_source_failure"):
            self.backend.drain_samples()

    def test_strict_profile_never_instantiates_public_provider(self):
        cfg = config()
        cfg["hardware"]["profile"] = "strict"
        import bridge_ros_public
        with patch.object(bridge_ros_public,"create") as create:
            with self.assertRaisesRegex(ValueError,"strict hardware contract"):
                load_hardware(cfg)
        create.assert_not_called()


if __name__ == "__main__":
    unittest.main()
