"""Pure geometry/evidence tests and a real SIGKILL of a ROS-free fake child."""
import copy
import json
from pathlib import Path
import signal
import tempfile
import unittest
from unittest.mock import Mock,patch

import numpy as np

from bottle_scan.config import demo_config
from bottle_scan.stop_validation import (stop_test_defaults,validate_stop_settings,stop_path,
    recheck_confirmed_path,stable_window,read_intent,_write_intent,verify_last_target,
    software_stop,stop_report_binding,gate_stop_report,run_stop_test,summarize_trial,_child_real)


def config():
    cfg=demo_config();cfg["stop_test"]=stop_test_defaults(demo=True)
    return cfg


class StopValidationTests(unittest.TestCase):
    def setUp(self):self.cfg=config()

    def test_nominal_50mm_free_path_and_width_are_checked(self):
        start=np.asarray(self.cfg["preparation"]["axis_test_pose"])
        path=stop_path(self.cfg,start,self.cfg["gripper"]["open_mm"])
        np.testing.assert_array_equal(np.asarray(path["target_tcp_mm"])[:3,3]-start[:3,3],[50,0,0])
        self.assertEqual(path["speed_mm_s"],self.cfg["motion"]["free_speed_mm_s"])
        with self.assertRaisesRegex(ValueError,"w_open"):stop_path(self.cfg,start,20.)
        far=start.copy();far[0,3]+=3
        with self.assertRaisesRegex(ValueError,"axis_test_pose"):stop_path(self.cfg,far,90.)

    def test_collision_rejected_without_starting_child(self):
        self.cfg["safety"]["workspace_max_mm"][0]=220.
        with self.assertRaisesRegex(ValueError,"path rejected"):
            stop_path(self.cfg,self.cfg["preparation"]["axis_test_pose"],90.)

    def test_recheck_allows_small_measurement_noise_without_changing_target(self):
        start=np.asarray(self.cfg["preparation"]["axis_test_pose"])
        path=stop_path(self.cfg,start,90.)
        measured=start.copy();measured[0,3]+=.01
        updated=recheck_confirmed_path(self.cfg,path,measured,90.)
        self.assertEqual(updated["target_tcp_mm"],path["target_tcp_mm"])
        measured[0,3]+=2
        with self.assertRaises(ValueError):recheck_confirmed_path(self.cfg,path,measured,90.)

    def test_unmeasured_parameters_and_nonmiddle_trigger_reject(self):
        self.cfg["stop_test"]=stop_test_defaults(demo=False)
        self.assertTrue(validate_stop_settings(self.cfg))
        self.cfg=config();self.cfg["stop_test"]["trigger_s"]=3.
        self.assertIn("precede",str(validate_stop_settings(self.cfg)))
        self.cfg=config();self.cfg["stop_test"]["repeats"]=2
        self.assertIn("at least 3",str(validate_stop_settings(self.cfg)))

    def test_shared_cancel_path_calls_only_controller_stop(self):
        arm=Mock()
        result=software_stop(arm)
        arm.stop_arm.assert_called_once_with()
        self.assertEqual(arm.mock_calls,[unittest.mock.call.stop_arm()])
        self.assertGreater(result["request_monotonic_ns"],0)

    def test_interrupted_intent_and_unreceived_final_frame_are_invalid(self):
        shared=[0.]*16;wire=[10.,20.,2.,.2,0.,.15,0.,0.,0.,1.]
        _write_intent(shared,wire)
        intent=read_intent(shared)
        self.assertEqual(intent["wire_values"],wire)
        self.assertTrue(verify_last_target(intent,[{"wire_values":wire}]))
        next_wire=wire.copy();next_wire[1]+=1
        _write_intent(shared,next_wire)
        self.assertFalse(verify_last_target(read_intent(shared),[{"wire_values":wire}]))
        shared[0]+=1
        self.assertIsNone(read_intent(shared))

    def test_reordered_or_duplicate_source_commands_are_not_verified(self):
        wire=[10.,20.,2.,.2,0.,.15,0.,0.,0.,1.]
        intent={"wire_values":wire}
        self.assertFalse(verify_last_target(intent,[{"wire_values":wire},{"wire_values":wire}]))

    def test_settling_requires_actual_independent_timed_measurements(self):
        pose=np.eye(4).tolist();options=self.cfg["stop_test"]
        samples=[{"monotonic_ns":1_000_000_000+i*10_000_000,"tcp_mm":pose} for i in range(30)]
        self.assertIsNotNone(stable_window(samples,1_000_000_000,options))
        repeated=[{"monotonic_ns":1_000_000_000,"tcp_mm":pose}]*100
        self.assertIsNone(stable_window(repeated,1_000_000_000,options))
        for i,row in enumerate(samples):
            row["tcp_mm"]=np.eye(4).tolist();row["tcp_mm"][0][3]=i*.5
        self.assertIsNone(stable_window(samples,1_000_000_000,options))

    def test_binding_reuses_new_t0_but_changes_for_hardware_speed_direction(self):
        original=stop_report_binding(self.cfg)
        self.cfg["T0"][0][3]+=1
        self.cfg["stop_test"]["report_path"]="somewhere_else.json"
        self.assertEqual(original,stop_report_binding(self.cfg))
        self.cfg["motion"]["free_speed_mm_s"]+=1
        self.assertNotEqual(original,stop_report_binding(self.cfg))

    def child_fixture(self):
        cfg=self.cfg
        cfg["hardware"]["public_ros"]={"arm_publish_hz":50.,"arm_command_step_mm":2.,"arm_command_step_deg":2.}
        cfg["hardware"]["gripper_width_calibration"]=[{"normalized":.3,"width_mm":20.},{"normalized":1.,"width_mm":90.}]
        arm=Mock();arm._publisher=Mock();arm.snapshot.return_value={"arms":{"right":{
            "T_base_tcp_mm":cfg["preparation"]["axis_test_pose"]}}}
        grip=Mock();grip.snapshot.return_value={"fresh":True,"valid":True,"width_mm":90.}
        path=stop_path(cfg,cfg["preparation"]["axis_test_pose"],90.)
        pipe=Mock();pipe.poll.return_value=True
        pipe.recv.side_effect=[{"kind":"start","path":path},{"kind":"stop"},{"kind":"close"}]
        return arm,grip,pipe

    def test_real_child_plumbing_uses_fake_objects_and_only_explicit_start_stop(self):
        arm,grip,pipe=self.child_fixture()
        with patch("bottle_scan.hardware_arm.RosArmController",return_value=arm), \
             patch("bottle_scan.hardware_gripper.RosNormalizedGripperObserver",return_value=grip), \
             patch("bottle_scan.command_ownership.guard_all_commands") as guard, \
             patch("bottle_scan.command_ownership.current_ros_node_names",return_value=("/fake",)), \
             patch("move_tcp.SampleDrain"),patch("bottle_scan.stop_validation.signal.signal"):
            _child_real(pipe,[0.]*16,self.cfg)
        self.assertEqual(guard.call_count,2)
        arm.move_linear.assert_called_once()
        arm.stop_arm.assert_called_once()
        self.assertEqual([call.args[0]["event"] for call in pipe.send.call_args_list],["ready","started","stopped"])
        grip.close.assert_called_once()

    def test_late_foreign_publisher_rejects_without_a_spurious_hold(self):
        arm,grip,pipe=self.child_fixture()
        with patch("bottle_scan.hardware_arm.RosArmController",return_value=arm), \
             patch("bottle_scan.hardware_gripper.RosNormalizedGripperObserver",return_value=grip), \
             patch("bottle_scan.command_ownership.guard_all_commands",side_effect=[{},RuntimeError("foreign publisher")]), \
             patch("bottle_scan.command_ownership.current_ros_node_names",return_value=("/fake",)), \
             patch("move_tcp.SampleDrain"),patch("bottle_scan.stop_validation.signal.signal"):
            _child_real(pipe,[0.]*16,self.cfg)
        arm.move_linear.assert_not_called()
        arm.stop_arm.assert_not_called()
        self.assertEqual(pipe.send.call_args_list[-1].args[0]["event"],"error")


class StopProcessSimulationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory=tempfile.TemporaryDirectory()
        cls.cfg=config();cls.path=Path(cls.directory.name)/"stop.json"
        cls.report=run_stop_test(cls.cfg,simulate=True,output_path=cls.path,
                                confirm=lambda _:(_ for _ in ()).throw(AssertionError("Simulation must not prompt")),output=lambda _:None)

    @classmethod
    def tearDownClass(cls):cls.directory.cleanup()

    def test_three_real_child_sigkills_and_three_software_stops(self):
        self.assertEqual(len(self.report["trials"]),6)
        self.assertTrue(self.report["passed"])
        self.assertFalse(self.report["operator_confirmed"])
        self.assertTrue(all(x["last_published_target_verified"] for x in self.report["trials"]))
        kills=[x for x in self.report["trials"] if x["mode"]=="sigkill"]
        software=[x for x in self.report["trials"] if x["mode"]=="software"]
        self.assertTrue(all(x["software_stop_acknowledged"] and x["stop_dispatch_latency_s"]>=0 for x in software))
        self.assertTrue(all(x["controller_stop_call_monotonic_ns"]>=x["stop_request_monotonic_ns"] for x in software))
        self.assertEqual([x["child_exitcode"] for x in kills],[-signal.SIGKILL]*3)
        self.assertTrue(all(x["max_extra_displacement_mm"]>.1 for x in kills))
        self.assertTrue(all(x["final_to_last_target_error_mm_deg"][0]<.05 for x in kills))
        for trial in kills:
            # No parent recovery command was introduced after the kill time.
            self.assertTrue(all(row["source_ros_ns"]<=trial["stop_request_monotonic_ns"] for row in trial["commands"]))

    def test_synthetic_report_never_passes_real_gate_even_with_manual_top_flag(self):
        with self.assertRaisesRegex(ValueError,"real PASS"):gate_stop_report(self.cfg,self.path)
        changed=copy.deepcopy(self.report)
        changed.update(simulated=False,operator_confirmed=True)
        other=Path(self.directory.name)/"edited.json";other.write_text(json.dumps(changed))
        with self.assertRaisesRegex(ValueError,"passing repeats"):gate_stop_report(self.cfg,other)

    def test_measured_overtravel_limit_is_enforced_from_trace(self):
        cfg=copy.deepcopy(self.cfg);cfg["stop_test"]["max_extra_mm"]=.001
        trial=next(row for row in self.report["trials"] if row["mode"]=="sigkill")
        summary=summarize_trial(cfg,trial)
        self.assertTrue(summary["valid"])
        self.assertFalse(summary["passed"])

    def test_unverified_last_frame_does_not_become_nominal_endpoint(self):
        trial=copy.deepcopy(self.report["trials"][-1]);trial["last_publish_intent"]["wire_values"][1]+=1
        result=summarize_trial(self.cfg,trial)
        self.assertFalse(result["valid"])
        self.assertFalse(result["last_published_target_verified"])
        self.assertNotEqual(result["last_observed_target_tcp_mm"],trial["path"]["target_tcp_mm"])

    def test_no_stop_ack_is_not_a_successful_stop_even_when_measured_pose_is_stable(self):
        trial=copy.deepcopy(self.report["trials"][0]);trial["software_stop_acknowledged"]=False
        summary=summarize_trial(self.cfg,trial)
        self.assertFalse(summary["valid"])
        self.assertIn("not acknowledged",summary["invalid_reason"])

    def test_refuses_overwriting_previous_report(self):
        with self.assertRaises(FileExistsError):run_stop_test(self.cfg,simulate=True,output_path=self.path)


if __name__=="__main__":unittest.main()
