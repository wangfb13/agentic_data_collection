import copy
import tempfile
from pathlib import Path
import types
import unittest
from unittest.mock import Mock

import numpy as np
from bottle_scan.config import demo_config
from bottle_scan.session import Session
from bottle_scan.park_guard import ParkGuard,ParkGuardError,park_guard_defaults,image_pose_metadata
from bottle_scan.hardware_observer import RosJointObserver


def observation(cfg,now=1_000_000_000):
    arms={side:{"T_base_tcp_mm":np.eye(4).tolist(),"received_monotonic_ns":now,
                 "source_ros_timestamp_s":100.,"base_frame":"base_link","tcp_frame":side+"_gripper_base_link"}
          for side in ("left","right")}
    joints={"arms":{side:{"names":cfg["park_guard"]["joint_names"][side],"position_rad":[.1]*7}
                      for side in ("left","right")},"received_monotonic_ns":now,
            "source_ros_timestamp_s":100.,"source_sequence":99,"topic":"/joint_states","unit":"rad"}
    return {"arms":arms,"joints":joints,"timestamp_ns":now}


class ParkTests(unittest.TestCase):
    def setUp(self):
        self.directory=tempfile.TemporaryDirectory()
        cfg=demo_config();cfg["park_guard"]=park_guard_defaults()
        self.session=Session.create(Path(self.directory.name)/"session",cfg,np.eye(4),simulated=True)
        self.obs=observation(cfg)
    def tearDown(self):self.directory.cleanup()

    def test_capture_keeps_joint_order_source_and_immutable_reference(self):
        guard=ParkGuard.capture(self.session,self.obs,now_ns=1_000_000_000)
        original=copy.deepcopy(self.session.meta["nonexperiment_arm_park"])
        self.assertEqual(original["joints"]["names"],self.session.cfg["park_guard"]["joint_names"]["left"])
        self.assertEqual(original["joint_source"]["source_sequence"],99)
        self.assertTrue(guard.check(self.obs,now_ns=1_000_000_000)["passed"])
        with self.assertRaisesRegex(ParkGuardError,"immutable"):ParkGuard.capture(self.session,self.obs,now_ns=1_000_000_000)
        self.assertEqual(self.session.meta["nonexperiment_arm_park"],original)

    def test_translation_or_rotation_change_stops_without_reference_update(self):
        from scipy.spatial.transform import Rotation
        guard=ParkGuard.capture(self.session,self.obs,now_ns=1_000_000_000)
        original=copy.deepcopy(self.session.meta["nonexperiment_arm_park"])
        for pose in (np.eye(4),np.eye(4)):
            if pose[0,3]==0:pose[0,3]=1.1
            self.obs["arms"]["left"]["T_base_tcp_mm"]=pose.tolist()
            with self.assertRaisesRegex(ParkGuardError,"moved"):guard.check(self.obs,now_ns=1_000_000_000)
        pose=np.eye(4);pose[:3,:3]=Rotation.from_euler("z",.6,degrees=True).as_matrix()
        self.obs["arms"]["left"]["T_base_tcp_mm"]=pose.tolist()
        with self.assertRaises(ParkGuardError):guard.check(self.obs,now_ns=1_000_000_000)
        self.assertEqual(self.session.meta["nonexperiment_arm_park"],original)

    def test_stale_joint_or_pose_cannot_pass_even_at_same_pose(self):
        guard=ParkGuard.capture(self.session,self.obs,now_ns=1_000_000_000)
        with self.assertRaisesRegex(ParkGuardError,"Stale"):guard.check(self.obs,now_ns=2_000_000_000)
        self.obs["arms"]["left"]["received_monotonic_ns"]=2_000_000_000
        with self.assertRaisesRegex(ParkGuardError,"joints"):guard.check(self.obs,now_ns=2_000_000_000)

    def test_every_image_carries_both_measured_poses_B_and_time_skew(self):
        ParkGuard.capture(self.session,self.obs,now_ns=1_000_000_000)
        self.obs["arms"]["right"]["received_monotonic_ns"]+=10_000_000
        meta=image_pose_metadata(self.obs,self.session,camera_stamp_ns=1_000_000_000)
        self.assertEqual(meta["T_base_leftTCP"],np.eye(4).tolist())
        self.assertEqual(meta["T_base_rightTCP"],np.eye(4).tolist())
        self.assertEqual(meta["B_base_mm"],self.session.meta["B"])
        self.assertEqual(meta["camera_roles"]["left_wrist"],"fixed_external_view_on_parked_arm")
        self.assertAlmostEqual(meta["pose_sources"]["right"]["image_to_pose_skew_s"],.01)
        with self.assertRaises(ParkGuardError):image_pose_metadata(self.obs,self.session,camera_stamp_ns=2_000_000_000)


class JointObserverTests(unittest.TestCase):
    def setUp(self):
        self.cfg=demo_config();self.cfg["park_guard"]=park_guard_defaults()
        self.now=1_000_000_000;self.ros=Mock();self.ros.core.is_initialized.return_value=True
        self.ros.Time.now.return_value.to_sec.return_value=100.
        self.observer=RosJointObserver(self.cfg,rospy_module=self.ros,message_type=object,clock=lambda:self.now)
        self.names=sum([self.cfg["park_guard"]["joint_names"][side] for side in ("right","left")],[])
        self.msg=types.SimpleNamespace(name=self.names,position=list(range(14)),
             header=types.SimpleNamespace(stamp=types.SimpleNamespace(to_sec=lambda:100.),seq=123))
    def test_subscribe_only_no_command_publisher_and_exact_name_order(self):
        self.observer.connect();self.observer._receive(self.msg)
        snapshot=self.observer.snapshot()
        self.assertEqual(snapshot["arms"]["left"]["position_rad"],list(range(7,14)))
        self.ros.Publisher.assert_not_called();self.ros.ServiceProxy.assert_not_called()
        self.observer.close();self.ros.Subscriber.return_value.unregister.assert_called_once()
    def test_missing_name_stale_source_and_host_receipt_rejected(self):
        self.observer._receive(self.msg)
        self.now+=300_000_000
        with self.assertRaisesRegex(RuntimeError,"receipt"):self.observer.snapshot()
        self.observer._receive(self.msg);self.ros.Time.now.return_value.to_sec.return_value=101.
        with self.assertRaisesRegex(RuntimeError,"source"):self.observer.snapshot()
        self.ros.Time.now.return_value.to_sec.return_value=100.
        self.msg.name=self.msg.name[:-1];self.msg.position=self.msg.position[:-1]
        with self.assertRaisesRegex(RuntimeError,"missing"):self.observer.snapshot()


if __name__=="__main__":unittest.main()
