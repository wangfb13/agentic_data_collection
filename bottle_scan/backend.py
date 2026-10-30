"""Hardware contract and deterministic offline backend.

No hardware libraries are imported here. Existing ROS wrappers are deliberately
not used as an emergency stop or as a speed-controlled Cartesian planner.
"""
from __future__ import annotations
from dataclasses import dataclass
import importlib
import time
import numpy as np
import cv2
from scipy.spatial.transform import Rotation, Slerp


@dataclass
class Sample:
    stream: str
    data: np.ndarray
    monotonic_ns: int
    device_timestamp: float | None = None
    metadata: dict | None = None


REQUIRED_CAPABILITIES = (
    "safe_connect_no_motion", "bounded_linear_motion", "verified_arm_stop",
    "gripper_mm_calibration", "per_finger_force_N", "raw_sdk_dd",
    "host_monotonic_timestamps", "independent_acquisition", "hold_on_disconnect",
    "force_limit_enforced", "width_limit_enforced", "watchdog_stops_arm_keeps_grip",
    "dynamic_gripper_width_limit",
    "verified_fresh_image_capture",
)

# Explicit user-selected public ROS profile. These are SOFTWARE behavior
# promises, not aliases for verified hardware stopping/path/watchdog guarantees.
PUBLIC_ROS_CAPABILITIES = (
    "safe_connect_no_motion", "software_interpolation", "software_pose_hold",
    "calibrated_gripper_mapping", "software_force_width_limits",
    "continuous_source_recording", "host_monotonic_timestamps",
    "fresh_image_capture", "processed_dd_provenance",
)


def load_hardware(cfg):
    """Called only after real-mode TODO, dry-run and commissioning gates."""
    from .config import unresolved
    missing = unresolved(cfg)
    if missing:
        raise ValueError("Real mode refused: unresolved TODO values: " + str(missing))
    module = cfg["hardware"].get("bridge_module")
    if not module or module.startswith("TODO"):
        raise ValueError("Hardware bridge is unbound; see hardware_bridge_template.py")
    provider = importlib.import_module(module)
    profile = cfg["hardware"].get("profile", "strict")
    if profile not in ("strict", "public_ros"):
        raise ValueError("Unknown hardware profile: " + str(profile))
    if profile == "public_ros" and getattr(provider, "PROFILE", None) != "public_ros":
        raise ValueError("Public ROS profile requires an explicitly matching provider")
    if profile == "strict" and getattr(provider, "PROFILE", "strict") != "strict":
        raise ValueError("Software ROS provider does not satisfy the strict hardware contract")
    capabilities = getattr(provider, "CAPABILITIES", {})
    required = PUBLIC_ROS_CAPABILITIES if profile == "public_ros" else REQUIRED_CAPABILITIES
    absent = [key for key in required if capabilities.get(key) is not True]
    if absent:
        raise ValueError("Unverified hardware capabilities: " + ", ".join(absent))
    if hasattr(provider, "validate_settings"):
        provider.validate_settings(cfg)
    if profile == "public_ros":
        from .command_ownership import guard_all_commands, current_ros_node_names, format_ownership_report
        report = guard_all_commands(cfg, own_nodes=current_ros_node_names())
        print(format_ownership_report(report))
    # create() MUST not move, set motor targets, initialize poses, or start force control.
    return provider.create(cfg)


class SimBackend:
    """Virtual-time plant, explicitly synthetic. Never opens a socket/device."""
    simulated = True

    def __init__(self, cfg, T0):
        self.cfg = cfg
        self.T0 = np.array(T0, dtype=float)
        self.pose = self.T0.copy()
        self.command_pose = self.pose.copy()
        self.width = float(cfg["gripper"]["open_mm"])
        self.width_command = self.width
        self.gripper_min_width = float(cfg["gripper"]["min_mm"])
        self.gripper_speed = 0.
        self.current = 0.
        self.force_target = 0.
        self.force_mode = False
        self.now_ns = time.monotonic_ns()
        self.seq = 0
        self.stage = "IDLE"
        self.phase_start_ns = self.now_ns
        self.scenario = "normal"
        self.nominal_empty = False
        self.bottle_shift_px = [0, 0]
        self.motion = None
        self.calls = []
        self.pending = []
        self.rng = np.random.default_rng(42)
        self.last_camera_ns = 0
        self.snapshot_cache = None
        self.step(0.)

    def monotonic_ns(self):
        return self.now_ns

    def set_stage(self, name):
        self.stage = name
        self.phase_start_ns = self.now_ns
        if name == "LIFT":
            self.pre_lift_z = float(self.pose[2,3])

    def set_trial(self, item, scenario="normal"):
        self.scenario = scenario
        self.offsets = item.get("delta_acl_mm", [0, 0, 0])
        from .geometry import nominal_contact_region
        target = np.asarray(item.get("T_target", self.T0), dtype=float)
        self.nominal_empty = nominal_contact_region(target, self.T0, self.cfg) == "empty"

    def move_linear(self, pose_mm, speed_mm_s, angular_speed_deg_s):
        self.calls.append(("move_linear", np.asarray(pose_mm).copy()))
        target = np.array(pose_mm, dtype=float)
        distance = np.linalg.norm(target[:3, 3] - self.pose[:3, 3])
        angle = np.degrees(Rotation.from_matrix(self.pose[:3, :3].T @ target[:3, :3]).magnitude())
        seconds = max(distance / speed_mm_s, angle / angular_speed_deg_s, .01)
        self.motion = (self.now_ns, self.pose.copy(), target, seconds)
        self.command_pose = target.copy()

    def motion_done(self):
        return self.motion is None

    def stop_arm(self):
        self.calls.append(("stop_arm", None))
        self.motion = None
        self.command_pose = self.pose.copy()

    def command_open(self, width_mm, speed_mm_s):
        self.calls.append(("open", width_mm))
        self.force_mode = False
        self.width_command = float(width_mm)
        self.gripper_speed = abs(speed_mm_s)

    def command_close(self, min_mm, speed_mm_s):
        self.calls.append(("close", min_mm))
        self.force_mode = False
        if min_mm < self.gripper_min_width:
            raise ValueError("Close command below dynamically enforced width floor")
        self.width_command = float(min_mm)
        self.gripper_speed = -abs(speed_mm_s)

    def set_force_target(self, per_finger_N):
        self.calls.append(("force", per_finger_N))
        self.force_mode = True
        from .force_control import target_as_mean, target_definition
        self.force_target = target_as_mean(per_finger_N, target_definition(self.cfg))

    def set_gripper_min_width(self, width_mm):
        if not np.isfinite(width_mm) or not self.cfg["gripper"]["min_mm"] <= width_mm <= self.cfg["gripper"]["open_mm"]:
            raise ValueError("Invalid dynamic gripper width floor")
        self.calls.append(("width_floor", float(width_mm)))
        self.gripper_min_width = float(width_mm)

    def exit_force_mode(self):
        self.calls.append(("exit_force", None))
        self.force_mode = False
        self.force_target = 0.
        self.gripper_speed = 0.
        self.width_command = self.width

    def hold_gripper(self):
        self.calls.append(("hold_gripper", None))
        self.force_mode = False
        self.width_command = self.width
        self.gripper_speed = 0.

    def snapshot(self):
        return self.snapshot_cache

    def control_metrics(self):
        return {name: {"count":0,"mean_s":None,"p95_s":None,"max_s":None,
                "unavailable":"Synthetic virtual-time plant; use record-stress for wall-clock timing"}
            for name in ("arm_interpolation", "force_control")}

    def drain_samples(self):
        pending, self.pending = self.pending, []
        return pending

    def park_observation(self):
        active = self.cfg["hardware"].get("arm_side", "right")
        arms, joints = {}, {}
        for side in ("left", "right"):
            pose = self.pose.copy() if side == active else self.T0.copy()
            if side != active:
                pose[1,3] += 150.
            arms[side] = {"T_base_tcp_mm":pose.tolist(), "received_monotonic_ns":self.now_ns,
                "source_ros_timestamp_s":None,"tcp_frame":"synthetic_"+side,"base_frame":"synthetic_base"}
            joints[side] = {"names":[f"arm_{side}_{i}_joint" for i in range(1,8)],"position_rad":[0.]*7}
        return {"arms":arms,"joints":{"received_monotonic_ns":self.now_ns,"arms":joints,"simulated":True}}

    def images(self):
        # A fixed, textured label permits a unique match; the bottle shifts only
        # during the explicitly synthetic calibration/reset test.
        result = {}
        for i, name in enumerate(("left_wrist", "right_wrist")):
            image = np.full((180, 240, 3), 45, np.uint8)
            x, y = 88 + int(self.bottle_shift_px[0]) + i * 4, 42 + int(self.bottle_shift_px[1])
            cv2.rectangle(image, (x, y), (x + 60, y + 100), (185, 210, 190), -1)
            cv2.putText(image, "RX" + str(i), (x + 4, y + 43), cv2.FONT_HERSHEY_SIMPLEX, .6, (20, 25, 30), 2)
            for k in range(7):
                cv2.line(image, (x + 5, y + 60 + k * 4), (x + 18 + k * 5, y + 60 + k * 4), (40, 80, 40), 1)
            result[name] = image
        return result

    def image_samples(self):
        return {name: Sample("camera/"+name, data, self.now_ns, None,
                    {"simulated":True,"stage":self.stage,"source_sequence":self.seq,
                     "image_timestamp_kind":"synthetic_capture_time"})
                for name,data in self.images().items()}

    def step(self, seconds):
        self.now_ns += int(seconds * 1e9)
        no_contact = self.scenario == "no_contact" or (self.scenario == "normal" and self.nominal_empty)
        if self.motion:
            start_ns, start, end, duration = self.motion
            t = min(1., (self.now_ns - start_ns) / 1e9 / duration)
            self.pose[:3, 3] = start[:3, 3] * (1 - t) + end[:3, 3] * t
            self.pose[:3, :3] = Slerp([0, 1], Rotation.from_matrix(np.stack((start[:3, :3], end[:3, :3]))))([t]).as_matrix()[0]
            if t >= 1:
                self.motion = None
        if not self.force_mode:
            diff = self.width_command - self.width
            step = min(abs(diff), abs(self.gripper_speed) * seconds)
            candidate = self.width + np.sign(diff) * step
            # Synthetic bottle body diameter agrees with demo geometry: 36 mm.
            if diff < 0 and not no_contact and candidate < 36.:
                candidate = 36.
            self.width = candidate
        elif self.force_target > 0:
            self.width = max(self.gripper_min_width, 35.8)
        contact_stage = self.stage in {"CLOSE", "PROBE", "SQUEEZE", "LIFT", "HOLD", "LOWER"}
        contact = contact_stage and self.width <= 36.1 and not no_contact and self.scenario != "stall"
        normal = np.zeros(2)
        if contact:
            level = self.force_target if self.force_mode else .7
            if self.scenario == "force_timeout" and self.stage == "SQUEEZE":
                level *= .4
            normal[:] = level
        if self.scenario == "dropped" and self.stage == "HOLD" and self.now_ns - self.phase_start_ns > 400_000_000:
            normal[0] = 0.
        if self.scenario == "single_contact":
            normal[1] = 0.
        wrench = np.zeros((2, 6))
        wrench[:, 2] = normal
        loaded = self.stage == "HOLD" or (self.stage in ("LIFT", "LOWER") and
            self.pose[2,3] > getattr(self, "pre_lift_z", self.T0[2,3]) + 1e-6)
        if loaded and np.all(normal > .1):
            wrench[:, 1] = .4
        wrench += self.rng.normal(0, .002, wrench.shape)
        dd = []
        for i in range(2):
            yy, xx = np.mgrid[:32, :48]
            offsets = getattr(self, "offsets", [0, 0, 0])
            u = 24 + offsets[0] * .2
            v = 16 + offsets[2] * .1
            if self.scenario == "slip" and self.stage == "HOLD":
                u += min(16., (self.now_ns - self.phase_start_ns) / 1e9 * 9.)
            raw = self.rng.normal(0, .0005, (32, 48, 3)).astype(np.float32)
            if normal[i] > .1:
                raw[:, :, 0] += (np.exp(-((xx-u)**2/45 + (yy-v)**2/24)) * normal[i] * .25).astype(np.float32)
            dd.append(raw)
        self.seq += 1
        self.current = float(np.mean(normal) * .05)
        self.snapshot_cache = {"timestamp_ns": self.now_ns, "gripper_timestamp_ns": self.now_ns, "tcp": self.pose.copy(),
            "width_mm": self.width, "velocity_mm_s": self.gripper_speed, "current_A": self.current,
            "normal_N": normal, "normal": normal, "wrench": wrench, "wrench_force": wrench[:, :3].copy(), "dd": dd, "force_target_N": self.force_target,
            "controller_state": "force" if self.force_mode else "position", "fault": None}
        payloads = {"dd_left": dd[0], "dd_right": dd[1], "wrench_left": wrench[0], "wrench_right": wrench[1],
            "tcp_measured": self.pose.copy(), "tcp_command": self.command_pose.copy(),
            "gripper": np.array([self.width, self.gripper_speed, self.current]),
            "gripper_command": np.array([self.width_command, self.gripper_speed]),
            "force_target": np.array(self.force_target), "normal_force_" + self.cfg["force"].get("unit", "N"): normal.copy(),
            "controller_state": np.array(int(self.force_mode), np.int8)}
        self.snapshot_cache["arms"] = self.park_observation()["arms"]
        meta = {"stage": self.stage, "source_sequence": self.seq, "simulated": True, "force_unit": self.cfg["force"].get("unit", "N")}
        for name, data in payloads.items():
            self.pending.append(Sample(name, data, self.now_ns, self.now_ns / 1e9, meta))
        if self.now_ns - self.last_camera_ns >= 100_000_000:
            for name, img in self.images().items():
                self.pending.append(Sample("camera/" + name, img, self.now_ns, None, meta))
            self.last_camera_ns = self.now_ns

    def close(self):
        # Never opens the gripper or changes force/position on teardown.
        self.calls.append(("disconnect_inert", None))
