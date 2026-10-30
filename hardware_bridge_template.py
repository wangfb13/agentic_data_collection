"""Explicit first-version hardware seam; NOT A WORKING ROBOT DRIVER.

Bind existing validated drivers/controller here after commissioning their units,
native DD access, stopping and path guarantees. Never fill these with guessed
ROS topic names or republish-current-pose as a supposed stop. Do not instantiate
Spark3PoseGripper: its default action can command both arms/grippers.

Required methods are documented in docs/HARDWARE_CONTRACT.md. This module can
be imported safely: no SDK imports, device construction or transport side effects.
"""
CAPABILITIES = {
    "safe_connect_no_motion": False,
    "bounded_linear_motion": False,
    "verified_arm_stop": False,
    "gripper_mm_calibration": False,
    "per_finger_force_N": False,
    "raw_sdk_dd": False,
    "host_monotonic_timestamps": False,
    "independent_acquisition": False,
    "hold_on_disconnect": False,
    "force_limit_enforced": False,
    "width_limit_enforced": False,
    "dynamic_gripper_width_limit": False,
    "verified_fresh_image_capture": False,
    "watchdog_stops_arm_keeps_grip": False,
}


def create(cfg):
    raise NotImplementedError("TODO_HARDWARE_BRIDGE: verify and bind existing drivers before physical operation")
