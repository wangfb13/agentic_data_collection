#!/usr/bin/env python3
"""Read current ROS/TF TCP and save a named measurement; never send commands."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from bottle_scan.hardware_arm import RosArmObserver
from bottle_scan.hardware_gripper import RosNormalizedGripperObserver
from bottle_scan.session import plain


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("left", "right"), default="right")
    parser.add_argument("--label", required=True, help="e.g. placeholder_1_T0 or placeholder_2_free")
    parser.add_argument("--output", required=True, help="New JSON file; existing files are never overwritten")
    parser.add_argument("--max-age-s", type=float, default=.25)
    parser.add_argument("--width-mm", type=float, help="Optional jaw opening actually measured by operator; does not move gripper")
    args = parser.parse_args(argv)
    output = Path(args.output)
    if output.exists():
        parser.error("Output already exists; choose a new measurement name")
    if args.width_mm is not None and (not 0 < args.width_mm < float("inf")):
        parser.error("--width-mm must be a finite positive manual measurement")
    arm = RosArmObserver(arm_side=args.arm, max_age_s=args.max_age_s)
    grip = RosNormalizedGripperObserver(side=args.arm, max_age_s=args.max_age_s)
    try:
        arm.connect()
        grip.connect()
        deadline = time.monotonic() + 5
        while not grip.snapshot().get("fresh") and time.monotonic() < deadline:
            time.sleep(.05)
        state = arm.snapshot()
        tcp = state["arms"][args.arm]["T_base_tcp_mm"]
        gripper = grip.snapshot()
        if not gripper.get("fresh") or not gripper.get("valid"):
            raise RuntimeError("No fresh gripper state; measurement not saved")
        result = {"label":args.label, "arm":args.arm, "read_only":True,
            "created_unix_ns":time.time_ns(), "T_base_tcp_mm":tcp,
            "pose_source":state, "gripper":gripper,
            "operator_measured_width_mm":args.width_mm,
            "note":"Measurement only. Does not change config, T0, reference images or any actuator."}
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8") as stream:
            json.dump(plain(result), stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        print(json.dumps(plain(result), ensure_ascii=False, indent=2, allow_nan=False))
        print(f"Saved measurement: {output.resolve()}")
    finally:
        grip.close()
        arm.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
