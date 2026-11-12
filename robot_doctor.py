#!/usr/bin/env python3
"""Read-only live ROS and broadcast-wrench checks. No actuator publishers."""
from __future__ import annotations

import argparse
import copy
import json
import time

from bottle_scan.config import load_config, unresolved
from bottle_scan.command_ownership import (inspect_all_commands, current_ros_node_names,
                                          format_ownership_report)
from bottle_scan.hardware_arm import RosArmObserver
from bottle_scan.hardware_gripper import RosNormalizedGripperObserver
from bottle_scan.hardware_sensors import inspect_wrench_pub, inspect_dd_pub, ActualX5SensorSource
from bottle_scan.session import atomic_json, plain


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("left", "right"), default="right")
    parser.add_argument("--config", default="config.ros.yaml")
    parser.add_argument("--output", default="outputs/live_diagnostics/latest.json")
    parser.add_argument("--no-wrench", action="store_true")
    parser.add_argument("--no-dd", action="store_true")
    parser.add_argument("--no-cameras", action="store_true", help="Skip the existing PUSH/PULL wrist streams")
    parser.add_argument("--wrench-address", default="tcp://192.168.9.11:39020")
    parser.add_argument("--dd-address", default="tcp://192.168.9.11:39030")
    parser.add_argument("--serials", nargs=2, default=["X26040505", "X26040305"])
    args = parser.parse_args(argv)
    report = {"read_only":True, "created_unix_ns":time.time_ns(), "arm":args.arm,
        "checks":{}, "errors":[], "notice":"This is connectivity evidence, not physical calibration or motion commissioning."}
    arm = RosArmObserver(arm_side=args.arm)
    grips = []
    try:
        cfg = None
        try:
            cfg = load_config(args.config)
        except Exception as exc:
            report["errors"].append("config: " + str(exc))
        ownership = inspect_all_commands(cfg or {}, own_nodes=current_ros_node_names())
        report["checks"]["command_ownership"] = ownership
        print(format_ownership_report(ownership))
        if not ownership["allowed"]:
            report["errors"].append("command publishers: " + format_ownership_report(ownership))
        try:
            arm.connect()
            report["checks"]["arm"] = arm.snapshot()
        except Exception as exc:
            report["errors"].append("arm: " + str(exc))
        for side in ("left", "right"):
            try:
                grip = RosNormalizedGripperObserver(side=side)
                grips.append(grip)
                grip.connect()
                end = time.monotonic()+3
                while not grip.snapshot().get("fresh") and time.monotonic()<end:
                    time.sleep(.05)
                report["checks"][side+"_gripper"] = grip.snapshot()
                if not report["checks"][side+"_gripper"].get("fresh"):
                    report["errors"].append(side+" gripper: no fresh state")
            except Exception as exc:
                report["errors"].append(side+" gripper: "+str(exc))
        if not args.no_wrench:
            try:
                wrench = inspect_wrench_pub(args.wrench_address, serials=tuple(args.serials), duration_s=5)
                report["checks"]["wrench"] = wrench
                if not wrench["ready"]:
                    report["errors"].append("wrench: both serials did not provide new samples")
            except Exception as exc:
                report["errors"].append("wrench: "+str(exc))
        if not args.no_dd:
            try:
                dd = inspect_dd_pub(args.dd_address, serials=tuple(args.serials), duration_s=5)
                report["checks"]["dd"] = dd
                if not dd["ready"]:
                    report["errors"].append("dd: no valid processed DD for both fingers")
            except Exception as exc:
                report["errors"].append("dd: "+str(exc))
        try:
            if cfg is None:
                raise ValueError("configuration was not loaded")
            report["unmeasured_parameters"] = unresolved(cfg)
            report["configured_bridge_module"] = cfg.get("hardware",{}).get("bridge_module")
            if not args.no_cameras and cfg.get("hardware",{}).get("profile") == "public_ros":
                observation_cfg = copy.deepcopy(cfg)
                settings = observation_cfg["hardware"]["public_ros"]["sensors"]
                # Diagnostic observation must not require a guessed N conversion
                # and never performs software zeroing or SDK calibration.
                settings.pop("force_conversion", None)
                settings.pop("startup_zero", None)
                source = ActualX5SensorSource(observation_cfg)
                try:
                    source.start(timeout_s=5)
                    images = source.images()
                    report["checks"]["wrist_images"] = {name:{"shape":list(img.shape),
                        "dtype":str(img.dtype),"color_order":"RGB","resolution":"source JPEG size",
                        "freshness":"received after request; source exposure timestamp unavailable"}
                        for name,img in images.items()}
                finally:
                    source.close()
        except Exception as exc:
            report["errors"].append("config/camera observation: "+str(exc))
        report["read_only_checks_passed"] = not report["errors"]
        atomic_json(args.output, report)
        print(json.dumps(plain(report), ensure_ascii=False, indent=2, allow_nan=False))
        print(f"Saved diagnostic report: {args.output}")
        return int(bool(report["errors"]))
    finally:
        for grip in grips:
            grip.close()
        arm.close()


if __name__ == "__main__":
    raise SystemExit(main())
