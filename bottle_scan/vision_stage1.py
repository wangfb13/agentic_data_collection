"""Stage-one CLI: offline planning, read-only teaching, supervised commissioning.

No device imports or connections until an explicit --real teach/commission call.
No gripper closure, contact, lift, skill training or autonomous experiment loop.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import time

import numpy as np
import yaml

from .geometry import pose_error
from .session import atomic_json, plain, fingerprint
from .vision_workspace import make_workspace_profile, teaching_from_captures, TaughtWorkspace
from .vision_localization import validate_calibration, localize_image, calibration_digest
from .vision_motion import plan_alignment, GuardedVisionAdapter, localize_sample


def _read(path):
    data = yaml.safe_load(Path(path).read_text())
    if not isinstance(data, (dict, list)):
        raise ValueError("Expected a JSON/YAML mapping or capture list")
    return data


def _new(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(plain(data), stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write("\n")


def _confirm(message, token, ask=input):
    if ask(message + " 输入 " + token + "：").strip() != token:
        raise ValueError("operator did not confirm " + token)


def _read_only_observation(reader, calibration):
    sample = reader.sensors.image_samples()[calibration["camera_name"]]
    arms = reader.arm.snapshot()["arms"]
    observation = localize_sample(sample, arms, calibration, time.monotonic_ns())
    if not observation["valid"]:
        raise ValueError("visual localization rejected: " + ";".join(observation["reasons"]))
    return sample, observation


def _live_calibration(calibration, station):
    if calibration["simulated"]:
        raise ValueError("synthetic calibration cannot authorize physical work")
    if calibration["camera_mount"] != "eye_in_hand":
        raise ValueError("physical stage one supports only wrist cameras using live TCP; fixed is offline-only")
    if calibration["camera_name"] != calibration["mount_tcp_side"]+"_wrist":
        raise ValueError("camera name does not match its physical wrist")
    if calibration["evidence"].get("station_hardware_hash") != station["hardware_hash"]:
        raise ValueError("calibration must be bound to the current measured station identity")


def teach(args, *, ask=input, say=print):
    if not args.real:
        raise ValueError("teach is read-only live capture; explicit --real is required")
    if Path(args.out).exists():
        raise ValueError("output already exists")
    from .wizard_real import require_real_profile, model_identity, read_model
    from .wizard_station import LiveReader
    station = require_real_profile(args.station)
    cfg = station["config"]
    calibration = validate_calibration(_read(args.calibration))
    _live_calibration(calibration, station)
    limits = _read(args.limits)
    if not isinstance(limits, dict):
        raise ValueError("limits must be a mapping")
    if "geometry_reference_tcp_mm" not in limits:
        raise ValueError("explicit measured geometry_reference_tcp_mm required; not inferred from desired grasp")
    captures = []
    evidence = Path(str(args.out)+".evidence")
    evidence.mkdir(parents=True, exist_ok=False)
    say("只读示教：只采位姿和图像，不创建运动后端。请自行遥操作并停止后记录。")
    with LiveReader(cfg) as reader:
        if model_identity(read_model(reader, cfg)) != model_identity(station["model"]):
            raise ValueError("station hardware identity changed")
        for role, label in (("lower_corner", "工作空间下界角"), ("upper_corner", "工作空间上界角"),
                            ("reference", "认可的目标夹持位置（保持全开、无接触）"),
                            ("observation", "观察起点（保持全开、无接触）")):
            _confirm("遥操作到"+label+"，停止遥操作", "RECORD", ask)
            tcp = reader.pose()
            from .hardware_gripper import WidthCalibration
            width = WidthCalibration(cfg["hardware"]["gripper_width_calibration"]).to_mm(reader.gripper()["normalized"])
            if abs(width-cfg["gripper"]["open_mm"]) > cfg["gripper"]["open_tolerance_mm"]:
                raise ValueError("all teaching poses require fully open gripper")
            row = {"role": role, "tcp_mm": tcp.tolist(), "recorded_monotonic_ns": time.monotonic_ns(),
                   "gripper": plain(reader.gripper()), "read_only": True}
            if role == "reference":
                sample, observation = _read_only_observation(reader, calibration)
                row["constraints"] = deepcopy(limits)
                row["constraints"]["open_width_mm"] = cfg["gripper"]["open_mm"]
                row["constraints"]["reference_calibration_sha256"] = calibration_digest(calibration)
                _confirm("确认整个示教空间内部无障碍，且当前瓶子与既有几何参考对应关系已核对", "SPACE", ask)
                row["constraints"]["clear_space_confirmed"] = True
                captures.append({"role": "bottle_reference", "pose_mm": observation["T_base_bottle_mm"],
                                 "visual_observation": observation})
                import cv2
                if not cv2.imwrite(str(evidence/"reference.png"), cv2.cvtColor(sample.data, cv2.COLOR_RGB2BGR)):
                    raise ValueError("failed to save reference image")
            after = reader.pose()
            error = pose_error(tcp, after)
            if error[0] > cfg["motion"]["pose_tolerance_mm"] or error[1] > cfg["motion"]["pose_tolerance_deg"]:
                raise ValueError("robot moved during teaching capture; record again in a new session")
            captures.append(row)
            atomic_json(evidence/"captures.json", captures)
    teaching = teaching_from_captures(captures, cfg)
    profile = make_workspace_profile(cfg, teaching, simulated=False)
    profile_path = Path(args.out)
    _new(profile_path, profile)
    say("Saved taught workspace: "+str(profile_path)+"；尚未执行或验收视觉运动。")
    return 0


def commission(args, *, ask=input, say=print):
    if not args.real:
        raise ValueError("commission requires explicit --real and现场 MOVE confirmation")
    if not 2 <= args.max_steps <= 50 or not np.isfinite(args.tolerance_mm) or args.tolerance_mm <= 0:
        raise ValueError("max_steps must be 2..50 and tolerance must be positive")
    if Path(args.out).exists():
        raise ValueError("commissioning output exists; use a new session")
    # Validate all offline artifacts before opening even read-only transports.
    profile, calibration = _read(args.workspace), validate_calibration(_read(args.calibration))
    workspace = TaughtWorkspace(profile)
    if profile["simulated"] or calibration["simulated"]:
        raise ValueError("synthetic artifacts cannot authorize physical commissioning")
    from .wizard_real import require_real_profile, doctor, model_identity, read_model
    from .wizard_station import LiveReader
    from .wizard import UI
    station = require_real_profile(args.station)
    _live_calibration(calibration, station)
    if fingerprint(profile["geometry_config"]) != fingerprint(station["config"]):
        raise ValueError("workspace geometry/hardware differs from commissioned station")
    cfg = workspace.config
    ui = UI()
    doctor(station["config"], Path(args.out).parent/(".doctor_"+Path(args.out).name+".json"), ui)
    _confirm("请置于示教观察位，夹爪完全张开且空载，停止遥操作并交出控制权", "READY", ask)
    with LiveReader(station["config"]) as reader:
        if model_identity(read_model(reader, station["config"])) != model_identity(station["model"]):
            raise ValueError("station hardware identity changed")
        current = reader.pose()
        error = pose_error(current, workspace.teaching["observation_tcp_mm"])
        if error[0] > cfg["motion"]["pose_tolerance_mm"] or error[1] > cfg["motion"]["pose_tolerance_deg"]:
            raise ValueError("start must match the taught observation pose")
        _, observed = _read_only_observation(reader, calibration)
        first = plan_alignment(profile, calibration, observed, current, now_ns=time.monotonic_ns(),
                               tolerance_mm=args.tolerance_mm)
        if not first["valid"]:
            raise ValueError("preflight rejected: "+";".join(first["reasons"]))
    # Existing zeroing, parked-arm and Runner guards are retained. No movement
    # occurs on construction, and no automatic opening/closing is requested.
    from .session import Session
    from .backend import load_hardware
    from .runtime import Runner
    from .park_guard import ParkGuard
    session = Session.create(args.out, cfg, workspace.geometry_anchor(observed["T_base_bottle_mm"]), simulated=False)
    session.meta.update(vision_stage1=True, station_hardware_hash=station["hardware_hash"],
                        operator_empty_confirmation=True, visual_alignment_is_contact_success=False)
    session.save_meta()
    _new(session.path/"vision_workspace.json", profile)
    _new(session.path/"vision_calibration.json", calibration)
    backend = None
    report = {"schema": "vision_stage1_commissioning_v1", "simulated": False, "status": "incomplete",
              "attempts": [], "physical_acceptance": "not_issued", "contact_success": None}
    try:
        backend = load_hardware(cfg)
        backend.prepare_zero(session, confirm=ask)
        ParkGuard.capture(session, backend.park_observation(), now_ns=backend.monotonic_ns())
        runner = Runner(session, backend, single_step=False, unattended=False)
        adapter = GuardedVisionAdapter(runner, profile, calibration)
        confirmations = 0
        last_confirmation = None
        for index in range(args.max_steps):
            ParkGuard(session).check(backend.park_observation(), now_ns=backend.monotonic_ns(), context="vision_before_image")
            preview_sample = backend.image_samples()[calibration["camera_name"]]
            preview_snapshot = runner._healthy_snapshot()
            preview_observation = localize_sample(preview_sample, preview_snapshot["arms"], calibration, int(runner.now()))
            preview = plan_alignment(profile, calibration, preview_observation, preview_snapshot["tcp"],
                now_ns=int(runner.now()), travelled_mm=adapter.travelled_mm,
                tolerance_mm=args.tolerance_mm, width_mm=float(preview_snapshot["width_mm"]))
            if not preview["valid"]:
                raise ValueError("preview rejected: "+";".join(preview["reasons"]))
            say(json.dumps({"status": preview["status"], "next_tcp_mm": preview["target_tcp_mm"],
                            "goal_tcp_mm": preview["goal_tcp_mm"], "contact_allowed": False}, ensure_ascii=False))
            _confirm("核对上述目标与活动空间；仅执行一步开爪平移", "MOVE", ask)
            ParkGuard(session).check(backend.park_observation(), now_ns=backend.monotonic_ns(), context="vision_after_confirmation")
            # Images are fetched AFTER approval; waiting cannot reuse a stale plan.
            sample = backend.image_samples()[calibration["camera_name"]]
            snapshot = runner._healthy_snapshot()
            observed = localize_sample(sample, snapshot["arms"], calibration, int(runner.now()))
            folder = session.path/"checks"/f"vision_{index:03d}"
            folder.mkdir()
            import cv2
            if not cv2.imwrite(str(folder/"image.png"), cv2.cvtColor(sample.data, cv2.COLOR_RGB2BGR)):
                raise ValueError("failed to persist image before motion")
            atomic_json(folder/"observation.json", observed)
            refreshed = plan_alignment(profile, calibration, observed, snapshot["tcp"],
                now_ns=int(runner.now()), travelled_mm=adapter.travelled_mm,
                tolerance_mm=args.tolerance_mm, width_mm=float(snapshot["width_mm"]))
            if not refreshed["valid"]:
                raise ValueError("post-confirmation vision invalid: "+";".join(refreshed["reasons"]))
            for old, new in ((preview["source_tcp_mm"], snapshot["tcp"]),
                             (preview["target_tcp_mm"], refreshed["target_tcp_mm"])):
                position_error, angle_error = pose_error(old, new)
                if position_error > cfg["motion"]["pose_tolerance_mm"] or angle_error > cfg["motion"]["pose_tolerance_deg"]:
                    raise ValueError("robot or visual target changed during confirmation; movement cancelled")
            receipt = adapter.move_step(observed, tolerance_mm=args.tolerance_mm)
            report["attempts"].append(receipt)
            atomic_json(folder/"receipt.json", receipt)
            if receipt["plan"]["status"] == "aligned":
                if last_confirmation is None or observed["timestamp_ns"]-last_confirmation >= 200_000_000:
                    confirmations += 1
                    last_confirmation = observed["timestamp_ns"]
                if confirmations >= 2:
                    report["status"] = "visual_alignment_confirmed"
                    break
            else:
                confirmations = 0
                last_confirmation = None
            atomic_json(session.path/"vision_report.json", report)
        report["travelled_mm"] = adapter.travelled_mm
    except BaseException as exc:
        report["error"] = str(exc)
        if backend is not None:
            backend.stop_arm()
        raise
    finally:
        atomic_json(session.path/"vision_report.json", report)
        if backend is not None:
            backend.close()
    say(json.dumps(report, ensure_ascii=False))
    return 0 if report["status"] == "visual_alignment_confirmed" else 2


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("build-appearance", help="offline: learn current bottle colour from an explicit body mask")
    p.add_argument("--image", required=True); p.add_argument("--mask", required=True)
    p.add_argument("--background-mask"); p.add_argument("--target-id", required=True)
    p.add_argument("--appearance-mode", choices=("green", "white"), default="green")
    p.add_argument("--evidence", required=True); p.add_argument("--settings")
    p.add_argument("--simulated", action="store_true"); p.add_argument("--out", required=True)
    p = sub.add_parser("detect-bottle", help="offline: unmarked bottle segmentation only; no metric pose")
    p.add_argument("--image", required=True); p.add_argument("--appearance", required=True)
    p.add_argument("--out", required=True); p.add_argument("--mask-out")
    p = sub.add_parser("build-workspace", help="offline: build from saved teaching/captures")
    p.add_argument("--config", required=True)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--teaching"); group.add_argument("--captures")
    p.add_argument("--simulated", action="store_true"); p.add_argument("--out", required=True)
    p = sub.add_parser("preview", help="offline: image -> bottle pose -> checked robot target")
    p.add_argument("--workspace", required=True); p.add_argument("--calibration", required=True)
    p.add_argument("--image", required=True); p.add_argument("--pose", required=True)
    p.add_argument("--tolerance-mm", type=float, default=1.); p.add_argument("--out", required=True)
    p = sub.add_parser("teach", help="read-only live teleoperation capture; no publishers")
    p.add_argument("--station", required=True); p.add_argument("--calibration", required=True)
    p.add_argument("--limits", required=True); p.add_argument("--out", required=True)
    p.add_argument("--real", action="store_true")
    p = sub.add_parser("commission", help="explicit live single-step open-hand visual positioning")
    p.add_argument("--station", required=True); p.add_argument("--workspace", required=True)
    p.add_argument("--calibration", required=True); p.add_argument("--out", required=True)
    p.add_argument("--real", action="store_true"); p.add_argument("--max-steps", type=int, default=10)
    p.add_argument("--tolerance-mm", type=float, default=1.)
    args = parser.parse_args(argv)
    try:
        if args.command == "teach": return teach(args)
        if args.command == "commission": return commission(args)
        if Path(args.out).exists(): raise ValueError("output exists; use a new path")
        if args.command == "build-workspace":
            cfg = _read(args.config)
            cfg = cfg.get("config", cfg)
            teaching = _read(args.teaching) if args.teaching else teaching_from_captures(_read(args.captures), cfg)
            profile = make_workspace_profile(cfg, teaching, simulated=args.simulated)
            _new(args.out, profile)
            print("Saved immutable workspace. No hardware connected.")
            return 0
        import cv2
        image = cv2.imread(args.image)
        if image is None: raise ValueError("cannot read image")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if args.command == "build-appearance":
            from .bottle_appearance import build_appearance_profile
            mask = cv2.imread(args.mask, cv2.IMREAD_GRAYSCALE)
            if mask is None: raise ValueError("cannot read foreground mask")
            background = None
            if args.background_mask:
                background = cv2.imread(args.background_mask, cv2.IMREAD_GRAYSCALE)
                if background is None: raise ValueError("cannot read background mask")
            appearance = build_appearance_profile(image, mask, target_id=args.target_id,
                simulated=args.simulated, evidence=_read(args.evidence), background_mask=background,
                settings=_read(args.settings) if args.settings else None, appearance_mode=args.appearance_mode)
            _new(args.out, appearance)
            print("Saved bottle appearance. No hardware connected; metric calibration still required.")
            return 0
        if args.command == "detect-bottle":
            from .bottle_appearance import segment_bottle
            if args.mask_out and Path(args.mask_out).exists(): raise ValueError("mask output already exists")
            if args.mask_out and Path(args.mask_out).resolve() == Path(args.out).resolve():
                raise ValueError("mask and report outputs must differ")
            detection = segment_bottle(image, _read(args.appearance))
            mask = detection.pop("mask", None)
            detection.update(metric_pose_available=False, hardware_connected=False)
            if mask is not None and args.mask_out:
                # Encode before writing, use exclusive create, never overwrite evidence.
                ok, encoded = cv2.imencode(".png", mask)
                if not ok: raise ValueError("could not encode segmentation mask")
                Path(args.mask_out).parent.mkdir(parents=True, exist_ok=True)
                with Path(args.mask_out).open("xb") as stream: stream.write(encoded.tobytes())
            _new(args.out, detection)
            print(json.dumps({k: detection[k] for k in ("valid", "reasons", "metric_pose_available", "hardware_connected")}))
            return 0 if detection["valid"] else 2
        profile, calibration, pose = _read(args.workspace), _read(args.calibration), _read(args.pose)
        observation = localize_image(image, calibration, T_base_tcp_mm=pose["T_base_tcp_mm"],
            image_timestamp_ns=pose["image_timestamp_ns"], pose_timestamp_ns=pose["pose_timestamp_ns"],
            now_ns=pose["now_ns"], pose_tcp_side=pose.get("pose_tcp_side"))
        plan = plan_alignment(profile, calibration, observation, pose["current_tcp_mm"],
                              now_ns=pose["now_ns"], tolerance_mm=args.tolerance_mm)
        _new(args.out, plan)
        print(json.dumps({"valid": plan["valid"], "reasons": plan["reasons"], "status": plan["status"], "hardware_connected": False}))
        return 0 if plan["valid"] else 2
    except (ValueError, KeyError, TypeError, OSError) as exc:
        print("vision-stage1 stopped: "+str(exc))
        return 2
