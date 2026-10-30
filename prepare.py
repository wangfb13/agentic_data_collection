#!/usr/bin/env python3
"""Session preparation; safe default prints unresolved fields without hardware."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import yaml
from bottle_scan.config import load_config, unresolved, require_real_ready, validate_config
from bottle_scan.geometry import check_pose, check_segment, translate_local
from bottle_scan.session import Session, atomic_json, plain


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=str(Path(__file__).with_name("config.ros.yaml")))
    modes = p.add_mutually_exclusive_group()
    modes.add_argument("--dry-run", action="store_true")
    modes.add_argument("--simulate", action="store_true")
    modes.add_argument("--real", action="store_true")
    p.add_argument("--demo", action="store_true")
    p.add_argument("--session", default="outputs/demo_session")
    p.add_argument("--single-step", action="store_true")
    p.add_argument("--small-demo", action="store_true", help="Reduce synthetic grids for integration testing only")
    p.add_argument("--rois-json", help="Optional manually drawn two-camera ROI JSON; otherwise OpenCV GUI")
    tuning = p.add_mutually_exclusive_group()
    tuning.add_argument("--tune", dest="tune", action="store_true", help="Probe/release/1 mm a/l refinement (default)")
    tuning.add_argument("--no-tune", dest="tune", action="store_false", help="Explicitly skip T0 refinement")
    p.set_defaults(tune=True)
    args = p.parse_args(argv)
    if args.real and (args.demo or args.small_demo):
        raise ValueError("Synthetic options cannot be used on hardware")
    cfg = load_config(args.config, demo=args.simulate or args.demo)
    if not cfg.get("demo") and unresolved(cfg):
        print("实机参数待填：\n"+"\n".join(unresolved(cfg)))
        if args.real:
            raise ValueError("TODO gate: no hardware module imported")
        print("可先运行 prepare.py --dry-run --demo 查看合成准备流程。")
        return 0
    if args.small_demo:
        cfg["design"].update(bottle_axis_range_mm=[-10,10], bottle_axis_step_mm=10,
            other_axis_range_mm=[-5,5], other_axis_step_mm=5, closure_offsets_mm=[-4,0,4],
            tilt_degrees=[-5,0,5], repeats=1)
        cfg["visual"]["count"] = 16
        cfg["gripper"]["close_speed_mm_s"] = 20.
        cfg["motion"].update(free_speed_mm_s=30., near_speed_mm_s=20., lift_speed_mm_s=20.)
        cfg["recording"]["control_hz"] = 10.
        cfg["sample_hz"] = cfg["tactile"]["fps"] = 10.
        if "close" in cfg:
            cfg["close"].update(observed_force_fps=10., observed_dd_fps=10., max_overtravel_mm=4.)
        for camera in cfg["cameras"].values():
            camera["fps"] = 10.
        cfg["motion"]["stage_timeouts_s"]["SQUEEZE"] = 2.
    errors = validate_config(cfg)
    if errors:
        raise ValueError("Invalid configuration: "+"; ".join(errors))
    T0 = np.asarray(cfg["T0"],float)
    T_check = translate_local(T0, -cfg["check"]["retreat_mm"], "a", cfg["axes"])
    reasons = check_pose(T0,cfg,T0)+check_pose(T_check,cfg,T0)+check_segment(T0,T_check,cfg,T0)
    preview = {"simulated_geometry":cfg.get("demo",False),"T0":T0.tolist(),"T_check":T_check.tolist(),
        "preparation_safety_reasons":sorted(set(reasons)),"stages":["record parked nonexperimental arm", "read T0","probe tuning (default)",
        "free-space +a/+l/+c verification","sensor directions","10 s unloaded reference",
        "two wrist images at T_check, operator ROIs","stationary repeat images",
        "operator ~2 mm displacement and pixel calibration","operator reset; compare ORIGINAL references"]}
    print(json.dumps(preview,ensure_ascii=False,indent=2))
    if not (args.real or args.simulate):
        return 0
    if reasons:
        raise ValueError("Preparation path rejected: "+str(reasons))
    if args.real:
        require_real_ready(cfg, preparing=True)
        if not args.single_step:
            raise ValueError("Initial hardware preparation requires --single-step")
    from bottle_scan.backend import SimBackend, load_hardware
    from bottle_scan.runtime import Runner
    from bottle_scan.sensing import TemplateGuard, dd_descriptors
    backend = SimBackend(cfg,T0) if args.simulate else load_hardware(cfg)
    try:
        if args.real:
            input("先将夹爪定位到理想T0（可用 move-tcp），退出定位工具并现场守护急停；回车只读取当前TCP为本session T0：")
            snap = backend.observation_snapshot()
            age = None if snap is None else backend.monotonic_ns()-int(snap["timestamp_ns"])
            if age is None or not 0 <= age <= cfg["recording"]["stale_timeout_s"]*1e9:
                raise ValueError("Cannot record stale/missing TCP")
            if not np.isfinite(snap["width_mm"]) or abs(snap["width_mm"]-cfg["gripper"]["open_mm"]) > cfg["gripper"]["open_tolerance_mm"]:
                raise ValueError("T0 must be recorded with gripper already open to measured w_open")
            T0 = np.asarray(snap["tcp"],float)
            cfg["T0"] = T0.tolist()
            if check_pose(T0,cfg,T0,width_mm=snap["width_mm"]):
                raise ValueError("Measured T0 is outside configured collision envelope")
        T_check = translate_local(T0, -cfg["check"]["retreat_mm"], "a", cfg["axes"])
        session = Session.create(args.session,cfg,T0,simulated=args.simulate)
        if args.real:
            backend.prepare_zero(session, confirm=input)
        if hasattr(backend, "provenance"):
            session.meta["hardware_runtime"] = backend.provenance()
            from bottle_scan.recording import dd_processing_metadata
            session.meta["dd_processing"] = dd_processing_metadata(backend.provenance())
            session.save_meta()
        atomic_json(session.path/"plans"/"preparation_dry_run.json",preview)
        runner = Runner(session,backend,single_step=args.single_step,unattended=args.simulate)
        if hasattr(backend, "park_observation"):
            from bottle_scan.park_guard import ParkGuard
            other = "left" if cfg["hardware"].get("arm_side", "right") == "right" else "right"
            runner.ask(f"非实验臂 {other} 应已用 move-tcp --arm {other} 停在能看清瓶子和实验夹爪的位置，并退出该工具。回车只记录停放 TCP 和关节角：")
            ParkGuard.capture(session, backend.park_observation(), now_ns=backend.monotonic_ns())
        # The first commanded movement always has its own concrete MOVE preview.
        runner.confirm_startup_move(T_check)
        with runner.phase("CHECK"):
            runner.ensure_open()
            runner.move(T_check, noncontact=True)
        if args.tune:
            with runner.phase("APPROACH"):
                runner.move(T0, noncontact=True)
        if args.tune:
            tune(runner,session)
            T0 = session.T0
            cfg["T0"] = T0.tolist()
        record_height_crosscheck(runner, session)
        # Record the actual measured/tuned frame used by every later target.
        T_check = translate_local(T0,-cfg["check"]["retreat_mm"],"a",cfg["axes"])
        actual_reasons=check_pose(T0,cfg,T0)+check_pose(T_check,cfg,T0)+check_segment(T0,T_check,cfg,T0)
        preview.update(T0=T0.tolist(),T_check=T_check.tolist(),preparation_safety_reasons=actual_reasons)
        atomic_json(session.path/"plans"/"preparation_dry_run.json",preview)
        if actual_reasons:
            raise ValueError("Measured preparation geometry rejected: "+str(actual_reasons))
        axis_checks(runner,session)
        sensor_checks(runner,session)
        T_check = translate_local(T0,-cfg["check"]["retreat_mm"],"a",cfg["axes"])
        with runner.phase("CHECK"):
            runner.ensure_open()
            runner.move(T_check,noncontact=True)
            runner.wait(cfg["motion"]["settle_s"],noncontact=True)
        runner.baseline()
        samples = backend.image_samples()
        images = {name:sample.data for name,sample in samples.items()}
        image_metadata = capture_pose_metadata(runner, samples)
        session.meta["bottle_reference_image_metadata"] = image_metadata
        preview_dir = session.path/"references"/"bottle_position_preview"
        save_reference_preview(images, preview_dir, backend.monotonic_ns(), image_metadata=image_metadata)
        print(f"本次原尺寸腕部参考预览已保存：{preview_dir.resolve()}")
        if args.rois_json:
            rois = load_or_wait_rois(args.rois_json, runner)
        elif args.simulate:
            rois = {"left_wrist":[88,42,61,101],"right_wrist":[92,42,61,101]}
        else:
            import cv2
            rois = {}
            for name,image in images.items():
                print(f"{name}: 框选必须包含瓶身左右两侧的轮廓边缘，不要只框标签纹理。回车确认；取消将停止。")
                rois[name] = list(map(int,cv2.selectROI(name,cv2.cvtColor(image,cv2.COLOR_RGB2BGR),False,False)))
                cv2.destroyWindow(name)
        guard = TemplateGuard(images,rois,search_radius_px=cfg["check"].get("search_radius_px",80),
                              min_score=cfg["check"].get("min_score",.8))
        # ROI selection/file creation may take minutes. Begin a NEW bounded
        # observation phase after that human wait, without any actuator command.
        captured_pairs = capture_observation_pairs(runner, 3)
        noise = [pair for _, pair in captured_pairs]
        runner.ask("请沿接近方向 a 将瓶子故意平移约2 mm（需两路均可分辨），放稳后按回车进行阈值标定：")
        if args.simulate:
            backend.bottle_shift_px = [8,0]
        # The intentional displacement is another unbounded human wait; only
        # the subsequent settling/capture belongs to the new CHECK timeout.
        moved_capture = capture_observation_pairs(runner, 1)
        moved = moved_capture[0][1]
        captured_pairs.extend(moved_capture)
        calibration = guard.calibrate(moved,noise_images=noise,movement_mm=cfg["check"].get("calibration_mm",2),
            threshold_fraction=cfg["check"].get("calibration_fraction",.5))
        print("像素阈值标定：",json.dumps(calibration,ensure_ascii=False))
        guard.save(session.path/"references"/"bottle_position")
        atomic_json(session.path/"references"/"bottle_position"/"image_pose_metadata.json", image_metadata)
        # Calibration observations are evidence, separate from immutable originals.
        from bottle_scan.recording import AsyncRecorder
        with AsyncRecorder(session.path/"references"/"calibration_observations", cfg) as writer:
            for captured_ns,pair in captured_pairs:
                for name,image in pair.items():
                    attrs = runner.calibration_image_metadata.get(captured_ns, {}).get(name, {})
                    writer.jpg.append(name,image,attrs.get("source_received_monotonic_ns", captured_ns), attrs=attrs)
        session.meta["T_check"] = T_check.tolist()
        session.meta["position_calibration"] = calibration
        session.meta["reference_sha256"] = guard.reference_sha256
        session.meta["camera_mounts_verified"] = True
        cfg["check"]["threshold_px"] = guard.thresholds_px
        session.cfg = cfg
        (session.path/"config_snapshot.yaml").write_text(yaml.safe_dump(plain(cfg),sort_keys=False),encoding="utf-8")
        session.save_meta()
        runner.guard = guard
        runner.ask("现在请将瓶子复位到最初的标记位置，再按回车；随后与初始参考图比较：")
        if args.simulate:
            backend.bottle_shift_px = [0,0]
        runner.check_bottle("calibration_reset")
        if args.simulate:
            session.meta["simulated_clock_last_ns"] = backend.monotonic_ns()
            session.save_meta()
        print(f"Session准备完成：{session.path}。原始参考已锁定；更换瓶位/T0请创建新session。")
    except BaseException:
        # No bottle is lifted during preparation. Attempt safe release only after
        # a verified stop; no blind retreat if preparation geometry is uncertain.
        if 'runner' in locals() and runner.actuation_started:
            backend.stop_arm()
            try:
                with runner.phase("RELEASE", emergency=True):
                    runner.emergency_release_pre_lift(already_stopped=True)
                with runner.phase("RETREAT", emergency=True):
                    retreat=translate_local(backend.snapshot()["tcp"],-cfg["motion"]["pre_distance_mm"],"a",cfg["axes"])
                    runner.move(retreat,noncontact=True)
            except BaseException:
                backend.stop_arm()
                print("准备异常的安全释放/后撤未完成，请人工处理。",file=sys.stderr)
        raise
    finally:
        backend.close()
    return 0


def save_reference_preview(images, directory, captured_ns, *, image_metadata=None):
    """Write the initial full-size RGB pixels once, before any ROI interaction.

    These lossless previews are separate from the calibrated immutable guard.
    They do not establish a usable bottle-position reference by themselves.
    """
    import cv2
    import hashlib
    if set(images) != {"left_wrist", "right_wrist"}:
        raise ValueError("Preview requires exactly the two wrist images")
    encoded, records = {}, {}
    for name, image in images.items():
        image = np.asarray(image)
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"{name}: expected native RGB uint8 image")
        ok, png = cv2.imencode(".png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        if not ok:
            raise OSError(f"Could not encode {name} preview")
        encoded[name] = png.tobytes()
        records[name] = {"file": name+".png", "shape": list(image.shape),
                         "pixel_sha256": hashlib.sha256(image.tobytes()).hexdigest()}
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    for name, data in encoded.items():
        with (directory/(name+".png")).open("xb") as stream:
            stream.write(data)
    atomic_json(directory/"preview.json", {
        "schema": "bottle_position_preview_v1", "images": records,
        "host_pair_received_ns": int(captured_ns),
        "timestamp_semantics": "host time after images() returned both fresh frames; not device exposure time",
        "reference_policy": "preview_only; calibrated immutable references are in sibling bottle_position",
        "per_image_source_and_poses": image_metadata or {},
    })


def capture_pose_metadata(runner, samples):
    from bottle_scan.park_guard import image_pose_metadata
    result = {}
    for name,sample in samples.items():
        attrs = dict(sample.metadata or {})
        if "pose_sources" not in attrs:
            attrs.update(image_pose_metadata(runner.backend.snapshot(), runner.session, camera_stamp_ns=sample.monotonic_ns))
        attrs.update(source_received_monotonic_ns=sample.monotonic_ns,
            source_device_timestamp=sample.device_timestamp)
        result[name] = attrs
    return result


def load_or_wait_rois(filename, runner):
    """Existing JSON loads unchanged; a real operator may create a missing file."""
    path = Path(filename).expanduser().resolve()
    while not path.exists():
        if runner.backend.simulated:
            raise FileNotFoundError(f"Synthetic unattended preparation needs an existing ROI JSON: {path}")
        runner.ask(f"ROI JSON 尚不存在：{path}\n请在另一终端根据本次原尺寸预览创建该文件，再按回车读取；Ctrl-C 中止。等待期间不移动机械臂：")
    return json.loads(path.read_text(encoding="utf-8"))


def capture_observation_pairs(runner, count):
    """Fresh pure-observation timeout after operator waits; never moves/opens."""
    if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
        raise ValueError("Observation count must be a positive integer")
    pairs = []
    with runner.phase("CHECK"):
        for _ in range(count):
            runner.wait(runner.cfg["motion"]["settle_s"], noncontact=True)
            if hasattr(runner.backend, "image_samples"):
                samples = runner.backend.image_samples()
                pair = {name:sample.data for name,sample in samples.items()}
                metadata = capture_pose_metadata(runner, samples)
            else:
                pair, metadata = runner.backend.images(), {}
            stamp = runner.backend.monotonic_ns()
            if not hasattr(runner, "calibration_image_metadata"):
                runner.calibration_image_metadata = {}
            runner.calibration_image_metadata[stamp] = metadata
            pairs.append((stamp, pair))
    return pairs


def axis_checks(runner,session):
    cfg=runner.cfg
    if runner.backend.simulated:
        session.meta["axis_verification"]={"simulated":True,"confirmed":["a","l","c"]}
        session.meta["axis_verified"] = True
        session.save_meta()
        return
    free_pose=np.asarray(cfg["preparation"]["axis_test_pose"],float)
    with runner.phase("MOVE_PRE"):
        runner.ensure_open()
        runner.move(free_pose,noncontact=True)
    for axis in ("a","l","c"):
        with runner.phase("APPROACH"):
            target=translate_local(free_pose,5.,axis,cfg["axes"])
            runner.move(target,speed=cfg["motion"]["near_speed_mm_s"],noncontact=True)
            if runner.ask(f"已沿 +{axis} 移动5mm。方向正确请输入 yes；否则立即停止并修改配置：").strip()!="yes":
                raise ValueError("Axis direction rejected by operator")
        with runner.phase("RETREAT"):
            runner.move(free_pose,speed=cfg["motion"]["near_speed_mm_s"],noncontact=True)
    session.meta["axis_verification"]={"confirmed":["a","l","c"],"pose":free_pose.tolist()}
    session.meta["axis_verified"] = True
    session.save_meta()


def sensor_checks(runner,session):
    if runner.backend.simulated:
        session.meta["sensor_axis_verification"]={"simulated":True}
        session.meta["sensor_verified"] = True
        session.save_meta()
        return
    from bottle_scan.sensing import dd_descriptors
    observations={}
    for side,i in (("left",0),("right",1)):
        observations[side]={}
        for direction in ("near_fingertip_+a","other_inplane_+l"):
            runner.ask(f"请按压 {side} 传感器的 {direction} 一侧，保持后按回车读取DD质心：")
            snapshot=runner.backend.snapshot()
            observations[side][direction]=dd_descriptors(snapshot["dd"][i],pixel_pitch_mm=runner.cfg["tactile"]["pixel_pitch_mm"],
                depth_channel=runner.cfg["tactile"].get("depth_channel",0),contact_threshold=runner.cfg["tactile"]["dd_threshold"])
            print(json.dumps(observations[side][direction],ensure_ascii=False))
            if runner.ask("质心方向与配置一致请输入 yes；不一致请停止修改映射：").strip()!="yes":
                raise ValueError("Sensor mapping rejected")
            runner.ask("请移开手指后按回车：")
    session.meta["sensor_axis_verification"]=observations
    session.meta["sensor_verified"] = True
    session.save_meta()


def tune(runner,session):
    """Probe then RELEASE before every 1 mm move; never slide a closed grasp."""
    from bottle_scan.sensing import dd_descriptors
    from bottle_scan.geometry import pose_offset
    while True:
        runner.meta={"trial_id":"preparation_tune","stages":[],"auto":{"contact_times_ns":[None,None],"contact_widths_mm":[None,None]}}
        # Preparation diagnostics are kept separately from experiment trials.
        (session.path/"trials"/"preparation_tune").mkdir(exist_ok=True)
        if not runner.close_contact():
            raise ValueError("T0 tuning failed to achieve bilateral contact")
        runner.probe()
        snap=runner.backend.snapshot()
        residuals = {}
        for side,i in (("left",0),("right",1)):
            desc = dd_descriptors(snap["dd"][i],pixel_pitch_mm=runner.cfg["tactile"].get("pixel_pitch_mm"),
                depth_channel=runner.cfg["tactile"].get("depth_channel",0),contact_threshold=runner.cfg["tactile"]["dd_threshold"])
            center = (np.asarray(snap["dd"][i].shape[1::-1])-1)/2
            residual = None if desc["centroid_px"] is None else np.asarray(desc["centroid_px"])-center
            pitch = runner.cfg["tactile"].get("pixel_pitch_mm")
            residuals[side] = {"centroid_px":desc["centroid_px"], "residual_px":None if residual is None else residual.tolist(),
                "residual_mm":None if residual is None or not isinstance(pitch,(int,float,list,tuple)) else (residual*np.asarray(pitch)).tolist(),
                "definition":"depth-weighted DD centroid minus sensor image center", "probe_not_settled":runner.meta["auto"]["probe_not_settled"]}
            print(side, json.dumps(residuals[side], ensure_ascii=False))
        with runner.phase("RELEASE"):
            runner._command_force(0.)
            runner.ensure_open()
        key=runner.ask("微调：a+/a-/l+/l- 每次1mm；回车确认T0：").strip()
        if not key:
            break
        if key not in ("a+","a-","l+","l-"):
            continue
        delta=[0.,0.,0.]
        delta[("a","c","l").index(key[0])]=1. if key[1]=="+" else -1.
        target=pose_offset(runner.backend.snapshot()["tcp"],delta,axes=runner.cfg["axes"])
        with runner.phase("APPROACH"):
            runner.move(target,speed=runner.cfg["motion"]["near_speed_mm_s"],noncontact=True)
    old_T0 = session.T0.copy()
    session.T0=np.asarray(runner.backend.snapshot()["tcp"])
    # The bottle stayed on the table while the open TCP was translated.
    # Express the same measured bottle-axis origin in the refined TCP frame.
    old_center = np.asarray(runner.cfg["geometry"]["bottle"]["center_offset_mm"], float)
    fixed_center_base = old_T0[:3,3] + old_T0[:3,:3] @ old_center
    new_center = session.T0[:3,:3].T @ (fixed_center_base-session.T0[:3,3])
    runner.cfg["geometry"]["bottle"]["center_offset_mm"] = new_center.tolist()
    session.meta["tune_geometry_reexpression"] = {"old_T0":old_T0.tolist(),
        "old_center_offset_mm":old_center.tolist(),"new_center_offset_mm":new_center.tolist(),
        "bottle_axis_origin_base_mm":fixed_center_base.tolist(),
        "reason":"bottle remains fixed while open-gripper translation refines T0"}
    session.meta["T0"]=session.T0.tolist()
    from bottle_scan.geometry import grasp_pose
    session.meta["B"]=grasp_pose(session.T0, runner.cfg).tolist()
    session.meta["B_legacy"]=session.T0.tolist()
    session.meta["tune_residuals"] = residuals
    runner.cfg["T0"]=session.T0.tolist()
    session.save_meta()
    runner.meta=None


def record_height_crosscheck(runner, session):
    from bottle_scan.geometry import grasp_pose
    computed = float(grasp_pose(session.T0, runner.cfg)[2,3] - runner.cfg["safety"]["table_z_mm"])
    if runner.backend.simulated:
        measured = computed
    else:
        measured = float(runner.ask(f"抓取中心距桌面计算值为 {computed:.3f} mm；请输入卡尺实测高度（mm）："))
        if not np.isfinite(measured) or measured < 0:
            raise ValueError("Caliper height must be a finite nonnegative mm measurement")
    threshold = float(runner.cfg["preparation"]["height_crosscheck_tolerance_mm"])
    delta = measured-computed
    session.meta["grasp_center_height"] = {"computed_mm":computed,"caliper_mm":measured,
        "difference_mm":delta,"warning":abs(delta)>threshold,"tolerance_mm":threshold,
        "simulated":runner.backend.simulated}
    if abs(delta)>threshold:
        print(f"警告：抓取中心高度与卡尺相差 {delta:.3f} mm，请复核 G、T0 和桌面 z。")
    session.save_meta()


if __name__=="__main__":
    try:
        raise SystemExit(main())
    except (ValueError,RuntimeError,OSError,KeyboardInterrupt) as exc:
        print(f"STOP: {exc}",file=sys.stderr)
        raise SystemExit(2)
