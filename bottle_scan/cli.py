"""Offline-first command line, dry-run reports, and resumable execution."""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import sys
import time
import numpy as np
from .config import load_config, unresolved, require_real_ready, validate_config
from .design import contact_trials, commissioning_trials, visual_points
from .geometry import translate_local
from .session import Session, atomic_json, fingerprint, boot_id, plain


def parser(task):
    p = argparse.ArgumentParser(description=f"Bottle {task} scan; default is offline dry-run")
    p.add_argument("--config", default=str(Path(__file__).resolve().parents[1]/"config.ros.yaml"))
    modes = p.add_mutually_exclusive_group()
    modes.add_argument("--dry-run", action="store_true")
    modes.add_argument("--simulate", action="store_true", help="Synthetic data only; no hardware imports")
    modes.add_argument("--real", action="store_true", help="Operator-started hardware run after all gates")
    p.add_argument("--demo", action="store_true", help="Use a coherent synthetic geometry for dry-run")
    p.add_argument("--session")
    p.add_argument("--output", default=f"outputs/dry_{task}")
    p.add_argument("--single-step", action="store_true")
    p.add_argument("--commissioning", action="store_true", help="Exactly zero/deep +a/above-cap; contact only")
    p.add_argument("--commissioning-lite", action="store_true", help="One zero-offset trial with prior matching full PASS")
    p.add_argument("--verified-session", help="Prior full three-point PASS session for unchanged hardware/bridge")
    p.add_argument("--limit", type=int, help="At most this many pending records; does not change saved full plan")
    p.add_argument("--scenarios", default="normal", help="Simulation fault cases, comma separated")
    p.add_argument("--resume", action="store_true", help="Explicitly resume completed/partial session")
    return p


def plan_for(task, cfg, T0, commissioning=False):
    if task == "contact":
        result = commissioning_trials(cfg, T0) if commissioning else contact_trials(cfg, T0)
        if commissioning:
            for item in result:
                item["trial_id"] = "commission_" + item["trial_id"].split("_")[-1]
        return result
    points = visual_points(cfg, T0)
    return [x for x in points if x["valid"]][:8] if commissioning else points


def dry_report(task, cfg, T0, plan, output, print_all=True):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    valid = [p for p in plan if p["valid"]]
    motion = cfg["motion"]
    per_seconds = []
    start = translate_local(T0, -cfg["check"]["retreat_mm"], "a", cfg["axes"])
    for item in valid:
        target = np.array(item["T_target"])
        if task == "contact":
            pre = np.array(item["T_pre"])
            distance = 2*np.linalg.norm(pre[:3,3]-start[:3,3]) + 2*np.linalg.norm(target[:3,3]-pre[:3,3]) + 2*motion["lift_height_mm"]
            closing = 2*(cfg["gripper"]["open_mm"]-cfg["gripper"]["min_mm"])/cfg["gripper"]["close_speed_mm_s"]
            seconds = distance/min(motion["near_speed_mm_s"], motion["lift_speed_mm_s"]) + closing + motion["hold_s"] + cfg["force"].get("probe_s", .3)+cfg["force"].get("stable_s", .5)+1.
        else:
            seconds = np.linalg.norm(target[:3,3]-start[:3,3])/motion["near_speed_mm_s"]+motion["settle_s"]
            start = target
        per_seconds.append(float(seconds))
    seconds = sum(per_seconds)
    if task == "contact":
        seconds += (1+len(valid)//cfg["design"].get("baseline_every",30))*cfg["tactile"].get("baseline_s",10)
    tactile = cfg["tactile"]
    shape = tactile.get("dd_shape", [0])
    # SDK dtype must be configured on hardware. Demo uses raw float32.
    dd_bytes = int(np.prod(shape))*np.dtype(tactile.get("dd_dtype", "float32")).itemsize
    raw_per_sec = 2*dd_bytes*tactile["fps"] + 2*6*8*tactile["fps"]
    camera_per_sec = sum(cam["fps"] for cam in cfg["cameras"].values())*cfg.get("estimated_bytes_per_image",80000)
    # Metadata, poses and controller logs are estimated separately; gzip ratios
    # are not assumed. Real JPEG size is estimated, not lossless raw size.
    stream_per_sec = raw_per_sec+camera_per_sec+4096*cfg.get("sample_hz",20)
    summary = {"task": task, "simulated_geometry": bool(cfg.get("demo")),
        "candidate_count": len(plan), "accepted_count": len(valid), "rejected_count": len(plan)-len(valid),
        "estimated_active_seconds": seconds, "estimated_storage_GiB": seconds*stream_per_sec/2**30,
        "estimate_excludes": "operator waits, resets, stage retries, transport overhead; images use configured bytes/frame",
        "upper_stage_timeout_seconds": sum(motion["stage_timeouts_s"].values())*len(valid),
        "config_sha256": fingerprint(cfg), "plan_sha256": fingerprint(plan),
        "unresolved_real_parameters": cfg.get("source_unresolved_fields", unresolved(cfg))}
    from .force_control import close_quality_report
    summary["closure_quality"] = close_quality_report(cfg)
    if summary["closure_quality"]["errors"]:
        raise ValueError("; ".join(summary["closure_quality"]["errors"]))
    atomic_json(output/"summary.json", summary)
    atomic_json(output/"plan.json", plan)
    lines = [json.dumps(summary, ensure_ascii=False, indent=2)]
    for item in plan:
        ident = item.get("trial_id",item.get("point_id"))
        from .geometry import grasp_pose
        center = grasp_pose(item['T_target'], cfg)[:3, 3]
        offset = center-grasp_pose(T0, cfg)[:3, 3]
        line = f"{ident} bottle={item['bottle']} kind={item['kind']} delta_acl_mm={item['delta_acl_mm']} rotation={item.get('rotation_axis')}:{item.get('angle_deg',item.get('tilt_deg'))} {'ACCEPT' if item['valid'] else 'SKIP'} reasons={item['safety_reasons']} grasp_center_base_mm={center.tolist()} grasp_center_offset_base_mm={offset.tolist()} T_target={item['T_target']}"
        lines.append(line)
    (output/"dry_run.txt").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print("\n".join(lines if print_all else lines[:1]))
    return summary


def main(task, argv=None):
    args = parser(task).parse_args(argv)
    if args.commissioning_lite and (task != "contact" or args.commissioning):
        raise ValueError("--commissioning-lite is a separate contact-only mode")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")
    if args.real and args.demo:
        raise ValueError("Demo configuration cannot run hardware")
    session = Session(args.session) if args.session else None
    cfg = session.cfg if session else load_config(args.config, demo=args.demo or args.simulate)
    if not cfg.get("demo") and unresolved(cfg):
        print("实测参数尚未填完，不能计算真实目标或运行硬件：\n"+"\n".join(unresolved(cfg)))
        if args.real:
            raise ValueError("Real mode refused before importing hardware")
        print("使用 --dry-run --demo 可查看明确标注的合成示例。")
        return 0
    T0 = session.T0 if session else np.array(cfg["T0"],float)
    invalid = validate_config(cfg)
    if invalid:
        raise ValueError("Invalid configuration: " + "; ".join(invalid))
    plan = plan_for(task, cfg, T0, args.commissioning)
    if args.commissioning_lite:
        plan = plan_for("contact", cfg, T0, True)[:1]
        plan[0]["trial_id"] = "commission_lite_0001"
    output = session.path/"plans"/(task+("_commissioning" if args.commissioning else "")+"_preflight") if session else Path(args.output)
    dry_report(task, cfg, T0, plan, output)
    if not (args.real or args.simulate):
        return 0
    if task == "contact" and cfg["force"]["squeeze_mode"] != "fixed":
        raise ValueError("First version supports fixed force only; ForcePolicy must not start any actuators")
    if session is None:
        raise ValueError("Run prepare.py first and supply --session")
    if bool(session.meta["simulated"]) != bool(args.simulate):
        raise ValueError("Cannot mix synthetic and real data in a session")
    key = task+("_commissioning_lite" if args.commissioning_lite else "_commissioning" if args.commissioning else "")
    session.lock_plan(key, plan)
    pending = [x for x in plan if x.get("trial_id", x.get("point_id")) not in session.progress["completed"]]
    if not pending:
        print("没有待做项；未连接硬件，也未发送机械臂或夹爪命令。")
        return 0
    if session.meta.get("clock_boot_id") != boot_id():
        raise ValueError("Host reboot changed monotonic time domain; create a new session/reference")
    existing = bool(session.progress["completed"] or session.progress["active"])
    if existing and not args.resume:
        if args.simulate:
            raise ValueError("Existing trial progress: use --resume explicitly")
        if input("已有断点，是否续跑且跳过已完成试验？输入 yes：").strip() != "yes":
            return 0
    if session.progress.get("requires_operator_recovery"):
        raise ValueError("Previous attempt may be airborne. Operator must physically recover and start a new session; no automatic reopening.")
    from .backend import SimBackend, load_hardware
    from .runtime import Runner
    scenarios = args.scenarios.split(",")
    if any(s not in {"normal","no_contact","stall","force_timeout","dropped","slip"} for s in scenarios):
        raise ValueError("Unknown synthetic scenario")
    if args.real and args.scenarios != "normal":
        raise ValueError("Synthetic fault injection cannot be used with hardware")
    if args.real:
        require_real_ready(cfg)
        from .stop_validation import gate_stop_report
        from .force_control import require_force_report
        from .commissioning import require_prior_session
        prior = None
        validation_run = args.commissioning or args.commissioning_lite
        if validation_run and not args.single_step:
            raise ValueError("Commissioning requires --single-step")
        if args.commissioning_lite or (task == "visual" and args.commissioning):
            prior = require_prior_session(cfg, args.verified_session)
        if not validation_run:
            from .commissioning import require_current_commissioning
            require_current_commissioning(session, task)
            gate_stop_report(cfg)
            if task == "contact":
                require_force_report(cfg)
        if task == "contact" and not session.progress["commissioning_passed"]:
            if not validation_run or not args.single_step or (args.limit is not None and args.limit != (1 if args.commissioning_lite else 3)):
                raise ValueError("First physical contact run must use --commissioning --single-step (exactly 3 planned points)")
        if task == "visual" and not args.commissioning and not (session.progress["commissioning_passed"] or session.progress.get("visual_commissioning_passed")):
            raise ValueError("Visual session requires first 8 single-step --commissioning points and matching --verified-session")
    else:
        prior = None
        if args.commissioning_lite:
            from .commissioning import require_prior_session
            prior = require_prior_session(cfg, args.verified_session, simulated=True)
    completed_count = 0
    attempts_count = 0
    backend = None
    runner = None
    try:
        backend = load_hardware(cfg) if args.real else SimBackend(cfg, T0)
        if args.real:
            backend.prepare_zero(session, confirm=input)
        if hasattr(backend, "provenance"):
            session.meta.setdefault("hardware_runs", []).append({
                "started_monotonic_ns":backend.monotonic_ns(), "task":key,
                "provenance":backend.provenance()})
            session.save_meta()
        if args.simulate:
            backend.now_ns = max(backend.now_ns, session.meta.get("simulated_clock_last_ns", 0)+1)
            backend.drain_samples()
            backend.step(0.)
        runner = Runner(session, backend, single_step=args.single_step, unattended=args.simulate)
        runner.check_bottle("run_start")
        for item in pending:
            if args.limit and attempts_count >= args.limit:
                break
            ident = item.get("trial_id",item.get("point_id"))
            if not item["valid"]:
                session.start_trial(ident)
                session.complete({"trial_id":ident,"task":task,"bottle":item["bottle"],"kind":item["kind"],
                    "status":"skipped_safety","skip_reason":item["safety_reasons"],"stages":[],"auto":{},"operator":{}})
                # A skipped candidate counts towards the commissioning cap.
                attempts_count += int(args.commissioning or args.commissioning_lite)
                continue
            previous = session.progress.get("last_bottle")
            if previous != item["bottle"]:
                runner.ask(f"请放置 {item['bottle']} 瓶到同一标记位置，然后按回车；T0和参考图保持原样：")
                runner.check_bottle("bottle_switch")
                # Persist only AFTER operator acknowledgement and the fixed-
                # reference position check. Skipped plan rows never set this.
                session.progress["last_bottle"] = item["bottle"]
                session.save_progress()
            if task == "contact":
                result = runner.contact_trial(item, scenario=scenarios[completed_count % len(scenarios)])
                completed_count += 1
                cumulative = sum(1 for ident in session.progress["completed"]
                    if ident.startswith(("trial_", "commission_")) and
                    json.loads((session.path/"trials"/ident/"meta.json").read_text()).get("status") == "completed")
                if cumulative and cumulative % cfg["design"].get("baseline_every",30) == 0:
                    runner.baseline()
            else:
                result = runner.visual_point(item)
                completed_count += 1
            attempts_count += 1
            print(f"{ident}: {result['status']} {result.get('auto',{})}")
        if task == "visual" and completed_count:
            runner.check_bottle("visual_end")
        if args.commissioning or args.commissioning_lite:
            from .commissioning import accept_commissioning
            records = [json.loads((session.path/"trials"/x.get("trial_id",x.get("point_id"))/"meta.json").read_text()) for x in plan
                       if (session.path/"trials"/x.get("trial_id",x.get("point_id"))/"meta.json").exists()]
            accept_commissioning(session, records, level="visual" if task == "visual" else "lite" if args.commissioning_lite else "full", prior=prior)
    except BaseException as exc:
        # Includes startup checks, bottle changes and baseline failures outside
        # contact_trial(). Closing a transport is never treated as an arm stop.
        if backend is not None and runner is not None and getattr(runner, "actuation_started", False):
            try:
                backend.stop_arm()
            finally:
                if session.progress.get("requires_operator_recovery") or (runner and runner.airborne):
                    session.progress["requires_operator_recovery"] = True
                    session.save_progress()
                    backend.hold_gripper()
        atomic_json(session.path/"last_run_error.json",{"error":str(exc),"stage":None if runner is None else runner.stage,
            "requires_operator_recovery":session.progress.get("requires_operator_recovery",False)})
        raise
    finally:
        if args.simulate and backend is not None:
            session.meta["simulated_clock_last_ns"] = backend.monotonic_ns()
            session.save_meta()
        if backend is not None:
            backend.close()
    return 0


def entry(task):
    try:
        return main(task)
    except (ValueError, RuntimeError, OSError, KeyboardInterrupt) as exc:
        print(f"STOP: {exc}", file=sys.stderr)
        return 2
