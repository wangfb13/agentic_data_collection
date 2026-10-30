#!/usr/bin/env python3
"""Operator-started force repeatability validation at measured T0 (no lift)."""
from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path
import sys
import time

import h5py
import numpy as np

from bottle_scan.config import load_config, require_real_ready
from bottle_scan.force_control import (force_unit, force_value, target_definition,
    force_report_binding, summarize_force_cycle, evaluate_force_report, close_quality_report)
from bottle_scan.geometry import pose_error
from bottle_scan.session import Session, atomic_json, plain


def observations_from_h5(path, unit):
    """Read committed source events, not repeated state-machine snapshots."""
    with h5py.File(path, "r") as handle:
        root = handle["streams"]
        name = "normal_force_" + unit
        if name not in root:
            raise ValueError("Missing unit-labelled recorded force stream: " + name)
        group = root[name]
        count = int(group.attrs["committed_count"])
        stamps = group["monotonic_ns"][:count]
        data = group["data"][:count]
        metadata = [json.loads(s) for s in group["metadata_json"][:count]]
    if not count:
        raise ValueError("No committed force samples")
    return [{"t_s": (int(stamp) - int(stamps[0])) / 1e9, "phase": meta.get("stage", "UNKNOWN"),
             "normal": np.asarray(value).tolist(), "source_metadata": meta}
            for stamp, value, meta in zip(stamps, data, metadata)]


def run_cycle(runner, folder, index):
    """Reuse guarded CLOSE/PROBE/SQUEEZE/HOLD/RELEASE; never move either arm."""
    from bottle_scan.runtime import Acquisition, SafetyFault
    cfg, backend = runner.cfg, runner.backend
    ident = "force_validation_%03d" % index
    item = {"trial_id": ident, "T_target": runner.session.T0.tolist(),
            "nominal_contact_region": "body", "delta_acl_mm": [0, 0, 0]}
    runner.meta = {"trial_id": ident, "task": "force_validation", "status": "running",
        "T_target": item["T_target"], "stages": [], "simulated": backend.simulated,
        "auto": {"contact_times_ns": [None, None], "contact_widths_mm": [None, None],
                 "dropped": False, "slip_suspected": False, "exceptions": []}}
    backend.set_trial(item)
    backend.drain_samples()
    runner.recorder = Acquisition(backend, folder)
    cycle_actuated = False
    try:
        with runner.phase("RELEASE"):
            cycle_actuated = True
            runner.ensure_open()
        if not runner.close_contact():
            raise SafetyFault("Force validation did not establish bilateral contact")
        if callable(getattr(runner, "probe", None)):
            runner.probe()
        else:
            with runner.phase("PROBE"):
                runner._command_force(cfg["force"].get("probe_N", cfg["force"]["probe_fraction"]*force_value(cfg, "target")))
                runner.wait(cfg["force"]["probe_s"])
        if not runner.squeeze():
            raise SafetyFault("Force validation failed to settle at target")
        with runner.phase("HOLD"):
            runner.wait(cfg["force"]["validation"]["hold_s"])
        with runner.phase("RELEASE"):
            runner.ensure_open()
        runner.meta["status"] = "completed"
    except BaseException as exc:
        runner.meta["status"] = "aborted"
        runner.meta["error"] = str(exc)
        # No automatic release on validation failure. Complete controller exit
        # and explicit measured hold preserve evidence; any failed handoff stays
        # locked. No arm motion was started by this tool.
        if cycle_actuated and runner.actuation_started:
            try:
                backend.hold_gripper()
            except BaseException as stop_error:
                runner.meta["stop_error"] = str(stop_error)
        raise
    finally:
        runner.recorder.close()
        runner.recorder = None
        if (folder / "streams.h5").exists():
            from bottle_scan.window_descriptors import enrich_trial_windows
            from bottle_scan.recording import finalize_gel_windows
            enrich_trial_windows(runner.meta, folder / "streams.h5", cfg, runner.reference)
            finalize_gel_windows(folder, runner.meta, cfg)
        atomic_json(folder / "meta.json", runner.meta)
    samples = observations_from_h5(folder / "streams.h5", force_unit(cfg))
    atomic_json(folder / "force_samples.json", samples)
    metrics = summarize_force_cycle(samples, cfg)
    return {"trial_id": ident, "status": runner.meta["status"], **metrics,
            "source": "committed native paired force events", "data_path": str(folder)}


def comparison_rows(reports):
    rows = []
    for report in reports:
        unit = report["unit"]
        for trial in report["trials"]:
            row = {"controller": report["controller"], "simulated": report["simulated"],
                   "evaluation_signal": report["evaluation_signal"], "control_signal": report["control_signal"],
                   "unit": unit, "target_definition": report["target_definition"],
                   "trial_id": trial["trial_id"], "status": trial["status"],
                   "target_" + unit: trial["target"], "time_to_target_band_s": trial["time_to_target_band_s"]}
            for side, statistics in trial["statistics"].items():
                row.update({side + "_" + key + "_" + unit: value for key, value in statistics.items()})
            rows.append(row)
    return rows


def write_comparison(reports, path):
    rows = comparison_rows(reports)
    keys = list(dict.fromkeys(k for row in rows for k in row))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--simulate", action="store_true")
    modes.add_argument("--real", action="store_true")
    modes.add_argument("--dry-run", action="store_true")
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--config", default="config.ros.yaml")
    parser.add_argument("--session")
    parser.add_argument("--controller", choices=("existing", "bridge_loop"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--compare", nargs="+", type=Path)
    args = parser.parse_args(argv)
    if args.compare:
        if args.real or args.simulate or args.session or args.output is None:
            raise ValueError("Offline --compare requires --output CSV and no execution/session flags")
        reports = [json.loads(p.read_text()) for p in args.compare]
        rows = write_comparison(reports, args.output)
        print("不同控制来源的 SDK 数值不自动等价；请按 evaluation_signal/unit 分组对比。")
        print(json.dumps(rows, ensure_ascii=False, indent=2))
        return 0
    if args.real and args.demo:
        raise ValueError("Synthetic config cannot execute hardware")
    session = Session(args.session) if args.session else None
    cfg = copy.deepcopy(session.cfg if session else load_config(args.config, demo=args.demo or args.simulate))
    if args.controller:
        cfg["force"]["controller"] = args.controller
    if session is not None and bool(session.meta["simulated"]) != bool(args.simulate):
        raise ValueError("Cannot mix synthetic/real validation and a prepared session")
    options = cfg["force"]["validation"]
    repetitions = int(options["repetitions"])
    if repetitions <= 0 or repetitions != options["repetitions"]:
        raise ValueError("force.validation.repetitions must be a positive integer")
    summary = {"controller": cfg["force"]["controller"], "unit": force_unit(cfg),
               "target_definition": target_definition(cfg), "target": force_value(cfg, "target"),
               "repetitions": repetitions, "arm_motion": "none; already measured at T0 required",
               "sequence": ["OPEN", "CLOSE", "PROBE", "SQUEEZE", "HOLD", "RELEASE"],
               "simulated": bool(args.simulate), "hold_s": options["hold_s"],
               "evaluation_target": options.get("evaluation_target"),
               "close_quality":close_quality_report(cfg)}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if summary["close_quality"]["errors"]:
        raise ValueError("; ".join(summary["close_quality"]["errors"]))
    if not (args.real or args.simulate):
        print("DRY-RUN：未连接硬件；existing 使用完整原控制器进程，bridge_loop 使用软件力环。")
        return 0
    destination = args.output or Path(options["report_path"])
    if destination.exists():
        raise FileExistsError("Choose a new evidence report; existing report retained: " + str(destination))
    run_dir = destination.parent / (destination.stem + "_data")
    run_dir.mkdir(parents=True, exist_ok=False)
    if session is None:
        if args.real:
            raise ValueError("Physical force validation requires --session from prepare")
        session = Session.create(run_dir / "synthetic_session", cfg, np.asarray(cfg["T0"]), simulated=True)
    else:
        session.cfg = cfg  # Validation config snapshot below records explicit controller selection.
    if args.real:
        require_real_ready(cfg)
        if session.progress.get("requires_operator_recovery"):
            raise ValueError("Operator recovery latch prevents validation")
    from bottle_scan.backend import SimBackend, load_hardware
    from bottle_scan.runtime import Runner
    backend = None
    report = {"schema": "bottle_force_validation_v2", **summary, "binding": force_report_binding(cfg),
              "operator_confirmation": None, "passed": False, "trials": [],
              "evaluation_signal": "synthetic" if args.simulate else "existing_X5_39020_after_guarded_software_zero",
              "control_signal": ("synthetic_controller_emulation" if args.simulate else
                                 "SDK_1.2.11_saved_reference_39010" if cfg["force"]["controller"] == "existing" else
                                 "existing_X5_39020_after_guarded_software_zero"),
              "note": "SDK values from different processing/reference sources are not interchangeable"}
    atomic_json(run_dir / "effective_config.json", cfg)
    try:
        backend = load_hardware(cfg) if args.real else SimBackend(cfg, session.T0)
        if args.real:
            backend.prepare_zero(session, confirm=input)
            from bottle_scan.park_guard import ParkGuard
            ParkGuard(session).check(backend.park_observation(), now_ns=backend.monotonic_ns())
        runner = Runner(session, backend, single_step=bool(args.real), unattended=bool(args.simulate))
        measured = runner._healthy_snapshot()["tcp"]
        error = pose_error(np.asarray(measured), session.T0)
        if error[0] > cfg["motion"]["pose_tolerance_mm"] or error[1] > cfg["motion"]["pose_tolerance_deg"]:
            raise ValueError("Move to measured T0 with move-tcp first; validation never moves either arm")
        if args.real and input("将在当前 T0 执行上述闭合/夹紧/保持/松开循环；输入 MOVE 开始：").strip() != "MOVE":
            raise ValueError("Operator cancelled before validation actuation")
        measured = runner._healthy_snapshot()["tcp"]
        error = pose_error(np.asarray(measured), session.T0)
        if error[0] > cfg["motion"]["pose_tolerance_mm"] or error[1] > cfg["motion"]["pose_tolerance_deg"]:
            raise ValueError("TCP changed during confirmation; no validation cycle started")
        for index in range(repetitions):
            folder = run_dir / ("cycle_%03d" % (index+1))
            folder.mkdir()
            report["trials"].append(run_cycle(runner, folder, index+1))
            atomic_json(destination, report)
        report["failures"] = evaluate_force_report(report, cfg)
        report["passed"] = not report["failures"]
        print(json.dumps(report["trials"], ensure_ascii=False, indent=2))
        if args.real and report["passed"]:
            if input("逐项核对力控验收摘要；输入 PASS 确认此控制器报告：").strip() == "PASS":
                report["operator_confirmation"] = "PASS"
        elif args.simulate:
            report["operator_confirmation"] = "SIMULATION_ONLY"
        write_comparison([report], destination.with_suffix(".csv"))
    except BaseException as exc:
        report["error"] = str(exc)
        report["passed"] = False
        raise
    finally:
        if backend is not None:
            report["hardware_provenance"] = plain(backend.provenance()) if hasattr(backend, "provenance") else {"simulated": True}
            backend.close()
        atomic_json(destination, report)
    print("Force validation report: " + str(destination))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, OSError, KeyboardInterrupt) as exc:
        print("STOP: " + str(exc), file=sys.stderr)
        raise SystemExit(2)
