#!/usr/bin/env python3
"""Offline end-to-end acceptance data. Cannot select or load a hardware backend."""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import numpy as np
from prepare import main as prepare
from bottle_scan.backend import SimBackend
from bottle_scan.design import _trial, visual_points
from bottle_scan.runtime import Runner
from bottle_scan.session import Session, atomic_json
from bottle_scan.analysis import analyze_session


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session",default="outputs/acceptance_demo")
    args=parser.parse_args()
    prepare(["--simulate","--small-demo","--session",args.session])
    session=Session(args.session)
    backend=SimBackend(session.cfg,session.T0)
    backend.now_ns=max(backend.now_ns,session.meta["simulated_clock_last_ns"]+1)
    backend.drain_samples()
    backend.step(0.)
    runner=Runner(session,backend,single_step=False,unattended=True)
    reference_hash=runner.guard.reference_sha256
    cases=[("normal",[0,0,0]),("no_contact",[-5,0,-10]),("stall",[5,0,0]),
           ("force_timeout",[0,0,10]),("dropped",[-5,0,0]),("slip",[5,0,10]),
           ("normal",[0,0,0]),("normal",[-5,0,-10])]
    records=[]
    try:
        runner.check_bottle("smoke_start")
        for i,(scenario,delta) in enumerate(cases):
            bottle="light" if i<4 else "heavy"
            if i==4:
                runner.check_bottle("same_position_heavy_bottle")
            item=_trial(session.cfg,session.T0,bottle,"smoke_"+scenario,delta)
            item["trial_id"]=f"trial_{i+1:04d}"
            assert item["valid"],item["safety_reasons"]
            result=runner.contact_trial(item,scenario=scenario)
            assert result["status"]=="completed",result
            if scenario=="normal":
                assert result["auto"]["bilateral_contact"] and result["auto"]["force_reached"]
            if scenario=="no_contact":
                assert not result["auto"]["bilateral_contact"]
                assert result["auto"].get("no_bilateral_contact") or result["auto"].get("safety_closure_limit")
            if scenario=="stall":
                assert result["auto"]["stall_without_contact"]
            if scenario=="force_timeout":
                assert result["auto"]["force_not_reached"]
            if scenario=="dropped":
                assert result["auto"]["dropped"]
            if scenario=="slip":
                assert result["auto"]["slip_suspected"]
            records.append({"scenario":scenario,"trial_id":item["trial_id"],"status":result["status"]})
            print(json.dumps(records[-1]),flush=True)
        calls_before=len(backend.calls)
        # The last synthetic contact block installed the heavy bottle. Encode
        # it in the visual plan itself instead of overriding recorded labels.
        visual_cfg=copy.deepcopy(session.cfg)
        visual_cfg["design"]["bottles"]=["heavy"]
        visual=[x for x in visual_points(visual_cfg,session.T0) if x["valid"]][:6]
        for item in visual:
            result=runner.visual_point(item)
            assert result["status"]=="completed",result
            assert result["bottle"]==item["bottle"]=="heavy",result
            np.testing.assert_allclose(result["label_B_in_tcp"],np.linalg.inv(result["T_measured"])@session.T0)
        assert not any(name in {"close","force"} for name,_ in backend.calls[calls_before:])
        assert reference_hash==runner.guard.reference_sha256
        session.meta["simulated_clock_last_ns"]=backend.monotonic_ns()
        session.save_meta()
        analysis=analyze_session(session.path)
        report={"simulated_only":True,"contact_cases":records,"visual_points":len(visual),
            "immutable_reference_verified":True,"visual_no_close_verified":True,"analysis":analysis}
        atomic_json(session.path/"acceptance.json",report)
        print(json.dumps(report,ensure_ascii=False,indent=2))
    finally:
        backend.close()


if __name__=="__main__":
    main()
