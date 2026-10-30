#!/usr/bin/env python3
"""Real-wall-clock disk pressure tests using synthetic data only; never ROS."""
from __future__ import annotations
import argparse
import json
import platform
from pathlib import Path
import threading
import time

import numpy as np
import cv2

from bottle_scan.config import load_config
from bottle_scan.recording import AsyncRecorder,validate_record_stress_settings
from bottle_scan.session import atomic_json


def _stats(values):
    array=np.asarray(values,dtype=float)
    if not len(values):return {"count":0,"mean_s":None,"p95_s":None,"max_s":None}
    return {"count":len(values),"mean_s":float(array.mean()),"p95_s":float(np.percentile(array,95)),"max_s":float(array.max())}


def _case(path,cfg,*,pressure):
    options=cfg["recording"]["stress"]
    period=1/options["control_hz"]
    durations={name:[] for name in ("state_machine","arm_interpolation","force_control")}
    submit=[];errors=[];stop=threading.Event();started=threading.Event();counts={"camera_frames":0,"dd_frames":0,"wrench_frames":0,"payload_bytes":0}
    writer=AsyncRecorder(path,cfg,_test_options={"delay_s":options["write_delay_s"]}) if pressure else None
    rng=np.random.default_rng(421)
    image=rng.integers(0,255,tuple(options["image_shape"]),dtype=np.uint8)
    # The public bridge queues source JPEG bytes verbatim; decoding/validation
    # runs inside JpegWriter in the child. Encode synthetic source bytes before
    # starting the wall-clock experiment so production API costs are matched.
    ok,encoded=cv2.imencode(".jpg",image,[cv2.IMWRITE_JPEG_QUALITY,95])
    if not ok:raise RuntimeError("Synthetic camera encoding failed")
    image=encoded.tobytes()
    dd=rng.normal(size=(96,128,3)).astype(np.float32)
    dd_encoded=rng.integers(0,65535,size=(96,128,3),dtype=np.uint16)
    def control(name):
        started.wait();last=time.monotonic();next_tick=last+period;count=0
        while not stop.wait(max(0.,next_tick-time.monotonic())):
            now=time.monotonic();durations[name].append(now-last);last=now;count+=1
            # Stage metadata submissions share the real asynchronous API used
            # by the state machine; no virtual clock advances these loops.
            if writer and name=="state_machine" and count%10==0:
                before=time.monotonic()
                try:writer.json("stress_meta.json",{"count":count,"stage":"HOLD"})
                except Exception as exc:errors.append(str(exc))
                submit.append(time.monotonic()-before)
            next_tick+=period
            if next_tick<now-period:next_tick=now+period
    def producer(kind):
        frequency=options["producer_hz"] if kind=="camera" else options.get("dd_hz",27.) if kind=="dd" else options.get("force_hz",30.)
        started.wait();period=1/frequency
        while not stop.is_set():
            before=time.monotonic();stamp=time.monotonic_ns()
            if writer:
                try:
                    for side in ("left","right"):
                        if kind=="camera":
                            writer.jpg.append(side+"_wrist",image,stamp)
                            counts["camera_frames"]+=1;counts["payload_bytes"]+=len(image)
                        elif kind=="dd":
                            writer.h5.append("dd_"+side,dd,stamp)
                            writer.h5.append("dd_encoded_"+side,dd_encoded,stamp)
                            counts["dd_frames"]+=1;counts["payload_bytes"]+=dd.nbytes+dd_encoded.nbytes
                        else:
                            wrench=np.array([1.,2.,3.,.1,.2,.3])
                            writer.h5.append("wrench_raw_"+side,wrench,stamp)
                            counts["wrench_frames"]+=1;counts["payload_bytes"]+=wrench.nbytes
                except Exception as exc:errors.append(str(exc));return
                submit.append(time.monotonic()-before)
            stop.wait(max(0.,period-(time.monotonic()-before)))
    threads=[threading.Thread(target=control,args=(name,),daemon=True) for name in durations]
    threads.extend(threading.Thread(target=producer,args=(kind,),daemon=True) for kind in ("camera","dd","force"))
    for thread in threads:thread.start()
    clock_start=time.monotonic();started.set();time.sleep(options["duration_s"]);stop.set()
    for thread in threads:thread.join(timeout=2)
    elapsed=time.monotonic()-clock_start
    result={"loops":{name:_stats(values) for name,values in durations.items()},"errors":errors,
            "submission_s":_stats(submit) if submit else None,"production_wall_s":elapsed,
            "submitted":counts,"submitted_payload_MB_s":counts["payload_bytes"]/elapsed/1e6,
            "camera_payload":"two original JPEG byte streams; random synthetic image, quality 95",
            "camera_bytes_per_frame":len(image),"dd_payload":"two decoded float32 plus two encoded uint16 streams, 96x128x3",
            "force_payload":"two six-axis float64 raw streams"}
    if writer:
        result["writer"]=writer.stats()
        try:writer.close()
        except Exception as exc:errors.append(str(exc))
    return result


def run_stress(cfg,output):
    errors=validate_record_stress_settings(cfg)
    if errors:raise ValueError("; ".join(errors))
    options=cfg["recording"]["stress"]
    output=Path(output)
    output.mkdir(parents=True,exist_ok=False)
    baseline=_case(output/"baseline",cfg,pressure=False)
    pressure=_case(output/"pressure",cfg,pressure=True)
    comparison={name:{"baseline_p95_s":baseline["loops"][name]["p95_s"],
                      "pressure_p95_s":pressure["loops"][name]["p95_s"],
                      "added_p95_s":(pressure["loops"][name]["p95_s"]-baseline["loops"][name]["p95_s"])
                        if pressure["loops"][name]["count"] and baseline["loops"][name]["count"] else None}
                for name in baseline["loops"]}
    loop_passed=all(x["added_p95_s"] is not None and x["added_p95_s"]<=options["max_added_p95_s"] for x in comparison.values())
    capacity_passed=not pressure["errors"]
    passed=capacity_passed and loop_passed
    report={"schema":"bottle_writer_wall_clock_stress_v1","simulated_data":True,"wall_clock":"time.monotonic",
            "host":platform.node(),"platform":platform.platform(),"created_unix_ns":time.time_ns(),
            "settings":options,"baseline":baseline,"pressure":pressure,"comparison":comparison,"passed":passed,
            "loop_period_bound_passed":loop_passed,"recording_capacity_passed":capacity_passed,
            "scope":"Synthetic timer loops and real asynchronous recording API measured on this host only; checks configured added-p95 bound, not deployed ROS controllers, hard-real-time or zero-jitter guarantee. Existing donor controller synchronous CSV remains outside this test.",
            "separate_failure_tests":"tests/test_async_recording.py covers byte queue full, disk exception and writer SIGKILL"}
    atomic_json(output/"report.json",report)
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",default=str(Path(__file__).with_name("config.ros.yaml")))
    parser.add_argument("--output",required=True)
    args=parser.parse_args(argv)
    # The stress tool has no hardware code path. Read the requested recording
    # settings verbatim so an operator's queue/load edits are not discarded.
    cfg=load_config(args.config,demo=False)
    report=run_stress(cfg,args.output)
    print(json.dumps({"passed":report["passed"],"comparison":report["comparison"],
                      "report":str(Path(args.output)/"report.json")},indent=2))
    return 0 if report["passed"] else 2


if __name__=="__main__":raise SystemExit(main())
