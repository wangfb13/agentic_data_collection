#!/usr/bin/env python3
"""Measure software stop / SIGKILL behaviour; dry-run is the default."""
import argparse
import json
from pathlib import Path

import numpy as np

from bottle_scan.config import load_config, require_real_ready
from bottle_scan.stop_validation import run_stop_test, stop_path, stop_test_defaults


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    modes=parser.add_mutually_exclusive_group()
    modes.add_argument("--real",action="store_true")
    modes.add_argument("--simulate",action="store_true")
    modes.add_argument("--dry-run",action="store_true")
    parser.add_argument("--demo",action="store_true")
    parser.add_argument("--config",default=str(Path(__file__).with_name("config.ros.yaml")))
    parser.add_argument("--output")
    args=parser.parse_args(argv)
    cfg=load_config(args.config,demo=args.demo or args.simulate)
    if "stop_test" not in cfg and cfg.get("demo"):cfg["stop_test"]=stop_test_defaults(demo=True)
    preview=stop_path(cfg,np.asarray(cfg["preparation"]["axis_test_pose"]),cfg["gripper"]["open_mm"])
    print(json.dumps({"test":"software and SIGKILL","repeats_each":cfg["stop_test"]["repeats"],
                      "simulated":bool(args.simulate),"nominal_preview_only":True,**preview},ensure_ascii=False,indent=2))
    if not (args.real or args.simulate):return 0
    if args.real:require_real_ready(cfg,preparing=True)
    output=args.output or cfg["stop_test"]["report_path"]
    report=run_stop_test(cfg,simulate=args.simulate,output_path=output)
    print("停止测试报告："+str(Path(output).resolve()))
    return 0 if report["passed"] else 2


if __name__=="__main__":
    raise SystemExit(main())
