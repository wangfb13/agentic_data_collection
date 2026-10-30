#!/usr/bin/env python3
"""Analyze a recorded or mock session without connecting hardware."""
import argparse
import json

from bottle_scan.analysis import analyze_session, analyze_sessions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session", nargs="+", help="One or more session directories containing session_meta.json")
    parser.add_argument("--output", help="Output directory (default: SESSION/analysis)")
    args = parser.parse_args()
    if len(args.session) > 1 and not args.output:
        parser.error("Multiple sessions require --output")
    report = (analyze_session(args.session[0], args.output) if len(args.session) == 1
              else analyze_sessions(args.session, args.output))
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
