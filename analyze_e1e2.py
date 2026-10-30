#!/usr/bin/env python3
"""Offline E1/E2 analysis; never imports a robot controller."""
from bottle_scan.e1e2_analysis import main
if __name__ == '__main__':
    raise SystemExit(main())
