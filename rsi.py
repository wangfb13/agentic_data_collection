#!/usr/bin/env python3
"""Recursive experiment improvement. Importing never connects to hardware."""
from bottle_scan.rsi import main

if __name__ == '__main__':
    raise SystemExit(main())
