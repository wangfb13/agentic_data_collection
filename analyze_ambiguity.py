#!/usr/bin/env python3
"""Offline-only contact ambiguity diagnostic entry point."""
from bottle_scan.ambiguity_analysis import main

if __name__ == '__main__':
    raise SystemExit(main())
