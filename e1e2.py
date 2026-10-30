#!/usr/bin/env python3
"""Independent real E1/E2 entry point. Importing does not connect hardware."""
from bottle_scan.e1e2 import main
if __name__ == '__main__':
    raise SystemExit(main())
