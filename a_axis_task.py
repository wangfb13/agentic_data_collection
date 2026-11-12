"""Explicit a-axis task entry. Importing does not connect to hardware."""
from bottle_scan.a_axis_task import main

if __name__ == "__main__":
    raise SystemExit(main())
