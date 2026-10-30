"""Supervise the unchanged, complete existing force CLI on the 5090 host.

Import is inert. ``--host-supervisor`` exposes a private stdio JSON protocol to
the configured local/SSH executor. It never starts a controller until ``start``.
The experiment remains responsible for contact search, releasing its publisher,
reacquiring ownership and explicitly holding measured opening after termination.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path
import queue
import signal
import subprocess
import sys
import threading
import time
from urllib.parse import urlsplit
import xmlrpc.client

from .force_control import force_unit, force_value, target_definition, target_as_mean, validate_control_target


CLI_OVERRIDES = ("position_min", "position_max", "settle_time", "tactile_timeout",
                 "position_timeout", "max_run_time", "max_pair_skew", "startup_timeout", "step")


def validate_existing_settings(cfg):
    options = cfg["force"].get("existing")
    if not isinstance(options, dict):
        raise ValueError("existing controller requires an explicit force.existing deployment configuration")
    for key in ("source_path", "source_sha256", "calibration_path", "calibration_sha256", "host_python", "report_dir"):
        if not isinstance(options.get(key), str) or not options[key] or "TODO" in options[key].upper():
            raise ValueError("Unverified existing controller parameter: " + key)
    for key in ("source_sha256", "calibration_sha256"):
        if len(options[key]) != 64 or any(c not in "0123456789abcdef" for c in options[key]):
            raise ValueError(key + " must be the measured lowercase SHA256")
    argv = options.get("host_executor_argv")
    if not isinstance(argv, list) or not argv or not all(isinstance(a, str) and a and "TODO" not in a.upper() for a in argv):
        raise ValueError("Configure the verified host supervisor argv; container Docker is not assumed")
    for key in ("startup_timeout_s", "shutdown_timeout_s", "poll_s", "exit_stable_s", "exit_velocity_tolerance_mm_s"):
        value = options.get(key)
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value <= 0:
            raise ValueError("Positive finite force.existing." + key + " required")
    overrides = options.get("cli_overrides", {})
    if set(overrides) != set(CLI_OVERRIDES):
        raise ValueError("existing cli_overrides must explicitly specify " + ", ".join(CLI_OVERRIDES))
    if any(not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v) or v <= 0 for v in overrides.values()):
        raise ValueError("existing CLI limits must be measured finite positive values")
    if not 300 <= overrides["position_min"] < overrides["position_max"] <= 1000:
        raise ValueError("Existing controller's unchanged 300..1000 operating bounds must be preserved")
    if int(overrides["step"]) != overrides["step"]:
        raise ValueError("Existing step must be an integer")
    limits = options.get("control_limits", {})
    if limits.get("signal") != "SDK_1.2.11_saved_reference_39010":
        raise ValueError("existing.control_limits.signal must identify the original 39010 SDK/reference source")
    for key in ("max", "contact_min", "target_tolerance"):
        value = limits.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError("Explicit positive existing.control_limits." + key + " required; 39020 thresholds cannot be reused implicitly")
    mean = target_as_mean(force_value(cfg, "target"), target_definition(cfg))
    tolerance = limits["target_tolerance"] / (2. if target_definition(cfg) == "sum" else 1.)
    if not limits["contact_min"] < mean-tolerance < mean+tolerance < limits["max"]:
        raise ValueError("Existing 39010 controller target band must fit its own contact/max limits")
    evaluation_probe = options.get("evaluation_probe_target")
    if isinstance(evaluation_probe, bool) or not isinstance(evaluation_probe, (int, float)) or not math.isfinite(evaluation_probe) or evaluation_probe <= 0:
        raise ValueError("existing.evaluation_probe_target requires a measured independent 39020 PROBE target")
    if target_definition(cfg) == "per_finger":
        raise ValueError("The existing controller regulates a mean, not independent per-finger targets; select mean or sum explicitly")
    if force_unit(cfg) == "N":
        if not isinstance(options.get("n_per_gf"), (int, float)) or options["n_per_gf"] <= 0:
            raise ValueError("N control requires explicit existing.n_per_gf")
        if not options.get("n_conversion_evidence") or "TODO" in str(options["n_conversion_evidence"]).upper():
            raise ValueError("N control requires the measured source-specific gf/N calibration evidence")
    hosts = options.get("ros_node_hosts")
    if not isinstance(hosts, list) or not hosts or not all(isinstance(x, str) and x for x in hosts):
        raise ValueError("Verified ROS worker API hosts are required to identify its exact node")
    uri = options.get("ros_master_uri", "")
    if urlsplit(uri).scheme != "http" or not urlsplit(uri).hostname:
        raise ValueError("existing.ros_master_uri must be the verified HTTP ROS master URL")
    return options


def build_existing_argv(cfg, target, csv_path):
    options = validate_existing_settings(cfg)
    mean_target = target_as_mean(target, target_definition(cfg))
    limits = options["control_limits"]
    tolerance = limits["target_tolerance"] / (2. if target_definition(cfg) == "sum" else 1.)
    if not limits["contact_min"] < mean_target-tolerance < mean_target+tolerance < limits["max"]:
        raise ValueError("Requested target band is outside original-source contact/max limits")
    scale = options["n_per_gf"] if force_unit(cfg) == "N" else 1.
    target_option = "--target-force" if force_unit(cfg) == "N" else "--target-sdk-force"
    argv = [options["host_python"], options["source_path"], "--calibration", options["calibration_path"],
            target_option, str(mean_target / scale), "--deadband", str(tolerance / scale),
            "--max-force", str(limits["max"] / scale),
            "--contact-min-force", str(limits["contact_min"] / scale),
            "--csv", str(csv_path), "--execute"]
    for key in CLI_OVERRIDES:
        value = options["cli_overrides"][key]
        argv.extend(["--" + key.replace("_", "-"), str(int(value) if key == "step" else value)])
    return argv


def find_source_workers(source_text, proc_root=Path("/proc")):
    """Match full original source in python -c argv, not a loose process-name grep.

    Fail if proc inventory cannot be read. Individual disappeared processes are
    normal; denied command lines cannot establish the required exit evidence.
    """
    workers = []
    encoded = source_text.encode()
    for entry in proc_root.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            args = (entry / "cmdline").read_bytes().split(b"\0")
        except FileNotFoundError:
            continue
        except PermissionError as exc:
            raise RuntimeError("Cannot verify host worker disappearance: " + str(entry)) from exc
        if not args or not Path(os.fsdecode(args[0])).name.startswith("python") or encoded not in args or b"--internal-worker" not in args:
            continue
        index = args.index(b"--internal-worker")
        if index + 1 >= len(args) or args[index+1] not in (b"ros_tag", b"tactile_tag"):
            continue
        try:
            status = (entry / "status").read_text()
            nspids = next((line.split()[1:] for line in status.splitlines() if line.startswith("NSpid:")), [])
            start_ticks = (entry / "stat").read_text().rsplit(")", 1)[1].split()[19]
        except FileNotFoundError:
            continue
        workers.append({"host_pid": int(entry.name), "namespace_pids": [int(p) for p in nspids],
                        "kind": args[index+1].decode(), "start_ticks": int(start_ticks)})
    return workers


class HostSupervisor:
    """Host-side owner. A stop timeout never pretends the old workers are gone."""
    def __init__(self, *, popen=subprocess.Popen, worker_scanner=find_source_workers,
                 monotonic=time.monotonic, sleep=time.sleep):
        self.process = None
        self.options = None
        self.source = None
        self.log = None
        self._popen, self._scanner = popen, worker_scanner
        self._clock, self._sleep = monotonic, sleep

    def start(self, cfg, target, csv_path):
        if self.process is not None:
            raise RuntimeError("Previous controller lifetime has not been closed")
        if Path("/.dockerenv").exists() or Path("/run/.containerenv").exists():
            raise RuntimeError("The unchanged controller requires the 5090 host with Docker, not the experiment container")
        self.options = validate_existing_settings(cfg)
        # The audited immutable CLI pins these endpoints internally. Changing
        # only the supervisor's discovery master must never bypass ownership.
        if self.options["ros_master_uri"].rstrip("/") != "http://192.168.9.77:11311":
            raise RuntimeError("Original immutable controller pins ROS master 192.168.9.77:11311")
        for key in ("source", "calibration"):
            raw = Path(self.options[key + "_path"]).read_bytes()
            if hashlib.sha256(raw).hexdigest() != self.options[key + "_sha256"]:
                raise RuntimeError("Existing " + key + " checksum changed; refusing controller startup")
        self.source = Path(self.options["source_path"]).read_text()
        if self._scanner(self.source):
            raise RuntimeError("Existing force workers remain; no new controller was started")
        cal = json.loads(Path(self.options["calibration_path"]).read_text())
        if cal.get("side") != cfg["hardware"]["arm_side"]:
            raise ValueError("Existing controller calibration selects another gripper")
        expected_serials = cfg["hardware"]["sensor_serials"]
        if [f["serial"] for f in cal["fingers"]] != [expected_serials[s] for s in ("left", "right")]:
            raise ValueError("Existing controller calibration does not match both recorded finger serials")
        for finger in cal["fingers"]:
            if hashlib.sha256(Path(finger["reference_image"]).read_bytes()).hexdigest() != finger["reference_sha256"]:
                raise ValueError("Existing SDK reference image checksum changed")
        path = Path(csv_path)
        if path.exists():
            raise FileExistsError("Controller CSV cannot overwrite evidence: " + str(path))
        path.parent.mkdir(parents=True, exist_ok=True)
        self.log = path.with_suffix(".process.log").open("x")
        argv = build_existing_argv(cfg, target, path)
        try:
            self.process = self._popen(argv, stdin=subprocess.DEVNULL, stdout=self.log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
        except BaseException:
            self.log.close()
            self.log = None
            raise
        return {"started": True, "host_pid": self.process.pid, "argv": argv,
                "source_sha256": self.options["source_sha256"], "csv_path": str(path)}

    def _owned_nodes(self, workers):
        from .command_ownership import _TimeoutTransport
        ros_workers = [w for w in workers if w["kind"] == "ros_tag"]
        if not ros_workers:
            return []
        timeout = self.options["poll_s"] + 1.
        with xmlrpc.client.ServerProxy(self.options["ros_master_uri"], transport=_TimeoutTransport(timeout)) as master:
            result = master.getSystemState("/bottle_existing_force_supervisor")
            if result[0] != 1:
                raise RuntimeError("Cannot verify existing controller ROS registration")
            topic = "/" + self.side + "_gripper_command"
            candidates = dict(result[2][0]).get(topic, [])
            owned = []
            for node in candidates:
                lookup = master.lookupNode("/bottle_existing_force_supervisor", node)
                if lookup[0] != 1 or urlsplit(lookup[2]).hostname not in self.options["ros_node_hosts"]:
                    continue
                with xmlrpc.client.ServerProxy(lookup[2], transport=_TimeoutTransport(timeout)) as api:
                    pid = api.getPid("/bottle_existing_force_supervisor")
                if pid[0] == 1 and any(pid[2] in worker["namespace_pids"] for worker in ros_workers):
                    owned.append(node)
        if len(owned) > 1:
            raise RuntimeError("More than one ROS node matches the supervised force worker")
        return owned

    def status(self):
        workers = self._scanner(self.source) if self.source is not None else []
        return {"running": self.process is not None and self.process.poll() is None,
                "returncode": None if self.process is None else self.process.poll(),
                "workers": workers, "workers_gone": not workers,
                "owned_nodes": self._owned_nodes(workers) if self.process is not None and workers else []}

    def stop(self):
        if self.process is None:
            return {"running": False, "workers_gone": True, "workers": [], "owned_nodes": []}
        if self.process.poll() is None:
            self.process.send_signal(signal.SIGINT)
        deadline = self._clock() + self.options["shutdown_timeout_s"]
        while self._clock() < deadline:
            report = self.status()
            if not report["running"] and report["workers_gone"]:
                self.process.wait(timeout=self.options["poll_s"])
                self.process = None
                self.log.close()
                self.log = None
                return report
            self._sleep(self.options["poll_s"])
        raise RuntimeError("Existing controller/worker exit not verified; publisher must NOT be reacquired")

    def dispatch(self, request):
        op = request["op"]
        if op == "start":
            self.side = request["cfg"]["hardware"]["arm_side"]
            return self.start(request["cfg"], request["target"], request["csv_path"])
        if op == "status":
            return self.status()
        if op in ("stop", "shutdown"):
            return self.stop()
        raise ValueError("Unknown host supervisor operation")


class HostTransport:
    """Private, bounded stdio transport; no shell interpolation or credentials."""
    def __init__(self, cfg, *, popen=subprocess.Popen):
        options = validate_existing_settings(cfg)
        live_master = os.environ.get("ROS_MASTER_URI", "").rstrip("/")
        if live_master != options["ros_master_uri"].rstrip("/"):
            raise RuntimeError("Existing supervisor and experiment must inspect the same explicit ROS_MASTER_URI")
        self.timeout = max(options["startup_timeout_s"], options["shutdown_timeout_s"]) + options["poll_s"]
        self.process = popen(options["host_executor_argv"], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                             stderr=None, text=True, bufsize=1)
        self.replies = queue.Queue(maxsize=4)
        self._lock = threading.Lock()
        def reader():
            try:
                for line in self.process.stdout:
                    self.replies.put_nowait(json.loads(line))
            except BaseException as exc:
                try: self.replies.put_nowait({"ok": False, "error": str(exc)})
                except queue.Full: pass
            finally:
                try: self.replies.put_nowait({"ok": False, "error": "Host supervisor disconnected"})
                except queue.Full: pass
        threading.Thread(target=reader, daemon=True, name="existing_force_supervisor_reader").start()

    def request(self, op, **payload):
        with self._lock:
            self.process.stdin.write(json.dumps({"op": op, **payload}, allow_nan=False) + "\n")
            self.process.stdin.flush()
            try:
                reply = self.replies.get(timeout=self.timeout)
            except queue.Empty as exc:
                raise RuntimeError("Host supervisor response timeout; controller state unknown") from exc
            if not reply.get("ok"):
                raise RuntimeError("Host supervisor: " + str(reply.get("error")))
            return reply["result"]

    def close(self):
        self.request("shutdown")
        self.process.stdin.close()
        self.process.wait(timeout=self.timeout)


class ExistingForceController:
    """Explicit one-owner handoff around the unchanged original controller."""
    def __init__(self, cfg, gripper, force_reader, *, ownership_check, transport_factory=HostTransport,
                 monotonic=time.monotonic, sleep=time.sleep):
        self.cfg, self.gripper, self.force_reader = cfg, gripper, force_reader
        self.options = validate_existing_settings(cfg)
        self.ownership_check = ownership_check
        self.transport_factory = transport_factory
        self.transport = None
        self.active = False
        self.fault = None
        self.target = 0.
        self.owned_nodes = ()
        self.events = []
        self._clock, self._sleep = monotonic, sleep
        self._monitor_stop = threading.Event()
        self._monitor = None

    def _fresh(self):
        state = self.gripper.snapshot()
        if not state.get("fresh") or state.get("width_mm") is None:
            raise RuntimeError("Existing-controller handoff needs fresh calibrated opening")
        force = self.force_reader()
        values = force.get("normal", force.get("normal_N"))
        if values is None or len(values) != 2 or any(not math.isfinite(float(x)) for x in values):
            raise RuntimeError("Existing-controller handoff needs fresh two-finger force")
        return state, values

    def set_target(self, target):
        if self.fault:
            raise RuntimeError("Existing controller fault latched: " + self.fault)
        target = float(target)
        if not math.isfinite(target) or target <= 0:
            raise ValueError("Positive finite existing-controller target required")
        validate_control_target(self.cfg, target)
        if self.active:
            self.exit_force_mode()  # No hidden hot update or concurrent CLI workers.
        measured, normal = self._fresh()
        from .hardware_gripper import WidthCalibration
        calibration = WidthCalibration(self.cfg["hardware"]["gripper_width_calibration"])
        floor = measured.get("min_width_mm")
        if floor is None or not math.isfinite(float(floor)):
            raise RuntimeError("Existing controller requires the current verified geometric width floor")
        effective_cfg = copy.deepcopy(self.cfg)
        limits = effective_cfg["force"]["existing"]["cli_overrides"]
        limits["position_min"] = max(float(limits["position_min"]), calibration.to_normalized(floor)*1000.)
        limits["position_max"] = min(float(limits["position_max"]), calibration.to_normalized(self.cfg["gripper"]["open_mm"])*1000.)
        validate_existing_settings(effective_cfg)
        if min(normal) <= force_value(self.cfg, "contact_min"):
            raise RuntimeError("Existing controller starts only after verified bilateral contact")
        self.ownership_check(())
        self.gripper.suspend_publication()
        try:
            self.ownership_check(())
            self.transport = self.transport_factory(self.cfg)
            csv_path = str(Path(self.options["report_dir"]) / ("force_" + str(time.time_ns()) + ".csv"))
            started = self.transport.request("start", cfg=effective_cfg, target=target, csv_path=csv_path)
            self.events.append({"event": "start", **started})
            deadline = self._clock() + self.options["startup_timeout_s"]
            while self._clock() < deadline:
                state = self.transport.request("status")
                if not state["running"]:
                    raise RuntimeError("Existing controller exited during startup: " + str(state))
                if state.get("owned_nodes") and {w["kind"] for w in state["workers"]} == {"ros_tag", "tactile_tag"}:
                    self.owned_nodes = tuple(state["owned_nodes"])
                    self.ownership_check(self.owned_nodes)
                    self.active, self.target = True, target
                    self._monitor_stop.clear()
                    self._monitor = threading.Thread(target=self._monitor_loop, daemon=True,
                                                     name="existing_force_health_monitor")
                    self._monitor.start()
                    return
                self._sleep(self.options["poll_s"])
            raise RuntimeError("Existing controller startup/worker ownership not verified")
        except BaseException as exc:
            self.fault = str(exc)
            # Exit/worker proof is mandatory before any bridge publisher returns.
            # A failed cleanup remains suspended and fault-locked; never release.
            try:
                self._stop_and_reacquire()
            except BaseException as cleanup:
                self.fault += "; cleanup: " + str(cleanup)
            raise

    def check(self):
        if self.fault:
            raise RuntimeError(self.fault)
        return self.owned_nodes

    def _monitor_loop(self):
        while not self._monitor_stop.wait(self.options["poll_s"]):
            try:
                state = self.transport.request("status")
                if not state["running"] or not state.get("owned_nodes"):
                    raise RuntimeError("Existing controller/verified ROS worker stopped unexpectedly")
                self.owned_nodes = tuple(state["owned_nodes"])
            except BaseException as exc:
                self.fault = "Existing controller health failure: " + str(exc)
                return

    def _stop_monitor(self):
        self._monitor_stop.set()
        if self._monitor is not None:
            self._monitor.join(timeout=self.options["shutdown_timeout_s"])
            if self._monitor.is_alive():
                raise RuntimeError("Existing controller status monitor did not stop; handoff refused")
            self._monitor = None

    def _stop_and_reacquire(self):
        self._stop_monitor()
        if self.transport is not None:
            report = self.transport.request("stop")
            self.events.append({"event": "stop", **report})
            if report.get("running") or report.get("workers_gone") is not True:
                raise RuntimeError("Cannot reacquire command ownership before all workers exit")
            self.transport.close()
            self.transport = None
        self.active, self.target, self.owned_nodes = False, 0., ()
        self.ownership_check(())
        before, _ = self._fresh()
        self.gripper.resume_publication(timeout_s=self.options["startup_timeout_s"], poll_s=self.options["poll_s"])
        self.gripper.hold_gripper()
        deadline = self._clock()+self.options["shutdown_timeout_s"]
        stable = None
        last_sequence = before.get("source_sequence")
        if last_sequence is None:
            raise RuntimeError("Exit confirmation needs advancing measured opening source sequence")
        while True:
            after, _ = self._fresh()
            velocity = after.get("velocity_mm_s")
            advanced = after.get("source_sequence") is not None and after["source_sequence"]>last_sequence
            quiet = (velocity is not None and math.isfinite(float(velocity)) and
                     abs(float(velocity))<=self.options["exit_velocity_tolerance_mm_s"] and
                     abs(after["width_mm"]-before["width_mm"])<=self.cfg["gripper"]["open_tolerance_mm"])
            if advanced:
                last_sequence=after["source_sequence"]
                stable = self._clock() if stable is None and quiet else stable if quiet else None
                if stable is not None and self._clock()-stable>=self.options["exit_stable_s"]:
                    break
            if self._clock()>=deadline:
                raise RuntimeError("Existing-controller exit measured-width/velocity confirmation failed")
            self._sleep(self.options["poll_s"])
        keys = ("width_mm", "normalized", "monotonic_ns", "source_sequence")
        self.events.append({"event": "reacquired_and_measured_hold",
                           "before": {k:before.get(k) for k in keys}, "after": {k:after.get(k) for k in keys}})

    def exit_force_mode(self):
        if self.transport is None and not self.active:
            return self.gripper.hold_gripper()
        try:
            self._stop_and_reacquire()
        except BaseException as exc:
            self.fault = "Existing-controller exit/handoff failed: " + str(exc)
            raise

    def close(self):
        # Transport cleanup must terminate old regulation but never implicitly
        # reopen/reacquire/hold. Runner performs explicit stop/handoff beforehand.
        self._stop_monitor()
        if self.transport is not None:
            self.transport.close()
            self.transport = None
        self.active = False


def host_main():
    supervisor = HostSupervisor()
    try:
        for line in sys.stdin:
            try:
                request = json.loads(line)
                response = {"ok": True, "result": supervisor.dispatch(request)}
            except BaseException as exc:
                response = {"ok": False, "error": type(exc).__name__ + ": " + str(exc)}
            print(json.dumps(response, allow_nan=False), flush=True)
    finally:
        supervisor.stop()
    return 0


if __name__ == "__main__":
    if sys.argv[1:] != ["--host-supervisor"]:
        raise SystemExit("Use --host-supervisor only through the configured private host executor")
    raise SystemExit(host_main())
