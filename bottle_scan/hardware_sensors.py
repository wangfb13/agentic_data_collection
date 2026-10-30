"""Passive native sensor transport and a thread-safe acquisition cache.

Importing this module opens no devices or sockets. The native wire format is an
ADDITIONAL lossless tap for the existing SDK owner; it is not the lossy 39030
format. SharedSensorHub accepts an existing owner/fanout's frames. The separately
named ActualX5SensorSource implements the user's explicit choice to consume the
existing PULL wrist-camera protocol and processed DD, with honest provenance.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import json
import hashlib
import math
import threading
import time

import numpy as np


class SensorFault(RuntimeError):
    pass


def inspect_wrench_pub(address="tcp://192.168.9.11:39020", *,
                       serials=("X2620410318", "X2620410347"), duration_s=5.0,
                       _context_factory=None, _clock=time.monotonic_ns):
    """Read-only, bounded diagnostic subscription to the existing wrench PUB.

    This function never opens a camera/PULL socket or imports/initializes an SDK.
    ``ready`` means both source serials produced data, NOT that units, calibration,
    delivery completeness, physical finger assignment or source clocks are verified.
    The private injection arguments support fake-I/O tests without real sockets.
    """
    if (not isinstance(address, str) or not address.startswith("tcp://") or
            not np.isfinite(duration_s) or not 0 < duration_s <= 5 or
            len(serials) != 2 or len(set(serials)) != 2 or not all(isinstance(s, str) and s for s in serials)):
        raise ValueError("Require explicit PUB endpoint, two serials and duration in (0,5] seconds")
    import zmq
    context = (_context_factory or zmq.Context)()
    socket = None
    report = {"endpoint": address, "transport": "ZMQ SUB of existing PUB", "ready": False,
              "native_dd_available": False, "physical_force_unit_verified": False,
              "wrench_unit": "unknown_sdk_raw", "components": ["Fx", "Fy", "Fz", "Mx", "My", "Mz"],
              "wire_dtype": "JSON numbers; not evidence of native SDK dtype", "malformed_messages": 0,
              "unexpected_serial_messages": 0, "sensors": {s:{"count":0, "shape":[6],
                  "sequence_gaps":0, "duplicate_or_reversed_fids":0, "session_changed":False,
                  "all_force_fids_verified":True} for s in serials}}
    start = _clock()
    try:
        socket = context.socket(zmq.SUB)
        socket.setsockopt(zmq.LINGER, 0)
        socket.setsockopt(zmq.RCVHWM, 2048)
        for serial in serials:
            socket.setsockopt(zmq.SUBSCRIBE, serial.encode())
        socket.connect(address)
        while _clock()-start < duration_s*1e9:
            remaining_ms = max(1, min(100, int((duration_s*1e9-(_clock()-start))/1e6)))
            if not socket.poll(remaining_ms, zmq.POLLIN):
                continue
            parts = socket.recv_multipart()
            received_ns = _clock()
            try:
                if len(parts) != 2:
                    raise ValueError("Wrong wrench multipart length")
                m = json.loads(parts[1])
                serial = m["serial"]
                if parts[0].decode() != serial or m["schema"] != "dm_x5_wrench_v1":
                    raise ValueError("Wrong topic/schema")
                if serial not in report["sensors"]:
                    report["unexpected_serial_messages"] += 1
                    continue
                raw = np.asarray(m["wrench"], dtype=np.float64)
                fid = m["fid"]
                if raw.shape != (6,) or not np.isfinite(raw).all() or type(fid) is not int or fid < 0:
                    raise ValueError("Invalid wrench/fid")
                source_timestamp = m.get("timestamp")
                if source_timestamp is not None and (not isinstance(source_timestamp,(int,float)) or not np.isfinite(source_timestamp)):
                    raise ValueError("Invalid source timestamp")
                info = report["sensors"][serial]
                if not info["count"]:
                    info.update(first_fid=fid, first_receive_monotonic_ns=received_ns,
                                source_session_id=m.get("session_id"), minimum=raw.tolist(), maximum=raw.tolist())
                else:
                    delta = fid-info["last_fid"]
                    info["sequence_gaps"] += max(0,delta-1)
                    info["duplicate_or_reversed_fids"] += int(delta <= 0)
                    info["session_changed"] |= info["source_session_id"] != m.get("session_id")
                    info["minimum"] = np.minimum(info["minimum"],raw).tolist()
                    info["maximum"] = np.maximum(info["maximum"],raw).tolist()
                info["count"] += 1
                info["all_force_fids_verified"] &= m.get("force_fid_verified") is True
                info.update(last_fid=fid, last_receive_monotonic_ns=received_ns, last_wrench=raw.tolist(),
                            source_timestamp=source_timestamp, source_reference_sha256=m.get("reference_sha256"),
                            timestamp_kind=m.get("timestamp_kind", "publisher_sdk_read_wall_time_not_hardware_exposure"))
            except (ValueError, KeyError, TypeError, UnicodeError):
                report["malformed_messages"] += 1
        report["ready"] = all(x["count"] >= 2 and x["last_fid"] > x["first_fid"]
                              for x in report["sensors"].values()) and not report["malformed_messages"]
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if socket is not None:
            socket.close(0)
        context.term()
    report["elapsed_s"] = (_clock()-start)/1e9
    for info in report["sensors"].values():
        elapsed = (info.get("last_receive_monotonic_ns",0)-info.get("first_receive_monotonic_ns",0))/1e9
        info["observed_receive_fps"] = (info["count"]-1)/elapsed if elapsed > 0 else None
        if not info["count"]:
            info["all_force_fids_verified"] = False
    return report


def _native_array(value, kind):
    a = np.asarray(value)
    if a.dtype.kind != "f" or not np.isfinite(a).all():
        raise SensorFault(f"{kind}: finite native floating-point array required")
    if kind == "depth" and not (a.ndim == 2 or a.ndim == 3 and a.shape[-1] == 1):
        raise SensorFault("depth: expected HxW or HxWx1")
    if kind == "flow" and not (a.ndim == 3 and a.shape[-1] == 2):
        raise SensorFault("flow: expected HxWx2")
    if kind == "wrench" and a.shape != (6,):
        raise SensorFault("wrench: expected six native SDK values")
    if not a.size:
        raise SensorFault(f"{kind}: empty array")
    return np.array(a, copy=True, order="C")


def encode_native_tactile(serial, depth, flow, wrench, metadata):
    """Called inside the sole SDK owner after copying aligned SDK outputs.

    No dtype casts, resizing, quantization, zeroing or SDK calls occur here.
    source_sequence is a contiguous emitted-frame counter, distinct from SDK fid.
    Metadata must state source read times and units honestly (unknown is allowed
    on the raw wire; control requires a separately verified calibration).
    """
    arrays = {k: _native_array(v, k) for k, v in
              (("depth", depth), ("flow", flow), ("wrench", wrench))}
    if arrays["depth"].shape[:2] != arrays["flow"].shape[:2]:
        raise SensorFault("Native depth/flow geometry mismatch")
    m = dict(metadata)
    for key in ("source_sequence", "fid", "depth_fid", "deformation_fid"):
        if type(m.get(key)) is not int or m[key] < 0:
            raise SensorFault(f"Missing nonnegative integer {key}")
    if m["fid"] != m["depth_fid"] or m["fid"] != m["deformation_fid"]:
        raise SensorFault("Depth/flow frame ids are not aligned")
    if type(m.get("force_fid_verified")) is not bool:
        raise SensorFault("Must state force_fid_verified explicitly")
    if m["force_fid_verified"] and m.get("force_fid") != m["fid"]:
        raise SensorFault("Tagged force does not match depth/flow frame")
    for key in ("session_id", "reference_sha256", "timestamp_kind", "depth_unit", "flow_unit", "wrench_unit"):
        if not isinstance(m.get(key), str) or not m[key].strip():
            raise SensorFault(f"Missing source metadata {key}")
    m.update(schema="bottle_native_tactile_v1", serial=str(serial), representation="sdk_native_arrays",
             arrays={k: {"shape": list(v.shape), "dtype": v.dtype.str, "bytes": v.nbytes}
                     for k, v in arrays.items()})
    return [str(serial).encode(), json.dumps(m, allow_nan=False).encode(),
            *(arrays[k].tobytes() for k in ("depth", "flow", "wrench"))]


def decode_native_tactile(parts, *, max_payload_bytes=64*1024*1024):
    if len(parts) != 5:
        raise SensorFault("Native tactile tap requires five multipart frames; 39030 is not native")
    try:
        m = json.loads(parts[1])
        if m["schema"] != "bottle_native_tactile_v1" or m["representation"] != "sdk_native_arrays":
            raise SensorFault("Not a verified native SDK wire representation")
        if parts[0].decode() != m["serial"]:
            raise SensorFault("Topic/serial mismatch")
        arrays = {}
        for kind, payload in zip(("depth", "flow", "wrench"), parts[2:]):
            spec = m["arrays"][kind]
            shape, dtype = spec["shape"], np.dtype(spec["dtype"])
            if not shape or any(type(x) is not int or x <= 0 for x in shape) or dtype.kind != "f":
                raise SensorFault("Unsafe array shape/dtype")
            size = int(np.prod(shape, dtype=object))*dtype.itemsize
            if not 0 < size <= max_payload_bytes or size != len(payload) or spec["bytes"] != size:
                raise SensorFault("Native payload length mismatch")
            arrays[kind] = np.frombuffer(payload, dtype=dtype).reshape(shape).copy()
        # Reuse the producer's semantic checks without opening any SDK object.
        encode_native_tactile(m["serial"], **arrays, metadata=m)
        return m, arrays
    except (KeyError, TypeError, ValueError, UnicodeError) as exc:
        raise SensorFault(f"Malformed native tactile packet: {exc}") from exc


@dataclass(frozen=True)
class WrenchCalibration:
    """calibrated = matrix @ (raw - offset), in N and explicit moment units."""
    matrix: np.ndarray
    offset: np.ndarray
    normal_axis: int
    normal_sign: float
    moment_unit: str
    evidence: str

    def __post_init__(self):
        matrix, offset = np.array(self.matrix, float, copy=True), np.array(self.offset, float, copy=True)
        if (matrix.shape != (6, 6) or offset.shape != (6,) or not np.isfinite(matrix).all()
                or not np.isfinite(offset).all() or np.linalg.matrix_rank(matrix) != 6
                or type(self.normal_axis) is not int or self.normal_axis not in (0, 1, 2)
                or self.normal_sign not in (-1, 1) or self.moment_unit not in ("Nm", "Nmm")
                or not self.evidence or "TODO" in self.evidence or "unknown" in self.evidence.lower()):
            raise ValueError("Explicit verified wrench calibration and provenance required")
        matrix.setflags(write=False)
        offset.setflags(write=False)
        object.__setattr__(self, "matrix", matrix)
        object.__setattr__(self, "offset", offset)

    def apply(self, raw):
        return self.matrix @ (np.asarray(raw)-self.offset)


@dataclass(frozen=True)
class SdkWrenchProjection:
    """Signed normal selection and software offset; no physical-unit claim."""
    offset: np.ndarray
    normal_axis: int
    normal_sign: float
    moment_unit: str = "sdk"
    evidence: str = "existing raw SDK signal; software unloaded mean subtraction only"

    def __post_init__(self):
        value = np.array(self.offset, dtype=float, copy=True)
        if value.shape != (6,) or not np.isfinite(value).all() or type(self.normal_axis) is not int or self.normal_axis not in (0, 1, 2) or self.normal_sign not in (-1, 1):
            raise ValueError("Explicit SDK offset and measured normal mapping required")
        value.setflags(write=False)
        object.__setattr__(self, "offset", value)

    def apply(self, raw):
        return np.asarray(raw)-self.offset


@dataclass(frozen=True)
class ForceOnlyWrenchCalibration(SdkWrenchProjection):
    """Verified N for forces, unconverted SDK moments; no inferred moment scale."""
    force_scale_to_N: float = 0.

    def __post_init__(self):
        super().__post_init__()
        if (not np.isfinite(self.force_scale_to_N) or self.force_scale_to_N <= 0
                or not self.evidence or "TODO" in self.evidence.upper()):
            raise ValueError("Measured positive N/SDK coefficient and source-specific evidence required")

    def apply(self, raw):
        result = np.asarray(raw)-self.offset
        result[:3] *= self.force_scale_to_N
        return result


class SharedSensorHub:
    """Receive callbacks only; the caller owns existing sources and their lifetime.

    DD shape/dtype must be measured, not inferred from the old encoded stream.
    All host receive timestamps use the SAME host clock as the robot backend.
    Source resets, gaps, queue overflow and invalid packets latch a fault. A new
    hub/session is required after a source/reference reset; no automatic rebase.
    """
    def __init__(self, *, serials, dd_shape, dd_dtype, calibrations, max_age_s=.25,
                 max_pair_skew_s=.10, max_queue=4096, clock=time.monotonic_ns):
        if set(serials) != {"left", "right"} or len(set(serials.values())) != 2:
            raise ValueError("Explicit distinct left/right sensor serials required")
        if set(calibrations) != {"left", "right"} or not all(isinstance(c, WrenchCalibration) for c in calibrations.values()):
            raise ValueError("Two verified physical wrench calibrations required")
        self.serials, self.calibrations = dict(serials), dict(calibrations)
        self.dd_shape, self.dd_dtype = tuple(dd_shape), np.dtype(dd_dtype)
        if len(self.dd_shape) != 3 or self.dd_shape[-1] != 3 or self.dd_dtype.kind != "f":
            raise ValueError("Configured lossless DD view must be HxWx3 native float")
        if max_age_s <= 0 or max_pair_skew_s <= 0 or max_queue < 1:
            raise ValueError("Invalid freshness/queue bounds")
        self.max_age_ns, self.max_skew_ns = int(max_age_s*1e9), int(max_pair_skew_s*1e9)
        self.max_queue, self.clock = int(max_queue), clock
        self._condition = threading.Condition(threading.RLock())
        self._latest, self._identity, self._sequence = {}, {}, {}
        self._pending, self._last_normal_pair = deque(), None
        self._fault = None
        self._context = {"stage": "IDLE"}

    def _fail(self, reason):
        self._fault = self._fault or str(reason)
        self._condition.notify_all()
        raise SensorFault(self._fault)

    def fail(self, reason):
        """Transport threads must call this on their exception/overflow."""
        with self._condition:
            self._fail(reason)

    @property
    def fault(self):
        with self._condition:
            return self._fault

    def set_context(self, *, stage, trial_id=None):
        with self._condition:
            self._context = {"stage": str(stage), "trial_id": trial_id}

    def _stamp(self, received_ns):
        stamp = self.clock() if received_ns is None else received_ns
        if type(stamp) is not int or not 0 <= self.clock()-stamp <= self.max_age_ns:
            self._fail("Source receive time is stale/future or not host monotonic ns")
        return stamp

    def _track(self, key, sequence, identity):
        if type(sequence) is not int or sequence < 0:
            self._fail(f"{key}: missing source sequence")
        if key in self._identity and identity != self._identity[key]:
            self._fail(f"{key}: source/reference identity changed")
        if key in self._sequence and sequence != self._sequence[key]+1:
            self._fail(f"{key}: source sequence gap/reset/duplicate {self._sequence[key]} -> {sequence}")
        self._identity[key], self._sequence[key] = identity, sequence

    def _append(self, stream, data, stamp, meta):
        from .backend import Sample
        if len(self._pending) >= self.max_queue:
            self._fail("Native acquisition queue overflow; source data would be lost")
        sample = Sample(stream, np.array(data, copy=True), stamp, meta.get("timestamp"),
                        {**meta, **self._context})
        self._pending.append(sample)
        self._latest[stream] = sample

    def ingest_tactile(self, parts, *, received_ns=None):
        with self._condition:
            if self._fault:
                raise SensorFault(self._fault)
            try:
                meta, arrays = decode_native_tactile(parts)
                side = next((s for s, serial in self.serials.items() if serial == meta["serial"]), None)
                if side is None:
                    self._fail("Unexpected tactile serial")
                stamp = self._stamp(received_ns)
                self._track(side, meta["source_sequence"], (meta["session_id"], meta["reference_sha256"]))
                # Track SDK ids separately: contiguous emitted ids cannot hide dropped SDK frames.
                self._track(side+"/sdk", meta["fid"], meta["session_id"])
                d, f, raw = arrays["depth"], arrays["flow"], arrays["wrench"]
                if d.dtype != self.dd_dtype or f.dtype != self.dd_dtype:
                    self._fail("Native depth/flow dtype differs from measured configuration")
                dd = np.concatenate((d[..., None] if d.ndim == 2 else d, f), axis=-1)
                if dd.shape != self.dd_shape:
                    self._fail("Native DD geometry differs from measured configuration")
                for kind, array in arrays.items():
                    stream = f"wrench_raw_{side}" if kind == "wrench" else f"sdk_{kind}_{side}"
                    self._append(stream, array, stamp, {**meta, "unit": meta[kind+"_unit"]})
                self._append("dd_"+side, dd, stamp, {**meta, "representation": "lossless_stack_depth_flow",
                             "channels": ["D", "Dx", "Dy"], "unit": meta["depth_unit"],
                             "channel_units": [meta["depth_unit"], meta["flow_unit"], meta["flow_unit"]]})
                cal = self.calibrations[side]
                wrench = cal.apply(raw)
                self._append("wrench_"+side, wrench, stamp, {**meta, "unit": ["N"]*3+[cal.moment_unit]*3,
                             "calibration_evidence": cal.evidence, "representation": "calibrated_wrench"})
                if all("wrench_"+s in self._latest for s in ("left", "right")):
                    pair = tuple(self._sequence[s] for s in ("left", "right"))
                    changed = self._last_normal_pair is None or all(a != b for a,b in zip(pair, self._last_normal_pair))
                    samples = [self._latest["wrench_"+s] for s in ("left", "right")]
                    if changed and abs(samples[0].monotonic_ns-samples[1].monotonic_ns) <= self.max_skew_ns:
                        values = [self.calibrations[s].normal_sign*x.data[self.calibrations[s].normal_axis]
                                  for s,x in zip(("left", "right"), samples)]
                        self._append("normal_force_N", values, stamp,
                                     {"unit": "N", "source_sequences": list(pair), "paired_by": "host_receive_time"})
                        self._last_normal_pair = pair
                self._condition.notify_all()
            except Exception as exc:
                self._fail(exc)

    def ingest_camera(self, name, image, *, source_sequence, source_session, received_ns=None,
                      metadata=None):
        with self._condition:
            if self._fault:
                raise SensorFault(self._fault)
            if name not in ("left_wrist", "right_wrist") or not source_session:
                self._fail("Two named wrist sources and explicit source session required")
            a = np.asarray(image)
            if a.ndim != 3 or a.shape[-1] != 3 or a.dtype != np.uint8 or not a.size:
                self._fail("Wrist source must supply RGB uint8 HxWx3")
            stamp = self._stamp(received_ns)
            stream = "camera/"+name
            self._track(stream, source_sequence, (source_session, a.shape))
            self._append(stream, a, stamp, {**(metadata or {}), "source_sequence": source_sequence,
                         "source_session": source_session, "color_order": "RGB"})
            self._condition.notify_all()

    def drain_samples(self):
        with self._condition:
            result = list(self._pending)
            self._pending.clear()
            return result

    def snapshot(self):
        """Sensor portion only; backend must merge independently fresh arm/gripper state."""
        with self._condition:
            if self._fault:
                raise SensorFault(self._fault)
            required = ["dd_left", "dd_right", "wrench_left", "wrench_right"]
            for key in required:
                if key not in self._latest:
                    raise SensorFault("Waiting for source "+key)
                if not 0 <= self.clock()-self._latest[key].monotonic_ns <= self.max_age_ns:
                    raise SensorFault("Stale source "+key)
            if abs(self._latest["wrench_left"].monotonic_ns-self._latest["wrench_right"].monotonic_ns) > self.max_skew_ns:
                raise SensorFault("Left/right wrench receive-time skew exceeds configured bound")
            wrench = np.stack([self._latest["wrench_"+s].data for s in ("left", "right")])
            normal = np.array([self.calibrations[s].normal_sign*wrench[i,self.calibrations[s].normal_axis]
                               for i,s in enumerate(("left","right"))])
            return {"dd": [self._latest["dd_"+s].data.copy() for s in ("left", "right")],
                    "wrench": wrench,"normal_N":normal,
                    "timestamp_ns":min(self._latest[k].monotonic_ns for k in required),"fault":None,
                    "source_timestamp_ns": {k:self._latest[k].monotonic_ns for k in required},
                    "source_sequences": {k:self._latest[k].metadata["fid"] for k in required}}

    def images(self, timeout_s=None):
        return {name:sample.data for name,sample in self.image_samples(timeout_s).items()}

    def image_samples(self, timeout_s=None):
        """Wait for BOTH host-received frames newer than the request.

        A verified producer/fanout must ALSO rule out a delayed pre-motion exposure;
        host receipt alone cannot prove exposure freshness. This class therefore
        does not assert verified_fresh_image_capture on behalf of the hardware.
        """
        requested = self.clock()
        timeout = self.max_age_ns/1e9 if timeout_s is None else min(timeout_s, self.max_age_ns/1e9)
        deadline = time.monotonic()+timeout
        with self._condition:
            while True:
                if self._fault:
                    raise SensorFault(self._fault)
                frames = {n:self._latest.get("camera/"+n) for n in ("left_wrist", "right_wrist")}
                if all(x is not None and x.monotonic_ns > requested and
                       0 <= self.clock()-x.monotonic_ns <= self.max_age_ns for x in frames.values()):
                    from .backend import Sample
                    return {name:Sample(x.stream,x.data.copy(),x.monotonic_ns,x.device_timestamp,
                                        dict(x.metadata or {})) for name,x in frames.items()}
                remaining = deadline-time.monotonic()
                if remaining <= 0:
                    raise SensorFault("Timed out waiting for both fresh wrist frames")
                self._condition.wait(remaining)


def decode_existing_dd(parts):
    """Decode 39030 exactly as the existing model, preserving the encoded input.

    The float16 round trip matches dd_codec.decode_model_dd; HWC is retained for
    runtime descriptors. Neither result is native SDK DD or proven physical mm.
    """
    try:
        if len(parts) != 3:
            raise SensorFault("39030 requires topic, JSON metadata, uint16 payload")
        m = json.loads(parts[1])
        frame = m["frame"]
        if (parts[0].decode() != m["serial"] or frame["dtype"] != "uint16" or
                frame["shape"] != [96,128,3] or frame["bytes"] != 96*128*3*2 or len(parts[2]) != 96*128*3*2):
            raise SensorFault("Existing DD wire contract differs from audited codec")
        if type(m.get("fid")) is not int or m["fid"] < 0:raise SensorFault("Missing DD source fid")
        if m.get("schema")=="dm_x5_dd_v1":
            encoding=m["encoding"]
            if (frame["byteorder"] != "little" or frame["channels"] != ["D","Dx","Dy"] or
                    encoding["scale"] != [1000.,1000.,1000.] or encoding["offset"] != [0.,30000.,30000.] or
                    encoding["quantization"] != "truncate_uint16_then_inter_area" or
                    m.get("depth_fid") != m["fid"] or m.get("deformation_fid") != m["fid"]):
                raise SensorFault("39030 encoding or depth/deformation alignment differs")
            m.update(dd_source="dm_x5_dd_v1",depth_flow_fid_alignment_verified=True)
        elif m.get("schema") is None and "raw" not in m:
            if (frame["channels"] != ["depth","deformation_x","deformation_y"] or
                    m["depth"]!={"channel":0,"scale":1000.,"offset":0.} or
                    m["deformation"]!={"channels":[1,2],"scale":1000.,"offset":30000.}):
                raise SensorFault("Legacy DD encoding differs from audited producer")
            m.update(dd_source="legacy_depth_deformation_u16",depth_flow_fid_alignment_verified=False,
                     quantization="rint_clip_uint16_then_inter_area",byteorder_provenance="audited_little_endian_X5_host")
        else:raise SensorFault("Unrecognized DD schema; raw tactile images are not DD")
        m.update(native_dd=False,timestamp=m.get("timestamp",m.get("t")),
                 timestamp_kind=m.get("timestamp_kind","sdk_read_wall_time_not_hardware_exposure"))
        encoded=np.frombuffer(parts[2],dtype="<u2").reshape(96,128,3).copy()
        decoded=encoded.astype(np.float32)
        decoded[...,1:]-=30000.
        decoded/=1000.
        decoded=decoded.astype(np.float16).astype(np.float32)
        return m,encoded,decoded
    except (KeyError,ValueError,TypeError,UnicodeError) as exc:
        raise SensorFault(f"Malformed existing DD packet: {exc}") from exc


def decode_existing_gel_raw(parts):
    """Audited 39010: topic, JSON raw spec, unchanged uint8 frame payload."""
    try:
        if len(parts) != 3:
            raise ValueError("39010 requires exactly three multipart frames")
        meta = json.loads(parts[1])
        spec = meta["raw"]
        shape = spec["shape"]
        if parts[0].decode() != meta["serial"] or spec["present"] is not True or spec["dtype"] != "uint8":
            raise ValueError("raw serial/presence/dtype differs from audited producer")
        if not isinstance(shape, list) or len(shape) not in (2, 3) or any(type(n) is not int or n <= 0 for n in shape):
            raise ValueError("invalid original raw shape")
        size = math.prod(shape)
        if size > 16*1024*1024 or spec["bytes"] != size or len(parts[2]) != size:
            raise ValueError("raw payload length mismatch")
        if type(meta.get("fid")) is not int or meta["fid"] < 0:
            raise ValueError("raw fid required")
        return meta, np.frombuffer(parts[2], dtype=np.uint8).reshape(shape).copy()
    except (KeyError, TypeError, ValueError, UnicodeError) as exc:
        raise SensorFault("Malformed 39010 gel raw packet: " + str(exc)) from exc


def inspect_dd_pub(address="tcp://192.168.9.11:39030", *,
                   serials=("X2620410318","X2620410347"),duration_s=5.,
                   _context_factory=None,_clock=time.monotonic_ns):
    """At most five seconds of SUB diagnostics; never turns raw images into DD."""
    if not isinstance(address,str) or not address.startswith("tcp://") or not np.isfinite(duration_s) or not 0<duration_s<=5:
        raise ValueError("Explicit known PUB address and duration in (0,5] required")
    if len(serials)!=2 or len(set(serials))!=2:raise ValueError("Two distinct expected serials required")
    import zmq
    context=(_context_factory or zmq.Context)();socket=None;start=_clock()
    report={"endpoint":address,"ready":False,"native_dd":False,"sensors":{s:{"count":0,"dd_count":0} for s in serials}}
    try:
        socket=context.socket(zmq.SUB);socket.setsockopt(zmq.LINGER,0);socket.setsockopt(zmq.RCVHWM,512)
        for serial in serials:socket.setsockopt(zmq.SUBSCRIBE,serial.encode())
        socket.connect(address)
        while _clock()-start<duration_s*1e9:
            if not socket.poll(min(50,max(1,int((duration_s*1e9-(_clock()-start))/1e6))),zmq.POLLIN):continue
            parts=socket.recv_multipart()
            try:
                m=json.loads(parts[1]);serial=m["serial"]
                if serial not in report["sensors"] or parts[0].decode()!=serial:raise ValueError("Unexpected serial/topic")
                info=report["sensors"][serial];info["count"]+=1
                info.update(last_receive_monotonic_ns=_clock(),last_fid=m.get("fid"),
                            payload_bytes=len(parts[-1]),source_timestamp=m.get("timestamp",m.get("t")))
                try:
                    meta,encoded,decoded=decode_existing_dd(parts)
                    info["dd_count"]+=1
                    info.update(wire_dtype=str(encoded.dtype),wire_shape=list(encoded.shape),
                                decoded_dtype=str(decoded.dtype),decoded_shape=list(decoded.shape),
                                dd_source=meta["dd_source"],native_dd=False,
                                depth_flow_fid_alignment_verified=meta["depth_flow_fid_alignment_verified"])
                except SensorFault as exc:
                    info["not_dd_reason"]=str(exc)
                    if "raw" in m:info["raw_image_metadata"]=m["raw"]
            except (KeyError,ValueError,TypeError,UnicodeError,IndexError) as exc:
                report["invalid_messages"]=report.get("invalid_messages",0)+1
                report["last_error"]=str(exc)
        report["ready"]=all(x["dd_count"]>=2 and x["dd_count"]==x["count"] for x in report["sensors"].values()) and not report.get("invalid_messages")
    except Exception as exc:report["error"]=f"{type(exc).__name__}: {exc}"
    finally:
        if socket is not None:socket.close(0)
        context.term()
    report["elapsed_s"]=(_clock()-start)/1e9
    return report


class ActualX5SensorSource:
    """User-authorized existing X5 streams, without SDK ownership or source changes.

    39020/39030 are broadcast SUB; 5552 uses the original camera PULL/JPEG
    protocol. This process must be the intended camera consumer: another PULL
    consumer splits frames. Unlike native taps, this existing latest-only path
    cannot guarantee no upstream loss, capture time or SDK raw fidelity. Those
    limitations are emitted in every sample's provenance, never upgraded to a
    hardware capability automatically.

    ``calibrations`` may be absent for observation, but snapshot() then refuses
    to manufacture normal_N. start()/close() only connect/close data transports.
    No SDK initialization, reset, robot command, software force zeroing, or
    source service management is performed.
    """
    def __init__(self, cfg, *, calibrations=None, clock=time.monotonic_ns, _context_factory=None):
        from .force_control import force_unit
        x5=cfg["hardware"]["public_ros"]["sensors"]
        self.cfg,self.clock=cfg,clock
        self.force_unit = force_unit(cfg)
        self.serials=dict(x5["serials"])
        if set(self.serials)!={"left","right"} or len(set(self.serials.values()))!=2:
            raise ValueError("Explicit distinct tactile serial mapping required")
        if calibrations is not None and (set(calibrations)!={"left","right"} or
                not all(isinstance(x,(WrenchCalibration,SdkWrenchProjection)) for x in calibrations.values())):
            raise ValueError("Expected two verified WrenchCalibration values")
        expected_calibration = (WrenchCalibration,ForceOnlyWrenchCalibration) if self.force_unit == "N" else SdkWrenchProjection
        if calibrations is not None and (not all(isinstance(x,expected_calibration) for x in calibrations.values()) or
                self.force_unit=="sdk" and any(isinstance(x,ForceOnlyWrenchCalibration) for x in calibrations.values())):
            raise ValueError("Supplied wrench processing does not match explicitly selected force.unit")
        self.calibrations=calibrations
        cameras=x5["wrist_endpoints"]
        if set(cameras)!={"left_wrist","right_wrist"}:raise ValueError("Exactly two wrist endpoints required")
        self.endpoints={"dd":x5["dd_address"],"wrench":x5["wrench_address"],**cameras}
        self._gel_mode = cfg.get("record", {}).get("gel_raw", "off")
        if self._gel_mode not in ("off", "windows", "full"):
            raise ValueError("record.gel_raw must be off, windows or full")
        if self._gel_mode != "off":
            self.endpoints["gel_raw"] = x5["gel_raw_address"]
        if any(not isinstance(x,str) or not x.startswith("tcp://") for x in self.endpoints.values()):
            raise ValueError("Explicit X5 TCP endpoints required")
        self._force_conversion=x5.get("force_conversion")
        self._zero_cfg=x5.get("startup_zero")
        self.startup_zero_report=None
        self.last_zero_attempt=None
        self._zero_lock=threading.Lock()
        if self._force_conversion is not None:
            f=self._force_conversion
            if set(f.get("normal_signs",{}))!={"left","right"}:
                raise ValueError("Explicit left/right normal signs required")
            if self.force_unit == "N":
                if not isinstance(f.get("force_scale_to_N"),(int,float)) or not np.isfinite(f["force_scale_to_N"]) or f["force_scale_to_N"]<=0:
                    raise ValueError("Measured positive force conversion factor required")
                if f.get("moment_unit") in ("Nm", "Nmm"):
                    if not isinstance(f.get("moment_scale_to_unit"),(int,float)) or not np.isfinite(f["moment_scale_to_unit"]) or f["moment_scale_to_unit"]<=0:
                        raise ValueError("Physical moment unit requires its measured conversion factor")
                    self._force_matrix=np.diag([f["force_scale_to_N"]]*3+[f["moment_scale_to_unit"]]*3)
                    for side in ("left","right"):
                        WrenchCalibration(self._force_matrix,np.zeros(6),f["normal_axis"],f["normal_signs"][side],f["moment_unit"],f["evidence"])
                else:
                    for side in ("left", "right"):
                        ForceOnlyWrenchCalibration(np.zeros(6),f["normal_axis"],f["normal_signs"][side],
                                                  "sdk",f["evidence"],f["force_scale_to_N"])
            else:
                for side in ("left", "right"):
                    SdkWrenchProjection(np.zeros(6),f["normal_axis"],f["normal_signs"][side])
        recording=cfg.get("recording",{})
        self.max_age_ns=int(recording.get("source_max_age_s",recording.get("stale_timeout_s",.25))*1e9)
        self.max_skew_ns=int(x5.get("max_pair_skew_s",.10)*1e9)
        self.max_queue=int(x5.get("max_queue",4096))
        self._condition=threading.Condition(threading.RLock())
        self._pending,self._latest=deque(),{}
        self._fids,self._identities,self._camera_ids={},{},{}
        self._fault=None
        self._context={"stage":"IDLE"}
        self._stop=threading.Event()
        self._threads=[]
        self._zmq_context=None
        self._context_factory=_context_factory
        self._last_normal_pair=None
        self._raw_history={side:deque(maxlen=4096) for side in ("left","right")}
        self.provenance={"native_dd":False,"dd_source":"existing_dm_x5_dd_v1",
            "dd_service_name":x5.get("dd_service_name","unreported"),
            "dd_service_version":x5.get("dd_service_version","unknown_not_transmitted"),
            "dd_encoded_shape":[96,128,3],"dd_encoded_dtype":"uint16_little_endian",
            "dd_decoded_dtype":"float32_after_float16_roundtrip", "gel_raw_recording":self._gel_mode,
            "force_unit":self.force_unit, "force_processing":"guarded software unloaded offset; no unit inference",
            "dd_decoding":"uint16_scale1000_offset30000_float16_roundtrip",
            "dd_source_unit":"SDK units unverified; not established mm",
            "upstream_complete_delivery_verified":False,"hardware_exposure_timestamp_available":False,
            "camera_protocol":"existing_X5_PUSH_PULL_JPEG_single_intended_consumer"}

    def _fail(self,reason):
        with self._condition:
            self._fault=self._fault or str(reason)
            self._condition.notify_all()

    @property
    def fault(self):
        with self._condition:return self._fault

    def set_context(self, *, stage, trial_id=None):
        with self._condition:self._context={"stage":str(stage),"trial_id":trial_id}

    def _record(self,stream,data,stamp,metadata):
        from .backend import Sample
        if len(self._pending)>=self.max_queue:
            raise SensorFault("X5 receive queue overflow; stopping rather than silently dropping")
        sample=Sample(stream,np.array(data,copy=True),stamp,metadata.get("timestamp"),
                      {**self.provenance,**metadata,**self._context})
        self._pending.append(sample)
        self._latest[stream]=sample

    def _track_existing(self,key,m):
        fid=m.get("fid")
        if type(fid) is not int or fid<0:raise SensorFault("Missing source fid")
        previous=self._fids.get(key)
        if previous is not None and fid<=previous:raise SensorFault(f"{key}: source fid reset/reordered/duplicate")
        identity=(m.get("session_id"),m.get("reference_sha256"))
        if key in self._identities and identity!=self._identities[key]:raise SensorFault(f"{key}: source/reference changed")
        self._fids[key],self._identities[key]=fid,identity
        return {"source_sequence":fid,"source_gap_count":0 if previous is None else fid-previous-1,
                "source_reference_verified":bool(identity[0] and identity[1])}

    def _ingest(self,kind,parts,stamp):
        # Decoding is CPU work and must not hold the wrench/force feedback lock.
        rgb = None
        if kind in ("left_wrist", "right_wrist"):
            import cv2
            if len(parts)!=1:raise SensorFault("Existing wrist source must emit a single JPEG frame")
            bgr=cv2.imdecode(np.frombuffer(parts[0],np.uint8),cv2.IMREAD_COLOR)
            if bgr is None:raise SensorFault("Wrist JPEG decode failed")
            rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
            jpeg_digest=hashlib.sha256(parts[0]).hexdigest()
        with self._condition:
            if self._fault:raise SensorFault(self._fault)
            if kind in ("dd","wrench","gel_raw"):
                if len(parts)<2:raise SensorFault("Missing source metadata")
                m=json.loads(parts[1]);serial=m.get("serial")
                side=next((s for s,v in self.serials.items() if v==serial),None)
                if side is None or parts[0].decode()!=serial:raise SensorFault("Unexpected source serial")
                if kind=="gel_raw":
                    m,raw=decode_existing_gel_raw(parts)
                    m={**m,**self._track_existing("gel_raw_"+side,m),"unit":"original_uint8_intensity",
                       "representation":"original_39010_no_resize_no_lossy_encoding"}
                    if self._gel_mode == "full" or self._gel_mode == "windows" and self._context.get("stage") in ("PROBE", "HOLD"):
                        self._record("gel_raw_"+side,raw,stamp,m)
                elif kind=="dd":
                    m,encoded,decoded=decode_existing_dd(parts)
                    m={**m,**self._track_existing("dd_"+side,m),"unit":"sdk_encoded_units_unverified"}
                    self.provenance["dd_encoding_observed"] = m.get("encoding", {
                        "depth":m.get("depth"),"deformation":m.get("deformation"),"quantization":m.get("quantization")})
                    self._record("dd_encoded_"+side,encoded,stamp,m)
                    self._record("dd_"+side,decoded,stamp,{**m,"representation":"decoded_existing_quantized_DD"})
                else:
                    if len(parts)!=2 or m.get("schema")!="dm_x5_wrench_v1":raise SensorFault("Bad wrench schema")
                    raw=np.asarray(m["wrench"],dtype=np.float64)
                    if raw.shape!=(6,) or not np.isfinite(raw).all():raise SensorFault("Invalid raw wrench")
                    m={**m,**self._track_existing("wrench_"+side,m),"unit":"unknown_sdk_raw",
                       "timestamp_kind":m.get("timestamp_kind","sdk_read_wall_time_not_hardware_exposure"),
                       "wire_representation":"JSON numbers; float64 receiver storage, not native SDK dtype"}
                    self._record("wrench_raw_"+side,raw,stamp,m)
                    self._raw_history[side].append((stamp,raw.copy(),m["fid"]))
                    if self.calibrations is not None:
                        cal=self.calibrations[side]
                        self._record("wrench_"+side,cal.apply(raw),stamp,
                            {**m,"unit":[self.force_unit]*3+[cal.moment_unit]*3,"calibration_evidence":cal.evidence})
                        if all("wrench_"+s in self._latest for s in ("left","right")):
                            values=[self._latest["wrench_"+s] for s in ("left","right")]
                            pair=tuple(x.metadata["fid"] for x in values)
                            changed=self._last_normal_pair is None or all(x!=y for x,y in zip(pair,self._last_normal_pair))
                            if changed and abs(values[0].monotonic_ns-values[1].monotonic_ns)<=self.max_skew_ns:
                                normal=[self.calibrations[s].normal_sign*x.data[self.calibrations[s].normal_axis]
                                        for s,x in zip(("left","right"),values)]
                                from .force_control import force_components
                                components=force_components(normal)
                                metadata={"unit":self.force_unit,"source_sequences":list(pair)}
                                self._record("normal_force_"+self.force_unit,normal,stamp,metadata)
                                self._record("normal_mean_"+self.force_unit,np.asarray(components["mean"]),stamp,metadata)
                                self._record("normal_sum_"+self.force_unit,np.asarray(components["sum"]),stamp,metadata)
                                self._last_normal_pair=pair
            else:
                if rgb is None:raise SensorFault("Unknown source kind: "+str(kind))
                sequence=self._camera_ids.get(kind,-1)+1;self._camera_ids[kind]=sequence
                # The source transmits no fid/time: never present this local counter as native.
                metadata={"receiver_sequence":sequence,"source_sequence":None,
                    "timestamp_kind":"host_receive_monotonic_only","color_order":"RGB",
                    "processing":"JPEG decode + BGR_to_RGB; original resolution, no crop/undistort/resize",
                    "receiver_validated_jpeg_sha256":jpeg_digest}
                previous=self._latest.get("camera/"+kind)
                if previous is not None and previous.data.shape!=rgb.shape:raise SensorFault("Wrist source dimensions changed")
                from .backend import Sample
                if len(self._pending)>=self.max_queue:raise SensorFault("X5 receive queue overflow")
                meta={**self.provenance,**metadata,**self._context}
                # The recorder already accepts original JPEG bytes. Keep RGB only
                # for the current template view, without a second lossy encoding.
                self._pending.append(Sample("camera/"+kind,bytes(parts[0]),stamp,None,meta))
                self._latest["camera/"+kind]=Sample("camera/"+kind,rgb,stamp,None,meta)
            self._condition.notify_all()

    def _receive(self,kind):
        import zmq
        socket=None
        try:
            socket=self._zmq_context.socket(zmq.SUB if kind in ("dd","wrench","gel_raw") else zmq.PULL)
            socket.setsockopt(zmq.LINGER,0)
            socket.setsockopt(zmq.RCVHWM,512)
            if kind in ("dd","wrench","gel_raw"):
                for serial in self.serials.values():socket.setsockopt(zmq.SUBSCRIBE,serial.encode())
            # The endpoint/protocol matches the existing camera client, but this
            # collector deliberately does not CONFLATE away acquisition frames.
            socket.connect(self.endpoints[kind])
            while not self._stop.is_set():
                if not socket.poll(50,zmq.POLLIN):continue
                parts=socket.recv_multipart();stamp=self.clock()
                self._ingest(kind,parts,stamp)
        except Exception as exc:
            if not self._stop.is_set():self._fail(f"{kind}: {type(exc).__name__}: {exc}")
        finally:
            if socket is not None:socket.close(0)

    def start(self, timeout_s=5.):
        if self._threads:raise SensorFault("X5 source already started")
        if not 0<timeout_s<=15:raise ValueError("Bounded startup timeout required")
        import zmq
        self._zmq_context=(self._context_factory or zmq.Context)()
        self._stop.clear()
        for kind in self.endpoints:
            thread=threading.Thread(target=self._receive,args=(kind,),daemon=True,name="bottle_x5_"+kind)
            self._threads.append(thread);thread.start()
        deadline=time.monotonic()+timeout_s
        required=["dd_left","dd_right","wrench_raw_left","wrench_raw_right","camera/left_wrist","camera/right_wrist"]
        try:
            with self._condition:
                while True:
                    if self._fault:raise SensorFault(self._fault)
                    missing=[k for k in required if k not in self._latest or self.clock()-self._latest[k].monotonic_ns>self.max_age_ns]
                    if not missing:break
                    remaining=deadline-time.monotonic()
                    if remaining<=0:raise SensorFault("X5 startup missing/stale streams: "+", ".join(missing))
                    self._condition.wait(min(.1,remaining))
            return self
        except BaseException:
            self.close();raise

    def collect_raw_zero_samples(self, count, timeout, *, monitor):
        """Return only source frames received after this request; never poll-copy."""
        requested=self.clock();deadline=time.monotonic()+timeout
        with self._condition:
            starting={side:self._fids.get("wrench_"+side,-1) for side in ("left","right")}
        while True:
            # Do not acquire a gripper lock while holding the sensor lock: the
            # force worker reads these in the opposite order during actuation.
            monitor()
            with self._condition:
                if self._fault:raise SensorFault(self._fault)
                samples={s:[x for x in self._raw_history[s] if x[0]>requested and x[2]>starting[s]][:count]
                         for s in ("left","right")}
                if all(len(v)>=count for v in samples.values()):return samples
                remaining=deadline-time.monotonic()
                if remaining<=0:raise SensorFault("Guarded software zero lacks 100 fresh independent source frames")
                self._condition.wait(min(.05,remaining))

    def zero_unloaded(self, *, width_reader=None, reference=None, confirm=None, previous=None):
        """The only offset-changing entry point; EVERY call must pass A6 guards."""
        from .zero_guard import guarded_zero_report
        if not callable(width_reader):
            raise SensorFault("Guarded software zero requires measured-width reader; direct zero is forbidden")
        if self._force_conversion is None:
            raise SensorFault("Explicit force conversion required before guarded software zero")
        if not self._zero_lock.acquire(blocking=False):
            raise SensorFault("Another guarded software zero is already in progress")
        self.last_zero_attempt=None
        try:
            report=guarded_zero_report(self.cfg,self,width_reader,reference=reference,confirm=confirm,previous=previous)
            self.last_zero_attempt=report
            f=self._force_conversion
            if self.force_unit == "N":
                if f.get("moment_unit") in ("Nm", "Nmm"):
                    calibrations={s:WrenchCalibration(self._force_matrix,np.asarray(report["fingers"][s]["offset_raw"]),
                                  f["normal_axis"],f["normal_signs"][s],f["moment_unit"],f["evidence"])
                                  for s in ("left","right")}
                else:
                    calibrations={s:ForceOnlyWrenchCalibration(np.asarray(report["fingers"][s]["offset_raw"]),
                                  f["normal_axis"],f["normal_signs"][s],"sdk",f["evidence"],f["force_scale_to_N"])
                                  for s in ("left","right")}
            else:
                calibrations={s:SdkWrenchProjection(np.asarray(report["fingers"][s]["offset_raw"]),
                              f["normal_axis"],f["normal_signs"][s]) for s in ("left", "right")}
            deadline=time.monotonic()+self.cfg["zero_guard"]["timeout_s"]
            with self._condition:
                report["source_identities"]={key:{"session_id":value[0],"reference_sha256":value[1]}
                                             for key,value in self._identities.items()}
                self.calibrations=calibrations
                self._last_normal_pair=None
                for key in ("wrench_left","wrench_right","normal_force_"+self.force_unit):self._latest.pop(key,None)
                while "normal_force_"+self.force_unit not in self._latest:
                    if self._fault:raise SensorFault(self._fault)
                    remaining=deadline-time.monotonic()
                    if remaining<=0:raise SensorFault("No new calibrated feedback arrived after guarded software zero")
                    self._condition.wait(min(.1,remaining))
                report["status"]="ready"
                self.startup_zero_report=report
                return report
        except BaseException as exc:
            self.last_zero_attempt={**(self.last_zero_attempt or {}),"status":"failed",
                                    "error":type(exc).__name__+": "+str(exc)}
            raise
        finally:
            self._zero_lock.release()

    def close(self):
        self._stop.set()
        for thread in self._threads:thread.join(timeout=1.)
        self._threads=[]
        if self._zmq_context is not None:
            self._zmq_context.term();self._zmq_context=None

    def drain_samples(self):
        with self._condition:
            values=list(self._pending);self._pending.clear();return values

    def observation_snapshot(self):
        """Fresh unzeroed readings for initialization; never label them newtons."""
        with self._condition:
            if self._fault:raise SensorFault(self._fault)
            keys=["dd_left","dd_right","wrench_raw_left","wrench_raw_right"]
            for key in keys:
                if key not in self._latest:raise SensorFault("Waiting for "+key)
                if not 0<=self.clock()-self._latest[key].monotonic_ns<=self.max_age_ns:
                    raise SensorFault("Stale source "+key)
            wrench=[self._latest["wrench_raw_"+s] for s in ("left","right")]
            if abs(wrench[0].monotonic_ns-wrench[1].monotonic_ns)>self.max_skew_ns:
                raise SensorFault("Raw finger wrench time skew")
            return {"dd":[self._latest["dd_"+s].data.copy() for s in ("left","right")],
                    "wrench_raw":np.stack([x.data for x in wrench]),"wrench_unit":"unknown_sdk_raw",
                    "timestamp_ns":min(self._latest[k].monotonic_ns for k in keys),
                    "source_timestamp_ns":{k:self._latest[k].monotonic_ns for k in keys},
                    "source_sequences":{k:self._latest[k].metadata.get("fid") for k in keys},
                    "fault":None,"sensor_provenance":dict(self.provenance)}

    def snapshot(self):
        with self._condition:
            if self._fault:raise SensorFault(self._fault)
            if self.calibrations is None:raise SensorFault("N conversion is unverified; raw wrench observation only")
            required=["dd_left","dd_right","wrench_left","wrench_right"]
            for key in required:
                if key not in self._latest:raise SensorFault("Waiting for "+key)
                if not 0<=self.clock()-self._latest[key].monotonic_ns<=self.max_age_ns:raise SensorFault("Stale source "+key)
            wrench=[self._latest["wrench_"+s] for s in ("left","right")]
            if abs(wrench[0].monotonic_ns-wrench[1].monotonic_ns)>self.max_skew_ns:raise SensorFault("Finger wrench time skew")
            # Feedback is derived from exactly the wrench frames whose source ids
            # and receive times are returned below. The independent recorded
            # normal_force_N stream can legitimately still be the previous pair.
            current_wrench=np.stack([x.data for x in wrench])
            normal=np.array([self.calibrations[s].normal_sign*current_wrench[i,self.calibrations[s].normal_axis]
                             for i,s in enumerate(("left","right"))])
            return {"dd":[self._latest["dd_"+s].data.copy() for s in ("left","right")],
                    "wrench":current_wrench,"wrench_force":current_wrench[:,:3],"normal_N":normal,
                    "normal":normal,"force_unit":self.force_unit,
                    "timestamp_ns":min(self._latest[k].monotonic_ns for k in required),"fault":None,
                    "source_timestamp_ns":{k:self._latest[k].monotonic_ns for k in required},
                    "source_sequences":{k:self._latest[k].metadata.get("fid",self._latest[k].metadata.get("source_sequences")) for k in required},
                    "sensor_provenance":dict(self.provenance)}

    def images(self,timeout_s=None):
        return {name:sample.data for name,sample in self.image_samples(timeout_s).items()}

    def image_samples(self,timeout_s=None):
        """Fresh keyframe RGB plus its actual source receipt time and provenance.

        Continuous acquisition separately preserves original JPEG bytes. These
        selected RGB copies are for template matching/keyframe views; encoding
        them again is labelled by JpegWriter rather than called source JPEG.
        """
        requested=self.clock()
        timeout=min(self.max_age_ns/1e9,timeout_s if timeout_s is not None else self.max_age_ns/1e9)
        deadline=time.monotonic()+timeout
        with self._condition:
            while True:
                if self._fault:raise SensorFault(self._fault)
                images={name:self._latest.get("camera/"+name) for name in ("left_wrist","right_wrist")}
                if all(x is not None and x.monotonic_ns>requested and 0<=self.clock()-x.monotonic_ns<=self.max_age_ns for x in images.values()):
                    from .backend import Sample
                    return {name:Sample(x.stream,x.data.copy(),x.monotonic_ns,x.device_timestamp,
                        {**(x.metadata or {}),"keyframe_selection_requested_monotonic_ns":requested,
                         "keyframe_pixel_representation":"RGB decoded from source JPEG; continuous stream preserves original JPEG"})
                        for name,x in images.items()}
                remaining=deadline-time.monotonic()
                if remaining<=0:raise SensorFault("Both wrist frames newer than request are unavailable")
                self._condition.wait(remaining)
