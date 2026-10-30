"""Crash-aware session metadata. Raw sensor arrays live in recording.py."""
from __future__ import annotations
import csv
import hashlib
import json
import os
import platform
import subprocess
import time
from pathlib import Path
import numpy as np
import yaml


def plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    return value


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as out:
        json.dump(plain(value), out, indent=2, ensure_ascii=False, allow_nan=False)
        out.flush()
        os.fsync(out.fileno())
    os.replace(temp, path)


def fingerprint(value):
    return hashlib.sha256(json.dumps(plain(value), sort_keys=True).encode()).hexdigest()


def code_version():
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for p in sorted(root.rglob("*.py")):
        if any(s.startswith(".") for s in p.relative_to(root).parts):
            continue
        digest.update(str(p.relative_to(root)).encode())
        digest.update(p.read_bytes())
    try:
        commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"],
                                         stderr=subprocess.DEVNULL, text=True).strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        commit = None
    return {"git_commit": commit, "source_sha256": digest.hexdigest()}


class Session:
    def __init__(self, path):
        self.path = Path(path)
        self.meta = json.loads((self.path / "session_meta.json").read_text())
        self.cfg = yaml.safe_load((self.path / "config_snapshot.yaml").read_text())
        self.progress_path = self.path / "progress.json"
        self.progress = json.loads(self.progress_path.read_text())
        self.T0 = np.array(self.meta["T0"], dtype=float)

    @classmethod
    def create(cls, path, cfg, T0, *, simulated):
        from .geometry import grasp_transform, grasp_pose
        G = grasp_transform(cfg)
        path = Path(path)
        path.mkdir(parents=True, exist_ok=False)
        for name in ("references", "trials", "checks", "plans"):
            (path / name).mkdir()
        (path / "config_snapshot.yaml").write_text(yaml.safe_dump(plain(cfg), sort_keys=False), encoding="utf-8")
        atomic_json(path / "session_meta.json", {
            "schema_version": 2, "T0": T0, "B": grasp_pose(T0, cfg), "G": G,
            "B_legacy": T0, "simulated": bool(simulated),
            "created_unix_ns": time.time_ns(), "clock": "synthetic virtual clock seeded from host monotonic_ns" if simulated else "host time.monotonic_ns",
            "clock_boot_id": boot_id(), "references": [], "camera_intrinsics": None,
            "sensor_serials": cfg.get("hardware", {}).get("sensor_serials", "TODO"),
            "code_version": code_version(), "reference_policy": "immutable until explicit new T0/new session",
        })
        atomic_json(path / "progress.json", {"completed": [], "active": None,
                    "requires_operator_recovery": False, "commissioning_passed": False,
                    "last_bottle": None, "plans": {}})
        return cls(path)

    def save_meta(self):
        atomic_json(self.path / "session_meta.json", self.meta)

    def save_progress(self):
        atomic_json(self.progress_path, self.progress)

    def lock_plan(self, task, plan):
        digest = fingerprint(plan)
        existing = self.progress["plans"].get(task)
        if existing and existing != digest:
            raise ValueError("Plan changed within session; create a new session instead.")
        self.progress["plans"][task] = digest
        self.save_progress()
        atomic_json(self.path / "plans" / f"{task}.json", plan)

    def start_trial(self, trial_id):
        if trial_id in self.progress["completed"]:
            raise ValueError("Refusing to overwrite a completed trial")
        path = self.path / "trials" / trial_id
        # Preserve a partial attempt, including its last airborne flag.
        if path.exists():
            archived = self.path / "interrupted"
            archived.mkdir(exist_ok=True)
            path.rename(archived / f"{trial_id}_{time.time_ns()}")
        path.mkdir()
        self.progress["active"] = trial_id
        self.save_progress()
        return path

    def complete(self, meta):
        trial_id = meta["trial_id"]
        atomic_json(self.path / "trials" / trial_id / "meta.json", meta)
        if trial_id not in self.progress["completed"]:
            self.progress["completed"].append(trial_id)
        self.progress["active"] = None
        # A recorded plan row (including a safety skip) is not evidence that
        # the operator has installed that bottle. Only the CLI's confirmed,
        # successful bottle-position check updates last_bottle.
        self.save_progress()
        self.rebuild_index()

    def rebuild_index(self):
        rows = []
        for trial_id in self.progress["completed"]:
            item = json.loads((self.path / "trials" / trial_id / "meta.json").read_text())
            row = {"trial_id": trial_id, "task": item.get("task"), "bottle": item.get("bottle"),
                   "status": item.get("status"), "kind": item.get("kind"),
                   **item.get("offsets", {}), **item.get("auto", {}),
                   "operator_label": item.get("operator", {}).get("label", ""),
                   "operator_note": item.get("operator", {}).get("note", "")}
            rows.append({k: json.dumps(plain(v), ensure_ascii=False) if isinstance(v, (list, dict)) else v
                         for k, v in row.items()})
        keys = list(dict.fromkeys(k for row in rows for k in row))
        temp = self.path / "trial_index.csv.tmp"
        with temp.open("w", newline="", encoding="utf-8") as out:
            writer = csv.DictWriter(out, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)
            out.flush()
            os.fsync(out.fileno())
        os.replace(temp, self.path / "trial_index.csv")


def boot_id():
    p = Path("/proc/sys/kernel/random/boot_id")
    return p.read_text().strip() if p.exists() else platform.node()
