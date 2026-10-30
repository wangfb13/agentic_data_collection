#!/usr/bin/env python3
"""Register validated real reports in a prepared session; offline, no actuators."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time

import yaml

from bottle_scan.session import Session, atomic_json
from bottle_scan.stop_validation import gate_stop_report
from bottle_scan.force_control import require_force_report


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _atomic_bytes(path, data):
    path = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(prefix=path.name + '.', suffix='.tmp', dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def register_reports(session_path, *, stop=None, force=None):
    """Change only report locators after the normal real-run gates accept them.

    All requested reports pass before any session write. Original snapshot bytes
    are retained, and an audit intent is saved before its atomic replacement.
    The caller must finish collection first; an active trial is rejected and
    changed source files are detected before commit.
    """
    if stop is None and force is None:
        raise ValueError('Provide at least one of --stop or --force')
    session = Session(Path(session_path).expanduser().resolve())
    if session.meta.get('simulated') is not False:
        raise ValueError('Report registration requires a real prepared session; simulation cannot authorize hardware')
    if session.progress.get('active') is not None:
        raise ValueError('Session has an active/incomplete trial; finish or recover it before registering reports')
    config_path = session.path / 'config_snapshot.yaml'
    meta_path = session.path / 'session_meta.json'
    progress_path = session.path / 'progress.json'
    before_config = config_path.read_bytes()
    before_meta = meta_path.read_bytes()
    before_progress = progress_path.read_bytes()
    updated = copy.deepcopy(session.cfg)
    changes = []
    verified_bytes = {}
    for kind, supplied in (('stop', stop), ('force', force)):
        if supplied is None:
            continue
        path = Path(supplied).expanduser().resolve(strict=True)
        raw = path.read_bytes()
        report = (gate_stop_report(session.cfg, path=path) if kind == 'stop'
                  else require_force_report(session.cfg, report_path=path))
        if path.read_bytes() != raw:
            raise ValueError('Report changed during validation: ' + str(path))
        original_section = session.cfg['stop_test'] if kind == 'stop' else session.cfg['force']['validation']
        new_section = updated['stop_test'] if kind == 'stop' else updated['force']['validation']
        old_path = original_section['report_path']
        old_resolved = Path(old_path).expanduser().resolve()
        new_section['report_path'] = str(path)
        changes.append({'kind': kind, 'field': 'stop_test.report_path' if kind == 'stop' else 'force.validation.report_path',
            'old_path': old_path, 'new_path': str(path),
            'old_report_sha256': _sha(old_resolved.read_bytes()) if old_resolved.is_file() else None,
            'new_report_sha256': _sha(raw), 'report_binding': report['binding'],
            'validation': 'existing_real_PASS_gate'})
        verified_bytes[path] = raw
    new_config = yaml.safe_dump(updated, sort_keys=False, allow_unicode=True).encode('utf-8')
    # No semantic change outside the two explicitly registered locator fields.
    comparison = copy.deepcopy(updated)
    comparison['stop_test']['report_path'] = session.cfg['stop_test']['report_path']
    comparison['force']['validation']['report_path'] = session.cfg['force']['validation']['report_path']
    if comparison != session.cfg:
        raise RuntimeError('Registration attempted to change a measured configuration value')
    if (config_path.read_bytes() != before_config or meta_path.read_bytes() != before_meta
            or progress_path.read_bytes() != before_progress):
        raise ValueError('Session changed during registration; stop collection and retry')
    if any(path.read_bytes() != raw for path, raw in verified_bytes.items()):
        raise ValueError('A report changed before commit; nothing was registered')
    stamp = time.time_ns()
    backup = session.path / 'validation_registration_backups' / str(stamp)
    backup.mkdir(parents=True, exist_ok=False)
    with (backup / 'config_snapshot.yaml').open('xb') as stream:
        stream.write(before_config)
        stream.flush()
        os.fsync(stream.fileno())
    entry = {'schema': 'bottle_validation_registration_v1', 'registered_unix_ns': stamp,
        'state': 'prepared', 'session': str(session.path), 'changes': changes,
        'snapshot_before_sha256': _sha(before_config), 'snapshot_after_sha256': _sha(new_config),
        'snapshot_backup': str((backup / 'config_snapshot.yaml').relative_to(session.path)),
        'scope': 'report paths only; measurements, commissioning acceptance, progress and plans unchanged'}
    session.meta.setdefault('validation_report_registrations', []).append(entry)
    # If interrupted, the prepared audit and exact backup remain inspectable.
    atomic_json(backup / 'audit.json', entry)
    session.save_meta()
    try:
        _atomic_bytes(config_path, new_config)
    except BaseException as exc:
        entry.update(state='failed_before_snapshot_commit', error=type(exc).__name__ + ': ' + str(exc))
        atomic_json(backup / 'audit.json', entry)
        session.save_meta()
        raise
    entry['state'] = 'committed'
    atomic_json(backup / 'audit.json', entry)
    session.save_meta()
    return entry


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session', required=True)
    parser.add_argument('--stop', type=Path, help='Existing real PASS stop-test JSON')
    parser.add_argument('--force', type=Path, help='Existing real PASS force-validation JSON')
    args = parser.parse_args(argv)
    if args.stop is None and args.force is None:
        parser.error('at least one of --stop or --force is required')
    result = register_reports(args.session, stop=args.stop, force=args.force)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print('离线登记完成：只更新经验证的报告路径，保留原配置备份；未连接机器人或发送动作。')
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, OSError, KeyboardInterrupt) as exc:
        print('STOP: ' + str(exc), file=sys.stderr)
        raise SystemExit(2)
