"""Thin entry to the existing V4 scheduler; no duplicate model/robot runtime.

All validation, ownership, force control and motion checks stay in the existing
scheduler. Check is the default and never requests an actuator launch. Review
and summary use separate processes to avoid mixing the two bottle_scan trees.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys

DEFAULT_SCHEDULER = Path('/home/dm/DM_das/contact_scheduler_20261104_experiment_b_v4')


def build_launch(args, environ):
    root = args.scheduler_root.expanduser().resolve()
    entry = root / 'run_contact_261010.sh'
    if not entry.is_file():
        raise ValueError('Existing scheduler entry is missing: ' + str(entry))
    env = dict(environ)
    if args.command == 'run' and not args.real:
        raise ValueError('run requires --real; use check for offline validation')
    if args.command == 'run' and env.get('CONTACT_TABLETOP_CONFIRMED') != '1':
        raise ValueError('现场确认原瓶位、桌面支撑、初始全开空载后，才可设置 CONTACT_TABLETOP_CONFIRMED=1')
    if args.route == 'main':
        if args.offset_mm is not None or env.get('CONTACT_EXPERIMENT_B_OFFSET_MM'):
            raise ValueError('main route forbids experimental offsets; use --route pilot explicitly')
        if args.after_skill not in (None, 'handoff'):
            raise ValueError('main route must hand off after successful skill completion')
        if env.get('CONTACT_SKILL_AFTER', 'handoff') != 'handoff' or env.get('CONTACT_SKILL_RECOVERY', 'enabled') != 'enabled':
            raise ValueError('main route requires enabled recovery and handoff; clear conflicting CONTACT_SKILL_* settings')
        env.pop('CONTACT_EXPERIMENT_B_OFFSET_MM', None)
        env.update(CONTACT_SKILL_AFTER='handoff', CONTACT_SKILL_RECOVERY='enabled')
        mode = 'task-' + args.command
    else:
        offset = args.offset_mm
        if offset is None or not math.isfinite(offset) or not (offset == 0 or 3 <= abs(offset) <= 10):
            raise ValueError('pilot requires --offset-mm 0 or an offset with 3 <= abs(offset) <= 10')
        env.update(CONTACT_EXPERIMENT_B_OFFSET_MM=str(offset),
                   CONTACT_SKILL_AFTER=args.after_skill or 'stop', CONTACT_SKILL_RECOVERY='enabled')
        mode = 'skill-' + args.command
    if args.skill_config:
        from .primary_skill_export import validate_export_config
        selected = validate_export_config(args.skill_config)
        env['CONTACT_SCHEDULER_CONFIG'] = selected['scheduler_config']
    if args.checkpoint:
        checkpoint = args.checkpoint.expanduser().resolve()
        if not (checkpoint / 'config.json').is_file():
            raise ValueError('Checkpoint config.json missing: ' + str(checkpoint))
        env['CONTACT_MAIN_POLICY_PATH'] = str(checkpoint)
    if args.trial_id:
        env['CONTACT_SKILL_TRIAL_ID'] = args.trial_id
    if args.log_dir:
        env['CONTACT_SCHEDULER_LOG_DIR'] = str(args.log_dir.expanduser().resolve())
    return ['bash', str(entry), mode], env


def parser():
    p = argparse.ArgumentParser(description='a轴完整主任务：离线check、显式实机run、独立结果review和summary')
    sub = p.add_subparsers(dest='command', required=True)
    for name in ('check', 'run'):
        q = sub.add_parser(name)
        q.add_argument('--scheduler-root', type=Path, default=DEFAULT_SCHEDULER)
        q.add_argument('--route', choices=('main', 'pilot'), default='main')
        q.add_argument('--offset-mm', type=float)
        q.add_argument('--after-skill', choices=('stop', 'handoff'))
        q.add_argument('--checkpoint', type=Path)
        q.add_argument('--skill-config', type=Path, help='immutable scheduler.json from primary-skill-export; validated before launch')
        q.add_argument('--trial-id')
        q.add_argument('--log-dir', type=Path)
        if name == 'run':
            q.add_argument('--real', action='store_true', help='允许现场真机执行；本次开发不会调用')
    q = sub.add_parser('summary')
    q.add_argument('path', type=Path)
    q.add_argument('--scheduler-root', type=Path, default=DEFAULT_SCHEDULER)
    q = sub.add_parser('review')
    q.add_argument('episode', type=Path)
    q.add_argument('--outcome', required=True, choices=('success', 'failure', 'unknown'))
    q.add_argument('--note', required=True)
    q.add_argument('--scheduler-root', type=Path, default=DEFAULT_SCHEDULER)
    return p


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    args = parser().parse_args(argv or ['check'])
    try:
        if args.command in ('check', 'run'):
            command, env = build_launch(args, os.environ)
            print(json.dumps({'route': args.route, 'mode': args.command,
                              'hardware_execution_requested': args.command == 'run',
                              'scheduler': str(args.scheduler_root),
                              'main_checkpoint': env.get('CONTACT_MAIN_POLICY_PATH', 'existing_261010_default'),
                              'skill_config': env.get('CONTACT_SCHEDULER_CONFIG', 'existing_legacy_default'),
                              'after_skill': env['CONTACT_SKILL_AFTER']}, ensure_ascii=False), flush=True)
        else:
            env = dict(os.environ)
            root = args.scheduler_root.expanduser().resolve()
            if args.command == 'summary':
                command = [sys.executable, str(root/'integration/summarize_contact_run.py'), str(args.path)]
            else:
                command = [sys.executable, str(root/'integration/task_outcome.py'), str(args.episode),
                           '--outcome', args.outcome, '--notes', args.note]
            if not Path(command[1]).is_file():
                raise ValueError('Existing scheduler tool is missing: ' + command[1])
        return subprocess.run(command, env=env, check=False).returncode
    except (OSError, ValueError) as exc:
        print('[a-axis-task] ' + str(exc), file=sys.stderr)
        return 2
