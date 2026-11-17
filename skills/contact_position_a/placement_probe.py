"""Blind single-PROBE placement experiment. Only the 'probe --real' subcommand uses hardware."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import signal
import time
from types import SimpleNamespace

import numpy as np
import yaml

from bottle_scan.contact_policy import _digest
from bottle_scan.e1e2 import Runner
from bottle_scan.e1e2_core import validate
from bottle_scan.session import atomic_json
from .contact_reference import probe_protocol, validate_reference

MODEL_DIR = Path('outputs/white_offset_all_20261116_v1')
MODEL_ID = '19c20374214fddd4392c'
LABEL = 'fixed_reference_a_placement_probe_v1'


def read(path):
    return json.loads(Path(path).read_text())


def hardware_for_placement(cfg, expected_digest, profile):
    primary = Path(cfg['hardware_config'])
    if hashlib.sha256(primary.read_bytes()).hexdigest() != expected_digest:
        raise ValueError('Hardware config differs from the verified config')
    original = yaml.safe_load(primary.read_text())
    if profile == 'configured':
        return original['hardware']
    if profile != '0318_0347':
        raise ValueError('WHITE_PLACEMENT_X5 must be configured or 0318_0347')
    path = Path(__file__).with_name('config.ros.x5_0318_0347.yaml')
    selected = yaml.safe_load(path.read_text())
    serials = {'left':'X2620410318', 'right':'X2620410347'}
    original['hardware']['sensor_serials'] = serials
    original['hardware']['public_ros']['sensors']['serials'] = serials
    if selected != original:
        raise ValueError('Placement hardware profile may only change the two serial mappings')
    cfg['hardware_config'] = str(path.resolve())
    return selected['hardware']


def assets():
    ref = validate_reference(read('skills/contact_position_a/white_reference.json'))
    model = read(MODEL_DIR / 'offset_all.json')
    scheduler = read('config.state_graph.v4_scheduler.json')
    cfg = scheduler['execution_config']
    validate(cfg)
    hw = hardware_for_placement(cfg, scheduler['hardware_config_sha256'], os.environ.get('WHITE_PLACEMENT_X5','configured'))
    if (model['model_id'] != MODEL_ID or _digest(model) != model['integrity_sha256']
            or model['integrity_sha256'][:20] != MODEL_ID
            or model['contact_reference_digest'] != ref['reference_digest']
            or model['target_definition_id'] != ref['target_definition_id']
            or ref['protocol'] != probe_protocol(cfg)
            or hw['sensor_serials'] != hw['public_ros']['sensors']['serials']
            or cfg['gripper']['min_normalized'] < hw['public_ros']['raw_normalized_min']):
        raise ValueError('Wrong model, probe protocol, hardware or fixed reference')
    return cfg, ref, model, hw['sensor_serials']


def tag(value):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}', value):
        raise argparse.ArgumentTypeError('Use a simple placement or trial name')
    return value


def difference(start, end, axis):
    a, b = np.asarray(start, float), np.asarray(end, float)
    if a.shape != (4, 4) or b.shape != (4, 4) or not np.isfinite([a, b]).all():
        raise ValueError('Invalid TCP pose')
    d = a[:3,:3].T @ (b[:3,3] - a[:3,3])
    x = float(d @ axis)
    angle = np.degrees(np.arccos(np.clip((np.trace(a[:3,:3].T @ b[:3,:3]) - 1)/2, -1, 1)))
    return x, float(np.linalg.norm(d - x * axis)), float(angle)


def capture(args, cfg, ref, model, factory=None, runner_type=Runner, ask=input, *, finger_serials=None):
    # No asset is passed to the touch controller except its existing probe config.
    if not args.real:
        raise ValueError('Physical capture requires --real')
    print('本次只在当前位置轻触并松爪。瓶底放稳，全开空载，整段停止遥操、无其他控制器。')
    if ask('核对两指顺序、空载和现场退路，输入 PROBE 开始：').strip() != 'PROBE':
        return
    directory = args.run / args.placement / args.trial
    directory.mkdir(parents=True, exist_ok=False)
    path = directory / 'observation.json'
    row = dict(schema=LABEL, placement=args.placement, trial=args.trial, started_ns=time.time_ns(),
               target_definition_id=ref['target_definition_id'], reference_digest=ref['reference_digest'],
               model_id=model['model_id'], state='started', bottle_static=None)
    if finger_serials is not None:
        row['finger_serials'] = dict(finger_serials)
        row['hardware_config_sha256'] = hashlib.sha256(Path(cfg['hardware_config']).read_bytes()).hexdigest()
    atomic_json(path, row)
    if factory is None:
        from bottle_scan.e1e2_hardware import Hardware

        class ProbeHardware(Hardware):
            def probe_window(self, since):
                window = super().probe_window(since)
                if window:
                    self.window_pose = self.arm.snapshot()['arms']['right']
                return window

        factory = ProbeHardware
    hw = factory(cfg)
    try:
        hw.connect()
        runner = runner_type(SimpleNamespace(out=directory, cond='A', step=False), cfg, None, hw)
        hw.begin_record(directory / 'record')
        hw.set_context('ZERO', args.trial)
        row['unloaded_zero'] = hw.zero_unloaded()
        np.savez_compressed(directory / 'unloaded_dd.npz', left=hw.dd_reference[0], right=hw.dd_reference[1])
        row['before_pose'] = hw.pose()
        try:
            status, probe = runner.touch()
            row.update(status=status, probe=probe, touch_report=runner.last_touch_report,
                       window_pose=getattr(hw, 'window_pose', None), after_pose=hw.pose())
        finally:
            # Bottle is table-supported throughout. There is no lift or arm translation.
            hw.recovery_open()
        hw.check()
        hw.end_record()
        row['state'] = 'released'
    except BaseException as exc:
        row.update(state='fault', error=str(exc) or type(exc).__name__)
        raise
    finally:
        try:
            hw.close()
        finally:
            atomic_json(path, row)
    row['bottle_static'] = ask('瓶子整次保持原位、直立且全程未碰动？输入 yes 或 YES 确认，其他输入保留为未确认：').strip().casefold() == 'yes'
    atomic_json(path, row)
    print('已保存', path, row['status'], '瓶位确认', row['bottle_static'])


def load_replay():
    spec = importlib.util.spec_from_file_location('white_shadow_replay', MODEL_DIR / 'replay.py')
    lib = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lib)
    return lib.replay


def score(run, ref, model, predict=None):
    if predict is None:
        predict = load_replay()
    rows = []
    summaries = {}
    axis = np.asarray(ref['protocol']['axes_tcp']['a'])
    for group in sorted(p for p in run.iterdir() if p.is_dir()):
        records = sorted((read(p) for p in group.glob('*/observation.json')), key=lambda r:r['started_ns'])
        if not records:
            continue
        baseline = None
        origin = None
        datum = group / 'datum.json'
        evidence = group / 'datum_source.txt'
        if datum.exists() or evidence.exists():
            d = read(datum)
            if (not evidence.read_text().strip() or d['label'] != group.name + '_standard_witness'
                    or d['arm'] != 'right' or d['read_only'] is not True
                    or d['pose_source']['arms']['right']['tcp_frame'] != 'right_gripper_base_link'
                    or d['pose_source']['base_frame'] != 'base_link'):
                raise ValueError('Invalid standard witness '+str(group))
            origin = d['T_base_tcp_mm']
        group_rows = []
        valid_poses = {}
        continuity = True
        for record in records:
            if (record['schema'] != LABEL or record['model_id'] != MODEL_ID
                    or record['reference_digest'] != ref['reference_digest'] or record['placement'] != group.name):
                raise ValueError('Mixed reference, placement or model')
            # A later new mark is a new group; never silently resume a displaced block.
            continuity = continuity and record.get('bottle_static') is True
            r = dict(placement=group.name, trial=record['trial'], status=record.get('status',record['state']),
                     gate=None, raw_mm=None, accepted_mm=None, measured_change_mm=None,
                     predicted_change_mm=None, change_error_mm=None, absolute_truth_mm=None, absolute_error_mm=None)
            group_rows.append(r)
            probe = record.get('probe') or {}
            if record['state'] != 'released' or record.get('status') != 'contact' or not probe.get('valid'):
                r['gate'] = 'no_valid_probe'
                continue
            prediction = predict(model, probe['descriptors'])
            r.update(gate=prediction['sample_gate'], raw_mm=prediction['raw_offset_a_mm'],
                     accepted_mm=prediction['accepted_offset_a_mm'], uncertainty_mm=prediction['uncertainty_mm'])
            fb = record['window_pose']
            pose = fb['T_base_tcp_mm']
            r['contact_xyz_base_mm'] = np.asarray(pose)[:3,3].tolist()
            r['tcp_dd_skew_ms'] = max(abs(fb['received_monotonic_ns']-w['frame_ns'][-1]) for w in probe['window'])/1e6
            motion = difference(record['before_pose'], record['after_pose'], axis)
            if (not continuity or r['tcp_dd_skew_ms'] > 100 or np.hypot(*motion[:2]) > 1 or motion[2] > 1):
                r['metrology'] = 'unconfirmed_placement_or_motion_or_timestamp'
                continue
            valid_poses[record['trial']] = (pose, r['accepted_mm'])
            if record['trial'] == 'center':
                baseline = (pose,r['accepted_mm'])
                center_orientation = np.asarray(pose)[:3,:3].tolist()
            if baseline is not None:
                x, off, deg = difference(baseline[0], pose, axis)
                r.update(off_axis_mm=off, angle_deg=deg)
                if off <= 1 and deg <= 1:
                    r['measured_change_mm'] = x
                    if baseline[1] is not None and r['accepted_mm'] is not None:
                        r['predicted_change_mm'] = r['accepted_mm']-baseline[1]
                        r['change_error_mm'] = r['predicted_change_mm']-x
            if origin is not None and continuity:
                x, off, deg = difference(origin, pose, axis)
                if off <= 1 and deg <= 1:
                    r['absolute_truth_mm'] = x
                    if r['accepted_mm'] is not None:
                        r['absolute_error_mm'] = r['accepted_mm']-x
        # If the bottle moved, a witness acquired after the block cannot certify earlier contacts.
        if not continuity:
            for r in group_rows:
                r.update(absolute_truth_mm=None, absolute_error_mm=None)
        relative = [r['change_error_mm'] for r in group_rows if r['trial'] != 'center' and r['change_error_mm'] is not None]
        absolute = [r['absolute_error_mm'] for r in group_rows if r['absolute_error_mm'] is not None]
        summaries[group.name] = dict(attempts=len(group_rows), accepted=sum(r['accepted_mm'] is not None for r in group_rows),
                                     relative_pairs=len(relative), relative_change_MAE_mm=float(np.mean(np.abs(relative))) if relative else None,
                                     absolute_scored=len(absolute), absolute_MAE_mm=float(np.mean(np.abs(absolute))) if absolute else None,
                                     method=evidence.read_text().strip() if origin is not None else None,
                                     center_R=center_orientation if baseline is not None else None)
        if 'minus' in valid_poses and 'plus' in valid_poses:
            minus, plus = (valid_poses[k] for k in ('minus','plus'))
            dx, off, deg = difference(minus[0], plus[0], axis)
            if off <= 1 and deg <= 1 and minus[1] is not None and plus[1] is not None:
                summaries[group.name]['plus_minus_span_mm'] = {
                    'measured':dx, 'predicted':plus[1]-minus[1], 'error':plus[1]-minus[1]-dx}
        rows.extend(group_rows)
    if not rows:
        raise ValueError('No observations in '+str(run))
    valid = [(name,s['center_R']) for name,s in summaries.items() if s['center_R'] is not None]
    for name, rotation in valid:
        cosine = (np.trace(np.asarray(valid[0][1]).T @ rotation)-1)/2
        summaries[name]['orientation_difference_from_'+valid[0][0]+'_deg'] = float(np.degrees(np.arccos(np.clip(cosine,-1,1))))
    report = dict(model_id=MODEL_ID, reference_digest=ref['reference_digest'], placements=summaries, rows=rows)
    atomic_json(run/'report.json', report)
    def fmt(v):
        return '—' if v is None else f'{v:+.2f}'
    print('摆放  试次  门槛  接受的偏移  TCP实测变化  预测变化  变化误差  独立标准偏移  绝对偏移误差 (mm)')
    for r in rows:
        print(r['placement'], r['trial'], r['gate'], *[fmt(r[k]) for k in
              ('accepted_mm','measured_change_mm','predicted_change_mm','change_error_mm','absolute_truth_mm','absolute_error_mm')])
    for name, summary in summaries.items():
        print(name, json.dumps({k:v for k,v in summary.items() if k != 'center_R'}, ensure_ascii=False))
    print('详细结果', (run/'report.json').resolve())
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest='cmd', required=True)
    subs.add_parser('check')
    probe = subs.add_parser('probe')
    probe.add_argument('--run', type=Path, required=True)
    probe.add_argument('--placement', type=tag, required=True)
    probe.add_argument('--trial', type=tag, required=True)
    probe.add_argument('--real', action='store_true')
    report = subs.add_parser('score')
    report.add_argument('--run', type=Path, required=True)
    args = parser.parse_args(argv)
    cfg, ref, model, serials = assets()
    if args.cmd == 'check':
        print(json.dumps(dict(model=MODEL_ID, train_range_mm=model['trained_offset_range_mm'],
                              reference=ref['target_definition_id'], serials=serials,
                              axes=cfg['axes_tcp'], probe_sdk=cfg['force']['target']*cfg['force']['probe_fraction'],
                              floor=cfg['gripper']['min_normalized']), ensure_ascii=False, indent=2))
    elif args.cmd == 'probe':
        def interrupted(sig, frame):
            raise KeyboardInterrupt(f'signal {sig}')
        previous = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGTERM, signal.SIGHUP)}
        try:
            capture(args, cfg, ref, model, finger_serials=serials)
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)
    else:
        score(args.run, ref, model)


if __name__ == '__main__':
    main()
