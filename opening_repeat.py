#!/usr/bin/env python3
"""Operator-started CLOSE+PROBE opening repeatability at T0; no arm commands."""
from __future__ import annotations
import argparse
import csv
import json
import math
from pathlib import Path
import sys

import h5py
import numpy as np

from bottle_scan.config import load_config, require_real_ready
from bottle_scan.force_control import close_quality_report, force_unit, force_report_binding
from bottle_scan.geometry import pose_error
from bottle_scan.session import Session, atomic_json, plain


def fit_manual_pairs(path):
    """Offline caliper measurements; retain measured knots, never auto-install fit."""
    from bottle_scan.hardware_gripper import WidthCalibration
    with Path(path).open(newline='', encoding='utf-8-sig') as stream:
        rows = list(csv.DictReader(stream))
    pairs = sorted(({'normalized': float(x['normalized']), 'width_mm': float(x['width_mm'])}
                    for x in rows), key=lambda x: x['normalized'])
    WidthCalibration(pairs)  # Enforces measured range and strictly increasing opening.
    x = np.asarray([p['normalized'] for p in pairs])
    y = np.asarray([p['width_mm'] for p in pairs])
    slope, intercept = np.polyfit(x, y, 1)
    residual = y - (slope*x+intercept)
    return {'source': str(Path(path).resolve()), 'pairs': pairs,
            'model': 'width_mm = slope_mm_per_normalized * normalized + intercept_mm',
            'slope_mm_per_normalized': float(slope), 'intercept_mm': float(intercept),
            'residual_mm': residual.tolist(), 'rmse_mm': float(np.sqrt(np.mean(residual**2))),
            'maximum_absolute_residual_mm': float(np.max(np.abs(residual))),
            'normalized_domain': [float(x.min()), float(x.max())],
            'automatic_config_update': False,
            'note': 'Review residuals; hardware calibration uses these measured knots, without extrapolation.'}


def distribution(values):
    values = np.asarray(values, float)
    if values.ndim != 1 or not values.size or not np.isfinite(values).all():
        raise ValueError('A nonempty finite distribution is required')
    return {'count': len(values), 'values': values.tolist(), 'mean': float(values.mean()),
            'std': float(values.std()), 'min': float(values.min()),
            'median': float(np.median(values)), 'max': float(values.max()),
            'range': float(np.ptp(values))}


def read_openings(path, cfg, simulated):
    """Native source feedback if available; derived values are explicitly labelled."""
    from bottle_scan.hardware_gripper import WidthCalibration
    with h5py.File(path, 'r') as handle:
        group = handle['streams/gripper']
        count = int(group.attrs['committed_count'])
        stamps, widths = group['monotonic_ns'][:count], group['data'][:count, 0]
        metadata = [json.loads(x) for x in group['metadata_json'][:count]]
    if not count:
        raise ValueError('No committed measured gripper feedback')
    calibration = None
    records = []
    for stamp, width, meta in zip(stamps, widths, metadata):
        value = meta.get('normalized')
        source = 'native_ROS_normalized_feedback'
        if value is None:
            if simulated:
                value = (float(width)-cfg['gripper']['min_mm'])/(cfg['gripper']['open_mm']-cfg['gripper']['min_mm'])
                source = 'synthetic_linear_min_open_mapping_not_a_physical_calibration'
            else:
                if calibration is None:
                    calibration = WidthCalibration(cfg['hardware']['gripper_width_calibration'])
                value = calibration.to_normalized(width)
                source = 'inverse_of_measured_width_calibration_not_native_feedback'
        if not math.isfinite(float(value)) or not 0 <= float(value) <= 1:
            raise ValueError('Invalid recorded normalized opening')
        records.append({'timestamp_ns': int(stamp), 'width_mm': float(width),
                        'normalized': float(value), 'source': source,
                        'source_sequence': meta.get('source_sequence'), 'stage': meta.get('stage')})
    return records


def cycle_measurements(meta, records, cfg):
    contacts = meta['auto']['contact_times_ns']
    measured = []
    for index, stamp in enumerate(contacts):
        if stamp is None:
            raise ValueError('Bilateral contact was not observed')
        # The closest received position event is recorded with its signed skew;
        # no claim of hardware synchronization is made.
        event = min(records, key=lambda x: abs(x['timestamp_ns']-stamp))
        skew = (event['timestamp_ns']-stamp)/1e9
        if abs(skew) > cfg['recording']['source_max_age_s']:
            raise ValueError('No sufficiently fresh opening event for contact')
        measured.append({'finger': ('left', 'right')[index], **event,
                         'contact_detected_ns': stamp, 'position_minus_contact_s': skew})
    return {'trial_id': meta['trial_id'], 'status': meta['status'], 'contacts': measured,
            'probe_not_settled': bool(meta['auto'].get('probe_not_settled', True)),
            'contact_peak': meta['auto'].get('contact_peak'), 'force_unit': force_unit(cfg)}


def run_cycle(runner, folder, index):
    """Exactly reuse guarded CLOSE and PROBE, then release; never move arms."""
    from bottle_scan.runtime import Acquisition, SafetyFault
    ident = 'opening_repeat_%03d' % index
    item = {'trial_id': ident, 'T_target': runner.session.T0.tolist(),
            'nominal_contact_region': 'body', 'delta_acl_mm': [0, 0, 0]}
    runner.meta = {'trial_id': ident, 'task': 'opening_repeat', 'status': 'running',
                   'T_target': item['T_target'], 'stages': [], 'simulated': runner.backend.simulated,
                   'auto': {'contact_times_ns': [None, None], 'contact_widths_mm': [None, None],
                            'dropped': False, 'slip_suspected': False, 'exceptions': []}}
    runner.backend.set_trial(item)
    runner.backend.drain_samples()
    runner.recorder = Acquisition(runner.backend, folder)
    cycle_actuated = False
    try:
        with runner.phase('RELEASE'):
            cycle_actuated = True
            runner.ensure_open()
        if not runner.close_contact():
            raise SafetyFault('Opening repeat did not establish bilateral contact')
        runner.probe()
        with runner.phase('RELEASE'):
            runner.ensure_open()
        runner.meta['status'] = 'completed'
    except BaseException as exc:
        runner.meta.update(status='aborted', error=str(exc))
        if cycle_actuated and runner.actuation_started:
            try:
                runner.backend.hold_gripper()
            except BaseException as stop_error:
                runner.meta['stop_error'] = str(stop_error)
        # No automatic release/arm retreat on failure.
        raise
    finally:
        try:
            runner.recorder.close()
        finally:
            runner.recorder = None
            if (folder/'streams.h5').exists():
                from bottle_scan.window_descriptors import enrich_trial_windows
                from bottle_scan.recording import finalize_gel_windows
                enrich_trial_windows(runner.meta, folder/'streams.h5', runner.cfg, runner.reference)
                finalize_gel_windows(folder, runner.meta, runner.cfg)
            atomic_json(folder/'meta.json', runner.meta)
    records = read_openings(folder/'streams.h5', runner.cfg, runner.backend.simulated)
    atomic_json(folder/'opening_samples.json', records)
    return cycle_measurements(runner.meta, records, runner.cfg), records


def require_first_single_step(session, enabled):
    if enabled:
        return
    confirmation = session.meta.get('opening_repeat_first_single_step', {})
    binding = force_report_binding(session.cfg)
    valid = (confirmation.get('operator_confirmation') == 'PASS' and
             confirmation.get('simulated') is False and confirmation.get('binding') == binding)
    if valid:
        try:
            report = json.loads(Path(confirmation['report_path']).read_text())
            if not isinstance(report, dict):
                raise ValueError('Opening report must be a JSON object')
            trials = report.get('trials', [])
            valid = (report.get('schema') == 'bottle_opening_repeat_v2' and
                     report.get('simulated') is False and report.get('completed') is True and
                     report.get('cleanup_complete') is True and report.get('operator_confirmation') == 'PASS' and
                     report.get('binding') == binding and report.get('repetitions') == len(trials) and
                     bool(trials) and all(trial.get('status') == 'completed' for trial in trials))
        except (KeyError, TypeError, ValueError, OSError):
            valid = False
    if not valid:
        raise ValueError('First real opening-repeat requires --single-step and operator PASS; prior aborted/synthetic/changed-config runs do not waive it')


def summarize(trials, records, cfg):
    values = [event['normalized'] for event in records]
    steps = np.abs(np.diff(values))
    steps = steps[steps > np.finfo(float).eps*8]
    return {'contact_normalized': {side: distribution([trial['contacts'][i]['normalized'] for trial in trials])
                                  for i, side in enumerate(('left', 'right'))},
            'opening_readout_resolution': {
                'observed_min_nonzero_delta_normalized': float(steps.min()) if steps.size else None,
                'method': 'smallest nonzero consecutive received feedback step; sampling and motion affect this estimate',
                'hardware_encoder_resolution_verified': False,
                'configured_measured_width_resolution_mm': cfg['close']['width_resolution_mm'],
                'sources': sorted(set(event['source'] for event in records))}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--real', action='store_true')
    modes.add_argument('--simulate', action='store_true')
    modes.add_argument('--dry-run', action='store_true')
    parser.add_argument('--demo', action='store_true')
    parser.add_argument('--config', default='config.ros.yaml')
    parser.add_argument('--session')
    parser.add_argument('--single-step', action='store_true')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--manual-pairs', type=Path, help='Offline CSV with normalized,width_mm measured caliper pairs')
    args = parser.parse_args(argv)
    if args.manual_pairs:
        if args.real or args.simulate or args.session or args.output is None:
            raise ValueError('--manual-pairs is offline and requires --output, without execution/session flags')
        if args.output.exists():
            raise FileExistsError('Refusing to overwrite mapping evidence: '+str(args.output))
        fitted = fit_manual_pairs(args.manual_pairs)
        atomic_json(args.output, fitted)
        print(json.dumps(fitted, ensure_ascii=False, indent=2))
        return 0
    if args.real and args.demo:
        raise ValueError('Synthetic configuration cannot execute hardware')
    session = Session(args.session) if args.session else None
    cfg = session.cfg if session else load_config(args.config, demo=args.demo or args.simulate)
    if session is not None and bool(session.meta['simulated']) != bool(args.simulate):
        raise ValueError('Cannot mix real/synthetic session and opening-repeat')
    options = cfg['opening_repeat']
    repetitions = options['repetitions']
    if type(repetitions) is not int or repetitions < 1:
        raise ValueError('opening_repeat.repetitions must be a positive integer')
    quality = close_quality_report(cfg)
    overview = {'sequence': ['OPEN', 'CLOSE', 'PROBE', 'RELEASE'], 'repetitions': repetitions,
                'arm_motion': 'none; operator must already place experimental arm at T0',
                'first_real_requires_single_step': True, 'close_quality': quality}
    print(json.dumps(overview, ensure_ascii=False, indent=2))
    if quality['errors']:
        raise ValueError('; '.join(quality['errors']))
    if not (args.real or args.simulate):
        print('DRY-RUN：未导入 ROS、未连接设备、未发命令。')
        return 0
    if args.real:
        if session is None:
            raise ValueError('Real opening-repeat requires a prepared --session')
        require_real_ready(cfg)
        require_first_single_step(session, args.single_step)
        if session.progress.get('requires_operator_recovery'):
            raise ValueError('Operator recovery latch prevents opening-repeat')
    destination = args.output or Path(options['report_path'])
    if destination.exists():
        raise FileExistsError('Refusing to overwrite opening evidence: '+str(destination))
    run_dir = destination.parent/(destination.stem+'_data')
    run_dir.mkdir(parents=True, exist_ok=False)
    if session is None:
        session = Session.create(run_dir/'synthetic_session', cfg, np.asarray(cfg['T0']), simulated=True)
    from bottle_scan.backend import SimBackend, load_hardware
    from bottle_scan.runtime import Runner
    backend = None
    report = {'schema': 'bottle_opening_repeat_v2', 'simulated': bool(args.simulate), **overview,
              'operator_confirmation': None, 'completed': False, 'cleanup_complete': False, 'binding': force_report_binding(cfg),
              'trials': [], 'unit': 'normalized_opening_0_1'}
    atomic_json(run_dir/'effective_config.json', cfg)
    all_records = []
    register_single_step = False
    try:
        backend = load_hardware(cfg) if args.real else SimBackend(cfg, session.T0)
        if args.real:
            backend.prepare_zero(session, confirm=input)
            from bottle_scan.park_guard import ParkGuard
            ParkGuard(session).check(backend.park_observation(), now_ns=backend.monotonic_ns())
        runner = Runner(session, backend, single_step=args.single_step, unattended=bool(args.simulate))
        error = pose_error(np.asarray(runner._healthy_snapshot()['tcp']), session.T0)
        if error[0] > cfg['motion']['pose_tolerance_mm'] or error[1] > cfg['motion']['pose_tolerance_deg']:
            raise ValueError('Use move-tcp to return to measured T0; opening-repeat never moves either arm')
        if args.real and input('将在当前 T0 执行闭合—轻触—松开；输入 MOVE 开始：').strip() != 'MOVE':
            raise ValueError('Operator cancelled before actuation')
        # Check the actual pose again after the prompt, before any gripper action.
        error = pose_error(np.asarray(runner._healthy_snapshot()['tcp']), session.T0)
        if error[0] > cfg['motion']['pose_tolerance_mm'] or error[1] > cfg['motion']['pose_tolerance_deg']:
            raise ValueError('TCP changed during confirmation; no opening cycle started')
        for index in range(repetitions):
            folder = run_dir/('cycle_%03d' % (index+1)); folder.mkdir()
            trial, records = run_cycle(runner, folder, index+1)
            report['trials'].append(trial); all_records.extend(records)
            atomic_json(destination, report)
        report.update(summarize(report['trials'], all_records, cfg), completed=True)
        print(json.dumps(plain(report), ensure_ascii=False, indent=2))
        if args.real and input('核对完整开度重复性摘要；输入 PASS 确认本次结果：').strip() == 'PASS':
            report['operator_confirmation'] = 'PASS'
            register_single_step = bool(args.single_step)
        elif args.simulate:
            report['operator_confirmation'] = 'SIMULATION_ONLY'
        with destination.with_suffix('.csv').open('x', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=['trial_id','finger','normalized','width_mm','source','position_minus_contact_s','probe_not_settled'])
            writer.writeheader()
            for trial in report['trials']:
                for event in trial['contacts']:
                    writer.writerow({'trial_id': trial['trial_id'], 'probe_not_settled': trial['probe_not_settled'],
                                     **{key: event[key] for key in writer.fieldnames if key in event}})
    except BaseException as exc:
        report.update(completed=False, error=str(exc))
        raise
    finally:
        try:
            if backend is not None:
                try:
                    report['hardware_provenance'] = plain(backend.provenance()) if hasattr(backend,'provenance') else {'simulated': True}
                finally:
                    backend.close()
                report['cleanup_complete'] = True
        except BaseException as exc:
            report.update(completed=False, cleanup_error=str(exc))
            raise
        finally:
            # This must succeed before any first-run permission is committed.
            atomic_json(destination, report)
    if register_single_step:
        session.meta['opening_repeat_first_single_step'] = {
            'operator_confirmation': 'PASS', 'simulated': False, 'report_path': str(destination.resolve()),
            'binding': report['binding']}
        session.save_meta()
    print('Opening repeatability report: '+str(destination))
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, OSError, KeyboardInterrupt) as exc:
        print('STOP: '+str(exc), file=sys.stderr)
        raise SystemExit(2)
