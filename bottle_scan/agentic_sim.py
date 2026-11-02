"""Explicitly synthetic, deterministic plant for complete campaign rehearsals.

No ROS, camera connection, sockets or wall-clock sleeps.  Simulated records are
marked at session, trial and stream level and cannot be presented as real data.
"""
from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from .e1e2_core import MotionBounds, LABELS
from .e1e2 import rebuild_summary
from .session import atomic_json, plain


def default_T0():
    out = np.eye(4)
    out[:3, :3] = [[0., 0., 1.], [0., -1., 0.], [1., 0., 0.]]
    out[:3, 3] = [100., -290., 800.]
    return out


def simulated_probe(actual_delta, T0, bounds, rng, clock, *, bottle='light'):
    """Finite nonlinear tactile plant; inputs never reach correction.decide."""
    if actual_delta < -18.:
        return 'stall_no_contact', None
    if actual_delta > 14.:
        return 'no_contact', None
    descriptors = []
    # Different gains, weak curvature, repeat noise and a small bottle effect
    # make selection of repeats / new offsets observable in offline tests.
    for i, (intercept, slope, curvature) in enumerate([(64., -3.3, .04), (62., -3.1, .035)]):
        u = float(np.clip(intercept + slope * actual_delta + curvature * actual_delta ** 2 + rng.normal(0, .55) + (.15 if bottle == 'heavy' else 0), 3, 124))
        v = float(48 + rng.normal(0, .25))
        left, right = max(0, int(u - 7)), min(127, int(u + 7))
        edges = {'left': left < 3, 'right': right >= 125, 'top': False, 'bottom': False}
        descriptors.append({'valid': True, 'centroid_px': [u, v], 'area_px': 180,
                            'bbox_xyxy_px': [left, int(v-6), right, int(v+6)],
                            'principal_angle_deg': 5. + i, 'touches_edge': any(edges.values()),
                            'edge_sides': edges, 'shape_hw': [96, 128], 'synthetic': True})
    measured = bounds.grasp(actual_delta)
    return 'contact', {'valid': True, 'descriptors': descriptors, 'normal_sdk': [1.5, 1.5],
                       'timestamp_ns': clock, 'measured_tcp_mm': measured.tolist(),
                       'actual_delta_mm': float(actual_delta), 'synthetic': True,
                       'position_reference': 'synthetic measured TCP projection; no real robot',
                       'window': [{'frame_ns': [clock-j*20_000_000 for j in range(4, -1, -1)], 'frame_fids': list(range(5))} for _ in range(2)]}


def _raw_record(folder, row, T0, cfg):
    """Save inspectable synthetic stream arrays, not fictitious full-speed IO."""
    stamps = np.asarray([s['start_ns'] for s in row['stages']], dtype=np.int64)
    n = len(stamps)
    initial = row.get('probe_initial') or {}
    measured = np.asarray(initial.get('measured_tcp_mm', T0), dtype=float)
    dd = np.zeros((2, 96, 128, 3), np.uint16)
    for i, d in enumerate(initial.get('descriptors', [])):
        x1, y1, x2, y2 = d['bbox_xyxy_px']
        dd[i, y1:y2+1, x1:x2+1, 0] = 120
    normal = np.zeros((n, 2))
    for i, stage in enumerate(row['stages']):
        if stage['name'] in ('PROBE', 'SQUEEZE', 'LIFT', 'HOLD', 'LOWER'):
            normal[i] = cfg['force']['target'] * (.3 if stage['name'] == 'PROBE' else 1.)
    arrays = {'monotonic_ns': stamps, 'tcp_measured': np.repeat(measured[None], n, axis=0),
              'normal_force_sdk': normal, 'synthetic_dd_encoded': dd,
              'gripper_normalized': np.full(n, .36)}
    np.savez_compressed(folder / 'synthetic_streams.npz', **arrays)
    try:
        import h5py
    except ImportError:
        h5py = None
    if h5py is not None:
        with h5py.File(folder / 'streams.h5', 'w') as h:
            h.attrs['simulated'] = True
            h.attrs['sampling_policy'] = 'synthetic stage samples; not full-speed hardware data'
            for name in ('tcp_measured', 'normal_force_sdk', 'gripper_normalized'):
                group = h.create_group('streams/' + name)
                group.create_dataset('data', data=arrays[name])
                group.create_dataset('monotonic_ns', data=stamps)
                group.create_dataset('sequence', data=np.arange(n))
    atomic_json(folder / 'recording_status.json', {'complete': True, 'simulated': True,
                'sampling_policy': 'synthetic stage samples only', 'source': 'agentic_sim_v1'})
    atomic_json(folder / 'keyframes.json', {'simulated': True, 'images_generated': False,
                'events': [{'stage': stage['name'], 'timestamp_ns': stage['start_ns']} for stage in row['stages']]})


def execute_simulated_batch(root, plan, cfg, fit, context, *, ask=input, say=print):
    from .agentic_executor import _correction, deadline_reached
    root = Path(root)
    meta = json.loads((root / 'session_meta.json').read_text())
    context = copy.deepcopy(context)
    T0 = np.asarray(context.get('T0', default_T0()), float)
    bounds = MotionBounds(cfg, T0)
    context['T0'] = T0.tolist()
    context.setdefault('T0_accepted_unix_ns', time.time_ns())
    context.setdefault('zero_samples', [])
    context.setdefault('initial_centroids', None)
    context.update(airborne=False, unsafe_recovery=False)
    meta.update(T0=T0.tolist(), T0_accepted_unix_ns=context['T0_accepted_unix_ns'], simulated=True)
    completed = {r['id'] for r in rebuild_summary(root) if r.get('complete')}
    count = 0
    quit_requested = False
    interactive = bool(cfg.get('agentic', {}).get('simulation_interactive', False))
    clock = int(context.get('simulation_clock_ns', 1_000_000_000))
    seed = int(cfg['design']['seed'])
    say('SIMULATION：本批全部为合成数据；不连接 ROS，不执行真实运动。')
    for item in plan:
        if item['id'] in completed:
            continue
        if deadline_reached(context):
            quit_requested = True
            say('SIMULATION：时间预算已耗尽，不再启动下一次合成试验。')
            break
        token = f"{seed}:{root.name}:{item['id']}:{item['cond']}:{item['delta_mm']}:{item['bottle']}"
        rng = np.random.default_rng(int.from_bytes(hashlib.sha256(token.encode()).digest()[:8], 'little'))
        target = bounds.grasp(item['delta_mm'])
        pre = bounds.pre(target)
        reason = bounds.check(target) or bounds.check(pre, 'retreat', target) or bounds.check(bounds.lift(target), 'lift', target)
        folder = root / 'trials' / item['id']
        folder.mkdir(parents=True, exist_ok=True)
        row = {**item, 'simulated': True, 'complete': False, 'recording_complete': False,
               'started_unix_ns': time.time_ns(), 'status': 'started', 'bilateral': False,
               'target_reached': False, 'lifted': False, 'hold_complete': False, 'dropped': False,
               'regrasps': 0, 'servo_exhausted': False, 'note': '', 'label': 'normal',
               'config_identity': meta['config_identity'], 'stages': []}
        def stage(name):
            nonlocal clock
            clock += 300_000_000
            row['stages'].append({'name': name, 'start_ns': clock, 'end_ns': clock+100_000_000, 'simulated': True})
        if reason:
            row.update(status='skipped_bound', skipped_reason=reason, label='invalid')
        else:
            for name in ('RELEASE', 'MOVE_PRE', 'APPROACH', 'CLOSE', 'PROBE'):
                stage(name)
            actual = float(item['delta_mm'] + rng.normal(0, .08))
            status, probe = simulated_probe(actual, T0, bounds, rng, clock, bottle=item['bottle'])
            row['initial_status'] = status
            row['probe_initial'] = copy.deepcopy(probe or {})
            if context['initial_centroids'] is None:
                _, baseline = simulated_probe(0, T0, bounds, rng, clock)
                context['initial_centroids'] = baseline
            if item['cond'] == 'B':
                correction = _correction(fit, cfg['servo'])
                history = []
                for attempt in range(cfg['servo']['max_regrasps']+1):
                    decision = correction.decide(status, probe['descriptors'] if probe else [], status == 'contact')
                    event = {'attempt': attempt, 'status': status, 'timestamp_ns': clock, **decision}
                    history.append(event)
                    if decision.get('rejected'):
                        status, probe = 'policy_rejected', None
                        row['servo_exhausted'] = True
                        break
                    if decision['accepted']:
                        break
                    if attempt >= cfg['servo']['max_regrasps']:
                        row['servo_exhausted'] = True
                        break
                    proposed = target.copy()
                    proposed[:3, 3] += bounds.a * decision['delta_mm']
                    retreat = bounds.pre(target, cfg['servo']['retreat_mm'])
                    corrected_pre = bounds.pre(proposed, cfg['servo']['retreat_mm'])
                    error = bounds.check(proposed) or bounds.check(retreat, 'retreat', target) or bounds.check(corrected_pre, 'retreat', proposed)
                    if error:
                        event.update(blocked=True, blocked_reason=error, applied_delta_mm=0.)
                        row['servo_exhausted'] = True
                        break
                    for name in ('RELEASE', 'REGRASP_RETREAT', 'REGRASP_CORRECT', 'REGRASP_APPROACH', 'CLOSE', 'PROBE'):
                        stage(name)
                    event['applied_delta_mm'] = decision['delta_mm']
                    actual += decision['delta_mm'] + rng.normal(0, .04)
                    target = proposed
                    status, probe = simulated_probe(actual, T0, bounds, rng, clock, bottle=item['bottle'])
                    row['regrasps'] += 1
                    row.setdefault('probe_regrasps', []).append({'status': status, 'probe': probe, 'target_tcp_mm': target.tolist()})
                row['correction_history'] = history
            row['probe_final'] = copy.deepcopy(probe or {})
            row['final_tcp_target_mm'] = target.tolist()
            row['bilateral'] = status == 'contact'
            row['status'] = status
            if status == 'contact':
                stage('SQUEEZE')
                row['target_reached'] = abs(actual) <= 11.
                if row['target_reached']:
                    for name in ('LIFT', 'HOLD', 'LOWER'):
                        stage(name)
                    row.update(lifted=True, hold_complete=True, status='hold_complete')
                else:
                    row['status'] = 'force_not_reached'
            stage('RELEASE')
            stage('RETREAT')
            if item['delta_mm'] == 0 and row['probe_initial'].get('valid'):
                vals = [d['centroid_px'] for d in row['probe_initial']['descriptors']]
                row['zero_centroids_px'] = vals
                samples = context['zero_samples']
                if len(samples) >= 3:
                    baseline = context.setdefault('zero_baseline_px', np.mean([s['centroids_px'] for s in samples[:3]], axis=0).tolist())
                    # Known plant slopes are simulation-only instrumentation;
                    # real runs use the current learned inverse model.
                    errors = [(vals[i][0]-baseline[i][0])/s for i, s in enumerate((-3.3, -3.1))]
                    row['zero_drift'] = {'checked': True, 'errors_equivalent_mm': errors,
                                         'alarm': max(map(abs, errors)) > cfg['design']['drift_mm'],
                                         'slope_source': 'synthetic plant instrumentation'}
                else:
                    row['zero_drift'] = {'checked': False, 'reason': 'collecting_first_three_campaign_zeros'}
            _raw_record(folder, row, T0, cfg)
            row['recording_complete'] = True
            if interactive:
                while True:
                    value = ask('SIMULATION 标签：回车/1正常，2倾倒，3推动，4掉落，5滑动，0作废，n备注，q本次后暂停：\n> ').strip().lower()
                    if value == 'q':
                        quit_requested = True
                        continue
                    if value == 'n':
                        row['note'] = ask('备注：\n> ')
                        continue
                    if value in LABELS:
                        row['label'] = LABELS[value]
                        break
            if row.get('zero_centroids_px') and row['label'] != 'invalid':
                context['zero_samples'].append({'trial_path': str(folder.resolve()), 'centroids_px': row['zero_centroids_px']})
                if len(context['zero_samples']) >= 3:
                    context.setdefault('zero_baseline_px', np.mean([s['centroids_px'] for s in context['zero_samples'][:3]], axis=0).tolist())
            context['last_safe_pose'] = bounds.pre(target).tolist()
            context['last_safe_target'] = bounds.pre(target).tolist()
            context['last_grasp_target'] = target.tolist()
            context['last_bottle'] = item['bottle']
        row.update(complete=True, finished_unix_ns=time.time_ns())
        atomic_json(folder / 'meta.json', row)
        completed.add(item['id'])
        count += 1
        context['simulation_clock_ns'] = clock
        meta.update(agentic_context=plain(context), initial_centroids=context['initial_centroids'])
        atomic_json(root / 'session_meta.json', meta)
        atomic_json(root / 'progress.json', {'completed': sorted(completed), 'active': None, 'airborne': False, 'unsafe_recovery': False})
        if quit_requested:
            break
    rebuild_summary(root)
    return {'complete': len(completed) == len(plan), 'quit': quit_requested, 'context': plain(context),
            'completed_count': count, 'total_completed': len(completed), 'root': str(root), 'simulated': True,
            'deadline_reached': deadline_reached(context)}
