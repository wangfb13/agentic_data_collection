"""Offline evidence, active experiment proposals and tactile-only correction.

This module never imports ROS, opens a device or publishes a command.  Planned
offsets are permitted in the experiment planner and offline evaluation only;
``AgenticCorrection`` accepts tactile observations, contact state and a frozen
model, never an experiment's hidden offset.  Every split is by physical trial
group, not the many frames recorded during that trial.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def _finite(value):
    try:
        return bool(np.all(np.isfinite(np.asarray(value, dtype=float))))
    except (TypeError, ValueError):
        return False


def _section(settings, name):
    return settings.get(name, {})


def _rotation_deg(a, b):
    return float(np.degrees(np.arccos(np.clip((np.trace(a.T @ b) - 1) / 2, -1, 1))))


def _session_role(meta):
    agentic = meta.get('agentic', {})
    return agentic.get('role', agentic.get('batch_role', meta.get('agentic_role', 'exploration')))


def _signature(meta):
    cfg = meta.get('config', {})
    # Plan, wall clock, file path and learned correction gain can differ. Force
    # and DD definitions, hardware identity, TCP definition cannot silently differ.
    return {**{key: cfg.get(key) for key in
            ('arm', 'axes_tcp', 'force', 'tactile', 'hold_s')},
        'hardware_hash': meta.get('hardware_hash', cfg.get('hardware_hash')),
        'tcp_definition': meta.get('tcp_definition', cfg.get('tcp_definition')),
        'simulated': bool(meta.get('simulated', False)),
    }


def _pose(value):
    if value is None or not _finite(value):
        return None
    array = np.asarray(value, float)
    if array.shape != (4, 4) or not np.allclose(array[3], [0, 0, 0, 1]):
        return None
    if not np.allclose(array[:3, :3].T @ array[:3, :3], np.eye(3), atol=1e-4):
        return None
    if np.linalg.det(array[:3, :3]) < .99:
        return None
    return array


def _axis(meta):
    T = _pose(meta.get('T0'))
    a = meta.get('config', {}).get('axes_tcp', {}).get('a')
    if T is None or a is None or not _finite(a):
        return None
    a = np.asarray(a, float)
    if a.shape != (3,) or abs(float(np.linalg.norm(a)) - 1.) > 1e-3:
        return None
    return T[:3, :3] @ a


def _load(roots, settings):
    sessions, trials, issues = [], [], []
    seen = set()
    physical_seen = {}
    for root in roots:
        root = Path(root).expanduser().resolve()
        if str(root) in seen:
            continue
        seen.add(str(root))
        path = root / 'session_meta.json'
        try:
            raw = path.read_bytes()
            meta = json.loads(raw)
        except (OSError, ValueError) as exc:
            issues.append(f'{root}: cannot read session metadata: {exc}')
            continue
        if meta.get('schema') != 'e1e2_session_v1':
            issues.append(f'{root}: unsupported session schema')
            continue
        expected_sim = bool(settings.get('simulate', False))
        if bool(meta.get('simulated', False)) != expected_sim:
            issues.append(f'{root}: simulation/real evidence mismatch')
            continue
        sess = {'path': str(root), 'meta': meta, 'signature': _signature(meta),
                'metadata_sha256': hashlib.sha256(raw).hexdigest(), 'role': _session_role(meta)}
        sessions.append(sess)
        for folder in ('trials', 'interrupted'):
            for p in sorted((root / folder).glob('*/meta.json')):
                try:
                    raw = p.read_bytes()
                    row = json.loads(raw)
                    status = p.parent / 'recording_status.json'
                    recorded = bool(row.get('recording_complete') and status.exists() and
                                    json.loads(status.read_text()).get('complete'))
                    canonical_trial = str(p.parent.resolve())
                    if meta.get('skill_label_review'):
                        from .skill_label_review import verify_view_row
                        canonical_session, canonical_trial = verify_view_row(root,meta,row,p)
                        row['canonical_session'] = canonical_session
                        row['canonical_trial_path'] = canonical_trial
                    if canonical_trial in physical_seen:
                        issues.append('duplicate physical trial evidence: ' + canonical_trial +
                                      ' appears in both ' + physical_seen[canonical_trial] + ' and ' + str(p.parent))
                        continue
                    physical_seen[canonical_trial] = str(p.parent)
                    row.update(session=str(root), trial_path=str(p.parent),
                               metadata_sha256=hashlib.sha256(raw).hexdigest(),
                               archived_attempt=folder == 'interrupted', recording_verified=recorded,
                               session_meta=meta, session_role=sess['role'])
                    trials.append(row)
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    issues.append(f'{p}: cannot read trial evidence: {exc}')
    return sessions, trials, issues


def _compatibility(sessions, settings):
    issues, evidence = [], []
    if not sessions:
        return ['no readable compatible sessions'], evidence
    ref = sessions[0]
    Tref, aref = _pose(ref['meta'].get('T0')), _axis(ref['meta'])
    if Tref is None or aref is None:
        issues.append('reference session has invalid or missing T0/axis')
        return issues, evidence
    learning = _section(settings, 'learning')
    max_pos = float(learning.get('t0_tolerance_mm', 2.))
    max_rot = float(learning.get('t0_tolerance_deg', 1.))
    signature = _hash(ref['signature'])
    for session in sessions:
        T, a = _pose(session['meta'].get('T0')), _axis(session['meta'])
        reasons = []
        if _hash(session['signature']) != signature:
            reasons.append('force/DD/arm/axis/hardware protocol differs')
        position_error, angle_error = None, None
        if T is None or a is None:
            reasons.append('invalid T0/axis')
        else:
            position_error = float(np.linalg.norm(T[:3, 3] - Tref[:3, 3]))
            angle_error = _rotation_deg(Tref[:3, :3], T[:3, :3])
            if position_error > max_pos or angle_error > max_rot:
                reasons.append('T0 differs beyond declared tolerance; re-establish common reference')
        if reasons:
            issues.append(session['path'] + ': ' + '; '.join(reasons))
        evidence.append({'session': session['path'], 'signature': _hash(session['signature']),
                         'T0_distance_mm': position_error, 'T0_rotation_deg': angle_error,
                         'compatible': not reasons, 'reasons': reasons})
    return issues, evidence


def _measured_probe(row, reference, settings):
    """Use measured PROBE-time TCP, including H5 on legacy sessions, not command.

    Return coordinates in the SAME reference T0 for all sessions.  A session's
    independently recorded T0 must not silently reset the mapping zero.
    """
    probe = row.get('probe_initial', {})
    Tref, a = _pose(reference.get('T0')), _axis(reference)
    if Tref is None or a is None:
        return {'x_mm': None, 'source': 'unavailable', 'reason': 'invalid reference'}
    Tlocal = _pose(row['session_meta'].get('T0'))
    if Tlocal is None:
        return {'x_mm': None, 'source': 'unavailable', 'reason': 'invalid session reference'}
    learning = _section(settings, 'learning')
    poses, source, detail = None, None, None
    stamps = [int(t) for win in probe.get('window', []) for t in win.get('frame_ns', [])]
    h5 = Path(row['trial_path']) / 'streams.h5'
    if h5.exists() and (stamps or probe.get('timestamp_ns')):
        try:
            import h5py  # Deliberately lazy: numerical tests need no hardware deps.
            with h5py.File(h5, 'r') as handle:
                group = handle['streams/tcp_measured']
                ts = np.asarray(group['monotonic_ns'], np.int64)
                lo = min(stamps) if stamps else int(probe['timestamp_ns'])
                hi = max(stamps) if stamps else lo
                indices = np.flatnonzero((ts >= lo) & (ts <= hi))
                if not len(indices) and len(ts):
                    closest = int(np.argmin(np.abs(ts - (lo + hi) // 2)))
                    if abs(int(ts[closest]) - (lo + hi) // 2) <= int(learning.get('tcp_max_age_s', .25) * 1e9):
                        indices = np.asarray([closest])
                if len(indices):
                    poses = np.asarray(group['data'][indices.tolist()], float)
                    source = 'measured_tcp_probe_window'
                    detail = {'file': str(h5), 'first_ns': int(ts[indices[0]]),
                              'last_ns': int(ts[indices[-1]]), 'n': len(indices)}
        except (ImportError, OSError, KeyError, ValueError) as exc:
            detail = {'h5_unavailable': str(exc)}
    if poses is None:
        inline = _pose(probe.get('measured_tcp_mm'))
        if inline is not None:
            poses = inline[None]
            source = 'measured_tcp_probe_end'
    if poses is not None:
        if poses.ndim != 3 or poses.shape[1:] != (4, 4) or not _finite(poses):
            return {'x_mm': None, 'source': 'invalid_measured_tcp', 'reason': 'invalid TCP matrix'}
        # Median translation in an already fixed rotation is robust to brief
        # feedback jitter; never infer the true position from the target pose.
        xyz = np.median(poses[:, :3, 3], axis=0)
        delta = xyz - Tref[:3, 3]
        x = float(delta @ a)
        off_axis = float(np.linalg.norm(delta - a * x))
        spread = float(np.max(np.linalg.norm(poses[:, :3, 3] - xyz, axis=1)))
        angle = max(_rotation_deg(Tref[:3, :3], p[:3, :3]) for p in poses)
        reason = None
        if off_axis > float(learning.get('max_cross_axis_mm', 2.)):
            reason = 'measured TCP outside one-dimensional a scan'
        elif spread > float(learning.get('max_probe_motion_mm', 1.)):
            reason = 'TCP not stable during PROBE window'
        elif angle > float(learning.get('t0_tolerance_deg', 1.)):
            reason = 'rotation changed during scan'
        return {'x_mm': x if not reason else None, 'source': source, 'off_axis_mm': off_axis,
                'spread_mm': spread, 'rotation_deg': angle, 'reason': reason, 'detail': detail}
    # Legacy summaries are still useful for choosing WHERE to reacquire; they
    # cannot establish the campaign's precision stopping criterion.
    shift = float((Tlocal[:3, 3] - Tref[:3, 3]) @ a)
    planned = float(row['delta_mm'])
    if _finite(planned):
        return {'x_mm': planned + shift, 'source': 'planned_delta_fallback',
                'reason': 'actual PROBE TCP unavailable; precision not verified', 'detail': detail}
    return {'x_mm': None, 'source': 'unavailable', 'reason': 'nonfinite offset'}


def _outcome_eligible(row):
    return bool(row.get('complete') and not row.get('archived_attempt') and
                row.get('recording_verified') and row.get('kind') == 'scan' and
                row.get('label') != 'invalid' and row.get('status') != 'skipped_bound')


def _success(row):
    return bool(row.get('lifted') and row.get('hold_complete') and
                not row.get('dropped') and row.get('label') == 'normal')


def _descriptor_valid(desc):
    return bool(desc and desc.get('valid') and _finite(desc.get('centroid_px')) and
                np.asarray(desc.get('centroid_px')).shape == (2,))


def _observations(rows, reference, settings):
    observations, excluded = [], Counter()
    for row in rows:
        reason = None
        if row.get('cond') != 'A':
            reason = 'B_diagnostic_only'
        elif row.get('session_role') in ('validation', 'frozen_validation', 'final_validation'):
            reason = 'frozen_validation_never_trains'
        elif not row.get('complete') or row.get('archived_attempt'):
            reason = 'incomplete_or_archived'
        elif not row.get('recording_verified'):
            reason = 'incomplete_recording'
        elif row.get('kind') not in ('scan', 'zero_monitor'):
            reason = 'not_initial_scan'
        elif row.get('label') in ('invalid', 'tipped', 'pushed'):
            reason = 'invalid_or_object_moved'
        elif not row.get('probe_initial', {}).get('valid'):
            reason = 'no_valid_initial_probe'
        else:
            descriptors = row['probe_initial'].get('descriptors', [])
            if len(descriptors) != 2 or not all(_descriptor_valid(d) for d in descriptors):
                reason = 'invalid_descriptor'
        if reason:
            excluded[reason] += 1
            continue
        measured = _measured_probe(row, reference, settings)
        if measured['x_mm'] is None:
            excluded[measured.get('reason') or 'unmeasured_position'] += 1
            continue
        observations.append({
            'id': row.get('id'), 'session': row['session'], 'trial_path': row['trial_path'],
            'metadata_sha256': row['metadata_sha256'], 'x_mm': measured['x_mm'],
            'planned_delta_mm': float(row['delta_mm']), 'measurement': measured,
            'bottle': row.get('bottle', 'unknown'), 'repeat': row.get('repeat'),
            'kind': row.get('kind'), 'descriptors': row['probe_initial']['descriptors'],
            'canonical_session': row.get('canonical_session',row['session']),
            'group': f"{row.get('canonical_session',row['session'])}|{row.get('bottle')}|{row.get('repeat') if row.get('repeat') is not None else 'monitor_' + str(row.get('id'))}",
        })
    return observations, dict(excluded)


def _pava(values, weights):
    """Weighted isotonic regression; no scipy/sklearn dependency."""
    blocks = []
    for i, (value, weight) in enumerate(zip(values, weights)):
        blocks.append([i, i, float(weight), float(value) * float(weight)])
        while len(blocks) > 1 and blocks[-2][3] / blocks[-2][2] > blocks[-1][3] / blocks[-1][2]:
            right, left = blocks.pop(), blocks.pop()
            blocks.append([left[0], right[1], left[2] + right[2], left[3] + right[3]])
    result = np.empty(len(values))
    for lo, hi, weight, total in blocks:
        result[lo:hi + 1] = total / weight
    return result


def _fit_finger(obs, finger, family, anchored, learning):
    clean = [o for o in obs if not o['descriptors'][finger].get('touches_edge')]
    if len(clean) < 4 or len(set(o['planned_delta_mm'] for o in clean)) < int(learning.get('min_levels', 3)):
        return None
    x = np.asarray([o['x_mm'] for o in clean])
    if np.ptp(x) < .1:
        return None
    options = []
    for idx, coordinate in enumerate(('u', 'v')):
        y = np.asarray([o['descriptors'][finger]['centroid_px'][idx] for o in clean])
        slope, intercept = np.polyfit(x, y, 1)
        if not _finite([slope, intercept]) or abs(slope) < float(learning.get('min_slope_px_per_mm', .05)):
            continue
        anchor = None
        if anchored:
            limit = float(learning.get('zero_anchor_tolerance_mm', .5))
            zeros = [o for o in clean if o['planned_delta_mm'] == 0 and
                     o['measurement']['source'].startswith('measured_tcp') and abs(o['x_mm']) <= limit]
            if len(zeros) < int(learning.get('min_zero_anchor', 3)):
                continue
            ax = float(np.mean([o['x_mm'] for o in zeros]))
            ay = float(np.mean([o['descriptors'][finger]['centroid_px'][idx] for o in zeros]))
            denominator = float(np.sum((x - ax) ** 2))
            if denominator < 1e-8:
                continue
            slope = float(np.sum((x - ax) * (y - ay)) / denominator)
            intercept = ay - slope * ax
            anchor = {'verified': True, 'n': len(zeros), 'measured_a_mm': ax,
                      'centroid_px': ay, 'max_abs_mm': max(abs(o['x_mm']) for o in zeros),
                      'meaning': 'measured near-zero anchor, not forced commanded-zero ground truth'}
        if abs(slope) < float(learning.get('min_slope_px_per_mm', .05)):
            continue
        sign = 1 if slope > 0 else -1
        if family == 'linear':
            kx = np.asarray([float(x.min()), float(x.max())])
            ky = slope * kx + intercept
            prediction = slope * x + intercept
        else:
            groups = defaultdict(list)
            for i, o in enumerate(clean):
                groups[o['planned_delta_mm']].append(i)
            bins = sorted([(float(np.mean(x[ind])), float(np.median(y[ind])), len(ind))
                           for ind in groups.values()])
            kx = np.asarray([g[0] for g in bins]); ky = np.asarray([g[1] for g in bins])
            ky = sign * _pava(sign * ky, [g[2] for g in bins])
            # A plateau has no unique inverse. Remove its interior and retain its
            # weighted midpoint; no fabricated epsilon slope to make it invertible.
            pooled = []
            for xp, yp, weight in zip(kx, ky, [g[2] for g in bins]):
                if pooled and abs(yp - pooled[-1][1]) < 1e-9:
                    ox, oy, ow = pooled[-1]
                    pooled[-1] = ((ox * ow + xp * weight) / (ow + weight), oy, ow + weight)
                else:
                    pooled.append((float(xp), float(yp), weight))
            kx = np.asarray([g[0] for g in pooled]); ky = np.asarray([g[1] for g in pooled])
            if len(kx) < 3 or np.any(np.diff(kx) <= 0) or np.any(sign * np.diff(ky) <= 0):
                continue
            if np.any(np.abs(np.diff(ky) / np.diff(kx)) < float(learning.get('min_slope_px_per_mm', .05))):
                continue
            prediction = np.interp(x, kx, ky)
        residual = (y - prediction) / abs(slope)
        fit = {'finger': ('left', 'right')[finger], 'coordinate': coordinate, 'family': family,
               'direction_sign': sign, 'slope_px_per_mm': float(slope), 'intercept_px': float(intercept),
               'knots_delta_mm': kx.tolist(), 'knots_centroid_px': ky.tolist(),
               'linear_range_mm': [float(kx[0]), float(kx[-1])], 'n': len(clean),
               'zero_anchor': anchor, 'training_rmse_equivalent_mm': float(np.sqrt(np.mean(residual ** 2))),
               'root_edge': ('right' if coordinate == 'u' else 'bottom') if sign > 0 else ('left' if coordinate == 'u' else 'top')}
        options.append(fit)
    return min(options, key=lambda f: f['training_rmse_equivalent_mm']) if options else None


def _fit(obs, family, anchored, learning):
    fingers = [_fit_finger(obs, finger, family, anchored, learning) for finger in range(2)]
    if any(f is None for f in fingers):
        return None
    low = max(f['knots_delta_mm'][0] for f in fingers)
    high = min(f['knots_delta_mm'][-1] for f in fingers)
    if high <= low:
        return None
    return {'schema': 'agentic_fit_v1', 'family': family, 'anchored': anchored,
            'fingers': fingers, 'common_range_mm': [low, high], 'common_linear_range_mm': [low, high],
            'usable_for_B': True, 'inverse_policy': 'reject_outside_observed_domain',
            'initial_A_only': True, 'training_observation_count': len(obs)}


def finger_estimate(finger_fit, centroid_px):
    """Invert a single finger only inside its observed, strictly monotone span."""
    if not _finite(centroid_px) or np.asarray(centroid_px).shape != (2,):
        return None
    coord = finger_fit.get('coordinate')
    if coord not in ('u', 'v'):
        return None
    x = np.asarray(finger_fit.get('knots_delta_mm', []), float)
    y = np.asarray(finger_fit.get('knots_centroid_px', []), float)
    if len(x) < 2 or len(x) != len(y) or not _finite([x, y]) or np.any(np.diff(x) <= 0):
        return None
    sign = finger_fit.get('direction_sign')
    if sign not in (-1, 1) or np.any(sign * np.diff(y) <= 0):
        return None
    v = float(centroid_px[0 if coord == 'u' else 1])
    if v < float(y.min()) - 1e-9 or v > float(y.max()) + 1e-9:
        return None
    return float(np.interp(v, y if sign == 1 else y[::-1], x if sign == 1 else x[::-1]))


def model_error(model, descriptors):
    """Tactile-only estimated displacement in mm, or None when unobservable."""
    fingers = model.get('fingers', [])
    if len(fingers) != 2 or len(descriptors or []) != 2 or not all(_descriptor_valid(d) for d in descriptors):
        return None
    estimates = [finger_estimate(f, d['centroid_px']) for f, d in zip(fingers, descriptors)]
    return float(np.mean(estimates)) if all(v is not None for v in estimates) else None


class AgenticCorrection:
    """Frozen-model controller: this API intentionally has no delta/pose input."""
    def __init__(self, fit, settings):
        self.fit, self.cfg = deepcopy(fit), deepcopy(settings)
        if fit.get('schema') != 'agentic_fit_v1' or fit.get('usable_for_B') is not True:
            raise ValueError('Agentic fit is not approved for B')
        if len(fit.get('fingers', [])) != 2:
            raise ValueError('Two finger models required')
        for f in fit['fingers']:
            x, y = f.get('knots_delta_mm', []), f.get('knots_centroid_px', [])
            if (len(x) < 2 or len(x) != len(y) or not _finite([x, y]) or f.get('coordinate') not in ('u', 'v') or
                    f.get('direction_sign') not in (-1, 1) or np.any(np.diff(x) <= 0) or
                    np.any(f['direction_sign'] * np.diff(y) <= 0)):
                raise ValueError('Invalid or non-invertible finger model')
        self.gain = float(settings.get('gain', settings.get('correction_gain', 1.)))
        if not .5 <= self.gain <= 1.:
            raise ValueError('Correction gain must be in [0.5, 1.0]')

    def estimate(self, descriptors):
        return model_error(self.fit, descriptors)

    def decide(self, status, descriptors, bilateral):
        e = self.estimate(descriptors)
        edges = []
        for desc, fit in zip(descriptors or [], self.fit['fingers']):
            if not desc:
                continue
            positive, negative = ('right', 'left') if fit['coordinate'] == 'u' else ('bottom', 'top')
            root, tip = (positive, negative) if fit['direction_sign'] > 0 else (negative, positive)
            sides = desc.get('edge_sides', {})
            if sides.get(root):
                edges.append('root')
            if sides.get(tip):
                edges.append('tip')
            if desc.get('touches_edge') and not (sides.get(root) or sides.get(tip)):
                edges.append('transverse')
        if bilateral and e is not None and abs(e) <= self.cfg['tol_mm'] and not any(d.get('touches_edge') for d in descriptors):
            return {'accepted': True, 'e_mm': e, 'delta_mm': 0., 'edges': edges, 'reason': 'within_tolerance', 'gain': self.gain}
        if status == 'no_contact':
            delta, reason = self.cfg['no_contact_delta_mm'], 'no_contact_forward'
        elif status == 'stall_no_contact':
            delta, reason = self.cfg['stall_delta_mm'], 'stall_back'
        elif 'root' in edges and 'tip' not in edges:
            delta, reason = self.cfg['root_edge_delta_mm'], 'root_edge'
        elif 'tip' in edges and 'root' not in edges:
            delta, reason = self.cfg['tip_edge_delta_mm'], 'tip_edge'
        elif e is not None:
            delta, reason = -self.gain * e, 'centroid_error'
        else:
            delta, reason = 0., 'unobservable_or_outside_model_domain'
        return {'accepted': False, 'e_mm': e, 'delta_mm': float(np.clip(delta, -self.cfg['max_delta_mm'], self.cfg['max_delta_mm'])),
                'edges': edges, 'reason': reason, 'gain': self.gain}


def _folds(obs):
    sessions = sorted(set(o['session'] for o in obs))
    if len(sessions) > 1:
        groups = {s: i % min(len(sessions), 5) for i, s in enumerate(sessions)}
        return [groups[o['session']] for o in obs], 'held_out_session'
    independent = sorted(set(o['group'] for o in obs))
    groups = {s: i % min(len(independent), 5) for i, s in enumerate(independent)} if independent else {}
    return [groups[o['group']] for o in obs], 'held_out_bottle_repetition_group'


def _metrics(errors, total):
    a = np.abs(np.asarray(errors, float))
    return {'n': len(a), 'total': total, 'validation_coverage': len(a) / total if total else 0.,
            'mae_mm': float(a.mean()) if len(a) else None,
            'p90_mm': float(np.quantile(a, .9)) if len(a) else None,
            'bias_mm': float(np.mean(errors)) if len(a) else None,
            'rmse_mm': float(np.sqrt(np.mean(a ** 2))) if len(a) else None}


def _cross_validate(obs, family, anchored, settings):
    assignment, method = _folds(obs)
    errors, predictions, group_errors = [], [], defaultdict(list)
    if len(set(assignment)) < 2:
        return {**_metrics([], len(obs)), 'method': method, 'reason': 'need independent trial groups'}, []
    for fold in sorted(set(assignment)):
        train = [o for o, f in zip(obs, assignment) if f != fold]
        test = [o for o, f in zip(obs, assignment) if f == fold]
        fit = _fit(train, family, anchored, _section(settings, 'learning'))
        for o in test:
            estimate = model_error(fit, o['descriptors']) if fit else None
            error = None if estimate is None else float(estimate - o['x_mm'])
            predictions.append({'session': o['session'], 'id': o['id'], 'group': o['group'],
                                'x_mm': o['x_mm'], 'planned_delta_mm': o['planned_delta_mm'],
                                'predicted_mm': estimate, 'error_mm': error, 'fold': fold})
            if error is not None:
                errors.append(error); group_errors[o['group']].append(abs(error))
    result = {**_metrics(errors, len(obs)), 'method': method, 'folds': len(set(assignment)),
                                         'independent_groups': len(group_errors)}
    # Descriptive group bootstrap, not sequentially valid confidence. These
    # intervals guide exploration; final success uses predeclared fixed samples.
    means = np.asarray([np.mean(v) for v in group_errors.values()])
    if len(means) >= 2:
        rng = np.random.default_rng(73019)
        draws = means[rng.integers(0, len(means), size=(400, len(means)))].mean(axis=1)
        result['group_mae_bootstrap_95_mm'] = np.quantile(draws, [.025, .975]).tolist()
    return result, predictions


def _level_stats(obs, model, predictions):
    groups, results = defaultdict(list), []
    residuals = {(p['session'], p['id']): p for p in predictions}
    for o in obs:
        if o['kind'] == 'scan':
            groups[o['planned_delta_mm']].append(o)
    for level, items in sorted(groups.items()):
        error = [residuals[(o['session'], o['id'])]['error_mm'] for o in items
                 if (o['session'], o['id']) in residuals and residuals[(o['session'], o['id'])]['error_mm'] is not None]
        per_finger = []
        for i in range(2):
            fit = model['fingers'][i] if model else None
            estimates = [finger_estimate(fit, o['descriptors'][i]['centroid_px']) for o in items] if fit else []
            estimates = [x for x in estimates if x is not None]
            sd = float(np.std(estimates, ddof=1)) if len(estimates) > 1 else None
            per_finger.append({'n': len(estimates), 'sd_equivalent_mm': sd,
                               'standard_error_mm': sd / math.sqrt(len(estimates)) if sd is not None else None})
        results.append({'offset_mm': level, 'n': len(items), 'measured_n': sum(o['measurement']['source'].startswith('measured_tcp') for o in items),
                        'bottles': dict(Counter(o['bottle'] for o in items)),
                        'actual_mean_mm': float(np.mean([o['x_mm'] for o in items])),
                        'actual_sd_mm': float(np.std([o['x_mm'] for o in items], ddof=1)) if len(items) > 1 else None,
                        'cv_bias_mm': float(np.mean(error)) if error else None,
                        'cv_mae_mm': float(np.mean(np.abs(error))) if error else None,
                        'fingers': per_finger})
    return results


def _diagnostics(rows, obs, levels, model, reference=None, settings=None):
    out = []
    B = [r for r in rows if r.get('cond') == 'B' and _outcome_eligible(r) and r.get('session_role') not in ('validation', 'frozen_validation', 'final_validation')]
    blocked = Counter(h.get('blocked_reason') for r in B for h in r.get('correction_history', []) if h.get('blocked_reason'))
    if blocked:
        out.append({'code': 'execution_bound', 'count': sum(blocked.values()), 'reasons': dict(blocked),
                    'action': 'keep fixed bounds; do not interpret blocked corrections as sensor limit'})
    zero_errors, overshoots = [], []
    for r in B:
        history = r.get('correction_history', [])
        es = [h['e_mm'] for h in history if _finite(h.get('e_mm')) and h.get('e_mm') is not None]
        if r.get('delta_mm') == 0 and es:
            zero_errors.append(es[0])
        if any(a * b < 0 and abs(a) > 2 and abs(b) > 2 for a, b in zip(es, es[1:])):
            overshoots.append(r.get('id'))
    if zero_errors and abs(float(np.mean(zero_errors))) > 1.:
        out.append({'code': 'zero_bias', 'n': len(zero_errors), 'mean_estimate_mm': float(np.mean(zero_errors)),
                    'action': 'measure actual zero and nearby A points before shifting an intercept'})
    if overshoots:
        out.append({'code': 'correction_overshoot', 'n': len(overshoots), 'trial_ids': overshoots,
                    'action': 'test reduced gain prospectively; no counterfactual success claim'})
    # Offline diagnosis can compare an estimate to the measured experimental
    # offset. This information is not sent to AgenticCorrection or model fitting.
    by_offset = defaultdict(list)
    for row in B:
        by_offset[float(row['delta_mm'])].append(row)
    for offset, group in sorted(by_offset.items()):
        residuals, sources, examples = [], Counter(), []
        for row in group:
            actual = _measured_probe(row, reference or row['session_meta'], settings or {})
            sources[actual['source']] += 1
            history = row.get('correction_history', [])
            estimate = history[0].get('e_mm') if history else None
            if estimate is None:
                used_fit = row['session_meta'].get('fit') or {}
                if used_fit.get('schema') == 'agentic_fit_v1':
                    estimate = model_error(used_fit, row.get('probe_initial', {}).get('descriptors', []))
            if (estimate is not None and _finite(estimate) and actual['x_mm'] is not None and
                    row.get('label') not in ('pushed', 'tipped')):
                error = float(estimate - actual['x_mm'])
                residuals.append(error)
                if len(examples) < 6:
                    examples.append({'id': row.get('id'), 'session': row['session'],
                                     'initial_estimate_mm': float(estimate), 'reference_a_mm': actual['x_mm'],
                                     'residual_mm': error, 'position_source': actual['source']})
        failed = sum(not _success(r) for r in group)
        exhausted = sum(bool(r.get('servo_exhausted')) for r in group)
        constrained = sum(any(h.get('blocked_reason') for h in r.get('correction_history', [])) for r in group)
        out.append({'code': 'B_offset_feedback', 'offset_mm': offset, 'n': len(group),
                    'failures': failed, 'failure_rate': failed / len(group),
                    'servo_exhausted': exhausted, 'servo_exhausted_rate': exhausted / len(group),
                    'execution_blocked': constrained,
                    'initial_residual_n': len(residuals),
                    'initial_bias_mm': float(np.mean(residuals)) if residuals else None,
                    'initial_mae_mm': float(np.mean(np.abs(residuals))) if residuals else None,
                    'position_sources': dict(sources), 'examples': examples,
                    'action': 'prioritize A at and adjacent to this offset for mapping/contact diagnosis; B rows remain excluded from fitting',
                    'limitations': 'residual is versus measured TCP, not measured bottle position; planned fallback is explicitly labelled'})
    disagreement = []
    if model:
        for o in obs:
            vals = [finger_estimate(f, d['centroid_px']) for f, d in zip(model['fingers'], o['descriptors'])]
            if all(v is not None for v in vals) and abs(vals[0] - vals[1]) > 2.:
                disagreement.append({'id': o['id'], 'session': o['session'], 'difference_mm': abs(vals[0] - vals[1])})
    if disagreement:
        out.append({'code': 'finger_disagreement', 'n': len(disagreement), 'examples': disagreement[:10],
                    'action': 'check contact/edge and repeat implicated positions'})
    biased = [x['offset_mm'] for x in levels if x['cv_bias_mm'] is not None and abs(x['cv_bias_mm']) > 1.]
    if biased:
        out.append({'code': 'local_bias', 'offsets_mm': biased, 'action': 'add neighboring positions; repeated identical points alone do not repair systematic bias'})
    return out


def analyze_sessions(roots, settings):
    """Return JSON-serializable evidence and candidate models; never fit B data."""
    sessions, rows, load_issues = _load(roots, settings)
    # Frozen validation is NEVER eligible to help select a model or tune its gain.
    training_sessions = [s for s in sessions if s['role'] not in ('validation', 'frozen_validation', 'final_validation')]
    issues, compatibility = _compatibility(training_sessions, settings) if training_sessions else ([], [])
    issues = load_issues + issues
    reference = training_sessions[0]['meta'] if training_sessions else {}
    observations, excluded = _observations(rows, reference, settings)
    learning = _section(settings, 'learning')
    models = []
    if not issues:
        for family, anchored in [('linear', False), ('linear', True), ('monotone_piecewise', False)]:
            model = _fit(observations, family, anchored, learning)
            if model is None:
                continue
            metrics, predictions = _cross_validate(observations, family, anchored, settings)
            model.update(validation=metrics, validation_predictions=predictions,
                         axes_tcp=reference.get('config', {}).get('axes_tcp'),
                         normal_axis=reference.get('config', {}).get('force', {}).get('normal_axis'),
                         normal_signs=reference.get('config', {}).get('force', {}).get('normal_signs'),
                         reference_T0=reference.get('T0'), source_sessions=sorted(set(o['session'] for o in observations)),
                         signature=_hash(_signature(reference)),
                         source_trials=[{'session': o['session'], 'id': o['id'], 'sha256': o['metadata_sha256']} for o in observations])
            model['usable_for_B'] = bool(metrics['n'] >= int(learning.get('min_validation_trials', 4)) and
                                        metrics['validation_coverage'] >= float(learning.get('min_validation_coverage', .75)))
            model['model_id'] = _hash({k: v for k, v in model.items() if k != 'validation_predictions'})[:16]
            models.append(model)
    eligible_models = [m for m in models if m['usable_for_B']]
    # Prefer simpler models unless inverse-mm validation actually improves.
    best = None
    for model in sorted(eligible_models, key=lambda m: (m['family'] != 'linear', m['anchored'])):
        if best is None or model['validation']['mae_mm'] < best['validation']['mae_mm'] - float(learning.get('min_gain_mm', .1)):
            best = model
    levels = _level_stats(observations, best, best['validation_predictions'] if best else [])
    diagnostics = _diagnostics(rows, observations, levels, best, reference, settings)
    sources = dict(Counter(o['measurement']['source'] for o in observations))
    scope = _section(settings, 'scope')
    lo, hi = float(scope.get('min_mm', -2)), float(scope.get('max_mm', 10))
    minimum = int(learning.get('min_per_level', 2))
    near = float(scope.get('step_mm', 2)) / 2
    endpoint_evidence = {str(x): sum(o['measurement']['source'].startswith('measured_tcp') and abs(o['x_mm'] - x) <= near
                                   for o in observations) for x in (lo, hi)}
    covered = bool(best and best['common_range_mm'][0] <= lo and best['common_range_mm'][1] >= hi and
                   all(n >= minimum for n in endpoint_evidence.values()))
    metrics = {**(deepcopy(best['validation']) if best else _metrics([], len(observations))),
        'coverage_scope_mm': [lo, hi], 'scope_covered': covered, 'endpoint_evidence': endpoint_evidence,
        'measured_fraction': sum(o['measurement']['source'].startswith('measured_tcp') for o in observations) / len(observations) if observations else 0.,
    }
    diagnostics += ([{'code': 'planned_position_only', 'count': sources['planned_delta_fallback'],
                       'action': 'reacquire measured PROBE TCP before precision acceptance'}] if sources.get('planned_delta_fallback') else [])
    measured_only = bool(observations and not sources.get('planned_delta_fallback'))
    stop = bool(best and not issues and covered and measured_only and
                metrics['mae_mm'] <= float(learning.get('target_mae_mm', 1.)) and
                metrics['p90_mm'] <= float(learning.get('target_p90_mm', 2.)))
    B_rows = [r for r in rows if r.get('cond') == 'B' and _outcome_eligible(r) and
              r.get('session_role') not in ('validation', 'frozen_validation', 'final_validation')]
    B_summary = []
    for session in sorted(set(r['session'] for r in B_rows)):
        group = [r for r in B_rows if r['session'] == session]
        cfg = group[0]['session_meta'].get('config', {})
        B_summary.append({'session': session, 'n': len(group), 'successes': sum(_success(r) for r in group),
                          'success_rate': sum(_success(r) for r in group) / len(group),
                          'mean_regrasps': float(np.mean([r.get('regrasps', 0) for r in group])),
                          'gain': cfg.get('servo', {}).get('gain', cfg.get('servo', {}).get('correction_gain', 1.)),
                          'model_id': (group[0]['session_meta'].get('fit') or {}).get('model_id'),
                          'prospective_only': True})
    return {'schema': 'agentic_analysis_v1', 'models': models, 'best_model': best, 'metrics': metrics,
            'stopping_eligible': stop, 'diagnostics': diagnostics, 'B_batches': B_summary,
            'quality': {'blocking_reasons': issues, 'excluded_counts': excluded, 'coordinate_sources': sources,
                        'trials_total': len(rows), 'mapping_trials': len(observations), 'compatibility': compatibility},
            'evidence': {'sessions': [{k: v for k, v in s.items() if k != 'meta'} for s in sessions],
                         'observations': observations, 'level_stats': levels,
                         'signature': _hash(_signature(reference)) if reference else None,
                         'reference_T0': reference.get('T0'), 'B_batches': B_summary},
            'caveats': ['PROBE TCP relative to fixed T0 is not bottle position if the bottle moved',
                        'CV and bootstrap guide exploration; final validation is a separate fixed sample',
                        'B outcomes never train the observation-to-offset mapping']}


def propose_candidates(report, settings, remaining_trials, round_index):
    """Deterministic bounded options for an agent to choose, each fully auditable."""
    if report.get('quality', {}).get('blocking_reasons'):
        return []
    scope, learning = _section(settings, 'scope'), _section(settings, 'learning')
    lo, hi, step = float(scope.get('min_mm', -2)), float(scope.get('max_mm', 10)), float(scope.get('step_mm', 2))
    if not _finite([lo, hi, step]) or lo > hi or step <= 0:
        raise ValueError('invalid immutable scope')
    batch = min(int(remaining_trials), int(_section(settings, 'budget').get('batch_trials', 12)))
    batch -= batch % 2
    if batch < 2:
        return []
    if not scope.get('allow_expand', False):
        lo = max(lo, float(scope.get('initial_min_mm', lo)))
        hi = min(hi, float(scope.get('initial_max_mm', hi)))
    grid = sorted(set([round(float(v), 8) for v in np.arange(lo, hi + step * .001, step)] + [hi]))
    if lo <= 0 <= hi:
        grid = sorted(set(grid + [0.]))
    levels = report.get('evidence', {}).get('level_stats', [])
    stats = {float(x['offset_mm']): x for x in levels}
    model = report.get('best_model')
    diag_codes = {d['code'] for d in report.get('diagnostics', [])}
    ranked = []
    for x in grid:
        stat = stats.get(x, {})
        n = stat.get('n', 0)
        nearest = min((abs(x - v) for v in stats), default=step * 2)
        bias = abs(stat.get('cv_bias_mm') or 0.)
        noise = max([f.get('sd_equivalent_mm') or 0. for f in stat.get('fingers', [])], default=0.)
        # High SEM calls for repeats; systematic bias calls for adjacent points.
        sem = noise / math.sqrt(max(1, n))
        score = 5. / (1 + n) + nearest / step + bias + sem
        reasons = ['new_position' if not n else 'repeat_uncertainty']
        if x == 0 and 'zero_bias' in diag_codes:
            score += 4; reasons.append('verify_zero_bias')
        if n and stat.get('measured_n', 0) < n:
            score += 3; reasons.append('replace_planned_with_measured_tcp')
        if x in (lo, hi) and not report.get('metrics', {}).get('scope_covered'):
            score += 2; reasons.append('scope_endpoint_evidence')
        for feedback in report.get('diagnostics', []):
            if feedback.get('code') != 'B_offset_feedback':
                continue
            distance = abs(x - float(feedback['offset_mm']))
            proximity = max(0., 1. - distance / (2. * step))
            if not proximity:
                continue
            # Bound-only failures cannot be repaired by an observation model;
            # keep them in reports but do not chase impossible motions forever.
            nonbound_failure = max(0, feedback['failures'] - feedback['execution_blocked']) / max(1, feedback['n'])
            nonbound_exhausted = max(0, feedback['servo_exhausted'] - feedback['execution_blocked']) / max(1, feedback['n'])
            mismatch = min(4., abs(feedback.get('initial_mae_mm') or 0.))
            boost = proximity * (5. * nonbound_failure + 3. * nonbound_exhausted + mismatch)
            if boost > 0:
                score += boost
                reasons.append(f'B_feedback_near_{feedback["offset_mm"]:g}')
        ranked.append((score, x, reasons))
    ranked.sort(key=lambda x: (-x[0], abs(x[1]), x[1]))
    selected = ranked[:max(1, batch // 2)]
    cells = [{'offset_mm': x, 'reps': 2} for _, x, _ in selected]
    spare = batch - sum(c['reps'] for c in cells)
    i = 0
    while spare >= 2 and cells:
        cells[i % len(cells)]['reps'] += 2; spare -= 2; i += 1
    candidates = []
    def append(cond, cells, reason, score, gain=1.):
        cand = {'cond': cond, 'cells': cells, 'reason': reason, 'score': float(score),
                'trial_count': sum(c['reps'] for c in cells), 'model_id': model.get('model_id') if model else None,
                'policy': {'gain': float(gain)}, 'round_index': int(round_index),
                'scope_mm': [lo, hi], 'requires_prospective_evaluation': cond == 'B'}
        cand['id'] = _hash(cand)[:16]
        candidates.append(cand)
    append('A', cells, 'Add positions/repetitions from held-out error, uncertainty and zero/endpoint evidence: ' +
           '; '.join(f'{x:g}: {",".join(rs)}' for _, x, rs in selected), sum(x[0] for x in selected))
    if model and model.get('usable_for_B'):
        support = [x for x in grid if model['common_range_mm'][0] <= x <= model['common_range_mm'][1]]
        # Test coverage across the support, including zero, rather than cherry
        # picking only the largest previous failure or changing the hidden delta.
        picks = sorted(set(support[i] for i in np.linspace(0, len(support) - 1, min(batch // 2, len(support)), dtype=int))) if support else []
        bcells = [{'offset_mm': x, 'reps': 2} for x in picks]
        gains = _section(settings, 'policy').get('gains', [.5, .75, 1.])
        for gain in gains:
            gain = float(gain)
            if not .5 <= gain <= 1.:
                raise ValueError('candidate gain outside immutable [0.5,1.0]')
            if not bcells:
                continue
            prior = [b for b in report.get('B_batches', []) if b.get('model_id') == model['model_id'] and b.get('gain') == gain]
            priority = 15. if report.get('stopping_eligible') else 4.
            priority += 3. if not prior else -len(prior)
            if 'correction_overshoot' in diag_codes and gain == .75:
                priority += 5.
            append('B', deepcopy(bcells), f'Prospective evaluation of frozen model {model["model_id"]}, gain={gain:g}; historical outcomes do not prove this gain will succeed', priority, gain)
    return sorted(candidates, key=lambda c: (-c['score'], c['id']))


def _wilson(k, n):
    if not n:
        return [None, None]
    z = 1.959963984540054
    den = 1 + z * z / n
    center = (k / n + z * z / (2 * n)) / den
    half = z * math.sqrt(k / n * (1 - k / n) / n + z * z / (4 * n * n)) / den
    return [max(0., center - half), min(1., center + half)]


def _rate(group):
    k, n = sum(_success(r) for r in group), len(group)
    contact = sum(bool(r.get('probe_final', {}).get('valid') and r.get('bilateral')) for r in group)
    return {'n': n, 'successes': k, 'success_rate': k / n if n else None, 'wilson_95': _wilson(k, n),
            'contact_valid': contact, 'contact_rate': contact / n if n else None,
            'mean_regrasps': float(np.mean([r.get('regrasps', 0) for r in group])) if n else None}


def frozen_evaluation(Aroots, Broots, model, settings):
    """Evaluate a predeclared final A/B plan without refitting or adaptive peeking."""
    sessions, rows, issues = _load(list(Aroots) + list(Broots), settings)
    compatibility, details = _compatibility(sessions, settings)
    issues += compatibility
    roots = {s['path'] for s in sessions}
    if roots & set(model.get('source_sessions', [])):
        issues.append('validation sessions overlap model training sources')
    validation = _section(settings, 'validation')
    B_sessions = {r['session'] for r in rows if r.get('cond') == 'B'}
    for s in sessions:
        if s['role'] not in ('validation', 'frozen_validation', 'final_validation'):
            issues.append(s['path'] + ': not marked as frozen validation before acquisition')
        contains_B = (s['path'] in B_sessions or s['meta'].get('cond') == 'B' or
                      any(item.get('cond') == 'B' for item in s['meta'].get('plan', [])))
        if contains_B:
            fit = s['meta'].get('fit') or {}
            if fit.get('model_id') != model.get('model_id'):
                issues.append(s['path'] + ': validation B used a different model')
            gain = s['meta'].get('config', {}).get('servo', {}).get('gain',
                   s['meta'].get('config', {}).get('servo', {}).get('correction_gain', 1.))
            expected_gain = validation.get('frozen_gain')
            if expected_gain is not None and (not _finite(gain) or not math.isclose(float(gain), float(expected_gain), abs_tol=1e-12)):
                issues.append(s['path'] + ': validation B did not use frozen gain')
            for row in (r for r in rows if r['session'] == s['path'] and r.get('cond') == 'B'):
                if row.get('model_id') is not None and row['model_id'] != model.get('model_id'):
                    issues.append(s['path'] + ': validation B trial used a different model')
                for h in row.get('correction_history', []):
                    if h.get('gain') is not None and (not _finite(h['gain']) or not math.isclose(float(h['gain']), float(gain), abs_tol=1e-12)):
                        issues.append(s['path'] + ': validation B changed gain within a trial')
                        break
        if model.get('signature') and _hash(s['signature']) != model['signature']:
            issues.append(s['path'] + ': validation sensing/force protocol differs from trained model')
        frozen_T0, current_T0 = _pose(model.get('reference_T0')), _pose(s['meta'].get('T0'))
        if frozen_T0 is None or current_T0 is None:
            issues.append(s['path'] + ': missing frozen or validation T0')
        elif (np.linalg.norm(frozen_T0[:3, 3] - current_T0[:3, 3]) > float(_section(settings, 'learning').get('t0_tolerance_mm', 2.)) or
              _rotation_deg(frozen_T0[:3, :3], current_T0[:3, :3]) > float(_section(settings, 'learning').get('t0_tolerance_deg', 1.))):
            issues.append(s['path'] + ': validation T0 differs from trained reference')
    gains = {float(s['meta'].get('config', {}).get('servo', {}).get('gain',
                   s['meta'].get('config', {}).get('servo', {}).get('correction_gain', 1.)))
             for s in sessions if s['path'] in B_sessions}
    if len(gains) > 1:
        issues.append('validation B changed correction gain within the frozen evaluation')
    offsets = [float(v) for v in validation.get('offsets_mm', [-2, 0, 2, 6, 10])]
    reps = int(validation.get('reps', 4))
    target = float(validation.get('success_target', .8))
    eligible = [r for r in rows if _outcome_eligible(r)]
    bins = []
    bottles = sorted(set(r.get('bottle', 'unknown') for r in rows))
    complete_plan = True
    for cond in ('A', 'B'):
        for offset in offsets:
            group = [r for r in eligible if r.get('cond') == cond and float(r['delta_mm']) == offset]
            n = len(group)
            if n != reps:
                complete_plan = False
            expected_bottles = validation.get('bottles', ['light', 'heavy'])
            bottle_counts = Counter(r.get('bottle') for r in group)
            if any(bottle_counts.get(b, 0) != reps // len(expected_bottles) for b in expected_bottles):
                complete_plan = False
            if n > reps:
                issues.append(f'{cond} {offset:g}: final sample exceeds predeclared count; optional stopping not a fixed-sample interval')
            bins.append({'cond': cond, 'offset_mm': offset, 'expected_n': reps, **_rate(group)})
    unexpected = [r for r in eligible if float(r['delta_mm']) not in offsets]
    if unexpected:
        issues.append('validation contains offsets outside its frozen plan')
    by_bottle = [{'cond': cond, 'bottle': bottle, **_rate([r for r in eligible if r.get('cond') == cond and r.get('bottle') == bottle])}
                 for cond in ('A', 'B') for bottle in bottles]
    errors, accuracy_points, unmeasured = [], [], 0
    reference = {'T0': model.get('reference_T0'), 'config': {'axes_tcp': model.get('axes_tcp')}}
    for row in eligible:
        if row.get('cond') != 'A' or not row.get('probe_initial', {}).get('valid') or row.get('label') in ('pushed', 'tipped'):
            continue
        actual = _measured_probe(row, reference, settings)
        if not actual['source'].startswith('measured_tcp') or actual['x_mm'] is None:
            unmeasured += 1
            continue
        estimate = model_error(model, row['probe_initial'].get('descriptors', []))
        if estimate is not None:
            errors.append(estimate - actual['x_mm'])
            accuracy_points.append({'session': row['session'], 'id': row.get('id'),
                                    'measured_a_mm': actual['x_mm'], 'estimated_a_mm': estimate,
                                    'residual_mm': estimate - actual['x_mm']})
    A_total = sum(r.get('cond') == 'A' for r in eligible)
    accuracy = _metrics(errors, A_total)
    Bbins = [b for b in bins if b['cond'] == 'B']
    learning = _section(settings, 'learning')
    accuracy_met = bool(accuracy['n'] and not unmeasured and
                        accuracy['validation_coverage'] >= float(learning.get('min_validation_coverage', .75)) and
                        accuracy['mae_mm'] <= float(learning.get('target_mae_mm', 1.)) and
                        accuracy['p90_mm'] <= float(learning.get('target_p90_mm', 2.)))
    success_met = all(b['n'] and b['wilson_95'][0] >= target for b in Bbins)
    success_rejected = any(b['n'] and b['wilson_95'][1] < target for b in Bbins)
    if issues or not complete_plan:
        outcome = 'inconclusive'
    elif success_rejected or (accuracy['n'] and not accuracy_met):
        outcome = 'not_met'
    elif success_met and accuracy_met:
        outcome = 'passed'
    else:
        outcome = 'inconclusive'
    return {'schema': 'agentic_frozen_evaluation_v1', 'outcome': outcome, 'model_id': model.get('model_id'),
            'fixed_plan_complete': complete_plan, 'issues': issues, 'compatibility': details,
            'accuracy': accuracy, 'unmeasured_A_trials': unmeasured, 'accuracy_met': accuracy_met,
            'accuracy_points': accuracy_points, 'frozen_gain': validation.get('frozen_gain'),
            'success_target': target, 'success_criterion': 'each predeclared B offset Wilson 95% lower bound >= target',
            'success_met': success_met, 'bins': bins, 'per_bottle': by_bottle,
            'totals': {cond: _rate([r for r in eligible if r.get('cond') == cond]) for cond in ('A', 'B')},
            'excluded_trials': len(rows) - len(eligible),
            'interpretation': 'An all-success small sample can remain inconclusive. Do not extend this fixed test after seeing results and reuse its 95% claim.',
            'source_sessions': sorted(roots)}


def write_plots(report, out):
    """Write standalone scientific PNGs for exploration or frozen evaluation.

    Returns paths and explicit optional dependency warnings. The plots never
    reinterpret inconclusive fixed-sample results as successful validation.
    """
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError as exc:
        return {'files': [], 'warnings': ['Plot dependency unavailable: ' + str(exc)]}
    files = []
    def save(fig, name):
        path = out / name
        fig.savefig(path, dpi=160, bbox_inches='tight')
        plt.close(fig)
        files.append(str(path.resolve()))
    if report.get('schema') == 'agentic_frozen_evaluation_v1':
        bins = report.get('bins', [])
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
        for cond, color in [('A', 'tab:blue'), ('B', 'tab:orange')]:
            rows = sorted((b for b in bins if b['cond'] == cond and b['n']), key=lambda b: b['offset_mm'])
            if not rows:
                continue
            x = [b['offset_mm'] for b in rows]
            y = np.asarray([b['success_rate'] for b in rows])
            low = np.asarray([b['wilson_95'][0] for b in rows]); high = np.asarray([b['wilson_95'][1] for b in rows])
            axes[0].errorbar(x, y, yerr=np.maximum(0., np.vstack([y - low, high - y])), fmt='o-', capsize=3, label=cond, color=color)
            axes[1].plot(x, [b['mean_regrasps'] for b in rows], 'o-', label=cond, color=color)
        axes[0].axhline(report.get('success_target', .8), linestyle='--', color='gray')
        axes[0].set(ylabel='Success rate (Wilson 95%)', ylim=(-.02, 1.02))
        axes[1].set(ylabel='Mean regrasps')
        for ax in axes:
            ax.set(xlabel='Predeclared offset (mm)'); ax.grid(alpha=.2)
            if ax.get_legend_handles_labels()[0]: ax.legend()
        fig.suptitle('Frozen independent validation: ' + str(report.get('outcome')))
        save(fig, 'validation_success_regrasps.png')
        points = report.get('accuracy_points', [])
        fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
        ax.scatter([p['measured_a_mm'] for p in points], [p['residual_mm'] for p in points], alpha=.65)
        ax.axhline(0., color='gray', linestyle='--')
        ax.set(xlabel='Measured PROBE TCP offset (mm)', ylabel='Estimated minus measured (mm)', title='Frozen model: independent A observations')
        ax.grid(alpha=.2); save(fig, 'validation_mapping_residuals.png')
    elif report.get('schema') == 'agentic_analysis_v1':
        observations = report.get('evidence', {}).get('observations', [])
        model = report.get('best_model')
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
        for i, ax in enumerate(axes):
            f = model['fingers'][i] if model else None
            index = 0 if not f or f['coordinate'] == 'u' else 1
            for edge, color in [(False, 'tab:blue'), (True, 'tab:orange')]:
                points = [o for o in observations if bool(o['descriptors'][i].get('touches_edge')) == edge]
                ax.scatter([o['x_mm'] for o in points], [o['descriptors'][i]['centroid_px'][index] for o in points],
                           s=18, alpha=.6, color=color, label='edge' if edge else 'interior')
            if f:
                ax.plot(f['knots_delta_mm'], f['knots_centroid_px'], 'k-', label=f['family'])
            ax.set(xlabel='PROBE TCP offset (mm; fallback labelled in JSON)', ylabel=('u' if index == 0 else 'v') + ' centroid (px)',
                   title=('Left', 'Right')[i] + ' finger, initial A only')
            ax.grid(alpha=.2); ax.legend()
        save(fig, 'mapping_fingers.png')
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
        predictions = (model or {}).get('validation_predictions', [])
        valid = [p for p in predictions if p.get('error_mm') is not None]
        axes[0].scatter([p['x_mm'] for p in valid], [p['error_mm'] for p in valid], alpha=.65)
        axes[0].axhline(0, color='gray', linestyle='--')
        axes[0].set(xlabel='Measured/fallback PROBE offset (mm)', ylabel='Held-out inverse error (mm)', title='Grouped cross-validation; no frame split')
        levels = report.get('evidence', {}).get('level_stats', [])
        axes[1].bar([v['offset_mm'] for v in levels], [v['n'] for v in levels], width=.6, label='all mapping observations')
        axes[1].bar([v['offset_mm'] for v in levels], [v['measured_n'] for v in levels], width=.3, label='measured TCP')
        axes[1].set(xlabel='Planned offset (mm)', ylabel='Physical trials', title='Coverage and coordinate evidence')
        axes[1].legend()
        for ax in axes: ax.grid(alpha=.2)
        save(fig, 'mapping_validation_coverage.png')
        feedback = [d for d in report.get('diagnostics', []) if d.get('code') == 'B_offset_feedback']
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
        axes[0].plot([d['offset_mm'] for d in feedback], [d['failure_rate'] for d in feedback], 'o-', label='task failure')
        axes[0].plot([d['offset_mm'] for d in feedback], [d['servo_exhausted_rate'] for d in feedback], 's-', label='servo exhausted')
        axes[0].set(ylabel='Observed fraction', ylim=(-.02, 1.02)); axes[0].legend()
        points = [d for d in feedback if d['initial_mae_mm'] is not None]
        axes[1].plot([d['offset_mm'] for d in points], [d['initial_mae_mm'] for d in points], 'o-')
        axes[1].set(ylabel='Initial tactile estimate MAE (mm)')
        for ax in axes: ax.set(xlabel='B initial offset (mm)'); ax.grid(alpha=.2)
        fig.suptitle('Exploratory B feedback: selects future A; never trains the mapping')
        save(fig, 'B_feedback.png')
    else:
        raise ValueError('Unknown agentic report schema')
    return {'files': files, 'warnings': []}
