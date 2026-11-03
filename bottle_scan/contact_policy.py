"""Offline-trained, tactile-only one-dimensional recovery policy.

This module imports no hardware.  A labels mean ``-measured PROBE displacement
from T0``: geometric supervision, not a proven optimal recovery action or bottle
position. Optional B labels are verified executed actions from successful
recovery trajectories; they are successful demonstrations, not optimal actions.
The frozen runtime never receives planned displacement or an absolute TCP pose.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import json
from pathlib import Path

import numpy as np

from .agentic_learning import (
    AgenticCorrection, _compatibility, _fit, _hash, _load, _measured_probe,
    _observations, _signature, _success,
)


SCHEMA = 'contact_policy_v1'
FINAL_ROLES = ('validation', 'frozen_validation', 'final_validation')
DEFAULTS = {
    'seed': 20261102, 'min_samples': 20, 'min_train_samples': 12,
    'min_validation_samples': 4, 'validation_fraction': .25,
    'ensemble_size': 5, 'hidden_units': 16, 'epochs': 400,
    'learning_rate': .01, 'l2': .001, 'dd_pool': False,
    'include_successful_B': False, 'max_validation_mae_mm': 2.,
    'max_validation_p90_mm': 4., 'max_uncertainty_mm': 4.,
    'ood_radius_multiplier': 1.5, 'ood_min_radius': .5,
    'max_action_mm': 15., 'B_tracking_tolerance_mm': 2.,
}


def feature_names(dd_pool=False):
    names = []
    for finger in ('left', 'right'):
        names.extend(f'{finger}.{key}' for key in (
            'centroid_u_px', 'centroid_v_px', 'bbox_x0', 'bbox_y0',
            'bbox_x1', 'bbox_y1', 'log1p_area', 'angle_sin2', 'angle_cos2',
            'angle_available'))
        if dd_pool:
            names.extend(f'{finger}.dd_depth_pool_{i}' for i in range(16))
    return names


def feature_vector(descriptors, *, dd_pool=False):
    """Use an explicit allowlist; extra delta/TCP/truth fields are ignored."""
    if len(descriptors or []) != 2:
        raise ValueError('ContactPolicy requires two tactile descriptors')
    values = []
    for d in descriptors:
        if not isinstance(d, dict) or not d.get('valid'):
            raise ValueError('invalid tactile descriptor')
        center, box = d.get('centroid_px'), d.get('bbox_xyxy_px')
        if np.asarray(center).shape != (2,) or np.asarray(box).shape != (4,):
            raise ValueError('missing centroid or contact bounding box')
        area = float(d.get('area_px', 0))
        if area <= 0 or box[0] > box[2] or box[1] > box[3]:
            raise ValueError('invalid contact area or bounding box')
        angle = d.get('principal_angle_deg')
        angle_available = angle is not None
        angle = np.radians(float(angle) * 2) if angle_available else 0.
        values.extend([*center, *box, np.log1p(area),
                       np.sin(angle) if angle_available else 0.,
                       np.cos(angle) if angle_available else 0., float(angle_available)])
        if dd_pool:
            pooled = np.asarray(d.get('dd_pool_4x4'), float)
            if pooled.shape != (4, 4):
                raise ValueError('missing numerical DD depth pooling feature')
            values.extend(pooled.ravel().tolist())
    array = np.asarray(values, float)
    if not np.isfinite(array).all():
        raise ValueError('nonfinite tactile feature')
    return array


def _digest(model):
    return _hash({k: v for k, v in model.items() if k not in ('model_id', 'integrity_sha256')})


def _seal(model):
    model['integrity_sha256'] = _digest(model)
    model['model_id'] = model['integrity_sha256'][:20]
    return model


def _settings(settings):
    whole = deepcopy(settings or {})
    cfg = {**DEFAULTS, **whole.get('contact_policy', {})}
    for key in ('seed', 'min_samples', 'min_train_samples', 'min_validation_samples',
                'ensemble_size', 'hidden_units', 'epochs'):
        if type(cfg[key]) is not int or cfg[key] < (0 if key == 'seed' else 1):
            raise ValueError(f'invalid contact_policy.{key}')
    if not 3 <= cfg['ensemble_size'] <= 20 or not 1 <= cfg['hidden_units'] <= 128 or cfg['epochs'] > 10000:
        raise ValueError('ContactPolicy model budget outside bounds')
    for key in ('validation_fraction', 'learning_rate', 'l2', 'max_validation_mae_mm',
                'max_validation_p90_mm', 'max_uncertainty_mm', 'ood_radius_multiplier',
                'ood_min_radius', 'max_action_mm', 'B_tracking_tolerance_mm'):
        if isinstance(cfg[key], bool) or not np.isfinite(cfg[key]) or cfg[key] <= 0:
            raise ValueError(f'invalid contact_policy.{key}')
    if not .1 <= cfg['validation_fraction'] <= .5 or cfg['max_action_mm'] > 15:
        raise ValueError('ContactPolicy split/action limit outside bounds')
    for key in ('dd_pool', 'include_successful_B'):
        if type(cfg[key]) is not bool:
            raise ValueError(f'contact_policy.{key} must be boolean')
    if 'balance_session_validation' in cfg and type(cfg['balance_session_validation']) is not bool:
        raise ValueError('contact_policy.balance_session_validation must be boolean')
    if 'frozen_validation_groups' in cfg:
        groups = cfg['frozen_validation_groups']
        if (not isinstance(groups, list) or not groups or
                any(not isinstance(group, str) or not group.strip() for group in groups) or
                len(set(groups)) != len(groups)):
            raise ValueError('contact_policy.frozen_validation_groups must be a nonempty unique list of group strings')
    return whole, cfg


def _B_observations(rows, reference, settings, cfg):
    """Only measured, applied, eventually successful recovery transitions."""
    observations, excluded = [], Counter()
    for row in rows:
        if row.get('cond') != 'B':
            continue
        if row.get('session_role') in FINAL_ROLES:
            excluded['B_final_validation_never_trains'] += 1
            continue
        if (not row.get('complete') or not row.get('recording_verified') or
                row.get('archived_attempt') or not _success(row) or
                row.get('servo_exhausted') or row.get('kind') != 'scan'):
            excluded['B_not_reliable_success'] += 1
            continue
        history = row.get('correction_history', [])
        regrasps = row.get('probe_regrasps', [])
        if not history or not history[-1].get('accepted') or not regrasps:
            excluded['B_no_verified_recovery'] += 1
            continue
        probes = [row.get('probe_initial', {})] + [r.get('probe') or {} for r in regrasps]
        for index, following in enumerate(probes[1:]):
            before = probes[index]
            step = history[index] if index < len(history) else {}
            if not before.get('valid') or not following.get('valid') or step.get('blocked'):
                excluded['B_invalid_transition'] += 1
                continue
            measured = [_measured_probe({**row, 'probe_initial': p}, reference, settings) for p in (before, following)]
            action = step.get('applied_delta_mm')
            if (any(m.get('x_mm') is None or not m['source'].startswith('measured_tcp') for m in measured) or
                    not isinstance(action, (int, float)) or not np.isfinite(action) or abs(action) > cfg['max_action_mm'] or
                    abs(measured[1]['x_mm'] - measured[0]['x_mm'] - action) > cfg['B_tracking_tolerance_mm']):
                excluded['B_unverified_applied_action'] += 1
                continue
            observations.append({
                'id': row['id'], 'session': row['session'], 'trial_path': row['trial_path'],
                'metadata_sha256': row['metadata_sha256'], 'descriptors': before['descriptors'],
                'x_mm': measured[0]['x_mm'], 'target_action_mm': float(action),
                'measurement': measured[0], 'group': row['trial_path'],
                'label_source': 'B_verified_successful_executed_action_not_optimal',
                'transition_index': index,
            })
    return observations, dict(excluded)


def _split(obs, cfg):
    """Hold out complete sessions when possible, otherwise complete trial groups."""
    if 'frozen_validation_groups' in cfg:
        held = set(cfg['frozen_validation_groups'])
        keys = [o['group'] for o in obs]
        missing = held - set(keys)
        if missing:
            raise ValueError('frozen validation groups missing from eligible observations: ' + str(sorted(missing)))
        test = np.asarray([i for i, key in enumerate(keys) if key in held], int)
        train = np.asarray([i for i, key in enumerate(keys) if key not in held], int)
        return train, test, {'method': 'frozen_trial_groups', 'seed': cfg['seed'],
                            'split_selection': 'preregistered group identities; all new groups train, no feature/label access',
                            'validation_groups': sorted(held), 'training_groups': sorted(set(keys) - held)}
    sessions = sorted({o['session'] for o in obs})
    method = 'held_out_session' if len(sessions) > 1 else 'held_out_trial_group'
    groups = sessions if len(sessions) > 1 else sorted({o['group'] for o in obs})
    if len(groups) < 2:
        raise ValueError('need at least two independent sessions or trial groups')
    rng = np.random.default_rng(cfg['seed'])
    order = [groups[i] for i in rng.permutation(len(groups))]
    count = max(1, min(len(groups) - 1, int(np.ceil(len(groups) * cfg['validation_fraction']))))
    held = set(order[:count])
    keys = [o['session'] if method == 'held_out_session' else o['group'] for o in obs]
    counts = Counter(keys)
    # Incremental collection creates small sessions beside a large seed
    # session. Choose a size-feasible whole-group split using counts only,
    # never observations, labels, predictions or validation losses.
    def feasible(chosen):
        n = sum(counts[k] for k in chosen)
        return n >= cfg['min_validation_samples'] and len(obs)-n >= cfg['min_train_samples']
    if not feasible(held) or cfg.get('balance_session_validation', False):
        candidates = [{g} for g in order if feasible({g})]
        for start in range(len(order)):
            chosen = set()
            for g in order[start:] + order[:start]:
                if sum(counts[k] for k in chosen) + counts[g] <= len(obs)-cfg['min_train_samples']:
                    chosen.add(g)
                if feasible(chosen):
                    candidates.append(chosen.copy())
                    if not cfg.get('balance_session_validation', False):
                        break
        if candidates:
            desired = cfg['validation_fraction'] * len(obs)
            held = min(candidates, key=lambda subset: abs(sum(counts[g] for g in subset)-desired))
    test = np.asarray([i for i, key in enumerate(keys) if key in held], int)
    train = np.asarray([i for i, key in enumerate(keys) if key not in held], int)
    selection = ('seeded whole groups, size-only feasibility and balance; no feature/label access' if cfg.get('balance_session_validation', False)
                 else 'seeded whole groups, size-only feasibility; no feature/label access')
    return train, test, {'method': method, 'seed': cfg['seed'], 'split_selection': selection,
                         'validation_groups': sorted(held),
                         'training_groups': sorted(set(keys) - held)}


def _network(X, y, cfg, rng):
    """Train all weights of a tanh MLP plus a linear skip using numpy Adam."""
    n, dims = X.shape
    take = rng.integers(0, n, n)
    X, y = X[take], y[take]
    z = np.column_stack([X, np.ones(n)])
    ridge = np.eye(dims + 1) * cfg['l2']; ridge[-1, -1] = 0
    linear = np.linalg.pinv(z.T @ z + ridge) @ z.T @ y
    params = [rng.normal(0, .3 / np.sqrt(dims), (dims, cfg['hidden_units'])),
              np.zeros(cfg['hidden_units']), np.zeros(cfg['hidden_units']),
              linear[:-1], np.array([linear[-1]])]
    first = [np.zeros_like(p) for p in params]; second = [np.zeros_like(p) for p in params]
    for epoch in range(1, cfg['epochs'] + 1):
        W, b, v, skip, bias = params
        hidden = np.tanh(X @ W + b)
        error = (hidden @ v + X @ skip + bias[0] - y) * (2. / n)
        dh = error[:, None] * v * (1 - hidden * hidden)
        grads = [X.T @ dh + 2 * cfg['l2'] * W, dh.sum(0),
                 hidden.T @ error + 2 * cfg['l2'] * v,
                 X.T @ error + 2 * cfg['l2'] * skip, np.asarray([error.sum()])]
        for i, grad in enumerate(grads):
            grad = np.clip(grad, -10., 10.)
            first[i] = .9 * first[i] + .1 * grad
            second[i] = .999 * second[i] + .001 * grad * grad
            params[i] -= cfg['learning_rate'] * (first[i] / (1 - .9 ** epoch)) / (np.sqrt(second[i] / (1 - .999 ** epoch)) + 1e-8)
    return {k: v.tolist() for k, v in zip(('W', 'b', 'v', 'skip', 'bias'), params)}


def _predict(network, X):
    return (np.tanh(X @ np.asarray(network['W']) + network['b']) @ np.asarray(network['v']) +
            X @ np.asarray(network['skip']) + network['bias'][0])


def _write(model, out):
    if out is not None:
        path = Path(out)
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_name(path.name + '.tmp')
        temp.write_text(json.dumps(model, indent=2, allow_nan=False) + '\n')
        temp.replace(path)
    return model


def train_contact_policy(roots, settings=None, *, out=None):
    """Return a sealed JSON model; insufficient evidence remains unusable for B.

    The exact trained weights are evaluated on held-out data and saved without
    a subsequent all-data refit. Validation data cannot affect normalization,
    weights, fallback direction fits or the OOD training support.
    """
    settings, cfg = _settings(settings)
    roots = [roots] if isinstance(roots, (str, Path)) else list(roots)
    sessions, rows, load_issues = _load(roots, settings)
    training_sessions = [s for s in sessions if s['role'] not in FINAL_ROLES]
    issues, compatibility = _compatibility(training_sessions, settings)
    reference = training_sessions[0]['meta'] if training_sessions else {}
    model = {'schema': SCHEMA, 'family': 'numpy_tanh_mlp_ensemble', 'usable_for_B': False,
             'settings': cfg, 'simulated': bool(settings.get('simulate', False)),
             'source_sessions': [], 'source_trials': [], 'fingers': [],
             'reference_T0': reference.get('T0'), 'axes_tcp': reference.get('config', {}).get('axes_tcp'),
             'normal_axis': reference.get('config', {}).get('force', {}).get('normal_axis'),
             'normal_signs': reference.get('config', {}).get('force', {}).get('normal_signs'),
             'signature': _hash(_signature(reference)) if reference else None,
             'blocking_reasons': load_issues + issues, 'compatibility': compatibility,
             'feature_definition': {'version': 'contact_descriptor_v1', 'names': feature_names(cfg['dd_pool']),
                                    'dd_pool': cfg['dd_pool'], 'runtime_inputs': ['two_current_tactile_descriptors'],
                                    'forbidden_runtime_inputs': ['delta_mm', 'TCP', 'trial_id', 'bottle_truth']},
             'label_definition': 'A: negative measured PROBE a displacement relative to shared T0; geometric supervision, not optimal recovery or measured bottle position',
             'caveats': ['Final validation never trains. Simulation and real data never mix.',
                         'Held-out error estimates geometry imitation, not prospective grasp success.',
                         'Digest detects artifact changes; it is not a cryptographic signature of a trusted author.']}
    if model['blocking_reasons']:
        return _write(_seal(model), out)
    observations, excluded = _observations(rows, reference, settings)
    row_labels = {r['trial_path']: r.get('label') for r in rows}
    usable = []
    for o in observations:
        # Historical exports use both "tipped" and the actual CLI's "tilted".
        # Neither can establish a stationary-bottle geometric target.
        if row_labels.get(o['trial_path']) in ('tilted', 'tipped', 'pushed', 'invalid'):
            excluded['invalid_or_object_moved'] = excluded.get('invalid_or_object_moved', 0) + 1
            continue
        if not o['measurement']['source'].startswith('measured_tcp'):
            excluded['unmeasured_position_not_policy_supervision'] = excluded.get('unmeasured_position_not_policy_supervision', 0) + 1
            continue
        o = deepcopy(o)
        o.update(target_action_mm=-o['x_mm'], label_source='A_measured_geometric_return_to_T0_not_optimal')
        usable.append(o)
    if cfg['include_successful_B']:
        B, B_excluded = _B_observations(rows, reference, settings, cfg)
        usable.extend(B); excluded.update(B_excluded)
    observations, features = [], []
    for o in usable:
        try:
            vector = feature_vector(o['descriptors'], dd_pool=cfg['dd_pool'])
        except (ValueError, TypeError, KeyError):
            excluded['missing_feature'] = excluded.get('missing_feature', 0) + 1
            continue
        if any(d.get('touches_edge') for d in o['descriptors']):
            excluded['edge_contact_uses_explicit_rule'] = excluded.get('edge_contact_uses_explicit_rule', 0) + 1
            continue
        observations.append(o); features.append(vector)
    model['excluded_counts'] = excluded
    model['source_sessions'] = sorted({o['session'] for o in observations})
    model['source_trials'] = [{k: o.get(k) for k in ('session', 'id', 'trial_path', 'metadata_sha256', 'group',
                                                    'label_source', 'transition_index', 'measurement', 'target_action_mm')}
                              for o in observations]
    model['training_snapshot_sha256'] = _hash({'features': [f.tolist() for f in features],
                                             'labels': [o['target_action_mm'] for o in observations],
                                             'provenance': model['source_trials']})
    if len(observations) < cfg['min_samples']:
        model['blocking_reasons'].append(f'need {cfg["min_samples"]} measured eligible samples; got {len(observations)}')
        return _write(_seal(model), out)
    try:
        train, valid, split = _split(observations, cfg)
    except ValueError as exc:
        model['blocking_reasons'].append(str(exc))
        return _write(_seal(model), out)
    split.update(training_indices=train.tolist(), validation_indices=valid.tolist())
    model['split'] = split
    if len(train) < cfg['min_train_samples'] or len(valid) < cfg['min_validation_samples']:
        model['blocking_reasons'].append('insufficient independent training or held-out validation samples')
        return _write(_seal(model), out)
    X = np.asarray(features); y = np.asarray([o['target_action_mm'] for o in observations])
    mean, scale = X[train].mean(0), X[train].std(0)
    scale = np.where(scale > 1e-6, scale, 1.)
    y_scale = max(float(y[train].std()), 1.)
    Z = (X - mean) / scale
    rng = np.random.default_rng(cfg['seed'])
    networks = [_network(Z[train], y[train] / y_scale, cfg, rng) for _ in range(cfg['ensemble_size'])]
    predictions = np.asarray([_predict(net, Z[valid]) * y_scale for net in networks])
    errors = np.abs(predictions.mean(0) - y[valid])
    fallback = _fit([observations[i] for i in train if observations[i]['label_source'].startswith('A_')],
                    'linear', False, settings.get('learning', {}))
    if fallback is None:
        model['blocking_reasons'].append('training split cannot establish two-finger direction fallback')
    else:
        model.update(fallback_fit=fallback, fingers=fallback['fingers'],
                     common_range_mm=fallback['common_range_mm'], common_linear_range_mm=fallback['common_linear_range_mm'])
    nearest = []
    for i in train:
        others = [j for j in train if observations[j]['group'] != observations[i]['group']]
        if others:
            nearest.append(float(np.min(np.sqrt(np.mean((Z[others] - Z[i]) ** 2, axis=1)))))
    radius = max(cfg['ood_min_radius'], float(np.quantile(nearest, .95)) * cfg['ood_radius_multiplier'] if nearest else cfg['ood_min_radius'])
    mae, p90 = float(errors.mean()), float(np.quantile(errors, .9))
    model.update(networks=networks, scaling={'mean': mean.tolist(), 'scale': scale.tolist(), 'y_scale': y_scale},
                 ood={'training_support': Z[train].tolist(), 'rms_radius': radius,
                      'calibration': 'training-only leave-trial-group-out nearest feature RMS distance'},
                 uncertainty={'validation_error_p90_mm': p90, 'max_mm': cfg['max_uncertainty_mm'],
                              'meaning': 'ensemble spread plus held-out residual floor; heuristic, not calibrated probability'},
                 validation={'n': len(valid), 'mae_mm': mae, 'p90_mm': p90, 'method': split['method'],
                             'predictions_mm': predictions.mean(0).tolist(), 'targets_mm': y[valid].tolist(),
                             'independent_final_validation': False},
                 training_count=len(train), validation_count=len(valid))
    if mae > cfg['max_validation_mae_mm'] or p90 > cfg['max_validation_p90_mm']:
        model['blocking_reasons'].append('held-out geometric prediction error exceeds configured gate')
    if p90 > cfg['max_uncertainty_mm']:
        model['blocking_reasons'].append('held-out residual exceeds runtime uncertainty allowance')
    if not all(np.isfinite(np.asarray(v)).all() for net in networks for v in net.values()):
        raise ValueError('ContactPolicy training produced nonfinite weights')
    model['usable_for_B'] = not model['blocking_reasons']
    return _write(_seal(model), out)


class ContactPolicy:
    """Frozen tactile-only policy implementing the existing Correction contract."""
    def __init__(self, model, servo_settings):
        if (model.get('schema') != SCHEMA or model.get('usable_for_B') is not True or
                model.get('integrity_sha256') != _digest(model) or
                model.get('model_id') != model.get('integrity_sha256', '')[:20]):
            raise ValueError('ContactPolicy is unusable or artifact integrity check failed')
        self.fit, self.cfg = deepcopy(model), deepcopy(servo_settings)
        self.fallback = AgenticCorrection(model['fallback_fit'], servo_settings)
        self.gain = self.fallback.gain
        self.mean = np.asarray(model['scaling']['mean'], float)
        self.scale = np.asarray(model['scaling']['scale'], float)
        self.support = np.asarray(model['ood']['training_support'], float)
        dims = len(feature_names(model['feature_definition']['dd_pool']))
        if model['feature_definition'].get('names') != feature_names(model['feature_definition']['dd_pool']):
            raise ValueError('ContactPolicy feature order differs from runtime implementation')
        if (self.mean.shape != (dims,) or self.scale.shape != (dims,) or np.any(self.scale <= 0) or
                self.support.ndim != 2 or self.support.shape[1] != dims or not len(self.support) or
                not np.isfinite(self.mean).all() or not np.isfinite(self.scale).all() or not np.isfinite(self.support).all()):
            raise ValueError('invalid ContactPolicy normalization/support')
        if model.get('fingers') != model['fallback_fit'].get('fingers') or model.get('blocking_reasons'):
            raise ValueError('ContactPolicy direction fallback or approval is inconsistent')
        _, checked = _settings({'contact_policy': model['settings']})
        positive = [model['scaling']['y_scale'], model['ood']['rms_radius'], model['uncertainty']['max_mm']]
        residual = model['uncertainty']['validation_error_p90_mm']
        if not np.isfinite(positive + [residual]).all() or min(positive) <= 0 or residual < 0:
            raise ValueError('invalid ContactPolicy numerical thresholds')
        if (model['uncertainty']['max_mm'] != checked['max_uncertainty_mm'] or
                model['validation']['mae_mm'] > checked['max_validation_mae_mm'] or
                model['validation']['p90_mm'] > checked['max_validation_p90_mm'] or
                model['validation']['p90_mm'] != residual or residual > checked['max_uncertainty_mm'] or
                model['validation']['n'] < checked['min_validation_samples']):
            raise ValueError('ContactPolicy validation/uncertainty gate inconsistent')
        if len(model['networks']) != checked['ensemble_size']:
            raise ValueError('ContactPolicy ensemble count mismatch')
        expected_shapes = {'W': (dims, checked['hidden_units']), 'b': (checked['hidden_units'],),
                           'v': (checked['hidden_units'],), 'skip': (dims,), 'bias': (1,)}
        for network in model['networks']:
            if set(network) != set(expected_shapes):
                raise ValueError('invalid ContactPolicy network keys')
            for name, shape in expected_shapes.items():
                array = np.asarray(network[name], float)
                if array.shape != shape or not np.isfinite(array).all():
                    raise ValueError('invalid ContactPolicy network shape or values')

    def predict(self, descriptors):
        try:
            X = feature_vector(descriptors, dd_pool=self.fit['feature_definition']['dd_pool'])
        except (ValueError, TypeError, KeyError) as exc:
            return {'valid': False, 'reason': 'policy_missing_features', 'detail': str(exc)}
        X = (X - self.mean) / self.scale
        distance = float(np.min(np.sqrt(np.mean((self.support - X) ** 2, axis=1))))
        if distance > self.fit['ood']['rms_radius']:
            return {'valid': False, 'reason': 'policy_out_of_distribution', 'ood_distance': distance}
        with np.errstate(over='ignore', invalid='ignore'):
            values = np.asarray([float(_predict(net, X[None])[0]) * self.fit['scaling']['y_scale'] for net in self.fit['networks']])
        if not np.isfinite(values).all():
            return {'valid': False, 'reason': 'policy_nonfinite_prediction'}
        uncertainty = float(values.std()) + self.fit['uncertainty']['validation_error_p90_mm']
        if not np.isfinite(uncertainty) or uncertainty > self.fit['uncertainty']['max_mm']:
            if not np.isfinite(uncertainty):
                return {'valid': False, 'reason': 'policy_nonfinite_uncertainty'}
            return {'valid': False, 'reason': 'policy_uncertain', 'uncertainty_mm': uncertainty}
        return {'valid': True, 'action_mm': float(values.mean()), 'uncertainty_mm': uncertainty,
                'ensemble_std_mm': float(values.std()), 'ood_distance': distance}

    def estimate(self, descriptors):
        prediction = self.predict(descriptors)
        return -prediction['action_mm'] if prediction['valid'] else None

    def decide(self, status, descriptors, bilateral):
        # These inherited directional recovery rules are explicit and audited.
        if status in ('no_contact', 'stall_no_contact') or any(d and d.get('touches_edge') for d in descriptors or []):
            result = self.fallback.decide(status, descriptors, bilateral)
            if result['reason'] not in ('no_contact_forward', 'stall_back', 'root_edge', 'tip_edge'):
                return {'accepted': False, 'rejected': True, 'e_mm': None, 'delta_mm': 0., 'edges': result['edges'],
                        'reason': 'policy_ambiguous_edge', 'model_id': self.fit['model_id']}
            return {**result, 'controller': 'explicit_direction_fallback', 'model_id': self.fit['model_id']}
        if status != 'contact' or not bilateral:
            return {'accepted': False, 'rejected': True, 'e_mm': None, 'delta_mm': 0., 'edges': [],
                    'reason': 'policy_requires_bilateral_contact', 'model_id': self.fit['model_id']}
        prediction = self.predict(descriptors)
        if not prediction['valid']:
            return {**prediction, 'accepted': False, 'rejected': True, 'e_mm': None,
                    'delta_mm': 0., 'edges': [], 'model_id': self.fit['model_id']}
        error = -prediction['action_mm']
        accepted = abs(error) <= self.cfg['tol_mm']
        limit = min(float(self.cfg['max_delta_mm']), self.fit['settings']['max_action_mm'])
        return {**prediction, 'accepted': accepted, 'rejected': False, 'e_mm': error,
                'delta_mm': 0. if accepted else float(np.clip(self.gain * prediction['action_mm'], -limit, limit)),
                'edges': [], 'reason': 'within_tolerance' if accepted else 'learned_contact_recovery',
                'gain': self.gain, 'controller': SCHEMA, 'model_id': self.fit['model_id']}
