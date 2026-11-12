"""Offline graph-supervised I/pi/beta and an opt-in RecoveryAdapter runtime.

This module opens no hardware and does not alter the existing V4 scheduler.
Whole-trial splits precede graph construction: held-out successful sinks cannot
create training labels. Only observation vectors enter any learned function.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import json
from pathlib import Path

import numpy as np

from bottle_scan.learned_skill import DEFAULTS as STATE_DEFAULTS, _partition, _design, _sigmoid, _metrics, _seal
from bottle_scan.contact_policy import DEFAULTS as POLICY_DEFAULTS, _network, _predict, _settings as policy_settings
from bottle_scan.recovery_skill import RecoveryAdapter
from .noise import MODALITIES, fit_noise, transformed
from .graph import build_graph, finite
from .labels import generate_labels, transition_weight
from .clusters import cluster_nodes, ground_symbol, observation_only
from .facts import _fact

SCHEMA = 'symbolic_graph_a_skill_v1'


def feature_vector(observation, noise_model, modalities):
    clean = observation_only(observation)
    pieces = [transformed(clean, name, noise_model) for name in modalities]
    if any(x is None for x in pieces):
        raise ValueError('required_observation_modality_unavailable')
    x = np.concatenate(pieces)
    if not np.isfinite(x).all():
        raise ValueError('nonfinite_observation_feature')
    return x


def partition_rows(rows, config=None):
    cfg = {**STATE_DEFAULTS, **(config or {})}
    samples = [{'group': str(r.get('group_id') or f"{r['session']}|{r['trial_id']}")} for r in rows]
    explicit = [r.get('split', r.get('training_split')) for r in rows]
    if any(x is not None for x in explicit):
        if any(x not in ('train', 'calibration', 'validation') for x in explicit):
            raise ValueError('explicit split must cover every row with train/calibration/validation')
        by_group = {}
        for s, part in zip(samples, explicit):
            old = by_group.setdefault(s['group'], part)
            if old != part:
                raise ValueError('one physical group crosses graph partitions')
        parts = {p: sorted(g for g, label in by_group.items() if label == p)
                 for p in ('train', 'calibration', 'validation')}
        indices = {p: np.asarray([i for i, s in enumerate(samples) if s['group'] in gs], int)
                   for p, gs in parts.items()}
    else:
        parts, indices = _partition(samples, cfg)
    partitions = {p: [deepcopy(rows[i]) for i in ii] for p, ii in indices.items()}
    for part, selected in partitions.items():
        for row in selected:
            row['split'] = part
    return parts, partitions


def partition_graphs(rows, config=None):
    """Fit transform/noise on train, then build three disconnected label graphs."""
    cfg = config or {}
    parts, partitions = partition_rows(rows, cfg.get('state_learning'))
    modalities = tuple(cfg.get('modalities', MODALITIES))
    noise = fit_noise(partitions['train'], {**cfg.get('noise', {}), 'modalities': modalities})
    graph_cfg = {**cfg.get('graph', {}), 'modalities': list(modalities)}
    if graph_cfg.get('validation_rows') or graph_cfg.get('include_evaluation_nodes'):
        raise ValueError('held-out validation nodes cannot supervise training graphs')
    graphs, labels = {}, []
    for part, selected in partitions.items():
        graphs[part] = build_graph(selected, noise, graph_cfg)
        labels.extend({**x, 'split': part} for x in generate_labels(graphs[part], selected))
    return parts, partitions, noise, graphs, labels


def _coverage(labels, state_cfg, policy_cfg):
    result = []
    for head, field in (('I', 'invocation_label'), ('beta', 'exit_label')):
        for part in ('train', 'calibration', 'validation'):
            for value in (0, 1):
                groups = {s['source_group'] for s in labels if s['split'] == part and s[field] == value}
                required = state_cfg['min_' + part + '_per_class']
                result.append({'component': head, 'partition': part, 'class': value,
                               'groups': len(groups), 'required': required, 'pass': len(groups) >= required})
    actions = [s for s in labels if s.get('pi_eligible') and finite(s.get('delta_a_target_mm'))]
    for part, minimum in ((None, policy_cfg['min_samples']), ('train', policy_cfg['min_train_samples']),
                          ('validation', policy_cfg['min_validation_samples'])):
        groups = {s['source_group'] for s in actions if part is None or s['split'] == part}
        result.append({'component': 'pi', 'partition': part or 'all', 'groups': len(groups),
                       'required': minimum, 'pass': len(groups) >= minimum})
    return result


def coverage_facts(rows, config=None):
    """Compute guide coverage from separately built whole-group label graphs.

    Extracted rows need no pre-existing split. This function assigns the same
    deterministic splits used by train_graph_skill, fits train-only transforms,
    and recomputes each partition's labels without importing another partition's
    successful sinks. The guide's pi fact remains >=20 qualified observations;
    the existing policy learner's additional sample gates are reported apart.
    """
    cfg = config or {}
    parts, partitioned, noise, graphs, labels = partition_graphs(rows, cfg)
    versions = {p: g['graph_version'] for p, g in graphs.items()}
    result = []
    for head, field in (('I', 'invocation_label'), ('beta', 'exit_label')):
        groups = {p: {str(k): set() for k in (0, 1)} for p in parts}
        evidence = []
        for label in labels:
            value = label[field]
            if value in (0, 1) and not label.get('conflicting_outcomes'):
                groups[label['split']][str(value)].add(label['source_group'])
                evidence.append(label['node_id'])
        numeric = {p: {k: len(v) for k, v in counts.items()} for p, counts in groups.items()}
        minimum = {'train': 4, 'calibration': 2, 'validation': 2}
        passed = all(numeric[p][str(k)] >= n for p, n in minimum.items() for k in (0, 1))
        result.append(_fact('coverage_ok', [head], evidence,
            {'counts': numeric, 'minimum': minimum, 'partition_graph_versions': versions,
             'label_source': 'separately_rebuilt_partition_graphs',
             'grouping': 'whole_physical_trial_groups_before_graph_labels'}, value=passed))
    qualified = [label for label in labels if label.get('pi_eligible')
                 and finite(label.get('delta_a_target_mm')) and not label.get('conflicting_outcomes')]
    result.append(_fact('coverage_ok', ['pi'], [label['node_id'] for label in qualified],
        {'qualified_samples': len(qualified), 'minimum': 20,
         'qualified_groups': len({label['source_group'] for label in qualified}),
         'partition_samples': {p: sum(label['split'] == p for label in qualified) for p in parts},
         'partition_groups': {p: len({label['source_group'] for label in qualified if label['split'] == p}) for p in parts},
         'partition_graph_versions': versions,
         'label_source': 'separately_rebuilt_partition_graphs'}, value=len(qualified) >= 20))
    sc = {**STATE_DEFAULTS, **cfg.get('state_learning', {})}
    _, pc = policy_settings({'contact_policy': cfg.get('contact_policy', {})})
    return {'facts': result, 'split_groups': parts, 'partition_graph_versions': versions,
            'partition_labels': labels,
            'partition_evidence': {p: [r['node_id'] for r in selected] for p, selected in partitioned.items()},
            'noise_fit_node_ids': noise['fit_node_ids'],
            'noise_status': {name: item['status'] for name, item in noise['modalities'].items()},
            'training_preconditions_summary': _coverage(labels, sc, pc),
            'label_partition_policy': 'split first; train-only noise/PCA; independently rebuild all three graphs'}


def _fit_head(samples, cfg):
    """Reuse the existing quadratic logistic implementation with shared splits."""
    X = np.asarray([s['features'] for s in samples]); y = np.asarray([s['label'] for s in samples], bool)
    ii = {p: np.asarray([i for i, s in enumerate(samples) if s['split'] == p], int)
          for p in ('train', 'calibration', 'validation')}
    tr, ca, va = (ii[p] for p in ('train', 'calibration', 'validation'))
    mean, scale = X[tr].mean(0), X[tr].std(0)
    scale[scale < 1e-6] = 1.
    Z = _design(X, mean, scale); w = np.zeros(Z.shape[1])
    counts = Counter(samples[i]['group'] for i in tr)
    weights = np.asarray([1. / counts[samples[i]['group']] for i in tr]); weights /= weights.sum()
    for _ in range(cfg['epochs']):
        penalty = cfg['l2'] * w; penalty[0] = 0
        w -= cfg['learning_rate'] * (Z[tr].T @ (weights * (_sigmoid(Z[tr] @ w) - y[tr])) + penalty)
    normalized = (X - mean) / scale
    neighbors = []
    for i in tr:
        other = [j for j in tr if samples[j]['group'] != samples[i]['group']]
        if other:
            neighbors.append(float(np.min(np.sqrt(np.mean((normalized[other] - normalized[i]) ** 2, axis=1)))))
    radius = max(cfg['ood_min_radius'], float(np.quantile(neighbors, .95)) * cfg['ood_radius_multiplier'] if neighbors else 0.)
    distances = np.min(np.sqrt(np.mean((normalized[:, None] - normalized[tr][None]) ** 2, axis=2)), axis=1)
    p = _sigmoid(Z @ w)
    low = min(.4, float(p[ca][y[ca]].min()) - cfg['score_margin'])
    high = max(.6, float(p[ca][~y[ca]].max()) + cfg['score_margin'])
    result = {'family': 'regularized_quadratic_logistic_observations', 'weights': w.tolist(),
              'mean': mean.tolist(), 'scale': scale.tolist(), 'thresholds': {'low': low, 'high': high},
              'support': normalized[tr].tolist(), 'support_radius': radius}
    reasons = []
    for name, indices in (('calibration', ca), ('validation', va)):
        met = _metrics(y[indices], p[indices], distances[indices] <= radius, low, high)
        result[name] = met
        if (not 0 <= low < .5 < high <= 1 or met['coverage'] < cfg['min_coverage'] or
                not met['true_positive'] or not met['true_negative'] or
                met['false_positive_rate'] > cfg['max_false_positive_rate'] or
                met['false_negative_rate'] > cfg['max_false_negative_rate']):
            reasons.append(name + '_discrimination_or_coverage_failed')
    result.update(usable=not reasons, blocking_reasons=reasons)
    return result


def _fit_policy(samples, cfg):
    X = np.asarray([s['features'] for s in samples]); y = np.asarray([s['target'] for s in samples])
    tr = np.asarray([i for i, s in enumerate(samples) if s['split'] == 'train'], int)
    ca = np.asarray([i for i, s in enumerate(samples) if s['split'] == 'calibration'], int)
    va = np.asarray([i for i, s in enumerate(samples) if s['split'] == 'validation'], int)
    mean, scale = X[tr].mean(0), X[tr].std(0); scale[scale < 1e-6] = 1.
    normalized = (X - mean) / scale
    rng = np.random.default_rng(cfg['seed'])
    counts = Counter(samples[i]['group'] for i in tr)
    weights = np.asarray([samples[i]['weight'] / counts[samples[i]['group']] for i in tr]); weights /= weights.sum()
    networks = []
    for _ in range(cfg['ensemble_size']):
        take = rng.choice(tr, size=len(tr), replace=True, p=weights)
        networks.append(_network(normalized[take], y[take], cfg, rng))
    predictions = np.asarray([_predict(net, normalized) for net in networks])
    errors = np.abs(predictions.mean(0) - y)
    calibration_error = float(np.quantile(errors[ca], .9)) if len(ca) else None
    mae, p90 = float(errors[va].mean()), float(np.quantile(errors[va], .9))
    reasons = []
    if calibration_error is None:
        reasons.append('missing_calibration_action_targets')
    if mae > cfg['max_validation_mae_mm'] or p90 > cfg['max_validation_p90_mm']:
        reasons.append('held_out_action_error_exceeds_existing_contact_policy_limits')
    return {'family': 'contact_policy_numpy_tanh_ensemble', 'networks': networks,
            'mean': mean.tolist(), 'scale': scale.tolist(), 'calibration_error_mm': calibration_error,
            'validation': {'n': len(va), 'mae_mm': mae, 'p90_mm': p90},
            'usable': not reasons, 'blocking_reasons': reasons,
            'sample_targets': [{k: v for k, v in s.items() if k != 'features'} for s in samples],
            'weighting': 'equal physical group; graph R-decreasing B weight via bootstrap probabilities'}


def _finish(status, model, out):
    result = {'training_status': status, 'model': model}
    if out is not None:
        dest = Path(out); dest.mkdir(parents=True, exist_ok=True)
        for name, value in (('training_status.json', status), ('symbolic_skill.json', model)):
            if value is not None:
                path = dest / name
                if path.exists():
                    raise FileExistsError('use a new training output directory: ' + str(path))
                path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    return result


def train_graph_skill(rows, graph, noise_model, clusters, offline_acceptance, config=None, out=None):
    """Train all three heads together, or write explicit not_trained evidence.

    The passed graph/noise/clusters describe the source analysis only. They are
    rebuilt per split; fitting on their full-data PCA or graph sinks would leak.
    """
    cfg = deepcopy(config or {}); sc = {**STATE_DEFAULTS, **cfg.get('state_learning', {})}
    _, pc = policy_settings({'contact_policy': cfg.get('contact_policy', {})})
    status = {'schema': 'symbolic_skill_training_status_v1', 'status': 'not_trained',
              'source_graph_version': graph.get('graph_version'), 'hardware_connected': False,
              'existing_scheduler_modified': False, 'blocking_reasons': [],
              'offline_acceptance': {k: offline_acceptance.get(k) for k in ('V1', 'V2', 'V4')}}
    reasons = status['blocking_reasons']
    for key in ('V1', 'V2', 'V4'):
        if (offline_acceptance.get(key) or {}).get('status') != 'pass':
            reasons.append('offline_acceptance_' + key + '_not_passed')
    if graph.get('diagnostic_only') or graph.get('training_labels_allowed') is False:
        reasons.append('diagnostic_or_sealed_graph_cannot_train')
    if any(r.get('evaluation_only') for r in rows):
        reasons.append('independent_evaluation_rows_cannot_train')
    if reasons:
        return _finish(status, None, out)
    parts, partitioned, noise, graphs, labels = partition_graphs(rows, cfg)
    modalities = list(cfg.get('modalities', MODALITIES))
    status.update(split_groups=parts, partition_graph_versions={p: g['graph_version'] for p, g in graphs.items()},
                  noise_fit_node_ids=noise['fit_node_ids'], labels=labels,
                  graph_partition_policy='rebuild topology and outcome labels separately in each whole-group split')
    unavailable = [m for m in modalities if noise['modalities'].get(m, {}).get('status') != 'ready']
    if unavailable:
        reasons.append('train_only_noise_unavailable:' + ','.join(unavailable))
    coverage = _coverage(labels, sc, pc); status['coverage'] = coverage
    if not all(c['pass'] for c in coverage):
        reasons.append('graph_label_coverage_insufficient')
    if reasons:
        return _finish(status, None, out)
    by_id = {r['node_id']: r for selected in partitioned.values() for r in selected}
    vectors = {}
    for ident, row in by_id.items():
        try:
            vectors[ident] = feature_vector(row, noise, modalities).tolist()
        except ValueError:
            reasons.append('missing_observation_features:' + ident)
    if reasons:
        return _finish(status, None, out)
    heads = {}
    for head, field in (('I', 'invocation_label'), ('beta', 'exit_label')):
        samples = [{'node_id': lab['node_id'], 'group': lab['source_group'], 'split': lab['split'],
                    'label': bool(lab[field]), 'features': vectors[lab['node_id']]}
                   for lab in labels if lab[field] is not None]
        heads[head] = _fit_head(samples, sc)
    action_samples = []
    for lab in labels:
        row = by_id[lab['node_id']]
        if row.get('phase') == 'A' and lab['pi_eligible']:
            action_samples.append({'node_id': lab['node_id'], 'group': lab['source_group'], 'split': lab['split'],
                                   'target': lab['delta_a_target_mm'], 'target_node_id': lab['target_node_id'],
                                   'features': vectors[lab['node_id']], 'weight': 1., 'source': 'A_graph_shortest_path_target'})
    for part, g in graphs.items():
        for edge in g['edges']:
            if edge['kind'] != 'observed':
                continue
            weight = transition_weight(g['R'][edge['source']], g['R'][edge['target']])
            action = edge.get('delta_a_mm')
            if weight > 0 and finite(action):
                source = by_id[edge['source']]
                action_samples.append({'node_id': edge['source'], 'target_node_id': edge['target'],
                                       'group': source.get('group_id') or f"{source['session']}|{source['trial_id']}",
                                       'split': part, 'target': action, 'features': vectors[edge['source']],
                                       'weight': weight, 'source': 'B_observed_R_decreasing', 'edge_id': edge['edge_id']})
    counts = Counter(s['split'] for s in action_samples)
    if counts['train'] < pc['min_train_samples'] or counts['validation'] < pc['min_validation_samples'] or not counts['calibration']:
        reasons.append('verified_graph_action_samples_insufficient')
        return _finish(status, None, out)
    heads['pi'] = _fit_policy(action_samples, pc)
    train_clusters = cluster_nodes(graphs['train'], partitioned['train'], noise,
                                   {**cfg.get('graph', {}), 'modalities': modalities})
    if not {'NeedShift', 'HandoffReady'} <= {c['symbol'] for c in train_clusters['clusters']}:
        reasons.append('training_clusters_missing_recovery_or_handoff_support')
    for name, head in heads.items():
        reasons.extend(name + ':' + reason for reason in head['blocking_reasons'])
    model = _seal({'schema': SCHEMA, 'usable': not reasons, 'graph_version': graphs['train']['graph_version'],
                   'source_graph_version': graph['graph_version'], 'heads': heads, 'noise_model': noise,
                   'modalities': modalities, 'clusters': train_clusters, 'split_groups': parts,
                   'partition_graph_versions': status['partition_graph_versions'], 'state_learning': sc,
                   'contact_policy': pc, 'three_components_updated_together': True,
                   'supervision': 'graph_R_and_shortest_path_success_set; never return_T0',
                   'runtime': {'max_corrections': 3, 'max_total_action_mm': 25., 'tracking_tolerance_mm': 2.},
                   'certified_for_hardware': False, 'scheduler_default_enabled': False,
                   'blocking_reasons': list(reasons)})
    status.update(status='trained' if model['usable'] else 'trained_candidate_failed_validation',
                  model_id=model['model_id'], validation={k: h['validation'] for k, h in heads.items()})
    return _finish(status, model, out)


def _head_result(head, x):
    mean, scale = np.asarray(head['mean']), np.asarray(head['scale'])
    normalized = (x - mean) / scale
    dist = float(np.min(np.sqrt(np.mean((np.asarray(head['support']) - normalized) ** 2, axis=1))))
    score = float(_sigmoid(_design(x[None], mean, scale) @ head['weights'])[0])
    value = None
    if dist <= head['support_radius']:
        if score >= head['thresholds']['high']:
            value = True
        elif score <= head['thresholds']['low']:
            value = False
    return {'value': value, 'score': score, 'support_distance': dist}


class FeatureRecoveryAdapter:
    """Wrap an existing RecoveryAdapter with the frozen observation extractor.

    The caller supplies extraction from its fresh stable sensor window. Motion,
    force limits, ownership and receipts remain in the original adapter.
    """
    def __init__(self, adapter: RecoveryAdapter, encode_observation):
        self.adapter, self.encode_observation = adapter, encode_observation

    def _encode(self, observation):
        result = observation_only(self.encode_observation(observation))
        result['timestamp_ns'] = observation.get('timestamp_ns', observation.get('t_ns'))
        return result

    def initial_probe(self):
        return self._encode(self.adapter.initial_probe())

    def observe(self):
        return self._encode(self.adapter.observe())

    def execute(self, action):
        return self.adapter.execute(action)

    def handoff(self, result):
        return self.adapter.handoff(result)


def model_execution_report(model_or_training_result):
    """Shape a real trained candidate for facts.model_executable, not B pass."""
    supplied = model_or_training_result or {}
    model = supplied.get('model') if 'training_status' in supplied else supplied
    model = model or {}
    supported = {c['symbol'] for c in model.get('clusters', {}).get('clusters', [])}
    usable = bool(model.get('schema') == SCHEMA and model.get('usable') is True
                  and all(model.get('heads', {}).get(h, {}).get('usable') is True for h in ('I', 'pi', 'beta'))
                  and {'NeedShift', 'HandoffReady'} <= supported)
    return {'model_id': model.get('model_id'), 'usable_for_B': usable,
            'graph_version': model.get('graph_version'),
            'evidence_node_ids': model.get('noise_model', {}).get('fit_node_ids', []),
            'meaning': 'offline executable development candidate; not B-validated or hardware enabled',
            'B_validated': False, 'independent_B_validated': False,
            'scheduler_default_enabled': False,
            'blocking_reasons': model.get('blocking_reasons') or supplied.get('training_status', {}).get('blocking_reasons', [])}


def load_symbolic_skill(model_or_path):
    model = json.loads(Path(model_or_path).read_text()) if isinstance(model_or_path, (str, Path)) else model_or_path
    return SymbolicRecoverySkill(model)


class ContactBackendRecoveryAdapter:
    """Opt-in facade over ContactRecoveryBackend's existing guarded methods.

    The enclosing backend invocation owns activation, recording, the exclusive
    lease and its finally/disarm block. Supply its active _SkillRunner and
    measured initial target; constructing this facade opens no hardware.

    encode_probe(probe, legacy_observation) must encode the actual stable DD and
    camera window using the frozen encoders/calibration. It must not synthesize
    embeddings from centroids. Output is filtered to deployment fields, and the
    timestamp always comes from the backend's independently checked window.
    """
    def __init__(self, backend, runner, target, encode_probe, *, caller_paused=False, on_handoff=None):
        if caller_paused is not True or getattr(backend, 'executing', False) is not True:
            raise ValueError('existing backend must own an exclusive paused invocation')
        self.backend, self.runner = backend, runner
        self.target = np.asarray(target, float).copy()
        self.encode_probe, self.on_handoff = encode_probe, on_handoff
        self.pending = None; self.latest_probe = None
        self.observations = []; self.corrections = []; self.result = None

    def _observe(self, status, probe):
        legacy = self.backend._observation(status, probe)
        if legacy.get('bilateral') is not True or legacy.get('force_stable') is not True:
            raise ValueError('existing backend did not produce a stable bilateral probe')
        observation = observation_only(self.encode_probe(probe, legacy))
        observation['bilateral_contact'] = legacy['bilateral']
        observation['timestamp_ns'] = legacy['timestamp_ns']
        self.latest_probe = deepcopy(probe)
        self.observations.append(deepcopy(observation))
        return observation

    def initial_probe(self):
        self.backend._last_window_end = self.backend._last_window_ids = None
        force = self.backend.cfg['force']
        probe = self.runner.regulate('PROBE', force['target'] * force['probe_fraction'],
                                     force['probe_tolerance'], force['probe_stable_s'], with_dd=True)
        return self._observe('contact' if probe is not None else 'probe_unstable', probe)

    def execute(self, action):
        row = {'requested_action': deepcopy(action), 'receipt': None}
        # This is the existing method, including release/retreat/correct/approach,
        # motion bounds, hardware checks and measured execution receipt.
        self.target, status, probe, row = self.backend._recover_once(self.runner, self.target, action, row)
        self.pending = (status, probe)
        self.corrections.append(deepcopy(row))
        return deepcopy(row['receipt'])

    def observe(self):
        if self.pending is None:
            raise ValueError('no executed correction awaiting observation')
        status, probe = self.pending; self.pending = None
        if probe is None:
            # The old single-open variant deliberately omits touch. This opt-in
            # facade obtains the graph-required fresh window without changing
            # that variant's flag, method, or default scheduler behavior.
            status, probe = self.runner.touch()
            self.target[:3, 3] = self.backend.hw.pose()[:3, 3]
        return self._observe(status, probe)

    def handoff(self, result):
        self.result = deepcopy(result)
        if result.get('completed'):
            self.backend.last_probe = deepcopy(self.latest_probe)
        if self.on_handoff is not None:
            self.on_handoff(deepcopy(result))


def make_contact_backend_runtime(model_or_path, backend, runner, target, encode_probe,
                                 *, caller_paused=False, on_handoff=None):
    """Return (skill, adapter) for an explicitly selected backend invocation.

    Caller executes skill.run(adapter) inside its existing ownership/recording
    scope and consumes the HandoffReady contract before ForcePolicy handoff.
    This factory never calls backend.execute(), replaces backend.recovery, or
    registers itself with the existing scheduler.
    """
    skill = load_symbolic_skill(model_or_path)
    adapter = ContactBackendRecoveryAdapter(backend, runner, target, encode_probe,
                                            caller_paused=caller_paused, on_handoff=on_handoff)
    return skill, adapter


class SymbolicRecoverySkill:
    """Opt-in observe → I → correction → fresh observe → beta handoff loop."""
    def __init__(self, model):
        if model.get('schema') != SCHEMA or not model.get('usable'):
            raise ValueError('a usable jointly trained symbolic skill is required')
        self.model = deepcopy(model)

    def evaluate(self, observation):
        m = self.model
        symbol = ground_symbol(observation, m['clusters'], m['noise_model'], {'modalities': m['modalities']})
        result = {'symbol': symbol, 'invoke': False, 'completed': False, 'action': None}
        if symbol['symbol'] in ('Unknown', 'Unrecoverable', 'Unsupported', 'Unreachable', 'Ambiguous'):
            result['reason'] = 'unreachable_in_current_graph' if symbol['symbol'] in ('Unrecoverable','Unreachable') else 'unsupported_symbol'
            return result
        try:
            x = feature_vector(observation, m['noise_model'], m['modalities'])
        except ValueError as exc:
            result['reason'] = str(exc)
            return result
        invocation, beta = (_head_result(m['heads'][h], x) for h in ('I', 'beta'))
        result.update(invocation=invocation, exit=beta)
        if symbol['symbol'] == 'HandoffReady' and beta['value'] is True:
            result.update(completed=True, reason='fresh_observed_HandoffReady')
            return result
        if symbol['symbol'] != 'NeedShift' or invocation['value'] is not True:
            result['reason'] = 'invocation_or_exit_not_established'
            return result
        head = m['heads']['pi']; z = (x - head['mean']) / head['scale']
        predictions = np.asarray([_predict(n, z[None])[0] for n in head['networks']])
        delta = float(predictions.mean()); uncertainty = float(predictions.std() + head['calibration_error_mm'])
        result['uncertainty_mm'] = uncertainty
        if uncertainty > m['contact_policy']['max_uncertainty_mm']:
            result['reason'] = 'recovery_uncertainty_exceeded'
        elif abs(delta) > min(15., m['contact_policy']['max_action_mm']):
            result['reason'] = 'recovery_action_exceeds_existing_limit'
        elif np.sign(delta) != symbol['args']['sign']:
            result['reason'] = 'recovery_direction_disagrees_with_graph'
        else:
            result.update(invoke=True, reason='graph_supported_recovery',
                          action={'type': 'regrasp', 'axis': 'a', 'unit': 'mm', 'delta_mm': delta,
                                  'interface': 'open_retreat_translate_reapproach_probe', 'reobserve_required': True})
        return result

    def run(self, adapter: RecoveryAdapter):
        events = []; total = 0.; corrections = 0; latest = None
        def finish(reason, completed=False, decision=None):
            symbol = (decision or {}).get('symbol', {})
            result = {'completed': completed, 'hand_back': completed, 'reason': reason,
                      'corrections': corrections, 'total_action_mm': total, 'events': events,
                      'graph_version': self.model['graph_version'], 'model_id': self.model['model_id']}
            if completed:
                result['handoff'] = {'symbol': 'HandoffReady', 'object': symbol['args']['object'],
                                     'protocol': symbol['args']['protocol'], 'cluster_id': symbol['cluster_id'],
                                     'R': 0, 'graph_version': symbol['graph_version'],
                                     'latest_contact': {**observation_only(latest), 'timestamp_ns': latest['timestamp_ns']}}
            adapter.handoff(result)
            return result
        try:
            latest = adapter.initial_probe()
            if 'timestamp_ns' not in latest:
                latest = {**latest, 'timestamp_ns': latest.get('t_ns')}
            if type(latest['timestamp_ns']) is not int:
                return finish('initial_observation_timestamp_missing')
            while True:
                decision = self.evaluate(latest); events.append({'observation_timestamp_ns': latest['timestamp_ns'], **decision})
                if decision['completed']:
                    return finish(decision['reason'], True, decision)
                if not decision['invoke']:
                    return finish(decision['reason'])
                if corrections >= min(3, self.model['runtime']['max_corrections']):
                    return finish('correction_budget_exhausted')
                delta = decision['action']['delta_mm']
                if total + abs(delta) > min(25., self.model['runtime']['max_total_action_mm']):
                    return finish('cumulative_action_budget_exhausted')
                receipt = adapter.execute(decision['action'])
                if receipt.get('applied') is not True or not finite(receipt.get('action_mm')):
                    return finish('action_not_confirmed_by_adapter')
                if abs(receipt['action_mm'] - delta) > self.model['runtime']['tracking_tolerance_mm']:
                    return finish('action_receipt_tracking_mismatch')
                following = adapter.observe()
                following = {**following, 'timestamp_ns': following.get('timestamp_ns', following.get('t_ns'))}
                done, ts = receipt.get('completed_ns'), following['timestamp_ns']
                if type(done) is not int or type(ts) is not int or not latest['timestamp_ns'] <= done < ts:
                    return finish('post_action_observation_not_fresh')
                corrections += 1; total += abs(receipt['action_mm']); latest = following
                events.append({'receipt': deepcopy(receipt), 'post_observation_timestamp_ns': ts})
        except Exception as exc:
            finish('adapter_or_observation_error:' + type(exc).__name__)
            raise
