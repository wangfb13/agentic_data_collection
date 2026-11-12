"""Deterministic, offline contact-state graph. No hardware or LLM imports."""
from __future__ import annotations

from collections import defaultdict
import hashlib
import heapq
import json
import math

import numpy as np

GATE_FIELDS = ('object_id', 'protocol_id', 'gripper_state', 'edge_class',
               'force_bin', 'encoder_version', 'time_period', 'sensor_version')
SINK = '__success__'


def finite(value):
    return isinstance(value, (int, float, np.number)) and not isinstance(value, bool) and math.isfinite(float(value))


def cost_value(value):
    """JSON null encodes an unreachable (infinite) path cost."""
    return float(value) if finite(value) else math.inf


def _key(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'))


def _known(value):
    if value is None:
        return False
    if isinstance(value, dict):
        return bool(value) and all(_known(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return bool(value) and all(_known(v) for v in value)
    return True


def gate(a, b, config=None):
    config = config or {}
    if a.get('bilateral_contact') is not True or b.get('bilateral_contact') is not True:
        return False
    for field in GATE_FIELDS:
        if field=='time_period' and config.get('ignore_time_period'):continue
        if field=='time_period' and not config.get('require_time_period') and a.get(field) is None and b.get(field) is None:continue
        if field=='sensor_version' and (not _known(a.get(field)) or not _known(b.get(field))):continue
        if field=='encoder_version' and 'modalities' in config and 'tac' not in config['modalities']:continue
        if field=='edge_class' and config.get('optional_edge_orientation',True) and (not _known(a.get(field)) or not _known(b.get(field))):continue
        if field == 'protocol_id' and config.get('ignore_protocol', False):
            continue
        # An unknown discrete gate cannot establish equal physical conditions.
        if field == 'edge_class' and config.get('ignore_unknown_edge_class') and (not _known(a.get(field)) or not _known(b.get(field))):
            continue
        if not _known(a.get(field)) or not _known(b.get(field)) or _key(a[field]) != _key(b[field]):
            return False
    return True


def _known_gate_mismatches(a, b, config):
    """Known physical differences can refute arrival; missing gates cannot."""
    fields = ('bilateral_contact',) + GATE_FIELDS
    return [field for field in fields
            if not (field == 'protocol_id' and config.get('ignore_protocol'))
            and not (field == 'time_period' and config.get('ignore_time_period'))
            and not (field == 'encoder_version' and 'modalities' in config and 'tac' not in config['modalities'])
            and _known(a.get(field)) and _known(b.get(field))
            and _key(a[field]) != _key(b[field])]


def measurement_distance(a, b, noise_model, config=None):
    config = config or {}
    if not gate(a, b, config):
        return math.inf
    from .noise import noise_distance
    return float(noise_distance(a, b, noise_model, modalities=config.get('modalities')))


def pairwise_distances(rows, noise_model, config=None):
    config = config or {}
    n = len(rows)
    if 'modalities' in noise_model:
        from .noise import transformed, MODALITIES, threshold
        distances = np.zeros((n, n), dtype=float)
        for name in config.get('modalities', MODALITIES):
            item = noise_model['modalities'].get(name, {})
            matrix = np.full((n, n), np.inf)
            if item.get('status') == 'ready':
                transformed_rows = [(i, transformed(row, name, noise_model)) for i, row in enumerate(rows)]
                valid = [(i, value) for i, value in transformed_rows if value is not None]
                if valid:
                    indexes, values = zip(*valid)
                    X = np.asarray(values); P = np.asarray(item['precision'])
                    XP = X @ P
                    diagonal = np.einsum('ij,ij->i', XP, X)
                    squared = np.maximum(0., diagonal[:, None] + diagonal[None, :] - 2. * XP @ X.T)
                    # Identical samples have zero distance despite cancellation.
                    np.fill_diagonal(squared, 0.)
                    scale = noise_model.get('w_vis', .5) if name == 'vis' else 1.
                    matrix[np.ix_(indexes, indexes)] = np.sqrt(squared) / threshold(noise_model,name,config.get('modalities',MODALITIES)) * scale
            distances = np.maximum(distances, matrix)
        for i, a in enumerate(rows):
            for j in range(i + 1):
                if not gate(a, rows[j], config):
                    distances[i, j] = distances[j, i] = math.inf
        return distances
    distances = np.full((n, n), np.inf, dtype=float)
    for i, row in enumerate(rows):
        distances[i, i] = measurement_distance(row, row, noise_model, config)
        for j in range(i):
            distances[i, j] = distances[j, i] = measurement_distance(row, rows[j], noise_model, config)
    return distances


def _edge(source, target, kind, cost, status, **extra):
    identity = _key([source, target, kind, extra.get('delta_a_mm')])
    return {'edge_id': 'e_' + hashlib.sha256(identity.encode()).hexdigest()[:16],
            'source': source, 'target': target, 'kind': kind, 'cost': float(cost),
            'status': status, **extra}


def _risk(row, config):
    limit = row.get('force_max', config.get('force_max', 15.0))
    force = row.get('peak_normal_force', row.get('normal_forces'))
    force_risk = 0.0
    if finite(force):
        force = [force]
    if isinstance(force, (list, tuple)) and force and all(finite(x) for x in force) and finite(limit) and limit > 0:
        force_risk = max(0., max(float(x) for x in force) / float(limit) - 1.)
    loss = float(row.get('contact_lost') is True)
    return {'force_risk': force_risk, 'contact_loss_risk': loss,
            'risk_definition': 'observed_force_limit_exceedance_and_observed_contact_loss',
            'force_risk_observed': isinstance(force, (list, tuple)) and bool(force)}


def _base_cost(row, config):
    r = _risk(row, config)
    return 1. + float(config.get('lambda_force', 1.)) * r['force_risk'] + float(config.get('lambda_contact', 1.)) * r['contact_loss_risk'], r


def reverse_costs(node_ids, edges):
    reverse = defaultdict(list)
    for edge in edges:
        if edge.get('status') != 'refuted':
            reverse[edge['target']].append((edge['source'], float(edge['cost'])))
    distances = {key: math.inf for key in node_ids}
    distances[SINK] = 0.
    heap = [(0., SINK)]
    while heap:
        value, current = heapq.heappop(heap)
        if value != distances[current]:
            continue
        for parent, step in reverse[current]:
            candidate = value + step
            if candidate < distances.get(parent, math.inf):
                distances[parent] = candidate
                heapq.heappush(heap, (candidate, parent))
    return {key: (value if math.isfinite(value) else None) for key, value in distances.items() if key != SINK}


def _observed_edges(rows, config):
    by_id = {row['node_id']: row for row in rows}
    edges, excluded = [], []
    candidates = list(config.get('observed_edges', []))
    for row in rows:
        if row.get('previous_node_id'):
            candidates.append({'source': row['previous_node_id'], 'target': row['node_id'],
                'delta_a_mm': row.get('executed_delta_a_mm'),
                'receipt': row.get('execution_receipt'), 'evidence': row.get('execution_evidence', [])})
    seen = set()
    for item in candidates:
        a, b = by_id.get(item.get('source')), by_id.get(item.get('target'))
        reason = None
        if a is None or b is None:
            reason = 'missing_pre_or_post_observation'
        elif a.get('session') != b.get('session') or a.get('trial_id') != b.get('trial_id'):
            reason = 'different_trial'
        elif a.get('object_id') != b.get('object_id') or a.get('protocol_id') != b.get('protocol_id'):
            reason = 'different_object_or_protocol'
        elif not item.get('receipt'):
            reason = 'no_execution_receipt'
        elif not finite(item.get('delta_a_mm')):
            reason = 'missing_executed_delta'
        elif 'ood_debug_plus5' in str(a.get('session', '')) or 'ood_debug_plus5' in str(b.get('session', '')):
            reason = 'post_touch_not_recorded_in_debug_protocol'
        elif int(b.get('window_index', 0)) <= int(a.get('window_index', 0)):
            reason = 'nonincreasing_window'
        if reason:
            excluded.append({'candidate': item, 'reason': reason})
            continue
        if (a['node_id'], b['node_id']) in seen:
            continue
        seen.add((a['node_id'], b['node_id']))
        cost, risk = _base_cost(b, config)
        edges.append(_edge(a['node_id'], b['node_id'], 'observed', cost, 'confirmed',
            delta_a_mm=float(item['delta_a_mm']), receipt=item['receipt'], evidence=item.get('evidence', []), **risk))
    return edges, excluded


def build_graph(rows, noise_model, config=None):
    """Build guide sections 4.1–4.3. R=None means +infinity, never NaN JSON.

    Post-freeze independent rows are not graph-development evidence. Historical
    B joins development; held-out B folds use ``validation_rows``. Their
    its outcomes cannot seed sinks or training labels in this graph.
    """
    config = dict(config or {})
    rows = [dict(row) for row in rows]
    if len({r['node_id'] for r in rows}) != len(rows):
        raise ValueError('node_id must be unique')
    invalid = [r['node_id'] for r in rows if r.get('bilateral_contact') is not True]
    if invalid:
        raise ValueError('graph nodes require bilateral contact: ' + ', '.join(invalid[:5]))
    from .provenance import frozen_independent
    sealed = [r['node_id'] for r in rows if r.get('evaluation_only') is True or frozen_independent(r) or r.get('evaluation_role') in ('independent', 'final', 'sealed')]
    if sealed and not config.get('include_evaluation_nodes'):
        raise ValueError('Independent rows cannot generate graph training labels')
    ids = [row['node_id'] for row in rows]
    distance = pairwise_distances(rows, noise_model, config)
    k = int(config.get('knn', 10))
    neighbors = []
    for i in range(len(rows)):
        order = sorted((j for j in range(len(rows)) if j != i and math.isfinite(distance[i, j])),
                       key=lambda j: (distance[i, j], ids[j]))
        neighbors.append(set(order[:k]))
    edges, conflicts = [], []
    for i in range(len(rows)):
        for j in range(i):
            if distance[i, j] <= 1.:
                a, b = rows[i], rows[j]
                if isinstance(a.get('downstream_success'), bool) and isinstance(b.get('downstream_success'), bool) and a['downstream_success'] != b['downstream_success']:
                    attributes = [field for field in ('closure_floor', 'protocol_id', 'session', 'sensor_version', 'time_period') if a.get(field) != b.get(field)]
                    conflicts.append({'nodes': [ids[j], ids[i]], 'distance': float(distance[i, j]),
                                      'attributes': attributes, 'outcomes': [b['downstream_success'], a['downstream_success']]})
                if i in neighbors[j] and j in neighbors[i]:
                    for source, target in ((ids[i], ids[j]), (ids[j], ids[i])):
                        edges.append(_edge(source, target, 'similar', 0., 'equivalent', distance=float(distance[i, j])))
    max_action = float(config.get('max_action_mm', 15.))
    kappa = float(config.get('kinematic_multiplier', 1.5))
    from .provenance import same_reference
    for a in rows:
        if a.get('phase') != 'A' or not finite(a.get('tcp_offset_a')):
            continue
        for b in rows:
            if a['node_id'] == b['node_id'] or b.get('phase') != 'A' or not finite(b.get('tcp_offset_a')):
                continue
            if not same_reference(a,b,config):
                continue
            delta = float(b['tcp_offset_a']) - float(a['tcp_offset_a'])
            if abs(delta) > max_action or abs(delta) < 1e-9:
                continue
            base, risk = _base_cost(b, config)
            edges.append(_edge(a['node_id'], b['node_id'], 'kinematic', base * kappa, 'unverified',
                               delta_a_mm=delta, base_cost=base, multiplier=kappa,
                               cross_session=a.get('session')!=b.get('session'), reference_ids=[a.get('T0_id'),b.get('T0_id')],
                               assumption='same_day_same_T0_same_object_same_protocol_a_translation', evidence=[a['node_id'], b['node_id']], **risk))
    observed, excluded = _observed_edges(rows, config)
    edges.extend(observed)
    validation_rows = list(config.get('validation_rows', []))
    external_observed, validation_excluded = _observed_edges(validation_rows, {}) if validation_rows else ([], [])
    validation_by_id = {r['node_id']: r for r in rows + validation_rows}
    validation_order = list(validation_by_id)
    validation_indexes = {ident: i for i, ident in enumerate(validation_order)}
    all_distances = pairwise_distances(list(validation_by_id.values()), noise_model, config) if (observed or external_observed) else None
    by_id = {r['node_id']: r for r in rows}
    delta_tolerance = float(config.get('delta_match_tolerance_mm', 1.5))
    refuted = []
    for edge in edges:
        if edge['kind'] != 'kinematic':
            continue
        a, b = by_id[edge['source']], by_id[edge['target']]
        evidence = []
        for real in observed + external_observed:
            if abs(real['delta_a_mm'] - edge['delta_a_mm']) > delta_tolerance:
                continue
            pre, post = validation_by_id[real['source']], validation_by_id[real['target']]
            start_distance = float(all_distances[validation_indexes[a['node_id']], validation_indexes[pre['node_id']]])
            if start_distance > 1.:
                continue
            arrival_distance = float(all_distances[validation_indexes[b['node_id']], validation_indexes[post['node_id']]])
            # An infinite distance may mean missing sensing/calibration rather
            # than a physically different arrival. Only an explicitly known
            # discrete mismatch can refute it without a finite distance.
            mismatches = _known_gate_mismatches(b, post, config)
            if not math.isfinite(arrival_distance):
                if not mismatches:
                    edge.setdefault('validation_unavailable', []).append({
                        'edge_id': real['edge_id'], 'source': real['source'], 'target': real['target'],
                        'reason': 'arrival_observation_or_gate_unknown', 'receipt': real['receipt']})
                    continue
            evidence.append({'edge_id': real['edge_id'], 'source': real['source'], 'target': real['target'],
                'start_distance': start_distance,
                'arrival_distance': arrival_distance if math.isfinite(arrival_distance) else None,
                'known_gate_mismatches': mismatches,
                'matches': bool(arrival_distance <= 1.), 'receipt': real['receipt']})
        if evidence:
            edge['validation'] = evidence
            if any(not e['matches'] for e in evidence):
                edge['status'] = 'refuted'
                refuted.append(dict(edge))
            else:
                edge['status'] = 'confirmed'
                edge['cost'] = edge['base_cost']
                edge['multiplier'] = 1.
    edges = [e for e in edges if e['status'] != 'refuted']
    for row in rows:
        if row.get('downstream_success') is True:
            edges.append(_edge(row['node_id'], SINK, 'sink', 0., 'observed_success', evidence=[row['node_id']]))
    R = reverse_costs(ids, edges)
    reachability=classify_reachability(rows,edges,R)
    zero_comparisons = []
    for i, a in enumerate(rows):
        if not a.get('zero_monitor'):
            continue
        for j in range(i):
            b = rows[j]
            if not b.get('zero_monitor') or any(a.get(f) != b.get(f) for f in ('session', 'object_id', 'protocol_id')):
                continue
            # Incomparable gates/missing features are reported, not called drift.
            d = float(distance[i, j])
            zero_comparisons.append({'nodes': [ids[j], ids[i]], 'session': a.get('session'),
                'object_id': a.get('object_id'), 'protocol_id': a.get('protocol_id'),
                'distance': d if math.isfinite(d) else None, 'measurable': math.isfinite(d)})
    nodes = [{k: row.get(k) for k in ('node_id', 'session', 'trial_id', 'window_index', 'phase',
              'object_id', 'protocol_id', 'downstream_success', 'tcp_offset_a', 'rule_id', 'group_id')} | {'R': R[row['node_id']], 'reachability':reachability[row['node_id']]}
             for row in rows]
    serial_config = {k: v for k, v in config.items() if k not in ('validation_rows', 'observed_edges')}
    noise_id = noise_model.get('model_id') or noise_model.get('noise_model_id') or hashlib.sha256(_key(noise_model).encode()).hexdigest()
    node_receipts = [{'node_id': row['node_id'], 'encoder_version': row.get('encoder_version'),
                     'time_period':row.get('time_period'),'sensor_version':row.get('sensor_version'),'T0_id':row.get('T0_id'),'protocol_id':row.get('protocol_id'),
                     'provenance': row.get('provenance'), 'feature_input_sha256': row.get('feature_input_sha256'),
                     'execution_receipt': row.get('execution_receipt'),
                     'wrench_receipts': row.get('wrench_receipts'), 'gripper_receipt': row.get('gripper_receipt'),
                     'tcp_receipt': row.get('tcp_receipt')} for row in rows + validation_rows]
    receipts_sha = hashlib.sha256(_key(node_receipts).encode()).hexdigest()
    encoder_versions = sorted({row['encoder_version'] for row in rows + validation_rows if row.get('encoder_version') is not None})
    signature = {'node_ids': ids, 'edges': edges, 'R': R, 'reachability':reachability, 'config': serial_config,
                 'noise_model_id': noise_id, 'input_receipts_sha256': receipts_sha,
                 'encoder_versions': encoder_versions}
    version = hashlib.sha256(_key(signature).encode()).hexdigest()[:20]
    return {'schema': 'symbolic_contact_graph_v2', 'graph_version': version, 'nodes': nodes,
        'edges': edges, 'refuted_edges': refuted, 'observed_exclusions': excluded,
        'validation_exclusions': validation_excluded, 'conflicts': conflicts,
        'R': R, 'reachability':reachability, 'R_infinity_encoding': None, 'sink': SINK, 'config': serial_config,
        'noise_model_id': noise_id, 'encoder_versions': encoder_versions,
        'input_receipts_sha256': receipts_sha,
        'zero_comparisons': zero_comparisons,
        'diagnostic_only': bool(config.get('ignore_protocol') or config.get('ignore_time_period') or config.get('ignore_unknown_edge_class') or sealed),
        'training_labels_allowed': not bool(config.get('ignore_protocol') or config.get('ignore_time_period') or config.get('ignore_unknown_edge_class') or sealed or validation_rows),
        'counts': {'nodes': len(nodes), 'edges': len(edges), 'success_sinks': sum(e['kind'] == 'sink' for e in edges),
                   'similar': sum(e['kind'] == 'similar' for e in edges),
                   'observed': len(observed), 'cross_session_kinematic':sum(e['kind']=='kinematic' and e.get('cross_session',False) for e in edges),
                   'unsupported':sum(x['status']=='unsupported' for x in reachability.values()),'unreachable':sum(x['status']=='unreachable' for x in reachability.values()), 'external_observed_validation': len(external_observed),
                   'kinematic_confirmed': sum(e['kind'] == 'kinematic' and e['status'] == 'confirmed' for e in edges),
                   'kinematic_unverified': sum(e['kind'] == 'kinematic' and e['status'] == 'unverified' for e in edges),
                   'kinematic_refuted': len(refuted), 'conflicts': len(conflicts)}}


distance = measurement_distance


def classify_reachability(rows,edges,R):
    by_id={r['node_id']:r for r in rows};result={}
    for r in rows:
        ident=r['node_id'];value=R[ident]
        if value is not None:
            result[ident]={'status':'handoff_ready' if value==0 else 'recoverable','evidence':[]};continue
        attempts=[e for e in edges if e['kind']=='observed' and (e['source']==ident or e['target']==ident)
                  and by_id[e['target']].get('downstream_success') is False and R[e['target']] is None]
        result[ident]={'status':'unreachable' if attempts else 'unsupported',
            'reason':'observed_correction_ended_in_failure_without_graph_path' if attempts else 'no_observed_failed_recovery_establishes_unreachability',
            'evidence':[{'edge_id':e['edge_id'],'source':e['source'],'target':e['target'],'receipt':e['receipt'],
                'trial_group':by_id[e['source']].get('group_id',e['source'])} for e in attempts],
            'outgoing_edges':sum(e['source']==ident for e in edges), 'scope':'current_observed_graph_only'}
    return result
