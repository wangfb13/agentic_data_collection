"""Offline R/TCP clustering and observation-only deployment grounding."""
from __future__ import annotations

from collections import defaultdict
import hashlib
import math
import numpy as np

from .graph import GATE_FIELDS, cost_value, finite, measurement_distance
from .labels import generate_labels

OBSERVATION_FIELDS = GATE_FIELDS + ('bilateral_contact', 'z_tac_left', 'z_tac_right',
    'sensor_contact_left', 'sensor_contact_right', 'sensor_contact_mean', 'sensor_contact_diff',
    'wrench_left', 'wrench_right', 'gripper_opening', 'z_vis', 'legacy_features')


def observation_only(row):
    """Explicit allowlist: no TCP, planned actions, R, outcomes or task labels."""
    return {key: row.get(key) for key in OBSERVATION_FIELDS}


def _components(ids, edges):
    parent = {x: x for x in ids}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for e in edges:
        if e['kind'] == 'similar':
            a, b = find(e['source']), find(e['target'])
            parent[max(a, b)] = min(a, b)
    groups = defaultdict(list)
    for x in ids:
        groups[find(x)].append(x)
    return [sorted(v) for k, v in sorted(groups.items())]


def cluster_nodes(graph, rows, noise_model, config=None):
    config = config or {}
    by_id = {r['node_id']: r for r in rows}
    labels = {x['node_id']: x for x in generate_labels(graph, rows, config)} if not graph.get('diagnostic_only') and graph.get('training_labels_allowed') is not False else {}
    groups = _components(list(by_id), graph['edges'])
    conflict_ids = {n for c in graph.get('conflicts', []) for n in c['nodes']}
    clusters, unassigned, methods = [], [], []
    for component in groups:
        usable = [i for i in component if finite(by_id[i].get('tcp_offset_a'))]
        unassigned.extend({'node_id': i, 'reason': 'missing_measured_tcp_for_offline_clustering'} for i in component if i not in usable)
        if len(usable) < 3:
            unassigned.extend({'node_id': i, 'reason': 'component_below_min_cluster_size'} for i in usable)
            continue
        from hdbscan import HDBSCAN
        values = [cost_value(graph['R'][i]) for i in usable]
        finite_r = [x for x in values if math.isfinite(x)]
        sentinel = max(finite_r, default=0.) + 1.
        X = np.asarray([[r if math.isfinite(r) else sentinel, float(by_id[i]['tcp_offset_a'])] for i, r in zip(usable, values)])
        scale = X.std(axis=0); scale[scale < 1e-12] = 1.
        X = (X - X.mean(axis=0)) / scale
        # min_samples=1 allows the requested min_cluster_size=3 to work for
        # exactly three observations; it is an explicit recorded parameter.
        assignments = HDBSCAN(min_cluster_size=3, min_samples=1, allow_single_cluster=True).fit_predict(X)
        methods.append({'component': component, 'features': ['R', 'tcp_offset_a'],
            'R_infinity_sentinel': sentinel, 'standardization_scale': scale.tolist(),
            'min_cluster_size': 3, 'min_samples': 1, 'allow_single_cluster': True})
        for label in sorted(set(int(x) for x in assignments)):
            members = [i for i, k in zip(usable, assignments) if k == label]
            if label < 0:
                unassigned.extend({'node_id': i, 'reason': 'hdbscan_noise'} for i in members)
                continue
            rs = [cost_value(graph['R'][i]) for i in members]
            directions = {int(np.sign(labels[i]['delta_a_target_mm'])) for i in members
                if i in labels and finite(labels[i].get('delta_a_target_mm'))}
            kind, args = 'Ambiguous', {}
            if all(r == 0 for r in rs):
                kind = 'HandoffReady'
            elif all(math.isinf(r) for r in rs):
                kind, args = ('Unreachable' if all(graph.get('reachability',{}).get(i,{}).get('status')=='unreachable' for i in members) else 'Unsupported'), {'axis': 'a', 'scope': 'current_tested_graph_only'}
            elif all(0 < r < math.inf for r in rs) and len(directions) == 1 and 0 not in directions:
                kind, args = 'NeedShift', {'axis': 'a', 'sign': next(iter(directions))}
            # Equal-cost propagation must not conceal observed outcome conflicts.
            if any(i in conflict_ids for i in members):
                kind, args = 'Ambiguous', {'reason': 'indistinguishable_opposite_recorded_outcomes'}
            ident = 'c_' + hashlib.sha256('|'.join(members).encode()).hexdigest()[:14]
            first = by_id[members[0]]
            clusters.append({'cluster_id': ident, 'node_ids': members, 'symbol': kind,
                'args': {'object': first.get('object_id'), 'protocol': first.get('protocol_id'), 'time_period':first.get('time_period'), **args},
                'R': [graph['R'][i] for i in members], 'evidence': members,
                'observations': [{'node_id': i, 'observation': observation_only(by_id[i])} for i in members]})
    return {'schema': 'symbolic_clusters_v1', 'graph_version': graph['graph_version'],
        'clusters': clusters, 'unassigned': unassigned, 'method_details': methods,
        'diagnostic_only': graph.get('diagnostic_only', False),
        'deployment_distance_fields': list(OBSERVATION_FIELDS),
        'cluster_distance': 'minimum_gated_measurement_distance_to_observed_member'}


def ground_symbol(observation, clusters, noise_model, config=None):
    """Match raw deployment observations; never accept labels as state features."""
    if clusters.get('diagnostic_only'):
        raise ValueError('Diagnostic clusters cannot be used for deployment')
    query = observation_only(observation)
    matches = []
    for cluster in clusters['clusters']:
        distances = [(measurement_distance(query, item['observation'], noise_model, config), item['node_id']) for item in cluster['observations']]
        value, nearest = min(distances, default=(math.inf, None))
        if value <= 1.:
            matches.append({'cluster_id': cluster['cluster_id'], 'symbol': cluster['symbol'],
                            'args': cluster['args'], 'distance': value, 'nearest_node_id': nearest})
    signatures = {(m['symbol'], str(sorted(m['args'].items()))) for m in matches}
    if not matches or len(signatures) != 1 or any(m['symbol'] == 'Ambiguous' for m in matches):
        return {'symbol': 'Unknown', 'reason': 'no_supported_cluster' if not matches else 'conflicting_clusters',
                'matches': matches, 'graph_version': clusters['graph_version']}
    best = min(matches, key=lambda x: (x['distance'], x['cluster_id']))
    return {**best, 'graph_version': clusters['graph_version'], 'matches': matches,
            'R': 0 if best['symbol'] == 'HandoffReady' else None}
