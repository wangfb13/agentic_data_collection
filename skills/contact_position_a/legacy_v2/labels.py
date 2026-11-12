"""Outcome-rooted I/pi/beta targets; no mutation of original labels."""
from __future__ import annotations

from collections import defaultdict
import heapq
import math
import numpy as np

from .graph import SINK, cost_value, finite


def _distances(start, edges):
    adj = defaultdict(list)
    for e in edges:
        if e['target'] != SINK and e.get('status') != 'refuted':
            adj[e['source']].append((e['target'], float(e['cost'])))
    distances = {start: 0.}
    heap = [(0., start)]
    while heap:
        value, node = heapq.heappop(heap)
        if value != distances[node]:
            continue
        for target, step in adj[node]:
            nxt = value + step
            if nxt < distances.get(target, math.inf):
                distances[target] = nxt
                heapq.heappush(heap, (nxt, target))
    return distances


def transition_weight(before, after, beta_w=0.5, gamma=1.):
    before, after = cost_value(before), cost_value(after)
    if not after < before:
        return 0.
    return float(np.clip(1. + beta_w * math.tanh(gamma * (before - after)), 0., 1.5))


def generate_labels(graph, rows, config=None):
    config = config or {}
    if graph.get('diagnostic_only') or graph.get('training_labels_allowed') is False:
        raise ValueError('Diagnostic or independent-validation graph cannot generate training labels')
    from .provenance import same_reference
    by_id = {r['node_id']: r for r in rows}
    R = graph['R']
    conflict_ids = {n for c in graph.get('conflicts', []) for n in c['nodes']}
    failures = [r for r in rows if r.get('downstream_success') is False and finite(r.get('tcp_offset_a'))]
    incoming = defaultdict(list)
    for e in graph['edges']:
        if e['kind'] == 'observed':
            incoming[e['target']].append({'edge_id': e['edge_id'], 'source': e['source'],
                'R_before': R[e['source']], 'R_after': R[e['target']],
                'weight': transition_weight(R[e['source']], R[e['target']],
                    float(config.get('beta_w', .5)), float(config.get('gamma', 1.)))})
    output = []
    for node in graph['nodes']:
        ident = node['node_id']; row = by_id[ident]; value = cost_value(R[ident])
        item = {'node_id': ident, 'graph_version': graph['graph_version'], 'R': R[ident],
            'invocation_label': 0 if value == 0 else (1 if math.isfinite(value) else None),
            'exit_label': int(value == 0), 'applicable': math.isfinite(value),
            'applicability': 'handoff_ready' if value == 0 else ('recoverable' if math.isfinite(value) else graph.get('reachability',{}).get(ident,{}).get('status','unsupported')),
            'target_node_id': ident if value == 0 else None, 'delta_a_target_mm': 0. if value == 0 else None,
            'target_failure_clearance_mm': None, 'incoming_transitions': incoming[ident],
            'source_group': row.get('group_id', f"{row.get('session')}|{row.get('trial_id')}"),
            'split': row.get('split', row.get('training_split')),
            'conflicting_outcomes': ident in conflict_ids,
            'evidence': [ident], 'pi_eligible': False}
        if 0. < value < math.inf and finite(row.get('tcp_offset_a')):
            distances = _distances(ident, graph['edges'])
            targets = [by_id[target] for target, dist in distances.items()
                if target in by_id and cost_value(R.get(target)) == 0 and math.isclose(dist, value, rel_tol=1e-7, abs_tol=1e-7)
                and by_id[target].get('object_id') == row.get('object_id')
                and by_id[target].get('protocol_id') == row.get('protocol_id')
                and same_reference(row,by_id[target],config)
                and finite(by_id[target].get('tcp_offset_a'))]
            matched_failures = [f for f in failures if f.get('object_id') == row.get('object_id') and f.get('protocol_id') == row.get('protocol_id') and same_reference(row,f,config)]
            def clearance(target):
                return min((abs(float(target['tcp_offset_a']) - float(f['tcp_offset_a'])) for f in matched_failures), default=None)
            if targets:
                targets.sort(key=lambda t: (-(clearance(t) if clearance(t) is not None else 0.),
                    abs(float(t['tcp_offset_a']) - float(row['tcp_offset_a'])), t['node_id']))
                target = targets[0]
                item.update(target_node_id=target['node_id'],
                    delta_a_target_mm=float(target['tcp_offset_a']) - float(row['tcp_offset_a']),
                    target_failure_clearance_mm=clearance(target), pi_eligible=True,
                    target_selection='shortest_cost_then_max_nearest_failure_clearance_then_min_motion',
                    evidence=[ident, target['node_id']] + [f['node_id'] for f in matched_failures])
        output.append(item)
    return output
