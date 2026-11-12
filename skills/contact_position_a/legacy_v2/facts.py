"""Evidence-backed symbolic facts calculated from graphs, never from LLM text."""
from __future__ import annotations

from collections import defaultdict, Counter
from itertools import combinations
import hashlib
import json
import math
import numpy as np

from .graph import cost_value, finite


def _fact(pred, args, evidence, metrics=None, value=True):
    evidence = sorted(set(evidence))
    stable = json.dumps([pred, args, evidence], ensure_ascii=False, sort_keys=True)
    return {'id': 'F_' + hashlib.sha256(stable.encode()).hexdigest()[:14],
            'pred': pred, 'args': args, 'evidence': evidence, 'metrics': metrics or {}, 'value': bool(value)}


def _split(row):
    s = row.get('split', row.get('training_split'))
    return {'fit': 'train', 'cal': 'calibration', 'calibrate': 'calibration',
            'val': 'validation', 'development': 'validation', 'dev': 'validation'}.get(s, s)


def _group(row):
    return str(row.get('group_id') or f"{row.get('session')}|{row.get('trial_id')}")


def generate_facts(graph, rows, labels, clusters, config=None):
    config = config or {}
    rows = list(rows)
    by_id = {r['node_id']: r for r in rows}
    by_label = {r['node_id']: r for r in labels}
    R = graph['R']
    facts = []
    rules = defaultdict(list)
    for row in rows:
        rules[str(row.get('rule_id') or row.get('session'))].append(row)
    for rule, members in sorted(rules.items()):
        if members and all(cost_value(R[r['node_id']]) == 0 for r in members):
            facts.append(_fact('no_failure', [rule], [r['node_id'] for r in members],
                               {'nodes': len(members), 'criterion': 'all_graph_costs_zero'}))
    failure_groups = defaultdict(list)
    for row in rows:
        x = row.get('tcp_offset_a')
        if finite(x) and (cost_value(R[row['node_id']]) > 0 or row.get('downstream_success') is False):
            side = 'negative' if x < 0 else ('positive' if x > 0 else 'zero')
            failure_groups[(side, row.get('object_id'))].append(row['node_id'])
    for (side, obj), members in sorted(failure_groups.items(), key=str):
        facts.append(_fact('failure_present', ['a', side, obj], members, {'count': len(members)}))

    partitions = defaultdict(list)
    for row in rows:
        if finite(row.get('tcp_offset_a')):
            partitions[(row.get('session'), row.get('object_id'), row.get('protocol_id'))].append(row)
    nominal = config.get('nominal_step_mm', {'a': 2.5})
    step = float(nominal.get('a', 2.5) if isinstance(nominal, dict) else nominal)
    for (session, obj, proto), members in sorted(partitions.items(), key=str):
        bins = defaultdict(list)
        for row in members:
            bins[float(row['tcp_offset_a'])].append(row)
        positions = sorted(bins)
        for x1, x2 in zip(positions, positions[1:]):
            if x2 - x1 <= 2 * step:
                continue
            def statuses(at):
                return {('failure' if r.get('downstream_success') is False or cost_value(R[r['node_id']]) > 0 else 'success') for r in bins[at]}
            if ('success' in statuses(x1) and 'failure' in statuses(x2)) or ('failure' in statuses(x1) and 'success' in statuses(x2)):
                facts.append(_fact('boundary_gap', ['a', [x1, x2], obj],
                    [r['node_id'] for r in bins[x1] + bins[x2]],
                    {'gap_mm': x2 - x1, 'nominal_step_mm': step, 'session': session, 'protocol_id': proto}))

    success_intervals = defaultdict(list)
    for row in rows:
        if cost_value(R[row['node_id']]) == 0 and finite(row.get('tcp_offset_a')):
            success_intervals[(row.get('protocol_id'), row.get('object_id'))].append(row)
    protocols = sorted({key[0] for key in success_intervals}, key=str)
    noise = config.get('position_noise_mm')
    for proto in protocols:
        objects = sorted([obj for pr, obj in success_intervals if pr == proto], key=str)
        for obj1, obj2 in combinations(objects, 2):
            r1, r2 = success_intervals[(proto, obj1)], success_intervals[(proto, obj2)]
            x1, x2 = [float(x['tcp_offset_a']) for x in r1], [float(x['tcp_offset_a']) for x in r2]
            a, b, c, d = min(x1), max(x1), min(x2), max(x2)
            center_delta = (c + d - a - b) / 2
            union = max(b, d) - min(a, c)
            overlap = max(0., min(b, d) - max(a, c)) / union if union > 0 else 1.
            difference = (finite(noise) and abs(center_delta) > 2 * noise) or overlap < .5
            if difference:
                facts.append(_fact('object_split', [obj1, obj2, 'a'], [r['node_id'] for r in r1 + r2],
                    {'intervals_mm': [[a, b], [c, d]], 'center_difference_mm': center_delta,
                     'intersection_over_union': overlap, 'position_noise_mm': noise, 'protocol_id': proto}))

    for conflict in graph.get('conflicts', []):
        attributes = conflict.get('attributes', [])
        attrs = attributes or [None]
        for attribute in attrs:
            args = list(conflict['nodes']) + ([attribute] if attribute is not None else [])
            facts.append(_fact('conflict', args, conflict['nodes'],
                {'distance': conflict['distance'], 'attribute': attribute,
                 'outcomes': conflict.get('outcomes'), 'diagnostic_only': graph.get('diagnostic_only', False)}))
    for cluster in clusters.get('clusters', []):
        if cluster['symbol'] == 'Ambiguous':
            facts.append(_fact('conflict', [cluster['cluster_id']], cluster['node_ids'],
                {'cluster_id': cluster['cluster_id'], 'reason': cluster['args'].get('reason', 'mixed_R_or_target_direction')}))
    for row in rows:
        state=graph.get('reachability',{}).get(row['node_id'])
        if state is None:
            from .graph import classify_reachability
            state=classify_reachability(rows,graph['edges'],R)[row['node_id']]
        if state['status'] in ('unsupported','unreachable'):
            evidence=[row['node_id']]+[n for e in state['evidence'] for n in (e['source'],e['target'])]
            facts.append(_fact(state['status'],[row['node_id'],'a'],evidence,
                {'observed_failure':state['status']=='unreachable','trial_groups':sorted({e['trial_group'] for e in state['evidence']}),
                 'execution_evidence':state['evidence'],'scope':'current_tested_graph_not_physical_impossibility'}))

    kinematic = [e for e in graph['edges'] if e['kind'] == 'kinematic']
    pending = [e for e in kinematic if e['status'] == 'unverified']
    # Count edges lying on at least one shortest path from a recoverable state.
    path_edges = [e for e in graph['edges'] if e['kind'] in ('kinematic', 'observed')
        and finite(R.get(e['source'])) and finite(R.get(e['target']))
        and math.isclose(R[e['source']], e['cost'] + R[e['target']], rel_tol=1e-7, abs_tol=1e-7)]
    path_pending = [e for e in path_edges if e['kind'] == 'kinematic' and e['status'] == 'unverified']
    if pending:
        source_edges = defaultdict(list)
        for edge in path_pending:
            source_edges[edge['source']].append(edge)
        verification_candidates = []
        for source, selected in source_edges.items():
            row = by_id[source]
            offset = row.get('planned_delta_mm')
            verification_candidates.append({
                'source_node_id': source, 'source_group': _group(row),
                'object_id': row.get('object_id'), 'protocol_id': row.get('protocol_id'),
                # A recorded experiment setting is appropriate for planning;
                # it remains excluded from similarity and learned inputs.
                'planned_offset_mm': float(offset) if finite(offset) else None,
                'delta_a_mm': sorted({float(e['delta_a_mm']) for e in selected if finite(e.get('delta_a_mm'))}),
                'edge_ids': sorted(e['edge_id'] for e in selected),
                'target_node_ids': sorted({e['target'] for e in selected}),
                'coverage_count': len(selected)})
        verification_candidates.sort(key=lambda item: (-item['coverage_count'], item['source_node_id']))
        facts.append(_fact('edge_unverified', [len(pending)], [x for e in pending for x in (e['source'], e['target'])],
            {'count': len(pending), 'total_kinematic': len(kinematic),
             'path_fraction': len(path_pending) / len(path_edges) if path_edges else 0.,
             'shortest_path_edge_count': len(path_edges), 'unverified_path_edge_count': len(path_pending),
             'edge_ids': [e['edge_id'] for e in pending],
             'verification_candidates': verification_candidates,
             'verification_candidate_ranking': 'descending unverified shortest-path edge count per observed source node'}))
    for edge in graph.get('refuted_edges', []):
        ids = [edge['source'], edge['target']] + [x for v in edge.get('validation', []) for x in (v['source'], v['target'])]
        facts.append(_fact('edge_refuted', [edge['edge_id']], ids,
            {'delta_a_mm': edge.get('delta_a_mm'), 'validation': edge.get('validation', [])}))

    # No invented model disagreement: requires measured ensemble score + training reference.
    training_scores = config.get('training_energy_scores', [])
    scores = [float(v) for v in training_scores if finite(v)]
    if scores:
        threshold = float(np.quantile(scores, .9))
        for row in rows:
            score = row.get('energy_disagreement')
            if finite(score) and score > threshold and finite(row.get('tcp_offset_a')):
                facts.append(_fact('high_disagreement', ['a', float(row['tcp_offset_a'])], [row['node_id']],
                    {'energy_score': score, 'training_p90': threshold, 'training_n': len(scores)}))

    for head, key in [('I', 'invocation_label'), ('beta', 'exit_label')]:
        counts = {split: {str(k): set() for k in (0, 1)} for split in ('train', 'calibration', 'validation')}
        evidence = []
        for row in rows:
            label = by_label.get(row['node_id'], {})
            split = _split(row)
            value = label.get(key)
            if split in counts and value in (0, 1) and not label.get('conflicting_outcomes'):
                counts[split][str(value)].add(_group(row)); evidence.append(row['node_id'])
        numeric = {s: {k: len(v) for k, v in d.items()} for s, d in counts.items()}
        ok = all(numeric[s][str(k)] >= n for s, n in [('train', 4), ('calibration', 2), ('validation', 2)] for k in (0, 1))
        facts.append(_fact('coverage_ok', [head], evidence, {'counts': numeric, 'minimum': {'train': 4, 'calibration': 2, 'validation': 2}}, value=ok))
    regression = [r['node_id'] for r in rows if by_label.get(r['node_id'], {}).get('pi_eligible')
                  and not by_label[r['node_id']].get('conflicting_outcomes')]
    facts.append(_fact('coverage_ok', ['pi'], regression, {'qualified_samples': len(regression), 'minimum': 20}, value=len(regression) >= 20))
    zero = defaultdict(list)
    for pair in graph.get('zero_comparisons', []):
        if pair.get('measurable'):
            zero[pair['session']].append(pair)
    for session, pairs in zero.items():
        fraction = sum(p['distance'] > 1 for p in pairs) / len(pairs)
        if fraction > .2:
            facts.append(_fact('drift', [session], [i for pair in pairs for i in pair['nodes']],
                {'fraction_above_one': fraction, 'pairs': len(pairs), 'threshold': .2}))
    model = config.get('model_report')
    if isinstance(model, dict) and model.get('model_id') and model.get('usable_for_B') is True:
        facts.append(_fact('model_executable', [model['model_id']], model.get('evidence_node_ids', []),
            {'source_report': model.get('report_path'), 'source_hash': model.get('report_hash')}))
    for key, pred in [('dev_B_report', 'dev_B_pass'), ('independent_B_report', 'indep_B_pass')]:
        report = config.get(key)
        if isinstance(report, dict) and report.get('passed') is True and report.get('model_id'):
            facts.append(_fact(pred, [report['model_id']], report.get('evidence_node_ids', []),
                {'source_report': report.get('report_path'), 'source_hash': report.get('report_hash')}))
    for row in rows:
        if row.get('human_label') is not None and row.get('downstream_success') is not None:
            label = row['human_label']
            normal = label in ('normal', 'ok', 'acceptable', 'success', True)
            if normal != row['downstream_success']:
                facts.append(_fact('label_conflict', [row.get('trial_id')], [row['node_id']],
                    {'human_label': label, 'downstream_success': row['downstream_success']}))
    return sorted(facts, key=lambda f: (f['pred'], str(f['args']), f['id']))
