from copy import deepcopy
import math
from unittest.mock import patch

import pytest

from skills.contact_position_a.legacy_v2.graph import build_graph, gate, measurement_distance, reverse_costs, pairwise_distances
from skills.contact_position_a.legacy_v2.labels import generate_labels, transition_weight
from skills.contact_position_a.legacy_v2.clusters import cluster_nodes, ground_symbol, observation_only
from skills.contact_position_a.legacy_v2.facts import generate_facts


def row(name, x=0., success=None, **kwargs):
    result = dict(node_id=name, session='session', trial_id=name, window_index=0,
        phase='A', t_ns=0, object_id='white_light', protocol_id='p',
        bilateral_contact=True, gripper_state='closing', edge_class=['none', 'none'],
        force_bin='probe', encoder_version='test', z_tac_left=[float(x)], z_tac_right=[float(x)],
        sensor_contact_left={'u_mm': x, 'v_mm': 0.}, sensor_contact_right={'u_mm': x, 'v_mm': 0.},
        wrench_left=[0., 0., -1., 0., 0., 0.], wrench_right=[0., 0., -1., 0., 0., 0.],
        gripper_opening=.36, z_vis=[0.], tcp_offset_a=float(x), downstream_success=success,
        planned_delta_mm=x, group_id=name, rule_id='r1', closure_floor=.336,
        normal_forces=[1., 1.], zero_monitor=False)
    result.update(kwargs)
    return result


@pytest.fixture(autouse=True)
def scalar_measurement():
    # The separate noise-module tests cover real PCA/covariance behavior. Here
    # an observation-only metric isolates graph topology and leakage checks.
    with patch('skills.contact_position_a.legacy_v2.noise.noise_distance',
               side_effect=lambda a, b, model, modalities=None: abs(a['z_tac_left'][0] - b['z_tac_left'][0])):
        yield


def test_gate_and_deployment_distance_do_not_use_labels():
    a, b = row('a'), row('b', .25)
    expected = measurement_distance(a, b, {})
    b.update(tcp_offset_a=999., planned_delta_mm=-100., downstream_success=True, R=-99., object_contact_h=987.)
    assert measurement_distance(a, b, {}) == expected
    for field, value in [('object_id', 'heavy'), ('protocol_id', 'different'),
                         ('encoder_version', 'new'), ('gripper_state', 'near_floor'),
                         ('edge_class', ['root', 'tip']), ('force_bin', 'higher')]:
        c = deepcopy(b); c[field] = value
        assert not gate(a, c)
    c = deepcopy(b); c['protocol_id'] = 'other'
    assert gate(a, c, {'ignore_protocol': True})
    for key in ['R', 'tcp_offset_a', 'planned_delta_mm', 'downstream_success', 'object_contact_h']:
        assert key not in observation_only(b)


def test_reverse_dijkstra_and_graph_targets():
    rows = [row('start', -10., False), row('failure', -1., False), row('g0', 0., True), row('g5', 5., True)]
    g = build_graph(rows, {}, {'max_action_mm': 15., 'knn': 0})
    assert g['R']['start'] == 1.5 and g['R']['g0'] == 0
    labels = {x['node_id']: x for x in generate_labels(g, rows)}
    assert labels['start']['target_node_id'] == 'g5'
    assert labels['start']['delta_a_target_mm'] == 15.
    assert labels['start']['invocation_label'] == 1
    assert labels['g0']['exit_label'] == 1
    assert labels['g0']['invocation_label'] == 0
    assert all(e['cost'] == 1.5 for e in g['edges'] if e['kind'] == 'kinematic')


def test_zero_equivalence_does_not_erase_outcome_conflict():
    rows = [row('failed', 0., False), row('passed', .2, True)]
    g = build_graph(rows, {})
    assert g['R'] == {'failed': 0., 'passed': 0.}
    assert len(g['conflicts']) == 1
    labels = generate_labels(g, rows)
    assert all(x['conflicting_outcomes'] for x in labels)
    f = generate_facts(g, rows, labels, {'clusters': []})
    assert any(x['pred'] == 'conflict' and x['evidence'] == ['failed', 'passed'] for x in f)


def test_protocol_relaxation_is_diagnostic_only():
    rows = [row('old', 0., False), row('new', .1, True, protocol_id='new', closure_floor=.330)]
    g = build_graph(rows, {})
    assert not g['conflicts'] and g['R']['old'] is None
    diagnostic = build_graph(rows, {}, {'ignore_protocol': True})
    assert diagnostic['conflicts'][0]['attributes'] == ['closure_floor', 'protocol_id']
    with pytest.raises(ValueError):generate_labels(diagnostic, rows)
    with pytest.raises(ValueError):ground_symbol(rows[0], {'diagnostic_only': True}, {})


def test_no_cross_object_recovery_or_sealed_evidence():
    rows = [row('bad', 0., False), row('good', 1., True, object_id='white_heavy')]
    g = build_graph(rows, {})
    labels = {x['node_id']: x for x in generate_labels(g, rows)}
    assert labels['bad']['applicable'] is False
    assert labels['bad']['invocation_label'] is None
    assert labels['bad']['exit_label'] == 0
    with pytest.raises(ValueError):build_graph([row('sealed', evaluation_role='independent')], {})
    with pytest.raises(ValueError):build_graph([row('sealed', evaluation_only=True)], {})
    with pytest.raises(ValueError):build_graph([row('one', bilateral_contact=False)], {})


def test_observed_needs_receipt_and_post_window():
    pre = row('pre', -5., None, phase='B_pre', trial_id='b')
    post = row('post', 0., True, phase='B_post', trial_id='b', window_index=1,
               previous_node_id='pre', executed_delta_a_mm=5.)
    g = build_graph([pre, post], {})
    assert not [e for e in g['edges'] if e['kind'] == 'observed']
    post['execution_receipt'] = {'source': 'meta.json', 'index': 0, 'executed': True}
    g = build_graph([pre, post], {})
    assert len([e for e in g['edges'] if e['kind'] == 'observed']) == 1
    assert g['R']['pre'] == 1.
    assert transition_weight(g['R']['pre'], g['R']['post']) > 1.


def test_independent_transitions_validate_but_do_not_seed_labels():
    A = [row('a0', -5., False), row('a1', 0., True)]
    B = [row('b0', -5., None, session='b', phase='B_pre', trial_id='bt', evaluation_role='independent'),
         row('b1', 0., True, session='b', phase='B_post', trial_id='bt', window_index=1,
             evaluation_role='independent', previous_node_id='b0', executed_delta_a_mm=5., execution_receipt={'executed': True})]
    g = build_graph(A, {}, {'validation_rows': B})
    forward = next(e for e in g['edges'] if e['source'] == 'a0' and e['target'] == 'a1')
    assert forward['status'] == 'confirmed' and forward['cost'] == 1.
    assert set(g['R']) == {'a0', 'a1'}
    with pytest.raises(ValueError):generate_labels(g, A)
    B[1]['z_tac_left'] = [9.]
    g = build_graph(A, {}, {'validation_rows': B})
    assert any(e['source'] == 'a0' and e['target'] == 'a1' for e in g['refuted_edges'])
    assert g['R']['a0'] is None


def test_transition_weight_and_no_empty_success_claim():
    assert transition_weight(2, 2) == 0
    assert transition_weight(1, 2) == 0
    assert transition_weight(None, None) == 0
    assert transition_weight(None, 0) == 1.5
    g = build_graph([], {})
    facts = generate_facts(g, [], [], {'clusters': []})
    assert not any(f['pred'] == 'no_failure' for f in facts)
    assert all(not f['value'] for f in facts if f['pred'] == 'coverage_ok')


def test_actual_hdbscan_and_observation_only_grounding():
    pytest.importorskip('hdbscan')
    rows = [row(str(i), i * .01, True) for i in range(8)]
    g = build_graph(rows, {})
    clusters = cluster_nodes(g, rows, {})
    assert clusters['clusters']
    symbol = ground_symbol(row('live', .03), clusters, {})
    assert symbol['symbol'] == 'HandoffReady'
    changed = row('live', .03); changed.update(tcp_offset_a=-999, R=999, downstream_success=False, planned_delta_mm=5)
    assert ground_symbol(changed, clusters, {}) == symbol
    assert ground_symbol(row('other', 900.), clusters, {})['symbol'] == 'Unknown'


def test_fact_coverage_and_edge_fraction():
    rows = [row('bad', -5., False), row('good', 0., True)]
    g = build_graph(rows, {}); labels = generate_labels(g, rows)
    f = generate_facts(g, rows, labels, {'clusters': []})
    unverified = next(x for x in f if x['pred'] == 'edge_unverified')
    assert unverified['metrics']['path_fraction'] == 1.
    assert set(x['args'][0] for x in f if x['pred'] == 'coverage_ok') == {'I', 'beta', 'pi'}
    assert all(x['id'].startswith('F_') and isinstance(x['evidence'], list) for x in f)


def test_missing_observation_is_not_physical_refutation():
    A = [row('a0', -5., False), row('a1', 0., True)]
    B = [row('b0', -5., None, session='b', phase='B_pre', trial_id='bt'),
         row('b1', 0., True, session='b', phase='B_post', trial_id='bt', window_index=1,
             previous_node_id='b0', executed_delta_a_mm=5., execution_receipt={'executed': True})]
    with patch('skills.contact_position_a.legacy_v2.noise.noise_distance',
               side_effect=lambda a, b, model, modalities=None: math.inf if 'b1' in (a.get('node_id'), b.get('node_id')) else abs(a['z_tac_left'][0] - b['z_tac_left'][0])):
        g = build_graph(A, {}, {'validation_rows': B})
    assert not g['refuted_edges']


@pytest.mark.parametrize('missing', [{'edge_class': {'left': None, 'right': None}},
                                    {'encoder_version': None}, {'force_bin': None}])
def test_unknown_arrival_gate_remains_unverified(missing):
    A = [row('a0', -5., False), row('a1', 0., True)]
    B = [row('b0', -5., None, session='b', phase='B_pre', trial_id='bt'),
         row('b1', 0., True, session='b', phase='B_post', trial_id='bt', window_index=1,
             previous_node_id='b0', executed_delta_a_mm=5., execution_receipt={'executed': True}, **missing)]
    g = build_graph(A, {}, {'validation_rows': B})
    assert not g['refuted_edges']
    edge = next(e for e in g['edges'] if e['source'] == 'a0' and e['target'] == 'a1')
    if 'edge_class' in missing:
        assert edge['status']=='confirmed'
    else:
        assert edge['status'] == 'unverified' and edge['cost'] == 1.5
        assert edge['validation_unavailable'][0]['reason'] == 'arrival_observation_or_gate_unknown'


def test_known_arrival_gate_difference_can_refute():
    A = [row('a0', -5., False), row('a1', 0., True)]
    B = [row('b0', -5., None, session='b', phase='B_pre', trial_id='bt'),
         row('b1', 0., True, session='b', phase='B_post', trial_id='bt', window_index=1,
             previous_node_id='b0', executed_delta_a_mm=5., execution_receipt={'executed': True},
             edge_class=['root', 'none'])]
    g = build_graph(A, {}, {'validation_rows': B})
    edge = next(e for e in g['refuted_edges'] if e['source'] == 'a0' and e['target'] == 'a1')
    assert edge['validation'][0]['known_gate_mismatches'] == ['edge_class']


def test_verify_candidates_rank_sources_by_shortest_path_edge_coverage():
    rows = [row('far', -20., False), row('branch', -10., False, planned_delta_mm=-9.),
            row('ok0', 0., True), row('ok5', 5., True)]
    graph = build_graph(rows, {}, {'knn': 0})
    facts = generate_facts(graph, rows, generate_labels(graph, rows), {'clusters': []})
    metrics = next(f['metrics'] for f in facts if f['pred'] == 'edge_unverified')
    candidates = metrics['verification_candidates']
    assert candidates[0]['source_node_id'] == 'branch'
    assert candidates[0]['coverage_count'] == 2
    assert candidates[0]['delta_a_mm'] == [10., 15.]
    assert candidates[0]['planned_offset_mm'] == -9.
    assert candidates[0]['target_node_ids'] == ['ok0', 'ok5']
    assert len(candidates[0]['edge_ids']) == 2
    assert candidates[1]['source_node_id'] == 'far' and candidates[1]['coverage_count'] == 1
    assert sum(c['coverage_count'] for c in candidates) == metrics['unverified_path_edge_count']


def test_graph_version_records_noise_encoder_and_input_receipts_even_without_similarity():
    rows = [row('bad', -5., False), row('ok', 0., True)]
    first = build_graph(rows, {'model_id': 'noise1'}, {'knn': 0})
    second = build_graph(rows, {'model_id': 'noise2'}, {'knn': 0})
    assert first['edges'] == second['edges'] and first['R'] == second['R']
    assert first['graph_version'] != second['graph_version']
    assert second['noise_model_id'] == 'noise2'
    new = [{**r, 'encoder_version': 'updated_encoder'} for r in rows]
    third = build_graph(new, {'model_id': 'noise1'}, {'knn': 0})
    assert third['graph_version'] != first['graph_version']
    assert third['encoder_versions'] == ['updated_encoder']
    receipts = [{**r, 'provenance': {'metadata_sha256': 'changed'}} for r in rows]
    fourth = build_graph(receipts, {'model_id': 'noise1'}, {'knn': 0})
    assert fourth['input_receipts_sha256'] != first['input_receipts_sha256']
    assert fourth['graph_version'] != first['graph_version']


def test_vectorized_distance_matches_mahalanobis_and_unknown_gate():
    model = {'modalities': {'tac': {'status': 'ready', 'input_dimension': 2,
        'pca_mean': [0., 0.], 'pca_components': None, 'precision': [[1., 0.], [0., 1.]], 'tau': 2.}}}
    d = pairwise_distances([row('a', 0.), row('b', 2.)], model, {'modalities': ['tac']})
    assert d[0, 1] == pytest.approx(math.sqrt(2))
    a = row('a', edge_class={'left': None, 'right': None})
    assert gate(a, a)
    assert not gate(a, a, {'optional_edge_orientation':False})
