from copy import deepcopy
from unittest.mock import patch

import numpy as np
import pytest

from skills.contact_position_a.legacy_v2.graph import build_graph
from skills.contact_position_a.legacy_v2.noise import fit_noise
from skills.contact_position_a.legacy_v2.skill import (
    partition_rows, partition_graphs, train_graph_skill, feature_vector, coverage_facts,
    SymbolicRecoverySkill, FeatureRecoveryAdapter, model_execution_report, make_contact_backend_runtime,
)


ACCEPTANCE = {key: {'status': 'pass'} for key in ('V1', 'V2', 'V4')}


def samples():
    rows = []
    # Three entirely separate groups of observed successes/failures. The
    # successful a=+2mm target deliberately differs from the old T0 origin.
    for part, n in [('train', 30), ('calibration', 10), ('validation', 10)]:
        for label in (False, True):
            for j in range(n):
                ident = f'{part}_{label}_{j}'
                rows.append(dict(node_id=ident, session='s', trial_id=ident, group_id=ident,
                    split=part, phase='A', window_index=0, t_ns=j,
                    object_id='white_light', protocol_id='p', bilateral_contact=True,
                    gripper_state='closing', edge_class=['none', 'none'], force_bin='probe',
                    encoder_version='frozen', gripper_opening=(.5 if label else .3) + .0005*(j%3-1),
                    tcp_offset_a=2. if label else -5., planned_delta_mm=2. if label else -5.,
                    downstream_success=label, normal_forces=[1., 1.]))
    return rows


def config():
    return {'modalities': ['open'], 'state_learning': {'epochs': 250},
            'contact_policy': {'epochs': 30, 'ensemble_size': 3}}


def source(rows):
    noise = fit_noise(rows, {'modalities': ['open']})
    return build_graph(rows, noise, {'modalities': ['open']}), noise


@pytest.fixture(scope='module')
def trained():
    rows = samples(); graph, noise = source(rows)
    result = train_graph_skill(rows, graph, noise, {}, ACCEPTANCE, config())
    assert result['training_status']['status'] == 'trained', result['training_status']
    return result, rows


def test_acceptance_blocks_model_and_saves_evidence(tmp_path):
    rows = samples(); graph, noise = source(rows)
    with patch('skills.contact_position_a.legacy_v2.skill._network') as fit:
        result = train_graph_skill(rows, graph, noise, {}, {'V1': {'status': 'fail'}}, config(), out=tmp_path)
    fit.assert_not_called()
    assert result['model'] is None
    assert 'offline_acceptance_V1_not_passed' in result['training_status']['blocking_reasons']
    assert (tmp_path/'training_status.json').exists()
    assert not (tmp_path/'symbolic_skill.json').exists()


def test_whole_group_split_cannot_cross_boundaries():
    rows = samples(); rows[-1]['group_id'] = rows[0]['group_id']
    with pytest.raises(ValueError, match='physical group'):
        partition_rows(rows)


def test_held_out_success_sink_cannot_make_training_nodes_recoverable():
    rows = samples()
    for row in rows:
        if row['split'] == 'train':
            row['downstream_success'] = False
    _, partitions, noise, graphs, labels = partition_graphs(rows, config())
    assert all(value is None for value in graphs['train']['R'].values())
    assert any(value == 0 for value in graphs['validation']['R'].values())
    assert all(x['invocation_label'] is None for x in labels if x['split'] == 'train')
    assert set(noise['fit_node_ids']) == {x['node_id'] for x in partitions['train']}
    for part, graph in graphs.items():
        assert all(e['source'].startswith(part) and (e['target'] == '__success__' or e['target'].startswith(part)) for e in graph['edges'])


def test_coverage_assigns_unsplit_rows_and_records_partition_label_provenance():
    rows = samples()
    for row in rows:
        del row['split']
    report = coverage_facts(rows, config())
    repeated = coverage_facts(rows, config())
    assert report['split_groups'] == repeated['split_groups']
    assert report['partition_graph_versions'] == repeated['partition_graph_versions']
    facts = {f['args'][0]: f for f in report['facts']}
    assert set(facts) == {'I', 'beta', 'pi'}
    assert all(f['value'] for f in facts.values())
    assert facts['pi']['metrics']['minimum'] == 20
    assert not any('split' in row for row in rows)
    train_ids = set(report['partition_evidence']['train'])
    assert set(report['noise_fit_node_ids']) == train_ids
    assert set().union(*map(set, report['partition_evidence'].values())) == {r['node_id'] for r in rows}
    assert not train_ids.intersection(report['partition_evidence']['validation'])


def test_coverage_cannot_borrow_held_out_success_labels():
    rows = samples()
    for row in rows:
        if row['split'] == 'train':
            row['downstream_success'] = False
    report = coverage_facts(rows, config())
    facts = {f['args'][0]: f for f in report['facts']}
    assert facts['I']['value'] is False and facts['beta']['value'] is False
    assert facts['I']['metrics']['counts']['train'] == {'0': 0, '1': 0}
    assert facts['beta']['metrics']['counts']['train']['1'] == 0
    assert all(label['invocation_label'] is None for label in report['partition_labels'] if label['split'] == 'train')


def test_joint_training_uses_graph_target_and_train_only_features(trained):
    result, rows = trained; model = result['model']
    assert model['three_components_updated_together']
    assert model['heads']['pi']['validation']['mae_mm'] < .05
    assert {s['target'] for s in model['heads']['pi']['sample_targets']} == {7.}
    assert not model['scheduler_default_enabled'] and not model['certified_for_hardware']
    row = rows[0]; bad = {**row, 'tcp_offset_a': 999., 'R': 999., 'planned_delta_mm': 999., 'downstream_success': True}
    assert np.array_equal(feature_vector(row, model['noise_model'], ['open']), feature_vector(bad, model['noise_model'], ['open']))
    assert all(n.startswith('train') for n in model['noise_model']['fit_node_ids'])
    assert all(n.startswith('train') for c in model['clusters']['clusters'] for n in c['node_ids'])


class FakeAdapter:
    def __init__(self, initial, following, *, stale=False):
        self.initial, self.following = deepcopy(initial), deepcopy(following)
        self.stale = stale; self.calls = []; self.received = []

    def initial_probe(self):
        self.calls.append('initial_probe')
        return {**self.initial, 'timestamp_ns': 10}

    def execute(self, action):
        self.calls.append('execute')
        assert action['reobserve_required'] is True
        self.action = action
        return {'applied': True, 'action_mm': action['delta_mm'], 'completed_ns': 20 + 20*len(self.received)}

    def observe(self):
        self.calls.append('observe'); self.received.append(True)
        return {**self.following, 'timestamp_ns': 20 if self.stale else 10 + 20*len(self.received)}

    def handoff(self, result):
        self.calls.append('handoff'); self.result = result


def test_fresh_post_action_handoff_and_one_correction(trained):
    result, rows = trained; skill = SymbolicRecoverySkill(result['model'])
    bad = rows[0]; good = next(r for r in rows if r['downstream_success'])
    adapter = FakeAdapter(bad, good)
    outcome = skill.run(adapter)
    assert adapter.calls == ['initial_probe', 'execute', 'observe', 'handoff']
    assert outcome['completed'] and outcome['corrections'] == 1
    assert abs(adapter.action['delta_mm'] - 7.) < .05
    assert outcome['handoff']['symbol'] == 'HandoffReady'
    assert outcome['handoff']['R'] == 0 and outcome['handoff']['latest_contact']['timestamp_ns'] == 30
    assert outcome['handoff']['graph_version'] == result['model']['graph_version']
    assert 'tcp_offset_a' not in outcome['handoff']['latest_contact']


def test_stale_post_action_does_not_handoff_success(trained):
    result, rows = trained
    adapter = FakeAdapter(rows[0], next(r for r in rows if r['downstream_success']), stale=True)
    out = SymbolicRecoverySkill(result['model']).run(adapter)
    assert not out['completed'] and out['reason'] == 'post_action_observation_not_fresh'


def test_unknown_never_sends_action_and_budget_is_three(trained):
    result, rows = trained; model = result['model']
    adapter = FakeAdapter({**rows[0], 'object_id': 'unseen'}, rows[0])
    out = SymbolicRecoverySkill(model).run(adapter)
    assert out['reason'] == 'unsupported_symbol' and 'execute' not in adapter.calls
    adapter = FakeAdapter(rows[0], rows[0])
    out = SymbolicRecoverySkill(model).run(adapter)
    assert out['reason'] == 'correction_budget_exhausted' and out['corrections'] == 3
    assert adapter.calls.count('execute') == 3 and adapter.calls.count('observe') == 3


def test_current_graph_unreachable_never_sends_action(trained):
    result, rows = trained; model = deepcopy(result['model'])
    for cluster in model['clusters']['clusters']:
        if cluster['symbol'] == 'NeedShift':
            cluster['symbol'] = 'Unrecoverable'
            cluster['args'] = {'object': 'white_light', 'protocol': 'p', 'axis': 'a', 'scope': 'current_graph_only'}
    adapter = FakeAdapter(rows[0], rows[0])
    out = SymbolicRecoverySkill(model).run(adapter)
    assert out['reason'] == 'unreachable_in_current_graph' and not out['completed']
    assert 'execute' not in adapter.calls


def test_insufficient_graph_labels_do_not_create_candidate_weights():
    rows = samples()
    for row in rows:
        row['downstream_success'] = True
    graph, noise = source(rows)
    with patch('skills.contact_position_a.legacy_v2.skill._network') as fit:
        result = train_graph_skill(rows, graph, noise, {}, ACCEPTANCE, config())
    fit.assert_not_called()
    assert result['model'] is None
    assert 'graph_label_coverage_insufficient' in result['training_status']['blocking_reasons']


def test_feature_wrapper_preserves_original_adapter_receipt_and_timestamp():
    rows = samples(); adapter = FakeAdapter(rows[0], rows[-1])
    wrapper = FeatureRecoveryAdapter(adapter, lambda x: {**x, 'R': 0, 'timestamp_ns': 999999})
    observed = wrapper.initial_probe()
    assert observed['timestamp_ns'] == 10 and 'R' not in observed
    receipt = wrapper.execute({'delta_mm': 7., 'reobserve_required': True})
    assert receipt == {'applied': True, 'action_mm': 7., 'completed_ns': 20}


def test_model_report_means_executable_candidate_not_B_pass(trained):
    result, _ = trained
    report = model_execution_report(result)
    assert report['usable_for_B'] is True and report['model_id'] == result['model']['model_id']
    assert report['B_validated'] is False and report['independent_B_validated'] is False
    assert not report['scheduler_default_enabled']
    blocked = model_execution_report({'training_status': {'status': 'not_trained', 'blocking_reasons': ['V1']}, 'model': None})
    assert blocked['usable_for_B'] is False and blocked['model_id'] is None
    assert blocked['blocking_reasons'] == ['V1']


class VirtualHardware:
    def __init__(self):
        self.T = np.eye(4); self.clock = 0

    def pose(self):
        return self.T.copy()


class BackendRunner:
    def __init__(self, backend, bad, good):
        self.backend, self.bad, self.good = backend, bad, good
        self.calls = []

    def regulate(self, mode, target, tolerance, stable_s, *, with_dd):
        self.calls.append(('regulate', mode, with_dd))
        self.backend.hw.clock += 10
        return {'timestamp_ns': self.backend.hw.clock, 'window_payload': self.bad, 'valid': True}

    def touch(self):
        self.calls.append(('touch',))
        self.backend.hw.clock += 10
        return 'contact', {'timestamp_ns': self.backend.hw.clock, 'window_payload': self.good, 'valid': True}


class BackendFacade:
    """Exact active ContactRecoveryBackend method signatures; no live hardware."""
    def __init__(self, single_open):
        self.executing = True; self.single_correction_open_handoff = single_open
        self.recovery = object(); self.hw = VirtualHardware()
        self.cfg = {'force': {'target': 5., 'probe_fraction': .2, 'probe_tolerance': .2, 'probe_stable_s': .2}}
        self.calls = []

    def _observation(self, status, probe):
        self.calls.append('_observation')
        return {'bilateral': True, 'force_stable': probe['valid'], 'timestamp_ns': probe['timestamp_ns']}

    def _recover_once(self, runner, target, action, row):
        self.calls.append('_recover_once')
        assert action['interface'] == 'open_retreat_translate_reapproach_probe'
        proposed = target.copy(); proposed[0, 3] += action['delta_mm']
        before = self.hw.pose(); self.hw.T = proposed.copy(); self.hw.clock += 10
        row['receipt'] = {'applied': True, 'action_mm': float(self.hw.T[0, 3] - before[0, 3]),
                          'completed_ns': self.hw.clock}
        if self.single_correction_open_handoff:
            return proposed, 'open_after_correction', None, row
        status, probe = runner.touch()
        return proposed, status, probe, row


@pytest.mark.parametrize('single_open', [False, True])
def test_opt_in_existing_backend_facade_reuses_execution_and_fresh_probe(trained, single_open):
    result, rows = trained
    bad, good = rows[0], next(r for r in rows if r['downstream_success'])
    backend = BackendFacade(single_open); original = backend.recovery
    runner = BackendRunner(backend, bad, good); handoffs = []
    def encode_probe(probe, observed):
        assert probe['timestamp_ns'] == observed['timestamp_ns']
        return {**probe['window_payload'], 'timestamp_ns': -999, 'R': 0, 'tcp_offset_a': 999}
    skill, adapter = make_contact_backend_runtime(result['model'], backend, runner, np.eye(4), encode_probe,
                                                  caller_paused=True, on_handoff=handoffs.append)
    assert not backend.calls and not runner.calls
    out = skill.run(adapter)
    assert out['completed'] and out['corrections'] == 1
    assert backend.calls.count('_recover_once') == 1
    assert runner.calls == [('regulate', 'PROBE', True), ('touch',)]
    assert out['handoff']['latest_contact']['timestamp_ns'] == 30
    assert 'tcp_offset_a' not in out['handoff']['latest_contact']
    assert backend.recovery is original and backend.single_correction_open_handoff is single_open
    assert backend.last_probe['timestamp_ns'] == 30 and handoffs[0]['completed']


def test_backend_factory_requires_existing_exclusive_invocation(trained):
    result, rows = trained; backend = BackendFacade(False); backend.executing = False
    runner = BackendRunner(backend, rows[0], rows[-1])
    with pytest.raises(ValueError, match='exclusive paused invocation'):
        make_contact_backend_runtime(result['model'], backend, runner, np.eye(4), lambda p, o: {}, caller_paused=True)
    assert not backend.calls and not runner.calls
