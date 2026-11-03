"""Offline regression checks for whole-cycle deduplication and staged evaluation."""
import copy
import fcntl
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from bottle_scan import rsi
from bottle_scan.rsi_methods import baseline_method, historical_coverage, select_cells
from bottle_scan.session import atomic_json
from tests.test_agentic_campaign import config
from tests.test_rsi import propose


class EfficientSamplingTests(unittest.TestCase):
    def test_session_validation_balance_keeps_large_seed_available_for_training(self):
        from bottle_scan.contact_policy import _settings, _split
        observations = [{'session': 'big', 'group': f'big:{i}'} for i in range(100)]
        observations += [{'session': f'small{j}', 'group': f'small{j}:{i}'} for j in range(4) for i in range(5)]
        _, cfg = _settings({'contact_policy': {'seed': 3, 'balance_session_validation': True}})
        train, valid, split = _split(observations, cfg)
        self.assertTrue(all(observations[i]['session'] != 'big' for i in valid))
        self.assertEqual(sum(observations[i]['session'] == 'big' for i in train), 100)
        self.assertGreaterEqual(len(valid), cfg['min_validation_samples'])
        self.assertEqual(split['method'], 'held_out_session')
        self.assertFalse({observations[i]['session'] for i in train} & {observations[i]['session'] for i in valid})

    def test_cycle_cap_spans_batches_and_fails_closed_when_full(self):
        method = copy.deepcopy(baseline_method())
        method.pop('method_id')
        method['sampling'].update(step_mm=5, max_reps_per_cell=4)
        scope = {'min_mm': -5, 'max_mm': 5}
        previous = {-5.: 4, 0.: 2}
        cells = select_cells(method, {}, scope=scope, trial_budget=4, prior_counts=previous,
                             efficiency={'max_reps_per_cell': 4}, seed=1)
        self.assertNotIn(-5., [c['offset_mm'] for c in cells])
        self.assertTrue(all(previous.get(c['offset_mm'], 0) + c['reps'] <= 4 for c in cells))
        with self.assertRaisesRegex(ValueError, 'cannot fill'):
            select_cells(method, {}, scope=scope, trial_budget=2,
                         prior_counts={-5.: 4, 0.: 4, 5.: 4}, efficiency={'max_reps_per_cell': 4})
        method['sampling'].update(step_mm=1, diversity_mm=2, max_reps_per_cell=2)
        cells = select_cells(method, {}, scope=scope, trial_budget=2, prior_counts={0.: 2},
                             efficiency={'max_reps_per_cell': 2}, seed=1)
        self.assertGreaterEqual(abs(cells[0]['offset_mm']), 2.)

    def test_repeat_exception_requires_explanation_and_remains_bounded(self):
        from bottle_scan.rsi_proposer import RSIProposalError, propose_method
        from tests.test_rsi_proposer import context, decision, FakeProvider
        ctx = context()
        ctx['efficiency_protocol'] = {'max_reps_per_cell':4,'max_override_reps_per_cell':8}
        value = decision(); value['method']['sampling']['max_reps_per_cell'] = 6
        with self.assertRaises(RSIProposalError) as error:
            propose_method({'backend':'chatgpt','max_proposal_attempts':1},ctx,provider=FakeProvider([value]))
        self.assertEqual(error.exception.audit['attempts'][0]['rejection']['code'],'repeat_justification')
        value['method']['repeat_justification'] = '该位置观测差异很大，需要重复判断是否有噪声。'
        accepted=propose_method({'backend':'chatgpt'},ctx,provider=FakeProvider([value]))
        self.assertIn('repeat_justification',accepted['method'])
        value['method']['sampling']['max_reps_per_cell'] = 10
        with self.assertRaises(RSIProposalError) as error:
            propose_method({'backend':'chatgpt','max_proposal_attempts':1},ctx,provider=FakeProvider([value]))
        self.assertEqual(error.exception.audit['attempts'][0]['rejection']['code'],'repeat_limit')
        method=copy.deepcopy(baseline_method());method.pop('method_id')
        method['sampling'].update(step_mm=5,max_reps_per_cell=6)
        previous={-5.:4,0.:4,5.:4}; scope={'min_mm':-5,'max_mm':5}
        options={'max_reps_per_cell':4,'max_override_reps_per_cell':8}
        with self.assertRaisesRegex(ValueError,'cannot fill'):
            select_cells(method,{},scope=scope,trial_budget=2,prior_counts=previous,efficiency=options)
        method['repeat_justification']='重复检查观测波动。'
        self.assertEqual(sum(x['reps'] for x in select_cells(method,{},scope=scope,trial_budget=2,
                             prior_counts=previous,efficiency=options)),2)

    def test_recorded_A_history_is_a_soft_bonus_without_evaluation_leakage(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            a, validation = root/'a', root/'evaluation'
            for i in range(8):
                folder = a/'trials'/str(i)
                atomic_json(folder/'meta.json', {'cond':'A','kind':'scan','complete':True,
                    'recording_complete':True,'label':'normal','bottle':'light' if i%2 else 'heavy','delta_mm':-5})
                atomic_json(folder/'recording_status.json', {'complete':True})
            atomic_json(validation/'trials/1/meta.json', {'cond':'B','kind':'scan','complete':True,
                'recording_complete':True,'label':'normal','bottle':'light','delta_mm':0})
            atomic_json(validation/'trials/1/recording_status.json', {'complete':True})
            report={'evidence': {'sessions':[{'path':str(a),'role':'exploration'},
                                            {'path':str(validation),'role':'validation'}]}}
            self.assertEqual(historical_coverage(report), [{'offset_mm':-5.,'bottles':{'heavy':4,'light':4},'total':8,'scan':8,'zero_monitor':0}])
            method=copy.deepcopy(baseline_method());method.pop('method_id')
            method['score']={'op':'constant','value':0}
            method['sampling'].update(step_mm=5,max_reps_per_cell=2)
            selected=select_cells(method,report,scope={'min_mm':-5,'max_mm':5},trial_budget=4,
                                  efficiency={'max_reps_per_cell':2,'coverage_weight':1.},seed=0)
            self.assertEqual({x['offset_mm'] for x in selected}, {0.,5.})

    def test_seed_campaign_imports_only_finished_A_and_requires_pause(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); seed=root/'seed'; campaign=root/'campaign'
            atomic_json(seed/'session_meta.json', {'schema':'e1e2_session_v1','cond':'A'})
            atomic_json(root/'seed_B'/'session_meta.json', {'schema':'e1e2_session_v1','cond':'B'})
            for name in ('a','b','pending'):
                atomic_json(campaign/'batches'/name/'session_meta.json', {'schema':'e1e2_session_v1','cond':'A'})
            state={'schema':'rsi_campaign_v1','status':'paused','active_batch':None,'seed_sessions':[str(seed),str(root/'seed_B')],
                   'batches':[{'phase':'collection','status':'complete','path':'batches/a','plan':[{'cond':'A'}]},
                              {'phase':'evaluation','status':'complete','path':'batches/b','plan':[{'cond':'B'}]},
                              {'phase':'collection','status':'paused','path':'batches/pending','plan':[{'cond':'A'}]}]}
            atomic_json(campaign/'rsi.json',state)
            (campaign/'.rsi.lock').touch()
            self.assertEqual(rsi.campaign_seed_sessions(campaign),[str(seed),str(campaign/'batches/a')])
            with (campaign/'.rsi.lock').open('r') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                with self.assertRaisesRegex(ValueError,'正在运行'):rsi.campaign_seed_sessions(campaign)
            state['status']='running'; atomic_json(campaign/'rsi.json',state)
            with self.assertRaisesRegex(ValueError,'必须停止'):rsi.campaign_seed_sessions(campaign)


class StagedSimulationTests(unittest.TestCase):
    def campaign(self, root, threshold):
        settings=rsi.load_settings(rsi.ROOT/'config.rsi.efficient.yaml')
        settings.update(simulate=True,seed=4)
        settings['memory']['path']=str(root/'memory')
        settings['provider']['backend']='mock'
        settings['budget'].update(max_cycles=1,max_trials=240)
        settings['contact_policy'].update(epochs=25,ensemble_size=3)
        settings['efficiency']['screening_min_difference']=threshold
        return rsi.RSICampaign(root/'run',settings=settings,say=lambda _:None,proposer=propose)

    def test_screen_only_is_saved_but_cannot_promote(self):
        with tempfile.TemporaryDirectory() as d, patch('bottle_scan.agentic.resolve_base',return_value=config()):
            c=self.campaign(Path(d),1.0)
            self.assertEqual(c.run(),0,c.state.get('last_error'))
            cycle=c.state['cycles'][0]
            self.assertEqual(cycle['result']['outcome'],'screen_only_inconclusive')
            self.assertIsNone(cycle['result']['winner'])
            self.assertEqual(c.state['incumbent_method_id'],cycle['branches']['old']['method_id'])
            self.assertEqual(cycle['screening']['branches']['old']['trials'],10)
            self.assertEqual(cycle['screening']['branches']['new']['trials'],10)
            self.assertTrue((c.root/'cycles/001/screening.json').exists())
            self.assertEqual(c.spent(),33+48+44)
            c.check_cycle_origin(cycle)
            with patch.object(c,'spent',return_value=999):
                with self.assertRaisesRegex(ValueError,'等预算不同'): c.finish_screen(cycle)
            c.settings['efficiency']['coverage_weight'] = 9.
            with self.assertRaisesRegex(ValueError,'规则被修改'): c.check_cycle_origin(cycle)
            c.settings['efficiency']['coverage_weight'] = .5
            captured = []
            def observe(_, context):
                captured.append(context)
                return {'action':'pause','method':None,'method_id':None,'reason':'test'}
            c.proposer = observe
            self.assertFalse(c.propose())
            memories = captured[0]['experience_memory']
            self.assertTrue(any(e.get('screening') for m in memories for e in m.get('past_experiences', [])))
            diagnostics = captured[0]['references']['diagnostics']
            self.assertTrue(any(d.get('code') == 'B_offset_feedback' and d.get('source') == 'completed_previous_screening' for d in diagnostics))
            for branch in cycle['branches'].values():
                counts={}
                for b in c.state['batches']:
                    if b['id'] in branch['collect']:
                        for row in b['plan']:
                            if row['kind']=='scan': counts[row['delta_mm']]=counts.get(row['delta_mm'],0)+1
                self.assertLessEqual(max(counts.values()),4)

    def test_confirmation_is_fresh_independent_and_compared_at_20_per_branch(self):
        with tempfile.TemporaryDirectory() as d, patch('bottle_scan.agentic.resolve_base',return_value=config()):
            c=self.campaign(Path(d),0.)
            self.assertEqual(c.run(),0,c.state.get('last_error'))
            cycle=c.state['cycles'][0]
            self.assertTrue(cycle['screening']['confirmation_requested'])
            self.assertNotEqual(cycle['result']['outcome'],'invalid',cycle['result'])
            self.assertEqual(cycle['comparison']['old']['trials'],20)
            self.assertEqual(cycle['comparison']['new']['trials'],20)
            self.assertEqual(c.spent(),33+92+44)
            for key,b in cycle['branches'].items():
                screen=set(cycle['screening']['branches'][key]['batches'])
                confirmation=set(b['evaluation'])
                self.assertTrue(screen and confirmation and not screen & confirmation)
                model=c.model(b['model_id'])
                self.assertFalse(set(model['source_sessions']) & set(c.roots(list(screen|confirmation))))


if __name__=='__main__': unittest.main()
