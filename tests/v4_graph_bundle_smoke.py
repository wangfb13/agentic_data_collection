"""Run with project Python; all robot/force components are the V4 fake fixtures."""
import sys
from pathlib import Path
REPO=Path(__file__).resolve().parents[1]
BUNDLE=Path('/home/dm/DM_das/contact_scheduler_20261104_experiment_b_v4_single_open_20261110')
sys.path.insert(0,str(BUNDLE/'runtime'))
import bottle_scan
bottle_scan.__path__.append(str(REPO/'bottle_scan'))
sys.path.append(str(REPO))
import unittest
import numpy as np
from unittest.mock import patch
from types import SimpleNamespace
from tests.test_contact_recovery_backend import RecoveryBackendTests,Runner
from bottle_scan.contact_scheduler import SchedulerRuntime
from skills.contact_position_a.runtime import GraphRecoveryProxy
from bottle_scan.state_graph.skill_interface import load_spec


class GraphBackendSmoke(unittest.TestCase):
    def test_probe_window_survives_later_force_tick_but_not_live_contact_loss(self):
        fixture=RecoveryBackendTests();fixture.setUp()
        try:
            backend=fixture.backend
            runner=Runner(backend,Path(fixture.temp.name))
            probe=runner.regulate()
            probe['normal_sdk']=[.8609203924735387,1.7371947954098383]
            sample=backend.hw.snapshot()
            # Actual failure: left finger updates after the certified window,
            # before the next right-finger frame reaches the observer.
            sample['normal']=np.array([1.9892195095618566,1.7371947954098383])
            backend.hw.snapshot=lambda:sample
            self.assertFalse(backend._observation('contact',probe)['force_stable']) # legacy unchanged
            backend.config['state_graph_execution']={'request':'test'}
            self.assertTrue(backend._observation('contact',probe)['force_stable'])
            self.assertFalse(backend._observation('contact',probe)['force_stable']) # no cached reuse
            backend._last_window_end=None;backend._last_window_ids=None
            sample['normal']=np.array([0.,1.7])
            lost=backend._observation('contact',probe)
            self.assertFalse(lost['bilateral']);self.assertFalse(lost['force_stable'])
            sample['normal']=np.array([backend.cfg['force']['max']+1.,1.7])
            with self.assertRaisesRegex(RuntimeError,'Force is invalid or exceeds'):
                backend._observation('contact',probe)
            sample['normal']=np.array([1.5,1.5])
            for recorded in ([0.,3.],[2.,2.],None):
                if recorded is None:probe.pop('normal_sdk',None)
                else:probe['normal_sdk']=recorded
                self.assertFalse(backend._observation('contact',probe)['force_stable'])
        finally:fixture.doCleanups()

    def test_graph_prelude_accepts_small_probes_only_under_explicit_graph_config(self):
        import os
        from bottle_scan.contact_scheduler import experiment_b_offset_from_environment
        from bottle_scan.contact_skill_backend import ContactSkillBackend
        fixture=RecoveryBackendTests();fixture.setUp()
        try:
            backend=fixture.backend;backend.config['state_graph_execution']={'request':'test'}
            with patch.dict(os.environ,{'RIGHT_GRIPPER_HANDOFF_MODE':'model_close_contact'}):
                for x in (2.5,-2.5,12.,25.):
                    with patch.dict(os.environ,{'CONTACT_EXPERIMENT_B_OFFSET_MM':str(x)}):
                        self.assertEqual(experiment_b_offset_from_environment(graph_config=backend.config),x)
                    with patch.object(backend.bounds,'grasp',side_effect=RuntimeError('offset_accepted_no_motion')):
                        with self.assertRaisesRegex(RuntimeError,'offset_accepted_no_motion'):
                            ContactSkillBackend.experiment_b_prelude(backend,offset_mm=x,caller_paused=True)
                with self.assertRaises(ValueError):ContactSkillBackend.experiment_b_prelude(backend,offset_mm=26,caller_paused=True)
                with patch.dict(os.environ,{'CONTACT_EXPERIMENT_B_OFFSET_MM':'2.5'}):
                    with self.assertRaises(ValueError):experiment_b_offset_from_environment()
        finally:fixture.doCleanups()

    def test_fresh_correction_exit_is_accepted_by_actual_v4_scheduler(self):
        fixture=RecoveryBackendTests();fixture.setUp()
        try:
            backend=fixture.backend;backend._graph_probe={};backend.single_correction_open_handoff=False
            spec=load_spec(REPO/'skills/contact_position_a/spec.campaign_v4.json')
            proxy=GraphRecoveryProxy(backend,{'mode':'planned','actions':[],'model_id':None},spec)
            proxy.mode='learned';backend.recovery=proxy
            predictions=iter([{'state':'recoverable','action':{'kind':'tcp_translation_a','u':[-5]},'value':{}},
                              {'state':'ready','policy':None,'value':{'p_hat':.95,'V':0}}])
            proxy.runtime=SimpleNamespace(infer=lambda row:next(predictions))
            def live(backend,probe,observation,folder,index,previous,receipt,spec):
                self.assertTrue(observation['force_stable'])
                row={'node_id':'obs'+str(index)}
                return {**row,'attrs':{}},row
            with patch('skills.contact_position_a.runtime.live_window',live),patch('bottle_scan.contact_recovery_backend._SkillRunner',Runner):
                result=backend.execute(caller_paused=True,retain_probe_for_handoff=True)
            self.assertTrue(result['success']);self.assertTrue(backend._handoff_probe_owned)
            self.assertEqual(result['model_schema'],'symbolic_recovery_v4')
            self.assertEqual(len(result['observations']),2);self.assertEqual(len(result['corrections']),1)
            self.assertAlmostEqual(result['corrections'][0]['receipt']['action_mm'],-5.)
            runtime=object.__new__(SchedulerRuntime);runtime.backend=backend
            self.assertTrue(runtime._successful_exit(result));self.assertTrue(runtime._actual_invocation(result))
            self.assertIsNone(runtime._checked_primary_contract(result))
        finally:fixture.doCleanups()

if __name__=='__main__':unittest.main(defaultTest='GraphBackendSmoke',verbosity=2)
