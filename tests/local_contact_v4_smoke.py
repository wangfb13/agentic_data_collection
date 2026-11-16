"""Opt-in offline integration against the installed V4 runtime and its fake hardware."""
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
V4=Path('/home/dm/DM_das/contact_scheduler_20261104_experiment_b_v4_single_open_20261110')
sys.path.insert(0,str(V4/'runtime'))
import bottle_scan
bottle_scan.__path__.append(str(ROOT/'bottle_scan'))
sys.path.append(str(ROOT))
from bottle_scan.contact_recovery_backend import ContactRecoveryBackend
from bottle_scan.contact_skill_backend import _SkillRunner
from tests.test_contact_recovery_backend import Hardware,Runner
from skills.contact_position_a.runtime import install_backend
from skills.contact_position_a.local_deploy import prepare, offline_station
from skills.contact_position_a.local_backend import read


class LocalBackend(ContactRecoveryBackend):
    @property
    def T0(self):raise AssertionError('Old T0 read')
    @property
    def bounds(self):raise AssertionError('Old bounds read')


class ProbeRunner(Runner):
    def regulate(self,*args,**kwargs):
        probe=super().regulate(*args,**kwargs)
        probe['normal_sdk']=[1.5,1.5]
        probe['descriptors']=self.backend.contact_reference['standard_sample']['descriptors']
        for side in ('left','right'):
            stamps=probe['window'][0]['frame_ns']
            data=np.zeros((32,32,2));data[5:18,10:25,0]=.1
            self.hw.dd[side]=[NS(monotonic_ns=stamp,data=data) for stamp in stamps]
            self.backend._graph_buffers['wrench_raw_'+side].extend(NS(monotonic_ns=t,data=np.ones(6)) for t in stamps)
        return probe


class Smoke(unittest.TestCase):
    def test_fitted_models_ignore_external_pose_and_hidden_labels(self):
        from copy import deepcopy
        import joblib
        from bottle_scan.contact_policy import ContactPolicy
        from bottle_scan.state_graph.skill import SymbolicRecoverySkill
        from skills.contact_position_a.local_decision import decide_local
        assets=ROOT/'outputs/local_contact_a_v1'
        g=SymbolicRecoverySkill(joblib.load(assets/'graph/models/bundle.joblib'))
        cfg=read(ROOT/'config.state_graph.v4_scheduler.json')['execution_config']
        b=NS(cfg=cfg,offset_policy=ContactPolicy(read(assets/'offset.json'),cfg['servo']),
             contact_reference=read(ROOT/'skills/contact_position_a/white_reference.json'),
             hw=NS(now=lambda:500),local_settings={'gain':1.,'offset_deadband_mm':2.,'max_steps':3,'observation_max_age_s':.5})
        # Use a measured offset-training PROBE with a valid geometric prediction.
        for trial in b.offset_policy.fit['source_trials']:
            ds=read(Path(trial['trial_path'])/'meta.json')['probe_initial']['descriptors']
            pred=b.offset_policy.predict(ds)
            if pred['valid'] and min(b.offset_policy.fit['trained_offset_range_mm'])<=-pred['action_mm']<=max(b.offset_policy.fit['trained_offset_range_mm']):break
        else:self.fail('No eligible measured offset probe')
        row=g.bundle['support'][0];o=dict(observation_id='probe',timestamp_ns=500,safety_ok=True,
            force_stable=True,bilateral=True,descriptors=ds)
        changed=deepcopy(row);changed.update(T0_mm='other robot',measured_tcp_mm='other pose',
            plan_coords={'a_offset_mm':1000},supervision={'offset_a_mm':-1000},placement_approach_group='never-seen',
            post={'hidden_downstream_outcome':False})
        for source in ('offset_servo','graph_policy'):
            b.action_source=source
            before=decide_local(b,o,row,g,0)
            after=decide_local(b,{**o,'T0_mm':'other','true_offset':-999},changed,g,0)
            self.assertEqual(before,after)
            if source=='offset_servo':self.assertIsNotNone(before[3]['offset_estimate'])

    def test_full_real_backend_fake_io(self):
        assets=ROOT/'outputs/local_contact_a_v1'
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            _,env=prepare(assets,offline_station(p),p,'offset_servo')
            config=read(p/'scheduler_config.json')
            with patch.dict(os.environ,env),patch('bottle_scan.contact_skill_backend._make_hardware',side_effect=AssertionError('real hardware')):
                backend=LocalBackend(config,output_dir=p)
                install_backend(backend,config['state_graph_execution'])
                runner=_SkillRunner(backend,p)
                self.assertNotIn('T0',runner.meta)
                # Test a genuinely issued action even when today's fitted Ready head
                # classifies every matching historical handoff as already acceptable.
                graph=backend.recovery.runtime
                hw=Hardware(np.eye(4));backend.hw=hw;backend.prepared=True
                hw.dd_reference=[np.zeros((32,32,2))]*2;hw.dd={};hw.lock=threading.Lock()
                hw.zero['offset_raw']=[[0]*6]*2
                sample=hw.snapshot
                hw.snapshot=lambda:{**sample(),'width':.4,'wrench_zeroed':np.ones((2,6))}
                with patch.object(graph,'infer',side_effect=[{'state':'not_ready'},{'state':'ready'}]), \
                     patch.object(backend.offset_policy,'predict',return_value={'valid':True,'action_mm':-3.,'uncertainty_mm':.5}), \
                     patch('bottle_scan.contact_recovery_backend._SkillRunner',ProbeRunner):
                    result=backend.execute(caller_paused=True,retain_probe_for_handoff=True)
                self.assertTrue(result['success'])
                self.assertTrue(backend._handoff_probe_owned)
                self.assertEqual(result['primary_resume_contract']['outcome'],'restored')
                from bottle_scan.contact_scheduler import SchedulerRuntime
                scheduler=SchedulerRuntime(NS(history_horizon=1),backend,clock=lambda: hw.now()*1e-9)
                self.assertEqual(scheduler._checked_primary_contract(result)['next_controller'],'existing_forcepolicy')
                receipt=result['corrections'][0]['receipt']
                self.assertAlmostEqual(receipt['action_mm'],-3.)
                self.assertEqual(len(result['observations']),2)
                self.assertIsNone(result['evaluation_only']['offset_ground_truth_mm'])
                self.assertFalse(hasattr(backend,'_graph_vision'))


if __name__=='__main__':unittest.main()
