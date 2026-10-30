"""V2 runtime quality and evidence gates; no device imports or live motion."""
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from bottle_scan.backend import SimBackend
from bottle_scan.config import demo_config
from bottle_scan.runtime import Runner, SafetyFault
from bottle_scan.session import Session
from bottle_scan.commissioning import hardware_binding, require_prior_session, accept_commissioning
from prepare import record_height_crosscheck


class QualityWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.cfg = demo_config()
        self.session = Session.create(Path(self.temp.name)/'s',self.cfg,np.asarray(self.cfg['T0']),simulated=True)
        self.backend = SimBackend(self.cfg,self.session.T0)
        self.runner = Runner(self.session,self.backend,single_step=False,unattended=True)
        self.runner.meta = {'trial_id':'unit','stages':[],'auto':{'exceptions':[]}}
        (self.session.path/'trials'/'unit').mkdir()
        self.backend.width = self.backend.width_command = 36.
        self.backend.set_force_target(2.)
        self.backend.step(.05)

    def test_probe_requires_continuous_band_and_flags_timeout(self):
        self.assertTrue(self.runner.probe())
        window = self.runner.meta['auto']['probe_stable_window']
        self.assertGreaterEqual(window['end_ns']-window['start_ns'],int(self.cfg['probe']['settle_s']*1e9))
        self.backend.scenario = 'no_contact'
        self.assertFalse(self.runner.probe())
        self.assertTrue(self.runner.meta['auto']['probe_not_settled'])
        self.assertIsNone(self.runner.meta['auto']['probe_stable_window'])

    def setup_lower(self):
        self.runner.grasp_pose = self.session.T0.copy()
        self.backend.pose[2,3] += 10.
        self.backend.step(.01)
        self.runner.airborne = True
        self.session.progress['requires_operator_recovery'] = True
        self.runner.meta['auto']['hold_vertical_tangent_sum'] = 2.

    def test_lower_stops_when_unloaded_and_never_commands_below_original(self):
        self.setup_lower()
        with patch.object(self.runner,'tangential_load',return_value=(.1,'test')):
            self.runner.lower()
        auto=self.runner.meta['auto']
        self.assertEqual(auto['lower_stop_reason'],'unloaded')
        self.assertGreater(auto['lower_unload_height_difference_mm'],0.)
        self.assertFalse(self.runner.airborne)
        self.assertTrue(any(name=='stop_arm' for name,_ in self.backend.calls))
        for name,target in self.backend.calls:
            if name=='move_linear': self.assertGreaterEqual(target[2,3],self.session.T0[2,3])

    def test_lower_original_height_fallback(self):
        self.setup_lower()
        with patch.object(self.runner,'tangential_load',return_value=(2.,'test')):
            self.runner.lower()
        self.assertEqual(self.runner.meta['auto']['lower_stop_reason'],'original_height')
        self.assertAlmostEqual(self.backend.pose[2,3],self.session.T0[2,3])

    def test_lower_refuses_measured_height_below_floor_keeps_latch(self):
        self.setup_lower()
        original=self.backend.snapshot
        def too_low():
            snap=copy.deepcopy(original()); snap['tcp'][2,3]=self.session.T0[2,3]-1
            return snap
        with patch.object(self.backend,'snapshot',side_effect=too_low), self.assertRaises((SafetyFault,RuntimeError)):
            self.runner.lower()
        self.assertTrue(self.session.progress['requires_operator_recovery'])

    def test_tangent_fallback_does_not_cancel_opposed_fingers(self):
        snap=copy.deepcopy(self.backend.snapshot())
        snap['wrench_force']=np.array([[3.,4.,2.],[-3.,-4.,2.]])
        self.assertEqual(self.runner.tangential_load(snap),(10.,'sum_of_tangential_magnitudes'))

    def test_lower_final_feedback_floor_checked_even_without_settle_samples(self):
        self.setup_lower()
        self.runner.cfg['motion']['settle_s'] = 0.
        start = copy.deepcopy(self.backend.snapshot())
        final = copy.deepcopy(start)
        final['tcp'][2,3] = self.session.T0[2,3] - .5
        with patch.object(self.runner,'move'), patch.object(self.runner,'wait',return_value=[]), \
                patch.object(self.runner,'_healthy_snapshot',side_effect=[start,final]):
            with self.assertRaisesRegex(SafetyFault,'final feedback'):
                self.runner.lower()
        self.assertTrue(self.runner.airborne)
        self.assertTrue(self.session.progress['requires_operator_recovery'])
        self.assertEqual(self.backend.calls[-1][0],'stop_arm')

    def test_height_uses_grasp_center_not_tcp(self):
        self.runner.cfg['tool']['grasp_center_in_tcp'][2][3]=10.
        record_height_crosscheck(self.runner,self.session)
        self.assertEqual(self.session.meta['grasp_center_height']['computed_mm'],80.)

    def test_tune_reexpresses_fixed_bottle_origin_in_new_t0(self):
        from prepare import tune
        self.runner.cfg['gripper']['close_speed_mm_s'] = 100.
        self.backend.cfg['gripper']['close_speed_mm_s'] = 100.
        before = self.session.T0[:3,3] + self.session.T0[:3,:3] @ np.asarray(self.runner.cfg['geometry']['bottle']['center_offset_mm'])
        with patch.object(self.runner,'ask',side_effect=['a+','']):
            tune(self.runner,self.session)
        after = self.session.T0[:3,3] + self.session.T0[:3,:3] @ np.asarray(self.runner.cfg['geometry']['bottle']['center_offset_mm'])
        np.testing.assert_allclose(before,after)
        self.assertAlmostEqual(self.session.T0[0,3],self.cfg['T0'][0][3]+1.)


class CommissioningTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.cfg=demo_config()
        self.session=Session.create(Path(self.temp.name)/'prior',self.cfg,np.asarray(self.cfg['T0']),simulated=True)

    def records(self):
        records=[]
        for i,kind in enumerate(('commission_zero','commission_deep_a','commission_above_cap')):
            records.append({'trial_id':str(i),'kind':kind,'status':'completed','auto':{
                'bilateral_contact':i!=2,'no_bilateral_contact':i==2,'force_reached':i!=2,
                'probe_not_settled':False,'max_force':[2.,2.],'exceptions':[]},'position_checks':[{'passed':True}]})
        return records

    def test_binding_excludes_new_t0_but_changes_for_hardware(self):
        other=copy.deepcopy(self.cfg);other['T0'][0][3]+=20
        self.assertEqual(hardware_binding(self.cfg),hardware_binding(other))
        other['gripper']['close_speed_mm_s']*=.5
        self.assertNotEqual(hardware_binding(self.cfg),hardware_binding(other))

    def test_simulated_pass_never_authorizes_real(self):
        accept_commissioning(self.session,self.records())
        self.assertTrue(require_prior_session(self.cfg,self.session.path,simulated=True))
        with self.assertRaisesRegex(ValueError,'Simulated'):
            require_prior_session(self.cfg,self.session.path)

    def test_pass_confirmation_is_required_for_real_evidence(self):
        self.session.meta['simulated']=False
        report=accept_commissioning(self.session,self.records(),confirm=lambda _: 'no')
        self.assertFalse(report['passed'])
        self.assertFalse(self.session.progress['commissioning_passed'])

    def test_flagged_or_skipped_trial_cannot_pass(self):
        records=self.records();records[0]['auto']['probe_not_settled']=True
        self.assertFalse(accept_commissioning(self.session,records)['passed'])
        records=self.records();records[2]['status']='skipped_safety'
        self.assertFalse(accept_commissioning(self.session,records)['passed'])

    def test_lite_cannot_bootstrap_another_session(self):
        self.session.meta['commissioning_acceptance']={'level':'lite','passed':True,'operator_confirmation':'PASS','binding':hardware_binding(self.cfg)}
        self.session.progress['commissioning_passed']=True
        self.session.save_meta();self.session.save_progress()
        with self.assertRaisesRegex(ValueError,'full three-point'):
            require_prior_session(self.cfg,self.session.path,simulated=True)
