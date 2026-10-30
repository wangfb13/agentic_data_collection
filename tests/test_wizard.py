"""Safety, resumability and UI boundaries of the offline wizard prototype."""
from __future__ import annotations
import argparse
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock,patch
import numpy as np
import yaml
from bottle_scan.session import Session,atomic_json
from bottle_scan.sensing import TemplateGuard
from bottle_scan.wizard import (UI,WizardRunner,WizardSimBackend,simulated_config,
    main,station_hash,require_profile,STEPS,save_yaml,run,check_zero_drift,teach_bounds,item_done,setup)
from bottle_scan.wizard_calibration import fit_response,center_correction


class WizardTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.cfg=simulated_config()
    def profile(self):
        p={'schema':'bottle_station_simulation_v1','simulation_only':True,'ready':True,
            'completed_steps':list(STEPS),'config':copy.deepcopy(self.cfg),
            'model':{'source':'synthetic'},'dimensions':{'open_mm':90},
            'camera_check':{'cameras':['right_wrist'],'roi_xywh':[92,42,61,101]}}
        p['hardware_hash']=station_hash(p)
        path=self.root/'station_profile.yaml';save_yaml(path,p)
        return path,p
    def runner(self,answers=()):
        session=Session.create(self.root/'session',self.cfg,np.asarray(self.cfg['T0']),simulated=True)
        backend=WizardSimBackend(self.cfg,session.T0)
        self.addCleanup(backend.close)
        stream=io.StringIO();iterator=iter(answers)
        ui=UI(stream=stream,read=lambda p:next(iterator))
        runner=WizardRunner(session,backend,ui)
        return runner,stream
    def record(self,ident,uv=(23.5,15.5)):
        return {'trial_id':ident,'task':'contact','status':'completed','kind':'drift_zero','stages':[{'name':'PROBE','start_ns':len(ident)}],
            'auto':{'bilateral_contact':True,'probe_not_settled':False},'offsets':{},
            'descriptor_windows':{'PROBE':{s:{'median':{'centroid_u_px':uv[0],'centroid_v_px':uv[1]}} for s in ('left','right')}}}
    def test_real_rehearsal_and_mixed_modes_fail_before_profile_or_backend(self):
        for args in (['setup','--real','--rehearsal'],['run','--mode','contact','--real','--rehearsal'],['run','--mode','visual','--simulate','--real']):
            with patch('bottle_scan.wizard.setup') as setup,patch('bottle_scan.wizard.run') as run_call,patch('bottle_scan.wizard.WizardSimBackend') as backend:
                with self.assertRaises(SystemExit) as error,patch('sys.stderr',io.StringIO()):main(args)
                self.assertEqual(error.exception.code,2);setup.assert_not_called();run_call.assert_not_called();backend.assert_not_called()
    def test_camera_tcp_and_code_hash_changes_reject_profile(self):
        path,p=self.profile();self.assertTrue(require_profile(path)['simulation_only'])
        for mutate in (lambda x:x['config']['tool']['grasp_center_in_tcp'][2].__setitem__(3,117),
                       lambda x:x['config']['cameras']['right_wrist'].__setitem__('width',1920),
                       lambda x:x['camera_check'].__setitem__('roi_xywh',[0,0,8,8])):
            changed=copy.deepcopy(p);mutate(changed);save_yaml(path,changed)
            with self.assertRaisesRegex(ValueError,'哈希变化'):require_profile(path)
    def test_incomplete_profile_never_marks_ready(self):
        path,p=self.profile();p['completed_steps']=list(STEPS[:-1]);save_yaml(path,p)
        with self.assertRaisesRegex(ValueError,'未完成'):require_profile(path)
    def test_no_pending_resume_never_creates_backend_or_doctor(self):
        path,p=self.profile()
        session=Session.create(self.root/'done',self.cfg,np.asarray(self.cfg['T0']),simulated=True)
        session.meta.update(wizard_simulation=True,wizard_mode='contact',station_hardware_hash=p['hardware_hash'],wizard_preparation_complete=True)
        session.save_meta();session.lock_plan('wizard_contact',[])
        args=argparse.Namespace(profile=str(path),session=str(session.path),mode='contact')
        with patch('bottle_scan.wizard.WizardSimBackend') as backend,patch('bottle_scan.wizard.sim_doctor') as doctor:
            self.assertEqual(run(args,UI(True,stream=io.StringIO())),0)
        backend.assert_not_called();doctor.assert_not_called()
    def test_airborne_resume_refuses_before_backend(self):
        path,p=self.profile()
        session=Session.create(self.root/'air',self.cfg,np.asarray(self.cfg['T0']),simulated=True)
        session.meta.update(wizard_simulation=True,wizard_mode='contact',station_hardware_hash=p['hardware_hash'],wizard_preparation_complete=True)
        session.save_meta();session.lock_plan('wizard_contact',[{'trial_id':'trial_1'}])
        session.progress['requires_operator_recovery']=True;session.save_progress()
        with patch('bottle_scan.wizard.WizardSimBackend') as backend:
            with self.assertRaisesRegex(ValueError,'空中'):run(argparse.Namespace(profile=str(path),session=str(session.path),mode='contact'),UI(True))
        backend.assert_not_called()
    def test_right_only_guard_ignores_left_motion_or_flat_image_and_roundtrips(self):
        backend=WizardSimBackend(self.cfg,np.asarray(self.cfg['T0']));images=backend.images()
        images['left_wrist'][:]=0
        guard=TemplateGuard(images,{'right_wrist':[92,42,61,101]}, {'right_wrist':2},checked_cameras=['right_wrist'])
        moved=copy.deepcopy(images);moved['left_wrist'][:]=255
        result=guard.compare(moved)
        self.assertTrue(result['passed']);self.assertEqual(set(result['cameras']),{'right_wrist'})
        guard.save(self.root/'reference');loaded=TemplateGuard.load(self.root/'reference')
        self.assertEqual(loaded.checked_cameras,('right_wrist',));self.assertEqual(loaded.reference_sha256,guard.reference_sha256)
        moved['right_wrist']=np.roll(moved['right_wrist'],6,axis=1)
        self.assertFalse(loaded.compare(moved)['passed'])
        self.assertEqual(loaded.reference_sha256,guard.reference_sha256)
    def test_missing_checked_right_image_fails_closed(self):
        b=WizardSimBackend(self.cfg,np.asarray(self.cfg['T0']));g=TemplateGuard(b.images(),{'right_wrist':[92,42,61,101]}, {'right_wrist':2},checked_cameras=['right_wrist'])
        self.assertFalse(g.compare({'left_wrist':b.images()['left_wrist']})['passed'])
    def test_response_fits_mirror_signs_and_rejects_degeneracy(self):
        J=np.array([[1.3,.2],[-.1,1.1]])
        rows=[]
        for xy in ([0,0],[5,0],[-5,0],[0,5],[0,-5]):
            rows.append({'actual_al_mm':xy,'centroids_px':{'left':(np.array([20,15])+J@xy).tolist(),'right':(np.array([20,15])-J@xy).tolist()}})
        model=fit_response(rows)
        np.testing.assert_allclose(model['left']['J_px_per_tcp_mm'],J)
        np.testing.assert_allclose(model['right']['J_px_per_tcp_mm'],-J)
        for row in rows:row['centroids_px']['left']=[20,15]
        with self.assertRaisesRegex(ValueError,'过弱'):fit_response(rows)
    def test_center_step_limit_and_both_fingers_required(self):
        models={s:{'J_px_per_tcp_mm':np.eye(2).tolist()} for s in ('left','right')}
        centers={s:[0,0] for s in models}
        result=center_correction({'left':[40,30],'right':[40,30]},centers,models)
        self.assertAlmostEqual(np.linalg.norm(result['delta_al_mm']),3);self.assertFalse(result['converged'])
        result=center_correction({'left':[0,0],'right':[2,0]},centers,models)
        self.assertFalse(result['converged'])
    def test_zero_baseline_locks_three_and_records_threshold_alarm(self):
        runner,stream=self.runner(['1'])
        runner.session.meta['wizard_sensor_response']={s:{'inverse_tcp_mm_per_px':np.eye(2).tolist()} for s in ('left','right')}
        for i in range(3):check_zero_drift(runner,self.record(f'zero_{i}'))
        baseline=copy.deepcopy(runner.session.meta['zero_drift']['baseline'])
        check_zero_drift(runner,self.record('zero_4',(25.5,15.5)))
        state=runner.session.meta['zero_drift']
        self.assertEqual(state['baseline'],baseline);self.assertTrue(state['samples'][-1]['alert'])
        self.assertEqual(state['samples'][-1]['operator_action'],'1');self.assertTrue(runner.need_move)
        self.assertIn('漂移报警',stream.getvalue())
        self.assertTrue(list((runner.session.path/'wizard_events').glob('*zero_drift_alarm*')))
    def test_zero_nonzero_offsets_or_unstable_probe_do_not_enter_baseline(self):
        runner,_=self.runner()
        runner.session.meta['wizard_sensor_response']={s:{'inverse_tcp_mm_per_px':np.eye(2).tolist()} for s in ('left','right')}
        row=self.record('nonzero');row['offsets']['a_mm']=1
        check_zero_drift(runner,row)
        row=self.record('unstable');row['auto']['probe_not_settled']=True
        check_zero_drift(runner,row)
        self.assertEqual(runner.session.meta['zero_drift']['samples'],[])
    def test_labels_only_ask_note_when_n_and_keep_legacy_mapping(self):
        runner,stream=self.runner(['n','sample note','5'])
        runner.meta=self.record('current');runner.meta['bottle']='light';runner.meta['auto']['slip_suspected']=True
        runner.label()
        self.assertEqual(runner.meta['operator'],{'label':'s','note':'sample note'})
        self.assertEqual(stream.getvalue().count('备注：'),1)
    def test_normal_after_automatic_drop_requires_second_confirmation(self):
        runner,stream=self.runner(['1','NO','4',''])
        runner.meta=self.record('current');runner.meta['bottle']='light';runner.meta['auto']['dropped']=True
        with patch.object(runner,'check_bottle'):runner.label()
        self.assertEqual(runner.meta['operator']['label'],'d');self.assertIn('矛盾',stream.getvalue())
    def test_pause_then_q_finishes_current_label_and_requires_next_move(self):
        runner,stream=self.runner(['p','','q','1'])
        runner.meta=self.record('current');runner.meta['bottle']='light';runner.need_move=False
        runner.label()
        self.assertTrue(runner.quit_requested);self.assertTrue(runner.need_move)
        self.assertEqual(runner.meta['operator']['label'],'')
    def test_undo_retains_original_label_and_does_not_rerun_trial(self):
        runner,stream=self.runner(['u','5','1'])
        old=self.record('trial_old');old['operator']={'label':'t','note':'old note'}
        runner.session.start_trial('trial_old');runner.session.complete(old)
        runner.meta=self.record('current');runner.meta['bottle']='light'
        runner.label()
        stored=json.loads((runner.session.path/'trials/trial_old/meta.json').read_text())
        self.assertEqual(stored['operator']['label'],'s');self.assertEqual(stored['operator_history'][0]['previous']['label'],'t')
        self.assertEqual(runner.session.progress['completed'],['trial_old'])
    def test_setup_checkpoints_resume_after_operator_interrupt(self):
        path=self.root/'station_profile.yaml'
        args=argparse.Namespace(profile=str(path))
        answers=iter([''])
        def interrupt(_):
            try:return next(answers)
            except StopIteration:raise KeyboardInterrupt('test interruption')
        with self.assertRaises(KeyboardInterrupt):
            setup(args,UI(stream=io.StringIO(),read=interrupt))
        checkpoint=yaml.safe_load((self.root/'station_profile_setup/checkpoint.yaml').read_text())
        self.assertEqual(checkpoint['completed_steps'],['doctor','geometry'])
        self.assertFalse(path.exists())
        stream=io.StringIO()
        with patch('bottle_scan.stop_validation.run_stop_test',return_value={'passed':True}) as stop:
            self.assertEqual(setup(args,UI(True,stream=stream)),0)
        stop.assert_called_once()
        self.assertIn('已保存，跳过：geometry',stream.getvalue())
        self.assertTrue(require_profile(path)['ready'])

    def test_visual_committed_images_with_missing_label_are_pending(self):
        runner,_=self.runner()
        item={'point_id':'point_0001'}
        runner.session.start_trial(item['point_id'])
        runner.session.complete({'trial_id':item['point_id'],'status':'completed','task':'visual'})
        self.assertFalse(item_done(runner.session,item,'visual'))
        runner.session.complete({'trial_id':item['point_id'],'status':'completed','task':'visual','operator':{'label':''}})
        self.assertTrue(item_done(runner.session,item,'visual'))
    def test_roi_change_keeps_reference_pixels_and_legacy_guard_file(self):
        runner,_=self.runner()
        runner.guard=TemplateGuard(runner.backend.images(),{'right_wrist':[92,42,61,101]},
            {'right_wrist':2},checked_cameras=['right_wrist'])
        hashes=runner.guard.reference_sha256.copy()
        runner._set_roi([90,40,65,105])
        self.assertEqual(runner.guard.reference_sha256,hashes)
        self.assertEqual(runner.guard.checked_cameras,('right_wrist',))

    def test_teaching_does_not_double_apply_safety_margin(self):
        cfg=copy.deepcopy(self.cfg);before=copy.deepcopy(cfg['safety'])
        teaching=teach_bounds(cfg,UI(True,stream=io.StringIO()))
        self.assertEqual(teaching['table_z_mm'],0)
        self.assertEqual(cfg['safety'],before)


if __name__=='__main__':unittest.main()
