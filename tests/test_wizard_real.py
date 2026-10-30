"""Real-route tests with injected observers/publishers. No ROS or simulated CLI."""
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
from bottle_scan.config import demo_config,validate_config
from bottle_scan.session import Session,atomic_json
from bottle_scan.wizard import UI,main,save_yaml,WizardRunner
from bottle_scan.wizard_real import (REAL_STEPS,require_real_profile,station_hash_real,
    run_real,setup_real,stop_station_gate,full_plan,gate_opening)
from bottle_scan.wizard_width import fit_width_points,WidthBootstrap
from bottle_scan.hardware_gripper import WidthCalibration
from bottle_scan.geometry import _cylinder_prism


class FakeUI:
    rehearsal=False
    def __init__(self,answers=()):self.answers=iter(answers);self.messages=[]
    def say(self,s):self.messages.append(s)
    def ask(self,s,*args,**kwargs):self.messages.append(s);return next(self.answers)
    def confirm(self,s,token=''):
        if self.ask(s)!=token:raise KeyboardInterrupt('cancel')


class Reader:
    def __init__(self):
        self.T=np.eye(4);self.norm=.8;self.force=np.zeros(2);self.lock=__import__('threading').Lock()
        self.grip=Mock();self.sensors=Mock();self.writer=None
    def pose(self):return self.T.copy()
    def gripper(self):return {'normalized':self.norm,'fresh':True,'valid':True,'source_sequence':1}
    def raw(self):return {'wrench_raw':np.column_stack([np.zeros((2,2)),self.force,np.zeros((2,3))]),
        'source_sequences':{'wrench_raw_left':1,'wrench_raw_right':1}}
    def noise(self):return {'wrench_mean':np.zeros((2,6)).tolist(),'normal_std':[.001,.001]}


class PhysicalWizardTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        self.cfg=demo_config();self.cfg['demo']=False
        self.cfg['force']['unit']=self.cfg['units']['force']=self.cfg['tactile']['force_unit']='sdk'
        self.cfg['hardware'].update(profile='public_ros',camera_mounts_declared=True,axis_mapping_declared=True,sensor_mapping_declared=True)
        self.cfg['hardware']['public_ros'].update(software_control_acknowledged=True,processed_dd_accepted=True,raw_normalized_min=.3,force_step_counts=1)
    def model(self):
        return {'urdf_sha256':'physical-fixture','mesh_sha256':{},'T_flange_tcp_mm':{},
            'camera_shapes':{'right_wrist':[180,240,3],'left_wrist':[180,240,3]},'serials':{},'tcp_frames':{}}
    def profile(self):
        profile={'schema':'bottle_station_real_v1','simulation_only':False,'ready':True,
            'completed_steps':list(REAL_STEPS),'config':self.cfg,'model':self.model(),'camera_check':{},'opening_report':'unused'}
        profile['hardware_hash']=station_hash_real(profile)
        path=self.root/'station.yaml';save_yaml(path,profile);return path,profile
    def test_default_routes_to_real_setup_without_automatic_answers(self):
        with patch('bottle_scan.wizard_real.setup_real',return_value=0) as setup:
            self.assertEqual(main(['setup']),0)
        self.assertEqual(setup.call_args.args[0].profile,'station_profile.yaml')
        self.assertFalse(setup.call_args.args[1].rehearsal)
    def test_real_rehearsal_refused_before_any_backend(self):
        with patch('bottle_scan.wizard_real.load_hardware') as backend,patch('sys.stderr',io.StringIO()):
            with self.assertRaises(SystemExit):main(['setup','--real','--rehearsal'])
        backend.assert_not_called()
    def test_check_only_dispatch_does_not_enter_control_setup(self):
        with patch('bottle_scan.wizard_real.check_only',return_value=0) as check,patch('bottle_scan.wizard_real.setup_real') as setup:
            self.assertEqual(main(['setup','--check-only']),0)
        check.assert_called_once();setup.assert_not_called()
    def test_simulated_profile_cannot_authorize_real_run(self):
        path,profile=self.profile();profile['simulation_only']=True;save_yaml(path,profile)
        with self.assertRaisesRegex(ValueError,'模拟'):require_real_profile(path)
    def test_changed_camera_or_tcp_invalidates_hash_before_report_or_backend(self):
        path,profile=self.profile();profile['model']['camera_shapes']['right_wrist'][0]=1080;save_yaml(path,profile)
        with patch('bottle_scan.wizard_real.require_force_report') as force:
            with self.assertRaisesRegex(ValueError,'哈希变化'):require_real_profile(path)
        force.assert_not_called()
    def test_completed_session_does_not_create_observer_or_control_backend(self):
        path,profile=self.profile()
        session=Session.create(self.root/'done',self.cfg,np.asarray(self.cfg['T0']),simulated=False)
        session.meta.update(wizard_domain='physical',wizard_mode='contact',station_hardware_hash=profile['hardware_hash'],wizard_preparation_complete=True)
        session.save_meta();session.lock_plan('wizard_contact',[])
        args=argparse.Namespace(profile=str(path),session=str(session.path),mode='contact')
        with patch('bottle_scan.wizard_real.require_real_profile',return_value=profile),patch('bottle_scan.wizard_real.LiveReader') as reader,patch('bottle_scan.wizard_real.load_hardware') as backend,patch('bottle_scan.wizard_real.doctor') as doctor:
            self.assertEqual(run_real(args,FakeUI()),0)
        reader.assert_not_called();backend.assert_not_called();doctor.assert_not_called()
    def test_resume_airborne_refuses_before_observation_or_control(self):
        path,profile=self.profile()
        session=Session.create(self.root/'air',self.cfg,np.asarray(self.cfg['T0']),simulated=False)
        session.meta.update(wizard_domain='physical',wizard_mode='contact',station_hardware_hash=profile['hardware_hash'],wizard_preparation_complete=True)
        session.save_meta();session.lock_plan('wizard_contact',[{'trial_id':'trial_1'}]);session.progress['requires_operator_recovery']=True;session.save_progress()
        with patch('bottle_scan.wizard_real.require_real_profile',return_value=profile),patch('bottle_scan.wizard_real.load_hardware') as backend:
            with self.assertRaisesRegex(ValueError,'空中恢复锁'):run_real(argparse.Namespace(profile=str(path),session=str(session.path),mode='contact'),FakeUI())
        backend.assert_not_called()
    def test_failed_doctor_stops_setup_before_readers_or_publishers(self):
        args=argparse.Namespace(profile=str(self.root/'station.yaml'),config='config.ros.yaml')
        with patch('bottle_scan.wizard_real.doctor',side_effect=RuntimeError('publisher conflict')),patch('bottle_scan.wizard_real.LiveReader') as reader,patch('bottle_scan.wizard_real.load_hardware') as backend:
            with self.assertRaisesRegex(RuntimeError,'publisher conflict'):setup_real(args,FakeUI())
        reader.assert_not_called();backend.assert_not_called()
    def test_width_mapping_no_extrapolation_and_two_points_not_linearity(self):
        result=fit_width_points([{'normalized':.4,'width_mm':30.},{'normalized':.8,'width_mm':60.}])
        self.assertFalse(result['linearity_check_possible'])
        with self.assertRaises(ValueError):WidthCalibration(result['points']).to_mm(.3)
        with self.assertRaises(ValueError):fit_width_points([{'normalized':.4,'width_mm':60.},{'normalized':.8,'width_mm':30.}])
    def bootstrap(self,reader=None,ui=None):
        reader=reader or Reader()
        b=WidthBootstrap(self.cfg,reader,ui or FakeUI(),self.root/'width',publisher_factory=Mock())
        b.publisher=Mock();b.start_pose=reader.pose();b.open_value=.8;b.offset=np.zeros(2)
        return b,reader
    def test_width_cancellation_before_MOVE_never_creates_publisher(self):
        reader=Reader();reader.norm=1.;ui=FakeUI(['EMPTY','cancel'])
        factory=Mock();b=WidthBootstrap(self.cfg,reader,ui,self.root/'width',publisher_factory=factory)
        with patch('bottle_scan.wizard_width.guard_all_commands'),patch('bottle_scan.wizard_width.current_ros_node_names',return_value=()):
            with self.assertRaises(KeyboardInterrupt):b.measure('body',36.,1.)
        factory.assert_not_called()
    def test_width_stale_force_or_arm_change_prevents_publication(self):
        b,reader=self.bootstrap();reader.T[0,3]=3.
        with patch('bottle_scan.wizard_width.guard_all_commands'),patch('bottle_scan.wizard_width.current_ros_node_names',return_value=()):
            with self.assertRaisesRegex(RuntimeError,'TCP'):b.publish(.799)
        b.publisher.publish.assert_not_called()
        reader.T[0,3]=0.;reader.force[:]=self.cfg['force']['max_N']+1
        with self.assertRaisesRegex(RuntimeError,'F_max'):b.check()
    def test_width_probe_force_and_normalized_bounds_prevent_closure(self):
        b,reader=self.bootstrap();reader.force[:]=1.
        with patch('bottle_scan.wizard_width.guard_all_commands'),patch('bottle_scan.wizard_width.current_ros_node_names',return_value=()):
            with self.assertRaisesRegex(RuntimeError,'PROBE'):b.publish(.799)
            with self.assertRaisesRegex(RuntimeError,'边界'):b.publish(.29)
        b.publisher.publish.assert_not_called()
    def test_negative_raw_Fz_is_positive_contact_when_configured(self):
        self.cfg['tactile']['normal_signs']=[-1.,-1.]
        b,reader=self.bootstrap();reader.force[:]=-.3
        _,_,force=b.check()
        np.testing.assert_allclose(force,[.3,.3])
    def test_wrong_force_polarity_prevents_any_further_closure(self):
        b,reader=self.bootstrap();reader.force[:]=-.8
        with patch('bottle_scan.wizard_width.guard_all_commands'),patch('bottle_scan.wizard_width.current_ros_node_names',return_value=()):
            with self.assertRaisesRegex(RuntimeError,'方向与配置不符'):b.publish(.799)
        b.publisher.publish.assert_not_called()
    def test_force_magnitude_limit_applies_in_both_raw_directions(self):
        b,reader=self.bootstrap();reader.force[:]=-self.cfg['force']['max_N']-1
        with self.assertRaisesRegex(RuntimeError,'F_max'):b.check()
    def test_full_stroke_deadline_accounts_for_feedback_without_larger_steps(self):
        b,_=self.bootstrap();b.open_value=1.
        self.assertGreater(b.leg_timeout_s(),120.)
        self.assertLessEqual(b.leg_timeout_s(),300.)
        self.assertEqual(b.step,.001)
        self.assertEqual(b.floor,.3)
    def test_partial_open_cannot_be_reused_as_full_open(self):
        reader=Reader();factory=Mock()
        b=WidthBootstrap(self.cfg,reader,FakeUI(),self.root/'width',publisher_factory=factory)
        with self.assertRaisesRegex(RuntimeError,'先全开'):b.measure('body',36.,.953)
        factory.assert_not_called()
    def test_unknown_rotation_only_allows_rotation_invariant_unloading(self):
        self.cfg['tactile']['sensor_to_tcp']={s:'unavailable' for s in ('left','right')}
        self.assertEqual(validate_config(self.cfg),[])
        self.cfg['lower']['vertical_mapping_confirmed']=True
        self.assertTrue(any('sensor_to_tcp' in x for x in validate_config(self.cfg)))
    def test_table_uncertainty_expands_collision_model(self):
        cfg=copy.deepcopy(self.cfg);seg=cfg['geometry']['bottle']['segments'][0]
        before=_cylinder_prism(np.asarray(cfg['T0']),cfg,seg)[0]
        cfg['geometry']['bottle']['axial_uncertainty_mm']=10.
        after=_cylinder_prism(np.asarray(cfg['T0']),cfg,seg)[0]
        self.assertAlmostEqual(after[:,2].min(),before[:,2].min()-10.)
        self.assertAlmostEqual(after[:,2].max(),before[:,2].max()+10.)
    def test_stop_reuse_cannot_change_control_or_workspace(self):
        profile={'config':copy.deepcopy(self.cfg),'stop_config_snapshot':copy.deepcopy(self.cfg)}
        profile['config']['T0'][0][3]+=1
        profile['config']['geometry']['bottle']['center_offset_mm'][0]-=1
        with patch('bottle_scan.wizard_real.gate_stop_report',return_value={}) as gate:stop_station_gate(profile)
        gate.assert_called_once_with(profile['stop_config_snapshot'])
        profile['config']['safety']['margin_mm']+=1
        with self.assertRaisesRegex(ValueError,'必须重测'):stop_station_gate(profile)
    def test_physical_runner_requires_explicit_authorized_construction(self):
        session=Session.create(self.root/'session',self.cfg,np.asarray(self.cfg['T0']),simulated=False)
        backend=Mock(simulated=False)
        with self.assertRaises(ValueError):WizardRunner(session,backend,UI())
    def test_actual_tip_pad_is_exempted_without_exempting_finger_middle(self):
        from bottle_scan.geometry import gripper_boxes,axis_basis
        cfg=demo_config();g=cfg['geometry']['gripper']
        g.update(finger_size_acl_mm=[145.,8.,44.],finger_center_a_mm=57.,finger_center_l_mm=9.,
                 pad_center_a_mm=117.,pad_center_l_mm=0.,pad_size_al_mm=[20.,10.],pad_depth_mm=2.)
        boxes=gripper_boxes(np.eye(4),cfg,36.,allow_contact=True)
        def occupied(point):
            point=np.asarray(point)
            return any(np.all(np.abs((point-b['center'])@b['basis'])<=b['half']+1e-9) for b in boxes)
        self.assertFalse(occupied([117.,18.5,0.]))
        self.assertTrue(occupied([57.,18.5,0.]))
        full=gripper_boxes(np.eye(4),cfg,36.,allow_contact=False)
        removed=sum(np.prod(2*b['half']) for b in full)-sum(np.prod(2*b['half']) for b in boxes)
        self.assertAlmostEqual(removed,2*20.*10.*2.)
    def test_stop_test_reset_uses_guarded_internal_move_not_operator_pose_json(self):
        from bottle_scan.wizard_real import stop_confirm
        ui=FakeUI(['PASS'])
        with patch('bottle_scan.wizard_real.return_free_for_stop') as reset:
            self.assertEqual(stop_confirm('用 move-tcp 将实验臂复位到 axis_test_pose',self.cfg,self.root,ui),'')
            reset.assert_called_once()
        self.assertEqual(stop_confirm('核对报告 PASS',self.cfg,self.root,ui),'PASS')

    def test_source_profile_unknowns_can_be_completed_without_demo_defaults(self):
        from bottle_scan.config import load_config,unresolved,require_real_ready
        from bottle_scan.wizard_station import configure_geometry,configure_noise_force,configure_policies
        from bottle_scan.wizard_real import locate_bottle
        from bottle_scan.geometry import axis_basis
        cfg=load_config('config.ros.yaml')
        audit=json.loads(Path('outputs/auto_calibration_20261028/geometry_candidates.json').read_text())
        class Answers:
            rehearsal=False
            def say(self,message):pass
            def confirm(self,*args):pass
            def ask(self,message,*args,**kwargs):
                if '实际允许接触的软胶厚度' in message:return '2'
                if '实际有效胶面尺寸' in message:return '20 10'
                if '实际 39020 的 F_target' in message:return '2'
                if '逐指 F_max(SDK)' in message:return '4'
                if '目标力容差(SDK)' in message:return '.2'
                if 'HOLD 疑似滑动' in message:return '3'
                if '[回车采用' in message or '回车接受以上软件限值' in message:return ''
                raise AssertionError('Unhandled physical input: '+message)
        ui=Answers()
        model={'robot_name':'fixture','tcp_frames':{'right':'right_gripper_base_link'},'geometry_candidates':audit}
        configure_geometry(cfg,model,ui)
        noise={'normal_std':[.001,.001],'dd_noise_q999':.001,
            'rates':{k:30. for k in ('dd_left','dd_right','wrench_raw_left','wrench_raw_right','camera/left_wrist','camera/right_wrist')},
            'stream_stats':{k:{'bytes':800000,'count':10} for k in ('camera/left_wrist','camera/right_wrist')}}
        configure_noise_force(cfg,noise,ui);configure_policies(cfg,noise,ui)
        self.assertEqual(cfg['tactile']['normal_signs'],[-1.,-1.])
        self.assertEqual(cfg['hardware']['public_ros']['sensors']['force_conversion']['normal_signs'],{'left':-1.,'right':-1.})
        cfg['gripper'].update(open_mm=90.,min_mm=22.)
        cfg['hardware']['gripper_width_calibration']=[{'normalized':.4,'width_mm':22.},{'normalized':.5,'width_mm':36.},{'normalized':1.,'width_mm':90.}]
        cfg['hardware']['camera_mounts_declared']=True
        cfg['close']['width_resolution_mm']=.05
        cfg['safety'].update(workspace_min_mm=[-1000]*3,workspace_max_mm=[1000]*3,table_z_mm=0.)
        T=np.eye(4);T[:3,:3]=axis_basis(cfg['axes']).T;T[:3,3]=[300,0,40]
        reader=Mock();reader.pose.return_value=T
        dimensions={part:{'diameter_mm':d,'height_mm':h} for part,d,h in [('body',36,65),('neck',22,15),('cap',28,15)]}
        locate_bottle(cfg,dimensions,reader,ui,0.)
        cfg['preparation']['axis_test_pose']=T.tolist();cfg['stop_test']['direction_base']=[1.,0,0]
        self.assertEqual(unresolved(cfg),[])
        self.assertEqual(validate_config(cfg),[])
        require_real_ready(cfg,preparing=True)
        self.assertFalse(cfg['demo']);self.assertEqual(cfg['force']['unit'],'sdk')
        self.assertEqual(cfg['tool']['grasp_center_in_tcp'][2][3],117)

    def test_real_visual_plan_count_preserved_across_orientation_blocks(self):
        self.cfg['visual']['count']=5
        self.cfg['wizard']={'visual_orientations_deg':[0.,90.]}
        fake=lambda cfg,T0:[{'point_id':'old'} for _ in range(cfg['visual']['count'])]
        with patch('bottle_scan.wizard_real.visual_points',side_effect=fake):plan=full_plan(self.cfg,np.asarray(self.cfg['T0']),'visual')
        self.assertEqual(len(plan),5);self.assertEqual(len({p['point_id'] for p in plan}),5)
        self.assertEqual([p['orientation_block'] for p in plan],[0,0,0,1,1])


if __name__=='__main__':unittest.main()
