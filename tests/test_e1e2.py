"""Offline protocol/safety tests with fake I/O. No ROS, sockets or actuators."""
import argparse
import copy
import inspect
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import numpy as np
import yaml
from bottle_scan.e1e2_core import defaults,validate,plan_trials,MotionBounds,describe,Correction
from bottle_scan.e1e2_analysis import fit_e1,wilson,rates,tolerance,gaussian_weighted,analyze
from bottle_scan.e1e2 import Runner,main,resolve_config
from bottle_scan.e1e2_hardware import NormalizedGripper,Hardware
from bottle_scan.session import atomic_json


def config():
    c=defaults();c['gripper']['min_normalized']=.336;return c


def fit():
    return {'schema':'e1e2_fit_v1','usable_for_B':True,'fingers':[
        {'coordinate':'u','slope_px_per_mm':2.,'intercept_px':50.,'direction_sign':1},
        {'coordinate':'v','slope_px_per_mm':-3.,'intercept_px':40.,'direction_sign':-1}]}


def desc(x=50.,y=40.,edges=None):
    es=dict(left=False,right=False,top=False,bottom=False);es.update(edges or {})
    return {'valid':True,'centroid_px':[x,y],'touches_edge':any(es.values()),'edge_sides':es,
            'area_px':50,'bbox_xyxy_px':[30,20,60,50],'principal_angle_deg':0.}


class CoreTests(unittest.TestCase):
    def test_defaults_resolve_without_profile(self):validate(config())
    def test_reject_invalid_numbers_and_limits(self):
        for path,value in [(('force','max'),float('nan')),(('motion','max_grasp_offset_mm'),26),
                           (('motion','lift_mm'),61),(('motion','pre_mm'),41),(('design','reps'),3),
                           (('servo','max_regrasps'),4),(('force','normal_signs'),[0,-1])]:
            c=config();c[path[0]][path[1]]=value
            with self.subTest(path=path),self.assertRaises(ValueError):validate(c)
    def test_plan_balanced_blocked_reproducible(self):
        c=config();plan=plan_trials(c,'A');self.assertEqual(plan,plan_trials(c,'A'))
        scans=[r for r in plan if r['kind']=='scan'];self.assertEqual(len(scans),90)
        for d in c['design']['offsets_mm']:
            for b in ('light','heavy'):self.assertEqual(sum(x['delta_mm']==d and x['bottle']==b for x in scans),5)
        self.assertEqual(sum(r['kind']=='zero_monitor' for r in plan),9)
        bottles=[r['bottle'] for r in plan];self.assertEqual(bottles,sorted(bottles,key=lambda b:0 if b=='light' else 1))
    def test_retreat_and_lift_separate_from_grasp_bounds(self):
        T=np.eye(4);T[:3,:3]=[[0,0,1],[0,1,0],[-1,0,0]]
        b=MotionBounds(config(),T);t=b.grasp(20)
        self.assertIsNone(b.check(t));self.assertIsNone(b.check(b.pre(t),'retreat',t))
        self.assertIsNone(b.check(b.lift(t),'lift',t))
        self.assertIsNotNone(b.check(b.grasp(26)))
        self.assertIsNotNone(b.check(b.pre(t),'grasp')) if np.linalg.norm(b.pre(t)[:3,3])>25 else None
        too=b.lift(t);too[2,3]+=21;self.assertIsNotNone(b.check(too,'lift',t))
    def test_height_applies_even_to_retreat(self):
        b=MotionBounds(config(),np.eye(4))
        self.assertIn('高度',b.check(b.pre(b.grasp(0)),'retreat',b.grasp(0)))
    def test_horizontal_a_has_valid_default_targets(self):
        T=np.eye(4);T[:3,:3]=[[0,0,1],[0,1,0],[-1,0,0]]
        b=MotionBounds(config(),T)
        for d in [-20,0,20]:
            g=b.grasp(d)
            self.assertIsNone(b.check(g));self.assertIsNone(b.check(b.pre(g),'retreat',g));self.assertIsNone(b.check(b.lift(g),'lift',g))
    def test_rotation_is_never_sent(self):
        b=MotionBounds(config(),np.eye(4));p=np.eye(4);p[0,0]=-1;p[1,1]=-1
        self.assertIsNotNone(b.check(p))
    def test_full_descriptors(self):
        dd=np.zeros((20,30,3),np.float32);dd[5:10,8:15,0]=1
        d=describe(dd,np.zeros_like(dd),config())
        self.assertEqual(d['bbox_xyxy_px'],[8,5,14,9]);self.assertEqual(d['area_px'],35)
        self.assertAlmostEqual(d['centroid_px'][0],11.);self.assertFalse(d['touches_edge'])
        dd[5:10,:3,0]=1;self.assertTrue(describe(dd,np.zeros_like(dd),config())['edge_sides']['left'])
    def test_servo_error_in_equivalent_mm_and_no_delta_input(self):
        corr=Correction(fit(),config()['servo'])
        d=[desc(60,40),desc(50,25)]
        decision=corr.decide('contact',d,True)
        self.assertEqual(decision['e_mm'],5.);self.assertEqual(decision['delta_mm'],-5.)
        self.assertNotIn('delta',inspect.signature(corr.decide).parameters)
    def test_servo_acceptance_edges_empty_stall_and_clamp(self):
        c=Correction(fit(),config()['servo'])
        self.assertTrue(c.decide('contact',[desc(),desc()],True)['accepted'])
        self.assertEqual(c.decide('no_contact',[],False)['delta_mm'],8)
        self.assertEqual(c.decide('stall_no_contact',[],False)['delta_mm'],-8)
        self.assertEqual(c.decide('contact',[desc(edges={'right':True}),desc()],True)['delta_mm'],-6)
        self.assertEqual(c.decide('contact',[desc(),desc(edges={'bottom':True})],True)['delta_mm'],6)
        self.assertEqual(c.decide('contact',[desc(150),desc(y=-110)],True)['delta_mm'],-15)
    def test_unusable_fit_rejected(self):
        f=fit();f['usable_for_B']=False
        with self.assertRaises(ValueError):Correction(f,config()['servo'])
    def test_body_contact_floor_no_station_profile(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);record=root/'body.json';atomic_json(record,{'release_complete':True,'recording_complete':True,'result':{'normalized':.366}})
            cfg=defaults();cfg['gripper']['body_contact_record']=str(record);cp=root/'config.yaml';cp.write_text(yaml.safe_dump(cfg))
            args=argparse.Namespace(out=root/'session',config=cp,cond='A',offsets=None,reps=None,seed=None,min_normalized=None,target=None,f_max=None,trial_run=False,fit=None)
            c,_=resolve_config(args,lambda _:self.fail('must reuse body'),lambda _:None)
            self.assertAlmostEqual(c['gripper']['min_normalized'],.336)
    def test_floor_prompt_and_grid_rounding(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);cfg=defaults();cfg['gripper']['body_contact_record']=str(root/'missing');cp=root/'c.yaml';cp.write_text(yaml.safe_dump(cfg))
            args=argparse.Namespace(out=root/'s',config=cp,cond='A',offsets=None,reps=None,seed=None,min_normalized=None,target=None,f_max=None,trial_run=False,fit=None)
            c,_=resolve_config(args,lambda _:'.3355',lambda _:None);self.assertEqual(c['gripper']['min_normalized'],.336)
    def test_import_and_help_do_not_import_ros(self):
        subprocess.run([sys.executable,'-c',"import sys;sys.modules['rospy']=None;sys.modules['zmq']=None;import e1e2;import analyze_e1e2"],check=True,capture_output=True)
        subprocess.run([sys.executable,'e1e2.py','--help'],check=True,capture_output=True)


class AnalysisTests(unittest.TestCase):
    def rows(self):
        rng=np.random.default_rng(123);out=[]
        for d in [-10,-5,0,5,10]:
            for rep in range(6):
                ds=[desc(50+2*d+rng.normal(0,.1),40),desc(50,40-3*d+rng.normal(0,.1))]
                out.append({'id':str(len(out)),'delta_mm':d,'bottle':'light' if rep<3 else 'heavy','cond':'A','kind':'scan',
                    'recording_complete':True,'analysis_recording_complete':True,'label':'normal','probe_initial':{'valid':True,'descriptors':ds},
                    'probe_final':{'valid':True},'bilateral':True,'lifted':True,'hold_complete':True,'dropped':False})
        return out
    def test_fit_selects_real_axes_and_noise_resolution(self):
        r=fit_e1(self.rows());self.assertTrue(r['usable_for_B']);a,b=r['fingers']
        self.assertEqual(a['coordinate'],'u');self.assertEqual(b['coordinate'],'v')
        self.assertAlmostEqual(a['slope_px_per_mm'],2,places=2);self.assertEqual(b['direction_sign'],-1)
        self.assertGreater(a['resolution_1sigma_mm'],0);self.assertEqual(a['linear_range_mm'],[-10,10])
    def test_fit_excludes_B_and_edges(self):
        rows=self.rows()
        for r in rows:r['cond']='B'
        self.assertFalse(fit_e1(rows)['usable_for_B'])
        rows=self.rows()
        for r in rows:r['probe_initial']['descriptors'][0]['touches_edge']=True
        self.assertFalse(fit_e1(rows)['usable_for_B'])
    def test_no_linear_segment_no_fake_fit(self):
        rows=self.rows()[:6];self.assertFalse(fit_e1(rows)['usable_for_B'])
    def test_wilson_extremes_and_zero(self):
        self.assertEqual(wilson(0,0),(None,None));lo,hi=wilson(10,10);self.assertLess(lo,1);self.assertAlmostEqual(hi,1)
        lo,hi=wilson(0,10);self.assertAlmostEqual(lo,0);self.assertGreater(hi,0)
    def test_rate_denominator_and_drop_definition(self):
        r=self.rows();r[0]['dropped']=True;r[1]['label']='invalid';r[2]['analysis_recording_complete']=False
        group=next(x for x in rates(r) if x['bottle']=='all' and x['delta_mm']==-10)
        self.assertEqual(group['n'],4);self.assertEqual(group['successes'],3)
    def test_tolerances_keep_direction_and_censoring(self):
        rr=[{'cond':'A','bottle':'all','delta_mm':d,'success_rate':p} for d,p in [(-10,.7),(-5,.9),(0,1),(5,.9),(10,.9)]]
        out=tolerance(rr,'A');self.assertEqual(out['negative_mm'],-10);self.assertIsNone(out['positive_mm']);self.assertIsNone(out['conservative_first_drop_abs_mm'])
    def test_gaussian_mass_and_unobserved_tail(self):
        rr=[dict(cond='A',bottle='all',delta_mm=d,success_rate=1,ci_low=.8,ci_high=1) for d in [-10,0,10]]
        weighted=gaussian_weighted(rr);self.assertEqual(len(weighted),7)
        for row in weighted:
            self.assertAlmostEqual(row['weighted_success_conditional'],1)
            self.assertLess(row['observed_gaussian_mass'],1)
            self.assertAlmostEqual(row['unconditional_upper_bound'],1)
    def test_full_analysis_creates_artifacts_without_robot(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);c=config();meta={'schema':'e1e2_session_v1','simulated':False,'config':c,'cond':'A'}
            atomic_json(root/'session_meta.json',meta)
            for i,row in enumerate(self.rows()):
                row.update(complete=True,status='hold_complete');path=root/'trials'/str(i)
                atomic_json(path/'meta.json',row);atomic_json(path/'recording_status.json',{'complete':True})
            report=analyze([root],root/'analysis');self.assertTrue(report['e1_fit_usable_for_B'])
            for name in ['e1_fit.yaml','e2_rates.csv','e1_descriptors_left.png','e2_success.png','e2_weighted_success.png']:
                self.assertTrue((root/'analysis'/name).exists())


class FakeHardware:
    """Virtual feedback only; no ROS imports or physical operations."""
    def __init__(self,cfg,statuses=None):
        self.cfg=cfg;self.clock=100.;self.pose_value=np.eye(4);self.pose_value[:3,:3]=[[0,0,1],[0,1,0],[-1,0,0]]
        self.pose_value[2,3]=500.;self.width=1.;self.normal=np.zeros(2);self.events=[];self.mode='hold';self.target=None
        self.zero={'contact_thresholds_sdk':[.05,.05]};self.dd_reference=[np.zeros((20,30,3),np.float32) for _ in range(2)]
        self.writer=None;self.motion_info={};self.stall_started=False;self.statuses=list(statuses or ['contact']);self.touch_index=-1;self.reached=True;self.current_stage='';self.legacy_evidence={}
    def now(self):return int(self.clock*1e9)
    def connect(self):self.events.append(('connect',))
    def close(self):self.events.append(('disconnect',))
    def check(self):pass
    def ownership(self):pass
    def pose(self):return self.pose_value.copy()
    def set_context(self,stage,trial_id=None):self.current_stage=stage;self.events.append(('stage',stage))
    def wait(self,seconds):self.clock+=max(.02,seconds)
    def snapshot(self):
        self.clock+=.02
        if self.mode=='close':
            status=self.statuses[min(max(0,self.touch_index),len(self.statuses)-1)]
            if status=='contact':self.normal[:]=.1
            elif status=='no_contact':self.width=self.cfg['gripper']['min_normalized'];self.normal[:]=0
            else:
                self.normal[:]=0
                if not self.stall_started:self.width-=.001;self.stall_started=True
        elif self.mode=='force':self.normal[:]=self.target
        elif self.mode=='open':self.width=1.;self.normal[:]=0
        stamp=self.now()
        return {'normal':self.normal.copy(),'width':self.width,'width_stamp':stamp,'width_sequence':stamp,'force_sequences':[stamp,stamp],
                'force_stamps':[stamp,stamp],'wrench_zeroed':np.tile([0,1,0,0,0,0],(2,1)).astype(float),
                'timestamp_ns':stamp,'pose':self.pose(),'gripper_motion':{**self.motion_info,'mode':self.mode}}
    def open(self,**kw):self.events.append(('open',));self.mode='open';self.width=1.;self.normal[:]=0
    def hold(self):self.events.append(('hold',));self.mode='hold'
    def close_to_floor(self):
        self.events.append(('close',));self.mode='close';self.touch_index+=1;self.stall_started=False
        self.motion_info={'requested_ns':self.now(),'start_normalized':self.width,'first_command_ns':self.now(),
                          'target_changed_ns':self.now(),'target_normalized':max(self.cfg['gripper']['min_normalized'],self.width-.001)}
    def pause_probe(self):self.events.append(('pause_probe',));self.mode='hold'
    def probe_force(self,target,tolerance):self.force(target,tolerance)
    def recontact(self,timeout_s):self.events.append(('recontact',timeout_s));self.mode='recontact'
    def force(self,target,tolerance):self.events.append(('force',target));self.mode='force';self.target=target
    def move(self,target,speed):self.events.append(('move',np.asarray(target).copy(),speed));self.pose_value=np.asarray(target).copy()
    def motion_done(self):return True
    def stop_arm(self):self.events.append(('stop_arm',))
    def recovery_open(self):self.events.append(('recovery_open',));self.open()
    def keyframes(self,reason,gel=False):self.events.append(('keyframe',reason,gel));return {'reason':reason}
    def probe_window(self,since):
        dd=np.zeros((20,30,3),np.float32);dd[6:12,10:18,0]=1
        return [{'median_dd':dd,'frame_ns':[since+i for i in range(5)],'frame_fids':list(range(5))} for _ in range(2)]
    def begin_record(self,path):self.writer=Path(path);self.writer.mkdir(parents=True,exist_ok=True)
    def end_record(self,error=None):
        if self.writer:atomic_json(self.writer/'recording_status.json',{'complete':not error});self.writer=None
    def zero_unloaded(self):return self.zero


class RunnerTests(unittest.TestCase):
    def runner(self,root,cond='A',statuses=None,extra=None):
        c=config();c['design'].update(step_first_trials=0,offsets_mm=[0.],reps=2)
        if extra:c=extra
        hw=FakeHardware(c,statuses);args=argparse.Namespace(out=Path(root),cond=cond,step=False,limit=None)
        r=Runner(args,c,fit() if cond=='B' else None,hw,ask=lambda _:'1',say=lambda _:None)
        r.T0=hw.pose();r.bounds=MotionBounds(c,r.T0)
        r.meta={'T0':r.T0,'config_identity':'test','plan':plan_trials(c,cond)}
        return r,hw
    def test_A_contact_full_flow_and_summary(self):
        with tempfile.TemporaryDirectory() as td:
            r,hw=self.runner(td)
            with patch('bottle_scan.e1e2.time',types.SimpleNamespace(monotonic=lambda:hw.clock,time_ns=lambda:hw.now())):
                r.trial(r.meta['plan'][0])
            self.assertTrue(r.current['complete']);self.assertTrue(r.current['lifted']);self.assertTrue(r.current['hold_complete'])
            self.assertFalse(r.progress['airborne']);self.assertEqual(r.current['label'],'normal')
            self.assertTrue((Path(td)/'summary.csv').exists())
            keys=[e[1] for e in hw.events if e[0]=='keyframe'];self.assertIn('PROBE_last',keys);self.assertIn('HOLD_mid',keys)
    def test_A_no_contact_and_stall_never_lift(self):
        for status in ['no_contact','stall_no_contact']:
            with self.subTest(status=status),tempfile.TemporaryDirectory() as td:
                r,hw=self.runner(td,statuses=[status])
                with patch('bottle_scan.e1e2.time',types.SimpleNamespace(monotonic=lambda:hw.clock,time_ns=lambda:hw.now())):r.trial(r.meta['plan'][0])
                self.assertEqual(r.current['status'],status);self.assertFalse(r.current['lifted'])
                self.assertNotIn(('stage','LIFT'),hw.events)
    def test_height_skip_does_not_open_or_move(self):
        with tempfile.TemporaryDirectory() as td:
            r,hw=self.runner(td);r.T0=np.eye(4);r.bounds=MotionBounds(r.cfg,r.T0);r.trial(r.meta['plan'][0])
            self.assertEqual(r.current['status'],'skipped_bound');self.assertFalse(any(e[0] in ('open','move','close') for e in hw.events))
    def test_B_exhausted_no_contact_skips_lift_and_records_all_retries(self):
        with tempfile.TemporaryDirectory() as td:
            r,hw=self.runner(td,cond='B',statuses=['no_contact']*5)
            # Start -8mm so three +8mm retries remain inside the 25mm bound.
            item=dict(r.meta['plan'][0],delta_mm=-8.)
            with patch('bottle_scan.e1e2.time',types.SimpleNamespace(monotonic=lambda:hw.clock,time_ns=lambda:hw.now())):r.trial(item)
            self.assertEqual(r.current['regrasps'],3);self.assertTrue(r.current['servo_exhausted']);self.assertFalse(r.current['lifted'])
            self.assertEqual(len(r.current['correction_history']),4)
            self.assertTrue(all(x['delta_mm']==8 for x in r.current['correction_history']))
    def test_interrupt_policy_pre_lift_open_airborne_hold(self):
        for airborne in [False,True]:
            with tempfile.TemporaryDirectory() as td:
                r,hw=self.runner(td);r.engaged=True;r.airborne=airborne;r.recover(KeyboardInterrupt())
                self.assertIn(('stop_arm',),hw.events)
                self.assertEqual(('recovery_open',) in hw.events,not airborne)
                self.assertEqual(r.progress['unsafe_recovery'],airborne)
    def test_motion_splits_final50_and_keeps_orientation(self):
        with tempfile.TemporaryDirectory() as td:
            r,hw=self.runner(td);hw.pose_value[0,3]=-100
            target=r.bounds.grasp(0)
            r.move(target)
            moves=[e for e in hw.events if e[0]=='move']
            self.assertEqual([e[2] for e in moves],[20,5])
            self.assertAlmostEqual(np.linalg.norm(moves[0][1][:3,3]-target[:3,3]),50)
            self.assertTrue(all(np.allclose(e[1][:3,:3],r.T0[:3,:3]) for e in moves))
    def test_fully_done_resume_does_not_construct_hardware(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);c=config();c['design'].update(offsets_mm=[0.],reps=2)
            plan=plan_trials(c,'A');atomic_json(root/'session_meta.json',{'schema':'e1e2_session_v1','cond':'A','config':c,'plan':plan,'fit':None})
            for row in plan:atomic_json(root/'trials'/row['id']/'meta.json',{**row,'complete':True,'recording_complete':True})
            result=main(['--cond','A','--out',str(root)],hardware_factory=lambda _:self.fail('must not connect'),output=lambda _:None)
            self.assertEqual(result,0)


class GripperTests(unittest.TestCase):
    class Ros:
        def __init__(self):self.values=[];self.core=types.SimpleNamespace(is_initialized=lambda:True)
        def Publisher(self,*args,**kw):return types.SimpleNamespace(publish=lambda m:self.values.append(m.data),get_num_connections=lambda:1,unregister=lambda:None)
        def Subscriber(self,*args,**kw):return types.SimpleNamespace(unregister=lambda:None)
    def gripper(self):
        self.clock=1_000_000_000;self.ros=self.Ros();self.normal=[0.,0.]
        def force():return {'normal':self.normal,'sequence':[self.clock,self.clock],'stamps':[self.clock,self.clock]}
        def step(c,mean,actual):
            if abs(mean-c['target_force'])<=c['deadband']:return 'hold',None
            v=actual+(-1 if mean<c['target_force'] else 1)
            if v<c['position_min'] or v>c['position_max']:raise RuntimeError('position_limit')
            return 'close',v
        g=NormalizedGripper(config(),force,clock_ns=lambda:self.clock,rospy_module=self.ros,message_type=lambda:types.SimpleNamespace(),legacy=types.SimpleNamespace(tag_step=step))
        g.connect();g._on_state(types.SimpleNamespace(data=.5));return g
    def test_connect_never_publishes(self):
        g=self.gripper();self.assertEqual(self.ros.values,[]);g.close();self.assertEqual(self.ros.values,[])
    def test_closed_targets_never_cross_floor(self):
        g=self.gripper();g.close_to_floor()
        for _ in range(220):
            self.clock+=50_000_000;g._on_state(types.SimpleNamespace(data=self.ros.values[-1] if self.ros.values else .5));g.tick()
        self.assertTrue(self.ros.values);self.assertGreaterEqual(min(self.ros.values),.336)
        self.assertTrue(all(abs(v*1000-round(v*1000))<1e-7 for v in self.ros.values))
    def test_force_max_cancels_closing_and_only_opens(self):
        g=self.gripper();g.close_to_floor();self.normal=[16.,0.];self.clock+=50_000_000;g.tick()
        self.assertTrue(g.overforce);self.assertEqual(g.mode,'open')
        for _ in range(10):
            self.clock+=50_000_000;g._on_state(types.SimpleNamespace(data=self.ros.values[-1]));g.tick()
        self.assertTrue(all(v>=.5 for v in self.ros.values))
    def test_force_at_floor_holds_until_runner_timeout(self):
        g=self.gripper();g._on_state(types.SimpleNamespace(data=.336));self.normal=[.1,.1];g.force(1.5,.25);g.tick()
        self.assertEqual(self.ros.values,[])


if __name__=='__main__':unittest.main()

class FaultAndIsolationTests(unittest.TestCase):
    def test_correction_output_cannot_depend_on_logged_delta(self):
        correction=Correction(fit(),config()['servo']);inputs=[desc(60),desc(y=25)]
        outputs=[]
        for logged_delta in [-20.,0.,20.]:
            metadata={'delta_mm':logged_delta}
            outputs.append(correction.decide('contact',inputs,True))
        self.assertEqual(outputs[0],outputs[1]);self.assertEqual(outputs[1],outputs[2])
    def test_force_not_reached_never_enters_lift(self):
        with tempfile.TemporaryDirectory() as td:
            helper=RunnerTests();r,hw=helper.runner(td)
            original=hw.force
            def never_target(target,tolerance):
                original(target,tolerance)
                if target>2:hw.target=.2
            hw.force=never_target
            with patch('bottle_scan.e1e2.time',types.SimpleNamespace(monotonic=lambda:hw.clock,time_ns=lambda:hw.now())):r.trial(r.meta['plan'][0])
            self.assertEqual(r.current['status'],'force_not_reached');self.assertFalse(r.current['lifted'])
    def test_step_prompts_before_motion_and_interrupt_stops_before_record_close(self):
        with tempfile.TemporaryDirectory() as td:
            helper=RunnerTests();r,hw=helper.runner(td);r.args.step=True
            def ask(msg):
                if 'APPROACH' in msg:raise KeyboardInterrupt('operator')
                return ''
            r.ask=ask
            old_end=hw.end_record
            def end(error=None):
                hw.events.append(('record_close',));old_end(error)
            hw.end_record=end
            with patch('bottle_scan.e1e2.time',types.SimpleNamespace(monotonic=lambda:hw.clock,time_ns=lambda:hw.now())):
                with self.assertRaises(KeyboardInterrupt):r.trial(r.meta['plan'][0])
            keys=[e[0] for e in hw.events]
            self.assertLess(keys.index('stop_arm'),keys.index('record_close'))
            self.assertLess(keys.index('recovery_open'),keys.index('record_close'))
            self.assertFalse(r.progress['unsafe_recovery'])
    def test_airborne_interrupt_holds_before_record_close(self):
        with tempfile.TemporaryDirectory() as td:
            helper=RunnerTests();r,hw=helper.runner(td)
            old_key=hw.keyframes
            def key(reason,gel=False):
                if reason=='HOLD_start':raise KeyboardInterrupt('operator')
                return old_key(reason,gel)
            hw.keyframes=key
            with patch('bottle_scan.e1e2.time',types.SimpleNamespace(monotonic=lambda:hw.clock,time_ns=lambda:hw.now())):
                with self.assertRaises(KeyboardInterrupt):r.trial(r.meta['plan'][0])
            self.assertTrue(r.progress['unsafe_recovery']);self.assertTrue(r.progress['airborne'])
            self.assertNotIn(('recovery_open',),hw.events)
    def test_force_limit_release_overrides_airborne_interrupt_hold(self):
        with tempfile.TemporaryDirectory() as td:
            helper=RunnerTests();r,hw=helper.runner(td);r.engaged=True;r.airborne=True;hw.overforce=True
            r.recover(RuntimeError('F_max'));self.assertIn(('recovery_open',),hw.events)
            self.assertTrue(r.progress['unsafe_recovery'])
    def test_continuous_images_filtered_but_numeric_sources_kept(self):
        from bottle_scan.backend import Sample
        h=Hardware(config());events=[Sample('camera/right_wrist',b'jpeg',1),Sample('gel_raw_left',np.zeros((3,4,3),np.uint8),2),
            Sample('dd_encoded_left',np.ones((2,3,3),np.uint16),3),Sample('dd_left',np.ones((2,3,3),np.float32),3),
            Sample('wrench_raw_left',np.ones(6),4)]
        h.sensors=types.SimpleNamespace(drain_samples=lambda:events)
        h.arm=types.SimpleNamespace(drain_samples=lambda:[]);h.grip=types.SimpleNamespace(drain_samples=lambda:[],armed=False)
        stored=[];h.writer=types.SimpleNamespace(h5=types.SimpleNamespace(append=lambda *args,**kwargs:stored.append(args[0])))
        h._drain()
        self.assertEqual(stored,['dd_encoded_left','dd_left','wrench_raw_left'])
        self.assertEqual(h.cameras['right_wrist'].data,b'jpeg');self.assertIn('left',h.gel)
    def test_actual_deployed_legacy_force_function_is_reusable(self):
        from bottle_scan.hardware_gripper import load_legacy_force_functions
        cfg=yaml.safe_load(Path('config.ros.yaml').read_text());path=Path(cfg['hardware']['public_ros']['force_source_path'])
        if not path.exists():self.skipTest('only available in 5090 read-only source tree')
        legacy=load_legacy_force_functions(path)
        c={'position_min':336.,'position_max':1000.,'step_size':1,'target_force':5.,'deadband':.5}
        self.assertEqual(legacy.tag_step(c,4.,500.),('close',499.))
        self.assertEqual(legacy.tag_step(c,5.,500.),('hold',None))
    def test_gripper_lost_contact_stops_force_chasing(self):
        helper=GripperTests();g=helper.gripper();helper.normal=[.1,.1];g.force(1.5,.25)
        helper.normal=[0.,.1];g.tick();self.assertEqual(g.mode,'hold');self.assertEqual(helper.ros.values,[.5])
    def test_failed_record_drain_still_closes_writer(self):
        h=Hardware(config());closed=[]
        def bad():raise RuntimeError('disk error')
        h._drain=bad;h.writer=types.SimpleNamespace(close=lambda error=None:closed.append(error))
        with self.assertRaises(RuntimeError):h.end_record()
        self.assertIsNone(h.writer);self.assertEqual(closed,['disk error'])

class NativeEventAndBTests(unittest.TestCase):
    def native_hardware(self,events):
        h=Hardware(config());h.zero={'offset_raw':[[0]*6]*2,'contact_thresholds_sdk':[.05,.05]}
        h.sensors=types.SimpleNamespace(drain_samples=lambda:events)
        calls=[]
        h.arm=types.SimpleNamespace(drain_samples=lambda:[],stop_arm=lambda:calls.append('arm_stop'))
        h.grip=types.SimpleNamespace(drain_samples=lambda:[],armed=True,mode='force',overforce=False,
            hold=lambda:calls.append('hold'),open=lambda **kw:calls.append('open'))
        return h,calls
    def test_native_burst_peak_is_not_hidden_by_later_normal_frame(self):
        from bottle_scan.backend import Sample
        # Compression is negative Fz in source frames.
        e=[Sample('wrench_raw_left',np.array([0.,0.,-16.,0.,0.,0.]),1),
           Sample('wrench_raw_left',np.zeros(6),2)]
        h,calls=self.native_hardware(e);h._drain()
        self.assertTrue(h.overforce);self.assertEqual(calls[:3],['arm_stop','hold','open'])
    def test_native_brief_HOLD_drop_is_latched(self):
        from bottle_scan.backend import Sample
        e=[Sample('wrench_raw_left',np.zeros(6),1,metadata={'stage':'HOLD','trial_id':'1'}),
           Sample('wrench_raw_left',np.array([0.,0.,-5.,0.,0.,0.]),2,metadata={'stage':'HOLD','trial_id':'1'})]
        h,calls=self.native_hardware(e);h.trial_id='1';h._drain();self.assertTrue(h.hold_drop_seen)
    def test_B_exhaustion_with_contact_and_target_still_lifts(self):
        with tempfile.TemporaryDirectory() as td:
            r,hw=RunnerTests().runner(td,cond='B',statuses=['contact']*5)
            with patch('bottle_scan.e1e2.time',types.SimpleNamespace(monotonic=lambda:hw.clock,time_ns=lambda:hw.now())):r.trial(r.meta['plan'][0])
            self.assertTrue(r.current['servo_exhausted']);self.assertEqual(r.current['regrasps'],3)
            self.assertTrue(r.current['target_reached']);self.assertTrue(r.current['lifted'])
    def test_B_satisfied_initial_contact_never_regrasps(self):
        with tempfile.TemporaryDirectory() as td:
            r,hw=RunnerTests().runner(td,cond='B')
            f=fit();f['fingers'][0]['intercept_px']=13.5;f['fingers'][1]['intercept_px']=8.5
            r.correction=Correction(f,r.cfg['servo'])
            with patch('bottle_scan.e1e2.time',types.SimpleNamespace(monotonic=lambda:hw.clock,time_ns=lambda:hw.now())):r.trial(r.meta['plan'][0])
            self.assertFalse(r.current['servo_exhausted']);self.assertEqual(r.current['regrasps'],0);self.assertTrue(r.current['lifted'])

class SummaryCrashTests(unittest.TestCase):
    def test_partial_summary_is_visible_but_never_counts_completed(self):
        with tempfile.TemporaryDirectory() as td:
            from bottle_scan.e1e2 import rebuild_summary
            r,hw=RunnerTests().runner(td);item=r.meta['plan'][0]
            atomic_json(Path(td)/'trials'/item['id']/'meta.json',{**item,'complete':False,'status':'failed'})
            rows=rebuild_summary(td);self.assertEqual(len(rows),1);self.assertFalse(rows[0]['complete'])
            r.progress['completed']=[item['id']]
            self.assertEqual(r.completed(),set())
    def test_analysis_reports_archived_interrupted_attempt(self):
        with tempfile.TemporaryDirectory() as td:
            from bottle_scan.e1e2_analysis import load_trials
            root=Path(td);atomic_json(root/'session_meta.json',{'schema':'e1e2_session_v1','simulated':False,'config':config()})
            atomic_json(root/'interrupted'/'00001_old'/'meta.json',{'id':'00001','complete':False,'cond':'A','kind':'scan','delta_mm':0.,'bottle':'light','status':'failed'})
            rows=load_trials([root]);self.assertEqual(len(rows),1);self.assertTrue(rows[0]['archived_attempt'])
            self.assertEqual(rates(rows),[])

class MeasuredBoundTests(unittest.TestCase):
    def test_observed_width_below_floor_never_keeps_closing(self):
        helper=GripperTests();g=helper.gripper();g.close_to_floor()
        helper.clock+=50_000_000;g._on_state(types.SimpleNamespace(data=.335))
        with self.assertRaises(RuntimeError):g.tick()
        self.assertEqual(g.mode,'hold');self.assertEqual(helper.ros.values,[.335])

class FullEntryTests(unittest.TestCase):
    def test_interactive_start_to_label_without_station_profile(self):
        with tempfile.TemporaryDirectory() as td:
            c=config();c['design'].update(offsets_mm=[0.],reps=2)
            hw=FakeHardware(c)
            def answer(prompt):
                if '输入 MOVE' in prompt:return 'MOVE'
                if '回车/1正常' in prompt:return '1'
                return ''
            with patch('bottle_scan.e1e2.time',types.SimpleNamespace(monotonic=lambda:hw.clock,time_ns=lambda:hw.now())):
                result=main(['--cond','A','--offsets','0','--reps','2','--min-normalized','.336',
                             '--limit','1','--out',td],hardware_factory=lambda _:hw,input_fn=answer,output=lambda _:None)
            self.assertEqual(result,0)
            meta=json.loads((Path(td)/'session_meta.json').read_text());self.assertIsNotNone(meta['initial_centroids'])
            progress=json.loads((Path(td)/'progress.json').read_text());self.assertEqual(progress['completed'],['00001'])
            self.assertFalse((Path(td)/'station_profile.yaml').exists())

class CloseStartupRegressionTests(unittest.TestCase):
    """Regression plants include the previously observed unchanged 1.000 feedback."""
    def delayed(self,td,delay,*,never=False):
        r,hw=RunnerTests().runner(td,statuses=['contact']);r.current=None
        original_snapshot=hw.snapshot;original_close=hw.close_to_floor
        request={'start':None}
        def close():
            original_close();request['start']=hw.clock
            # The actual command worker starts 50ms after the request.
            hw.motion_info['first_command_ns']=int((hw.clock+.05)*1e9)
            hw.motion_info['target_changed_ns']=int((hw.clock+.05)*1e9)
        def snapshot():
            if request['start'] is not None and hw.mode=='close':
                t=hw.clock-request['start']
                s=original_snapshot()
                s['normal']=np.zeros(2);hw.normal[:]=0
                if not never and t>=delay:
                    hw.width=.999;s['width']=.999
                    hw.normal[:]=.1;s['normal']=hw.normal.copy()
                else:s['width']=1.;hw.width=1.
                return s
            return original_snapshot()
        hw.close_to_floor=close;hw.snapshot=snapshot
        return r,hw
    def test_feedback_arrives_after_old_03s_cutoff_probe_still_succeeds(self):
        for latency in [.34,.60,.90]:
            with self.subTest(latency=latency),tempfile.TemporaryDirectory() as td:
                r,hw=self.delayed(td,latency)
                with patch('bottle_scan.e1e2.time',types.SimpleNamespace(monotonic=lambda:hw.clock,time_ns=lambda:hw.now())):
                    status,probe=r.touch()
                self.assertEqual(status,'contact');self.assertTrue(probe['valid'])
                self.assertNotIn(('move',),hw.events)
    def test_permanent_unresponsive_start_is_not_a_bottle_depth_stall(self):
        with tempfile.TemporaryDirectory() as td:
            r,hw=self.delayed(td,0,never=True);r.baseline_folder=Path(td)
            with patch('bottle_scan.e1e2.time',types.SimpleNamespace(monotonic=lambda:hw.clock,time_ns=lambda:hw.now())):
                with self.assertRaisesRegex(RuntimeError,'闭合起步未响应'):r.touch()
            self.assertEqual(r.last_touch_report['status'],'motion_start_failed')
            self.assertLess(hw.clock-100,1.3)
            self.assertTrue(list(Path(td).glob('touch_*.json')))
    def test_actual_stall_after_motion_still_returns_stall_no_contact(self):
        with tempfile.TemporaryDirectory() as td:
            r,hw=RunnerTests().runner(td,statuses=['stall_no_contact']);r.current=None
            with patch('bottle_scan.e1e2.time',types.SimpleNamespace(monotonic=lambda:hw.clock,time_ns=lambda:hw.now())):status,probe=r.touch()
            self.assertEqual(status,'stall_no_contact');self.assertIsNone(probe)
            self.assertEqual(r.last_touch_report['status'],'stall_no_contact')
            self.assertGreaterEqual(hw.clock-100,2.0)
            self.assertLess(hw.clock-100,2.2)
    def test_same_target_republication_does_not_restart_stall_timer(self):
        helper=GripperTests();g=helper.gripper();g.close_to_floor();rows=[]
        for _ in range(8):
            helper.clock+=50_000_000;g._on_state(types.SimpleNamespace(data=.5));g.tick();rows.append(g.motion_status())
        first=rows[0]['first_command_ns'];changed=rows[1]['target_changed_ns']
        self.assertTrue(all(r['first_command_ns']==first for r in rows))
        self.assertTrue(all(r['target_changed_ns']==changed for r in rows[1:]))
        self.assertEqual(set(helper.ros.values),{.499,.498})
        # One-count lag is accepted; at two counts the command stops advancing.
        self.assertGreater(g.last_command_ns,first)
    def test_missing_new_options_in_old_session_snapshot_is_supported(self):
        c=config();c['gripper'].pop('close_start_timeout_s');c['gripper'].pop('command_response_s')
        validate(c)
    def test_initial_failure_diagnostic_distinguishes_no_contact(self):
        with tempfile.TemporaryDirectory() as td:
            r,hw=RunnerTests().runner(td,statuses=['no_contact']);r.current=None
            with patch('bottle_scan.e1e2.time',types.SimpleNamespace(monotonic=lambda:hw.clock,time_ns=lambda:hw.now())):
                status,probe=r.touch()
            self.assertEqual(status,'no_contact');self.assertAlmostEqual(r.last_touch_report['width_normalized'],.336)
            self.assertEqual(r.last_touch_report['contact_thresholds_sdk'],[.05,.05])


class ProbeRecontactTests(unittest.TestCase):
    def plant(self,td,*,brief=False,recover=True,second_loss=False,floor=False,cond='A'):
        r,hw=RunnerTests().runner(td,cond=cond);r.current=None
        state={'force_calls':0,'epoch':None,'recontact':None,'dd_since':[]}
        original_force=hw.force;original_snapshot=hw.snapshot;original_window=hw.probe_window
        def force(target,tolerance):
            state['force_calls']+=1
            if state['epoch'] is None:state['epoch']=hw.clock
            original_force(target,tolerance)
        def recontact(timeout_s):
            hw.events.append(('recontact',timeout_s));state['recontact']=hw.clock;hw.mode='recontact'
        def snapshot():
            sample=original_snapshot()
            if state['epoch'] is not None:
                if state['recontact'] is not None and hw.mode=='recontact':
                    normal=[.1,.1] if recover and hw.clock-state['recontact']>=.12 else [.1,0.]
                    if recover and hw.clock-state['recontact']>=.34:
                        hw.mode='probe_ready';sample['gripper_motion']['mode']='probe_ready'
                        sample['gripper_motion']['recontact_stable']=True
                    if floor:hw.width=r.cfg['gripper']['min_normalized'];sample['width']=hw.width
                elif state['force_calls']==1:
                    normal=[.1,.1] if brief and hw.clock-state['epoch']>=.15 else [.1,0.]
                elif second_loss:normal=[.1,0.]
                else:normal=[hw.target,hw.target]
                hw.normal[:]=normal;sample['normal']=hw.normal.copy()
            return sample
        def window(since):state['dd_since'].append(since);return original_window(since)
        hw.force=force;hw.recontact=recontact;hw.snapshot=snapshot;hw.probe_window=window
        return r,hw,state
    def run_touch(self,r,hw):
        with patch('bottle_scan.e1e2.time',types.SimpleNamespace(monotonic=lambda:hw.clock,time_ns=lambda:hw.now())):
            return r.touch()
    def test_brief_contact_dip_resumes_force_without_recontact(self):
        with tempfile.TemporaryDirectory() as td:
            r,hw,state=self.plant(td,brief=True);status,probe=self.run_touch(r,hw)
            self.assertEqual(status,'contact');self.assertTrue(probe['valid'])
            self.assertEqual(state['force_calls'],2);self.assertFalse(any(e[0]=='recontact' for e in hw.events))
            self.assertTrue(state['dd_since']);self.assertGreater(state['dd_since'][0],int((state['epoch']+.15)*1e9))
    def test_persistent_loss_recontacts_once_and_resumes_probe(self):
        with tempfile.TemporaryDirectory() as td:
            r,hw,state=self.plant(td);status,probe=self.run_touch(r,hw)
            self.assertEqual(status,'contact');self.assertTrue(probe['valid'])
            self.assertEqual([e for e in hw.events if e[0]=='recontact'],[('recontact',2.)])
            self.assertEqual(state['force_calls'],2);self.assertTrue(all(e[0]!='move' for e in hw.events))
            self.assertIn('recontact_recovered',[e['event'] for e in r.last_touch_report['probe_contact_events']])
    def test_recontact_timeout_is_explicit_and_never_waits_30s(self):
        with tempfile.TemporaryDirectory() as td:
            r,hw,state=self.plant(td,recover=False);status,probe=self.run_touch(r,hw)
            self.assertEqual(status,'probe_contact_failed');self.assertIsNone(probe)
            self.assertEqual(hw.mode,'hold');self.assertEqual(state['force_calls'],1)
            self.assertLess(hw.clock-state['epoch'],2.5);self.assertIn('2秒',r.last_touch_report['probe_failure'])
    def test_floor_stops_recontact_before_timeout(self):
        with tempfile.TemporaryDirectory() as td:
            r,hw,state=self.plant(td,recover=False,floor=True);status,_=self.run_touch(r,hw)
            self.assertEqual(status,'probe_contact_failed');self.assertIn('下限',r.last_touch_report['probe_failure'])
            self.assertLess(hw.clock-state['epoch'],.5);self.assertEqual(hw.mode,'hold')
    def test_second_persistent_loss_cannot_start_another_recontact(self):
        with tempfile.TemporaryDirectory() as td:
            r,hw,state=self.plant(td,second_loss=True);status,_=self.run_touch(r,hw)
            self.assertEqual(status,'probe_contact_failed')
            self.assertEqual(sum(e[0]=='recontact' for e in hw.events),1);self.assertIn('再次',r.last_touch_report['probe_failure'])
    def test_worker_hold_between_polls_is_rearmed_even_if_force_recovered(self):
        with tempfile.TemporaryDirectory() as td:
            r,hw=RunnerTests().runner(td);r.current=None;original=hw.snapshot;pending={'dip':True}
            def snapshot():
                sample=original()
                if hw.mode=='force' and pending['dip']:
                    pending['dip']=False;hw.mode='hold';sample['gripper_motion']['mode']='hold'
                    sample['normal']=np.array([.1,.1])
                return sample
            hw.snapshot=snapshot;status,probe=self.run_touch(r,hw)
            self.assertEqual(status,'contact');self.assertTrue(probe['valid'])
            self.assertEqual(sum(e[0]=='force' for e in hw.events),2)
    def test_missing_dd_does_not_trigger_recontact_or_fake_success(self):
        with tempfile.TemporaryDirectory() as td:
            r,hw=RunnerTests().runner(td);r.current=None;r.cfg['force']['timeout_s']=.6
            hw.probe_window=lambda since:None;status,probe=self.run_touch(r,hw)
            self.assertEqual(status,'probe_unstable');self.assertIsNone(probe)
            self.assertFalse(any(e[0]=='recontact' for e in hw.events))
    def test_failed_recontact_in_B_skips_servo_and_lift(self):
        with tempfile.TemporaryDirectory() as td:
            r,hw,state=self.plant(td,recover=False,cond='B')
            with patch('bottle_scan.e1e2.time',types.SimpleNamespace(monotonic=lambda:hw.clock,time_ns=lambda:hw.now())):
                r.trial(r.meta['plan'][0])
            self.assertEqual(r.current['status'],'probe_contact_failed');self.assertFalse(r.current['lifted'])
            self.assertEqual(r.current['regrasps'],0);self.assertNotIn(('stage','LIFT'),hw.events)
    def test_squeeze_contact_loss_does_not_enable_probe_recontact(self):
        with tempfile.TemporaryDirectory() as td:
            r,hw,state=self.plant(td,recover=False);r.cfg['force']['timeout_s']=.4
            with patch('bottle_scan.e1e2.time',types.SimpleNamespace(monotonic=lambda:hw.clock,time_ns=lambda:hw.now())):
                result=r.regulate('SQUEEZE',5.,.5,.5)
            self.assertIsNone(result);self.assertFalse(any(e[0]=='recontact' for e in hw.events))
    def test_interrupt_during_recontact_cancels_before_propagating(self):
        with tempfile.TemporaryDirectory() as td:
            r,hw,state=self.plant(td);old_wait=hw.wait
            def wait(seconds):
                if state['recontact'] is not None:raise KeyboardInterrupt('operator')
                old_wait(seconds)
            hw.wait=wait
            with self.assertRaises(KeyboardInterrupt):self.run_touch(r,hw)
            self.assertEqual(hw.mode,'hold');self.assertEqual(sum(e[0]=='recontact' for e in hw.events),1)


class RecontactWorkerTests(unittest.TestCase):
    def test_worker_stops_at_deadline_without_runner(self):
        helper=GripperTests();g=helper.gripper();g.recontact(2.)
        helper.clock+=50_000_000;g.tick();self.assertLess(helper.ros.values[-1],.5)
        helper.clock+=2_000_000_000;g._on_state(types.SimpleNamespace(data=.5));g.tick()
        self.assertEqual(g.mode,'hold');self.assertEqual(helper.ros.values[-1],.5)
        count=len(helper.ros.values);helper.clock+=50_000_000;g.tick();self.assertEqual(len(helper.ros.values),count)
    def test_worker_stops_when_bilateral_contact_returns(self):
        helper=GripperTests();g=helper.gripper();g.recontact(2.);helper.normal=[.1,.1]
        helper.clock+=50_000_000;g.tick();self.assertEqual(g.mode,'recontact');self.assertEqual(helper.ros.values,[.5])
        helper.clock+=210_000_000;g._on_state(types.SimpleNamespace(data=.5));g.tick()
        self.assertEqual(g.mode,'probe_ready');self.assertTrue(g.motion_status()['recontact_stable'])
    def test_recontact_still_enforces_force_limit(self):
        helper=GripperTests();g=helper.gripper();g.recontact(2.);helper.normal=[16.,0.]
        helper.clock+=50_000_000;g.tick();self.assertTrue(g.overforce);self.assertEqual(g.mode,'open')
        self.assertTrue(all(v>=.5 for v in helper.ros.values))
    def test_recontact_never_commands_below_floor(self):
        helper=GripperTests();g=helper.gripper();g._on_state(types.SimpleNamespace(data=.336));g.recontact(2.)
        helper.clock+=50_000_000;g.tick();self.assertFalse(any(v<.336 for v in helper.ros.values))
        self.assertEqual(g.mode,'hold')


class FastCloseAndHandoffTests(unittest.TestCase):
    def tick(self,helper,g,width,ns=50_000_000):
        helper.clock+=ns;g._on_state(types.SimpleNamespace(data=width));g.tick()
    def test_fast_close_uses_larger_step_only_above_500(self):
        helper=GripperTests();g=helper.gripper();g._on_state(types.SimpleNamespace(data=1.));g.close_to_floor()
        self.tick(helper,g,1.);self.assertAlmostEqual(helper.ros.values[-1],.990)
        self.tick(helper,g,.501);self.assertAlmostEqual(helper.ros.values[-1],.500)
        self.tick(helper,g,.500);self.assertAlmostEqual(helper.ros.values[-1],.499)
        g._on_state(types.SimpleNamespace(data=.490));g.close_to_floor()
        self.tick(helper,g,.490);self.assertAlmostEqual(helper.ros.values[-1],.489)
    def test_either_contact_cancels_fast_for_rest_of_CLOSE(self):
        helper=GripperTests();g=helper.gripper();g._on_state(types.SimpleNamespace(data=.8));g.close_to_floor()
        helper.normal=[.1,0.];self.tick(helper,g,.8);self.assertAlmostEqual(helper.ros.values[-1],.799)
        helper.normal=[0.,0.];self.tick(helper,g,.8);self.assertAlmostEqual(helper.ros.values[-1],.798)
        self.assertTrue(g._fast_close_cancelled)
    def test_fast_target_respects_floor_above_switch(self):
        helper=GripperTests();g=helper.gripper();g.cfg['gripper']['min_normalized']=.7
        g._on_state(types.SimpleNamespace(data=.705));g.close_to_floor();self.tick(helper,g,.705)
        self.assertAlmostEqual(helper.ros.values[-1],.7)
    def test_recontact_is_slow_even_above_500(self):
        helper=GripperTests();g=helper.gripper();g._on_state(types.SimpleNamespace(data=.8));g.recontact(2.)
        self.tick(helper,g,.8);self.assertAlmostEqual(helper.ros.values[-1],.799)
    def test_pause_and_probe_force_never_reverse_pending_tiny_close(self):
        helper=GripperTests();g=helper.gripper();g._on_state(types.SimpleNamespace(data=.369));g.close_to_floor()
        self.tick(helper,g,.369);self.assertAlmostEqual(helper.ros.values[-1],.368)
        g.pause_probe();self.assertAlmostEqual(helper.ros.values[-1],.368)
        helper.normal=[.1,.1];g.force(1.5,.25,probe=True)
        self.tick(helper,g,.369,110_000_000);self.assertAlmostEqual(helper.ros.values[-1],.368)
        helper.normal=[0.,.1];self.tick(helper,g,.369)
        self.assertEqual(g.mode,'hold');self.assertTrue(all(abs(v-.368)<1e-9 for v in helper.ros.values))
    def test_probe_pause_does_not_retain_large_fast_request(self):
        helper=GripperTests();g=helper.gripper();g._on_state(types.SimpleNamespace(data=1.));g.close_to_floor()
        self.tick(helper,g,1.);g.pause_probe();self.assertGreaterEqual(helper.ros.values[-1],.998)
    def test_single_force_spike_does_not_finish_recontact(self):
        helper=GripperTests();g=helper.gripper();g._on_state(types.SimpleNamespace(data=.369));g.recontact(2.)
        self.tick(helper,g,.369);helper.normal=[.15,.31];self.tick(helper,g,.369)
        self.assertEqual(g.mode,'recontact');self.assertFalse(g.motion_status()['recontact_stable'])
        self.assertAlmostEqual(helper.ros.values[-1],.368)
        helper.normal=[.026,.30];self.tick(helper,g,.369)
        self.assertEqual(g.mode,'recontact');self.assertIsNone(g._recontact_contact_since_ns)
        helper.normal=[.15,.31];self.tick(helper,g,.368)
        self.tick(helper,g,.368,150_000_000);self.assertFalse(g.motion_status()['recontact_stable'])
        self.tick(helper,g,.368,60_000_000);self.assertTrue(g.motion_status()['recontact_stable'])
        self.assertEqual(g.mode,'probe_ready')
        self.assertLessEqual(helper.ros.values[-1],.368)
    def test_old_force_timestamps_cannot_establish_stable_contact(self):
        helper=GripperTests();g=helper.gripper();helper.normal=[.1,.1];stamp=helper.clock
        g.force_reader=lambda:{'normal':helper.normal,'sequence':[1,1],'stamps':[stamp,stamp]}
        g.recontact(2.);self.tick(helper,g,.5)
        for _ in range(4):self.tick(helper,g,.5,100_000_000)
        self.assertFalse(g.motion_status()['recontact_stable']);self.assertEqual(g.mode,'recontact')
    def test_non_probe_force_loss_still_uses_measured_hold(self):
        helper=GripperTests();g=helper.gripper();g._on_state(types.SimpleNamespace(data=.369));g.close_to_floor()
        self.tick(helper,g,.369);g.force(5.,.5);helper.normal=[0.,.1];self.tick(helper,g,.369)
        self.assertEqual(g.mode,'hold');self.assertAlmostEqual(helper.ros.values[-1],.369)
    def test_old_session_without_fast_options_uses_new_defaults(self):
        helper=GripperTests();g=helper.gripper()
        g.cfg['gripper'].pop('fast_close_until_normalized');g.cfg['gripper'].pop('fast_close_speed_normalized_s')
        validate(g.cfg);g._on_state(types.SimpleNamespace(data=1.));g.close_to_floor();self.tick(helper,g,1.)
        self.assertAlmostEqual(helper.ros.values[-1],.990)


class SlowCloseFeedbackToleranceTests(unittest.TestCase):
    def tick(self,helper,g,width):
        helper.clock+=50_000_000;g._on_state(types.SimpleNamespace(data=width));g.tick()
    def started(self):
        helper=GripperTests();g=helper.gripper()
        g._on_state(types.SimpleNamespace(data=.447));g.close_to_floor()
        self.tick(helper,g,.447)
        self.assertAlmostEqual(helper.ros.values[-1],.446)
        return helper,g
    def test_command_446_accepts_feedback_445_446_and_447(self):
        for feedback in [.445,.446,.447]:
            with self.subTest(feedback=feedback):
                helper,g=self.started();self.tick(helper,g,feedback)
                self.assertAlmostEqual(helper.ros.values[-1],.445)
                self.assertLessEqual(abs(feedback-helper.ros.values[-1]),.002+1e-9)
    def test_two_count_error_waits_without_advancing(self):
        for feedback in [.444,.448]:
            helper,g=self.started();changed=g.motion_status()['target_changed_ns']
            for _ in range(5):self.tick(helper,g,feedback)
            self.assertTrue(all(abs(v-.446)<1e-9 for v in helper.ros.values))
            self.assertEqual(g.motion_status()['target_changed_ns'],changed)
    def test_over_two_counts_stops_closing_in_both_directions(self):
        for feedback in [.443,.449]:
            helper,g=self.started()
            with self.assertRaisesRegex(RuntimeError,'超过2个刻度'):
                self.tick(helper,g,feedback)
            self.assertEqual(g.mode,'hold')
            self.assertAlmostEqual(helper.ros.values[-1],feedback)
    def test_jittering_feedback_plant_keeps_advancing_without_accumulated_error(self):
        helper,g=self.started()
        commands=[helper.ros.values[-1]]
        for noise in [-1,0,1]*15:
            feedback=commands[-1]+noise/1000.
            self.tick(helper,g,feedback);command=helper.ros.values[-1];commands.append(command)
            self.assertLessEqual(abs(command-feedback),.002+1e-9)
        self.assertTrue(all(b<=a+1e-9 for a,b in zip(commands,commands[1:])))
        self.assertLess(commands[-1],.42)
        self.assertGreaterEqual(min(commands),g.cfg['gripper']['min_normalized'])
    def test_one_count_jitter_does_not_hide_two_second_stall(self):
        with tempfile.TemporaryDirectory() as td:
            r,hw=RunnerTests().runner(td,statuses=['stall_no_contact']);r.current=None
            original=hw.snapshot;counter=[0]
            def snapshot():
                s=original()
                if hw.mode=='close':
                    counter[0]+=1;s['width']=.445 if counter[0]%2 else .447
                return s
            hw.snapshot=snapshot
            with patch('bottle_scan.e1e2.time',types.SimpleNamespace(monotonic=lambda:hw.clock,time_ns=lambda:hw.now())):
                status,_=r.touch()
            self.assertEqual(status,'stall_no_contact')
            self.assertGreaterEqual(hw.clock-100,2.)
            self.assertLess(hw.clock-100,2.3)
    def test_old_session_without_tolerance_keys_uses_same_policy(self):
        helper,g=self.started()
        for name in ['position_feedback_tolerance_counts','position_max_tracking_error_counts']:
            g.cfg['gripper'].pop(name)
        validate(g.cfg)
        self.tick(helper,g,.447);self.assertAlmostEqual(helper.ros.values[-1],.445)
        self.tick(helper,g,.447);self.assertAlmostEqual(helper.ros.values[-1],.445)


class RecontactSourceContinuityTests(unittest.TestCase):
    def tick(self,helper,g,ns=50_000_000):
        helper.clock+=ns;g._on_state(types.SimpleNamespace(data=.369));g.tick()
    def test_low_native_frame_between_polls_restarts_stability_clock(self):
        helper=GripperTests();g=helper.gripper();helper.normal=[.1,.1];g.recontact(2.)
        self.tick(helper,g);self.tick(helper,g,150_000_000)
        g.observe_recontact_force(0,0.,helper.clock+1)
        self.tick(helper,g,60_000_000);self.assertFalse(g.motion_status()['recontact_stable'])
        self.tick(helper,g,210_000_000);self.assertTrue(g.motion_status()['recontact_stable'])
    def test_loss_after_ready_cannot_bypass_original_two_second_deadline(self):
        helper=GripperTests();g=helper.gripper();helper.normal=[.1,.1];g.recontact(2.)
        self.tick(helper,g);self.tick(helper,g,210_000_000);self.assertEqual(g.mode,'probe_ready')
        g.observe_recontact_force(0,0.,helper.clock);self.assertEqual(g.mode,'recontact')
        self.assertFalse(g.motion_status()['recontact_stable']);helper.normal=[0.,0.]
        self.tick(helper,g,1_800_000_000);self.assertEqual(g.mode,'hold')
    def test_frame_from_before_recontact_does_not_break_stability(self):
        helper=GripperTests();g=helper.gripper();helper.normal=[.1,.1];g.recontact(2.)
        self.tick(helper,g);g.observe_recontact_force(0,0.,helper.clock-1_000_000_000)
        self.tick(helper,g,210_000_000);self.assertTrue(g.motion_status()['recontact_stable'])


class ErrorbarRoundoffTests(unittest.TestCase):
    def test_wilson_endpoints_are_exact_at_zero_and_all_success(self):
        from bottle_scan.e1e2_analysis import proportion_errorbars
        for n in range(1,101):
            self.assertEqual(wilson(0,n)[0],0.)
            self.assertEqual(wilson(n,n)[1],1.)
            for k in range(n+1):
                lo,hi=wilson(k,n)
                errors=proportion_errorbars([k/n],[lo],[hi])
                self.assertTrue(np.all(errors>=0.));self.assertTrue(lo<=k/n<=hi)
    def test_tiny_negative_roundoff_is_zeroed_but_bad_bounds_fail(self):
        from bottle_scan.e1e2_analysis import proportion_errorbars
        errors=proportion_errorbars([0.,1.],[5e-17,.72],[.28,1.-1e-16])
        self.assertEqual(errors[0,0],0.);self.assertEqual(errors[1,1],0.)
        with self.assertRaises(ValueError):proportion_errorbars([.5],[.6],[.8])
        with self.assertRaises(ValueError):proportion_errorbars([.5],[0.],[float('nan')])
    def test_analysis_renders_ten_repeats_at_zero_and_full_success(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);c=config()
            atomic_json(root/'session_meta.json',{'schema':'e1e2_session_v1','simulated':False,'config':c,'cond':'A'})
            templates=AnalysisTests().rows()
            for delta in [-10.,-5.,0.,5.,10.]:
                template=next(r for r in templates if r['delta_mm']==delta)
                for rep in range(10):
                    row=copy.deepcopy(template);row.update(id=str(delta)+'_'+str(rep),complete=True,status='hold_complete',lifted=delta!=10.,hold_complete=delta!=10.)
                    path=root/'trials'/row['id'];atomic_json(path/'meta.json',row);atomic_json(path/'recording_status.json',{'complete':True})
            report=analyze([root],root/'analysis')
            self.assertTrue((root/'analysis/e2_success.png').exists());self.assertTrue((root/'analysis/e2_contact.png').exists())
            self.assertTrue((root/'analysis/analysis_report.json').exists());self.assertEqual(report['completed_rows'],50)
