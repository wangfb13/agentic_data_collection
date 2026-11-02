"""Interactive independent E1/E2 collector; setup/run remain untouched."""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import csv
import fcntl
import json
from pathlib import Path
import signal
import time
import traceback
import numpy as np
import yaml
from .e1e2_core import (defaults,load_settings,validate,plan_trials,MotionBounds,describe,
                         Correction,LABELS,config_identity)
from .session import atomic_json,plain,fingerprint


class UserExit(Exception):pass


def parser():
    p=argparse.ArgumentParser(description='独立 E1/E2 实机采集；不依赖 setup 完成；导入和 --help 不连接硬件')
    p.add_argument('--cond',choices=['A','B'],required=True)
    p.add_argument('--offsets',nargs='+',type=float,metavar='MM')
    p.add_argument('--reps',type=int,help='每档总次数，轻／重瓶各一半，须为偶数')
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--config',type=Path,default=Path(__file__).resolve().parents[1]/'config.e1e2.yaml')
    p.add_argument('--fit',type=Path,help='B 读取的 e1_fit.yaml；也可配置 servo.fit')
    p.add_argument('--step',action='store_true',help='全部阶段前回车；默认前2次试验已开启')
    p.add_argument('--limit',type=int,help='本次最多执行几次（含插入零点，跳过项不发动作）')
    p.add_argument('--trial-run',action='store_true',help='试跑：-10/-5/0/5/10各2次，仍为实机')
    p.add_argument('--seed',type=int)
    p.add_argument('--min-normalized',type=float)
    p.add_argument('--target',type=float,help='两指均值目标，SDK单位')
    p.add_argument('--f-max',type=float,help='逐指 SDK 法向力上限')
    return p


def resolve_config(args,ask=input,say=print):
    existing=args.out/'session_meta.json'
    if existing.exists():
        meta=json.loads(existing.read_text())
        if meta.get('schema')!='e1e2_session_v1' or meta.get('cond')!=args.cond:raise ValueError('输出目录模式不符或不是 E1/E2 session')
        cfg=meta['config']
        for value,expected,name in [(args.offsets,cfg['design']['offsets_mm'],'offsets'),(args.reps,cfg['design']['reps'],'reps'),
              (args.seed,cfg['design']['seed'],'seed'),(args.min_normalized,cfg['gripper']['min_normalized'],'min-normalized'),
              (args.target,cfg['force']['target'],'target'),(args.f_max,cfg['force']['max'],'f-max')]:
            if value is not None and value!=expected:raise ValueError('续跑不能更换 '+name+'；请新建输出目录')
        fit=meta.get('fit')
        if args.fit and yaml.safe_load(args.fit.read_text())!=fit:raise ValueError('续跑不能替换 E1 拟合')
        if args.trial_run and (cfg['design']['offsets_mm']!=[-10.,-5.,0.,5.,10.] or cfg['design']['reps']!=2):raise ValueError('续跑计划不是试跑计划')
    else:
        cfg=load_settings(args.config);fit=None
        if args.trial_run:cfg['design'].update(offsets_mm=[-10.,-5.,0.,5.,10.],reps=2,step_first_trials=2)
        for attr,sec,key in [('offsets','design','offsets_mm'),('reps','design','reps'),('seed','design','seed'),
                ('min_normalized','gripper','min_normalized'),('target','force','target'),('f_max','force','max')]:
            v=getattr(args,attr)
            if v is not None:cfg[sec][key]=v
        # Read ONLY a completed first-contact number; never require a profile,
        # import the wizard, or modify its calibration/checkpoint files.
        if cfg['gripper']['min_normalized'] is None:
            path=Path(cfg['gripper']['body_contact_record'])
            if path.exists():
                rec=json.loads(path.read_text());result=rec.get('result') or {}
                if rec.get('release_complete') and rec.get('recording_complete') and isinstance(result.get('normalized'),(int,float)):
                    cfg['gripper']['min_normalized']=max(0.,float(result['normalized'])-cfg['gripper']['contact_margin_normalized'])
                    cfg['gripper']['floor_source']={'record':str(path),'contact_normalized':result['normalized'],'margin':cfg['gripper']['contact_margin_normalized']}
                    say(f"复用瓶身初接触读数 {result['normalized']:.3f}；本实验归一化下限={cfg['gripper']['min_normalized']:.3f}")
            if cfg['gripper']['min_normalized'] is None:
                while True:
                    try:
                        value=ask('未找到完整瓶身初接触记录，请输入归一化闭合下限 [0,1)：\n> ').strip()
                        if value.lower()=='q':raise UserExit()
                        value=float(value)
                        if np.isfinite(value) and 0<=value<cfg['gripper']['open_normalized']:break
                    except ValueError:pass
                    say('需要 [0, 张开值) 内的有限数值。')
                cfg['gripper']['min_normalized']=value;cfg['gripper']['floor_source']={'operator_input':True}
        path=args.fit or cfg['servo']['fit']
        if args.cond=='B':
            if not path:raise ValueError('B 需要 --fit E1分析输出的e1_fit.yaml')
            fit=yaml.safe_load(Path(path).read_text());cfg['servo']['fit']=str(path)
    if not existing.exists():
        configured=cfg['gripper']['min_normalized']
        cfg['gripper']['min_normalized']=float(np.ceil(configured*1000.-1e-8)/1000.)
        if abs(configured-cfg['gripper']['min_normalized'])>1e-9:
            cfg['gripper']['requested_min_normalized']=configured
            say('控制步长为0.001；闭合下限向更张开方向取整为 '+str(cfg['gripper']['min_normalized']))
    validate(cfg)
    if args.cond=='B':
        Correction(fit,cfg['servo'])
        if fit.get('axes_tcp') is not None and not all(np.allclose(fit['axes_tcp'][k],cfg['axes_tcp'][k]) for k in ('a','c','l')):
            raise ValueError('E1 拟合的 a/c/l 定义与当前配置不同')
        if fit.get('normal_signs') is not None and fit['normal_signs']!=cfg['force']['normal_signs']:
            raise ValueError('E1 与当前配置的法向力符号不同')
    return cfg,fit


def rebuild_summary(root):
    rows=[]
    for path in sorted((Path(root)/'trials').glob('*/meta.json')):
        meta=json.loads(path.read_text())
        row={k:meta.get(k) for k in ('id','cond','bottle','kind','delta_mm','status','label','note','regrasps','servo_exhausted','bilateral','target_reached','lifted','dropped','complete','recording_complete','skipped_reason')}
        row['contact_valid']=bool(meta.get('probe_initial',{}).get('valid'))
        row['final_contact_valid']=bool(meta.get('probe_final',{}).get('valid'))
        row['success']=bool(meta.get('lifted') and meta.get('hold_complete') and not meta.get('dropped') and meta.get('label')=='normal' and meta.get('recording_complete'))
        rows.append(row)
    fields=['id','cond','bottle','kind','delta_mm','status','label','note','regrasps','servo_exhausted','bilateral','target_reached','lifted','dropped','complete','recording_complete','skipped_reason','contact_valid','final_contact_valid','success']
    p=Path(root)/'summary.csv';tmp=p.with_suffix('.csv.tmp')
    with tmp.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    tmp.replace(p)
    return rows


class Runner:
    def __init__(self,args,cfg,fit,hw,*,ask=input,say=print):
        self.args=args;self.cfg=cfg;self.fit=fit;self.hw=hw;self.ask=ask;self.say=say;self.root=args.out
        self.meta_path=self.root/'session_meta.json';self.progress_path=self.root/'progress.json'
        self.meta=json.loads(self.meta_path.read_text()) if self.meta_path.exists() else None
        self.progress=json.loads(self.progress_path.read_text()) if self.progress_path.exists() else {'completed':[],'active':None,'airborne':False,'unsafe_recovery':False}
        self.current=None;self.touch_count=0;self.stage_step=args.step;self.airborne=bool(self.progress.get('airborne'))
        self.engaged=False;self.quit=False;self.bounds=None;self.folder=None;self.recovery_done=False
        self.correction=Correction(fit,cfg['servo']) if args.cond=='B' else None
    def save(self):
        # Commit the actual trial result before marking its plan entry done.
        if self.current and self.folder:atomic_json(self.folder/'meta.json',self.current)
        if self.meta is not None:atomic_json(self.meta_path,self.meta)
        atomic_json(self.progress_path,self.progress)
    def read(self,message):
        self.hw.check();value=self.ask(message+'\n> ').strip();self.hw.check();return value
    def authorize(self,message,word='MOVE'):
        while self.read(message)!=word:self.say('请输入 '+word+'，或 Ctrl+C 停止。')
    def completed(self):
        # Metadata is committed before progress. Reconcile that narrow crash
        # window; never rerun an already labelled physical trial.
        rows=rebuild_summary(self.root)
        self.progress['completed']=[row['id'] for row in rows if row['complete']]
        self.physical_completed=sum(r['complete'] and r['status']!='skipped_bound' for r in rows)
        return set(self.progress['completed'])
    def initial(self):
        if self.progress.get('unsafe_recovery') or self.progress.get('airborne'):
            raise RuntimeError('上次中断可能仍夹瓶／在空中。请现场安全放回并张开；新建输出目录，旧数据保留。')
        if self.meta and self.meta.get('T0'):
            self.T0=np.asarray(self.meta['T0'],float);self.bounds=MotionBounds(self.cfg,self.T0)
            self.say('续跑保留原 T0 基座 xyz(mm)：'+str(self.T0[:3,3].tolist()))
            self.read('请把瓶子放回原标记，夹爪张开，恢复到上述原 T0（包括原朝向），停止遥操后按回车。此次不重定义 T0。')
            now=self.hw.pose()
            if np.linalg.norm(now[:3,3]-self.T0[:3,3])>self.cfg['motion']['pose_tolerance_mm'] or not np.allclose(now[:3,:3],self.T0[:3,:3],atol=.01):
                raise RuntimeError('当前位姿不是原 T0，请按打印位置恢复后续跑，或使用新目录')
        else:
            self.read('把瓶子放到标记处，夹爪张开，移到理想抓取位（瓶身中部在两指之间、胶面未接触），然后按回车。接下来只在此处轻触，微调时会开爪平移。')
            self.T0=self.hw.pose();self.bounds=MotionBounds(self.cfg,self.T0)
            self.meta={'schema':'e1e2_session_v1','cond':self.args.cond,'config':self.cfg,'fit':self.fit,
                'config_identity':config_identity(self.cfg,self.args.cond,self.fit),'T0':self.T0,
                'plan':plan_trials(self.cfg,self.args.cond),'created_unix_ns':time.time_ns(),
                'simulated':False,'delta_visibility':'planned delta is logged; Correction accepts only tactile/status/bilateral',
                'initial_centroids':None,'zero_history':[],'source_policy':'processed DD, SDK force, keyframe-only images'}
            self.save()
        tilt=np.degrees(np.arccos(np.clip(self.bounds.l[2],-1,1)))
        self.say(f'+l 与基座向上偏差 {tilt:.2f}°（仅提示，不拦截）。a 基座方向='+str(self.bounds.a.tolist()))
        self.read('确认夹爪全开且两块胶面均未接触任何物体，按回车扣零。')
        self.baseline_folder=self.root/'baseline'/str(time.time_ns())
        self.hw.begin_record(self.baseline_folder)
        try:
            self.hw.set_context('ZERO','baseline');zero=self.hw.zero_unloaded()
            np.savez(self.baseline_folder/'dd_reference.npz',left=self.hw.dd_reference[0],right=self.hw.dd_reference[1])
            zero['dd_reference_file']=str(self.baseline_folder/'dd_reference.npz')
            self.meta['zero_history'].append(zero)
            self.say('扣零完成；逐指接触阈值 SDK='+str(zero['contact_thresholds_sdk']))
            self.say('以下在 T0 轻触；接受后回开。a+/a-/l+/l- 会先松开，再平移 1 mm 并重新轻触。')
            while True:
                status,probe=self.touch('BASELINE')
                if status!='contact' or not probe or not probe['valid']:
                    detail=self.last_touch_report
                    self.release()
                    names={'no_contact':'已到闭合下限，仍未双侧接触',
                           'stall_no_contact':'闭合已起步，之后停滞且未双侧接触',
                           'probe_unstable':'已双侧接触，但 PROBE 力或独立 DD 窗口未稳定',
                           'probe_contact_failed':'PROBE 接触恢复失败：'+str(getattr(self,'probe_failure','未知原因')),
                           'contact':'已达到 PROBE 力带，但 DD 接触区描述子无效'}
                    raise RuntimeError('T0 轻触失败：'+names.get(status,status)+
                        f"；开度={detail['width_normalized']:.3f}，法向力={detail['normal_sdk']}，阈值={detail['contact_thresholds_sdk']}。详见 touch 记录。")
                self.say('两指 DD 质心 px：'+str([x['centroid_px'] for x in probe['descriptors']]))
                value=self.read('回车接受；a+ / a- / l+ / l- 微调1 mm；q 停止：').lower()
                self.release()
                if value=='q':raise UserExit()
                if not value:
                    if self.meta['initial_centroids'] is None:
                        self.meta['initial_centroids']=probe;self.meta['T0']=self.T0;self.meta['T0_accepted_unix_ns']=time.time_ns()
                    break
                if value not in ('a+','a-','l+','l-'):self.say('未知输入，重新轻触。');continue
                if self.meta.get('T0_accepted_unix_ns'):
                    raise RuntimeError('续跑不能改 T0，请新建目录以微调')
                direction=self.bounds.a if value[0]=='a' else self.bounds.l
                target=self.T0.copy();target[:3,3]+=direction*(1 if value[1]=='+' else -1)
                self.move(target,'grasp');actual=self.hw.pose();self.T0=target.copy();self.T0[:3,3]=actual[:3,3]
                # Do not accumulate unlimited baseline tweaks: the original
                # bounds remain active until the operator accepts the new T0.
            self.bounds=MotionBounds(self.cfg,self.T0)
            self.hw.keyframes('T0_accepted',gel=True)
            self.meta['T0']=self.T0;self.meta['force_controller']=getattr(self.hw,'legacy_evidence',None)
            self.save()
        except BaseException as exc:
            self.recover(exc)
            try:self.hw.end_record(error=str(exc) or type(exc).__name__)
            except Exception:pass
            raise
        else:self.hw.end_record()
    def print_plan(self,done):
        self.say('全部试验：抓取／纠正≤25mm；退让单独限幅；竖直提起≤60mm；无旋转。')
        for item in self.meta['plan']:
            target=self.bounds.grasp(item['delta_mm']);pre=self.bounds.pre(target);lift=self.bounds.lift(target)
            reason=self.bounds.check(target) or self.bounds.check(pre,'retreat',target) or self.bounds.check(lift,'lift',target)
            v=np.round(target[:3,3]-self.T0[:3,3],3).tolist();pv=np.round(pre[:3,3]-self.T0[:3,3],3).tolist();lv=np.round(lift[:3,3]-self.T0[:3,3],3).tolist()
            self.say(f"{item['id']} {item['bottle']} {item['kind']} δ={item['delta_mm']:+g}mm Δbase抓取={v} 预位={pv} 提起={lv}"+(' [已完成]' if item['id'] in done else '')+(' [跳过：'+reason+']' if reason else ''))
        self.say('速度：自由 '+str(self.cfg['motion']['free_speed_mm_s'])+' mm/s；每段目标最后'+str(self.cfg['motion']['near_distance_mm'])+'mm '+str(self.cfg['motion']['near_speed_mm_s'])+' mm/s。')
    @contextmanager
    def stage(self,name):
        if self.stage_step:self.read('阶段 '+name+'：按回车执行。')
        self.hw.set_context(name,self.current['id'] if self.current else 'baseline')
        entry={'name':name,'start_ns':self.hw.now()}
        if self.current:self.current.setdefault('stages',[]).append(entry);self.save()
        self.hw.keyframes(name+'_start')
        try:
            yield entry
        except BaseException as exc:
            entry['error']=str(exc) or type(exc).__name__
            # Cancel motion/control BEFORE disk finalization or image waits.
            if self.engaged:
                try:self.hw.stop_arm()
                except Exception:pass
                try:self.hw.hold()
                except Exception:pass
            raise
        else:
            self.hw.keyframes(name+'_end')
        finally:
            entry['end_ns']=self.hw.now();self.save()
    def release(self):
        with self.stage('RELEASE'):
            self.engaged=True;self.hw.open();deadline=time.monotonic()+self.cfg['gripper']['timeout_s']
            while True:
                s=self.hw.snapshot()
                if abs(s['width']-self.cfg['gripper']['open_normalized'])<=self.cfg['gripper']['open_tolerance']:break
                if time.monotonic()>deadline:raise RuntimeError('张开超时')
                self.hw.wait(.01)
            self.hw.hold();self.airborne=False;self.progress['airborne']=False;self.save()
    def move(self,target,role='grasp',anchor=None,*,lower_monitor=None):
        error=self.bounds.check(target,role,anchor)
        if error:raise ValueError(error)
        now=self.hw.pose()
        if not np.allclose(now[:3,:3],self.T0[:3,:3],atol=.01):raise RuntimeError('当前姿态已改变；本脚本不发送旋转来恢复')
        distance=np.linalg.norm(target[:3,3]-now[:3,3]);near=self.cfg['motion']['near_distance_mm']
        waypoints=[]
        if distance>near+1e-6:
            mid=target.copy();mid[:3,3]=target[:3,3]-(target[:3,3]-now[:3,3])/distance*near
            waypoints.append((mid,self.cfg['motion']['free_speed_mm_s']))
        waypoints.append((target,self.cfg['motion']['near_speed_mm_s']))
        for waypoint,speed in waypoints:
            self.engaged=True;self.hw.move(waypoint,speed)
            deadline=time.monotonic()+np.linalg.norm(waypoint[:3,3]-self.hw.pose()[:3,3])/speed+self.cfg['motion']['timeout_margin_s']
            while True:
                s=self.hw.snapshot()
                if lower_monitor and lower_monitor(s):self.hw.stop_arm();return 'unloaded'
                if self.hw.motion_done():break
                if time.monotonic()>deadline:raise RuntimeError('机械臂移动超时')
                self.hw.wait(.01)
        return 'reached'
    def bilateral(self,s):return bool(np.all(np.asarray(s['normal'])>=np.asarray(self.hw.zero['contact_thresholds_sdk'])))
    def note_touch(self,status,snapshot,probe=None):
        self.touch_count+=1
        report={'status':status,'timestamp_ns':self.hw.now(),'width_normalized':float(snapshot['width']),
                'normal_sdk':np.asarray(snapshot['normal']).tolist(),
                'contact_thresholds_sdk':list(self.hw.zero['contact_thresholds_sdk']),
                'motion':snapshot.get('gripper_motion'),
                'probe_valid':bool(probe and probe.get('valid')),
                'probe_target_sdk':self.cfg['force']['probe_fraction']*self.cfg['force']['target'],
                'dd_valid_by_finger':[d.get('valid') for d in probe.get('descriptors',[])] if probe else [],
                'probe_contact_events':list(getattr(self,'probe_contact_events',[])),
                'probe_failure':getattr(self,'probe_failure',None),
                'close_policy':{'fast_until_normalized':self.cfg['gripper'].get('fast_close_until_normalized',.5),
                                'feedback_tolerance_counts':self.cfg['gripper'].get('position_feedback_tolerance_counts',1),
                                'max_tracking_error_counts':self.cfg['gripper'].get('position_max_tracking_error_counts',2),
                                'fast_speed_normalized_s':self.cfg['gripper'].get('fast_close_speed_normalized_s',.20),
                                'slow_speed_normalized_s':self.cfg['gripper']['close_speed_normalized_s']}}
        self.last_touch_report=report
        if self.current:
            self.current.setdefault('touch_reports',[]).append(report);self.save()
        elif getattr(self,'baseline_folder',None):
            atomic_json(self.baseline_folder/f'touch_{self.touch_count:03d}.json',report)
        return report
    def touch(self,prefix=''):
        status='no_contact';probe=None;self.engaged=True
        self.probe_contact_events=[];self.probe_failure=None
        with self.stage('CLOSE'):
            self.hw.close_to_floor();start=time.monotonic();rows=[];last_seq=None
            start_width=None;responded=False
            start_timeout=self.cfg['gripper'].get('close_start_timeout_s',1.)
            response_wait=max(self.cfg['gripper']['stall_window_s'],self.cfg['gripper'].get('command_response_s',.5))
            while True:
                s=self.hw.snapshot();now=time.monotonic()
                motion=s.get('gripper_motion') or {}
                if start_width is None:start_width=motion.get('start_normalized',s['width'])
                first_command=motion.get('first_command_ns')
                changed_command=motion.get('target_changed_ns',first_command)
                if s['width']<start_width-self.cfg['gripper']['stall_epsilon_normalized']:responded=True
                if self.bilateral(s) and motion.get('mode')!='handoff_wait':self.hw.pause_probe();status='contact';break
                if s['width']<=self.cfg['gripper']['min_normalized']+1e-6:
                    self.hw.hold();status='no_contact';floor_time=time.monotonic();floor_stamp=s['width_stamp']
                    # Give force frames received after the floor feedback a
                    # short settling interval; do not classify an old unloaded
                    # force sample as the result of the final closing step.
                    while time.monotonic()-floor_time<.2:
                        final=self.hw.snapshot()
                        if min(final['force_stamps'])>=floor_stamp and self.bilateral(final):status='contact';break
                        self.hw.wait(.01)
                    break
                if s['width_sequence']!=last_seq:
                    rows.append((now,s['width']));last_seq=s['width_sequence']
                    cutoff=now-self.cfg['gripper']['stall_window_s']
                    # Keep one boundary sample so a full configured window is observable.
                    while len(rows)>2 and rows[1][0]<=cutoff:rows.pop(0)
                    # +/-1 count feedback jitter spans two counts but is not
                    # continued physical closing. Keep the two-second guard.
                    flat_span=max(self.cfg['gripper']['stall_epsilon_normalized'],2*self.cfg['gripper'].get('position_feedback_tolerance_counts',1)/1000.)
                    flat=len(rows)>1 and rows[-1][0]-rows[0][0]>=self.cfg['gripper']['stall_window_s'] and np.ptp([x[1] for x in rows])<=flat_span+1e-9
                    # Repeated publication of the SAME target is not a new
                    # physical request and must not postpone a real stall.
                    command_waited=changed_command is not None and (self.hw.now()-changed_command)/1e9>=response_wait
                    if responded and flat and command_waited:
                        self.hw.hold();status='stall_no_contact';break
                if not responded:
                    elapsed=(self.hw.now()-first_command)/1e9 if first_command is not None else now-start
                    if elapsed>=start_timeout:
                        self.note_touch('motion_start_failed',s);self.hw.hold()
                        reason='首条闭合指令发出后，实测开度仍未减小' if first_command is not None else '未记录到有效闭合指令'
                        raise RuntimeError(f"夹爪闭合起步未响应：{reason}；等待 {elapsed:.2f}s，实测={s['width']:.3f}，指令={motion.get('target_normalized')}，下限={self.cfg['gripper']['min_normalized']:.3f}。已取消闭合；此状态不作为瓶位过深，不自动重抓。")
                if now-start>self.cfg['gripper']['timeout_s']:self.hw.hold();raise RuntimeError('CLOSE超时，既非有效接触也非稳定停滞')
                self.hw.wait(.01)
        if status=='contact':
            probe=self.regulate('PROBE',self.cfg['force']['probe_fraction']*self.cfg['force']['target'],self.cfg['force']['probe_tolerance'],self.cfg['force']['probe_stable_s'],with_dd=True)
            if probe is None:status='probe_contact_failed' if self.probe_failure else 'probe_unstable'
        self.note_touch(status,self.hw.snapshot(),probe)
        return status,probe
    def probe_event(self,event,s,**extra):
        row={'event':event,'timestamp_ns':self.hw.now(),'width_normalized':float(s['width']),
             'normal_sdk':np.asarray(s['normal']).tolist(),**extra}
        self.probe_contact_events.append(row)
        if self.current:self.current.setdefault('probe_contact_events',[]).append(row)
    def recontact_probe(self):
        start=self.hw.now();deadline=start+2_000_000_000;recovered=False;reason='补触2秒仍未双侧接触'
        self.probe_event('recontact_start',self.hw.snapshot(),timeout_s=2.)
        try:
            # No arm movement, opening or force target increase during recontact.
            self.hw.recontact(2.)
            while self.hw.now()<deadline:
                s=self.hw.snapshot()
                if self.hw.now()>=deadline:break
                if self.bilateral(s) and (s.get('gripper_motion') or {}).get('recontact_stable'):
                    recovered=True;reason='双侧接触连续稳定0.2秒';break
                if s['width']<=self.cfg['gripper']['min_normalized']+1e-6:
                    reason='补触到达开度下限仍未双侧接触';break
                self.hw.wait(.01)
        finally:
            # Cancel before recording/keyframes, including Ctrl+C/data faults.
            if not recovered:self.hw.hold()
        if not recovered:self.probe_failure=reason
        self.probe_event('recontact_recovered' if recovered else 'recontact_failed',self.hw.snapshot(),
                         elapsed_s=(self.hw.now()-start)/1e9,reason=reason)
        return recovered
    def regulate(self,name,target,tolerance,stable_s,*,with_dd=False):
        with self.stage(name):
            probe_contact=(name=='PROBE' and with_dd);lost_since=None;recontacts=0
            set_force=self.hw.probe_force if probe_contact else self.hw.force
            set_force(target,tolerance);deadline=time.monotonic()+self.cfg['force']['timeout_s'];stable=None;last_pair=None
            while time.monotonic()<deadline:
                s=self.hw.snapshot();pair=tuple(s['force_sequences'])
                if probe_contact:
                    bilateral=self.bilateral(s)
                    # A worker can latch hold on a brief dip between runner polls.
                    held=(s.get('gripper_motion') or {}).get('mode')=='hold'
                    if not bilateral or lost_since is not None or held:
                        stable=None;last_pair=None
                        if not bilateral:
                            if lost_since is None:
                                lost_since=time.monotonic();self.hw.pause_probe()
                                self.probe_event('contact_lost_wait',s,grace_s=.2)
                            if time.monotonic()-lost_since>=.2:
                                if recontacts:
                                    self.probe_failure='已补触一次，接触再次丢失超过0.2秒'
                                    self.hw.hold();self.probe_event('recontact_failed',s,reason=self.probe_failure)
                                    return None
                                recontacts+=1
                                if not self.recontact_probe():return None
                                set_force(target,tolerance);lost_since=None
                            self.hw.wait(.01);continue
                        set_force(target,tolerance);lost_since=None
                        self.probe_event('force_resumed',s,recontact_count=recontacts)
                        self.hw.wait(.01);continue
                if pair!=last_pair:
                    last_pair=pair
                    if self.bilateral(s) and abs(float(np.mean(s['normal']))-target)<=tolerance:
                        if stable is None:stable=self.hw.now()
                    else:stable=None
                    if stable is not None and (min(s['force_stamps'])-stable)/1e9>=stable_s:
                        if not with_dd:return {'normal_sdk':s['normal'].tolist(),'wrench_zeroed':s['wrench_zeroed'].tolist(),'timestamp_ns':s['timestamp_ns']}
                        window=self.hw.probe_window(stable)
                        if window:
                            desc=[describe(row['median_dd'],ref,self.cfg) for row,ref in zip(window,self.hw.dd_reference)]
                            key=self.hw.keyframes('PROBE_last',gel=True)
                            return {'valid':all(x['valid'] for x in desc),'descriptors':desc,'normal_sdk':s['normal'].tolist(),
                                    'window':[{'frame_ns':x['frame_ns'],'frame_fids':x['frame_fids']} for x in window],
                                    'timestamp_ns':s['timestamp_ns'],'keyframe':key,'stable_start_ns':stable}
                self.hw.wait(.01)
            self.hw.hold();return None
    def apply_B(self,target,status,probe):
        history=[];exhausted=False
        for attempt in range(self.cfg['servo']['max_regrasps']+1):
            s=self.hw.snapshot();decision=self.correction.decide(status,probe['descriptors'] if probe else [],self.bilateral(s))
            row={'attempt':attempt,'status':status,**decision,'timestamp_ns':self.hw.now()};history.append(row)
            self.current['correction_history']=history;self.save()
            if decision['accepted']:break
            if attempt>=self.cfg['servo']['max_regrasps']:
                exhausted=True;break
            proposed=target.copy();proposed[:3,3]+=self.bounds.a*decision['delta_mm']
            retreat=self.bounds.pre(target,self.cfg['servo']['retreat_mm'])
            corrected_pre=self.bounds.pre(proposed,self.cfg['servo']['retreat_mm'])
            error=self.bounds.check(proposed) or self.bounds.check(retreat,'retreat',target) or self.bounds.check(corrected_pre,'retreat',proposed)
            if error:
                row.update(blocked=True,blocked_reason=error,applied_delta_mm=0.);exhausted=True;break
            self.release()
            with self.stage('REGRASP_RETREAT'):self.move(retreat,'retreat',target)
            with self.stage('REGRASP_CORRECT'):self.move(corrected_pre,'retreat',proposed)
            with self.stage('REGRASP_APPROACH'):self.move(proposed)
            row['applied_delta_mm']=decision['delta_mm'];target=proposed
            status,probe=self.touch();self.current['regrasps']+=1
            self.current.setdefault('probe_regrasps',[]).append({'status':status,'probe':probe,'target_tcp_mm':target.tolist()})
            self.save()
            if status=='probe_contact_failed':exhausted=True;break
        self.current['servo_exhausted']=exhausted;self.current['correction_history']=history
        return target,status,probe
    def hold_and_lower(self,target,squeeze):
        # Empirical vertical-load proxy: tangential increment caused by lift.
        # Missing sensor mounting rotations are not fabricated as known axes.
        normal_axis=self.cfg['force']['normal_axis'];axes=[i for i in range(3) if i!=normal_axis]
        before=np.asarray(squeeze['wrench_zeroed'])[:,:3][:,axes].reshape(-1)
        with self.stage('LIFT'):
            self.airborne=True;self.progress['airborne']=True;self.current['lift_started']=True;self.save()
            self.move(self.bounds.lift(target),'lift',target);self.current['lifted']=True
            lifted=self.hw.snapshot()['wrench_zeroed'][:,:3][:,axes].reshape(-1)-before
        hold_samples=[];last=None;mid_saved=False
        with self.stage('HOLD'):
            start=time.monotonic()
            while time.monotonic()-start<self.cfg['hold_s']:
                s=self.hw.snapshot();pair=tuple(s['force_sequences'])
                if pair!=last:
                    last=pair;hold_samples.append(s['wrench_zeroed'][:,:3][:,axes].reshape(-1)-before)
                    if not self.bilateral(s):
                        self.current['dropped']=True;self.hw.hold()
                if not mid_saved and time.monotonic()-start>=self.cfg['hold_s']/2:
                    self.hw.keyframes('HOLD_mid',gel=True);mid_saved=True
                self.hw.wait(.01)
            self.current['hold_complete']=True
        self.current['dropped']=self.current['dropped'] or getattr(self.hw,'hold_drop_seen',False)
        load=np.median(np.stack(hold_samples),axis=0) if hold_samples else np.zeros(4)
        amplitude=float(np.linalg.norm(load));ln=float(np.linalg.norm(lifted))
        cosine=float(load@lifted/(amplitude*ln)) if amplitude>0 and ln>0 else 0.
        enabled=amplitude>=self.cfg['lower']['unload_abs_sdk'] and cosine>=self.cfg['lower']['mapping_min_cosine'] and not self.current['dropped']
        self.current['unload_proxy']={'enabled':enabled,'source':'empirical tangential increment from pre-lift to HOLD; not calibrated base-frame vertical force',
                                    'tangent_axes':axes,'direction':(load/amplitude).tolist() if amplitude else None,'amplitude_sdk':amplitude,'lift_hold_cosine':cosine}
        count=0;previous=None;high=self.hw.pose()[2,3]
        def monitor(s):
            nonlocal count,previous
            if not enabled:return False
            seq=tuple(s['force_sequences'])
            if seq==previous:return False
            previous=seq
            value=float((s['wrench_zeroed'][:,:3][:,axes].reshape(-1)-before)@(load/amplitude))
            descent=high-s['pose'][2,3]
            enough=descent>=self.cfg['lower']['min_descent_mm'] and value<=amplitude*self.cfg['lower']['unload_ratio'] and amplitude-value>=self.cfg['lower']['unload_abs_sdk']
            count=count+1 if enough else 0
            if count>=self.cfg['lower']['frames']:
                self.current['unload_detected']=True;self.current['unload_stopped_tcp_mm']=s['pose'].tolist();return True
            return False
        with self.stage('LOWER'):
            result=self.move(target,'lower',target,lower_monitor=monitor)
            if result=='reached':self.airborne=False;self.progress['airborne']=False
        self.save()
    def zero_drift(self,probe):
        if not probe or not probe['valid'] or self.current['delta_mm']!=0:return
        values=[x['centroid_px'] for x in probe['descriptors']]
        self.current['zero_centroids_px']=values
        old=[]
        for p in sorted((self.root/'trials').glob('*/meta.json')):
            m=json.loads(p.read_text())
            if m.get('complete') and m.get('recording_complete') and m.get('label')!='invalid' and m.get('zero_centroids_px'):old.append(m['zero_centroids_px'])
        if len(old)<3:return
        if self.meta.get('zero_drift_baseline_px') is None:
            self.meta['zero_drift_baseline_px']=np.mean(np.asarray(old[:3]),axis=0).tolist();self.save()
        baseline=np.asarray(self.meta['zero_drift_baseline_px']);now=np.asarray(values)
        if self.fit:
            errors=[float((now[i,0 if f['coordinate']=='u' else 1]-baseline[i,0 if f['coordinate']=='u' else 1])/f['slope_px_per_mm']) for i,f in enumerate(self.fit['fingers'])]
            source='supplied_E1_fit'
        else:
            # A learns a provisional response only from its completed initial
            # scans; no five-point motion or planned delta in B correction.
            from .e1e2_analysis import fit_e1,load_trials
            fitted=fit_e1(load_trials([self.root]),min_r2=.8)
            if not fitted.get('usable_for_B'):
                self.current['zero_drift']={'checked':False,'reason':'E1 response not yet observable; px retained; no invented mm'};return
            errors=[float((now[i,0 if f['coordinate']=='u' else 1]-baseline[i,0 if f['coordinate']=='u' else 1])/f['slope_px_per_mm']) for i,f in enumerate(fitted['fingers'])];source='provisional_completed_A_scans'
        event={'checked':True,'errors_equivalent_mm':errors,'baseline_centroids_px':baseline.tolist(),'slope_source':source,'alarm':max(abs(x) for x in errors)>self.cfg['design']['drift_mm']}
        self.current['zero_drift']=event
        if event['alarm']:event['operator_response']=self.read('零偏移质心漂移超过阈值，请检查并复位瓶子；回车继续，q结束。');self.quit=event['operator_response'].lower()=='q'
    def label(self):
        while True:
            self.say('请确认瓶子在标记处，必要时复位，再按键。')
            self.say(f"试验 {self.current['id']}/{len(self.meta['plan'])} {self.current['bottle']} δ={self.current['delta_mm']:+g} {self.args.cond} 状态={self.current['status']} 双接触={self.current['bilateral']} 达力={self.current['target_reached']} 提起={self.current['lifted']} 掉落={self.current['dropped']} 重抓={self.current['regrasps']} exhausted={self.current['servo_exhausted']}")
            value=self.read('回车/1正常，2倾倒，3推动，4掉落，5滑动，0作废，n备注，q本次后退出：').lower()
            if value=='n':self.current['note']=self.read('备注：');continue
            if value=='q':self.quit=True;self.say('已登记退出，请再给本次选标签。');continue
            if value not in LABELS:continue
            self.current['label']=LABELS[value]
            if value in ('2','3','4'):self.read('请把瓶子复位到标记处，再按回车。')
            return
    def trial(self,item):
        self.folder=self.root/'trials'/item['id']
        if self.folder.exists():
            archive=self.root/'interrupted'/f"{item['id']}_{time.time_ns()}";archive.parent.mkdir(exist_ok=True);self.folder.rename(archive)
        self.folder.mkdir(parents=True)
        self.current={**item,'complete':False,'status':'started','bilateral':False,'target_reached':False,
                      'lifted':False,'dropped':False,'hold_complete':False,'regrasps':0,'servo_exhausted':False,'note':'',
                      'config_identity':self.meta['config_identity'],'zero':self.hw.zero,'started_unix_ns':time.time_ns()}
        self.progress['active']=item['id'];self.save();self.stage_step=self.args.step or getattr(self,'physical_completed',0)<self.cfg['design']['step_first_trials']
        target=self.bounds.grasp(item['delta_mm']);pre=self.bounds.pre(target);lift=self.bounds.lift(target)
        reason=self.bounds.check(target) or self.bounds.check(pre,'retreat',target) or self.bounds.check(lift,'lift',target)
        if reason:
            self.current.update(status='skipped_bound',skipped_reason=reason,label='invalid',recording_complete=False,complete=True)
            self.say('跳过 '+item['id']+'：'+reason)
        else:
            self.hw.begin_record(self.folder);error=None
            try:
                # A new trial always starts unloaded/open. Release is explicit
                # even when the previous trial ended with a fault-free open.
                self.release()
                with self.stage('MOVE_PRE'):self.move(pre,'retreat',target)
                with self.stage('APPROACH'):self.move(target)
                status,probe=self.touch();self.current['initial_status']=status;self.current['probe_initial']=probe or {}
                if self.correction and status!='probe_contact_failed':target,status,probe=self.apply_B(target,status,probe)
                self.current['probe_final']=probe or {};self.current['status']=status
                self.current['final_tcp_target_mm']=target.tolist();self.current['bilateral']=self.bilateral(self.hw.snapshot())
                if status=='contact' and self.current['bilateral']:
                    squeeze=self.regulate('SQUEEZE',self.cfg['force']['target'],self.cfg['force']['squeeze_tolerance'],self.cfg['force']['squeeze_stable_s'])
                    self.current['target_reached']=squeeze is not None
                    if squeeze:self.hold_and_lower(target,squeeze);self.current['status']='hold_complete'
                    else:self.current['status']='force_not_reached'
                else:self.current['status']=status
                self.release()
                with self.stage('RETREAT'):self.move(self.bounds.pre(target),'retreat',target)
                self.current['dropped']=self.current['dropped'] or getattr(self.hw,'hold_drop_seen',False)
                self.zero_drift(self.current['probe_initial'])
            except BaseException as exc:
                error=str(exc) or type(exc).__name__;self.current['status']='interrupted' if isinstance(exc,(KeyboardInterrupt,UserExit)) else 'failed'
                self.current['error']=error;self.recover(exc);raise
            finally:
                self.save()
                try:
                    try:self.hw.end_record(error=error)
                    except Exception:
                        if error is None:raise
                finally:
                    statuspath=self.folder/'recording_status.json'
                    self.current['recording_complete']=bool(statuspath.exists() and json.loads(statuspath.read_text()).get('complete'))
                    self.save();rebuild_summary(self.root)
            self.hw.set_context('LABEL',item['id']);self.label();self.current['complete']=True
            self.physical_completed=getattr(self,'physical_completed',0)+1
        self.current['finished_unix_ns']=time.time_ns();self.progress['completed'].append(item['id']);self.progress['active']=None;self.save();rebuild_summary(self.root)
    def run(self):
        done=self.completed()
        self.initial();self.print_plan(done);self.authorize('已打印全部偏移和基座变化。确认场地与瓶子，输入 MOVE 开始自动试验。')
        self.hw.ownership();last_bottle=None;count=0
        for item in self.meta['plan']:
            if item['id'] in done:continue
            if self.args.limit is not None and count>=self.args.limit:break
            if item['bottle']!=last_bottle:
                self.read('请放置 '+item['bottle']+' 瓶（同一外形、同一标记），确认已开爪，按回车开始此块。');last_bottle=item['bottle']
            self.trial(item);count+=1
            if self.quit:break
        self.say('本次完成 '+str(count)+' 次；累计 '+str(len(self.progress['completed']))+'/'+str(len(self.meta['plan']))+'。数据：'+str(self.root.resolve()))
        self.say('离线分析：bash run_on_5090.sh analyze_e1e2 '+str(self.root))
        return 0
    def recover(self,exc):
        if not self.engaged or self.recovery_done:return
        self.recovery_done=True
        recovery={'error':str(exc) or type(exc).__name__,'airborne':self.airborne,'started_unix_ns':time.time_ns()}
        try:self.hw.stop_arm();recovery['arm_stop']='software_measured_pose_hold'
        except Exception as e:recovery['arm_stop_error']=str(e)
        try:
            if self.airborne and not getattr(self.hw,'overforce',False):
                self.hw.hold();recovery['gripper']='hold_measured_no_open'
                self.say('已提起或处于可能离桌阶段：已停止机械臂，保持夹爪；请现场接管，把瓶子安全放回后再释放。')
            else:
                self.hw.recovery_open();recovery['gripper']='opened_force_limit' if getattr(self.hw,'overforce',False) else 'opened_pre_lift'
                self.say('F_max 超限：按实验规则优先张开，请现场确认瓶子状态。' if getattr(self.hw,'overforce',False) else '尚未提起／已回到原高度：停止机械臂并张开夹爪。')
        except Exception as e:recovery['gripper_recovery_error']=str(e);self.say('恢复未完成，请现场处理：'+str(e))
        self.progress['airborne']=self.airborne;self.progress['unsafe_recovery']=self.airborne or 'gripper_recovery_error' in recovery
        atomic_json(self.root/'recovery'/f'{time.time_ns()}.json',recovery);self.save();rebuild_summary(self.root)


def main(argv=None,*,hardware_factory=None,input_fn=input,output=print):
    p=parser();args=p.parse_args(argv)
    if args.limit is not None and args.limit<1:p.error('--limit 必须为正整数')
    args.out=args.out.expanduser().resolve();args.out.mkdir(parents=True,exist_ok=True)
    lock=(args.out/'.e1e2.lock').open('a');hw=runner=None;handlers={}
    try:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise ValueError('同一输出目录已有采集进程')
        cfg,fit=resolve_config(args,input_fn,output)
        if (args.out/'session_meta.json').exists():
            meta=json.loads((args.out/'session_meta.json').read_text());rows=rebuild_summary(args.out)
            done={x['id'] for x in rows if x['complete']}
            if meta.get('plan') and all(x['id'] in done for x in meta['plan']):output('全部试验已完成；直接退出，不连接硬件、不发动作。');return 0
        output('独立 E1/E2 实机流程；不执行 setup/run 的几何、空间、停止测试、图像检查或验收。')
        output('F_target='+str(cfg['force']['target'])+' SDK（两指均值）；逐指 F_max='+str(cfg['force']['max'])+'；归一化下限='+str(cfg['gripper']['min_normalized']))
        output('闭合：开度大于 '+str(cfg['gripper'].get('fast_close_until_normalized',.5))+' 时快闭，速度 '+str(cfg['gripper'].get('fast_close_speed_normalized_s',.20))+' 归一化/s；以下慢闭；任一指提前接触即取消快闭。')
        output('慢闭合：1000分制反馈允许±1刻度；相差2刻度时等待，不再加深；超过2刻度停止闭合。无进展2秒仍停止。')
        output('Ctrl+C：软件停止；未提起时张开，已提起时保持。F_max 超限按本实验规则始终优先松开。')
        if hardware_factory is None:
            from .e1e2_hardware import Hardware
            hardware_factory=Hardware
        hw=hardware_factory(cfg);runner=Runner(args,cfg,fit,hw,ask=input_fn,say=output)
        def interrupted(signum,frame):raise KeyboardInterrupt('收到信号 '+str(signum))
        if __import__('threading').current_thread() is __import__('threading').main_thread():
            for sig in (signal.SIGINT,signal.SIGTERM):handlers[sig]=signal.signal(sig,interrupted)
        hw.connect()
        return runner.run()
    except (Exception,KeyboardInterrupt) as exc:
        if runner:
            try:runner.recover(exc)
            except Exception as recovery:output('恢复处理失败，请现场接管：'+str(recovery))
        atomic_json(args.out/'errors'/f'{time.time_ns()}.json',{'error':str(exc) or type(exc).__name__,'traceback':traceback.format_exc()})
        output('E1/E2 已停止：'+(str(exc) or type(exc).__name__))
        return 130 if isinstance(exc,(KeyboardInterrupt,UserExit)) else 2
    finally:
        if hw:hw.close()
        for sig,old in handlers.items():signal.signal(sig,old)
        lock.close()
