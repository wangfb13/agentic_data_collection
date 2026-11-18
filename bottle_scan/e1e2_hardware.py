"""Public-ROS E1/E2 I/O; no setup/run gates and no millimetre jaw calibration.

Imports are inert. Connection creates command transports without publishing.
Only explicit runner requests enable movement. Existing observation, X5 codecs,
SDK projection, legacy force step rule and lossless async writer are reused.
"""
from __future__ import annotations
from collections import deque
import copy
from pathlib import Path
import threading
import time
import numpy as np
import yaml
from .backend import Sample
from .e1e2_core import opening_within_tolerance
from .hardware_gripper import RosNormalizedGripperObserver, load_legacy_force_functions
from .hardware_sensors import SdkWrenchProjection
from .recording import AsyncRecorder
from .session import plain

SIDES=('left','right')


class OverForce(RuntimeError):pass


class NormalizedGripper(RosNormalizedGripperObserver):
    """Own normalized command loop. Never fabricates a width in mm."""
    def __init__(self,cfg,force_reader,*,clock_ns=time.monotonic_ns,legacy=None,**kwargs):
        super().__init__(cfg['arm'],clock_ns=clock_ns,state_queue_size=1024,**kwargs)
        self.cfg=cfg;self.force_reader=force_reader;self.legacy=legacy
        self.control_lock=threading.RLock();self.events_lock=threading.Lock();self.events=deque()
        self.publisher=None;self.mode='idle';self.goal=None;self.force_target=None
        self.last_tick=None;self.last_position_seq=None;self.last_force_seq=None;self.last_command_ns=None
        self.fault=None;self.overforce=False;self.armed=False
        self._position_request=None;self._first_position_command_ns=None
        self._position_target=None;self._position_target_changed_ns=None
        self._slow_close_tracking=False
        self._previous_close_was_fast=False;self._fast_handoff_ns=None
        self._handoff_wait=None;self._handoff_recovery_used=False
        self._recontact_deadline_ns=None;self._recontact_contact_since_ns=None;self._recontact_stable=False;self._recontact_last_loss_ns=None
        self._last_sent_value=None;self._last_sent_kind=None;self._probe_force=False;self._fast_close_cancelled=False
    def _event(self,stream,data,stamp,metadata):
        with self.events_lock:
            if len(self.events)>=10000:raise RuntimeError('归一化夹爪事件队列溢出')
            self.events.append(Sample(stream,np.asarray(data),int(stamp),metadata=metadata))
    def _on_state(self,message):
        super()._on_state(message)
        s=super().snapshot()
        if s['valid']:
            try:self._event('gripper_normalized',np.asarray(s['normalized']),s['monotonic_ns'],
                {'unit':'normalized','source_sequence':s['source_sequence'],'device_timestamp_available':False})
            except Exception as e:self.fault=str(e)
    def connect(self):
        super().connect()
        self.publisher=self._ros.Publisher('/'+self.side+'_gripper_command',self._message_type,queue_size=1,latch=False)
        return self
    def fresh(self):
        s=super().snapshot()
        if not s['fresh']:raise RuntimeError('夹爪开度缺失／过期：'+str(s.get('error')))
        return s
    def drain_samples(self):
        with self.events_lock:r=list(self.events);self.events.clear();return r
    def _publish(self,value,kind):
        value=float(value);g=self.cfg['gripper'];s=self.fresh()
        one_way_open=self.mode=='open' and 0<=s['normalized']<=value<=g['open_normalized']
        if not g['min_normalized']<=value<=g['open_normalized'] and not one_way_open:raise RuntimeError('归一化指令越过上下限')
        if self.publisher is None or self.publisher.get_num_connections()<1:raise RuntimeError('夹爪命令话题无订阅者')
        msg=self._message_type();msg.data=value;self.publisher.publish(msg)
        now=self._clock_ns();self.last_command_ns=now
        self._last_sent_value=value;self._last_sent_kind=kind
        if kind in ('open','close'):
            if self._first_position_command_ns is None:self._first_position_command_ns=now
            if self._position_target is None or abs(self._position_target-value)>1e-9:
                self._position_target=value;self._position_target_changed_ns=now
        self._event('gripper_command_normalized',np.asarray(value),now,{'unit':'normalized','kind':kind,
            'measured_normalized':s['normalized'],'source_state_sequence':s['source_sequence'],'software_only':True,
            'first_position_command_ns':self._first_position_command_ns,
            'position_target_changed_ns':self._position_target_changed_ns,
            'fast_close_until_normalized':g.get('fast_close_until_normalized',.5),
            'fast_close_speed_normalized_s':g.get('fast_close_speed_normalized_s',.20),
            'fast_close_cancelled':self._fast_close_cancelled,
            'position_feedback_tolerance_counts':g.get('position_feedback_tolerance_counts',1),
            'position_max_tracking_error_counts':g.get('position_max_tracking_error_counts',2),
            'slow_close_tracking':self._slow_close_tracking,
            'handoff_wait':copy.deepcopy(self._handoff_wait)})
    def hold(self):
        with self.control_lock:
            self.mode='hold';self.goal=None;self.force_target=None;self._recontact_deadline_ns=None
            self._probe_force=False;self._recontact_stable=False;self._handoff_wait=None
            s=self.fresh()
            # Holding an actual position slightly below a bound must not close
            # further. Transport cancellation records the unchanged observation.
            if self.cfg['gripper']['min_normalized']<=s['normalized']<=self.cfg['gripper']['open_normalized']:
                self._publish(s['normalized'],'hold_measured')
            else:
                msg=self._message_type();msg.data=s['normalized'];self.publisher.publish(msg)
                self._event('gripper_command_normalized',np.asarray(s['normalized']),self._clock_ns(),{'unit':'normalized','kind':'fault_hold_actual_outside_range'})
    def _preserve_probe_target(self,value,actual):
        # Preserve an already-issued tiny closing request, without accumulating
        # additional error or retaining a large pre-contact fast step.
        if self._last_sent_value is not None and self._last_sent_kind in ('close','force_close','probe_pause_target','probe_settle_target'):
            value=min(value,self._last_sent_value)
        return float(max(self.cfg['gripper']['min_normalized'],actual-.002,value))
    def _publish_probe_pause(self,kind):
        actual=float(self.fresh()['normalized'])
        self._publish(self._preserve_probe_target(actual,actual),kind)
    def pause_probe(self):
        with self.control_lock:
            if self._handoff_wait is not None:
                raise RuntimeError('快慢闭合切换的反馈仍在核对，禁止进入 PROBE')
            self.mode='hold';self.goal=None;self.force_target=None;self._recontact_deadline_ns=None
            self._recontact_stable=False
            self._publish_probe_pause('probe_pause_target')
    def _reset_position_tracking(self,sample):
        self._recontact_deadline_ns=None;self._recontact_contact_since_ns=None;self._recontact_stable=False;self._recontact_last_loss_ns=None
        self._probe_force=False;self._fast_close_cancelled=False
        self._position_request={'requested_ns':int(self._clock_ns()),'start_normalized':sample['normalized']}
        self._first_position_command_ns=None;self._position_target=None;self._position_target_changed_ns=None
        self._slow_close_tracking=False
        self._previous_close_was_fast=False;self._fast_handoff_ns=None
        self._handoff_wait=None;self._handoff_recovery_used=False
    def motion_status(self):
        with self.control_lock:
            return {**(self._position_request or {}),'mode':self.mode,
                    'first_command_ns':self._first_position_command_ns,
                    'target_normalized':self._position_target,'target_changed_ns':self._position_target_changed_ns,
                    'position_feedback_tolerance_counts':self.cfg['gripper'].get('position_feedback_tolerance_counts',1),
                    'position_max_tracking_error_counts':self.cfg['gripper'].get('position_max_tracking_error_counts',2),
                    'recontact_stable':self._recontact_stable,'recontact_deadline_ns':self._recontact_deadline_ns}
    def open(self,*,recovery=False):
        with self.control_lock:
            if self.fault and not recovery:raise RuntimeError(self.fault)
            self._reset_position_tracking(self.fresh())
            self.mode='open';self.goal=self.cfg['gripper']['open_normalized'];self.force_target=None
            self.armed=True;self.last_tick=self._clock_ns();self.last_position_seq=None
    def close_to_floor(self):
        with self.control_lock:
            if self.fault or self.overforce:raise RuntimeError(self.fault or 'F_max 已触发')
            self._reset_position_tracking(self.fresh());self.mode='close';self.goal=self.cfg['gripper']['min_normalized'];self.force_target=None
            self.armed=True;self.last_tick=self._clock_ns();self.last_position_seq=None
    def recontact(self,timeout_s):
        # The worker owns this deadline: a delayed runner cannot extend closure.
        if not np.isfinite(timeout_s) or not 0<timeout_s<=2.:
            raise ValueError('PROBE 补触最多2秒')
        with self.control_lock:
            self.close_to_floor();self.mode='recontact';self._fast_close_cancelled=True
            self._recontact_deadline_ns=self._clock_ns()+int(timeout_s*1e9)
    def observe_recontact_force(self,finger,normal,stamp_ns):
        # A low source frame between worker polls also breaks continuity.
        with self.control_lock:
            if self.mode not in ('recontact','probe_ready'):return
            if stamp_ns<(self._position_request or {}).get('requested_ns',0):return
            thresholds=self.force_reader().get('contact_thresholds',[self.cfg['force']['min_contact']]*2)
            if normal<thresholds[finger]:
                self._recontact_last_loss_ns=max(int(stamp_ns),self._recontact_last_loss_ns or 0)
                self._recontact_contact_since_ns=None;self._recontact_stable=False;self.mode='recontact'
    def force(self,target,tolerance,*,probe=False):
        with self.control_lock:
            if self._handoff_wait is not None:raise RuntimeError('快慢闭合切换的反馈仍在核对，禁止进入力控')
            if self.fault or self.overforce:raise RuntimeError(self.fault or 'F_max 已触发')
            if not 0<target<self.cfg['force']['max']:raise ValueError('力目标需小于 F_max')
            self._recontact_deadline_ns=None;self._recontact_stable=False;self._probe_force=bool(probe)
            self.mode='force';self.force_target=float(target);self.tolerance=float(tolerance)
            self.armed=True;self.last_force_seq=None;self.last_position_seq=None
    def tick(self):
        with self.control_lock:
            now=self._clock_ns()
            if self.mode=='idle' or not self.armed:return
            s=self.fresh();force=self.force_reader();normals=np.asarray(force['normal'],float)
            if np.max(normals)>self.cfg['force']['max'] and not self.overforce:
                self.overforce=True
                self.hold();self.open(recovery=True)
            if self.mode=='probe_ready':
                thresholds=force.get('contact_thresholds',[self.cfg['force']['min_contact']]*2)
                if np.all(normals>=np.asarray(thresholds)):return
                self._recontact_stable=False;self._recontact_contact_since_ns=None;self.mode='recontact'
            if self.mode=='hold':return
            if self.mode in ('close','force','recontact','handoff_wait') and s['normalized']<self.cfg['gripper']['min_normalized']-1e-6:
                self.hold();raise RuntimeError('实际归一化开度已低于配置下限，停止进一步闭合')
            if self.mode=='handoff_wait':
                waiting=self._handoff_wait
                if waiting is None:
                    self.hold();raise RuntimeError('快慢闭合切换核对状态丢失；已停止闭合')
                if now-waiting['started_ns']>=250_000_000:
                    previous,actual=waiting['reference_counts'],round(s['normalized']*1000.)
                    self.hold();raise RuntimeError('慢闭合指令与反馈相差超过2个刻度，暂停250毫秒后仍未连续恢复：指令=%d，反馈=%d；已停止闭合' % (previous,actual))
                if s['source_sequence']==waiting['last_sequence']:return
                waiting['last_sequence']=s['source_sequence']
                actual=round(s['normalized']*1000.)
                if abs(actual-waiting['reference_counts'])>3:
                    previous=waiting['reference_counts']
                    self.hold();raise RuntimeError('快慢闭合切换期间反馈继续偏离：指令=%d，反馈=%d；已停止闭合' % (previous,actual))
                prior=waiting['last_good_counts']
                if abs(actual-waiting['reference_counts'])<=2:
                    waiting['stable_frames']=waiting['stable_frames']+1 if prior is not None and abs(actual-prior)<=1 else 1
                    waiting['last_good_counts']=actual
                else:
                    waiting['stable_frames']=0;waiting['last_good_counts']=None
                if waiting['stable_frames']>=2:
                    # Only new independent feedback within the original two-count
                    # limit permits a new slow-close command on a later tick.
                    self.mode='close';self._handoff_wait=None;self._fast_close_cancelled=True
                    self._slow_close_tracking=False;self._position_target=None
                    self.last_position_seq=s['source_sequence'];self.last_tick=now
                return
            if self.mode=='recontact':
                if now>=self._recontact_deadline_ns:
                    self.hold();return
                thresholds=force.get('contact_thresholds',[self.cfg['force']['min_contact']]*2)
                if np.all(normals>=np.asarray(thresholds)):
                    stamp=int(min(force['stamps']))
                    if self._recontact_last_loss_ns is not None and stamp<=self._recontact_last_loss_ns:return
                    if self._recontact_contact_since_ns is None:
                        self._recontact_contact_since_ns=stamp
                        self._publish_probe_pause('probe_settle_target')
                    if stamp-self._recontact_contact_since_ns>=200_000_000:
                        # Leave the existing closing target intact for force
                        # handoff; do not publish a more-open measured hold.
                        self.mode='probe_ready';self._recontact_stable=True
                    return
                self._recontact_contact_since_ns=None
            if self.mode=='force':
                thresholds=force.get('contact_thresholds',[self.cfg['force']['min_contact']]*2)
                if np.any(normals<np.asarray(thresholds)):
                    if self._probe_force:self.pause_probe()
                    else:self.hold()
                    return
                seq=tuple(force['sequence']);stamps=force['stamps']
                if self.last_force_seq is not None and not all(a>b for a,b in zip(seq,self.last_force_seq)):return
                if self.last_command_ns is not None and (now-self.last_command_ns<self.cfg['force']['command_settle_s']*1e9 or min(stamps)<self.last_command_ns):return
                # Reuse the deployed controller's pure step rule; SDK positive
                # compression and mean definition are already explicit here.
                c={'position_min':self.cfg['gripper']['min_normalized']*1000.,
                   'position_max':self.cfg['gripper']['open_normalized']*1000.,
                   'step_size':self.cfg['force']['step_counts'],'target_force':self.force_target,
                   'deadband':self.tolerance,'control_force_mode':'sdk_raw'}
                mean=float(normals.mean())
                actual_counts=round(s['normalized']*1000.)
                # The public API's inherited force rule uses integer normalized
                # thousandths. All new position commands use this same grid.
                try:direction,counts=self.legacy.tag_step(c,mean,actual_counts)
                except Exception as exc:
                    if 'position_limit' not in str(exc):raise
                    direction,counts='at_bound',None
                self.last_force_seq=seq
                if counts is not None:
                    value=float(np.clip(counts/1000.,self.cfg['gripper']['min_normalized'],self.cfg['gripper']['open_normalized']))
                    if self._probe_force and direction=='close':value=self._preserve_probe_target(value,float(s['normalized']))
                    self._publish(value,'force_'+str(direction))
                return
            if s['source_sequence']==self.last_position_seq:return
            dt=min(max(0.,(now-(self.last_tick or now))/1e9),2*self.cfg['force']['control_period_s']);self.last_tick=now
            g=self.cfg['gripper'];actual=float(s['normalized']);delta=self.goal-actual
            thresholds=force.get('contact_thresholds',[self.cfg['force']['min_contact']]*2)
            if self.mode=='close' and np.any(normals>=np.asarray(thresholds)):
                self._fast_close_cancelled=True
            fast_until=g.get('fast_close_until_normalized',.5)
            fast=self.mode=='close' and not self._fast_close_cancelled and actual>fast_until
            if self.mode=='close' and self._previous_close_was_fast and not fast:
                self._fast_handoff_ns=now
            if self.mode=='close':self._previous_close_was_fast=fast
            speed=g.get('fast_close_speed_normalized_s',.20) if fast else g[('open' if self.mode=='open' else 'close')+'_speed_normalized_s']
            if self.mode=='open' and delta<0:raise RuntimeError('张开目标小于当前读数，拒绝反向闭合')
            if self.mode in ('close','recontact') and delta>0:raise RuntimeError('闭合下限大于当前读数')
            if abs(delta)<=1e-7:self.mode='hold';return
            steps=int(np.floor(speed*dt*1000.+1e-7))
            if steps<1:return
            actual_counts=round(actual*1000.)
            slow_close=self.mode in ('close','recontact') and not fast
            if slow_close:
                tolerance=g.get('position_feedback_tolerance_counts',1)
                max_error=g.get('position_max_tracking_error_counts',2)
                previous=round(self._position_target*1000.) if self._slow_close_tracking and self._position_target is not None else None
                if previous is not None and abs(actual_counts-previous)>max_error:
                    # A single 3-count discrepancy immediately after the
                    # fast-to-slow handoff may be transient. Stop requesting
                    # additional closure immediately; never expand the 2-count
                    # limit, and permit only one short, independently sampled
                    # recovery per close. Other mismatches still fail at once.
                    handoff=(self.mode=='close' and abs(actual_counts-previous)==max_error+1 and
                             self._fast_handoff_ns is not None and now-self._fast_handoff_ns<=300_000_000 and
                             not self._handoff_recovery_used)
                    if handoff:
                        self._handoff_recovery_used=True
                        self._handoff_wait={'started_ns':int(now),'reference_counts':previous,
                                            'first_feedback_counts':actual_counts,
                                            'last_sequence':s['source_sequence'],'stable_frames':0,
                                            'last_good_counts':None}
                        self.mode='handoff_wait'
                        self._publish(actual_counts/1000.,'handoff_wait_hold')
                        return
                    self.hold()
                    raise RuntimeError('慢闭合指令与反馈相差超过%d个刻度：指令=%d，反馈=%d；已停止闭合' % (max_error,previous,actual_counts))
                if previous is not None and abs(actual_counts-previous)>tolerance:
                    # Exactly two counts away: wait for feedback, never deepen.
                    value=previous/1000.
                else:
                    # Advance from the accepted command so +/-1 feedback noise
                    # cannot reverse or trap the incremental closing target.
                    base=actual_counts if previous is None else previous
                    value=max(base-steps,actual_counts-max_error)/1000.
                self._slow_close_tracking=True
            else:
                self._slow_close_tracking=False
                value=actual_counts/1000.+np.sign(delta)*steps/1000.
            value=float(np.clip(value,max(g['min_normalized'],fast_until) if fast else g['min_normalized'],g['open_normalized']))
            if self.mode=='recontact':value=self._preserve_probe_target(value,actual)
            self._publish(value,'close' if self.mode=='recontact' else self.mode);self.last_position_seq=s['source_sequence']
    def close(self):
        with self.control_lock:
            self.mode='idle';self.armed=False
            if self.publisher is not None:self.publisher.unregister();self.publisher=None
        super().close()


class Hardware:
    def __init__(self,cfg):
        self.cfg=cfg;self.arm=self.grip=self.sensors=None;self.worker=None
        self.stop=threading.Event();self.lock=threading.RLock();self.writer=None
        self.error=None;self.zero=None;self.dd_reference=None;self.stage='IDLE';self.trial_id=None
        self.dd={s:deque(maxlen=256) for s in SIDES};self.gel={};self.cameras={}
        self.arm_active=False;self.active=False;self.overforce=False;self.hold_drop_seen=False;self.event_log=[]
    def ownership(self):
        from .command_ownership import guard_all_commands,current_ros_node_names
        return guard_all_commands(self.hwcfg,own_nodes=current_ros_node_names())
    def connect(self):
        from .hardware_arm import RosArmController
        from .hardware_sensors import ActualX5SensorSource
        self.hwcfg=yaml.safe_load(Path(self.cfg['hardware_config']).read_text())
        self.hwcfg=copy.deepcopy(self.hwcfg)
        self.hwcfg.setdefault('force',{})['unit']='sdk'
        self.hwcfg['record']={'gel_raw':'full'}
        self.hwcfg['hardware']['public_ros']['sensors'].pop('force_conversion',None)
        self.hwcfg['recording']=dict(self.hwcfg.get('recording',{}),source_max_age_s=.25)
        if self.hwcfg['hardware']['public_ros']['sensors'].get('gripper_side')!=self.cfg['arm']:
            raise ValueError('X5 配置所属夹爪与实验臂不同；不能用右指数据控制左夹爪')
        self.ownership()
        source=self.hwcfg['hardware']['public_ros']['force_source_path']
        legacy=load_legacy_force_functions(source)
        self.legacy_evidence={'source_path':legacy.source_path,'source_sha256':legacy.source_sha256}
        try:
            self.arm=RosArmController(arm_side=self.cfg['arm'],pose_tolerance_mm=self.cfg['motion']['pose_tolerance_mm'],pose_tolerance_deg=self.cfg['motion']['pose_tolerance_deg'])
            self.arm.connect()
            self.grip=NormalizedGripper(self.cfg,self.force_sample,legacy=legacy).connect()
            self.sensors=ActualX5SensorSource(self.hwcfg);self.sensors.start()
            deadline=time.monotonic()+5
            while not self.grip.snapshot()['fresh'] and time.monotonic()<deadline:time.sleep(.02)
            self.grip.fresh();self.pose()
            self.worker=threading.Thread(target=self._loop,daemon=True,name='e1e2_acquisition_control');self.worker.start()
            return self
        except BaseException:self.close();raise
    def now(self):return time.monotonic_ns()
    def pose(self):return np.asarray(self.arm.read_pose(),float)
    def force_sample(self, *, pair_skew_s=None, pair_max_age_s=None):
        raw=(self.sensors.observation_snapshot() if pair_skew_s is None else
             self.sensors.observation_snapshot(pair_skew_s=pair_skew_s,pair_max_age_s=pair_max_age_s))
        if self.zero is None:raise RuntimeError('尚未扣零')
        f=self.cfg['force'];delta=raw['wrench_raw']-np.asarray(self.zero['offset_raw'])
        return {'normal':delta[:,f['normal_axis']]*np.asarray(f['normal_signs']),
                'sequence':[raw['source_sequences']['wrench_raw_'+s] for s in SIDES],
                'stamps':[raw['source_timestamp_ns']['wrench_raw_'+s] for s in SIDES],
                'wrench_zeroed':delta,'raw':raw,'contact_thresholds':self.zero['contact_thresholds_sdk']}
    def snapshot(self):
        self.check();f=self.force_sample();g=self.grip.fresh()
        return dict(normal=f['normal'],force_sequences=f['sequence'],force_stamps=f['stamps'],
                    wrench_zeroed=f['wrench_zeroed'],width=g['normalized'],width_sequence=g['source_sequence'],
                    width_stamp=g['monotonic_ns'],pose=self.pose(),timestamp_ns=self.now(),gripper_motion=self.grip.motion_status())
    def check(self):
        if self.error:raise RuntimeError('E1/E2 采集／控制失败：'+str(self.error))
        if self.grip and self.grip.overforce and not self.overforce:self._force_limit_release()
        if self.overforce or (self.grip and self.grip.overforce):raise OverForce('逐指法向力超过 F_max，已停止闭合并执行张开；本轮停止')
        if self.writer:self.writer.check()
    def set_context(self,stage,trial_id=None):
        with self.lock:
            self.stage=stage
            if trial_id is not None:
                if trial_id!=self.trial_id:self.hold_drop_seen=False
                self.trial_id=trial_id
            self.sensors.set_context(stage=stage,trial_id=self.trial_id)
    def _append(self,event):
        if not self.writer:return
        meta={**(event.metadata or {}),'e1e2_stage':self.stage,'e1e2_trial_id':self.trial_id}
        self.writer.h5.append(event.stream,event.data,event.monotonic_ns,event.device_timestamp,attrs=meta)
    def _drain(self):
        events=self.sensors.drain_samples()+self.arm.drain_samples()+self.grip.drain_samples()
        # Producers' queues are independent. Each writer stream retains its own
        # order; deliberately no polling copies of DD/force feedback are logged.
        with self.lock:
            for ev in events:
                if ev.stream.startswith('camera/'):
                    self.cameras[ev.stream.split('/',1)[1]]=ev
                    continue
                if ev.stream.startswith('gel_raw_'):
                    self.gel[ev.stream.rsplit('_',1)[1]]=ev
                    continue
                if ev.stream in ('dd_left','dd_right'):
                    self.dd[ev.stream[3:]].append(ev)
                if ev.stream in ('wrench_raw_left','wrench_raw_right') and self.zero is not None and self.grip.armed:
                    i=0 if ev.stream.endswith('left') else 1
                    f=self.cfg['force'];normal=(ev.data[f['normal_axis']]-self.zero['offset_raw'][i][f['normal_axis']])*f['normal_signs'][i]
                    # Inspect EVERY source event, including a burst whose latest
                    # sample has already recovered. Never miss a peak/drop by
                    # looking only at the newest polled value.
                    if normal>f['max']:self._force_limit_release()
                    if self.grip.mode in ('recontact','probe_ready'):
                        self.grip.observe_recontact_force(i,normal,ev.monotonic_ns)
                    if (ev.metadata or {}).get('stage')=='HOLD' and (ev.metadata or {}).get('trial_id')==self.trial_id and normal<self.zero['contact_thresholds_sdk'][i]:
                        self.hold_drop_seen=True
                        if self.grip.mode=='force':self.grip.hold()
                self._append(ev)
    def _force_limit_release(self):
        if self.overforce:return
        self.grip.overforce=True;self.overforce=True
        try:self.arm.stop_arm();self.arm_active=False
        except Exception:pass
        try:self.grip.hold()
        finally:self.grip.open(recovery=True)
    def _loop(self):
        previous=0.
        try:
            while not self.stop.wait(.005):
                self._drain()
                if self.writer:self.writer.check()
                if self.grip.armed and self.zero is not None:
                    f=self.force_sample()
                    if max(f['normal'])>self.cfg['force']['max'] and not self.grip.overforce:
                        # Force release has priority even if airborne, as
                        # explicitly requested. Ctrl+C has a separate policy.
                        self._force_limit_release()
                if time.monotonic()-previous>=self.cfg['force']['control_period_s']:
                    self.grip.tick();previous=time.monotonic()
                    if self.grip.overforce and not self.overforce:self._force_limit_release()
        except BaseException as exc:
            self.error=exc
            if self.active:
                try:self.arm.stop_arm();self.arm_active=False
                except Exception:pass
                try:self.grip.hold()
                except Exception:pass
    def wait(self,seconds):
        deadline=time.monotonic()+seconds
        while time.monotonic()<deadline:self.check();time.sleep(min(.01,max(0.,deadline-time.monotonic())))
    def zero_unloaded(self):
        g=self.grip.fresh()
        if not opening_within_tolerance(g['normalized'],self.cfg['gripper']):raise RuntimeError('扣零需要已全开；当前读数 '+str(g['normalized']))
        start=self.now();n=self.cfg['zero']['frames']
        def monitor():
            self.check()
            if not opening_within_tolerance(self.grip.fresh()['normalized'],self.cfg['gripper']):raise RuntimeError('扣零期间夹爪开度改变')
        samples=self.sensors.collect_raw_zero_samples(n,self.cfg['zero']['timeout_s'],monitor=monitor)
        means=[];std=[]
        for s in SIDES:
            values=np.stack([x[1] for x in samples[s]])
            means.append(values.mean(axis=0));std.append(float(values[:,self.cfg['force']['normal_axis']].std(ddof=1)))
        with self.lock:
            refs=[]
            for s in SIDES:
                dd=[x.data for x in self.dd[s] if x.monotonic_ns>=start]
                if len(dd)<5:raise RuntimeError('空载 DD 独立帧不足5帧')
                refs.append(np.median(np.stack(dd),axis=0).astype(np.float32))
            self.dd_reference=refs
            thresholds=np.maximum(5*np.asarray(std),self.cfg['force']['min_contact'])
            self.zero={'offset_raw':np.stack(means).tolist(),'normal_std_sdk':std,'contact_thresholds_sdk':thresholds.tolist(),
                       'force_source_ids':{s:[x[2] for x in samples[s]] for s in SIDES},'created_unix_ns':time.time_ns()}
            with self.sensors._condition:
                self.sensors.calibrations={s:SdkWrenchProjection(np.asarray(means[i]),self.cfg['force']['normal_axis'],self.cfg['force']['normal_signs'][i]) for i,s in enumerate(SIDES)}
                self.sensors._last_normal_pair=None
        if max(thresholds)>=self.cfg['force']['probe_fraction']*self.cfg['force']['target']-self.cfg['force']['probe_tolerance']:
            raise RuntimeError('5σ 接触阈值已进入 PROBE 力带；请消除空载扰动或配置适当目标')
        return plain(self.zero)
    def begin_record(self,path):
        writer=AsyncRecorder(path,self.cfg)
        try:
            with self.lock:
                if self.writer:raise RuntimeError('已有记录未关闭')
                # Do not attach old idle events to the new physical trial.
                self._drain();self.writer=writer
        except BaseException:
            try:writer.close(error='recording start failed')
            except Exception:pass
            raise
        return writer
    def end_record(self,error=None):
        drain_error=None
        with self.lock:
            writer=self.writer
            try:self._drain()
            except BaseException as exc:drain_error=exc
            finally:self.writer=None
        if writer:writer.close(error=error or (str(drain_error) if drain_error else None))
        if drain_error:raise drain_error
    def keyframes(self,reason,*,gel=False):
        requested=self.now();deadline=time.monotonic()+.5
        while time.monotonic()<deadline:
            self.check()
            with self.lock:
                cameras={s:self.cameras.get(s) for s in ('left_wrist','right_wrist')}
                gels={s:self.gel.get(s) for s in SIDES} if gel else {}
                ready=all(e and e.monotonic_ns>requested for e in [*cameras.values(),*gels.values()])
                if ready:
                    pose=self.pose().tolist()
                    for name,ev in cameras.items():
                        if self.writer:self.writer.jpg.append(name,ev.data,ev.monotonic_ns,ev.device_timestamp,
                            attrs={**(ev.metadata or {}),'keyframe_reason':reason,'T_base_tcp_mm':pose,'e1e2_stage':self.stage})
                    for name,ev in gels.items():
                        if self.writer:self.writer.h5.append('gel_keyframe_'+name,ev.data,ev.monotonic_ns,ev.device_timestamp,
                            attrs={**(ev.metadata or {}),'keyframe_reason':reason,'e1e2_stage':self.stage})
                    return {'reason':reason,'requested_ns':requested,'cameras':{n:e.monotonic_ns for n,e in cameras.items()},'gel':{n:e.monotonic_ns for n,e in gels.items()}}
            time.sleep(.005)
        raise RuntimeError('关键帧缺失：'+reason)
    def probe_window(self,since):
        with self.lock:
            result=[]
            for s in SIDES:
                rows=[x for x in self.dd[s] if x.monotonic_ns>=since][-self.cfg['tactile']['probe_frames']:]
                if len(rows)<self.cfg['tactile']['probe_frames']:return None
                result.append({'median_dd':np.median(np.stack([x.data for x in rows]),axis=0).astype(np.float32),
                               'frame_ns':[x.monotonic_ns for x in rows],
                               'frame_fids':[x.metadata.get('fid') for x in rows]})
            return result
    def _reset_position_tracking(self,sample):
        self._position_request={'requested_ns':int(self._clock_ns()),'start_normalized':sample['normalized']}
        self._first_position_command_ns=None;self._position_target=None;self._position_target_changed_ns=None
    def motion_status(self):
        with self.control_lock:
            return {**(self._position_request or {}),'mode':self.mode,
                    'first_command_ns':self._first_position_command_ns,
                    'target_normalized':self._position_target,'target_changed_ns':self._position_target_changed_ns}
    def open(self,*,recovery=False):self.active=True;self.grip.open(recovery=recovery)
    def close_to_floor(self):self.active=True;self.grip.close_to_floor()
    def recontact(self,timeout_s):self.active=True;self.grip.recontact(timeout_s)
    def hold(self):self.grip.hold()
    def pause_probe(self):self.grip.pause_probe()
    def probe_force(self,target,tolerance):self.grip.force(target,tolerance,probe=True)
    def force(self,target,tolerance):self.grip.force(target,tolerance)
    def move(self,target,speed):self.active=True;self.arm_active=True;self.arm.move_linear(target,speed,1.)
    def motion_done(self):
        done=self.arm.motion_done()
        if done:self.arm_active=False
        return done
    def stop_arm(self):
        if self.arm_active or self.active:self.arm.stop_arm()
        self.arm_active=False
    def recovery_open(self):
        """Works after a data fault; only measured opening, no further closure."""
        self.grip.hold();self.grip.open(recovery=True)
        deadline=time.monotonic()+self.cfg['gripper']['timeout_s']
        while time.monotonic()<deadline:
            g=self.grip.fresh()
            if opening_within_tolerance(g['normalized'],self.cfg['gripper']):self.grip.hold();return
            # If acquisition worker died, a force-independent one-way opening
            # tick avoids a dead force read preventing pre-lift release.
            if self.error:
                with self.grip.control_lock:
                    value=min(self.cfg['gripper']['open_normalized'],g['normalized']+self.cfg['gripper']['open_speed_normalized_s']*.05)
                    self.grip._publish(value,'recovery_open_only')
            time.sleep(.05)
        raise RuntimeError('恢复张开超时，请现场处理')
    def close(self):
        self.stop.set()
        if self.worker:self.worker.join(timeout=3)
        if self.writer:
            try:self.end_record(error=str(self.error or 'interrupted'))
            except Exception:pass
        for obj in (self.grip,self.arm,self.sensors):
            if obj:
                try:obj.close()
                except Exception:pass
