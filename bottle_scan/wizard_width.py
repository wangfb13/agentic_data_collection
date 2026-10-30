"""Operator-started, no-arm normalized jaw calibration before mm mapping exists.

The lower normalized operating bound is inherited from the existing public ROS
controller. No mm position, speed, encoder resolution or extrapolation is faked.
"""
from __future__ import annotations
from pathlib import Path
import time
import numpy as np
from .geometry import pose_error
from .hardware_gripper import WidthCalibration
from .session import atomic_json
from .recording import AsyncRecorder
from .command_ownership import guard_all_commands,current_ros_node_names


def fit_width_points(points):
    pairs=sorted(points,key=lambda x:x['normalized'])
    calibration=WidthCalibration(pairs)
    x=np.asarray([p['normalized'] for p in pairs]);y=np.asarray([p['width_mm'] for p in pairs])
    slope,intercept=np.polyfit(x,y,1);residual=y-(slope*x+intercept)
    return {'points':list(calibration.points),'slope_mm_per_normalized':float(slope),
        'intercept_mm':float(intercept),'residual_mm':residual.tolist(),
        'linearity_check_possible':len(pairs)>=3,'distinct_widths':len(pairs),
        'scope':'measured knots only; no extrapolation; first-contact compression remains a bias'}


def compliant_width_mapping(open_point,body_point,cfg):
    """Measured anchors + explicitly modelled, finite equivalent compression.

    This does NOT measure the loaded gel gap or change bottle diameter. Below
    first contact, use the body/open slope as an equivalent actuator coordinate.
    Physical travel remains bounded by the inherited normalized floor and both
    gel thicknesses; runtime certifies hard-part geometry and requires bilateral
    contact before force control may enter the modelled interval.
    """
    opened,body=dict(open_point),dict(body_point)
    measured=fit_width_points([body,opened])
    n0,w0=float(body['normalized']),float(body['width_mm'])
    n1,w1=float(opened['normalized']),float(opened['width_mm'])
    if not n0<n1 or not w0<w1:raise ValueError('瓶身接触点与全开点顺序不合法')
    slope=(w1-w0)/(n1-n0)
    limit=2*float(cfg['geometry']['gripper']['pad_depth_mm'])
    hard_min=float(cfg['hardware']['public_ros']['raw_normalized_min'])
    if not 0<limit<w0:raise ValueError('双指允许压缩窗口必须为正且小于瓶身直径')
    lower=max(hard_min,n0-limit/slope)
    if lower>=n0-1e-9:raise ValueError('瓶身接触点已到现有归一化下限，没有允许的硅胶压缩行程；不能扩大原操作范围')
    equivalent=w0+(lower-n0)*slope
    point={'normalized':lower,'width_mm':equivalent,'kind':'compression_equivalent_estimate',
        'contact_normalized':n0,'contact_width_mm':w0,'max_compression_mm':limit,
        'normalized_hard_min':hard_min,'evidence':'bounded continuation of measured body/open slope; not measured loaded gel spacing'}
    points=[point]+measured['points']
    WidthCalibration(points)
    model={'schema':'bottle_compliant_aperture_v1','coordinate':'equivalent_unloaded_aperture_mm',
        'points':points,'measured_points':measured['points'],'slope_mm_per_normalized':slope,
        'estimated_normalized_range':[lower,n0],'estimated_width_range_mm':[equivalent,w0],
        'max_compression_mm':limit,'normalized_hard_min':hard_min,
        'precontact_min_mm':w0,'linearity_check_possible':False,'distinct_measured_widths':2,
        'scope':'bounded engineering model; compressed gap is not measured; hard-part geometry and force gates remain mandatory'}
    return model


class WidthBootstrap:
    def __init__(self,cfg,reader,ui,directory,*,publisher_factory=None,sleep=time.sleep,clock=time.monotonic):
        signs=np.asarray(cfg['tactile']['normal_signs'],float)
        if cfg['force']['unit']!='sdk' or cfg['tactile']['normal_axis']!=2 or signs.shape!=(2,) or not np.isin(signs,[-1.,1.]).all():
            raise ValueError('首次归一化开度标定需要已确认的 SDK Fz 与各指正负号 ±1')
        self.signs=signs.copy()
        self.cfg=cfg;self.reader=reader;self.ui=ui;self.path=Path(directory)
        self.publisher_factory=publisher_factory;self.sleep=sleep;self.clock=clock
        self.publisher=None;self.actuated=False;self.trace=[]
        self.floor=float(cfg['hardware']['public_ros']['raw_normalized_min'])
        self.step=float(cfg['hardware']['public_ros']['force_step_counts'])/1000.
        if not 0<self.step<=.002:raise ValueError('首次开度标定每步不得超过 0.002 归一化值')
    def publish(self,value):
        if not self.floor<=value<=self.open_value:raise RuntimeError('未标定归一化目标超出现有操作边界')
        guard_all_commands(self.cfg,own_nodes=current_ros_node_names())
        _,current,force=self.check()
        if value<current['normalized'] and np.max(force)>self.cfg['force']['probe_fraction']*self.cfg['force']['target_N']:
            raise RuntimeError('发布前最新反馈已超过 PROBE，拒绝进一步闭合')
        if abs(value-current['normalized'])>self.step*2+.0001:raise RuntimeError('开度反馈未跟随，拒绝继续推进')
        if self.publisher_factory:self.publisher.publish(float(value))
        else:
            message=self.reader.grip._message_type();message.data=float(value);self.publisher.publish(message)
        self.actuated=True
        with self.reader.lock:
            if self.reader.writer:
                stamp=current.get('monotonic_ns',time.monotonic_ns())
                self.reader.writer.h5.append('bootstrap_gripper_normalized',np.array(current['normalized']),stamp,
                    attrs={'source_sequence':current['source_sequence'],'unit':'normalized','simulated':False})
                self.reader.writer.h5.append('bootstrap_gripper_command',np.array(value),time.monotonic_ns(),
                    attrs={'unit':'normalized','speed_unit':'unavailable_until_width_mapping','simulated':False})
        self.trace.append({'monotonic_s':self.clock(),'measured_normalized':current['normalized'],
            'commanded_normalized':float(value),'normal_force_sdk':force.tolist(),'normal_signs':self.signs.tolist()})
    def check(self):
        raw=self.reader.raw();grip=self.reader.gripper()
        diff=pose_error(self.start_pose,self.reader.pose())
        if diff[0]>self.cfg['motion']['pose_tolerance_mm'] or diff[1]>self.cfg['motion']['pose_tolerance_deg']:
            raise RuntimeError('开度标定期间 TCP 改变；不发布机械臂命令，停止闭合')
        delta=np.asarray(raw['wrench_raw'])[:,2]-self.offset
        if not np.isfinite(delta).all():raise RuntimeError('原始法向力不合法')
        # Magnitude guard must work even when configured polarity is wrong.
        if np.any(np.abs(delta)>self.cfg['force']['max_N']):raise RuntimeError('开度标定逐指 |ΔFz| 超过 F_max；停止')
        force=delta*self.signs
        reverse_limit=max(3*self.cfg['force']['contact_min_N'],getattr(self,'reverse_noise_limit',0.))
        if np.any(force < -reverse_limit):
            raise RuntimeError('法向力方向与配置不符，停止闭合；原始 ΔFz='+str(delta.tolist())+
                ' SDK；normal_signs='+str(self.signs.tolist())+'；需核对当前数据通道，不能继续夹紧')
        return raw,grip,force
    def leg_timeout_s(self):
        # The physical feedback advances more slowly than 40 ms submissions.
        # Budget the bounded full stroke without changing increment or speed.
        steps=(self.open_value-self.floor)/self.step
        return min(300.,max(120.,steps*.25+10.))
    def measure(self,label,width_mm,open_value):
        self.path.mkdir(parents=True,exist_ok=False)
        self.open_value=float(open_value);self.start_pose=self.reader.pose()
        if not .998<=self.open_value<=1.:
            raise RuntimeError('当前归一化张开读数='+str(self.open_value)+'，请先全开到 1.0；不能把上次失败的残留开度当作全开标定点')
        guard_all_commands(self.cfg,own_nodes=current_ros_node_names())
        self.ui.confirm(f'{label}：瓶子由桌面支撑、夹爪已完全张开，输入 EMPTY 确认空载采样：','EMPTY')
        first=self.reader.gripper()
        if abs(first['normalized']-self.open_value)>.002:raise RuntimeError('开度不是此前测量的张开值')
        baseline=self.reader.noise();self.offset=np.asarray(baseline['wrench_mean'])[:,2]
        self.reverse_noise_limit=6*max(baseline['normal_std'])
        if max(baseline['normal_std'])>=self.cfg['zero_guard'].get('force_std_max',self.cfg['zero_guard'].get('force_std_max_N')):
            raise RuntimeError('开度标定的空载力波动过大')
        self.check()
        self.ui.confirm('本步只慢速闭合/回开夹爪，不移动机械臂；输入 MOVE：','MOVE')
        # Revalidate pose and feedback AFTER the human wait, BEFORE publisher creation.
        self.check()
        if abs(self.reader.gripper()['normalized']-self.open_value)>.002:raise RuntimeError('确认期间开度改变')
        if self.publisher_factory:self.publisher=self.publisher_factory()
        else:self.publisher=self.reader.grip._ros.Publisher('/'+self.cfg['hardware']['arm_side']+'_gripper_command',self.reader.grip._message_type,queue_size=1)
        result=None;release_complete=False;recording_complete=False
        try:
            with AsyncRecorder(self.path,self.cfg) as writer:
                with self.reader.lock:self.reader.writer=writer
                try:
                    self.reader.sensors.set_context(stage='PROBE')
                    last_pair=None;deadline=self.clock()+self.leg_timeout_s()
                    while self.clock()<deadline:
                        raw,grip,force=self.check()
                        pair=tuple(raw['source_sequences']['wrench_raw_'+s] for s in ('left','right'))
                        if pair==last_pair:self.sleep(.01);continue
                        last_pair=pair
                        if np.all(force>=self.cfg['force']['contact_min_N']):
                            self.publish(grip['normalized'])
                            result={'normalized':float(grip['normalized']),'width_mm':float(width_mm),
                                'part':label,'force_sdk':force.tolist(),'normal_signs':self.signs.tolist(),
                                'raw_delta_Fz_sdk':(force*self.signs).tolist(),'force_source_fids':list(pair),
                                'T_base_tcp_mm':self.start_pose.tolist(),'bias':'first bilateral force contact, possible gel compression'}
                            break
                        probe=self.cfg['force'].get('probe_fraction',.3)*self.cfg['force']['target_N']
                        if np.max(force)>probe:raise RuntimeError('单侧力超过 PROBE，停止，未用另一指继续推动瓶子')
                        proposed=float(grip['normalized'])-self.step
                        if proposed<self.floor:raise RuntimeError('到现有归一化操作下限仍未双侧接触；不能伪造开度点')
                        self.publish(proposed);self.sleep(.04)
                    if result is None:raise RuntimeError('开度标定闭合超时；当前归一化开度='+str(grip['normalized'])+'；法向力='+str(force.tolist())+' SDK；请核对反馈与接触，不自动扩大行程')
                    deadline=self.clock()+self.leg_timeout_s()
                    while self.clock()<deadline:
                        _,grip,_=self.check()
                        value=float(grip['normalized'])
                        if abs(value-self.open_value)<.0005:break
                        self.publish(min(self.open_value,value+self.step));self.sleep(.04)
                    else:raise RuntimeError('开度标定回开超时')
                    release_complete=True
                finally:
                    with self.reader.lock:self.reader.writer=None
                    self.reader.sensors.set_context(stage='IDLE')
            recording_complete=True
        except BaseException:
            # Only a fresh measured hold is allowed on failure, never a guessed
            # reopen after a source fault or an arm command from this tool.
            if self.actuated:
                try:self.publish(self.reader.gripper()['normalized'])
                except BaseException:pass
            self.ui.say('开度标定已停止；夹爪可能仍接触瓶子，请现场确认后释放。没有移动机械臂。')
            raise
        finally:
            atomic_json(self.path/'calibration.json',{'result':result,'commands':self.trace,'zero':baseline,
                'simulated':False,'arm_commands':False,'release_complete':release_complete,'recording_complete':recording_complete,'normalized_lower_bound':self.floor,
                'normal_signs':self.signs.tolist(),'leg_timeout_s':self.leg_timeout_s()})
            if self.publisher is not None:self.publisher.unregister()
        return result
