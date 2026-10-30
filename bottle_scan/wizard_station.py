"""Read-only station discovery and operator-entered physical configuration.

No command publisher in this module. Unknown physical values require explicit
terminal input; synthetic defaults are never used for real hardware.
"""
from __future__ import annotations
import copy
import hashlib
import itertools
import json
from pathlib import Path
import threading
import time
import xml.etree.ElementTree as ET
import numpy as np
from .session import atomic_json,plain,fingerprint
from .geometry import axis_basis,gripper_boxes,_box_vertices


def numeric(ui,label,default=None,count=1,*,positive=True):
    while True:
        hint='' if default is None else ' [回车采用 '+str(default)+']'
        value=ui.ask(label+hint+'：')
        if not value and default is not None:
            result=np.asarray(default if isinstance(default,(list,tuple)) else [default],float)
        else:
            try:result=np.asarray([float(x) for x in value.replace(',',' ').split()],float)
            except ValueError:result=np.asarray([])
        if result.shape==(count,) and np.isfinite(result).all() and (not positive or np.all(result>0)):
            return float(result[0]) if count==1 else result.tolist()
        ui.say(f'需要 {count} 个有限'+('正数' if positive else '数字')+'，以空格分隔；q 可中断保存。')
        if value.lower()=='q':raise KeyboardInterrupt('操作员中断设置')


def get(cfg,path):
    out=cfg
    for key in path.split('.'):out=out[key]
    return out


def set_value(cfg,path,value):
    out=cfg
    keys=path.split('.')
    for key in keys[:-1]:out=out.setdefault(key,{})
    out[keys[-1]]=value


def known(value):
    return value is not None and not (isinstance(value,str) and 'TODO' in value.upper())


def fill(cfg,ui,path,label,default=None,count=1,positive=True):
    current=get(cfg,path)
    if known(current):return current
    value=numeric(ui,label,default,count,positive=positive)
    set_value(cfg,path,value)
    return value


class LiveReader:
    """Subscriptions + X5 transports only. Idle frames are explicitly discarded."""
    def __init__(self,cfg):
        self.cfg=cfg;self.side=cfg['hardware']['arm_side']
        self.arm=self.grip=self.sensors=None;self.stop=threading.Event();self.worker=None
        self.error=None;self.lock=threading.Lock();self.writer=None;self.stats={}
    def __enter__(self):
        from .hardware_arm import RosArmObserver
        from .hardware_gripper import RosNormalizedGripperObserver
        from .hardware_sensors import ActualX5SensorSource
        try:
            self.arm=RosArmObserver(arm_side=self.side,base_frame=self.cfg['hardware']['base_frame'])
            self.arm.connect()
            self.grip=RosNormalizedGripperObserver(side=self.side);self.grip.connect()
            observation_cfg=copy.deepcopy(self.cfg)
            observation_cfg['hardware']['public_ros']['sensors'].pop('force_conversion',None)
            self.sensors=ActualX5SensorSource(observation_cfg);self.sensors.start()
            self.worker=threading.Thread(target=self._drain,daemon=True);self.worker.start()
            deadline=time.monotonic()+5
            while not self.grip.snapshot()['fresh'] and time.monotonic()<deadline:time.sleep(.03)
            self.pose();self.gripper()
            return self
        except BaseException:
            self.__exit__(None,None,None);raise
    def _drain(self):
        try:
            while not self.stop.wait(.005):
                events=self.sensors.drain_samples()
                with self.lock:
                    for sample in events:
                        row=self.stats.setdefault(sample.stream,{'count':0,'first_ns':sample.monotonic_ns,'last_ns':sample.monotonic_ns,'bytes':0})
                        row['count']+=1;row['last_ns']=sample.monotonic_ns;row['bytes']+=np.asarray(sample.data).nbytes
                        if self.writer:
                            if sample.stream.startswith('camera/'):
                                arms=self.arm.snapshot()['arms']
                                attrs=dict(sample.metadata or {})
                                attrs.update(T_base_leftTCP=arms['left']['T_base_tcp_mm'],T_base_rightTCP=arms['right']['T_base_tcp_mm'],
                                    bottle_frame_status='unavailable_before_daily_T0',pose_sources={side:arms[side] for side in ('left','right')})
                                # Source JPEG bytes/metadata are retained by the same async writer API.
                                self.writer.jpg.append(sample.stream.split('/',1)[1],sample.data,sample.monotonic_ns,
                                    sample.device_timestamp,attrs=attrs)
                            else:self.writer.h5.append(sample.stream,sample.data,sample.monotonic_ns,sample.device_timestamp,attrs=sample.metadata)
        except BaseException as exc:self.error=exc
    def pose(self):
        if self.error:raise RuntimeError('只读采集失败：'+str(self.error))
        return np.asarray(self.arm.snapshot()['arms'][self.side]['T_base_tcp_mm'],float)
    def gripper(self):
        state=self.grip.snapshot()
        if not state.get('fresh') or not state.get('valid'):raise RuntimeError('夹爪归一化开度不新鲜')
        return state
    def raw(self):
        self.pose();self.gripper();return self.sensors.observation_snapshot()
    def images(self):return self.sensors.images()
    def noise(self,count=100):
        rows=[];last=None;deadline=time.monotonic()+12
        while len(rows)<count and time.monotonic()<deadline:
            obs=self.raw();ids=tuple(obs['source_sequences']['wrench_raw_'+s] for s in ('left','right'))
            if last is None or all(a!=b for a,b in zip(ids,last)):
                rows.append((obs['wrench_raw'].copy(),[d[:,:,self.cfg['tactile']['depth_channel']].copy() for d in obs['dd']],ids))
                last=ids
            time.sleep(.01)
        if len(rows)<count:raise RuntimeError('空载统计没有获得 100 个独立双指帧')
        wrench=np.asarray([x[0] for x in rows]);dd=np.asarray([x[1] for x in rows])
        dd_noise=float(np.quantile(np.abs(dd-np.median(dd,axis=0)),.999))
        with self.lock:stats=copy.deepcopy(self.stats)
        rates={k:(v['count']-1)/((v['last_ns']-v['first_ns'])/1e9) for k,v in stats.items() if v['count']>2 and v['last_ns']>v['first_ns']}
        return {'wrench_mean':wrench.mean(axis=0).tolist(),'normal_std':wrench[:,:,2].std(axis=0).tolist(),
            'wrench_samples_raw':wrench.tolist(),'dd_noise_q999':dd_noise,'rates':rates,'stream_stats':stats,'source_fids':[list(x[2]) for x in rows]}
    def __exit__(self,*args):
        self.stop.set()
        if self.worker:self.worker.join(timeout=3)
        for component in (self.sensors,self.grip,self.arm):
            if component is not None:component.close()


def read_model(reader,cfg):
    urdf=reader.arm._rospy.get_param('/robot_description')
    digest=hashlib.sha256(urdf.encode()).hexdigest();xml=ET.fromstring(urdf)
    meshes={}
    for link in xml.findall('link'):
        name=link.get('name','')
        if not any(word in name for word in ('gripper','finger','pad','camera')):continue
        for mesh in link.findall('.//mesh'):
            uri=mesh.get('filename','');resolved=None
            if uri.startswith('package://'):
                part=uri[len('package://'):];package,relative=part.split('/',1)
                candidates=[Path('/home/dm/DM_das/tianji_lerobot_replay/models/ros_packages')/package/relative]
                try:
                    import rospkg
                    candidates.insert(0,Path(rospkg.RosPack().get_path(package))/relative)
                except Exception:pass
                resolved=next((p for p in candidates if p.exists()),None)
            elif uri.startswith('file://'):resolved=Path(uri[7:])
            elif Path(uri).is_absolute():resolved=Path(uri)
            if resolved is None or not resolved.is_file():raise ValueError('无法读取模型网格：'+uri)
            meshes[uri]=hashlib.sha256(resolved.read_bytes()).hexdigest()
    snap=reader.arm.snapshot()
    frame={s:np.asarray(snap['arms'][s]['T_flange_tcp_mm']).round(8).tolist() for s in ('left','right')}
    images=reader.images()
    model={'urdf_sha256':digest,'robot_name':xml.get('name'),'mesh_sha256':meshes,
        'T_flange_tcp_mm':frame,'camera_shapes':{k:list(v.shape) for k,v in images.items()},
        'serials':copy.deepcopy(cfg['hardware']['sensor_serials']),
        'tcp_frames':{s:snap['arms'][s]['tcp_frame'] for s in ('left','right')}}
    audit_path=Path(__file__).resolve().parents[1]/'outputs/auto_calibration_20261028/geometry_candidates.json'
    candidates=json.loads(audit_path.read_text()) if audit_path.exists() else {}
    if candidates.get('urdf_sha256')==digest:model['geometry_candidates']=candidates
    # Only static definitions participate; WBC positions and jaw opening do not.
    return model


def model_identity(model):
    return {k:model[k] for k in ('urdf_sha256','mesh_sha256','T_flange_tcp_mm','camera_shapes','serials','tcp_frames')}


def configure_geometry(cfg,model,ui):
    candidate=model.get('geometry_candidates',{})
    axes=copy.deepcopy(candidate.get('candidate_axes_in_tcp',{}))
    if all(a in axes for a in ('a','c','l')):
        reference=np.asarray(cfg['T0'],float)
        if reference.shape==(4,4) and float((reference[:3,:3]@np.asarray(axes['l']))[2])<0:
            axes['c']=(-np.asarray(axes['c'])).tolist();axes['l']=(-np.asarray(axes['l'])).tolist()
            ui.say('c 的符号约定取反，使 +l 在已记录 T0 处朝瓶盖方向；a 保持基座到指尖方向。')
    ui.say('模型：'+model['robot_name']+'；TCP='+model['tcp_frames'][cfg['hardware']['arm_side']])
    ui.say('G 使用已确认的灰色短段近似，不改 TCP 原点；必须核对完整开合范围和实际传感器凸起。')
    for axis in ('a','c','l'):
        if not known(cfg['axes'][axis]) and axis in axes:
            cfg['axes'][axis]=axes[axis]
        else:fill(cfg,ui,'axes.'+axis,axis+' 轴在 TCP 中的 xyz',None,3,False)
    basis=axis_basis(cfg['axes'])
    geometry=candidate.get('collision_meshes',{})
    def acl_bounds(item):
        points=np.array(list(itertools.product(*zip(item['tcp_bbox_min_mm'],item['tcp_bbox_max_mm']))))@basis
        return points.min(0),points.max(0)
    fingers=[acl_bounds(v) for k,v in geometry.items() if 'finger_' in k and '_pad_' not in k]
    palm=candidate.get('palm_including_camera_tcp_bbox_mm')
    pdefault=None
    if palm:
        points=np.array(list(itertools.product(*zip(palm['min'],palm['max']))))@basis
        pdefault=(points.min(0),points.max(0))
    if fingers:
        size=np.max([hi-lo for lo,hi in fingers],axis=0).tolist()
        center=np.mean([(hi+lo)/2 for lo,hi in fingers],axis=0)
    else:size=center=None
    if not known(cfg['geometry']['gripper']['finger_size_acl_mm']) and (size) is not None:
        cfg['geometry']['gripper']['finger_size_acl_mm']=size
    else:
        fill(cfg,ui,'geometry.gripper.finger_size_acl_mm','完整硬指体包络尺寸 a c l(mm)，可修改 CAD 建议',size,3)
    if not known(cfg['geometry']['gripper']['finger_center_a_mm']) and (None if center is None else float(center[0])) is not None:
        cfg['geometry']['gripper']['finger_center_a_mm']=None if center is None else float(center[0])
    else:
        fill(cfg,ui,'geometry.gripper.finger_center_a_mm','硬指体中心 a(mm)',None if center is None else float(center[0]),positive=False)
    if not known(cfg['geometry']['gripper']['finger_center_l_mm']) and (None if center is None else float(center[2])) is not None:
        cfg['geometry']['gripper']['finger_center_l_mm']=None if center is None else float(center[2])
    else:
        fill(cfg,ui,'geometry.gripper.finger_center_l_mm','硬指体中心 l(mm)',None if center is None else float(center[2]),positive=False)
    if not known(cfg['geometry']['gripper']['palm_center_acl_mm']) and (None if pdefault is None else ((pdefault[0]+pdefault[1])/2).tolist()) is not None:
        cfg['geometry']['gripper']['palm_center_acl_mm']=None if pdefault is None else ((pdefault[0]+pdefault[1])/2).tolist()
    else:
        fill(cfg,ui,'geometry.gripper.palm_center_acl_mm','手掌和相机联合包络中心 a c l(mm)',None if pdefault is None else ((pdefault[0]+pdefault[1])/2).tolist(),3,False)
    if not known(cfg['geometry']['gripper']['palm_size_acl_mm']) and (None if pdefault is None else (pdefault[1]-pdefault[0]).tolist()) is not None:
        cfg['geometry']['gripper']['palm_size_acl_mm']=None if pdefault is None else (pdefault[1]-pdefault[0]).tolist()
    else:
        fill(cfg,ui,'geometry.gripper.palm_size_acl_mm','手掌和相机联合包络尺寸 a c l(mm)',None if pdefault is None else (pdefault[1]-pdefault[0]).tolist(),3)
    fill(cfg,ui,'geometry.gripper.pad_depth_mm','实际允许接触的软胶厚度(mm)，不能用橙色原厂 pad 代替')
    fill(cfg,ui,'geometry.gripper.pad_size_al_mm','实际有效胶面尺寸 a l(mm)，只能豁免这块软胶',None,2)
    g=cfg['geometry']['gripper']
    pad_center=basis.T@np.asarray(cfg['tool']['grasp_center_in_tcp'],float)[:3,3]
    g['pad_center_a_mm']=float(pad_center[0]);g['pad_center_l_mm']=float(pad_center[2])
    # Union the CAD hard envelope and the operator-declared active gel rectangle;
    # never place the pad at the long whole-finger midpoint or use orange pads.
    for dim,field,pad_dim in [(0,'finger_center_a_mm',0),(2,'finger_center_l_mm',1)]:
        lo=g[field]-g['finger_size_acl_mm'][dim]/2;hi=g[field]+g['finger_size_acl_mm'][dim]/2
        lo=min(lo,pad_center[dim]-g['pad_size_al_mm'][pad_dim]/2)
        hi=max(hi,pad_center[dim]+g['pad_size_al_mm'][pad_dim]/2)
        g[field]=float((lo+hi)/2);g['finger_size_acl_mm'][dim]=float(hi-lo)
    if g['pad_depth_mm']>=g['finger_size_acl_mm'][1] or np.any(np.asarray(g['pad_size_al_mm'])>np.asarray(g['finger_size_acl_mm'])[[0,2]]):
        raise ValueError('软胶豁免必须位于硬指体包络内；请重新输入尺寸，未创建控制器')
    ui.say(json.dumps({'axes':cfg['axes'],'G':cfg['tool']['grasp_center_in_tcp'],'gripper':cfg['geometry']['gripper']},ensure_ascii=False))
    ui.confirm('确认包络覆盖实物与允许的整个开合范围、胶面与外壳区分正确，输入 GEOMETRY：','GEOMETRY')
    cfg['hardware']['axis_mapping_declared']=True


def configure_noise_force(cfg,noise,ui):
    sigma=max(noise['normal_std']);tiny=1e-9
    ui.say('力来源：现有公开 ROS/X5 39020，SDK 单位；F_target='+cfg['force']['target_definition']+'。不把旧来源的 gf/N 数值直接复制成 39020 目标。')
    ui.confirm('已按当前 39020 实机接触记录确认：两指受压原始 Fz 都减小；沿用 index=2、normal_signs=-1/-1，使接触力为正，回车确认：')
    cfg['tactile'].update(normal_axis=2,normal_signs=[-1.,-1.],sensor_to_tcp={'left':'unavailable','right':'unavailable'})
    cfg['lower']['vertical_mapping_confirmed']=False
    cfg['hardware']['public_ros']['sensors']['force_conversion'].update(normal_axis=2,normal_signs={'left':-1.,'right':-1.})
    fill(cfg,ui,'force.contact_min_N','接触阈值(SDK)，建议不小于空载噪声 5σ',max(5*sigma,tiny))
    fill(cfg,ui,'force.target_N','实际 39020 的 F_target(SDK)，必须填写已知可接受目标')
    fill(cfg,ui,'force.max_N','逐指 F_max(SDK)，必须填写接受的最大值')
    fill(cfg,ui,'force.target_tolerance_N','目标力容差(SDK)')
    fill(cfg,ui,'hardware.public_ros.force_deadband_N','同来源 SDK 力环死区',cfg['force']['target_tolerance_N']/2)
    while not (0<cfg['force']['contact_min_N']<cfg['force']['probe_fraction']*cfg['force']['target_N']
            and cfg['force']['target_N']+cfg['force']['target_tolerance_N']<cfg['force']['max_N']):
        ui.say('需要 0 < 接触阈值 < PROBE，且 F_target+容差 < 逐指 F_max；请重新输入。')
        for key,label in [('contact_min_N','接触阈值'),('target_N','F_target'),('max_N','逐指 F_max'),('target_tolerance_N','目标容差')]:
            cfg['force'][key]=numeric(ui,label+'(SDK)',cfg['force'][key])
    cfg['force']['validation']['evaluation_target']=cfg['force']['target_N']
    for path,label,default in [
        ('zero_guard.force_std_max','空载标准差验收上限',max(3*sigma,tiny)),
        ('zero_guard.offset_delta_warn','换次扣零变化警告阈值',max(10*sigma,tiny)),
        ('zero_guard.dd_contact_threshold','DD 空载接触阈值',max(2*noise['dd_noise_q999'],tiny)),
        ('zero_guard.dd_contact_max_fraction','空载允许接触像素比例',.001),
        ('tactile.dd_threshold','DD 描述子阈值',max(2*noise['dd_noise_q999'],tiny)),
        ('tactile.centroid_drift_px','HOLD 疑似滑动质心漂移阈值(px)',None)]:
        fill(cfg,ui,path,label,default)
    cfg['analysis']['dd_threshold']=cfg['tactile']['dd_threshold']
    cfg['analysis']['centroid_to_al']={'left':'unavailable','right':'unavailable'}
    # Unknown 3D sensor rotation remains unavailable. Norm-based unloading is
    # invariant to tangent rotation; no world-z force or CoP is fabricated.
    cfg['hardware']['sensor_mapping_declared']=True
    for key,label,default in [('max_contact_peak','接触峰值验收上限(SDK)',cfg['force']['max_N']),
        ('max_settle_s','夹紧稳定时间验收上限(s)',3.),('max_steady_std','HOLD 标准差验收上限(SDK)',cfg['force']['target_tolerance_N'])]:
        fill(cfg,ui,'force.validation.'+key,label,default)


def configure_policies(cfg,noise,ui):
    # These are explicitly proposed software limits, never measured hardware
    # properties. The operator confirms them before any motion and the same
    # configuration is used for stopping/force validation.
    policies={'motion.free_speed_mm_s':5.,'motion.near_speed_mm_s':2.,'motion.lift_speed_mm_s':2.,
        'motion.angular_speed_deg_s':2.,'motion.pose_tolerance_mm':1.,'motion.pose_tolerance_deg':1.,
        'safety.margin_mm':2.,'safety.max_step_mm':150.,'safety.max_rotation_step_deg':30.,
        'gripper.open_tolerance_mm':.5,'gripper.close_speed_mm_s':1.,'gripper.stall_epsilon_mm':.05,
        'check.min_score':.8,'check.search_radius_px':80.,'preparation.height_crosscheck_tolerance_mm':5.,
        'close.max_overtravel_mm':.5,'stop_test.trigger_s':.2,'stop_test.settle_window_s':.5,
        'stop_test.settle_translation_mm':.5,'stop_test.settle_rotation_deg':.5,
        'stop_test.timeout_s':5.,'stop_test.max_extra_mm':1.}
    proposed={p:v for p,v in policies.items() if not known(get(cfg,p))}
    for stage in cfg['motion']['stage_timeouts_s']:
        path='motion.stage_timeouts_s.'+stage
        if not known(get(cfg,path)):proposed[path]=120.
    ui.say('以下是待确认的软件限值/验收界限，不称为硬件实测：'+json.dumps(proposed,ensure_ascii=False))
    # One review plus optional edits; no silently imported demo defaults.
    if proposed:
        answer=ui.ask('回车接受以上软件限值，edit 逐项修改，q 取消：')
        if answer=='q':raise KeyboardInterrupt('未确认限值')
        if answer not in ('','edit'):raise ValueError('未确认软件限值')
        for path,value in proposed.items():
            set_value(cfg,path,numeric(ui,path,value) if answer=='edit' else value)
    rates=noise['rates'];stats=noise['stream_stats']
    def observed(prefix):
        values=[v for k,v in rates.items() if k.startswith(prefix)]
        if not values:raise ValueError('无法实测来源频率：'+prefix)
        return min(values)
    cfg['close']['observed_force_fps']=observed('wrench_raw_')
    cfg['close']['observed_dd_fps']=observed('dd_')
    cfg['tactile']['fps']=cfg['close']['observed_dd_fps']
    cfg['sample_hz']=cfg['recording']['control_hz']
    image_sizes=[]
    for camera in ('left_wrist','right_wrist'):
        key='camera/'+camera
        cfg['cameras'][camera]['fps']=observed(key)
        data=stats[key];image_sizes.append(data['bytes']/data['count'])
    cfg['estimated_bytes_per_image']=int(max(image_sizes))
    cfg.setdefault('wizard',{}).update(zero_drift_threshold_mm=1.5,visual_orientations_deg=[0.,90.,180.,270.])
    cfg['check']['threshold_px']='calibrate_at_session_start'
    cfg['check']['enabled_cameras']=['right_wrist']


def teach_real(cfg,reader,ui):
    ui.confirm('夹爪张开，把最低指尖移到桌面上方约 10 mm（确认在 0～20 mm 内且未触桌），停止遥操后回车记录：')
    pose=reader.pose();boxes=gripper_boxes(pose,cfg,cfg['gripper']['open_mm'])
    finger=[b for b in boxes if b['name']!='palm']
    table=min(_box_vertices(b)[:,2].min() for b in finger)-10.
    # Geometry uses the upper table bound; nominal height remains separate.
    cfg['safety']['table_z_mm']=float(table+10.)
    cfg['geometry']['bottle']['axial_uncertainty_mm']=10.
    points=[]
    for role,reduce in [('下界角','min'),('上界角','max')]:
        ui.confirm(f'将完整工具包络放在已确认无障碍工作空间的{role}，停止遥操后回车记录：')
        pose=reader.pose();vertices=np.concatenate([_box_vertices(b) for b in gripper_boxes(pose,cfg,cfg['gripper']['open_mm'])])
        value=vertices.min(0) if reduce=='min' else vertices.max(0)
        points.append({'role':role,'tcp':pose.tolist(),'boundary_base_mm':value.tolist()})
    low,high=[np.asarray(x['boundary_base_mm']) for x in points]
    if np.any(high<=low):raise ValueError('示教上下角顺序无效')
    cfg['safety'].update(workspace_min_mm=low.tolist(),workspace_max_mm=high.tolist())
    ui.say(f'桌面 z={table:.2f} mm；工作空间 '+str(low.round(1))+' 到 '+str(high.round(1))+'；安全余量由原几何模块仅计入一次。')
    ui.confirm('确认该矩形空间内部也无障碍（两个角本身不能证明无障碍），输入 SPACE：','SPACE')
    ui.confirm('移至停止测试的自由空位，夹爪张开，停止遥操后按回车记录：')
    free=reader.pose();cfg['preparation']['axis_test_pose']=free.tolist()
    # Direction comes from the measured free TCP orientation and confirmed -a.
    cfg['stop_test']['direction_base']=(-free[:3,:3]@np.asarray(cfg['axes']['a'])).tolist()
    return {'nominal_table_z_mm':float(table),'conservative_table_z_mm':float(table+10.),'table_gap_bounds_mm':[0.,20.],'points':points,'free_pose':free.tolist(),'margin_application':'existing geometry once'}
