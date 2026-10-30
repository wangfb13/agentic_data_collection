"""Offline prototype of the setup/run operator experience.

This delivery deliberately has no real backend selection. Existing expert
commands retain their own physical gates. Simulation can never certify hardware.
"""
from __future__ import annotations
import argparse
from collections import Counter
from contextlib import redirect_stdout
import copy
import datetime as dt
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
import yaml
from .backend import SimBackend
from .cli import dry_report
from .commissioning import build_acceptance, hardware_binding
from .config import demo_config, validate_config
from .design import _trial, commissioning_trials, visual_points
from .geometry import (axis_basis, check_segment, gripper_boxes, _box_vertices,
    pose_offset, translate_local, grasp_pose)
from .runtime import Runner, SafetyFault
from .session import Session, atomic_json, fingerprint, plain, boot_id
from .sensing import TemplateGuard
from .wizard_calibration import calibrate_and_center, calibrate_right_camera, SIDES

BANNER='模拟交互预览 · SIMULATION_ONLY · 不连接 ROS、不驱动实机'
LABELS={'':'','1':'','2':'t','3':'m','4':'d','5':'s','0':'x'}
LABEL_NAMES={'':'正常','t':'倾倒','m':'被推动','d':'掉落','s':'滑动/转动','x':'作废'}
MENU='回车/1 正常 | 2 倾倒 | 3 被推动 | 4 掉落 | 5 滑动/转动 | 0 作废 | n 备注 | u 撤销上一标签 | p 暂停 | q 本次后安全退出'
STEPS=('doctor','geometry','dimensions','teaching','force','stop','roi_and_hash')


class UI:
    def __init__(self,rehearsal=False,*,stream=None,read=None):
        self.rehearsal=rehearsal;self.stream=stream or sys.stdout
        self.read=read or input
    def say(self,message):print(message,file=self.stream,flush=True)
    def ask(self,message,default='',*,key=None):
        if self.rehearsal:
            self.say(message+(' '+str(default) if default else ' [回车]'))
            return str(default)
        self.say(message)
        return self.read('> ').strip()
    def confirm(self,message,token=''):
        while True:
            answer=self.ask(message,token)
            if answer==token:return
            if answer.lower() in ('q','quit','取消'):raise KeyboardInterrupt('操作员取消')
            self.say('未确认；请按提示输入，q 取消。')
    def numbers(self,message,default,count):
        while True:
            raw=self.ask(message,' '.join(map(str,default)))
            if not raw:raw=' '.join(map(str,default))
            try:
                values=np.asarray([float(x) for x in raw.replace(',',' ').split()])
                if len(values)==count and np.isfinite(values).all() and np.all(values>0):return values.tolist()
            except ValueError:pass
            self.say(f'请输入 {count} 个正数，以空格分隔，单位 mm。')


def save_yaml(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    with tmp.open('w') as f:
        yaml.safe_dump(plain(value),f,sort_keys=False,allow_unicode=True)
        f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)


def simulated_config():
    cfg=demo_config()
    cfg['gripper']['close_speed_mm_s']=20.
    cfg['motion'].update(free_speed_mm_s=30.,near_speed_mm_s=20.,lift_speed_mm_s=20.)
    cfg['recording']['control_hz']=10.
    cfg['sample_hz']=cfg['tactile']['fps']=10.
    cfg['probe']['settle_s']=.6
    cfg['close'].update(observed_force_fps=10.,observed_dd_fps=10.,max_overtravel_mm=4.)
    cfg['motion']['stage_timeouts_s']['SQUEEZE']=2.
    cfg['simulation']['max_speedup']=8.
    for cam in cfg['cameras'].values():cam.update(fps=10.,width=240,height=180)
    cfg['design'].update(other_axis_range_mm=[-5.,5.],bottle_axis_range_mm=[-10.,10.],
        other_axis_step_mm=5.,bottle_axis_step_mm=10.,repeats=1)
    cfg['visual']['count']=6
    # These two legacy fields remain for existing configuration validation only;
    # the new TemplateGuard explicitly compares right_wrist alone.
    cfg['check']['enabled_cameras']=['right_wrist']
    cfg['wizard']={'zero_drift_threshold_mm':1.5}
    return cfg


class WizardSimBackend(SimBackend):
    """Synthetic, pose-dependent calibration plant. Never opens sockets."""
    def __init__(self,cfg,T0):
        self.calibrating=False;self.check_origin=None;self.zero_drift_px=0.
        self.contact_origin=np.asarray(T0).copy()
        self.true_center_al=np.array([2.,-1.])
        super().__init__(cfg,T0)
    def images(self):
        import cv2
        images=super().images()
        if self.check_origin is not None:
            local=self.check_origin[:3,:3].T@(self.pose[:3,3]-self.check_origin[:3,3])
            acl=axis_basis(self.cfg['axes']).T@local
            shift=np.array([[-4.,0.,1.],[0.,-3.,1.]])@acl
            images['right_wrist']=cv2.warpAffine(images['right_wrist'],
                np.array([[1.,0.,shift[0]],[0.,1.,shift[1]]]),(240,180),borderValue=(45,45,45))
        return images
    def step(self,seconds):
        super().step(seconds)
        yy,xx=np.mgrid[:32,:48]
        for i,side in enumerate(SIDES):
            if self.snapshot_cache['normal_N'][i]<=.1:continue
            if self.calibrating:
                local=self.contact_origin[:3,:3].T@(self.pose[:3,3]-self.contact_origin[:3,3])
                acl=axis_basis(self.cfg['axes']).T@local
                J=np.array([[1.4,.15],[.1,1.3]]) if i==0 else np.array([[-1.3,.2],[.1,-1.4]])
                u,v=np.array([23.5,15.5])+J@(acl[[0,2]]-self.true_center_al)
            elif self.zero_drift_px:
                offsets=getattr(self,'offsets',[0,0,0]);u=24+offsets[0]*.2+self.zero_drift_px;v=16+offsets[2]*.1
            else:continue
            raw=self.snapshot_cache['dd'][i]
            # Mutate the same native arrays queued by SimBackend.step, so the
            # logged stream and live descriptor represent identical data.
            raw[:,:,0]=(np.exp(-((xx-u)**2/18+(yy-v)**2/14))*self.snapshot_cache['normal_N'][i]*.25).astype(np.float32)


def sim_doctor(cfg,output):
    errors=validate_config(cfg)
    backend=WizardSimBackend(cfg,np.asarray(cfg['T0']))
    try:
        if set(backend.image_samples())!={'left_wrist','right_wrist'}:errors.append('缺少腕部图像')
        snap=backend.snapshot()
        if not np.isfinite(snap['tcp']).all():errors.append('TCP 无效')
    finally:backend.close()
    report={'simulated':True,'physical_connectivity_checked':False,'command_publishers':[],
        'checks':['synthetic TCP','synthetic two wrist images','configuration validation'], 'errors':errors}
    atomic_json(output,report)
    if errors:raise ValueError('doctor 拒绝：'+'；'.join(errors))
    return report


def station_hash(profile):
    cfg=profile['config']
    return fingerprint({'simulation_only':True,'binding':hardware_binding(cfg),
        'model':profile.get('model'),'camera_check':profile.get('camera_check'),
        'safety':cfg['safety'],'bottle_dimensions':profile.get('dimensions'),'workflow':cfg.get('wizard'),
        'wizard_sources':{n:hashlib.sha256((Path(__file__).parent/n).read_bytes()).hexdigest()
            for n in ('wizard.py','wizard_calibration.py','wizard_analysis.py','sensing.py')}})


def require_profile(path):
    profile=yaml.safe_load(Path(path).read_text())
    if (profile.get('schema')!='bottle_station_simulation_v1' or not profile.get('simulation_only')
            or profile.get('completed_steps')!=list(STEPS) or not profile.get('ready')):
        raise ValueError('station_profile 未完成或不是本版模拟 profile；请重新 setup')
    if profile.get('hardware_hash')!=station_hash(profile):
        raise ValueError('硬件/相机/TCP/安全配置或控制源码哈希变化，请重新 setup；未连接后端')
    return profile


def select_roi(ui,images,path,default=None):
    import cv2
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    image=images['right_wrist']
    if not cv2.imwrite(str(path),cv2.cvtColor(image,cv2.COLOR_RGB2BGR)):raise OSError('保存原尺寸图失败')
    ui.say('右腕原尺寸图：'+str(path.resolve()))
    while True:
        if os.environ.get('DISPLAY') and not ui.rehearsal:
            roi=list(map(int,cv2.selectROI('right_wrist',cv2.cvtColor(image,cv2.COLOR_RGB2BGR),False,False)))
            cv2.destroyWindow('right_wrist')
        else:
            raw=ui.ask('框选瓶身区域，输入 x y w h（像素；左腕只记录）：',
                       ' '.join(map(str,default or [92,42,61,101])))
            try:roi=list(map(int,raw.split()))
            except ValueError:roi=[]
        try:
            TemplateGuard(images,{'right_wrist':roi},checked_cameras=['right_wrist'])
            preview=image.copy();x,y,w,h=roi
            cv2.rectangle(preview,(x,y),(x+w-1,y+h-1),(255,80,0),2)
            cv2.imwrite(str(path.with_name(path.stem+'_roi.png')),cv2.cvtColor(preview,cv2.COLOR_RGB2BGR))
            return roi
        except (ValueError,TypeError) as exc:ui.say('ROI 无效：'+str(exc))


def teach_bounds(cfg,ui):
    # Semantic faces, not min/max of arbitrary TCP samples. In the real adapter
    # these must be read-only observations of operator-taught clear boundary faces.
    ui.say('  示教点均为模拟夹具数据；实机版将读取机器人反馈。安全余量由原几何模块统一计入一次。')
    example=np.asarray(cfg['T0']).copy();example[2,3]=20.
    ui.confirm('把最低指尖移到桌面上方约 10 mm，按回车记录：')
    tip_boxes=[b for b in gripper_boxes(example,cfg,cfg['gripper']['open_mm']) if b['name']!='palm']
    tip_z=min(float(_box_vertices(b)[:,2].min()) for b in tip_boxes)
    table=tip_z-10.
    low=np.asarray(cfg['safety']['workspace_min_mm']);high=np.asarray(cfg['safety']['workspace_max_mm'])
    records=[{'role':'table_tip_10mm','tcp':example.tolist(),'gap_mm':10.,'lowest_finger_z_mm':tip_z}]
    for role,point in (('下界角',low),('上界角',high)):
        ui.confirm(f'将工具外包络边界示教到安全工作空间{role}，按回车记录：')
        records.append({'role':role,'boundary_base_mm':point.tolist(),'simulated':True})
    if np.any(high<=low) or not np.isfinite(table):raise ValueError('示教边界无效')
    cfg['safety']['table_z_mm']=table
    return {'records':records,'table_z_mm':table,'workspace_min_mm':low.tolist(),'workspace_max_mm':high.tolist(),
        'safety_margin_mm':cfg['safety']['margin_mm'],
        'margin_application':'raw boundary faces stored; existing check_pose expands tool and tests table margin ONCE',
        'scope':'operator-declared obstacle-free rectangular volume; corners alone cannot establish obstacle freedom'}


def setup(args,ui):
    profile_path=Path(args.profile);evidence=profile_path.parent/(profile_path.stem+'_setup')
    state_path=evidence/'checkpoint.yaml'
    evidence.mkdir(parents=True,exist_ok=True)
    if profile_path.exists():
        try:
            require_profile(profile_path)
        except ValueError:pass
        else:
            ui.say('setup 已完成且哈希一致；没有待做步骤，不发动作。');return 0
    state=yaml.safe_load(state_path.read_text()) if state_path.exists() else {
        'schema':'bottle_station_simulation_v1','simulation_only':True,'ready':False,
        'completed_steps':[],'config':simulated_config()}
    if state.get('simulation_only') is not True:raise ValueError('拒绝混用实机 checkpoint')
    if state.get('ready') and state.get('hardware_hash')!=station_hash(state):
        save_yaml(evidence/f'checkpoint_before_hash_change_{time.time_ns()}.yaml',state)
        state['completed_steps']=[];state['ready']=False
    cfg=state['config']
    ui.say(BANNER);ui.say('setup：每步完成即保存；同一条命令可以续接。')
    for number,step in enumerate(STEPS,1):
        if step in state['completed_steps']:
            ui.say(f'[{number}/7] 已保存，跳过：{step}');continue
        if step=='doctor':
            sim_doctor(cfg,evidence/'doctor.json');ui.say('[1/7] doctor 通过（模拟输入；未检测真实网络）。')
        elif step=='geometry':
            state['model']={'source':'synthetic fixture; not a live TF/URDF lookup',
                'axes':cfg['axes'],'G':cfg['tool']['grasp_center_in_tcp'],'gripper':cfg['geometry']['gripper']}
            ui.say('[2/7] 模拟模型：a=TCP+X，c=TCP+Y，l=TCP+Z，G=[0,0,0] mm。')
            ui.say('  手指 50×8×20 mm，手掌 16×110×24 mm；这些仅是模拟夹具，实机已有 G≈[0,0,117] mm 未被覆盖。')
            ui.confirm('轴向、抓取中心与外包络确认一次；回车接受此模拟模型：')
        elif step=='dimensions':
            ui.say('[3/7] 输入卡尺尺寸（本次演示使用合成夹具的默认值）。')
            defaults={'body':[36.,65.],'neck':[22.,15.],'cap':[28.,15.]};dimensions={}
            for key,title in (('body','瓶身'),('neck','瓶颈'),('cap','瓶盖')):
                diameter,height=ui.numbers(f'{title}直径 高度（mm）：',defaults[key],2)
                dimensions[key]={'diameter_mm':diameter,'height_mm':height}
            dimensions['open_mm']=ui.numbers('夹爪张开宽度（mm）：',[90.],1)[0]
            state['dimensions']=dimensions
            # The preview plant is the documented fixed fixture. Unknown/custom
            # dimensions must not silently alter only one half of that plant.
            if any(dimensions[k]!={'diameter_mm':v[0],'height_mm':v[1]} for k,v in defaults.items()) or dimensions['open_mm']!=90.:
                raise ValueError('本轮交互预览仅支持上列模拟夹具尺寸；实机尺寸请等接入后输入，不写入真实配置')
            state['width_mapping']={'points':[{'normalized':1.,'width_mm':90.},{'normalized':.4,'width_mm':36.}],
                'simulated':True,'linearity_verified':False,'reason':'只有两种独立宽度，不能检查线性；实机接触读数尚未接入'}
            ui.say('  开度配对：2 个不同宽度点；不足以检验线性，已记录。')
        elif step=='teaching':
            ui.say('[4/7] 示教桌面与工作空间。');state['teaching']=teach_bounds(cfg,ui)
            ui.say(f"  桌面 z={cfg['safety']['table_z_mm']:.1f} mm；安全余量 {cfg['safety']['margin_mm']:.1f} mm，统一计入一次。")
        elif step=='force':
            ui.say('[5/7] 预填来源：bottle_scan/config.py 的 demo_config/force_defaults（模拟）。')
            ui.say(f"  单位={cfg['force']['unit']}，F_target 定义={cfg['force']['target_definition']}，目标={cfg['force']['target_N']}，每指上限={cfg['force']['max_N']}。")
            ui.say('  真实公开 ROS 配置仍使用 SDK 单位，本次不会改成 N。')
            ui.confirm('确认以上模拟力设置，按回车：')
        elif step=='stop':
            ui.say('[6/7] 自由空间停止测试：软件停止×3、SIGKILL×3（仅模拟子进程）。')
            from .stop_validation import run_stop_test
            report_path=evidence/'stop_report.json'
            # A partially written stop report is preserved before a full retry.
            if report_path.exists():report_path.rename(evidence/f'stop_interrupted_{time.time_ns()}.json')
            report=run_stop_test(cfg,simulate=True,output_path=report_path,output=lambda s:None)
            if not report['passed']:raise ValueError('模拟停止测试失败，setup 停止；见 '+str(report_path))
            ui.say('  6/6 通过；报告：'+str(report_path))
            ui.confirm('核对停止测试摘要；输入 PASS：','PASS')
            state['stop_report']=str(report_path.resolve())
        elif step=='roi_and_hash':
            ui.say('[7/7] 到 T_check 并设置一次右腕检查区域。')
            backend=WizardSimBackend(cfg,np.asarray(cfg['T0']))
            temp=evidence/'roi_session'
            if temp.exists():temp.rename(evidence/f'roi_interrupted_{time.time_ns()}')
            session=Session.create(temp,cfg,np.asarray(cfg['T0']),simulated=True)
            runner=WizardRunner(session,backend,ui)
            target=translate_local(session.T0,-cfg['check']['retreat_mm'],'a',cfg['axes'])
            backend.check_origin=target.copy()
            try:
                with runner.phase('CHECK'):runner.ensure_open();runner.move(target,noncontact=True)
                images=backend.images();roi=select_roi(ui,images,evidence/'right_wrist_full.png')
                state['camera_check']={'cameras':['right_wrist'],'roi_xywh':roi,'image_size':[240,180],
                    'left_wrist':'record_only','min_score':cfg['check']['min_score']}
            finally:backend.close()
        state['completed_steps'].append(step)
        save_yaml(state_path,state)
    state['ready']=True;state['hardware_hash']=station_hash(state)
    save_yaml(state_path,state);save_yaml(profile_path,state)
    ui.say('setup 完成：'+str(profile_path.resolve()))
    ui.say('硬件哈希（模拟）：'+state['hardware_hash'][:16]+'；实机放行=否。')
    return 0


class WizardRunner(Runner):
    def __init__(self,session,backend,ui,*,physical_authorized=False):
        if not isinstance(backend,WizardSimBackend):
            if not physical_authorized or backend.simulated or session.meta.get('simulated') or ui.rehearsal:
                raise ValueError('Physical wizard requires validated real profile and interactive operator')
        super().__init__(session,backend,single_step=False,unattended=False)
        self.ui=ui;self.need_move=True;self.quit_requested=False;self.menu_progress=''
        self.roi_reselected=False
        if self.guard and session.meta.get('wizard_roi_override'):
            self._set_roi(session.meta['wizard_roi_override'])
    def event(self,kind,**data):
        record={'kind':kind,'monotonic_ns':self.now(),**plain(data)}
        folder=self.session.path/'wizard_events';folder.mkdir(exist_ok=True)
        atomic_json(folder/f'{self.now()}_{time.time_ns()}_{kind}.json',record)
    def ask(self,message):
        if message.startswith('[LABEL]'):return ''
        if message.startswith('['):
            return self.ui.ask(message.replace('确认现场安全后按回车；Ctrl-C 中止：','阶段确认，回车继续：'))
        result=self.ui.ask(message)
        if '复位' in message:
            result=(self.meta or {}).get('auto',{}).get('check_displacements',{})
            if not result:
                files=sorted((self.session.path/'checks').glob('*/result.json'),key=lambda p:p.stat().st_mtime_ns)
                if files:result=json.loads(files[-1].read_text())
            quality=result.get('cameras',{}).get('right_wrist',{})
            if quality.get('reason') in ('low_match_confidence','ambiguous_match') and not self.roi_reselected:
                self.ui.say('右腕匹配质量不足：从本 session 原始参考图重选一次区域，参考图像素保持不变。')
                roi=select_roi(self.ui,self.guard._references,self.session.path/'references/right_wrist_quality_reselect.png',
                    default=self.guard.rois['right_wrist'])
                self._set_roi(roi);self.roi_reselected=True
                self.session.meta['wizard_roi_override']=roi;self.session.save_meta()
                self.event('roi_reselected',roi=roi,reference_sha256=self.guard.reference_sha256)
            if self.backend.simulated:self.backend.bottle_shift_px=[0,0]
            self.need_move=True
            self.event('operator_reset')
        return result
    def _set_roi(self,roi):
        original=self.guard
        self.guard=TemplateGuard(original._references,{'right_wrist':roi},original.thresholds_px,
            checked_cameras=['right_wrist'],search_radius_px=original.search_radius_px,
            min_score=original.min_score,min_template_std=original.min_template_std,
            min_peak_margin=original.min_peak_margin)
        self.guard.calibration=original.calibration
        if self.guard.reference_sha256!=original.reference_sha256:
            raise SafetyFault('重新框选意外改变原始参考；拒绝继续')

    def authorize_action(self,target):
        if not self.need_move:return
        from .geometry import pose_error
        start=self._healthy_snapshot()
        reasons=check_segment(np.asarray(start['tcp']),np.asarray(target),self.cfg,self.session.T0,width_mm=start['width_mm'])
        if reasons:raise SafetyFault('首段路径被否决：'+'；'.join(reasons))
        self.ui.say('首段移动目标 TCP xyz(mm)：'+str(np.asarray(target)[:3,3].round(2).tolist()))
        self.ui.confirm('确认现场安全，输入 MOVE 启动本段及后续自动流程：','MOVE')
        change=pose_error(np.asarray(start['tcp']),np.asarray(self._healthy_snapshot()['tcp']))
        if change[0]>self.cfg['motion']['pose_tolerance_mm'] or change[1]>self.cfg['motion']['pose_tolerance_deg']:
            raise SafetyFault('确认期间 TCP 已改变；取消命令，请重新预览')
        self.need_move=False;self.event('move_confirmation',start_tcp=start['tcp'],target_tcp=target)
    def ensure_open(self):
        if self.need_move:self.authorize_action(self._healthy_snapshot()['tcp'])
        return super().ensure_open()
    def move(self,target,**kwargs):
        self.authorize_action(target)
        return super().move(target,**kwargs)
    def confirm_startup_move(self,target):
        self.authorize_action(target)
    def check_bottle(self,tag='after_retreat'):
        with redirect_stdout(io.StringIO()):
            return super().check_bottle(tag)
    def _undo(self):
        previous=[]
        for ident in self.session.progress['completed']:
            path=self.session.path/'trials'/ident/'meta.json'
            item=json.loads(path.read_text())
            if 'operator' in item and item.get('status')=='completed':previous.append((path,item))
        if not previous:self.ui.say('没有可撤销的上一条标签。');return
        path,item=previous[-1];old=copy.deepcopy(item.get('operator',{}))
        replacement=self.ui.ask(f"撤销 {item['trial_id']} 的标签 {LABEL_NAMES.get(old.get('label',''),'未知')}；输入替代数字（回车正常）：",'1')
        if replacement not in LABELS:self.ui.say('未改动上一条标签。');return
        label=LABELS[replacement]
        if label=='' and any(item.get('auto',{}).get(k) for k in ('dropped','slip_suspected')):
            self.ui.confirm('上一条自动异常与正常矛盾；输入 YES 才改为正常：','YES')
        item.setdefault('operator_history',[]).append({'previous':old,'changed_unix_ns':time.time_ns(),'action':'undo_replace'})
        item['operator']={'label':label,'note':old.get('note','')}
        atomic_json(path,item);self.session.rebuild_index();self.event('label_undo',trial_id=item['trial_id'],old=old,new=item['operator'])
        self.ui.say('已更正上一条；原标签保留在历史中。')
    def label(self):
        auto=self.meta.get('auto',{});offsets=self.meta.get('offsets',{})
        lifted=any(x['name']=='LIFT' for x in self.meta['stages'])
        check=auto.get('check_displacements',{}).get('passed',False)
        self.ui.say(f"试验 {self.meta['trial_id']} · {self.menu_progress} · {self.meta['bottle']} · 偏移(a,c,l)={[offsets.get(k,0) for k in ('a_mm','c_mm','l_mm')]}")
        self.ui.say('  自动：双侧接触={} 目标力={} 提起={} 掉落={} 疑似滑动={} 瓶位={}'.format(
            auto.get('bilateral_contact',False),auto.get('force_reached',False),lifted,
            auto.get('dropped',False),auto.get('slip_suspected',False),'通过' if check else '未通过'))
        self.ui.say(MENU)
        note='';default='4' if auto.get('dropped') else '5' if auto.get('slip_suspected') else '1'
        while True:
            answer=self.ui.ask('标签：',default,key='label:'+self.meta['trial_id']).lower()
            if answer=='n':note=self.ui.ask('备注：');continue
            if answer=='u':self._undo();continue
            if answer=='p':
                self.event('pause',trial_id=self.meta['trial_id']);self.ui.say('已暂停：在 T_check，夹爪张开，未提瓶。')
                self.ui.confirm('回车恢复，q 可在随后标签菜单安全退出：')
                self.event('resume');self.need_move=True;self.stage_started_ns=self.now();continue
            if answer=='q':
                self.quit_requested=True;self.event('quit_requested',trial_id=self.meta['trial_id'])
                answer=self.ui.ask('已请求安全退出；请先选择本次标签（回车正常）：',default)
            if answer not in LABELS:self.ui.say('无效选项；仅在当前标签菜单接收这些按键。');continue
            label=LABELS[answer]
            contradictory=(label=='' and (auto.get('dropped') or auto.get('slip_suspected')
                or not auto.get('bilateral_contact',True) and self.meta.get('kind')!='commission_above_cap'))
            if contradictory:
                confirm=self.ui.ask('自动异常与“正常”矛盾；仍按肉眼判断记正常？输入 YES，其他输入重选：','YES')
                if confirm!='YES':continue
                self.event('label_contradiction_confirmed',trial_id=self.meta['trial_id'],auto=auto)
            self.meta['operator']={'label':label,'note':note};break
        if label in ('t','m','d'):
            self.ui.confirm('请把瓶子复位到标记处；回车后自动检查：')
            if self.backend.simulated:self.backend.bottle_shift_px=[0,0]
            self.need_move=True
            self.check_bottle('operator_reset')


def compact_plan(cfg,T0,mode):
    """A bounded, labelled rehearsal plan; the expert full experiment is unchanged."""
    if mode=='visual':
        points=[]
        for block,angle in enumerate((0,90)):
            vc=copy.deepcopy(cfg);vc['visual']['count']=3
            for i,item in enumerate(visual_points(vc,T0)):
                item.update(point_id=f'point_{len(points)+1:04d}',orientation_block=block,bottle_orientation_deg=angle)
                points.append(item)
        return points
    specs=[('drift_zero',[0,0,0]),('drift_zero',[0,0,0]),('drift_zero',[0,0,0]),
        ('grid',[-5,0,-10]),('grid',[5,0,10]),('drift_zero',[0,0,0]),
        ('grid',[-5,0,10]),('grid',[5,0,-10])]
    result=[]
    for i,(kind,delta) in enumerate(specs):
        item=_trial(cfg,T0,'light' if i<4 else 'heavy',kind,delta)
        item.update(trial_id=f'trial_{i+1:04d}',sequence=i+1)
        result.append(item)
    # Keep an explicit geometry rejection in the preview; never move to it.
    rejected=_trial(cfg,T0,'heavy','rehearsal_rejected',[20,0,0])
    rejected.update(trial_id='trial_0009',sequence=9)
    result.append(rejected)
    return result


def zero_values(record,models,shape):
    off=record.get('offsets',{})
    if (record.get('task')!='contact' or record.get('status')!='completed'
        or any(abs(float(off.get(k,0)))>1e-8 for k in ('a_mm','c_mm','l_mm','rot_c_deg','rot_other_deg'))):return None
    auto=record.get('auto',{})
    if (not auto.get('bilateral_contact') or auto.get('probe_not_settled',True)
            or auto.get('dropped') or auto.get('slip_suspected')):return None
    result={}
    for side in SIDES:
        med=record.get('descriptor_windows',{}).get('PROBE',{}).get(side,{}).get('median',{})
        uv=np.asarray([med.get('centroid_u_px'),med.get('centroid_v_px')],float)
        if not np.isfinite(uv).all():return None
        center=np.array([(shape[1]-1)/2,(shape[0]-1)/2])
        result[side]=(np.asarray(models[side]['inverse_tcp_mm_per_px'])@(uv-center)).tolist()
    return result


def check_zero_drift(runner,record,threshold=None):
    threshold=float(runner.cfg.get('wizard',{}).get('zero_drift_threshold_mm',1.5) if threshold is None else threshold)
    if not np.isfinite(threshold) or threshold<=0:raise ValueError('零偏移漂移阈值必须是正数')
    state=runner.session.meta.setdefault('zero_drift',{'threshold_mm':threshold,'baseline_trial_ids':[],
        'samples':[],'units':'equivalent TCP a/l mm from recorded response calibration'})
    models=runner.session.meta.get('wizard_sensor_response',{})
    if set(models)!=set(SIDES):return
    if any(x['trial_id']==record['trial_id'] for x in state['samples']):return
    value=zero_values(record,models,runner.cfg['tactile']['dd_shape'])
    if value is None:return
    stamp=next((s['start_ns'] for s in record['stages'] if s['name']=='PROBE'),runner.now())
    sample={'trial_id':record['trial_id'],'monotonic_ns':stamp,'centroid_al_mm':value}
    state['samples'].append(sample)
    if len(state['samples'])<=3:
        state['baseline_trial_ids'].append(record['trial_id'])
        if len(state['samples'])==3:
            state['baseline']={s:np.mean([x['centroid_al_mm'][s] for x in state['samples']],axis=0).tolist() for s in SIDES}
            runner.ui.say('零偏移后备检查：前 3 次有效双指质心基准已锁定。')
    else:
        residual={s:(np.asarray(value[s])-state['baseline'][s]).tolist() for s in SIDES}
        sample['residual_al_mm']=residual
        sample['alert']=any(np.any(np.abs(residual[s])>threshold) for s in SIDES)
        if sample['alert']:
            runner.ui.say(f'零偏移漂移报警：任一指 a/l 超过 {threshold} mm；请检查瓶子是否仍在标记处。')
            answer=runner.ui.ask('1 继续 / 2 已复位后继续 / q 安全退出：','2')
            while answer not in ('1','2','q'):answer=runner.ui.ask('请选择 1、2 或 q：','2')
            sample['operator_action']=answer;runner.event('zero_drift_alarm',**sample)
            runner.need_move=True
            if answer=='q':runner.quit_requested=True
            elif answer=='2':
                if runner.backend.simulated:
                    runner.backend.zero_drift_px=0.;runner.backend.bottle_shift_px=[0,0]
                runner.check_bottle('zero_drift_reset')
    runner.session.save_meta()


def analyze_and_print(session,ui):
    from .analysis import analyze_session
    from .wizard_analysis import write_zero_drift_plot
    report=analyze_session(session.path)
    drift=report.get('zero_offset_centroid_drift') or write_zero_drift_plot(session.path,session.path/'analysis')
    rows=[]
    for ident in session.progress['completed']:
        rows.append(json.loads((session.path/'trials'/ident/'meta.json').read_text()))
    measured=[r for r in rows if r.get('status')=='completed' and not r.get('commissioning') and not r['trial_id'].startswith('commission_')]
    counts=Counter(LABEL_NAMES.get(r.get('operator',{}).get('label',''),'未知') for r in measured)
    ui.say(f"分析：已完成 {len(measured)} 条，安全剔除 {sum(r.get('status')=='skipped_safety' for r in rows)} 条。")
    ui.say('标签比例：'+('；'.join(f'{k} {v}/{len(measured)} ({v/len(measured):.0%})' for k,v in counts.items()) or '暂无完成记录'))
    import csv
    with (session.path/'analysis/linear_fits.csv').open() as f:fits=list(csv.DictReader(f))
    valid=[r for r in fits if r.get('slope') not in ('nan','','None')]
    ui.say('拟合斜率：'+('；'.join(f"{r.get('bottle')}/{r.get('stage')}/{r.get('finger')}/{r.get('axis')}={float(r['slope']):.4g} {r.get('slope_unit','')}" for r in valid[:4]) or '有效独立偏移不足；未报告虚假斜率'))
    ui.say('结果目录：'+str((session.path/'analysis').resolve()))
    ui.say('零偏移质心时间图：'+str(drift['plot']))
    return report


def item_done(session,item,mode):
    ident=item.get('trial_id',item.get('point_id'))
    if ident not in session.progress['completed']:return False
    if mode=='visual':
        record=json.loads((session.path/'trials'/ident/'meta.json').read_text())
        # A crash between exposure and terminal annotation must not discard the
        # pending label or reacquire/overwrite the already committed images.
        if record.get('status')=='completed' and 'operator' not in record:return False
    return True


def run(args,ui):
    profile=require_profile(args.profile)
    existing=args.session and (Path(args.session)/'session_meta.json').exists()
    session=Session(args.session) if existing else None
    ui.say(BANNER)
    # No-pending is intentionally before doctor/backend/calibration, even on a
    # different boot. It cannot generate any robot command or new reference.
    if session:
        if not session.meta.get('simulated') or not session.meta.get('wizard_simulation'):
            raise ValueError('本版向导不能续接实机会话或高级工具会话')
        if session.meta['wizard_mode']!=args.mode:raise ValueError('session 模式不一致')
        if session.meta['station_hardware_hash']!=profile['hardware_hash']:raise ValueError('session 与 station_profile 哈希不一致')
        if not session.meta.get('wizard_preparation_complete') or not (session.path/'plans'/f'wizard_{args.mode}.json').exists():
            raise ValueError('准备阶段中断，尚无锁定计划；请新建 session，原始记录保留')
        plan=json.loads((session.path/'plans'/f'wizard_{args.mode}.json').read_text())
        pending=[x for x in plan if not item_done(session,x,args.mode)]
        if not pending:
            ui.say('没有待做项；直接退出，没有创建后端，没有发送运动。');return 0
        if session.progress.get('requires_operator_recovery'):
            raise ValueError('上次可能仍在空中：禁止自动张开/续跑；请现场恢复后创建新 session')
        if session.meta['clock_boot_id']!=boot_id():raise ValueError('主机已重启，原单调时钟失效；请新建 session')
        if not session.meta.get('wizard_preparation_complete'):
            raise ValueError('准备阶段中断，未形成锁定参考；请新建 session，已采原始数据保留')
        cfg=session.cfg
        sim_doctor(cfg,session.path/'doctor_resume.json')
        ui.say('续跑：已完成项和原始参考保留，跳过校准。')
    else:
        cfg=copy.deepcopy(profile['config'])
        path=Path(args.session) if args.session else Path(args.output_root)/dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).strftime('%Y%m%d_%H%M%S_%f')
        sim_doctor(cfg,path.parent/f'.wizard_doctor_{path.name}.json')
        ui.confirm('请把瓶子放到标记处，把夹爪张开并移到瓶子旁大致的抓取位置，然后按回车。')
        session=Session.create(path,cfg,np.asarray(cfg['T0']),simulated=True)
        session.meta.update(wizard_simulation=True,wizard_mode=args.mode,station_hardware_hash=profile['hardware_hash'])
        session.save_meta()
        ui.say('session：'+str(session.path.resolve())+'；初始 T0 已自动读取（模拟反馈）。')
    backend=WizardSimBackend(cfg,session.T0)
    if existing:
        backend.now_ns=max(backend.now_ns,session.meta.get('simulated_clock_last_ns',0)+1)
        backend.drain_samples();backend.step(0.)
    runner=WizardRunner(session,backend,ui)
    Tcheck=translate_local(session.T0,-cfg['check']['retreat_mm'],'a',cfg['axes'])
    backend.check_origin=Tcheck.copy()
    try:
        if not existing:
            ui.say('自动准备：每次移动前松开夹爪；方向标定与居中全程不提瓶。')
            # Initial unloaded acquisition supports honest reference subtraction
            # during calibration; another is recorded at final T_check.
            runner.baseline()
            residual=calibrate_and_center(runner,session,ui)
            cfg=session.cfg
            Tcheck=translate_local(session.T0,-cfg['check']['retreat_mm'],'a',cfg['axes'])
            backend.check_origin=Tcheck.copy()
            with runner.phase('CHECK'):runner.ensure_open();runner.move(Tcheck,noncontact=True)
            runner.baseline()
            images=backend.images();roi=profile['camera_check']['roi_xywh']
            try:
                guard=calibrate_right_camera(runner,images,roi,session.path/'calibration/right_camera')
            except ValueError as exc:
                ui.say('右腕模板质量/标定失败：'+str(exc))
                roi=select_roi(ui,images,session.path/'references/right_wrist_reselect.png')
                guard=calibrate_right_camera(runner,images,roi,session.path/'calibration/right_camera_retry')
            guard.save(session.path/'references/bottle_position');runner.guard=guard
            for camera,image in images.items():
                import cv2
                cv2.imwrite(str(session.path/'references'/f'{camera}_full.png'),cv2.cvtColor(image,cv2.COLOR_RGB2BGR))
            # Overlay uses saved ROI only; run does not request a new selection.
            import cv2
            preview=images['right_wrist'].copy();x,y,w,h=roi
            cv2.rectangle(preview,(x,y),(x+w-1,y+h-1),(255,80,0),2)
            preview_path=session.path/'references/right_wrist_roi.png'
            cv2.imwrite(str(preview_path),cv2.cvtColor(preview,cv2.COLOR_RGB2BGR))
            session.meta.update(T_check=Tcheck.tolist(),position_calibration=guard.calibration,
                reference_sha256=guard.reference_sha256,wizard_preparation_complete=True)
            save_yaml(session.path/'config_snapshot.yaml',cfg);session.save_meta()
            peak=max(np.linalg.norm(v) for v in residual['residual_equivalent_al_mm'].values())
            ui.say(f"准备完成：T0 双指最大残差={peak:.3f} 等效 mm；右腕阈值={guard.thresholds_px['right_wrist']:.2f} px；区域预览={preview_path}")
            if guard.calibration['unobservable_directions']:
                ui.say('不可观测方向：'+str(guard.calibration['unobservable_directions'])+'；图像检查不能证明全方向 3D 复位。')
            ui.confirm('核对准备摘要，按回车确认：')
            plan=compact_plan(cfg,session.T0,args.mode)
            session.lock_plan('wizard_'+args.mode,plan)
            pending=plan
        else:
            runner.check_bottle('run_start')
        with redirect_stdout(io.StringIO()):
            summary=dry_report(args.mode,cfg,session.T0,plan,session.path/'plans/wizard_preflight',print_all=False)
        rejected=Counter(reason for x in plan if not x['valid'] for reason in {':'.join(r.split(':')[-3:]) for r in x['safety_reasons']})
        ui.say(f"计划：候选 {summary['candidate_count']}，可执行 {summary['accepted_count']}，剔除 {summary['rejected_count']}；预计 {summary['estimated_active_seconds']/60:.1f} 分钟，{summary['estimated_storage_GiB']:.2f} GiB。")
        if rejected:ui.say('主要剔除原因：'+str(rejected.most_common(3)))
        ui.say('估计不含标注、复位、验收和其他等待；完整 dry-run 已保存。')
        ui.confirm('确认计划，按回车继续：')
        validation_path=Path(args.profile).with_suffix('.simulation_acceptance.json')
        reusable=None
        if validation_path.exists():
            prior=json.loads(validation_path.read_text())
            if (prior.get('hardware_hash')==profile['hardware_hash'] and prior.get('simulation_only')
                    and prior.get('passed') and prior.get('level')=='full'):
                from .commissioning import require_prior_session
                try:
                    require_prior_session(cfg,prior.get('session'),simulated=True)
                except (ValueError,OSError):
                    ui.say('旧验收证据不完整，重新做三点验收。')
                else:reusable=prior
        if not session.progress.get('commissioning_passed'):
            # Match the existing acceptance rules, including above-cap no-contact.
            level='lite' if reusable else 'full'
            ui.say('验收：'+('哈希已验证，本 session 第一次试验逐阶段确认。' if reusable else '此哈希首次使用，零偏移/最深 +a/越瓶盖 3 点逐阶段确认。'))
            commission=commissioning_trials(cfg,session.T0)[:1 if reusable else 3]
            records=[]
            for i,item in enumerate(commission):
                item['trial_id']=f'commission_{i+1:04d}'
                path=session.path/'trials'/item['trial_id']/'meta.json'
                if item['trial_id'] in session.progress['completed']:
                    records.append(json.loads(path.read_text()));continue
                if not item['valid']:raise SafetyFault('验收点被几何否决，保留否决；需要调整站点定义后重试')
                runner.single_step=True;runner.menu_progress=f'验收 {i+1}/{len(commission)}'
                record=runner.contact_trial(item);records.append(record)
                check_zero_drift(runner,record)
                if runner.quit_requested:break
            if runner.quit_requested:return finish_run(runner,ui)
            approval=build_acceptance(session,records,level=level,prior=reusable)
            ui.say(f"验收摘要：{len(records)}/{len(commission)} 完成；稳定轻触、力限值、越瓶盖无接触和最终瓶位检查："+('通过' if approval['measurements_passed'] else '失败'))
            atomic_json(session.path/'wizard_commissioning.json',approval)
            if not approval['measurements_passed']:raise SafetyFault('验收未通过：'+'；'.join(approval['failures']))
            ui.confirm('输入 PASS 才继续（本次只能登记模拟验收）：','PASS')
            approval.update(passed=True,operator_confirmation='PASS')
            atomic_json(session.path/'wizard_commissioning.json',approval)
            session.meta['commissioning_acceptance']=approval
            session.progress['commissioning_passed']=True
            session.save_meta();session.save_progress()
            if not reusable:
                atomic_json(validation_path,{'hardware_hash':profile['hardware_hash'],'simulation_only':True,
                    'passed':True,'level':level,'session':str(session.path.resolve())})
        runner.single_step=False
        count=0;last_block=session.meta.get('visual_orientation_block')
        for item in pending:
            ident=item.get('trial_id',item.get('point_id'))
            if item_done(session,item,args.mode):continue
            if not item['valid']:
                session.start_trial(ident)
                session.complete({'trial_id':ident,'task':args.mode,'bottle':item['bottle'],'kind':item['kind'],
                    'status':'skipped_safety','skip_reason':item['safety_reasons'],'auto':{},'stages':[],'operator':{}})
                continue
            if args.limit is not None and count>=args.limit:break
            if session.progress.get('last_bottle')!=item['bottle']:
                ui.confirm(f"请放置 {item['bottle']} 瓶到同一标记处，回车后自动检查；T0 和原始参考不变：")
                runner.check_bottle('bottle_switch')
                session.progress['last_bottle']=item['bottle'];session.save_progress()
            if args.mode=='visual' and item['orientation_block']!=last_block:
                # Rotation reference is a separate block artifact. It never
                # overwrites the original position-check reference.
                angle=item['bottle_orientation_deg']
                ui.confirm(f'请将瓶子转到 {angle}°，保持底部在标记处；回车检查后记录本朝向参考：')
                runner.check_bottle('orientation_switch')
                import cv2
                folder=session.path/'references'/f"orientation_{item['orientation_block']}";folder.mkdir(exist_ok=True)
                for cam,img in backend.images().items():
                    cv2.imwrite(str(folder/(cam+'.png')),cv2.cvtColor(img,cv2.COLOR_RGB2BGR))
                last_block=item['orientation_block'];session.meta['visual_orientation_block']=last_block;session.save_meta()
            completed=sum(x.get('trial_id',x.get('point_id')) in session.progress['completed'] for x in plan)
            runner.menu_progress=f'{completed+1}/{len(plan)}'
            runner.single_step=bool(existing and count==0)
            if args.mode=='contact':
                # Explicit scenario injection is confined to the prerecorded
                # rehearsal example. Normal interactive simulations stay normal.
                scenario='dropped' if ui.rehearsal and ident=='trial_0004' else 'slip' if ui.rehearsal and ident=='trial_0005' else 'normal'
                backend.zero_drift_px=4. if ui.rehearsal and ident=='trial_0006' else 0.
                record=runner.contact_trial(item,scenario=scenario)
                check_zero_drift(runner,record)
            else:
                if ident in session.progress['completed']:
                    record=json.loads((session.path/'trials'/ident/'meta.json').read_text())
                    runner.meta=record
                    ui.say('恢复已拍摄点的待补标签；不会重新拍摄或覆盖原始图像。')
                else:record=runner.visual_point(item)
                record['offsets']={k:float(v) for k,v in zip(('a_mm','c_mm','l_mm'),item['delta_acl_mm'])}
                record.update(orientation_block=item['orientation_block'],bottle_orientation_deg=item['bottle_orientation_deg'])
                runner.check_bottle('visual_after_capture')
                if record['status']=='completed':
                    with runner.phase('LABEL'):runner.label()
                    session.complete(runner.meta)
            count+=1;runner.single_step=False
            if runner.quit_requested:break
            # Preserve periodic independent unloaded references from the expert
            # collector; these do not replace the immutable image reference.
            if count % cfg['design'].get('baseline_every',30)==0:runner.baseline()
        return finish_run(runner,ui)
    except BaseException as exc:
        runner.event('exception',error=str(exc),airborne=runner.airborne,
            recovery_latched=session.progress.get('requires_operator_recovery',False),stage=runner.stage)
        if runner.actuation_started:backend.stop_arm()
        if runner.airborne or session.progress.get('requires_operator_recovery'):
            ui.say('已停止：可能仍在空中，禁止自动张开。请守护急停、人工放回瓶子并恢复，随后新建 session。')
        else:
            released=False
            if runner.actuation_started:
                try:
                    with runner.phase('RELEASE',emergency=True):
                        runner.emergency_release_pre_lift(already_stopped=True)
                    released=True
                except BaseException as release_error:
                    runner.event('release_incomplete',error=str(release_error))
            ui.say(f'已停止于 {runner.stage}；未提瓶/空中恢复锁未置位；'+('夹爪已安全释放。' if released else '夹爪开度未保证，请现场处理。'))
        try:analyze_and_print(session,ui)
        except Exception as analysis_error:ui.say('部分数据分析未完成：'+str(analysis_error))
        raise
    finally:
        session.meta['simulated_clock_last_ns']=backend.monotonic_ns();session.save_meta();backend.close()


def finish_run(runner,ui):
    target=translate_local(runner.session.T0,-runner.cfg['check']['retreat_mm'],'a',runner.cfg['axes'])
    runner.single_step=False
    with runner.phase('CHECK'):
        runner.ensure_open();runner.move(target,noncontact=True)
    runner.event('safe_exit',tcp=runner.backend.snapshot()['tcp'],gripper_open=True)
    ui.say('本轮结束：已到 T_check，夹爪张开；断点已保存。')
    analyze_and_print(runner.session,ui)
    return 0


def main(argv=None):
    parser=argparse.ArgumentParser(description='药瓶采集向导：setup 首次设置，run 日常采集；运动由现场 MOVE 确认')
    sub=parser.add_subparsers(dest='command',required=True)
    for name in ('setup','run'):
        p=sub.add_parser(name)
        modes=p.add_mutually_exclusive_group()
        modes.add_argument('--simulate',action='store_true',help='合成数据；绝不连接硬件')
        modes.add_argument('--real',action='store_true',help='使用真实公开 ROS；不加模拟选项时默认实机')
        p.add_argument('--rehearsal',action='store_true',help='仅模拟：自动填写演示回答')
        p.add_argument('--profile')
        p.add_argument('--config',default=str(Path(__file__).resolve().parents[1]/'config.ros.yaml'))
        p.add_argument('--check-only',action='store_true',help='实机只读诊断和缺项报告；绝不创建控制后端')
        if name=='setup':p.add_argument('--redo',action='store_true',help='实机：归档旧设置并重新执行 setup')
        if name=='run':
            p.add_argument('--mode',choices=('contact','visual'),required=True)
            p.add_argument('--session')
            p.add_argument('--output-root')
            p.add_argument('--limit',type=int)
    args=parser.parse_args(argv)
    if args.rehearsal and not args.simulate:parser.error('--rehearsal 仅限 --simulate，实机不能自动回答 MOVE/PASS')
    if args.simulate and args.check_only:parser.error('--check-only 是实机只读入口')
    if getattr(args,'limit',None) is not None and args.limit<=0:parser.error('--limit 必须为正整数')
    if args.profile is None:args.profile='outputs/wizard_preview/station_profile.yaml' if args.simulate else 'station_profile.yaml'
    if args.command=='run' and args.output_root is None:args.output_root='outputs/wizard_preview/sessions' if args.simulate else 'outputs/sessions'
    ui=UI(args.rehearsal)
    try:
        if args.simulate:return setup(args,ui) if args.command=='setup' else run(args,ui)
        from .wizard_real import setup_real,run_real,check_only
        if args.check_only:return check_only(args,ui)
        return setup_real(args,ui) if args.command=='setup' else run_real(args,ui)
    except (Exception,KeyboardInterrupt) as exc:
        import traceback
        folder=Path('outputs/wizard_errors');folder.mkdir(parents=True,exist_ok=True)
        report=folder/(str(time.time_ns())+'.json')
        atomic_json(report,{'error':str(exc),'type':type(exc).__name__,'traceback':traceback.format_exc(),
            'command':args.command,'simulate':bool(args.simulate)})
        ui.say('向导停止：'+str(exc)+'；错误详情：'+str(report));return 2
