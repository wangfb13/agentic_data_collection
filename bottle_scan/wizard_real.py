"""Physical setup/run adapter over the existing public ROS bridge.

Import is inert. Only an interactive operator invocation may create a control
backend. check_only uses observers exclusively; no simulator may authorize it.
"""
from __future__ import annotations
import copy
from collections import Counter
from contextlib import redirect_stdout
import datetime as dt
import hashlib
import io
import json
from pathlib import Path
import time
import numpy as np
import yaml
from .backend import load_hardware
from .config import load_config,unresolved,require_real_ready,validate_config
from .session import Session,atomic_json,fingerprint,plain,boot_id
from .geometry import translate_local,grasp_pose,axis_basis,check_pose,check_segment
from .commissioning import hardware_binding,require_prior_session,build_acceptance,require_current_commissioning
from .stop_validation import run_stop_test,gate_stop_report
from .force_control import require_force_report,force_report_binding,evaluate_force_report
from .park_guard import ParkGuard
from .design import contact_trials,commissioning_trials,visual_points
from .cli import dry_report
from .wizard import (UI,WizardRunner,save_yaml,select_roi,item_done,finish_run,
    analyze_and_print,check_zero_drift)
from .wizard_calibration import calibrate_and_center,calibrate_right_camera
from .wizard_station import (LiveReader,read_model,model_identity,configure_geometry,
    configure_noise_force,configure_policies,teach_real,numeric,fill,known)
from .wizard_width import WidthBootstrap,fit_width_points,compliant_width_mapping
from .wizard_dimensions import bottle_dimensions,bottle_sections

REAL_STEPS=('doctor','model','dimensions','calibration','teaching','stop','roi_and_validation')


def doctor(cfg,path,ui,*,quiet=True):
    import robot_doctor
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    config_path=path.with_suffix('.config.yaml');save_yaml(config_path,cfg)
    capture=io.StringIO()
    with redirect_stdout(capture):
        result=robot_doctor.main(['--arm',cfg['hardware']['arm_side'],'--config',str(config_path),'--output',str(path)])
    if not path.exists():raise RuntimeError('doctor 没有生成报告：'+capture.getvalue()[-2000:])
    report=json.loads(path.read_text())
    if result not in (None,0) or report.get('errors'):
        raise RuntimeError('doctor 未通过：'+'；'.join(map(str,report.get('errors',[])))+'；报告 '+str(path))
    ownership=report.get('checks',{}).get('command_ownership',{})
    if ownership.get('allowed') is not True:raise RuntimeError('命令话题发布者门禁未通过')
    if not quiet:ui.say('doctor 通过；真实双腕图像、DD、六维力和公开 ROS 状态可用。')
    return report


def source_binding():
    root=Path(__file__).resolve().parents[1]
    names=('bottle_scan/wizard.py','bottle_scan/wizard_real.py','bottle_scan/wizard_station.py',
        'bottle_scan/wizard_width.py','bottle_scan/wizard_dimensions.py','bottle_scan/wizard_calibration.py','bottle_scan/runtime.py',
        'bottle_scan/config.py','bottle_scan/geometry.py','bottle_scan/sensing.py','bridge_ros_public.py')
    return {n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in names}


def station_hash_real(profile):
    return fingerprint({'domain':'physical_station_v1','config':profile['config'],
        'model':model_identity(profile['model']),'camera_check':profile['camera_check'],
        'binding':hardware_binding(profile['config']),'sources':source_binding()})


def require_real_profile(path):
    profile=yaml.safe_load(Path(path).read_text())
    if (profile.get('schema')!='bottle_station_real_v1' or profile.get('simulation_only') is not False
            or not profile.get('ready') or profile.get('completed_steps')!=list(REAL_STEPS)):
        raise ValueError('需要完成真实 setup；模拟 profile/未完成 checkpoint 不能授权实机')
    if profile.get('hardware_hash')!=station_hash_real(profile):
        raise ValueError('硬件/配置/控制源码哈希变化，请重新 setup；尚未创建控制后端')
    require_real_ready(profile['config'],preparing=True)
    stop_station_gate(profile)
    require_force_report(profile['config'])
    gate_opening(profile)
    return profile


def gate_opening(profile):
    path=Path(profile['opening_report'])
    report=json.loads(path.read_text())
    if (report.get('schema')!='bottle_opening_repeat_v2' or report.get('simulated') is not False
            or report.get('operator_confirmation')!='PASS' or report.get('completed') is not True
            or report.get('cleanup_complete') is not True or report.get('binding')!=force_report_binding(profile['config'])
            or report.get('repetitions')!=profile['config']['opening_repeat']['repetitions']
            or len(report.get('trials',[]))!=report.get('repetitions')
            or any(t.get('status')!='completed' or t.get('probe_not_settled',True) for t in report.get('trials',[]))):
        raise ValueError('首次开度重复性实测及 PASS 尚未完成或配置已改变')
    return report


def check_only(args,ui):
    cfg=load_config(args.config)
    directory=Path('outputs/wizard_readonly')/dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).strftime('%Y%m%d_%H%M%S_%f')
    doctor(cfg,directory/'doctor.json',ui,quiet=False)
    with LiveReader(cfg) as reader:
        model=read_model(reader,cfg)
        atomic_json(directory/'model.json',model)
        ui.say('只读：模型='+model['robot_name']+'；TCP='+model['tcp_frames'][cfg['hardware']['arm_side']])
    ui.say('当前原始配置待确认项：'+str(len(unresolved(cfg)))+'；setup 将引导填写/采样，不会伪造这些值。')
    ui.say('只读检查结束；没有创建控制后端，没有发布机械臂或夹爪命令。报告：'+str(directory.resolve()))
    return 0


def capture_widths(cfg,state,reader,ui,evidence):
    from .hardware_gripper import WidthCalibration
    ui.confirm('夹爪完全张开且空载，按回车读取与卡尺张开宽度对应的归一化值：')
    open_value=float(reader.gripper()['normalized']);state['width_open_normalized']=open_value
    if not .998<=open_value<=1.:raise ValueError('请先把夹爪完全张开到归一化 1.0，不能把残留开度作为全开点')
    points=[{'normalized':open_value,'width_mm':cfg['gripper']['open_mm'],'part':'open'}]
    # Only the body anchors the actuator opening map. Neck/cap are one upper
    # collision region and no longer require separate contact measurements.
    part='body';diameter=state['dimensions'][part]['diameter_mm']
    folder=evidence/'width_body'
    cached=json.loads((folder/'calibration.json').read_text()) if (folder/'calibration.json').exists() else {}
    result=cached.get('result')
    status=json.loads((folder/'recording_status.json').read_text()) if (folder/'recording_status.json').exists() else {}
    reusable=(cached.get('release_complete') and cached.get('recording_complete') and status.get('complete')
        and result and result.get('width_mm')==diameter and result.get('normal_signs')==cfg['tactile']['normal_signs'])
    if reusable:
        points.append(result);ui.say('复用已完成且完整记录的瓶身接触点；不重复闭合。')
    else:
        if folder.exists():folder.rename(evidence/(folder.name+'_interrupted_'+str(time.time_ns())))
        ui.confirm('把瓶身放在两指胶面正中，由桌面支撑；夹爪保持全开，停止遥操后按回车：')
        points.append(WidthBootstrap(cfg,reader,ui,folder).measure(part,diameter,open_value))
    atomic_json(evidence/'width_points.json',points)
    fit=compliant_width_mapping(points[0],points[1],cfg)
    calibration=WidthCalibration(fit['points'])
    cfg['hardware']['gripper_width_calibration']=fit['points']
    cfg['hardware']['width_calibration_model']=fit
    cfg['gripper']['min_mm']=fit['estimated_width_range_mm'][0]
    cfg['gripper']['precontact_min_mm']=fit['precontact_min_mm']
    # Independent readout steps, converted through measured calibration knots.
    changes=[]
    for file in evidence.glob('width_body/calibration.json'):
        data=json.loads(file.read_text());values=[x['measured_normalized'] for x in data.get('commands',[])]
        for a,b in zip(values,values[1:]):
            if a!=b:
                try:changes.append(abs(calibration.to_mm(a)-calibration.to_mm(b)))
                except ValueError:pass
    if not changes:raise ValueError('没有观测到可用于开度分辨率估计的独立反馈步长')
    fill(cfg,ui,'close.width_resolution_mm','实测反馈最小非零开度变化(mm)，不是硬件编码器精度',min(changes))
    cfg['gripper']['stall_window_s']=max(cfg['gripper']['stall_window_s'],5.1*cfg['close']['width_resolution_mm']/cfg['gripper']['close_speed_mm_s'])
    atomic_json(evidence/'width_mapping.json',fit)
    ui.say(f"瓶身 {diameter:g} mm 保持不变；全开/瓶身两点标定，不能检查线性。")
    lo,hi=fit['estimated_width_range_mm']
    ui.say(f"接触后等效开度估算区间 {lo:.2f}～{hi:.2f} mm；归一化下限 {fit['estimated_normalized_range'][0]:.4f}；双侧接触后才允许力控进入，力限和硬部件几何检查仍生效。")
    ui.confirm('该压缩段是有界等效开度估算，不是胶面间距实测；确认后按回车继续：')
    state['width_model_confirmation']={'operator_confirmed':True,'kind':fit['schema'],'compressed_gap_measured':False}
    return fit


def locate_bottle(cfg,dimensions,reader,ui,table_z):
    while True:
        ui.confirm('请把瓶子放到标记处，把夹爪张开并移到瓶身大致抓取位置；停止遥操后按回车，自动记录 T0：')
        T0=reader.pose();G=grasp_pose(T0,cfg)
        bottle_axis='l' if cfg['grasp_mode']=='side' else 'a'
        direction=T0[:3,:3]@np.asarray(cfg['axes'][bottle_axis])
        if direction[2]<np.cos(np.deg2rad(5.)):
            ui.say(f'瓶轴 +{bottle_axis} 与基座竖直向上未对齐（z 分量 {direction[2]:.3f}）。请调整末端姿态后重读；自动居中只改平移，不伪造瓶轴。')
            continue
        cfg['T0']=T0.tolist()
        cfg['geometry']['bottle']['center_offset_mm']=np.asarray(cfg['tool']['grasp_center_in_tcp'])[:3,3].tolist()
        height=0.;segments=[]
        for part,data in bottle_sections(dimensions):
            lo=(table_z+height-G[2,3])/direction[2]
            height+=data['height_mm'];hi=(table_z+height-G[2,3])/direction[2]
            segments.append({'name':part,'radius_mm':data['diameter_mm']/2,'axial_min_mm':float(lo),'axial_max_mm':float(hi)})
        cfg['geometry']['bottle']['segments']=segments
        cfg['design']['bottle_axis_range_mm']=[segments[0]['axial_min_mm'],segments[-1]['axial_max_mm']+10.]
        return T0


def validate_cycles(runner,evidence,ui):
    import validate_force
    import opening_repeat
    cfg=runner.cfg
    force_path=Path(cfg['force']['validation']['report_path'])
    force_data=evidence/('force_data_'+str(time.time_ns()));force_data.mkdir()
    force={'schema':'bottle_force_validation_v2','simulated':False,'controller':cfg['force']['controller'],
        'unit':cfg['force']['unit'],'target_definition':cfg['force']['target_definition'],
        'repetitions':cfg['force']['validation']['repetitions'],'trials':[],'binding':force_report_binding(cfg),
        'passed':False,'operator_confirmation':None,'evaluation_signal':'existing_X5_39020_after_guarded_software_zero',
        'control_signal':'existing_X5_39020_after_guarded_software_zero','hold_s':cfg['force']['validation']['hold_s']}
    ui.say('首次力控验证：10 次，不提瓶、不移动机械臂；保留稳定窗口、逐指上限、原始流与验收。')
    atomic_json(force_path,force)
    for i in range(force['repetitions']):
        verify_at_T0(runner)
        runner.single_step=(i==0)
        folder=force_data/f'cycle_{i+1:03d}';folder.mkdir()
        force['trials'].append(validate_force.run_cycle(runner,folder,i+1));atomic_json(force_path,force)
    force['failures']=evaluate_force_report(force,cfg)
    if force['failures']:raise ValueError('力控验收失败：'+'；'.join(force['failures']))
    ui.confirm('力控验证全部达标；核对报告后输入 PASS：','PASS')
    force.update(passed=True,operator_confirmation='PASS');atomic_json(force_path,force)
    require_force_report(cfg)
    opening_path=evidence/'opening_report.json'
    data=evidence/('opening_data_'+str(time.time_ns()));data.mkdir()
    opening={'schema':'bottle_opening_repeat_v2','simulated':False,'binding':force_report_binding(cfg),
        'operator_confirmation':None,'completed':False,'cleanup_complete':False,'trials':[],
        'repetitions':cfg['opening_repeat']['repetitions']}
    records=[]
    for i in range(cfg['opening_repeat']['repetitions']):
        verify_at_T0(runner)
        runner.single_step=(i==0);folder=data/f'cycle_{i+1:03d}';folder.mkdir()
        trial,events=opening_repeat.run_cycle(runner,folder,i+1)
        opening['trials'].append(trial);records.extend(events);atomic_json(opening_path,opening)
    if any(t.get('probe_not_settled',True) for t in opening['trials']):raise ValueError('开度重复性有未稳定的 PROBE，不能验收')
    opening.update(opening_repeat.summarize(opening['trials'],records,cfg),completed=True,cleanup_complete=True)
    ui.say('开度重复性分布：'+json.dumps(opening['contact_normalized'],ensure_ascii=False))
    ui.confirm('核对全部开度重复性测量，输入 PASS：','PASS')
    opening['operator_confirmation']='PASS';opening['first_cycle_single_step']=True;atomic_json(opening_path,opening)
    runner.session.meta['opening_repeat_first_single_step']={'operator_confirmation':'PASS','simulated':False,
        'report_path':str(opening_path.resolve()),'binding':opening['binding']}
    runner.session.save_meta();runner.single_step=False;runner.meta=None
    return str(opening_path.resolve())


def _setup_real_impl(args,ui):
    if ui.rehearsal:raise ValueError('实机 setup 不能使用自动回答')
    path=Path(args.profile);evidence=path.parent/(path.stem+'_setup');evidence.mkdir(parents=True,exist_ok=True)
    checkpoint=evidence/'checkpoint.yaml'
    state=yaml.safe_load(checkpoint.read_text()) if checkpoint.exists() else yaml.safe_load(path.read_text()) if path.exists() else {
        'schema':'bottle_station_real_v1','simulation_only':False,'ready':False,'completed_steps':[],
        'config':load_config(args.config)}
    if state.get('schema')!='bottle_station_real_v1' or state.get('simulation_only') is not False:
        raise ValueError('不能复用模拟设置；请选择真实 profile 路径')
    if state.get('source_binding') and state['source_binding']!=source_binding():
        raise ResetStation('设置期间控制源码变化')
    state['source_binding']=source_binding()
    cfg=state['config']
    if state.get('dimensions'):state['dimensions']=bottle_dimensions(state['dimensions'])
    # Reconnectivity is always observed again, even when resuming an old checkpoint.
    doctor(cfg,evidence/'doctor_latest.json',ui,quiet=False)
    if state.get('ready'):
        try:require_real_profile(path)
        except (ValueError,OSError) as exc:raise ResetStation(str(exc)) from exc
        with LiveReader(cfg) as reader:
            if model_identity(read_model(reader,cfg))!=model_identity(state['model']):
                raise ResetStation('现场 URDF/网格/TCP/相机身份改变')
        ui.say('真实 setup 已完成且身份未变；没有待做设置，不发动作。');return 0
    with LiveReader(cfg) as reader:
        model=read_model(reader,cfg)
        if state.get('model') and model_identity(state['model'])!=model_identity(model):
            raise ResetStation('设置中断后硬件身份改变')
        state['model']=model
        def commit(step):
            if step not in state['completed_steps']:state['completed_steps'].append(step)
            save_yaml(checkpoint,state)
        commit('doctor')
        if 'model' not in state['completed_steps']:
            configure_geometry(cfg,model,ui);commit('model')
        if 'dimensions' not in state['completed_steps']:
            dimensions={}
            for part,name in [('body','瓶身'),('upper','上部（瓶颈和瓶盖合并；最大直径、两者总高度）')]:
                values=numeric(ui,name+'直径 高度(mm)',None,2)
                dimensions[part]={'diameter_mm':values[0],'height_mm':values[1]}
            cfg['gripper']['open_mm']=numeric(ui,'夹爪实际张开宽度(mm)')
            state['dimensions']=dimensions;commit('dimensions')
        if 'calibration' not in state['completed_steps']:
            if not state.get('calibration_inputs_ready'):
                ui.confirm('两指胶面空载，按回车采集真实噪声与帧率：')
                noise=reader.noise();atomic_json(evidence/'noise_and_rates.json',noise)
                configure_noise_force(cfg,noise,ui);configure_policies(cfg,noise,ui)
                for camera,shape in model['camera_shapes'].items():
                    cfg['cameras'][camera].update(height=int(shape[0]),width=int(shape[1]))
                state['calibration_inputs_ready']=True;save_yaml(checkpoint,state)
            state['width_mapping']=capture_widths(cfg,state,reader,ui,evidence)
            cfg['hardware']['camera_mounts_declared']=True
            commit('calibration')
        if 'teaching' not in state['completed_steps']:
            state['teaching']=teach_real(cfg,reader,ui)
            # Record rough T0 without any handwritten pose file, then return to
            # the free pose by manual teaching before the stop-test adapter starts.
            locate_bottle(cfg,state['dimensions'],reader,ui,state['teaching']['nominal_table_z_mm'])
            ui.confirm('回到刚才停止测试的自由空位，保持夹爪张开，停止遥操后回车：')
            commit('teaching')
    # All read-only transports above are closed before control objects are created.
    cfg['force']['validation']['report_path']=str((evidence/'force_report.json').resolve())
    cfg['stop_test']['report_path']=str((evidence/'stop_report.json').resolve())
    missing=unresolved(cfg)
    if missing:raise ValueError('设置尚缺参数（已保存断点）：'+', '.join(missing))
    repair_configuration(cfg,state,checkpoint,ui)
    require_real_ready(cfg,preparing=True)
    if 'stop' not in state['completed_steps']:
        report_path=Path(cfg['stop_test']['report_path'])
        if report_path.exists():report_path.rename(evidence/('stop_previous_'+str(time.time_ns())+'.json'))
        run_stop_test(cfg,simulate=False,output_path=report_path,
            confirm=lambda message: stop_confirm(message,cfg,evidence,ui),output=ui.say)
        gate_stop_report(cfg);state['stop_config_snapshot']=copy.deepcopy(cfg);commit_external(state,checkpoint,'stop')
    else:gate_stop_report(state['stop_config_snapshot'])
    if 'roi_and_validation' not in state['completed_steps']:
        with LiveReader(cfg) as reader:
            ui.confirm('把张开的夹爪移回首次瓶身抓取位，停止遥操后回车，自动重读 T0：')
            T0=reader.pose()
        # The actual bottle origin in base coordinates stays fixed when a new
        # TCP is read. Never transplant the old center without reexpression.
        old=np.asarray(cfg['T0']);center=np.asarray(cfg['geometry']['bottle']['center_offset_mm'])
        from .geometry import pose_error
        if pose_error(old,T0)[1]>cfg['motion']['pose_tolerance_deg']:
            raise ValueError('回读 T0 姿态与示教瓶轴不一致；请以原姿态回到抓取位，再重复 setup')
        fixed=old[:3,3]+old[:3,:3]@center
        cfg['geometry']['bottle']['center_offset_mm']=(T0[:3,:3].T@(fixed-T0[:3,3])).tolist()
        cfg['T0']=T0.tolist()
        # Stop evidence is intentionally evaluated against its original station
        # snapshot; this readback/reexpression changes no stop control settings.
        setup_session=Session.create(evidence/('setup_session_'+str(time.time_ns())),cfg,T0,simulated=False)
        backend=None;runner=None
        try:
            backend=load_hardware(cfg);backend.prepare_zero(setup_session,confirm=ui.ask)
            ParkGuard.capture(setup_session,backend.park_observation(),now_ns=backend.monotonic_ns())
            runner=WizardRunner(setup_session,backend,ui,physical_authorized=True)
            runner.authorize_action(T0);runner.baseline()
            state['opening_report']=validate_cycles(runner,evidence,ui)
            target=translate_local(T0,-cfg['check']['retreat_mm'],'a',cfg['axes'])
            with runner.phase('CHECK'):runner.ensure_open();runner.move(target,noncontact=True);runner.wait(cfg['motion']['settle_s'],noncontact=True)
            images=backend.images();roi=select_roi(ui,images,evidence/'right_wrist_full.png')
            state['camera_check']={'cameras':['right_wrist'],'roi_xywh':roi,
                'image_size':[images['right_wrist'].shape[1],images['right_wrist'].shape[0]],'left_wrist':'record_only'}
            setup_session.meta['setup_complete']=True;setup_session.save_meta()
        except BaseException as exc:
            cleanup(runner,backend,exc,ui);raise
        finally:
            if backend is not None:backend.close()
        commit_external(state,checkpoint,'roi_and_validation')
    state['completed_steps']=list(REAL_STEPS)
    state['ready']=True;state['hardware_hash']=station_hash_real(state)
    save_yaml(checkpoint,state);save_yaml(path,state)
    ui.say('实机 setup 已完成：'+str(path.resolve())+'；日常运行 bash run_on_5090.sh run --mode contact')
    return 0


def commit_external(state,path,step):
    if step not in state['completed_steps']:state['completed_steps'].append(step)
    save_yaml(path,state)


def cleanup(runner,backend,error,ui):
    if backend is None or runner is None or not runner.actuation_started:return
    runner.event('exception',stage=runner.stage,error=str(error),airborne=runner.airborne)
    try:backend.stop_arm()
    except BaseException as stop_error:
        runner.session.progress['requires_operator_recovery']=True;runner.session.save_progress()
        runner.event('stop_not_confirmed',error=str(stop_error))
        ui.say('停止未能确认；保持夹爪，不自动释放，请现场急停/接管：'+str(stop_error));return
    if runner.airborne or runner.session.progress.get('requires_operator_recovery'):
        ui.say('已停止，可能仍在空中：保持夹爪，不自动张开；现场接管，安全放回瓶子后重新建 session。');return
    try:
        with runner.phase('RELEASE',emergency=True):runner.emergency_release_pre_lift(already_stopped=True)
        ui.say('已停止并完成未提瓶安全释放；请检查现场。')
    except BaseException as recovery_error:
        ui.say('已停止，但释放未完成：'+str(recovery_error)+'；请现场处理。')


def stop_station_gate(profile):
    recorded=profile.get('stop_config_snapshot',profile['config'])
    current=copy.deepcopy(profile['config']);original=copy.deepcopy(recorded)
    # Reading a new T0 reexpresses the same fixed bottle center. Everything
    # controlling stop performance, free-pose path, workspace and tool stays bound.
    for cfg in (current,original):
        cfg.pop('T0',None)
        cfg['geometry']['bottle'].pop('center_offset_mm',None)
    if current!=original:raise ValueError('停止测试后站点安全/运控配置改变，必须重测')
    return gate_stop_report(recorded)


def full_plan(cfg,T0,mode):
    if mode=='contact':return contact_trials(cfg,T0)
    angles=cfg.get('wizard',{}).get('visual_orientations_deg',[0.,90.,180.,270.])
    if not angles:raise ValueError('visual 朝向块不能为空')
    count=int(cfg['visual']['count']);result=[]
    if count<len(angles):raise ValueError('视觉点数应不少于朝向块数')
    for block,angle in enumerate(angles):
        vc=copy.deepcopy(cfg)
        vc['visual']['count']=count//len(angles)+(1 if block<count%len(angles) else 0)
        vc['visual']['seed']=int(cfg['visual']['seed'])+block
        for item in visual_points(vc,T0):
            item.update(point_id=f'point_{len(result)+1:05d}',orientation_block=block,bottle_orientation_deg=float(angle))
            result.append(item)
    return result


def run_real(args,ui):
    if ui.rehearsal:raise ValueError('实机不能自动填写操作员确认')
    profile=require_real_profile(args.profile)
    existing=bool(args.session and (Path(args.session)/'session_meta.json').exists())
    session=Session(args.session) if existing else None
    if existing:
        if session.meta.get('simulated') or session.meta.get('wizard_domain')!='physical':
            raise ValueError('不能用实机向导续接模拟或高级工具会话')
        if session.meta.get('wizard_mode')!=args.mode:raise ValueError('续跑模式与 session 不一致')
        if session.meta.get('station_hardware_hash')!=profile['hardware_hash']:raise ValueError('session 站点哈希改变，需要重新 setup 和新建 session')
        plan_path=session.path/'plans'/('wizard_'+args.mode+'.json')
        if not session.meta.get('wizard_preparation_complete') or not plan_path.exists():
            raise ValueError('准备阶段中断、参考/计划尚未锁定；原记录保留，请新建 session')
        plan=json.loads(plan_path.read_text())
        pending=[x for x in plan if not item_done(session,x,args.mode)]
        if not pending:
            ui.say('没有待做项；未运行 doctor、未连接硬件、未发送运动。');return 0
        if session.progress.get('requires_operator_recovery'):
            raise ValueError('存在空中恢复锁；禁止自动张开/续跑，请现场恢复后新建 session')
        if session.meta['clock_boot_id']!=boot_id():raise ValueError('主机重启，单调时钟域改变；请新建 session')
        cfg=session.cfg;directory=session.path
    else:
        cfg=copy.deepcopy(profile['config'])
        directory=Path(args.session) if args.session else Path(args.output_root)/dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).strftime('%Y%m%d_%H%M%S_%f')
        if directory.exists():raise ValueError('新 session 路径已存在，拒绝覆盖')
    diagnostic=directory/'doctor.json' if existing else directory.parent/('.doctor_'+directory.name+'.json')
    doctor(cfg,diagnostic,ui)
    with LiveReader(cfg) as reader:
        identity=read_model(reader,cfg)
        if model_identity(identity)!=model_identity(profile['model']):
            raise ValueError('现场相机/网格/URDF/TCP 定义改变，请重新 setup；未创建控制后端')
        if not existing:
            ui.confirm('请把瓶子放到标记处，把夹爪完全张开、两指胶面空载无接触，并移到瓶身旁大致抓取位置，然后按回车（同时确认空载）：')
            T0=reader.pose()
            from .hardware_gripper import WidthCalibration
            width=WidthCalibration(cfg['hardware']['gripper_width_calibration']).to_mm(reader.gripper()['normalized'])
            if abs(width-cfg['gripper']['open_mm'])>cfg['gripper']['open_tolerance_mm']:
                raise ValueError('启动时夹爪未达到实测张开宽度，未发布命令')
            # The bottle frame follows the marked table position; its height
            # ranges are recomputed from the new measured TCP, never a stale T0.
            direction=T0[:3,:3]@np.asarray(cfg['axes'][cfg['geometry']['bottle']['axis']])
            if direction[2]<np.cos(np.deg2rad(5.)):raise ValueError('瓶轴与基座竖直向上未对齐，请先调整末端姿态')
            nominal_table=profile['teaching']['nominal_table_z_mm']
            G=grasp_pose(T0,cfg);height=0.;segments=[]
            for part,data in bottle_sections(profile['dimensions']):
                lo=(nominal_table+height-G[2,3])/direction[2];height+=data['height_mm']
                hi=(nominal_table+height-G[2,3])/direction[2]
                segments.append({'name':part,'radius_mm':data['diameter_mm']/2,'axial_min_mm':float(lo),'axial_max_mm':float(hi)})
            cfg['T0']=T0.tolist();cfg['geometry']['bottle'].update(center_offset_mm=np.asarray(cfg['tool']['grasp_center_in_tcp'])[:3,3].tolist(),segments=segments)
            cfg['design']['bottle_axis_range_mm']=[segments[0]['axial_min_mm'],segments[-1]['axial_max_mm']+10.]
    # There is no source-sharing window between read-only X5 PULL streams and
    # the control bridge. All physical gates precede its publisher creation.
    require_real_ready(cfg,preparing=not existing)
    require_force_report(cfg);stop_station_gate(profile);gate_opening(profile)
    if not existing:
        reasons=check_pose(T0,cfg,T0)
        target=translate_local(T0,-cfg['check']['retreat_mm'],'a',cfg['axes'])
        reasons+=check_segment(T0,target,cfg,T0)
        if reasons:raise ValueError('初始 T0/T_check 几何否决：'+'；'.join(reasons))
        session=Session.create(directory,cfg,T0,simulated=False)
        session.meta.update(wizard_domain='physical',wizard_mode=args.mode,station_hardware_hash=profile['hardware_hash'],
            operator_empty_confirmation={'method':'explicit startup ENTER','meaning':'both gel faces unloaded, gripper fully open'},
            station_stop_scope=profile['stop_config_snapshot'])
        session.save_meta()
        ui.say('Session：'+str(session.path.resolve())+'；T0 已从真实反馈自动记录。')
    backend=None;runner=None
    try:
        backend=load_hardware(cfg)
        def zero_confirm(message):
            if not existing and 'EMPTY' in message and session.meta.get('operator_empty_confirmation'):
                return 'EMPTY'  # Explicit startup ENTER has the same stated meaning, recorded above.
            return ui.ask(message)
        backend.prepare_zero(session,confirm=zero_confirm)
        if not existing:ParkGuard.capture(session,backend.park_observation(),now_ns=backend.monotonic_ns())
        else:ParkGuard(session).check(backend.park_observation(),now_ns=backend.monotonic_ns(),context='wizard_resume')
        session.meta.setdefault('hardware_runs',[]).append({'task':args.mode,'provenance':backend.provenance(),'started_monotonic_ns':backend.monotonic_ns()})
        from .recording import dd_processing_metadata
        session.meta['dd_processing']=dd_processing_metadata(backend.provenance());session.save_meta()
        runner=WizardRunner(session,backend,ui,physical_authorized=True)
        if not existing:
            runner.authorize_action(session.T0)
            runner.baseline()
            residual=calibrate_and_center(runner,session,ui)
            cfg=session.cfg
            target=translate_local(session.T0,-cfg['check']['retreat_mm'],'a',cfg['axes'])
            with runner.phase('CHECK'):
                runner.ensure_open();runner.move(target,noncontact=True);runner.wait(cfg['motion']['settle_s'],noncontact=True)
            runner.baseline();images=backend.images();roi=profile['camera_check']['roi_xywh']
            try:guard=calibrate_right_camera(runner,images,roi,session.path/'calibration/right_camera')
            except ValueError as exc:
                ui.say('右腕模板质量/标定失败：'+str(exc))
                roi=select_roi(ui,images,session.path/'references/right_reselect.png')
                guard=calibrate_right_camera(runner,images,roi,session.path/'calibration/right_camera_retry')
            guard.save(session.path/'references/bottle_position');runner.guard=guard
            from prepare import save_reference_preview
            from .park_guard import image_pose_metadata
            # Full-resolution reference and ROI overlay are local files; no web prediction.
            preview=session.path/'references/wrist_preview'
            save_reference_preview(images,preview,backend.monotonic_ns(),image_metadata=image_pose_metadata(backend.snapshot(),session,camera_stamp_ns=backend.monotonic_ns()))
            import cv2
            picture=images['right_wrist'].copy();x,y,w,h=roi
            cv2.rectangle(picture,(x,y),(x+w-1,y+h-1),(255,80,0),2)
            cv2.imwrite(str(preview/'right_roi.png'),cv2.cvtColor(picture,cv2.COLOR_RGB2BGR))
            threshold=guard.thresholds_px['right_wrist']
            # Keep the legacy left numerical slot only for schema compatibility;
            # actual CHECK explicitly compares right_wrist alone.
            cfg['check']['threshold_px']={'right_wrist':threshold,'left_wrist':threshold}
            session.meta.update(T_check=target.tolist(),reference_sha256=guard.reference_sha256,
                position_calibration=guard.calibration,wizard_preparation_complete=True,camera_mounts_verified=True)
            save_yaml(session.path/'config_snapshot.yaml',cfg);session.save_meta()
            maximum=max(np.linalg.norm(x) for x in residual['residual_equivalent_al_mm'].values())
            ui.say(f'准备完成：双指最大残差 {maximum:.3f} 等效 mm，右腕阈值 {threshold:.2f} px；区域预览 {preview}/right_roi.png')
            if guard.calibration['unobservable_directions']:
                ui.say('相机不可观测方向：'+str(guard.calibration['unobservable_directions'])+'；图像不能证明这些方向的 3D 位置。')
            ui.confirm('核对准备摘要，回车确认：')
            require_real_ready(cfg)
            plan=full_plan(cfg,session.T0,args.mode);session.lock_plan('wizard_'+args.mode,plan);pending=plan
        else:runner.check_bottle('run_start')
        with redirect_stdout(io.StringIO()):summary=dry_report(args.mode,cfg,session.T0,plan,session.path/'plans/wizard_preflight',print_all=False)
        counts=Counter(reason for item in plan if not item['valid'] for reason in {':'.join(r.split(':')[-3:]) for r in item['safety_reasons']})
        ui.say(f"计划：可执行 {summary['accepted_count']}，剔除 {summary['rejected_count']}；预计 {summary['estimated_active_seconds']/60:.1f} 分钟、{summary['estimated_storage_GiB']:.2f} GiB（不含人工等待/验收）。")
        if counts:ui.say('主要剔除原因：'+str(counts.most_common(3)))
        ui.confirm('确认计划，回车继续：')
        cache=Path(args.profile).with_suffix('.real_acceptance.json');prior=None
        if cache.exists():
            saved=json.loads(cache.read_text())
            if saved.get('hardware_hash')==profile['hardware_hash']:
                try:prior=require_prior_session(cfg,saved.get('session'),simulated=False)
                except (ValueError,OSError):pass
        if not session.progress.get('commissioning_passed'):
            level='lite' if prior else 'full';trials=commissioning_trials(cfg,session.T0)[:1 if prior else 3]
            ui.say('首次三点逐阶段验收。' if not prior else '该哈希已通过三点验收；本次首次试验逐阶段确认。')
            rows=[]
            for index,item in enumerate(trials):
                item['trial_id']=f'commission_{index+1:04d}'
                if item['trial_id'] in session.progress['completed']:
                    rows.append(json.loads((session.path/'trials'/item['trial_id']/'meta.json').read_text()));continue
                if not item['valid']:raise ValueError('验收点被几何否决，不能用 PASS 覆盖：'+str(item['safety_reasons']))
                runner.single_step=True;runner.menu_progress=f'验收 {index+1}/{len(trials)}'
                row=runner.contact_trial(item);rows.append(row);check_zero_drift(runner,row)
                if runner.quit_requested:return finish_run(runner,ui)
            report=build_acceptance(session,rows,level=level,prior=prior)
            atomic_json(session.path/'wizard_commissioning.json',report)
            ui.say('验收摘要：'+('通过' if report['measurements_passed'] else '失败')+'；'+str(report['failures']))
            if not report['measurements_passed']:raise ValueError('实测验收未通过，停止采集')
            ui.confirm('现场核对并输入 PASS：','PASS');report.update(passed=True,operator_confirmation='PASS')
            atomic_json(session.path/'wizard_commissioning.json',report)
            session.meta['commissioning_acceptance']=report;session.progress['commissioning_passed']=True
            session.save_meta();session.save_progress()
            if not prior:atomic_json(cache,{'hardware_hash':profile['hardware_hash'],'session':str(session.path.resolve()),'simulated':False})
        require_current_commissioning(session,args.mode);require_force_report(cfg);stop_station_gate(profile);gate_opening(profile)
        runner.single_step=False;count=0;last_block=session.meta.get('visual_orientation_block')
        completed_marked=sum(item_done(session,item,args.mode) for item in plan)
        for item in pending:
            if item_done(session,item,args.mode):continue
            ident=item.get('trial_id',item.get('point_id'))
            if not item['valid']:
                session.start_trial(ident);session.complete({'trial_id':ident,'task':args.mode,'bottle':item['bottle'],
                    'kind':item['kind'],'status':'skipped_safety','skip_reason':item['safety_reasons'],'auto':{},'stages':[],'operator':{}})
                completed_marked+=1;continue
            if args.limit is not None and count>=args.limit:break
            if session.progress.get('last_bottle')!=item['bottle']:
                ui.confirm(f"换为 {item['bottle']} 瓶，放回同一标记处，回车自动检查：")
                runner.check_bottle('bottle_switch');session.progress['last_bottle']=item['bottle'];session.save_progress()
            if args.mode=='visual' and item['orientation_block']!=last_block:
                angle=item['bottle_orientation_deg']
                ui.confirm(f'把瓶子转到 {angle}°、底部不离标记，回车检查后记录该朝向参考：')
                runner.check_bottle('orientation_switch')
                from prepare import save_reference_preview
                folder=session.path/'references'/('orientation_'+str(item['orientation_block'])+'_'+str(time.time_ns()))
                samples=backend.image_samples();save_reference_preview({k:v.data for k,v in samples.items()},folder,backend.monotonic_ns())
                last_block=item['orientation_block'];session.meta['visual_orientation_block']=last_block;session.save_meta()
            runner.menu_progress=f"{completed_marked+1}/{len(plan)}"
            runner.single_step=bool(existing and count==0)
            if args.mode=='contact':
                row=runner.contact_trial(item);check_zero_drift(runner,row)
            else:
                if ident in session.progress['completed']:
                    row=json.loads((session.path/'trials'/ident/'meta.json').read_text());runner.meta=row
                else:row=runner.visual_point(item)
                row['offsets']={k:float(v) for k,v in zip(('a_mm','c_mm','l_mm'),item['delta_acl_mm'])}
                row.update(orientation_block=item['orientation_block'],operator_bottle_orientation_deg=item['bottle_orientation_deg'])
                runner.check_bottle('visual_after_capture')
                if row['status']=='completed':
                    with runner.phase('LABEL'):runner.label()
                    session.complete(runner.meta)
            count+=1;completed_marked+=1;runner.single_step=False
            if runner.quit_requested:break
            if count%cfg['design'].get('baseline_every',30)==0:runner.baseline()
        return finish_run(runner,ui)
    except BaseException as exc:
        cleanup(runner,backend,exc,ui)
        if session:
            try:analyze_and_print(session,ui)
            except Exception as analysis_error:ui.say('部分分析未完成：'+str(analysis_error))
        raise
    finally:
        if backend is not None:backend.close()


def verify_at_T0(runner):
    from .geometry import pose_error
    error=pose_error(np.asarray(runner._healthy_snapshot()['tcp']),runner.session.T0)
    if error[0]>runner.cfg['motion']['pose_tolerance_mm'] or error[1]>runner.cfg['motion']['pose_tolerance_deg']:
        raise ValueError('验收循环必须保持在实测 T0；TCP 已改变')


def stop_confirm(message,cfg,evidence,ui):
    if '用 move-tcp 将实验臂复位' in message:
        return_free_for_stop(cfg,evidence,ui)
        return ''
    return ui.ask(message)


def return_free_for_stop(cfg,evidence,ui):
    """Automatic guarded reset BETWEEN stop trials, never during SIGKILL observation.

    The stop-test parent remains read-only during each measured child run. A
    temporary ordinary bridge exists only after that child has fully exited.
    """
    from .geometry import pose_error
    from .runtime import Acquisition
    target=np.asarray(cfg['preparation']['axis_test_pose'])
    with LiveReader(cfg) as reader:
        error=pose_error(reader.pose(),target)
        if error[0]<=cfg['motion']['pose_tolerance_mm'] and error[1]<=cfg['motion']['pose_tolerance_deg']:
            return
    ui.say('停止子试验已结束；现在自动回到已示教自由位，保持几何/新鲜度/非接触与发布者检查。')
    folder=Path(evidence)/('stop_return_'+str(time.time_ns()))
    session=Session.create(folder,cfg,np.asarray(cfg['T0']),simulated=False)
    backend=None;runner=None
    try:
        backend=load_hardware(cfg);backend.prepare_zero(session,confirm=ui.ask)
        ParkGuard.capture(session,backend.park_observation(),now_ns=backend.monotonic_ns())
        runner=WizardRunner(session,backend,ui,physical_authorized=True)
        runner.authorize_action(target)
        runner.meta={'trial_id':'stop_return','task':'preparation','stages':[],'auto':{'exceptions':[]},'simulated':False}
        raw=folder/'return_motion';raw.mkdir()
        runner.recorder=Acquisition(backend,raw)
        try:
            with runner.phase('CHECK'):runner.ensure_open();runner.move(target,noncontact=True)
        finally:
            runner.recorder.close();runner.recorder=None;atomic_json(raw/'meta.json',runner.meta)
    except BaseException as exc:
        cleanup(runner,backend,exc,ui);raise
    finally:
        if backend is not None:backend.close()


class ResetStation(ValueError):
    pass


def setup_real(args,ui):
    for attempt in range(2):
        try:
            if attempt==0 and getattr(args,'redo',False):raise ResetStation('操作员主动要求重新设置')
            return _setup_real_impl(args,ui)
        except ResetStation as exc:
            if attempt:raise ValueError('硬件/代码持续改变，暂停设置：'+str(exc)) from exc
            path=Path(args.profile);evidence=path.parent/(path.stem+'_setup')
            stamp=str(time.time_ns())
            if evidence.exists():evidence.rename(evidence.with_name(evidence.name+'_previous_'+stamp))
            if path.exists():path.rename(path.with_name(path.stem+'_previous_'+stamp+path.suffix))
            ui.say('旧 profile/断点和原始数据已归档；原因：'+str(exc)+'。重新开始 setup，不沿用旧放行。')


def repair_configuration(cfg,state,checkpoint,ui):
    """Repair typed numeric parameters inside the wizard, never hand-edit JSON."""
    fields=[('force.contact_min_N','接触阈值'),('force.target_N','目标力'),('force.max_N','逐指力上限'),
        ('force.target_tolerance_N','力容差'),('force.probe_fraction','PROBE 比例'),
        ('hardware.public_ros.force_deadband_N','力环死区'),('motion.free_speed_mm_s','自由空间速度'),
        ('motion.near_speed_mm_s','接近速度'),('motion.lift_speed_mm_s','提起速度'),
        ('gripper.close_speed_mm_s','闭合速度'),('close.width_resolution_mm','实测开度分辨率'),
        ('gripper.stall_window_s','停滞观察时间'),('close.max_overtravel_mm','允许闭合过冲'),
        ('stop_test.trigger_s','停止触发时间'),('stop_test.timeout_s','停止观察超时'),
        ('stop_test.max_extra_mm','允许额外位移'),('geometry.gripper.pad_depth_mm','软胶豁免厚度'),
        ('geometry.gripper.pad_size_al_mm','胶面 a/l 尺寸')]
    from .wizard_station import get,set_value
    while True:
        errors=validate_config(cfg)
        if not errors:return
        ui.say('以下配置组合尚未通过；未创建正式控制后端：'+'；'.join(errors))
        for index,(path,label) in enumerate(fields,1):ui.say(f'{index}. {label} = {get(cfg,path)}')
        answer=ui.ask('输入要修改的参数编号；q 保存并退出：')
        if answer=='q':save_yaml(checkpoint,state);raise KeyboardInterrupt('设置已保存，参数组合尚未通过')
        try:index=int(answer)-1
        except ValueError:continue
        if not 0<=index<len(fields):continue
        path,label=fields[index];current=get(cfg,path)
        count=len(current) if isinstance(current,list) else 1
        set_value(cfg,path,numeric(ui,label,current,count))
        save_yaml(checkpoint,state)
