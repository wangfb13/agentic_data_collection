"""White-bottle a/h transition collection through the existing V4 symbolic director."""
import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
import time
import numpy as np

from bottle_scan.geometry import pose_error, validate_transform
from bottle_scan.e1e2_core import opening_within_tolerance
from bottle_scan.state_graph import campaign
from bottle_scan.state_graph.extract_nodes import extract_nodes
from . import ah_evidence as evidence, placement_probe as probe
from . import ah_current
from .campaign import technical_error
from .ah_geometry import COORDS, KIND, basis, measured, position_allowed, position_limits, target, validate_cell, horizontal, coord_names
from bottle_scan.state_graph.common import digest
from .placement_al_probe import child, contact

HISTORY=Path('outputs/white_placement_20261117_112520')
DEFAULT_OUT=Path('outputs/state_graph/campaign_white_ah_P2')
ADAPTER='skills.contact_position_a.campaign_ah:'
POSITION_TOLERANCE_MM=2.
A_RANGE=(-10,10)
H_RANGE=(-3,45)
RANGE_VERSION="P2_a10_h_minus3_plus45_v1"
WBC_STAMP_FAULT = r'(?:left|right) WBC timestamp is stale or clocks disagree: -?\d+\.\d+ s'


def configuration(out, *, center=None, legacy=False,version=4,overrides=None):
    bounds=((-3,3),(-3,3)) if legacy else (A_RANGE,H_RANGE)
    steps=[max(abs(lo),abs(hi)) for lo,hi in bounds]
    grid=[[-3,0],[3,0],[0,-3],[0,3]] if legacy else (
        [[a,0] for a in (-10,-5,-3,3,5,10)] + [[0,h] for h in (-45,-30,-15,-10,-5,-3,3,5,10,15,30,45)])
    spec=probe.read('skills/contact_position_a/spec.campaign_v4.json')
    spec['task']['action_space']=dict(id=KIND,dim=2,names=['delta_tcp_a_mm','delta_base_h_mm'],max_step=steps,
        frame='mixed: [TCP a at motion start, base_link Z]',candidate_actions=grid)
    spec['capabilities']={KIND:dict(kind='vector',dim=2,max_action=steps,handler=ADAPTER+'execute_trial')}
    spec['safety_envelope']['max_action']=steps
    spec['runtime']['max_steps']=1
    spec['budget'].update(max_batch_trials=4,max_new_trials=24)
    variables=spec['perturbation_variables']
    for key,bound,step in zip(COORDS,bounds,steps):
        variables[key]=dict(kind='continuous',envelope=list(bound),initial_range=list(bound),reference=0.,
                            nominal_step=3,max_expansion=step,active=True,unit='mm',executor='placement.'+key)
    adapters=spec['adapters']
    adapters.pop('campaign_prepare')
    adapters.update(trials='skills.contact_position_a.ah_evidence:iter_trials',campaign_executor=ADAPTER+'execute_trial',
        executor=ADAPTER+'translate_plan',campaign_recover=ADAPTER+'recover_trial',
        exploration_context='skills.contact_position_a.ah_evidence:exploration_context',
        validate_collection_cell='skills.contact_position_a.ah_geometry:validate_cell')
    del spec['observation']['modalities']['vis']
    cfg=probe.read('config.state_graph.campaign_v4.json')
    cfg.update(spec=str(out/'instance.json'),sources=[str(out/'history_windows.json')])
    cfg['execution']=dict(executable_variables=[*COORDS,'mass_class'],executable_actions=[KIND],
        fixed_settings={'l_offset_mm':0,'approach_pose':'pose_0','closure_floor':.33},required_blocks=['mass_class'],
        sampling_preference={
            'source':'operator_reported_deployment_preference',
            'primary_comparisons':['no_translation','tcp_a_positive_or_negative','base_h_positive'],
            'base_h_negative':'optional',
            'instruction':'常见需求是向上修正。把试次优先用于比较同类起点不移动、沿 a 调整、向 +h 调整后的轻触。'
                          '−h 无覆盖不构成必须补采的理由，也不要求正负等量。若选择 −h，在 reason 中说明它能区分的具体假设。'
                          '该偏好只针对恢复探索动作，不限制准备起点或返回途中的下降。'},
        placement='P2',center_file=str((HISTORY/'P2/center/observation.json').resolve()),
        phase='仅采集触觉、双指去零六维力及实测动作前后转移。无下游标签。基于缺口自由选择方向、幅度、起点、重复数。'
              '每个动作只选一个分量：[0,+d]沿基座Z向上，[0,-d]向下，[d,0]沿当前TCP的a。'
              '新计划起点及终点每个系数在±3mm内。补噪声请从同一个设置、同一质量重复轻触。'
              '证据足够时 pause。历史未知瓶重不能作为轻重结论。')
    if not legacy:
        if center is None:
            frozen=Path(out)/'placement_basis.json'
            center=(probe.read(frozen)['measured_pose'] if frozen.is_file() else
                    probe.read(cfg['execution']['center_file'])['window_pose']['T_base_tcp_mm'])
        spec['placement_geometry']={'range_version':RANGE_VERSION,'tcp_a_base_z':float(basis(center,[0,0,1])[2,0])}
        cfg['execution']['phase']=(
            '采集同类起点的轻触与实测转移。单次只动一个轴，a 单步不超过 10 mm，h 单步不超过 45 mm。'
            'P2 起点和终点 a ∈ [−10,+10]，h ∈ [−3,+45] mm，实际基座 Z 相对 P2 也在 [−3,+45] mm。'
            '倾斜 a 轴也改变实际高度，基座高度等于 h + tcp_a_base_z × a。'
            '优先探索瓶身向瓶盖方向目前未观测的高度，可以直接在新高度轻触，或沿 +h 移动后重触。'
            '小幅向上有样本只说明该局部有数据；根据证据选幅度，不要求等距扫点，不假定 +45 即为瓶盖或 Ready。'
            '允许从高处返回，−h 不需要与 +h 对称配额。返回数据不代替下游结果。')
    if version==5:
        from bottle_scan.state_graph.v5_profile import configure
        from . import ah_v5_config
        spec=configure(spec,overrides)
        ah_v5_config.configure(spec,cfg)
        cfg['execution'].pop('sampling_preference',None)
        cfg['execution'].pop('phase',None)
    return spec,cfg


def translate_plan(plan,spec):
    for cell in plan['cells']:
        validate_cell(cell,spec,'',{})
    return dict(schema='ah_placement_plan_v1',executed=False,trials=campaign.expand_plan(plan,spec))


def check_assets():
    hw,ref,model,serials=probe.assets()
    if not np.array_equal(hw['axes_tcp']['a'],[0,0,1]) or serials!={'left':'X2620410318','right':'X2620410347'}:
        raise ValueError('当前 P2 资产要求已验证的 TCP z 为 a 轴以及 0318/0347 双指')
    return hw,(ref,model,serials)


def cancelled_before_moving(attempt):
    if attempt['reason']=='未授权本次动作':
        return True
    if not attempt.get('archive'):
        return False
    archive=Path(attempt['archive'])
    if (archive/'P2').exists() or not (archive/'interruption.json').exists():
        return False
    logs=list(archive.glob('*_move*/tcp_move_*.json'))
    return bool(logs) and all(probe.read(p).get('result')=='operator_cancelled' for p in logs)


def resume_range(out,spec,cfg):
    # Validate all old/new files before writing. Each replacement is atomic; state commits last.
    old_spec,old_cfg=configuration(out,legacy=True)
    updates=[]
    for name in ('spec','instance','input','config'):
        path=out/(name+'.json')
        is_spec=name in ('spec','instance')
        expected=spec if is_spec else cfg
        saved=probe.read(path)
        if saved==expected:
            continue
        if not is_spec:
            saved['execution'].setdefault('sampling_preference',cfg['execution']['sampling_preference'])
        if saved!=expected and saved!=(old_spec if is_spec else old_cfg):
            raise ValueError('现有 campaign 与可迁移的版本不一致：'+name)
        updates.append((path,expected))
    state=probe.read(out/'state.json')
    if state.get('range_version')!=RANGE_VERSION:
        if state.get('frozen') or state.get('complete'):
            raise ValueError('冻结或完成的实验需使用独立输出目录')
        for trial in state['trials']:
            if trial['status']=='started':
                # A known cancelled/failed attempt can be archived without invoking hardware.
                try:
                    recover_trial(trial['request'],Path(trial['directory']),spec,cfg,interactive=False)
                except campaign.RetryTrial as exc:
                    campaign.requeue_attempt(out,state,trial,str(exc))
        rule=state['rule']
        parent=rule['rule_id']
        for key in COORDS:
            rule['variables'][key]['range']=list(spec['perturbation_variables'][key]['initial_range'])
        rule.update(rule_id='rule_'+digest([parent,RANGE_VERSION])[:16],parent_rule_id=parent,
                    reason='现场更新 a/h 探索范围',evidence_refs=[])
        state['range_version']=RANGE_VERSION
        pending=state.get('pending')
        if pending and pending['role']=='exploration':
            retired=[]
            for t in state['trials']:
                if t['trial_id'] not in pending['trial_ids'] or t['status']!='queued':
                    continue
                attempts=t.get('incomplete_attempts',[])
                if not all(cancelled_before_moving(a) for a in attempts):
                    continue  # A technical retry keeps its original physical protocol.
                if any(f.name!='request.json' for f in Path(t['directory']).iterdir()):
                    continue
                t.update(status='superseded',superseded_by_range=RANGE_VERSION)
                retired.append(t['trial_id'])
            if retired:
                pending['superseded_trial_ids']=retired
                pending['trial_ids']=[i for i in pending['trial_ids'] if i not in retired]
                if not any(t['trial_id'] in pending['trial_ids'] and t['status']!='complete' for t in state['trials']):
                    state.update(last_batch=pending,pending=None,phase='analyze')
        updates.append((out/'state.json',state))
    for path,value in updates:
        campaign.atomic(path,value)


def initialize(out):
    out=Path(out).resolve()
    with campaign.ownership(out):
        spec,cfg=configuration(out)
        if (out/'state.json').exists():
            resume_range(out,spec,cfg)
            return
        _,identity=check_assets()
        trials=evidence.historical(HISTORY.resolve(),identity)
        campaign.atomic(out/'history_windows.json',trials)
        campaign.atomic(out/'instance.json',spec)
        campaign.atomic(out/'input.json',cfg)
        center,_=contact(Path(cfg['execution']['center_file']),*identity)
        campaign.atomic(out/'placement_basis.json',dict(placement='P2',source=cfg['execution']['center_file'],
                      measured_pose=center.tolist(),reference_digest=identity[0]['reference_digest']))
        state=campaign.initialize(out/'input.json',out)
        state['range_version']=RANGE_VERSION
        # Historical measured overshoot is evidence, not an enlargement of the commanded envelope.
        for key in COORDS:
            state['rule']['variables'][key]['range']=spec['perturbation_variables'][key]['initial_range']
        campaign.atomic(out/'state.json',state)


def current_pose(folder, label, hw, invoke):
    path=folder/(label+'.json')
    try:
        invoke('record_pose.py','--arm','right','--label',label,'--output',path,
               stdout=subprocess.DEVNULL,stderr=subprocess.PIPE,text=True)
    except subprocess.CalledProcessError as exc:
        diagnostic=(exc.stderr or '').strip()
        reason=diagnostic.rsplit('\n',1)[-1]
        if exc.returncode>0 and re.fullmatch(
                r'bottle_scan\.hardware_arm\.ArmObservationError: Read-only arm connection timed out: '
                + WBC_STAMP_FAULT,reason):
            campaign.atomic(folder/'pose_clock_failure.json',dict(stage=label,reason=reason,
                            command='record_pose.py',returncode=exc.returncode,source='child_stderr',
                            stderr=diagnostic))
            requeue_technical_interruption(folder)
        if diagnostic:
            print(diagnostic,file=sys.stderr)
        raise
    r=probe.read(path)
    arm=r['pose_source']['arms']['right']
    if arm['tcp_frame']!='right_gripper_base_link' or r['pose_source']['base_frame']!='base_link':
        raise ValueError('当前 TCP / base frame 不匹配')
    g=r['gripper']
    settings=hw['gripper']
    if not (g['fresh'] and g['valid'] and opening_within_tolerance(g['normalized'],settings)):
        reason=(f"移动前右爪须全开、空载。反馈={g.get('normalized')}，"
                f"目标={settings['open_normalized']} ± {settings['open_tolerance']}，"
                f"fresh={g['fresh']}，valid={g['valid']}")
        if label=='prepare_pose_0':
            campaign.atomic(folder/'initial_open_rejection.json',dict(stage=label,reason=reason,measurement=path.name))
            requeue_technical_interruption(folder)
        raise ValueError(reason)
    return validate_transform(r['T_base_tcp_mm'])


def move(folder,label,arguments,limit,z_bounds,invoke):
    logs=folder/label
    logs.mkdir()
    height_args=[] if z_bounds is None else ['--base-z-range-mm',*z_bounds]
    invoke('move_tcp.py','--arm','right',*arguments,'--speed-mm-s',2,'--max-distance-mm',limit,*height_args,
           '--pose-tolerance-mm',POSITION_TOLERANCE_MM,'--log-dir',logs,'--real')
    paths=list(logs.glob('tcp_move_*.json'))
    if len(paths)!=1 or probe.read(paths[0])['result']!='reached_by_measured_feedback':
        raise RuntimeError('未取得实测到位回执，停止当前试次')
    return paths[0]


def check_measured_pose(center,pose,axis,spec):
    if pose_error(center,pose)[1]>1:
        raise ValueError('当前姿态与 P2 朝向相差超过 1°')
    coords,residual=measured(center,pose,center,axis,spec)
    if residual>POSITION_TOLERANCE_MM or not position_allowed(coords,spec,
            base_height=pose[2,3]-center[2,3],tolerance=POSITION_TOLERANCE_MM):
        raise ValueError('实测 TCP 不在 P2 的 a/h 工作区')
    return coords


def prepare_position(folder,center,start_u,hw,spec,invoke,*,station_z_bounds=None):
    axis=hw['axes_tcp']['a']
    bounds=position_limits(spec)
    limit=max(spec['task']['action_space']['max_step'])
    goal=target(center,center,start_u,axis,spec)
    safe_z=center[2,3]+bounds[1]
    if station_z_bounds is not None:
        safe_z=[max(safe_z[0],station_z_bounds[0]),min(safe_z[1],station_z_bounds[1])]
        if not safe_z[0]<=goal[2,3]<=safe_z[1]:raise ValueError('Requested start exceeds station z bounds')
    # A straight line between any two nominal points is at most the sum of the two spans.
    # Leave room for the two measured endpoints and progress allowed by the existing feedback tolerance.
    max_legs=1+int(np.ceil((np.sum(bounds[:,1]-bounds[:,0])+2*POSITION_TOLERANCE_MM)/(limit-POSITION_TOLERANCE_MM-1)))
    for leg in range(max_legs):
        now=current_pose(folder,'prepare_pose_'+str(leg),hw,invoke)
        check_measured_pose(center,now,axis,spec)
        delta=goal[:3,3]-now[:3,3]
        distance=np.linalg.norm(delta)
        if distance<=.25:return
        end=goal[:3,3] if distance<=limit else now[:3,3]+delta*((limit-1)/distance)
        move(folder,'prepare_move_'+str(leg),['--xyz-mm',*end],limit,safe_z,invoke)
        if distance<=limit:return
    raise RuntimeError('准备移动未在分段预算内到达起点')


def issued_action(center,now,u,axis,spec):
    coords=check_measured_pose(center,now,axis,spec)
    goal=target(now,now,u,axis,spec)
    change,_=measured(now,goal,center,axis,spec)
    bounds=position_limits(spec)
    # Both axes and the actual base height must respect the same nominal endpoint contract.
    # At a boundary, account for measured starting error without rotating the requested action.
    fraction=1.
    for x,dx,(lo,hi) in zip([*coords,now[2,3]-center[2,3]],
                             [*change,goal[2,3]-now[2,3]], [*bounds,bounds[1]]):
        if dx>1e-9:
            fraction=min(fraction,(hi-x)/dx)
        elif dx< -1e-9:
            fraction=min(fraction,(lo-x)/dx)
    fraction=max(0.,fraction)
    if fraction<=1e-9 or np.linalg.norm(u)*(1-fraction)>POSITION_TOLERANCE_MM+1e-9:
        raise ValueError('实测起点处无法在 P2 边界内执行所选动作')
    return u*fraction


def requeue_technical_interruption(directory):
    """A terminal device fault can survive the child process or the campaign itself."""
    directory=Path(directory)
    fault=None
    pose_fault=directory/'pose_clock_failure.json'
    if pose_fault.is_file():
        failure=probe.read(pose_fault)
        fault=(failure['stage'],pose_fault,failure['reason'])
    initial_rejection=directory/'initial_open_rejection.json'
    if initial_rejection.is_file():
        failure=probe.read(initial_rejection)
        fault=(failure['stage'],initial_rejection,failure['reason'])
    for slot in ('reprobe','initial'):
        paths=list(directory.glob('*/'+slot+'/observation.json'))
        if len(paths)!=1:
            continue
        path=paths[0]
        observation=probe.read(path)
        if observation.get('state')=='fault':
            message=str(observation.get('error',''))
            feedback_stop=('慢闭合指令与反馈相差超过' in message or '快慢闭合切换期间反馈继续偏离' in message)
            if technical_error(message) or re.fullmatch(WBC_STAMP_FAULT,message) or (feedback_stop and '已停止闭合' in message):
                fault=(slot,path,message)
        break  # A later capture takes precedence over any earlier one.
    moves=sorted(directory.glob('*_move*/tcp_move_*.json'))
    reached=[]
    execution_starts=[]
    for path in moves:
        receipt=probe.read(path)
        if receipt.get('execution_start_mm') is not None:
            execution_starts.append(str(path.relative_to(directory)))
        if receipt.get('result')=='reached_by_measured_feedback':
            reached.append(str(path.relative_to(directory)))
        elif fault is None and receipt.get('finished_unix_ns'):
            reason=str(receipt.get('error',''))
            interpolation_stopped=(reason=='机械臂反馈/控制故障：invalid rotation axis' and
                                   receipt.get('software_hold')=='fresh_measured_pose_published')
            if (receipt.get('result')=='operator_cancelled' or
                    (receipt.get('result')=='failed' and
                     (interpolation_stopped or 'TCP 移动等待实测到位超时' in reason or technical_error(reason)))):
                fault=(path.parent.name,path,reason or '现场取消移动')
    if fault:
        stage,path,reason=fault
        campaign.atomic(directory/'interruption.json',dict(stage=stage,reason=reason,
            evidence=str(path.relative_to(directory)),reached_move_receipts=reached,
            move_receipts_with_execution_start=execution_starts))
        raise campaign.RetryTrial(f'{stage}：{reason}')


def acquire(folder,trial,slot,identity,center,settings,mass,invoke,placement='P2',*,index=None):
    invoke('-m','skills.contact_position_a.placement_probe','probe','--run',folder,
           '--placement',placement,'--trial',slot,'--real')
    path=folder/placement/slot/'observation.json'
    if not path.exists():raise campaign.RetryTrial('现场取消了轻触，当前试次未完成')
    r=probe.read(path)
    if r.get('state')=='released' and (r.get('status')!='contact' or not (r.get('probe') or {}).get('valid')):
        return None,'无有效双侧轻触：'+str(r.get('status'))
    if r.get('bottle_static') is not True:return None,'本次瓶位连续性未经确认'
    return evidence.window(path,identity,center,trial,(0 if slot=='initial' else 1) if index is None else index,settings,mass,placement),None


def execute_trial(request,directory,spec,cfg,*,real,interactive,invoke=child,ask=input):
    if spec.get('graph_schema')=='state_graph_v5':
        from .p3_trial import execute_trial as run_rh
        return run_rh(request,directory,spec,cfg,real=real,interactive=interactive)
    if not real or not interactive or request['role']!='exploration' or request['mode']!='planned':
        raise ValueError('此入口仅执行现场确认的探索采集')
    validate_cell({'settings':request['settings'],'actions':request['actions'],'blocks':[request['block']]},spec,'',{})
    hw,identity=check_assets()
    axis=hw['axes_tcp']['a']
    current=spec.get('current_placement')
    placement=current or 'P2'
    center=None
    if not current:
        center,_=contact(Path(cfg['execution']['center_file']),*identity)
        frozen=probe.read(Path(request['campaign'])/'placement_basis.json')
        if frozen['reference_digest']!=identity[0]['reference_digest'] or not np.array_equal(frozen['measured_pose'],center):
            raise ValueError('P2 采集基准与建批时不同')
        basis(center,axis)
    trial=request['trial_id'];directory=Path(directory)
    mass=request['block']['mass_class']
    start=None if current else [request['settings'][k] for k in COORDS]
    bottle='轻白瓶' if mass=='light' else '重白瓶'
    if current:
        print(f'{trial} 记录组 {placement}，{bottle}。在本次位置对准瓶身，原地轻触。'
              f'起始偏移未知，动作 {request["actions"]}。',flush=True)
    else:
        print(f'{trial} 放{bottle}到 P2 标记。初始 [a,h]={start} mm，动作 {request["actions"]}。',flush=True)
    if ask('瓶底放稳、右爪张开空载、已停止遥操，确认整段直线路径与桌面有间隙。输入 MOVE：').strip()!='MOVE':
        raise campaign.RetryTrial('未授权本次动作')
    def run_child(*args,**kwargs):
        try:
            invoke(*args,**kwargs)
        except subprocess.CalledProcessError:
            requeue_technical_interruption(directory)
            raise

    t0=time.monotonic()
    group=trial
    if current:
        group=current+'_'+trial
        current_pose(directory,'prepare_pose_0',hw,run_child)
        z_limits=cfg['execution'].get('table_tcp_z_mm')  # Only an independently configured station limit.
        settings={}
    else:
        prepare_position(directory,center,start,hw,spec,run_child)
        settings=dict(zip(COORDS,start))
        z_limits=center[2,3]+position_limits(spec)[1]
    windows=[];moves=[]
    first,reason=acquire(directory,group,'initial',identity,center,settings,mass,run_child,placement)
    if first:
        windows.append(first)
        cumulative=np.zeros(2)
        for step,action in enumerate(request['actions'],1):
            first=windows[-1]
            suffix='' if step==1 else '_'+str(step)
            # Each relative command is transformed by move_tcp from its own actual start snapshot.
            u=np.array(action['u'])
            now=current_pose(directory,'action_pose'+suffix,hw,run_child)
            prev=first['provenance']['measured_pose']
            if pose_error(prev,now)[0]>1 or pose_error(prev,now)[1]>1:
                raise ValueError('释放后位置改变，不能把此动作归给原轻触')
            if current:
                issued=u
            else:
                if not first['plan_coords']:
                    raise ValueError('首触未取得有效的 a/h 位移测量')
                issued=issued_action(center,now,u,axis,spec)
            frame_field='preflight_pose' if current else 'frame_start'
            campaign.atomic(directory/('action_command'+suffix+'.json'),{'requested_u':u.tolist(),'preflight_u':issued.tolist(),
                            frame_field:now.tolist(),'range_version':ah_current.MODE if current else RANGE_VERSION})
            index=0 if u[0] else 1
            frame=(('horizontal_tcp_a' if horizontal(spec) else 'tcp'),'base')[index]
            amount=issued[index]
            print(f'本次预览 {("a","h")[index]}={amount:+.2f} mm（计划 {u[index]:+g} mm）',flush=True)
            options=['--relative-at-confirm'] if current or horizontal(spec) else []
            if frame=='horizontal_tcp_a':options+=['--min-horizontal-projection',spec['geometry']['minimum_horizontal_projection']]
            path=move(directory,'action_move'+suffix,[*options,'--frame',frame,'--direction','z+' if amount>0 else 'z-',
                      '--distance-mm',abs(amount)],spec['task']['action_space']['max_step'][index],
                      z_limits,run_child)
            moves.append(str(path))
            cumulative+=u
            terminal_settings={} if current else dict(zip(coord_names(spec),np.asarray(start)+cumulative))
            terminal,reason=acquire(directory,group,'reprobe'+suffix,identity,center,terminal_settings,mass,run_child,placement,index=step)
            if terminal:
                evidence.link(first,terminal,[path],axis,command=issued if current else None,spec=spec)
                windows.append(terminal)
                if not terminal.get('u_receipt'):
                    reason='前后轻触与实测移动回执不一致，未计作转移'
                    break
            else:break
    if windows:windows[-1]['is_terminal']=True
    data=directory/'windows.json'
    campaign.atomic(data,[dict(trial_id=trial,windows=windows)])
    nodes,_=extract_nodes([data],spec,directory/'extracted')
    result=dict(trial_id=trial,status='failed' if reason else 'complete',nodes=nodes,assessment=None,
                attempt_reason=reason,move_receipts=moves,cost={'actions':len(moves),'wall_seconds':time.monotonic()-t0})
    campaign.atomic(directory/'completed_capture.json',result)
    return result


def recover_trial(request,directory,spec,cfg,*,interactive):
    path=Path(directory)/'completed_capture.json'
    if path.is_file():
        return probe.read(path)
    requeue_technical_interruption(directory)
    return None


def offer_defer(trial, reason):
    print(f"[graph-loop] {trial['trial_id']} 中断：{reason}", flush=True)
    try:
        answer = input('确认机械臂已停止、瓶已放稳、夹爪已张开空载后，输入 SKIP 暂缓本条、继续本批；回车暂停：')
    except EOFError:
        return False
    return answer.strip().upper() == 'SKIP'


def show_deferred(state):
    trials = campaign.deferred_trials(state)
    print(f"[graph-loop] 本批暂缓 {len(trials)} 条。")
    for trial in trials:
        evidence = trial['incomplete_attempts'][-1]
        print(f"  {trial['trial_id']}  {evidence['reason']}  {evidence['archive']}")
    if trials:
        print('问题修复后，用 --retry-deferred all 或 --retry-deferred 试次ID 恢复原请求，再用原采集命令继续。')


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path)
    mode=p.add_mutually_exclusive_group()
    mode.add_argument('--from-current',metavar='RUN_GROUP',type=probe.tag,
                      help='每次独立从当前位置轻触；组名仅用于续跑，默认沿用 P3')
    mode.add_argument('--fixed-p2',action='store_true',help='复现原固定 P2 对照实验')
    task=p.add_mutually_exclusive_group()
    task.add_argument('--real',action='store_true')
    task.add_argument('--preview-plan',action='store_true',help='从已结算证据离线预览下一批，保留现场队列')
    task.add_argument('--list-deferred',action='store_true',help='只读列出本批暂缓试次及证据目录')
    task.add_argument('--retry-deferred',nargs='+',metavar='ID',help='把指定暂缓试次或 all 放回原队列，仅改队列')
    p.add_argument('--graph-version',type=int,choices=(4,5),default=4,help='Python entry preserves legacy replay; start_state_graph_ah.sh defaults to version 5')
    p.add_argument('--allow-agent',action='store_true')
    opts=p.parse_args(argv)
    if opts.graph_version==5:
        from .ah_transition_v5 import entry
        return entry(opts)
    if not opts.fixed_p2 and not opts.from_current:
        if opts.out is None:
            opts.from_current='P3'
        elif (opts.out/'spec.json').is_file():
            opts.from_current=probe.read(opts.out/'spec.json').get('current_placement')
    if opts.list_deferred or opts.retry_deferred:
        out = opts.out or (DEFAULT_OUT.with_name('campaign_white_ah_here_'+opts.from_current) if opts.from_current else DEFAULT_OUT)
        if not (out/'state.json').is_file():
            p.error(f'找不到已有采集记录 {out}')
        if opts.list_deferred:
            show_deferred(campaign.read(out/'state.json'))
        else:
            ids = campaign.release_deferred(out, opts.retry_deferred)
            print('[graph-loop] 已恢复到原队列：' + (', '.join(ids) or '无暂缓试次'))
        return 0
    if opts.preview_plan:
        if not opts.from_current or not opts.allow_agent:
            p.error('--preview-plan 需要自由起点组和 --allow-agent')
        out=opts.out or DEFAULT_OUT.with_name('campaign_white_ah_here_'+opts.from_current)
        print(ah_current.preview_plan(out,opts.from_current))
        return 0
    if opts.from_current:
        opts.out=opts.out or DEFAULT_OUT.with_name('campaign_white_ah_here_'+opts.from_current)
        ah_current.initialize(opts.out,opts.from_current,DEFAULT_OUT)
    else:
        opts.out=opts.out or DEFAULT_OUT
        initialize(opts.out)
    state=campaign.run(opts.out,real=opts.real,allow_agent=opts.allow_agent,
                       on_interruption=offer_defer if opts.real else None)
    if state['phase']=='deferred_review':
        show_deferred(state)
    if state['phase']=='awaiting_result':
        for trial in state['trials']:
            if trial['status']=='started' and trial.get('error'):
                print(f"[graph-loop] {trial['trial_id']} 等待处理上次中断：{trial['error']}",flush=True)
    print(json.dumps({'out':str(opts.out),'phase':state['phase'],'spent':state['spent'],'pending':state['pending']},ensure_ascii=False))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
