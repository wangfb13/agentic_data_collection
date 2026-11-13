"""Actual a-axis trial bridge: V4 -> ForcePolicy -> outcome review -> graph rows."""
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import time
from bottle_scan.state_graph.campaign import atomic
from bottle_scan.state_graph.extract_nodes import extract_nodes
from bottle_scan.state_graph.skill_interface import normalize_action
from .scheduler import append_reviewed_outcome


def launch_request(request,directory,spec,cfg):
    execution=cfg['execution'];settings=request['settings']
    # This task adapter currently implements a translation and its existing probe.
    # New task/action adapters can implement other capabilities, without altering
    # the generic graph or campaign loop.
    if settings.get('l_offset_mm',0)!=0 or settings.get('approach_pose','pose_0')!='pose_0':
        raise ValueError('a-axis executor implements the existing fixed pose and a translation only')
    for action in request['actions']:
        value=normalize_action(action,spec)
        if value['kind']!='tcp_translation_a':raise ValueError('No a-axis hardware handler for '+value['kind'])
    config=json.loads(Path(execution['scheduler_config']).read_text())
    config['state_graph_execution']={'repository':str(Path(__file__).resolve().parents[2]),'request':str(directory/'request.json')}
    config['single_correction_open_handoff']=False
    floor=float(settings.get('closure_floor',.33))
    if floor!=spec['post_conditions'][0]['consumer']['params']['closure_floor']:raise ValueError('Trial floor differs from active consumer')
    config['execution_config']['gripper']['min_normalized']=floor
    atomic(directory/'scheduler_config.json',config)
    env=dict(os.environ);env.update(execution.get('environment',{}))
    env.update(CONTACT_SCHEDULER_CONFIG=str(directory/'scheduler_config.json'),CONTACT_SCHEDULER_LOG_DIR=str(directory/'episodes'),
        CONTACT_SKILL_AFTER='handoff',CONTACT_SKILL_RECOVERY='enabled',CONTACT_EXPERIMENT_B_OFFSET_MM=str(settings.get('a_offset_mm',0)),
        CONTACT_GRAPH_MASS_CLASS=str(request['block'].get('mass_class','unrecorded')),CONTACT_GRAPH_APPROACH_POSE=settings.get('approach_pose','pose_0'),
        CONTACT_SKILL_TRIAL_ID=request['trial_id'],RIGHT_FORCE_POSITION_MIN=str(floor),RIGHT_GRIPPER_CLOSE_LIMIT=str(floor),
        CONTACT_GRAPH_TIME_PERIOD=time.strftime('%Y%m%d'),CONTACT_GRAPH_SENSOR_VERSION=str(execution.get('sensor_version','current_recorded_session')))
    return ['bash',execution['entry'],'skill-run'],env


def execute_trial(request,directory,spec,cfg,*,real=False,interactive=True):
    if not real:raise ValueError('Real collection requires --real')
    directory=Path(directory)
    command,env=launch_request(request,directory,spec,cfg)
    if not interactive:raise ValueError('This instance needs on-site reset and real L2 outcome review; use interactive execution')
    print('\n[graph-loop] '+request['trial_id']+' '+request['role']+' '+json.dumps({'settings':request['settings'],'block':request['block'],'actions':request['actions']},ensure_ascii=False),flush=True)
    print('将本次瓶子放稳在原标记；机械臂停止、右夹爪全开空载，恢复到本配置 T0 或允许的退让位。停止遥操。',flush=True)
    if input('确认后输入 MOVE 执行这一次；其他输入保留待执行记录：').strip()!='MOVE':return None
    env['CONTACT_TABLETOP_CONFIRMED']='1'
    atomic(directory/'launch.json',{'command':command,'started_ns':time.time_ns(),'request':request,'model_id':request['model_id']})
    started=time.monotonic()
    # Existing V4 entry forwards the CONTACT_* / RIGHT_* namespace into Docker.
    rc=subprocess.run(command,env=env,check=False).returncode
    atomic(directory/'process.json',{'returncode':rc,'wall_seconds':time.monotonic()-started,'finished_ns':time.time_ns()})
    return recover_trial(request,directory,spec,cfg,interactive=interactive)


def recover_trial(request,directory,spec,cfg,*,interactive=True):
    directory=Path(directory);process=directory/'process.json'
    events_paths=sorted(directory.glob('episodes/episode_*/events.jsonl'))
    if not events_paths:return None
    if len(events_paths)!=1:raise ValueError('Expected one episode per campaign trial; review duplicate launches')
    events_path=events_paths[0]
    events=[json.loads(line) for line in events_path.read_text().splitlines() if line.strip()]
    selected=[e for e in events if e.get('event')=='graph_runtime_selected']
    if not selected or selected[-1]['request']['trial_id']!=request['trial_id']:raise ValueError('Episode did not load the requested graph adapter')
    if selected[-1]['model_id']!=(request['model_id'] or 'planned_exploration'):raise ValueError('Episode used a different model')
    outcomes=[e for e in events if e.get('event')=='state_graph_consumer_result']
    if not process.exists():
        # No terminal process receipt: a still-running session must not be reviewed.
        return None
    outcome=outcomes[-1]['consumer_outcome'] if outcomes else None
    if outcome and outcome.get('L1') is True and type(outcome.get('L2')) is not bool:
        if not interactive:return None
        print('复核实际结果：ForcePolicy 接手后，瓶子是否提起并保持、不滑落？不能用“两指接触”替代。')
        value=input('输入 y=成功 / n=失败 / u=未观察到（保留待复核）：').strip().lower()
        if value not in ('y','n'):return None
        task=input('主任务最终是否完成？ y / n / u=未验证：').strip().lower()
        result=append_reviewed_outcome(events_path,L2=value=='y',L3={'y':True,'n':False}.get(task),evidence={'source':'onsite_review','trial_id':request['trial_id'],'reviewed_ns':time.time_ns()})
        outcome=result['consumer_outcome']
    rows,report=extract_nodes([str(events_path)],spec,directory/'extracted')
    terminal=next((r for r in reversed(rows) if r['post'][request['post_condition']]['evaluated']),None)
    decisions=[e for e in events if e.get('event')=='recovery_skill_decision']
    invoked=any(e.get('skill_outputs',{}).get('invocation',{}).get('invoke') is True for e in decisions)
    exit_declared=any(e.get('skill_outputs',{}).get('exit',{}).get('completed') is True for e in decisions)
    assessment=None;review_file=directory/'initial_truth_review.json'
    if request['role']!='exploration':
        if review_file.exists():review=json.loads(review_file.read_text())
        elif interactive:
            print('首次接触的真实类别：直接交给同一 ForcePolicy 是否可完成提起保持？依据配对的无修正试验或独立复核，不以 T0 距离判断。')
            label=input('r=已验证可直接交接 / a=已验证需位置恢复 / u=未知：').strip().lower()
            evidence=input('记录上述判断所依据的试验编号或复核证据（未知可留空）：').strip() if label in ('r','a') else ''
            review={'initial_truth':{'r':'Ready','a':'NeedAction'}.get(label) if evidence else None,'evidence':evidence}
            atomic(review_file,review)
        else:return None
        evaluated=terminal is not None
        assessment={'evaluated':evaluated,'success':terminal['post'][request['post_condition']]['success'] if evaluated else None,
                    'initial_truth':review['initial_truth'],'initial_truth_evidence':review['evidence'],
                    'target_truth':terminal['post'][request['post_condition']]['success'] if evaluated else None,
                    'invoked':invoked,'exit_declared':exit_declared,'rejected':not exit_declared,
                    'task_success':(outcome or {}).get('L3'),'evidence':[str(events_path),str(review_file)]}
    proc=json.loads(process.read_text())
    applied=[r for r in rows if (r.get('u_receipt') or {}).get('applied')]
    return {'trial_id':request['trial_id'],'status':'complete' if terminal and proc['returncode']==0 else 'failed','nodes':rows,'assessment':assessment,
            'cost':{'actions':len(applied),'wall_seconds':proc['wall_seconds']},'evidence':[str(events_path)],
            'unassessed_reason':None if terminal else 'No actual consumer endpoint; hardware/policy failures are not invented consumer labels.'}


def prepare(spec,cfg):
    import hashlib
    spec=deepcopy(spec)
    root=Path(__file__).resolve().parents[2]
    visual=Path(spec['data']['visual_model_directory'])
    spec['data']['visual_model_directory']=str(visual if visual.is_absolute() else root/visual)
    for key in ('artifact','cache_dir'):
        if key in spec.get('tactile_encoder',{}):
            value=Path(spec['tactile_encoder'][key]);spec['tactile_encoder'][key]=str(value if value.is_absolute() else root/value)
    path=cfg['execution']['environment']['RIGHT_FORCE_POLICY_PATH']
    version=hashlib.sha256(Path(path).read_bytes()).hexdigest()
    for condition in spec['post_conditions']:
        if condition['consumer']['controller']=='existing_forcepolicy':condition['consumer']['version']=version
    return spec
