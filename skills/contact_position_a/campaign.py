"""Actual a-axis trial bridge: V4 -> ForcePolicy -> outcome review -> graph rows."""
from copy import deepcopy
import json
import os
import re
import sys
from pathlib import Path
import subprocess
import time
from bottle_scan.state_graph.campaign import RetryTrial, atomic
from bottle_scan.state_graph.extract_nodes import extract_nodes
from bottle_scan.state_graph.skill_interface import normalize_action
from .scheduler import append_reviewed_outcome
from .ready import FIXED_DEVICE, FIXED_GOALS, MEASURED_GOALS, check_fixed_config, device_floor, select


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
    active=spec['active_post_condition']
    if request['post_condition']!=active:raise ValueError('Request belongs to a different Ready goal')
    condition=next(c for c in spec['post_conditions'] if c['id']==active)
    floor=float(settings.get('closure_floor',condition['consumer']['params']['closure_floor']))
    if active == FIXED_DEVICE:
        config['state_graph_execution']['fixed5_device_range'] = True
    if floor!=condition['consumer']['params']['closure_floor']:raise ValueError('Trial floor differs from active consumer')
    config['execution_config']['gripper']['min_normalized']=floor
    after='stop' if active in FIXED_GOALS else 'handoff'
    config['skill_after']=after
    if active in FIXED_GOALS:check_fixed_config(config['execution_config'], active)
    atomic(directory/'scheduler_config.json',config)
    env=dict(os.environ);env.update(execution.get('environment',{}))
    env.update(CONTACT_SCHEDULER_CONFIG=str(directory/'scheduler_config.json'),CONTACT_SCHEDULER_LOG_DIR=str(directory/'episodes'),
        CONTACT_SKILL_AFTER=after,CONTACT_SKILL_RECOVERY='enabled',CONTACT_EXPERIMENT_B_OFFSET_MM=str(settings.get('a_offset_mm',0)),
        CONTACT_GRAPH_MASS_CLASS=str(request['block'].get('mass_class','unrecorded')),CONTACT_GRAPH_APPROACH_POSE=settings.get('approach_pose','pose_0'),
        CONTACT_SKILL_TRIAL_ID=request['trial_id'],RIGHT_FORCE_POSITION_MIN=str(floor),RIGHT_GRIPPER_CLOSE_LIMIT=str(floor),
        CONTACT_GRAPH_TIME_PERIOD=time.strftime('%Y%m%d'),CONTACT_GRAPH_SENSOR_VERSION=str(execution.get('sensor_version','current_recorded_session')))
    return ['bash',execution['entry'],'skill-run'],env


def technical_error(message):
    text = str(message).strip()
    lower = text.lower()
    sensor_faults = ('raw finger wrench time skew', 'x5 startup missing/stale streams',
                     'guarded software zero lacks', '扣零需要已全开', '扣零期间夹爪开度改变')
    camera_timeout = 'camera' in lower and 'timed out waiting for a new frame' in lower
    python_error = re.match(r'^(ImportError|ModuleNotFoundError|TypeError|KeyError|AttributeError|NameError):', text)
    if camera_timeout or python_error or any(s in lower for s in sensor_faults):
        return text[:600]
    return None


def run_with_diagnostics(command, env):
    # Preserve interactive stdin/stdout, forward stderr and retain the fault
    # category in the terminal process receipt for offline recovery.
    try:
        child = subprocess.Popen(command, env=env, stderr=subprocess.PIPE,
                                 text=True, errors='replace')
    except OSError as exc:
        return 127, str(exc)
    reason = None
    with child:
        for line in child.stderr:
            sys.stderr.write(line)
            sys.stderr.flush()
            reason = technical_error(line) or reason
        return child.wait(), reason


def execute_trial(request,directory,spec,cfg,*,real=False,interactive=True):
    if not real:raise ValueError('Real collection requires --real')
    directory=Path(directory)
    command,env=launch_request(request,directory,spec,cfg)
    if not interactive:raise ValueError('This instance needs on-site reset; use interactive execution')
    print('\n[graph-loop] '+request['trial_id']+' '+request['role']+' '+json.dumps({'settings':request['settings'],'block':request['block'],'actions':request['actions']},ensure_ascii=False),flush=True)
    if request['post_condition'] in MEASURED_GOALS:
        print('本次 Ready 目标：'+next(c['description'] for c in spec['post_conditions'] if c['id']==request['post_condition']), flush=True)
    mass = request['block'].get('mass_class')
    bottle = {'light': '轻瓶', 'heavy': '重瓶'}.get(mass, '本次指定的瓶子')
    print(f"本次请放【{bottle}】，a 偏移 {request['settings'].get('a_offset_mm', 0):+g} mm。", flush=True)
    print('将本次瓶子放稳在原标记；机械臂停止、右夹爪全开空载，恢复到本配置 T0 或允许的退让位。停止遥操。',flush=True)
    if input('确认后输入 MOVE 执行这一次；其他输入保留待执行记录：').strip()!='MOVE':
        raise RetryTrial('未输入 MOVE，本次未启动。')
    env['CONTACT_TABLETOP_CONFIRMED']='1'
    atomic(directory/'launch.json',{'command':command,'started_ns':time.time_ns(),'request':request,'model_id':request['model_id']})
    started=time.monotonic()
    # Existing V4 entry forwards the CONTACT_* / RIGHT_* namespace into Docker.
    rc, reason = run_with_diagnostics(command, env)
    atomic(directory/'process.json',{'returncode':rc,'technical_error':reason,'wall_seconds':time.monotonic()-started,'finished_ns':time.time_ns()})
    return recover_trial(request,directory,spec,cfg,interactive=interactive)


def recover_trial(request,directory,spec,cfg,*,interactive=True):
    directory=Path(directory);process=directory/'process.json'
    events_paths=sorted(directory.glob('episodes/episode_*/events.jsonl'))
    proc = json.loads(process.read_text()) if process.exists() else None
    if not events_paths:
        if not (directory/'launch.json').exists() and all(p.name in ('request.json', 'scheduler_config.json') for p in directory.iterdir()):
            raise RetryTrial('尚未确认启动。')
        episodes = directory/'episodes'
        episode_started = episodes.exists() and any(episodes.iterdir())
        if proc and proc['returncode'] != 0 and (directory/'launch.json').exists() and not episode_started:
            raise RetryTrial(proc.get('technical_error') or f"子进程在建立 episode 前退出（返回码 {proc['returncode']}）。")
        return None
    if len(events_paths)!=1:raise ValueError('Expected one episode per campaign trial; review duplicate launches')
    events_path=events_paths[0]
    events=[json.loads(line) for line in events_path.read_text().splitlines() if line.strip()]
    if proc is None:
        return None
    selected=[e for e in events if e.get('event')=='graph_runtime_selected']
    outcomes=[e['consumer_outcome'] for e in events if e.get('event')=='state_graph_consumer_result']
    outcome=outcomes[-1] if outcomes else None
    if selected and (selected[-1]['request']['trial_id'] != request['trial_id'] or
                     selected[-1]['model_id'] != (request['model_id'] or 'planned_exploration')):
        raise ValueError('Episode did not load the requested graph trial and model')
    active=request['post_condition']
    if active in MEASURED_GOALS:
        from bottle_scan.state_graph.skill_interface import post_spec, load_callable
        condition=post_spec(spec,active)
        terminal_observed=any(load_callable(condition['evaluator'])({}, {'is_terminal':True,'consumer_outcome':o}, condition)['evaluated'] for o in outcomes)
    else:
        terminal_observed = any(o.get('L1') is False or type(o.get('L2')) is bool for o in outcomes)
    faults = [str(e.get('reason') or '') for e in events if e.get('event') in ('fault', 'episode_end')]
    reason = next((r for f in faults if (r := technical_error(f))), None) or proc.get('technical_error')
    if reason and not terminal_observed and (proc['returncode'] != 0 or any(technical_error(f) for f in faults)):
        raise RetryTrial(reason)
    if not selected:
        raise ValueError('Episode did not load the requested graph adapter')
    if active not in MEASURED_GOALS and outcome and outcome.get('L1') is True and type(outcome.get('L2')) is not bool:
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
            goal=next(c['description'] for c in spec['post_conditions'] if c['id']==active)
            print(f'首次接触能否直接达到本轮目标（{goal}）？依据配对无修正试验或独立复核。')
            label=input('r=已验证无需调整即可达到目标 / a=已验证需位置恢复 / u=未知：').strip().lower()
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
    measured = [r['u_executed'] for r in rows if (r.get('u_receipt') or {}).get('applied')]
    receipts = [r['u_receipt'] for r in rows if (r.get('u_receipt') or {}).get('applied')]
    attempt_reason = None
    for event in events:
        if event.get('event') not in ('contact_completed','contact_incomplete','contact_failed','position_action_completed'):
            continue
        attempt_reason = event.get('reason')
        # Motion can finish even when the next probe yields no observation.
        for correction in event.get('corrections', []):
            receipt = correction.get('receipt') or {}
            if receipt.get('applied') is True and receipt not in receipts:
                receipts.append(receipt)
                measured.append([receipt['action_mm']])
    if active in MEASURED_GOALS:
        passed = terminal is not None and terminal['post'][active]['success'] is True
    else:
        passed = terminal is not None and proc['returncode'] == 0
    return {'trial_id':request['trial_id'],'status':'complete' if passed else 'failed','nodes':rows,'assessment':assessment,
            'cost':{'actions':len(measured),'wall_seconds':proc['wall_seconds']},'evidence':[str(events_path)],
            'measured_actions':measured,'attempt_reason':attempt_reason,
            'unassessed_reason':None if terminal else 'No actual consumer endpoint; hardware/policy failures are not invented consumer labels.'}


def prepare(spec,cfg):
    import hashlib
    spec=deepcopy(spec)
    if cfg.get('ready_mode'):
        floor = None
        if cfg['ready_mode'] == 'fixed5':
            scheduler = json.loads(Path(cfg['execution']['scheduler_config']).read_text())
            floor = device_floor(scheduler['execution_config'])
            cfg['execution']['fixed_settings']['closure_floor'] = floor
        select(spec,cfg['ready_mode'],device_min=floor)
        spec['budget']['max_new_trials']-=cfg.get('seed_trial_count',0)
    root=Path(__file__).resolve().parents[2]
    visual=Path(spec['data']['visual_model_directory'])
    spec['data']['visual_model_directory']=str(visual if visual.is_absolute() else root/visual)
    for key in ('artifact','cache_dir'):
        if key in spec.get('tactile_encoder',{}):
            value=Path(spec['tactile_encoder'][key]);spec['tactile_encoder'][key]=str(value if value.is_absolute() else root/value)
    for condition in spec['post_conditions']:
        if condition['consumer']['controller']=='existing_forcepolicy':
            path=cfg['execution']['environment']['RIGHT_FORCE_POLICY_PATH']
            condition['consumer']['version']=hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return spec
