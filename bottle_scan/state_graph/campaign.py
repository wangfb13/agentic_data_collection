"""Persistent graph/director/collection/consumer/B loop. Hardware is opt-in."""
import argparse
from collections import deque
from contextlib import contextmanager
from copy import deepcopy
import fcntl
import json
from pathlib import Path
import time
import traceback
import joblib
from .common import digest, clean, independent
from .skill_interface import load_spec, load_callable, task_id, normalize_action
from .extract_nodes import extract_nodes
from .cli import build, initial_rule
from .director_bridge import context, decide, update_hypotheses
from .facts import coordinate_evidence, observed_positions
from .operators import validate_decision
from .acceptance import evaluate_B

COLLECT={'collect_seed','collect_outcomes','collect_transitions','expand_range','densify','activate_variable',
         'probe_hidden_variable','block_by_property','repeat_state','verify_edges','validate_B','probe_amplitude','recalibrate_reference','probe_recovery_frontier','request_expert_recovery'}


def atomic(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(clean(value),ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def read(path):return json.loads(Path(path).read_text())


@contextmanager
def ownership(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    with (out/'campaign.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('Campaign already running') from None
        yield


def initialize(config,out,*,seed_rows=None):
    out=Path(out).resolve();cfg=read(config);spec=load_spec(cfg['spec'])
    if (out/'state.json').exists():
        if read(out/'config.json')!=cfg:raise ValueError('Existing campaign configuration differs; resume it or use a new output')
        return read(out/'state.json')
    prepare=spec['adapters'].get('campaign_prepare')
    if prepare:spec=load_callable(prepare)(spec,cfg)
    if cfg.get('graph_version') == 5:
        from .v5_profile import configure
        spec=configure(spec,cfg.get('graph_overrides'))
    rows=list(seed_rows or [])
    if seed_rows is not None and cfg.get('sources'):raise ValueError('Choose a single source of seed evidence')
    if cfg.get('sources'):rows,_=extract_nodes(cfg['sources'],spec,out/'initial_extraction')
    rows=[r for r in rows if r['task_id']==task_id(spec) and not independent(r)]
    atomic(out/'config.json',cfg);atomic(out/'spec.json',spec);atomic(out/'seed_nodes.json',rows)
    state={'schema':'symbolic_campaign_v4','phase':'analyze','revision':0,'rule':initial_rule(rows,spec),
           'hypotheses':[],'history':[],'trials':[],'pending':None,'model':None,'frozen':None,
           'spent':{'trials':0,'actions':0,'failures':0,'seconds':0.},'created_ns':time.time_ns(),
           'complete':False,'robot_started':False}
    if spec.get('loop', {}).get('rounds'):
        from .rounds import initialize_rounds
        initialize_rounds(state, spec, cfg)
    if spec.get('import',{}).get('consumer_audit') and (out/'import_receipt.json').exists():
        warnings=read(out/'import_receipt.json').get('consumer_audit',{}).get('warnings',[])
        state['import_warnings']=warnings
        if warnings:
            import warnings as notices
            notices.warn(f'{len(warnings)} unmatched consumer outcomes; see import_receipt.json',RuntimeWarning)
    atomic(out/'state.json',state);return state


def all_rows(out,state):
    rows=read(Path(out)/'seed_nodes.json')
    for trial in state['trials']:
        path=Path(trial['directory'])/'result.json'
        if path.exists() and trial.get('status')=='complete':rows.extend(read(path).get('nodes',[]))
    ids=[r['node_id'] for r in rows]
    if len(ids)!=len(set(ids)):raise ValueError('Repeated node IDs in campaign evidence')
    return rows


def assessments(state,role,model_id):
    result=[]
    for trial in state['trials']:
        if trial.get('status')!='complete' or trial['request']['role']!=role or trial['request'].get('model_id')!=model_id:continue
        a=read(Path(trial['directory'])/'result.json').get('assessment')
        if a:result.append(a)
    return result


def expand_plan(plan,spec):
    trials=[]
    for cell in plan['cells']:
        for repeat in range(cell['reps']):
            for block in cell.get('blocks',[{}]):
                settings={k:v['reference'] for k,v in spec['perturbation_variables'].items() if 'reference' in v}
                settings.update(cell.get('settings',{}))
                trials.append({'settings':settings,'block':block,'actions':[normalize_action(a,spec) for a in cell.get('actions',[])],
                               'source_node_id':cell.get('source_node_id'),'repeat':repeat,
                               **({'amplitude_target':deepcopy(cell['amplitude_target'])} if 'amplitude_target' in cell else {})})
    return trials


def source_initial_settings(plan, spec, rows):
    """A cell referencing an initial window inherits its recorded trial settings."""
    sources = {row['node_id']:row for row in rows}
    plan = deepcopy(plan)
    for cell in plan['cells']:
        ident = cell.get('source_node_id')
        if not ident:
            continue
        source = sources[ident]
        if source.get('previous_node_id'):
            continue  # The original start settings do not reproduce a later window.
        settings = cell.setdefault('settings', {})
        for key, value in source.get('repeat_settings', {}).items():
            variable = spec['perturbation_variables'].get(key)
            if not variable or variable['kind'] == 'object_property' or value is None:
                continue
            if variable['kind'] == 'continuous':
                if not isinstance(value,(int,float)) or not variable['envelope'][0] <= value <= variable['envelope'][1]:
                    continue
            elif value not in variable.get('levels', []):
                continue
            settings.setdefault(key, value)
    return plan


def queue_plan(out,state,spec,plan,role,*,source_review=None):
    if not spec.get('campaign',{}).get('integrated_frontier') and role in ('direct_baseline','policy_frontier','expert_recovery'):raise ValueError('Use the frozen frontier round, not campaign exploration')
    if any(c.get('source_node_id') for c in plan['cells']):
        plan=source_initial_settings(plan,spec,all_rows(out,state))
    items=expand_plan(plan,spec)
    v5=spec.get('graph_schema')=='state_graph_v5'
    reserve=state.get('independent_reserve',0) if role!='independent' else 0
    if state['spent']['trials']+len(items)+reserve>spec['budget']['max_new_trials']:raise ValueError('Trial budget exhausted')
    model=state['frozen'] if role=='independent' else state.get('model')
    if role in ('development','independent','policy_frontier') and (not model or not model.get('trained')):raise ValueError('B needs a trained candidate')
    signature=[plan,role,state['revision']]
    if v5:
        signature.extend([state.get('round_index'),state.get('round_stage'),len(state['trials'])])
        if spec.get('facts',{}).get('amplitude_coverage') or spec.get('director',{}).get('score_based_axis_choice'):
            # A continuation can pick the same round-one plan as its source.
            # Keep their trial/node IDs distinct, including after a restart.
            signature.append(state['created_ns'])
    batch_id='batch_'+str(len(state['history']))+'_'+digest(signature)[:8]
    queued=[]
    for i,item in enumerate(items):
        trial_id=batch_id+'_'+str(i+1).zfill(4);directory=str((Path(out)/'trials'/trial_id).resolve())
        request={'schema':'graph_trial_request_v4','trial_id':trial_id,'campaign':str(Path(out).resolve()),'role':role,
                 'post_condition':spec['active_post_condition'],'spec':str((Path(out)/'spec.json').resolve()),
                 'rule_id':state['rule']['rule_id'],'model_bundle':model['bundle'] if model else None,
                 'model_id':model['model_id'] if model else None,'frozen_model_id':model['model_id'] if role=='independent' else None,
                 'post_freeze':role=='independent','mode':'learned' if role in ('development','independent') else 'planned',**item}
        if v5:
            from .round_data import bind_request
            bind_request(request, state, spec, role)
        record={'trial_id':trial_id,'directory':directory,'status':'queued','request':request}
        atomic(Path(directory)/'request.json',request);state['trials'].append(record);queued.append(trial_id)
    state['pending']={'batch_id':batch_id,'trial_ids':queued,'role':role,'plan':plan}
    if source_review is not None:
        state['pending']['source_review']=source_review
    state['phase']='collect';atomic(Path(out)/'state.json',state)


def check_result(result,request,spec):
    if result.get('trial_id')!=request['trial_id']:raise ValueError('Result trial identity mismatch')
    if result.get('status') not in ('complete','failed'):raise ValueError('Result must be terminal, with unevaluated outcomes explicit')
    for row in result.get('nodes',[]):
        if row['task_id']!=task_id(spec):raise ValueError('Consumer evidence belongs to another task')
        row['rule_id']=request['rule_id']
        row['frozen_independent']=request['post_freeze'];row['evaluation_only']=request['post_freeze']
        row['evaluation_scope']='post_freeze_independent' if request['post_freeze'] else 'development'
    if 'data_role' in request:
        from .round_data import tag_result
        tag_result(result,request)
    a=result.get('assessment')
    if a:
        if a.get('evaluated') and (type(a.get('success')) is not bool or not a.get('evidence')):
            raise ValueError('B evaluation requires actual consumer result and evidence')
        a.update(role=request['role'],post_condition=request['post_condition'],model_id=request['model_id'],
                 frozen_model_id=request['frozen_model_id'],post_freeze=request['post_freeze'],group_id=request['trial_id'])
    return result


def accept_result(out,state,trial,spec):
    path=Path(trial['directory'])/'result.json';result=check_result(read(path),trial['request'],spec);atomic(path,result)
    if trial.get('status')=='complete':return
    trial['status']='complete';trial['outcome_status']=result['status']
    state['spent']['actions']+=int(result.get('cost',{}).get('actions',0))
    state['spent']['failures']+=int(result['status']=='failed' or (result.get('assessment') or {}).get('success') is False)
    state['spent']['seconds']+=float(result.get('cost',{}).get('wall_seconds',0))
    atomic(Path(out)/'state.json',state)


class RetryTrial(Exception):
    """The attempt ended without a usable outcome. Repeat this plan cell."""


def retry_archive(out, trial):
    return Path(out)/'incomplete_attempts'/trial['trial_id']/str(trial['started_ns'])


def requeue_attempt(out, state, trial, reason, *, defer=False):
    directory = Path(trial['directory'])
    archive = retry_archive(out, trial)
    # Keeping the receipt with the original directory makes an interrupted move
    # recoverable even if the process dies before the state file is committed.
    if not (archive/'retry.json').exists():
        archive.parent.mkdir(parents=True, exist_ok=True)
        if not (directory/'retry.json').exists():
            receipt = {'trial_id': trial['trial_id'], 'reason': reason}
            if defer:
                receipt.update(disposition='deferred', started_ns=trial['started_ns'], deferred_ns=time.time_ns())
            atomic(directory/'retry.json', receipt)
        directory.rename(archive)
    receipt = read(archive/'retry.json')
    if receipt['trial_id'] != trial['trial_id']:
        raise ValueError('Retry receipt belongs to another trial')
    atomic(directory/'request.json', trial['request'])
    trial.setdefault('incomplete_attempts', []).append({**receipt, 'archive': str(archive)})
    deferred = receipt.get('disposition') == 'deferred'
    trial['status'] = 'deferred' if deferred else 'queued'
    trial.pop('started_ns')
    trial.pop('error', None)
    state['spent']['trials'] -= 1
    state['phase'] = 'ready_to_collect'
    atomic(Path(out)/'state.json', state)
    label = '已暂缓，保留记录，继续本批下一条' if deferred else '本次未计数，保留同一试次'
    print(f"[graph-loop] {trial['trial_id']} {label}。{receipt['reason']}", flush=True)


def deferred_trials(state):
    ids = set((state.get('pending') or {}).get('trial_ids', []))
    return [t for t in state['trials'] if t['trial_id'] in ids and t['status'] == 'deferred']


def release_deferred(out, selected):
    # Restore only original requests in the current batch. No dispatch or new plan.
    out = Path(out)
    with ownership(out):
        state = read(out/'state.json')
        deferred = {t['trial_id']: t for t in deferred_trials(state)}
        ids = set(deferred) if selected == ['all'] else set(selected)
        if ids - deferred.keys():
            raise ValueError('试次不在本批暂缓清单：' + ', '.join(sorted(ids - deferred.keys())))
        if ids:
            for ident in ids:
                deferred[ident]['status'] = 'queued'
            state['phase'] = 'ready_to_collect'
            atomic(out/'state.json', state)
        return sorted(ids)


def collect_pending(out,state,spec,cfg,*,real=False,executor=None,interactive=True,on_interruption=None):
    pending=state['pending'];ids=set(pending['trial_ids'])
    fn=executor
    remaining=deque(t for t in state['trials']
                    if t['trial_id'] in ids and t['status'] not in ('complete','deferred','budget_skipped'))
    while remaining:
        if pending.get('role')=='development' and state.get('round_stage')=='dev_validate':
            from .development_gate import stop_pending
            if stop_pending(out,state,spec):break
        trial=remaining.popleft()
        directory=Path(trial['directory']);result_path=directory/'result.json'
        if result_path.exists():accept_result(out,state,trial,spec);continue
        if trial['status']=='started':
            archive = retry_archive(out, trial)
            receipt = next((p for p in (archive/'retry.json', directory/'retry.json') if p.exists()), None)
            if receipt:
                requeue_attempt(out, state, trial, read(receipt)['reason'])
                if trial['status'] == 'deferred':
                    continue

        recovering = trial['status'] == 'started'
        if not recovering:
            if not real and executor is None:
                state['phase']='ready_to_collect';atomic(out/'state.json',state);return
            if 'dispatch_attempts' in state:
                limit=spec['budget'].get('max_dispatch_attempts',spec['budget']['max_new_trials'])
                reserve=0 if trial['request']['post_freeze'] else state['independent_reserve']
                if state['dispatch_attempts']>=limit-reserve:
                    trial.update(status='budget_skipped',reason='dispatch_budget_preserves_independent_reserve')
                    atomic(out/'state.json',state)
                    continue
                state['dispatch_attempts']+=1
                trial.setdefault('dispatch_ordinals',[]).append(state['dispatch_attempts'])
            trial['status']='started';trial['started_ns']=time.time_ns();state['spent']['trials']+=1
            state['robot_started']=bool(state['robot_started'] or real);atomic(out/'state.json',state)
        choose = on_interruption if interactive else None
        try:
            if recovering:
                name=spec['adapters'].get('campaign_recover')
                result=load_callable(name)(trial['request'],directory,spec,cfg,interactive=interactive) if name else None
            else:
                if fn is None:fn=load_callable(spec['adapters']['campaign_executor'])
                result=fn(trial['request'],directory,spec,cfg,real=real,interactive=interactive)
        except RetryTrial as exc:
            reason = str(exc)
            defer = bool(choose and choose(trial, reason))
            requeue_attempt(out, state, trial, reason, defer=defer)
            if defer:
                continue
            # Legacy recovery of a classified interruption may retry in this same invocation.
            if recovering and choose is None:
                remaining.appendleft(trial)
                continue
            return
        except BaseException as exc:
            reason = str(exc) or type(exc).__name__
            trial['error']=reason;state['phase']='awaiting_result';atomic(out/'state.json',state)
            if not isinstance(exc, Exception) or not choose:
                raise
            atomic(directory/'exception.json', {
                'reason': reason, 'traceback': traceback.format_exc(),
                'child_stderr': getattr(exc, 'stderr', None) if isinstance(getattr(exc, 'stderr', None), str) else None,
            })
            if choose(trial, reason):
                requeue_attempt(out, state, trial, reason, defer=True)
                continue
            return
        if result is None:
            state['phase']='awaiting_result';atomic(out/'state.json',state)
            reason = trial.get('error') or '上次执行没有可结算结果'
            if choose and choose(trial, reason):
                requeue_attempt(out, state, trial, reason, defer=True)
                continue
            return
        atomic(result_path,result);accept_result(out,state,trial,spec)
    if pending.get('role')=='development' and state.get('round_stage')=='dev_validate' and deferred_trials(state):
        from .development_gate import stop_pending
        stop_pending(out,state,spec)
    if deferred_trials(state):
        state['phase']='deferred_review'
        atomic(out/'state.json',state)
        return
    state['last_batch']=state['pending'];state['pending']=None;state['phase']='analyze';atomic(Path(out)/'state.json',state)


def source_comparison(out, batch, trials, rows, condition):
    """Review the two arms against the graph that selected the source."""
    from .noise import pairwise_distances
    from .common import group_id
    reference = batch['source_review']
    noise = read(Path(out)/'revisions'/f"{reference['revision']:04d}"/'noise_model.json')
    by_id = {row['node_id']: row for row in rows}
    source = by_id[reference['source_node_id']]
    prior = [by_id[ident] for ident in noise['fit_node_ids']]
    prior_success = [r for r in prior if r.get('is_terminal') and
                     r['post'][condition].get('evaluated') and r['post'][condition].get('success') is True]
    arms = []
    for trial in trials:
        if trial['trial_id'] not in batch['trial_ids']:
            continue
        result = read(Path(trial['directory'])/'result.json')
        nodes = result['nodes']
        first = next((r for r in nodes if r.get('window_index') == 0 and r.get('window_valid')), None)
        terminal = next((r for r in reversed(nodes) if r.get('is_terminal') and
                         r['post'][condition].get('evaluated')), None)
        arm = {'trial_id': trial['trial_id'], 'block': trial['request']['block'],
               'settings': trial['request']['settings'], 'requested_actions': trial['request']['actions'],
               'measured_actions': result.get('measured_actions', [r.get('u_executed') for r in nodes if (r.get('u_receipt') or {}).get('applied')]),
               'attempt_reason': result.get('attempt_reason'),
               'initial_node': first['node_id'] if first else None,
               'actual_initial_coords': {k:x for k in ('a_offset_mm','l_offset_mm','r_offset_mm','h_offset_mm') if (x:=coordinate_evidence(first,k)) is not None} if first else None,
               'distance_to_source': None, 'source_matched': None, 'prior_success_groups_nearby': None,
               'consumer_success': terminal['post'][condition]['success'] if terminal else None,
               'consumer_level': terminal['post'][condition].get('level') if terminal else None}
        if first:
            distances = pairwise_distances([first, source, *prior_success], noise)[0]
            arm.update(distance_to_source=float(distances[1]), source_matched=bool(distances[1] <= 1),
                       prior_success_groups_nearby=len({group_id(r) for r,d in zip(prior_success, distances[2:]) if d<=1}))
        arms.append(arm)
    return {'batch_id': batch['batch_id'], **reference, 'post_condition': condition, 'arms': arms,
            'interpretation': 'Judge reproducibility from first contact before any correction using this pre-batch metric. '
             'A source match that also matches historical success is ambiguous for reproducing the isolated state. '
             'Compare actual movements and measured endpoints across comparable starts. Missing evidence is unknown. '
             'Use the actual number of comparable trials in each arm. Decide whether further collection is justified and explain why.'}


def next_decision(out,state,spec,cfg,allow_agent,provider=None):
    out=Path(out);rows=[r for r in all_rows(out,state) if not independent(r)]
    previous=read(state['model']['clusters']) if state.get('model') and Path(state['model']['clusters']).exists() else None
    # Independent evidence never fits noise, graph, parameters, hypotheses or rules.
    if state['frozen']:
        frozen=state['frozen'];b=joblib.load(frozen['bundle'])
        facts=deepcopy(frozen['facts']);graph=b['graph'];training={'active_model_trained':True}
        result=evaluate_B(assessments(state,'independent',frozen['model_id']),spec,'independent',model_id=frozen['model_id'])
        facts+=result['facts'];atomic(out/'independent_acceptance.json',result)
        if not result['passed'] and state.get('last_batch',{}).get('role')=='independent':
            on_gap=spec['adapters'].get('after_independent_gap')
            if on_gap and load_callable(on_gap)(out,state,spec,result):
                # The failed confirmation remains sealed. Build the next menu from development data.
                return next_decision(out,state,spec,cfg,allow_agent,provider)
            state['phase']='independent_failed';state['independent_result']=result;atomic(out/'state.json',state);return None
    elif state.get('model') and state.get('last_batch',{}).get('role')=='development' and not state.pop('retrain_requested',False):
        # Repeated B batches test the same candidate until the director chooses
        # new collection or retraining; small batches can accumulate coverage.
        candidate=state['model'];bundle=joblib.load(candidate['bundle'])
        facts=[f for f in deepcopy(candidate['facts']) if f['pred'] not in ('dev_B_pass','dev_B_gap')]
        graph=bundle['graph'];training={'active_model_trained':candidate['trained']}
        result=evaluate_B(assessments(state,'development',candidate['model_id']),spec,'development',model_id=candidate['model_id'])
        facts+=result['facts'];atomic(out/'development_acceptance.json',result)
        if result['passed']:state['candidate_to_freeze']=deepcopy(candidate)
        state['hypotheses']=update_hypotheses(state['hypotheses'],facts,graph['graph_version'])
    else:
        state['revision']+=1;dest=out/'revisions'/str(state['revision']).zfill(4)
        built=build(rows,spec,dest,validate=False,previous_clusters=previous)
        graph=built['graph'];facts=built['facts'];training=built['training'];previous=built['clusters']
        prior=state.get('model')
        if prior and prior['model_id'] not in state.get('retired_candidates',[]):
            result=evaluate_B(assessments(state,'development',prior['model_id']),spec,'development',model_id=prior['model_id'])
            # A passed development candidate is frozen exactly as tested, before
            # incorporating its B outcomes into any new fitted model.
            if result['passed']:
                state['candidate_to_freeze']=deepcopy(prior);facts+=result['facts']
            atomic(dest/'development_acceptance.json',result)
        state['model']={'bundle':str((dest/'models/bundle.joblib').resolve()),'model_id':built['bundle']['model_id'],
                        'clusters':str((dest/'clusters.json').resolve()),'facts':facts,'trained':training['active_model_trained']}
        state['hypotheses']=update_hypotheses(state['hypotheses'],facts,graph['graph_version'])
    enrich=spec['adapters'].get('exploration_context')
    if enrich:
        facts, collection_ops = load_callable(enrich)(rows,graph,facts,spec)
    positions=observed_positions(rows,spec)
    ctx=context(spec,state['rule'],facts,hypotheses=state['hypotheses'],history=state['history'],
                state={'frozen_independent':bool(state['frozen'])},graph_version=graph['graph_version'],
                observed_positions=positions,budget_left=spec['budget']['max_new_trials']-state['spent']['trials'])
    if enrich:
        menu=spec['adapters'].get('campaign_menu')
        if spec.get('director',{}).get('mode') in ('llm','fixed_priority'):
            pass  # Phase facts and the operator registry supply the menu.
        elif menu:
            load_callable(menu)(ctx, collection_ops, state, training)
        else:
            ctx['applicable_operators']=collection_ops
    # Honest executor capabilities, not hypothetical actions that only translate a plan.
    if state['frozen']:ctx['independent_plan']=state['frozen']['independent_plan']
    ctx['execution']=cfg.get('execution',{})
    supported=ctx['execution'].get('executable_variables')
    if supported is not None:
        allowed=[]
        for op in ctx['applicable_operators']:
            var=op['bindings'].get('var')
            if var is not None and var not in supported and op['op'] in COLLECT:
                allowed.append({**op,'op':'request_capability','bindings':{'var':var,'requested_operator':op['op']}})
            else:allowed.append(op)
        ctx['applicable_operators']=allowed
    ctx['collection_regions']=[{'region':c['cluster_id'],'anchor':c.get('region_anchor'),'symbols':c['symbols'],
        'examples':[{'node_id':r['node_id'],'settings':r.get('repeat_settings',{}),'blocks':{v:r['attrs'].get(v) for v,d in spec['perturbation_variables'].items() if d['kind']=='object_property'}}
        for r in rows if r['node_id'] in c['node_ids']][:2]} for c in (previous or {}).get('clusters',[])]
    last_batch = state.get('last_batch') or {}
    if last_batch.get('source_review') and not state['frozen']:
        review = source_comparison(out, last_batch, state['trials'], rows, spec['active_post_condition'])
        atomic(out/'reviews'/(last_batch['batch_id']+'.json'), review)
        ctx['source_comparison'] = clean(review)
    ctx['training']=training;ctx['spent']=state['spent']
    if spec.get('graph_schema') == 'state_graph_v5':
        from .director_bridge import finalize_context
        allowed=cfg.get('execution',{}).get('allowed_operations')
        if allowed is not None:
            choices=ctx['applicable_operators']+(collection_ops if enrich else [])
            ctx['applicable_operators']=list({digest([x['op'],x['bindings']]):x for x in choices if x['op'] in allowed}.values())
        finalize_context(ctx)
    state['context_path']=str(out/'director_context.json');atomic(state['context_path'],ctx)
    answer=decide(ctx,out/'director',allow_agent=allow_agent,provider=provider,provider_settings=cfg.get('provider_settings'))
    if answer['status']!='accepted':
        state['phase']='awaiting_director' if answer['status']=='not_requested' else 'director_error';atomic(out/'state.json',state);return None
    return answer


def apply_decision(out,state,spec,answer):
    decision=answer['answer'];op=decision['operator']
    state['history'].append({'decision':decision,'model_id':(state.get('model') or {}).get('model_id'),'at_ns':time.time_ns()})
    state['hypotheses']=answer.get('hypotheses',state['hypotheses']);state['rule']=answer.get('next_rule',state['rule'])
    if op in COLLECT:
        role='independent' if state['frozen'] else 'development' if op=='validate_B' else decision.get('args',{}).get('role','expert_recovery' if op=='request_expert_recovery' else 'exploration')
        queue_plan(out,state,spec,decision['plan'],role)
    elif op=='freeze':
        candidate=state.get('candidate_to_freeze')
        if not candidate or candidate['model_id'] in state.get('retired_candidates',[]):
            raise ValueError('No eligible development-passed candidate to freeze')
        plan=deepcopy(read(Path(out)/'config.json').get('independent_plan') or state['last_batch']['plan'])
        count=len(expand_plan(plan,spec));minimum=spec['acceptance']['independent_B'].get('min_trials',count)
        if count<minimum:
            import math
            for cell in plan['cells']:cell['reps']*=math.ceil(minimum/count)
        state['frozen']={**candidate,'frozen_at_ns':time.time_ns(),'independent_plan':plan};state['phase']='analyze'
        atomic(Path(out)/'frozen.json',state['frozen'])
    elif op=='complete':
        if not state['frozen'] or not evaluate_B(assessments(state,'independent',state['frozen']['model_id']),spec,'independent',model_id=state['frozen']['model_id'])['passed']:
            raise ValueError('Completion requires frozen independent B evidence')
        state['complete']=True;state['phase']='complete'
        atomic(Path(out)/'certificate.json',{'model':state['frozen'],'acceptance':read(Path(out)/'independent_acceptance.json'),'spent':state['spent'],
                                           'scope':'Measured skill/consumer outcome on this campaign distribution; not universal task success.'})
    elif op=='train':
        state['phase']='analyze';state['retrain_requested']=True
    else:state['phase']=op
    atomic(Path(out)/'state.json',state)


def run(out,*,real=False,allow_degraded=False,allow_agent=False,provider=None,executor=None,interactive=True,max_decisions=50,on_interruption=None):
    out=Path(out).resolve()
    with ownership(out):
        state=read(out/'state.json');spec=load_spec(out/'spec.json');cfg=read(out/'config.json')
        if spec.get('graph_schema')=='state_graph_v5':
            from .calibrations import preflight
            state['calibration_preflight']=preflight(spec,real,allow_degraded)
            atomic(out/'state.json',state)
        if spec.get('loop',{}).get('rounds'):
            from .rounds import run_locked
            return run_locked(out,state,spec,cfg,real=real,executor=executor,provider=provider,
                              allow_agent=allow_agent,interactive=interactive,on_interruption=on_interruption,
                              max_decisions=max_decisions)
        for _ in range(max_decisions):
            if state['complete']:break
            if state['pending']:
                collect_pending(out,state,spec,cfg,real=real,executor=executor,interactive=interactive,on_interruption=on_interruption)
                if state['pending']:break
            if state['spent']['trials']>=spec['budget']['max_new_trials'] and not state.get('frozen'):
                state['phase']='budget_exhausted';atomic(out/'state.json',state);break
            try:answer=next_decision(out,state,spec,cfg,allow_agent,provider)
            except Exception as exc:
                state['phase']='director_or_analysis_error';state['last_error']={'type':type(exc).__name__,'message':str(exc)}
                atomic(out/'state.json',state);raise
            if answer is None:break
            apply_decision(out,state,spec,answer)
            if (state.get('last_batch') or {}).get('source_review'):
                print('[graph-loop] Agent', answer['answer']['operator'], answer['answer']['reason'], flush=True)
                if state['pending']:
                    state['phase']='ready_to_collect';atomic(out/'state.json',state)
                    break
            if state['phase'] not in ('collect','analyze'):break
        else:
            state['phase']='decision_budget_exhausted';atomic(out/'state.json',state)
        return state


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=['init','advance','run','status','review','resolve-interrupted'])
    parser.add_argument('--config',default='config.state_graph.campaign_v4.json');parser.add_argument('--out',required=True)
    parser.add_argument('--allow-degraded',action='store_true')
    parser.add_argument('--real',action='store_true');parser.add_argument('--allow-agent',action='store_true');parser.add_argument('--trial');parser.add_argument('--evidence');parser.add_argument('--noninteractive',action='store_true')
    args=parser.parse_args(argv);out=Path(args.out).resolve()
    if args.command=='init':
        with ownership(out):state=initialize(args.config,out)
    elif args.command=='status':state=read(out/'state.json')
    elif args.command in ('review','resolve-interrupted'):
        with ownership(out):
            state=read(out/'state.json');spec=load_spec(out/'spec.json');cfg=read(out/'config.json')
            trial=next(t for t in state['trials'] if t['trial_id']==args.trial)
            if trial['status']!='started':raise ValueError('Trial is not awaiting a result')
            if args.command=='resolve-interrupted':
                if not args.evidence:raise ValueError('Provide the actual stopped/reset state and evidence with --evidence')
                result={'trial_id':trial['trial_id'],'status':'failed','nodes':[],'assessment':({'evaluated':False,'rejected':True,'exit_declared':False,'evidence':[args.evidence]} if trial['request']['role']!='exploration' else None),'cost':{'actions':0,'actions_unknown':True,'wall_seconds':(time.time_ns()-trial['started_ns'])/1e9},'evidence':args.evidence}
                atomic(Path(trial['directory'])/'result.json',result)
            else:
                try:
                    result=load_callable(spec['adapters']['campaign_recover'])(trial['request'],Path(trial['directory']),spec,cfg,interactive=True)
                except RetryTrial as exc:
                    requeue_attempt(out, state, trial, str(exc))
                    print(json.dumps({'phase': state['phase'], 'spent': state['spent']}, ensure_ascii=False))
                    return 0
                if result is None:raise ValueError('No completed evidence available; do not replay this trial')
                atomic(Path(trial['directory'])/'result.json',result)
            accept_result(out,state,trial,spec)
    else:state=run(out,allow_degraded=args.allow_degraded,real=args.real and args.command=='run',allow_agent=args.allow_agent,interactive=not args.noninteractive)
    print(json.dumps(clean({'out':str(out),'phase':state['phase'],'spent':state['spent'],'pending':state['pending'],'complete':state['complete']}),ensure_ascii=False));return 0
