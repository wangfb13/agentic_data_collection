"""Persistent graph/director/collection/consumer/B loop. Hardware is opt-in."""
import argparse
from contextlib import contextmanager
from copy import deepcopy
import fcntl
import json
from pathlib import Path
import time
import joblib
from .common import digest, clean, independent
from .skill_interface import load_spec, load_callable, task_id, normalize_action
from .extract_nodes import extract_nodes
from .cli import build, initial_rule
from .director_bridge import context, decide, update_hypotheses
from .operators import validate_decision
from .acceptance import evaluate_B

COLLECT={'collect_seed','collect_outcomes','collect_transitions','expand_range','densify','activate_variable',
         'probe_hidden_variable','block_by_property','repeat_state','verify_edges','validate_B'}


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


def initialize(config,out):
    out=Path(out).resolve();cfg=read(config);spec=load_spec(cfg['spec'])
    if (out/'state.json').exists():
        if read(out/'config.json')!=cfg:raise ValueError('Existing campaign configuration differs; resume it or use a new output')
        return read(out/'state.json')
    prepare=spec['adapters'].get('campaign_prepare')
    if prepare:spec=load_callable(prepare)(spec,cfg)
    rows=[]
    if cfg.get('sources'):rows,_=extract_nodes(cfg['sources'],spec,out/'initial_extraction')
    rows=[r for r in rows if r['task_id']==task_id(spec) and not independent(r)]
    atomic(out/'config.json',cfg);atomic(out/'spec.json',spec);atomic(out/'seed_nodes.json',rows)
    state={'schema':'symbolic_campaign_v4','phase':'analyze','revision':0,'rule':initial_rule(rows,spec),
           'hypotheses':[],'history':[],'trials':[],'pending':None,'model':None,'frozen':None,
           'spent':{'trials':0,'actions':0,'failures':0,'seconds':0.},'created_ns':time.time_ns(),
           'complete':False,'robot_started':False}
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
                               'source_node_id':cell.get('source_node_id'),'repeat':repeat})
    return trials


def queue_plan(out,state,spec,plan,role):
    items=expand_plan(plan,spec)
    if state['spent']['trials']+len(items)>spec['budget']['max_new_trials']:raise ValueError('Trial budget exhausted')
    model=state['frozen'] if role=='independent' else state.get('model')
    if role in ('development','independent') and (not model or not model.get('trained')):raise ValueError('B needs a trained candidate')
    batch_id='batch_'+str(len(state['history']))+'_'+digest([plan,role,state['revision']])[:8]
    queued=[]
    for i,item in enumerate(items):
        trial_id=batch_id+'_'+str(i+1).zfill(4);directory=str((Path(out)/'trials'/trial_id).resolve())
        request={'schema':'graph_trial_request_v4','trial_id':trial_id,'campaign':str(Path(out).resolve()),'role':role,
                 'post_condition':spec['active_post_condition'],'spec':str((Path(out)/'spec.json').resolve()),
                 'rule_id':state['rule']['rule_id'],'model_bundle':model['bundle'] if model else None,
                 'model_id':model['model_id'] if model else None,'frozen_model_id':model['model_id'] if role=='independent' else None,
                 'post_freeze':role=='independent','mode':'learned' if role in ('development','independent') else 'planned',**item}
        record={'trial_id':trial_id,'directory':directory,'status':'queued','request':request}
        atomic(Path(directory)/'request.json',request);state['trials'].append(record);queued.append(trial_id)
    state['pending']={'batch_id':batch_id,'trial_ids':queued,'role':role,'plan':plan}
    state['phase']='collect';atomic(Path(out)/'state.json',state)


def check_result(result,request,spec):
    if result.get('trial_id')!=request['trial_id']:raise ValueError('Result trial identity mismatch')
    if result.get('status') not in ('complete','failed'):raise ValueError('Result must be terminal, with unevaluated outcomes explicit')
    for row in result.get('nodes',[]):
        if row['task_id']!=task_id(spec):raise ValueError('Consumer evidence belongs to another task')
        row['rule_id']=request['rule_id']
        row['frozen_independent']=request['post_freeze'];row['evaluation_only']=request['post_freeze']
        row['evaluation_scope']='post_freeze_independent' if request['post_freeze'] else 'development'
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


def requeue_attempt(out, state, trial, reason):
    directory = Path(trial['directory'])
    archive = retry_archive(out, trial)
    # Keeping the receipt with the original directory makes an interrupted move
    # recoverable even if the process dies before the state file is committed.
    if not (archive/'retry.json').exists():
        archive.parent.mkdir(parents=True, exist_ok=True)
        atomic(directory/'retry.json', {'trial_id': trial['trial_id'], 'reason': reason})
        directory.rename(archive)
    receipt = read(archive/'retry.json')
    if receipt['trial_id'] != trial['trial_id']:
        raise ValueError('Retry receipt belongs to another trial')
    atomic(directory/'request.json', trial['request'])
    trial.setdefault('incomplete_attempts', []).append({**receipt, 'archive': str(archive)})
    trial['status'] = 'queued'
    trial.pop('started_ns')
    trial.pop('error', None)
    state['spent']['trials'] -= 1
    state['phase'] = 'ready_to_collect'
    atomic(Path(out)/'state.json', state)
    print(f"[graph-loop] {trial['trial_id']} 本次未计数，保留同一试次。{receipt['reason']}", flush=True)


def collect_pending(out,state,spec,cfg,*,real=False,executor=None,interactive=True):
    pending=state['pending'];ids=set(pending['trial_ids'])
    fn=executor
    for trial in state['trials']:
        if trial['trial_id'] not in ids or trial['status']=='complete':continue
        directory=Path(trial['directory']);result_path=directory/'result.json'
        if result_path.exists():accept_result(out,state,trial,spec);continue
        if trial['status']=='started':
            archive = retry_archive(out, trial)
            receipt = next((p for p in (archive/'retry.json', directory/'retry.json') if p.exists()), None)
            if receipt:
                requeue_attempt(out, state, trial, read(receipt)['reason'])
            else:
                recover_name=spec['adapters'].get('campaign_recover')
                try:
                    result=load_callable(recover_name)(trial['request'],directory,spec,cfg,interactive=interactive) if recover_name else None
                except RetryTrial as exc:
                    requeue_attempt(out, state, trial, str(exc))
                else:
                    if result:
                        atomic(result_path,result);accept_result(out,state,trial,spec);continue
                    state['phase']='awaiting_result';atomic(Path(out)/'state.json',state);return
        if not real and executor is None:
            state['phase']='ready_to_collect';atomic(Path(out)/'state.json',state);return
        trial['status']='started';trial['started_ns']=time.time_ns();state['spent']['trials']+=1
        state['robot_started']=bool(state['robot_started'] or real);atomic(Path(out)/'state.json',state)
        try:
            if fn is None:fn=load_callable(spec['adapters']['campaign_executor'])
            result=fn(trial['request'],directory,spec,cfg,real=real,interactive=interactive)
        except RetryTrial as exc:
            requeue_attempt(out, state, trial, str(exc))
            return
        except BaseException as exc:
            trial['error']=str(exc);state['phase']='awaiting_result';atomic(Path(out)/'state.json',state);raise
        if result is None:
            state['phase']='awaiting_result';atomic(Path(out)/'state.json',state);return
        atomic(result_path,result);accept_result(out,state,trial,spec)
    state['last_batch']=state['pending'];state['pending']=None;state['phase']='analyze';atomic(Path(out)/'state.json',state)


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
        if prior:
            result=evaluate_B(assessments(state,'development',prior['model_id']),spec,'development',model_id=prior['model_id'])
            # A passed development candidate is frozen exactly as tested, before
            # incorporating its B outcomes into any new fitted model.
            if result['passed']:
                state['candidate_to_freeze']=deepcopy(prior);facts+=result['facts']
            atomic(dest/'development_acceptance.json',result)
        state['model']={'bundle':str((dest/'models/bundle.joblib').resolve()),'model_id':built['bundle']['model_id'],
                        'clusters':str((dest/'clusters.json').resolve()),'facts':facts,'trained':training['active_model_trained']}
        state['hypotheses']=update_hypotheses(state['hypotheses'],facts,graph['graph_version'])
    positions={v:sorted({r['plan_coords'][v] for r in rows if isinstance(r.get('plan_coords',{}).get(v),(int,float))})
               for v,d in spec['perturbation_variables'].items() if d['kind']=='continuous'}
    ctx=context(spec,state['rule'],facts,hypotheses=state['hypotheses'],history=state['history'],
                state={'frozen_independent':bool(state['frozen'])},graph_version=graph['graph_version'],
                observed_positions=positions,budget_left=spec['budget']['max_new_trials']-state['spent']['trials'])
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
    ctx['training']=training;ctx['spent']=state['spent']
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
        role='independent' if state['frozen'] else 'development' if op=='validate_B' else 'exploration'
        queue_plan(out,state,spec,decision['plan'],role)
    elif op=='freeze':
        candidate=state.get('candidate_to_freeze')
        if not candidate:raise ValueError('No development-passed candidate to freeze')
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


def run(out,*,real=False,allow_agent=False,provider=None,executor=None,interactive=True,max_decisions=50):
    out=Path(out).resolve()
    with ownership(out):
        state=read(out/'state.json');spec=load_spec(out/'spec.json');cfg=read(out/'config.json')
        for _ in range(max_decisions):
            if state['complete']:break
            if state['pending']:
                collect_pending(out,state,spec,cfg,real=real,executor=executor,interactive=interactive)
                if state['pending']:break
            if state['spent']['trials']>=spec['budget']['max_new_trials'] and not state.get('frozen'):
                state['phase']='budget_exhausted';atomic(out/'state.json',state);break
            try:answer=next_decision(out,state,spec,cfg,allow_agent,provider)
            except Exception as exc:
                state['phase']='director_or_analysis_error';state['last_error']={'type':type(exc).__name__,'message':str(exc)}
                atomic(out/'state.json',state);raise
            if answer is None:break
            apply_decision(out,state,spec,answer)
            if state['phase'] not in ('collect','analyze'):break
        else:
            state['phase']='decision_budget_exhausted';atomic(out/'state.json',state)
        return state


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=['init','advance','run','status','review','resolve-interrupted'])
    parser.add_argument('--config',default='config.state_graph.campaign_v4.json');parser.add_argument('--out',required=True)
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
    else:state=run(out,real=args.real and args.command=='run',allow_agent=args.allow_agent,interactive=not args.noninteractive)
    print(json.dumps(clean({'out':str(out),'phase':state['phase'],'spent':state['spent'],'pending':state['pending'],'complete':state['complete']}),ensure_ascii=False));return 0
