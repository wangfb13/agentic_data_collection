"""Durable, role-separated collection rounds. Called under the campaign lock."""
from collections import Counter
from copy import deepcopy
from pathlib import Path
import json

from . import campaign as c
from .acceptance import evaluate_B
from .common import digest
from .director_bridge import context, decide
from .facts import fact, observed_positions
from .operators import update_rule
from .round_data import contributions, episodes, training_rows
from .round_report import report, summary_markdown, paired_report
from .skill_interface import load_callable


def validation_plan(cfg, spec, role):
    plan=deepcopy(cfg.get(role+'_plan'))
    n=spec['acceptance'][role+'_B']['min_trials']
    if not plan:
        plan={'cluster_classes':['Ready','NeedAction'],'cells':[{'settings':{},'reps':n,'actions':[]}]}
    if len(c.expand_plan(plan,spec))<n or any(cell.get('actions') for cell in plan['cells']):
        raise ValueError('Validation requires a prespecified, sufficiently sized, action-free starting distribution')
    return plan


def initialize_rounds(state, spec, cfg):
    independent=validation_plan(cfg,spec,'independent')
    reserve=len(c.expand_plan(independent,spec))
    if spec['acceptance'].get('lower_bound_gates'):
        from .rate_gates import check_plan
        check_plan(independent,reserve,spec['acceptance'])
    if reserve>min(spec['budget']['max_new_trials'],spec['budget'].get('max_dispatch_attempts',spec['budget']['max_new_trials'])):raise ValueError('Independent reservation exceeds the campaign budget')
    rule=state['rule']; variables=rule['variables']
    axes={k:kdef for k,kdef in variables.items() if 'range' in kdef}
    policy=spec['loop']
    rule.update(rule_schema='state_graph_rule_v5',target_consumer=spec['active_post_condition'],
                allocation=deepcopy(policy['default_exploration_allocation']),start_points=[],
                directional_ranges={k:{'negative':None if a['range'][0]>0 else [a['range'][0],min(0,a['range'][1])],
                                       'positive':None if a['range'][1]<0 else [max(0,a['range'][0]),a['range'][1]]} for k,a in axes.items()})
    declared=fact('initial_rule_declared',[rule['rule_id']],[digest({'spec':spec,'rule':rule})])
    rule['field_provenance']={k:{'operator':'initialize','fact_ids':[declared['id']]} for k in tuple(rule)}
    state.update(schema='symbolic_campaign_v5',round_index=1,round_stage='plan',
                 dispatch_attempts=0,independent_reserve=reserve,independent_plan=independent,
                 independent_plan_digest=digest(independent),seed_rule_fact=declared,phase='plan')


def fit(out, state, spec, name, rows):
    if state['rule'].get('representation'):
        from .representation_apply import apply_representation
        rows,spec=apply_representation(rows,spec,state['rule']['representation'])
    dest=out/'revisions'/name
    checkpoint=dest/'fit_receipt.json'
    signature=digest({'spec':spec,'nodes':rows})
    if checkpoint.exists():
        saved=c.read(checkpoint)
        if saved['inputs']!=signature:raise ValueError('Committed training inputs changed')
        return saved
    previous=state.get('model')
    result=c.build(rows,spec,dest,validate=False,previous_clusters=c.read(previous['clusters']) if previous else None)
    model=dict(bundle=str(dest/'models/bundle.joblib'),model_id=result['bundle']['model_id'],
               clusters=str(dest/'clusters.json'),facts_path=str(dest/'facts.json'),
               graph_path=str(dest/'graph.json'),graph_version=result['graph']['graph_version'],
               trained=result['training']['active_model_trained'],target_consumer=spec['active_post_condition'],
               fit_contributions=contributions(rows),effective_spec=str(dest/'spec.json'),effective_spec_digest=digest(spec),
               base_spec_digest=digest(c.read(out/'spec.json')),inputs=signature)
    c.atomic(checkpoint,model)
    return model


def augmented_facts(out, state, spec, model, rows, episode_list):
    from .v5_facts import episode_frontiers
    facts=c.read(model['facts_path']);graph=c.read(model['graph_path'])
    if spec.get('campaign',{}).get('integrated_frontier') or spec.get('facts',{}).get('capability_predicates'):
        for pc in spec['post_conditions']:
            facts.extend(episode_frontiers(episode_list,spec,pc['id'],rows=rows,model_id=model['model_id']))
    enrich=spec['adapters'].get('exploration_context')
    if enrich:facts,_=load_callable(enrich)(rows,graph,facts,spec)
    return [*{f['id']:f for f in [state['seed_rule_fact'],*facts]}.values()]


def choose(out,state,spec,cfg,model,rows,facts,left,allow_agent,provider,phase):
    from .director_bridge import finalize_context
    ctx=context(spec,state['rule'],facts,history=state['history'],hypotheses=state['hypotheses'],
                state={'phase':phase,'frozen_independent':False,'trained':model['trained']},
                observed_positions=observed_positions(rows,spec),graph_version=model['graph_version'],budget_left=left)
    ctx.update(execution=deepcopy(cfg.get('execution',{})),training={'active_model_trained':model['trained']},
               spent=state['spent'],prediction_results=state.get('prediction_results',[]))
    ctx['execution'].pop('sampling_preference',None)
    ctx['frontier_context']=[f for f in facts if f['pred'] in ('direct_boundary','policy_frontier','expert_frontier','frontier_gap')]
    finalize_context(ctx)
    c.atomic(out/'director_context.json',ctx)
    answer=decide(ctx,out/'director',provider=provider,allow_agent=allow_agent,provider_settings=cfg.get('provider_settings'))
    if answer['status']!='accepted':
        state['phase']='awaiting_director' if answer['status']=='not_requested' else 'director_error'
        state['director_status']=answer['status'];c.atomic(out/'state.json',state)
        return None
    d=answer['answer']
    role=d['args'].get('role',{'request_expert_recovery':'expert_recovery','collect_outcomes':'direct_baseline'}.get(d['operator'],'exploration'))
    if not spec.get('campaign',{}).get('integrated_frontier') and role=='direct_baseline':role='exploration'
    if d['operator'] not in c.COLLECT|{'set_allocation','select_representation'}:
        state['phase']=d['operator'];state['last_director_decision']=d;c.atomic(out/'state.json',state);return None
    rule=answer.get('next_rule') or update_rule(state['rule'],d,spec=spec)
    schedules=[]
    if d['operator']=='set_allocation':
        points=rule['start_points']
        items=c.expand_plan(d['plan'],spec) if d.get('plan') else []
        if items and len(items)!=sum(rule['allocation'].values()):
            raise ValueError('Mixed plan size differs from declared role counts')
        start=0
        for kind,n in rule['allocation'].items():
            if not n:continue
            if not points:raise ValueError('Allocation requires explicit initial conditions')
            role_name={'direct':'direct_baseline','policy':'policy_frontier','expert':'expert_recovery'}[kind]
            if items:
                cells=[{'settings':x['settings'],'blocks':[x['block']],'actions':x['actions'],'reps':1} for x in items[start:start+n]]
            else:
                cells=[{'settings':points[i%len(points)],'reps':1,'actions':[]} for i in range(n)]
            start+=n
            schedules.append({'role':role_name,'plan':{'cells':cells},'operation':d['operator']})
    elif d['operator']!='select_representation':
        schedules=[{'role':role,'plan':d['plan'],'operation':d['operator']}]
    if any(s['role']=='policy_frontier' for s in schedules) and not model['trained']:
        raise ValueError('Policy frontier requires an executable candidate')
    if sum(len(c.expand_plan(s['plan'],spec)) for s in schedules)>left:
        raise ValueError('Proposed exploration would consume reserved validation trials')
    return dict(decision=d,rule=rule,schedules=schedules,pre_facts=facts,
                hypotheses=answer.get('hypotheses',state['hypotheses']),priors=ctx.get('effective_priors'),
                development_plan=validation_plan(cfg,spec,'development'))


def dispatch(out,state,spec,cfg,batches,options):
    # queue_plan commits the requests and the cursor tag together. Recover an
    # already collected sub-batch from the trial journal, including after a crash.
    stage=state['round_stage'];index=state['round_index']
    for ordinal,batch in enumerate(batches):
        token=f'{index}/{stage}/{ordinal}'
        existing=[t for t in state['trials'] if t['request'].get('stage_token')==token]
        if existing and all(t['status'] in ('complete','budget_skipped') for t in existing) and not state['pending']:continue
        state['stage_token']=token
        state['round_operation']=batch.get('operation','validate_B' if stage in ('independent','dev_validate') else None)
        if not state['pending']:
            if existing:raise ValueError('Stage has unresolved requests without a pending batch')
            c.queue_plan(out,state,spec,batch['plan'],batch['role'])
        if not all(t['request'].get('stage_token')==token for t in state['trials']
                   if t['trial_id'] in state['pending']['trial_ids']):raise ValueError('Pending batch belongs to another stage')
        c.collect_pending(out,state,spec,cfg,**options)
        if state['pending']:return False
    return True


def remaining_exploration(state,spec):
    budget=spec['budget']
    logical=budget['max_new_trials']-state['spent']['trials']
    physical=budget.get('max_dispatch_attempts',budget['max_new_trials'])-state.get('dispatch_attempts',state['spent']['trials'])
    return min(logical,physical)-state['independent_reserve']


def stop_reason(state,spec,facts,current,passed,next_cost):
    if passed:return 'development_passed'
    axes=[v for v,x in state['rule']['variables'].items() if x['active'] and 'range' in x]
    ends={(v,side) for v in axes for side,k in (('positive',1),('negative',0)) if spec['perturbation_variables'][v]['envelope'][k]*(1 if k else -1)>0}
    def frontier_cells(pred):
        return {(f['args'][0],f['args'][1],digest(f['metrics'].get('condition'))):f for f in facts if f['pred']==pred and spec['active_post_condition'] in f['args']}
    policy=frontier_cells('policy_frontier'); expert=frontier_cells('expert_frontier')
    def matches(cell):
        if cell not in policy or cell not in expert:return False
        a,b=policy[cell]['metrics'],expert[cell]['metrics']
        if not a.get('directional_evaluated_trials') or not b.get('directional_evaluated_trials'):return False
        if spec.get('facts',{}).get('rate_based_frontier'):
            from .frontier_rates import overlap
            x,y=a.get('verified_edge_mm'),b.get('verified_edge_mm')
            if x is None or y is None or not overlap(a,b):return False
        else:x,y=a.get('farthest_observed_success'),b.get('farthest_observed_success')
        resolution=b.get('resolution_mm',spec['facts'].get('boundary_resolution_mm'))
        return all(isinstance(i,(float,int)) for i in (x,y,resolution)) and abs(x-y)<=resolution
    conditions={k[2] for k in set(policy)|set(expert)}
    if ends and conditions and all(matches((*end,cid)) for end in ends for cid in conditions):return 'frontiers_converged'
    failures=[e for e in current if e['status']=='failed' or (e['target_consumer'] in e['terminal_outcomes'] and not e['terminal_outcomes'][e['target_consumer']]['success'])]
    if spec.get('facts',{}).get('rate_based_frontier'):
        nearby={}
        for cell,f in policy.items():
            if cell[:2] in ends:nearby.update(f['metrics'].get('adjacent_failures',{}))
        cfg=spec['facts']
        if len(nearby)>=cfg['frontier_min_trials'] and sum(c in ('contact_lost','edge_contact','contact_edge') for c in nearby.values())/len(nearby)>=cfg['physical_failure_fraction']:
            return 'remaining_observed_failures_physical'
    elif failures and all(e['failure_cause'] in ('contact_lost','edge_contact') for e in failures):return 'remaining_observed_failures_physical'
    if remaining_exploration(state,spec)<next_cost:return 'independent_reserve'
    return None


def summarize(out,directory,state,spec,cfg,data,allow_agent,provider):
    claims_file=directory/'summary_claims.json';c.atomic(directory/'round_report.json',data)
    if not spec['loop']['structured_summary']:return
    if claims_file.exists():claims=c.read(claims_file)
    else:
        backend=provider;own=False
        if backend is None and allow_agent and spec.get('director',{}).get('mode')=='llm':
            from ..agentic_provider import AgentProvider
            backend=AgentProvider(cfg.get('provider_settings') or {});own=True
        try:
            if hasattr(backend,'symbolic_round_summary'):
                claims=backend.symbolic_round_summary(data)['claims']
            else:
                n=sum(sum(v['attempts'] for v in bins.values()) for bins in data['development'].values())
                claims=[{'text':f'各下游合计记录 {n} 次开发验证评估机会。','paths':['/development']},
                        {'text':f"本轮目标下游为 {data['target_consumer']}。",'paths':['/target_consumer']}]
        finally:
            if own:backend.close()
    markdown=summary_markdown(claims,data)
    c.atomic(claims_file,claims)
    tmp=directory/'round_summary.md.tmp';tmp.write_text(markdown);tmp.replace(directory/'round_summary.md')



def run_locked(out,state,spec,cfg,*,real,executor,provider,allow_agent,interactive,on_interruption,max_decisions):
    options=dict(real=real,executor=executor,interactive=interactive,on_interruption=on_interruption)
    if digest(state['independent_plan'])!=state['independent_plan_digest']:
        raise ValueError('Reserved independent distribution changed')
    for _ in range(max_decisions):
        if state['complete'] or state['phase'] in ('independent_failed','inconclusive'):break
        index=state['round_index'];stage=state['round_stage']
        directory=out/'rounds'/f'round_{index:04d}'
        spec=deepcopy(spec)
        if spec.get('consumer',{}).get('target_from_rule'):
            spec['active_post_condition']=state['rule']['target_consumer']
        folder=directory/(stage+'.json')
        records=(c.read(out/'seed_episodes.json') if (out/'seed_episodes.json').exists() else [])+list(episodes(state,spec))
        rows=training_rows(c.all_rows(out,state),spec,index)
        devplan=validation_plan(cfg,spec,'development')
        dev_count=len(c.expand_plan(devplan,spec))
        available=remaining_exploration(state,spec)
        if stage=='plan':
            if folder.exists():
                planned=c.read(folder)
            elif state.get('next_plan'):
                planned=state['next_plan']
            else:
                if available < dev_count+1:
                    state.update(phase='inconclusive',stop_reason='insufficient_budget_for_first_round')
                    c.atomic(out/'state.json',state);break
                model=state.get('model') or fit(out,state,spec,'bootstrap',rows)
                state['model']=model
                facts=augmented_facts(out,state,spec,model,rows,records)
                planned=choose(out,state,spec,cfg,model,rows,facts,available-dev_count,allow_agent,provider,'plan')
                if planned is None:break
            planned['dev_plan_digest']=digest(planned['development_plan'])
            c.atomic(folder,planned)
            state['rule']=planned['rule'];state['next_plan']=None
            state['history'].append({'decision':planned['decision'],'model_id':state['model']['model_id'],'priors':planned['priors'],'round_index':index})
            state['hypotheses']=planned['hypotheses']
            state['expert_source']=planned['decision']['args'].get('expert_source','scripted_coordinate')
            state['round_stage']='execute'
        elif stage=='execute':
            planned=c.read(directory/'plan.json')
            if not dispatch(out,state,spec,cfg,planned['schedules'],options):break
            done=[t['trial_id'] for t in state['trials'] if t['request'].get('round_index')==index]
            c.atomic(folder,{'trial_ids':done,'spent':state['spent']})
            state['round_stage']='train'
        elif stage=='train':
            model=fit(out,state,spec,f'round_{index:04d}',rows)
            state['model']=model;state['revision']=index
            c.atomic(folder,model)
            state['round_stage']='dev_validate'
        elif stage=='dev_validate':
            model=c.read(directory/'train.json');planned=c.read(directory/'plan.json')
            if model['model_id']!=state['model']['model_id'] or digest(planned['development_plan'])!=planned['dev_plan_digest']:
                raise ValueError('Frozen development candidate or planned distribution changed')
            if model['trained'] and not dispatch(out,state,spec,cfg,[dict(role='development',plan=planned['development_plan'])],options):break
            evaluated=[e['assessment'] for e in episodes(state,spec) if e['round_index']==index and e['data_role']=='dev_validation' and e['assessment']]
            acceptance=evaluate_B(evaluated,spec,'development',model_id=model['model_id'])
            c.atomic(folder,dict(model_id=model['model_id'],plan_digest=planned['dev_plan_digest'],
                                 attempted=bool(model['trained']),acceptance=acceptance))
            state['round_stage']='summarize'
        elif stage=='summarize':
            from .v5_director import adjudicate_predictions
            model=state['model'];planned=c.read(directory/'plan.json')
            facts=augmented_facts(out,state,spec,model,rows,records)+c.read(directory/'dev_validate.json')['acceptance']['facts']
            decisions=adjudicate_predictions(planned['decision'].get('predictions',[]),facts,planned['pre_facts'])
            state['prediction_results']=decisions
            c.atomic(directory/'hypotheses.json',decisions)
            archive={f'round_{k:04d}':c.read(out/'rounds'/f'round_{k:04d}'/'hypotheses.json') for k in range(1,index+1)}
            c.atomic(out/'hypotheses.json',archive)
            previous=c.read(out/'rounds'/f'round_{index-1:04d}'/'round_report.json') if index>1 else None
            data=report(records,c.read(model['graph_path']),facts,decisions,spec,index,previous)
            acceptance=c.read(directory/'dev_validate.json')['acceptance']
            if spec['acceptance'].get('lower_bound_gates'):
                data['development_point_pass_only']=acceptance['metrics'].get('development_point_pass_only',False)
                data['primary_intervals']=acceptance['metrics'].get('primary_intervals',{})
            if 'calibration_preflight' in state:data['calibration_preflight']=state['calibration_preflight']
            data['dispatch_attempts_cumulative']=state.get('dispatch_attempts')
            data['budget_skipped']=[t['trial_id'] for t in state['trials'] if t['request'].get('round_index')==index and t['status']=='budget_skipped']
            summarize(out,directory,state,spec,cfg,data,allow_agent,provider)
            c.atomic(folder,{'facts':facts,'report':'round_report.json','model':model['model_id']})
            state['round_stage']='update_rule'
        elif stage=='update_rule':
            model=state['model'];facts=c.read(directory/'summarize.json')['facts']
            accepted=c.read(directory/'dev_validate.json')['acceptance']
            current=[e for e in records if e['round_index']==index]
            if folder.exists():
                update=c.read(folder)
            else:
                why=stop_reason(state,spec,facts,current,accepted['passed'],dev_count+1)
                if not spec['loop'].get('termination_rules') and why not in ('independent_reserve','development_passed'):why=None
                update={'parent_rule_id':state['rule']['rule_id'],'stop_reason':why,'next_plan':None}
                if why is None:
                    next_plan=choose(out,state,spec,cfg,model,rows,facts,available-dev_count,allow_agent,provider,'plan')
                    if next_plan is None:break
                    if next_plan['rule']['parent_rule_id']!=state['rule']['rule_id']:raise ValueError('Rule ancestry mismatch')
                    update['next_plan']=next_plan
                c.atomic(folder,update)
            if update['stop_reason']:
                state['stop_reason']=update['stop_reason']
                if not model['trained']:
                    state['phase']='inconclusive'
                    c.atomic(out/'final_report.json',dict(status='inconclusive',model_id=model['model_id'],
                             stop_reason=update['stop_reason'],independent_run=False,spent=state['spent'],
                             dispatch_attempts=state.get('dispatch_attempts'),latest_development=c.read(directory/'round_report.json'),
                             rounds=[f'rounds/round_{k:04d}/round_report.json' for k in range(1,index+1)]))
                    c.atomic(out/'state.json',state);break
                state['frozen']={**model,'independent_plan':deepcopy(state['independent_plan']),
                                 'independent_plan_digest':state['independent_plan_digest'],'stop_reason':state['stop_reason']}
                c.atomic(out/'frozen.json',state['frozen']);state['round_stage']='independent'
            else:
                state['next_plan']=update['next_plan'];state['round_index']+=1;state['round_stage']='plan'
        elif stage=='independent':
            model=state['frozen']
            if model['model_id']!=state['model']['model_id'] or digest(model['independent_plan'])!=state['independent_plan_digest']:
                raise ValueError('Independent freeze mismatch')
            if not dispatch(out,state,spec,cfg,[dict(role='independent',plan=model['independent_plan'])],options):break
            result=evaluate_B(c.assessments(state,'independent',model['model_id']),spec,'independent',model_id=model['model_id'])
            sealed=[e for e in episodes(state,spec,include_independent=True) if e['data_role']=='independent_validation']
            final=dict(model_id=model['model_id'],stop_reason=state['stop_reason'],acceptance=result,dispatch_attempts=state.get('dispatch_attempts'),
                       independent_paired=paired_report(sealed,c.read(model['graph_path']),spec),spent=state['spent'],
                       rounds=[f'rounds/round_{k:04d}/round_report.json' for k in range(1,index+1)])
            c.atomic(out/'independent_acceptance.json',result);c.atomic(out/'final_report.json',final)
            state.update(complete=result['passed'],phase='complete' if result['passed'] else 'independent_failed')
            if result['passed']:c.atomic(out/'certificate.json',final)
            c.atomic(out/'state.json',state);break
        else:raise ValueError(f'Unknown round stage {stage}')
        state['phase']=state['round_stage'];c.atomic(out/'state.json',state)
    return state
