"""Deterministic collection proposals, epistemic scores and checkable fact predictions."""
from copy import deepcopy
from itertools import product
import operator
import numpy as np
from .common import digest
from .v5_facts import continuous,number
from .v5_operators import NEW_COLLECTION


PREDICATES=set('''repeat_calibration_missing consumer_evidence_missing transition_evidence_missing
no_failure failure_present boundary_gap boundary_within_cluster coverage stochastic_outcome
hidden_variable_candidate unsupported unreachable inferred_edge_unverified drift
dynamics_uncertain high_disagreement model_executable coverage_ok development_class_missing
dev_B_pass dev_B_gap indep_B_pass indep_B_gap label_conflict axis_observable axis_unobservable
axis_consequential axis_inconsequential action_effect_unidentified dynamics_unavailable
direct_boundary policy_frontier expert_frontier frontier_gap frontier_direction_closed
coordinate_resolution recovery_evidence_missing representation_ranking resolution_scale_missing
posterior_variance phase_plan phase_execute phase_train phase_dev_validate phase_summarize
phase_update_rule phase_independent'''.split())
METRICS={'frontier_gap':['length_mm'],'coordinate_resolution':['resolution_mm','noise_mm'],
         'action_effect_unidentified':['relative_gain'],'axis_observable':['sensitivity'],
         'axis_unobservable':['sensitivity'],'axis_consequential':['p_adjusted'],'axis_inconsequential':['p_adjusted'],
         'boundary_gap':['gap'],'boundary_within_cluster':['p_adjusted'],'no_failure':['terminal_count']}
for _p in ('direct_boundary','policy_frontier','expert_frontier'):
    METRICS[_p]=['farthest_observed_success','trials','n_success','evaluated_trials']
COMPARE={'lt':operator.lt,'le':operator.le,'eq':operator.eq,'ge':operator.ge,'gt':operator.gt}
COLLECTION=NEW_COLLECTION|set('collect_seed collect_outcomes collect_transitions expand_range densify activate_variable probe_hidden_variable block_by_property repeat_state verify_edges validate_B'.split())


def validate_predictions(predictions,ctx):
    if not isinstance(predictions,list) or not predictions:raise ValueError('predictions must be a nonempty list')
    known=PREDICATES|{f['pred'] for f in ctx['facts']}
    for p in predictions:
        if not isinstance(p,dict) or p.get('pred') not in known or not isinstance(p.get('args'),list):
            raise ValueError('Unknown predicate or invalid prediction arguments')
        if not isinstance(p.get('scope',{}),dict) or set(p.get('scope',{}))-{'model_id','condition'}:
            raise ValueError('Predictions may only scope by condition/model_id')
        examples=[f for f in ctx['facts'] if f['pred']==p['pred']]
        if examples and len(p['args']) not in {len(f['args']) for f in examples}:
            raise ValueError('Prediction predicate argument arity differs from its schema')
        if p.get('expect') not in ('appears','disappears','metric'):raise ValueError('Unknown prediction expectation')
        if p['expect']=='metric':
            schema=set(METRICS.get(p['pred'],()))
            schema.update(k for f in ctx['facts'] if f['pred']==p['pred'] for k,v in f.get('metrics',{}).items() if number(v))
            if p.get('metric') not in schema or p.get('comparison') not in COMPARE or not number(p.get('value')):
                raise ValueError('Unknown numeric fact metric, comparison or bound')


def adjudicate_predictions(predictions,facts,previous_facts=None):
    results=[]
    for p in predictions:
        def match(rows):
            return [f for f in (rows or []) if f.get('value') and f['pred']==p['pred'] and f['args']==p['args']
                    and all(f.get('metrics',{}).get(k)==v for k,v in p.get('scope',{}).items())]
        now=match(facts);prev=match(previous_facts)
        confirmed=None
        if p['expect']=='metric':
            xs=[f['metrics'].get(p['metric']) for f in now]
            if len(xs)==1 and number(xs[0]):confirmed=COMPARE[p['comparison']](xs[0],p['value'])
        elif previous_facts is not None:
            if p['expect']=='appears' and not prev:confirmed=bool(now)
            if p['expect']=='disappears' and prev:confirmed=not now
        results.append({**p,'status':'undetermined' if confirmed is None else 'confirmed' if confirmed else 'refuted',
                        'fact_ids':[f['id'] for f in now],'previous_fact_ids':[f['id'] for f in prev]})
    return results


def proposal(ctx,offer):
    spec=ctx['spec'];decl=spec['perturbation_variables'];rule=ctx['current_rule']
    op=offer['op'];a=deepcopy(offer['bindings']);cfg=spec['director'];cid=spec['active_post_condition']
    axes=continuous(spec);space=spec['task']['action_space'];action_vars=space.get('continuous_displacement_variables',axes[:space['dim']])
    budget=min(ctx['budget_left'],spec['budget'].get('max_batch_trials',ctx['budget_left']))
    condition=a.get('condition') if isinstance(a.get('condition'),dict) else {}
    setting=deepcopy(ctx.get('execution',{}).get('fixed_settings',{}))
    properties=[k for k,d in decl.items() if d['kind']=='object_property']
    choices=[[condition[k]] if k in condition else list(decl[k]['levels']) for k in properties]
    blocks=[dict(zip(properties,c)) for c in product(*choices)]
    var=a.get('var');v=decl.get(var,{});cells=[]
    def add(s=None,acts=None,group=None):
        cell={'settings':{**setting,**(s or {})},'blocks':deepcopy(blocks if group is None else group),
              'reps':cfg['plan_reps'],'actions':acts or []};cells.append(cell)
        return cell
    def action(i,value):
        u=[0.]*space['dim'];u[i]=float(value);return {'kind':space['id'],'u':u}
    interval=a.get('interval');points=[]
    if op=='set_allocation':
        bounds=[f for f in ctx['facts'] if f['id'] in offer['premises_satisfied_by']]
        for f in bounds:
            m=f['metrics']
            if f['pred']=='frontier_gap': points.append({f['args'][0]:float(np.mean(m['interval']))})
            if f['pred']=='recovery_evidence_missing' and m['intervals']:points.append({k:float(np.mean(x)) for k,x in m['intervals'].items()})
        if not points:points=[{k:decl[k].get('reference',0) for k in action_vars}]
        policy=any(f['pred']=='model_executable' for f in ctx['facts'])
        roles=['expert','direct']+(['policy'] if policy else [])
        # A combined plan has one independent trial per declared role and block.
        cap=budget//len(blocks)
        if not cap:raise ValueError('Insufficient batch budget for required blocks')
        roles=roles[:cap]
        a.update(start_points=points[:1],target_consumer=cid,allocation={r:(len(blocks) if r in roles else 0) for r in ('direct','expert','policy')})
        if a.get('phase')=='plan':
            for role in roles:
                cell=add(points[0]);cell['reps']=1
    elif op in ('probe_recovery_frontier','request_expert_recovery'):
        add({var:a['start_mm'] if 'start_mm' in a else float(np.mean(interval))})
    elif op=='recalibrate_reference':
        add({v:decl[v].get('reference',0.) for v in action_vars})['reps']=max(2,cfg['plan_reps'])
    elif op=='expand_range':
        idx=0 if a['side']=='negative' else 1
        x=rule['variables'][var]['range'][idx]
        a['by']=min(abs(v['envelope'][idx]-x),v.get('max_expansion',v.get('nominal_step',1)))
        add({var:x+a['by']*(-1 if idx==0 else 1)})
    elif op=='densify':
        lo,hi=interval or rule['variables'][var]['range']
        a['interval']=[lo,hi];xs=np.linspace(lo,hi,cfg['densify_points']+2)[1:-1]
        for x in xs:add({var:float(x)})
    elif op=='activate_variable':
        xs=v.get('levels',[]) if v['kind']!='continuous' else sorted(set([*rule['variables'][var]['range'],v['reference']]))
        for x in xs:add({var:x})
    elif op=='probe_hidden_variable':
        for x in v['levels']:add({var:x})
    elif op in ('collect_transitions','verify_edges'):
        origin={k:float(np.mean(x)) for k,x in a.get('intervals',{}).items()}
        if op=='verify_edges':
            f=next(f for f in ctx['facts'] if f['pred']=='inferred_edge_unverified' and f['metrics']['edge_id']==a['edge_id'])
            add(acts=[{'kind':space['id'],'u':f['metrics']['u']}])['source_node_id']=a['source_node_id']
        else:
            counter=a.get('action_design')=='crossed_sign_amplitude'
            for i,k in enumerate(action_vars):
                lo,hi=decl[k]['envelope'];lim=spec['safety_envelope']['max_action'][i]
                pos=origin.get(k,decl[k].get('reference',0))
                for sign in ((1,-1) if counter else ((1,) if hi>pos else (-1,))):
                    room=min(lim,hi-pos if sign==1 else pos-lo)
                    if room<=0:continue
                    n=spec['dynamics']['min_amplitude_levels'] if counter else 1
                    for j in range(1,n+1):
                        magnitude=room*j/(n+1)
                        # Compare signs/amplitudes under the same object condition.
                        add(origin,[action(i,sign*magnitude)],group=blocks[:1] if counter else blocks)
                if not counter and cells:break
    elif op=='validate_B' and ctx.get('independent_plan'):cells=deepcopy(ctx['independent_plan']['cells'])
    elif op in COLLECTION:add()
    elif op=='request_capability':a['description']='Acquire a capability that distinguishes the unresolved property.'
    plan={'cells':cells} if cells else None
    if op=='validate_B':plan=deepcopy(ctx.get('independent_plan') or {'cells':cells,'cluster_classes':['Ready','NeedAction']})
    premises=offer['premises_satisfied_by']
    current=[f for f in ctx['facts'] if f['id'] in premises and not f['pred'].startswith('phase_')]
    if current:
        f=current[0];m=f.get('metrics',{})
        if f['pred'] in ('expert_frontier','policy_frontier','direct_boundary'):
            pred={'pred':f['pred'],'args':f['args'],'expect':'metric','metric':'evaluated_trials','comparison':'gt','value':m['evaluated_trials'],
                  'scope':{k:m[k] for k in ('model_id','condition')}}
        else:pred={'pred':f['pred'],'args':f['args'],'expect':'disappears'}
    else:pred={'pred':'coverage_ok','args':['V'],'expect':'appears'}
    return {'operator':op,'args':a,'premises':list(premises),'hypotheses_update':[],'discriminates':[],
            'expected':'下一轮按所列事实检验预测。','reason':'按已声明的事实、目标区域、试次数和预算选择。','plan':plan,'predictions':[pred]}


def scoring(ctx,offer,answer):
    spec=ctx['spec'];cid=spec['active_post_condition'];facts=ctx['facts']
    variance_stats=next((f['metrics'] for f in facts if f['pred']=='posterior_variance' and f['args']==[cid]),{})
    variances=variance_stats.get('by_node',{})
    targeted=[f for f in facts if f['id'] in offer['premises_satisfied_by'] and not f['pred'].startswith('phase_')]
    a=answer['args'];gap=0;var=a.get('var');cells=(answer['plan'] or {}).get('cells',[])
    region=a.get('intervals',{}).copy()
    if var and a.get('interval'):region[var]=a['interval']
    if var and 'start_mm' in a:
        tol=next((f['metrics']['resolution_mm'] for f in facts if f['pred']=='coordinate_resolution' and f['args']==[var]),
                   spec['facts']['boundary_resolution_mm'])
        region[var]=(a['start_mm']-tol,a['start_mm']+tol)
    exact={i for f in targeted for i in f['evidence']}
    known=variance_stats.get('locations',{})
    ids={i for i in exact if i in variances and i not in known}
    condition=a.get('condition') or {}
    for nid,loc in known.items():
        if nid not in variances:continue
        attrs=loc.get('attrs',{})
        if any(k in attrs and attrs[k]!=v for k,v in condition.items()):continue
        if region:
            if all(k in loc['coords'] and lo<=loc['coords'][k]<=hi for k,(lo,hi) in region.items()):ids.add(nid)
        elif targeted and (nid in exact or loc.get('trial_id') in exact):
            ids.add(nid)
        elif not targeted:
            # Unconditional calibration/outcome proposals still target the sampled starts.
            for cell in cells:
                starts={k:cell['settings'].get(k,decl.get('reference',0)) for k,decl in spec['perturbation_variables'].items()
                        if decl['kind']=='continuous' and decl.get('active',True)}
                if starts and all(k in loc['coords'] and abs(loc['coords'][k]-val)<=spec['facts']['boundary_resolution_mm']
                                  for k,val in starts.items()):ids.add(nid)
    for f in facts:
        if f['pred']!='frontier_gap' or f['args'][-1]!=cid:continue
        if f in targeted:
            gap+=f['metrics']['length_mm'];continue
        if (a.get('var')!=f['args'][0] or a.get('side')!=f['args'][1] or
                a.get('condition',{})!=f['metrics'].get('condition',{})):continue
        lo,hi=f['metrics']['interval']
        if 'start_mm' in a and lo<=a['start_mm']<=hi:gap+=hi-lo
        elif 'interval' in a:
            x,y=a['interval'];gap+=max(0.,min(hi,y)-max(lo,x))
    n=sum(c['reps']*len(c.get('blocks',[{}])) for c in (answer['plan'] or {}).get('cells',[]))
    total=sum(variances[i] for i in ids)
    return {'target_node_ids':sorted(ids),'variance_sum':total,'frontier_gap_mm':gap,'planned_trials':n,
            'score':(total+spec['director']['frontier_weight']*gap)/n if n else None}


def finalize_context(ctx):
    if ctx['spec'].get('graph_schema')!='state_graph_v5':return ctx
    spec=ctx['spec'];cfg=spec.get('director',{})
    if cfg.get('use_priors',True):ctx['effective_priors']=deepcopy(spec.get('priors',{}))
    else:
        spec.pop('priors',None);ctx.pop('effective_priors',None)
        def strip(obj):
            if isinstance(obj,dict):
                for key in ('priors','effective_priors','sampling_preference'):obj.pop(key,None)
                for child in obj.values():strip(child)
            elif isinstance(obj,list):
                for child in obj:strip(child)
        strip(ctx)
    if cfg.get('checkable_predictions'):
        ctx['prediction_contract']={'required':'non-empty predictions with pred, exact args, expect=appears|disappears|metric; metric additionally requires metric,comparison,value; optional scope by condition/model_id',
                                    'comparisons':list(COMPARE),'predicate_schema':sorted(PREDICATES|{f['pred'] for f in ctx['facts']}),'numeric_metric_schema':METRICS}
    if cfg.get('operator_scoring') or cfg.get('mode')=='fixed_priority':
        from .operators import validate_decision
        items=[]
        for offer in ctx['applicable_operators']:
            item=dict(operator=offer['op'],bindings=offer['bindings'],premises=offer['premises_satisfied_by'])
            try:
                candidate=proposal(ctx,offer);validate_decision(candidate,ctx)
                item.update(scoring(ctx,offer,candidate),proposal=candidate)
            except (KeyError,ValueError,TypeError) as exc:item.update(score=None,proposal=None,unavailable_reason=str(exc))
            items.append(item)
        ctx['operator_scores']=items
    return ctx


def fixed_decision(ctx):
    options=[v for v in ctx['operator_scores'] if v.get('proposal')]
    if not options:raise ValueError('No feasible validated operation within remaining budget')
    if any(v['operator']=='complete' for v in options):
        return deepcopy(next(v['proposal'] for v in options if v['operator']=='complete'))
    def order(v):
        m=v.get('score')
        # Positive exchange of data takes precedence over idle, with an explicit stable tie break.
        gap_ids={f['id'] for f in ctx['facts'] if f['pred']=='frontier_gap'}
        direct_gap=bool(gap_ids.intersection(v['premises'])) and v['operator']=='request_expert_recovery'
        return (v['operator']=='pause',m is None,-float(m or 0),not direct_gap,digest(v['proposal']))
    return deepcopy(min(options,key=order)['proposal'])



def prediction_snapshot(directory,ctx):
    """Only adjudicate against a subsequently rebuilt evidence graph."""
    import json
    from pathlib import Path
    path=Path(directory)/'hypotheses.json'
    journal=json.loads(path.read_text()) if path.is_file() else {'pending':None,'history':[]}
    pending=journal.get('pending')
    evidence_version=digest([ctx.get('graph_version'),[f for f in ctx['facts'] if not f['pred'].startswith('phase_')]])
    if pending and pending['evidence_version']!=evidence_version:
        results=adjudicate_predictions(pending['predictions'],ctx['facts'],pending['facts'])
        journal['history'].append({'graph_version':ctx.get('graph_version'),'decision_id':pending['decision_id'],'results':results})
        journal['pending']=None
    assessed=[p for item in journal['history'] for p in item['results'] if p['status']!='undetermined']
    journal['accuracy']=sum(p['status']=='confirmed' for p in assessed)/len(assessed) if assessed else None
    journal['assessed']=len(assessed)
    ctx['previous_prediction_results']=journal['history'][-1:]
    return journal,evidence_version
