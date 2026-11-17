"""Specification-driven collection operators; proposals never execute hardware."""
from copy import deepcopy
import numpy as np
from .common import digest


def applicable_operators(facts,spec,rule,state=None):
    state=state or {};active=spec['active_post_condition'];variables=spec.get('perturbation_variables',{});out=[]
    def add(op,fs=(),**bindings):
        item={'op':op,'bindings':bindings,'premises_satisfied_by':[f['id'] for f in fs]}
        if item not in out:out.append(item)
    def find(pred):return [f for f in facts if f['pred']==pred and f['value']]
    if state.get('frozen_independent'):
        for f in find('indep_B_pass'):add('complete',[f])
        add('validate_B',post_condition=active,role='independent',required_symbols=['Ready','NeedAction'])
        add('pause');return out
    for pred,op in [('repeat_calibration_missing','collect_seed'),('consumer_evidence_missing','collect_outcomes'),('transition_evidence_missing','collect_transitions')]:
        for f in find(pred):add(op,[f],post_condition=active)
    for f in find('dev_B_gap'):
        add('collect_transitions',[f],post_condition=active)
        add('collect_outcomes',[f],post_condition=active)
        add('train',[f])
    for name,definition in variables.items():
        setting=rule.get('variables',{}).get(name,{})
        enabled=setting.get('active',definition.get('active',True))
        if enabled and definition['kind']=='continuous':
            for f in find('no_failure'):
                if f['args']==[rule['rule_id'],active]:
                    for side in ('negative','positive'):add('expand_range',[f],var=name,side=side)
            for f in find('boundary_gap'):
                if f['args'][0]==name and f['args'][2]==active:add('densify',[f],var=name,interval=f['args'][1])
            for pred in ('dynamics_uncertain','high_disagreement'):
                for f in find(pred):add('densify',[f],var=name,region=f['args'][0])
        if not enabled:
            failures=[f for f in find('unreachable') if f['args'][-1]==active]
            groups={g for f in failures for g in f['metrics'].get('trial_groups',[])}
            coverage=[f for f in find('coverage') if f['args']==[name,active] and len(f['metrics']['values'])==1]
            if len(groups)>=3:add('activate_variable',failures,var=name)
            elif definition['kind']=='categorical' and coverage:add('activate_variable',coverage,var=name)
    hidden=find('hidden_variable_candidate')
    for f in hidden:
        cluster,name,cid=f['args']
        if cid!=active or name not in variables:continue
        kind=variables[name]['kind']
        if kind=='protocol':add('probe_hidden_variable',[f],var=name,cluster=cluster)
        if kind=='object_property':
            add('block_by_property',[f],var=name,cluster=cluster)
            if not f['metrics'].get('shared_ready_regions'):add('request_capability',[f],var=name,cluster=cluster)
    for f in find('stochastic_outcome'):
        if f['args'][-1]==active and not any(h['args'][0]==f['args'][0] and h['args'][-1]==active for h in hidden):add('repeat_state',[f],cluster=f['args'][0])
    for f in find('inferred_edge_unverified'):
        if f['args'][-1]==active and f['metrics']['path_fraction']>.5:
            add('verify_edges',[f],source_node_id=f['metrics']['source_node_id'],edge_id=f['metrics']['edge_id'])
    gaps = [f for f in find('development_class_missing') if f['args'] == [active]]
    for f in gaps:
        add('collect_outcomes', [f], post_condition=active)
        add('collect_transitions', [f], post_condition=active)
    # Mixed or unknown regions can still yield both classes in a prospective B.
    # A model calling every observed region Ready cannot exercise its correction.
    uniform = any(f['metrics'].get('observed_classes') in (['Ready'], ['NeedAction']) for f in gaps)
    coverage={f['args'][0]:f for f in find('coverage_ok')}
    if not uniform and all(h in coverage for h in ('V','pi','f')):add('train',[coverage[h] for h in ('V','pi','f')])
    if not uniform:
        for f in find('model_executable'):add('validate_B',[f],post_condition=active,required_symbols=['Ready','NeedAction'])
    for f in find('dev_B_pass'):add('freeze',[f])
    for f in find('label_conflict'):add('review_labels',[f])
    add('pause');return out


def validate_decision(answer,context):
    required={'hypotheses_update','operator','args','premises','discriminates','plan','expected','reason'}
    if not isinstance(answer,dict) or not required<=set(answer):raise ValueError('Missing symbolic fields')
    if not isinstance(answer['hypotheses_update'],list) or not all(isinstance(h,dict) for h in answer['hypotheses_update']):raise ValueError('Hypotheses update must be an array of objects')
    if not isinstance(answer['discriminates'],list):raise ValueError('Discriminates must be an array')
    if not isinstance(answer['args'],dict) or not isinstance(answer['premises'],list):raise ValueError('Invalid arguments/premises')
    if not all(isinstance(answer[k],str) and answer[k] for k in ('expected','reason')):raise ValueError('Expected and reason must be strings')
    facts={f['id']:f for f in context['facts']}
    if not all(p in facts and facts[p]['value'] for p in answer['premises']):raise ValueError('Unknown premise')
    op=answer['operator'];args=answer['args'];spec=context['spec'];variables=spec['perturbation_variables']
    matches=[c for c in context['applicable_operators'] if c['op']==op and all(args.get(k)==v for k,v in c['bindings'].items()) and set(c['premises_satisfied_by'])<=set(answer['premises'])]
    if not matches:raise ValueError('Operator is not applicable')
    collection={'collect_seed','collect_transitions','collect_outcomes','expand_range','densify','activate_variable','probe_hidden_variable','block_by_property','repeat_state','verify_edges','validate_B'}
    plan=answer['plan']
    if op not in collection:
        if plan is not None:raise ValueError('Noncollection operator must not carry a collection plan')
        if op=='request_capability' and not args.get('description'):raise ValueError('Describe the requested sensing/action capability')
        return answer
    if op=='validate_B' and context.get('independent_plan') is not None and plan!=context['independent_plan']:
        raise ValueError('Independent B must execute the predeclared frozen plan unchanged')
    if not isinstance(plan,dict) or not isinstance(plan.get('cells'),list) or not plan['cells']:raise ValueError('Collection cells required')
    count=0
    for cell in plan['cells']:
        if type(cell.get('reps')) is not int or cell['reps']<1:raise ValueError('Positive repeat count required')
        settings=cell.get('settings',{});blocks=cell.get('blocks',[{}]);
        if not isinstance(settings,dict) or not isinstance(blocks,list) or not blocks:raise ValueError('Invalid settings/blocks')
        from .skill_interface import normalize_action
        actions=[normalize_action(a,spec) for a in cell.get('actions',[])]
        if len(actions)>spec.get('runtime',{}).get('max_steps',3):raise ValueError('Too many actions in one trial')
        if op=='collect_transitions' and not actions:raise ValueError('Transition collection requires explicit actions')
        if op=='validate_B' and actions:raise ValueError('B executes the frozen candidate, not scripted corrections')
        validator=spec.get('adapters',{}).get('validate_collection_cell')
        if validator:
            from .skill_interface import load_callable
            load_callable(validator)(cell,spec,op,args)
        count+=cell['reps']*len(blocks)
        for key,value in context.get('execution',{}).get('fixed_settings',{}).items():
            if key in settings and settings[key]!=value:raise ValueError('Setting differs from the selected consumer protocol: '+key)
        for name,value in settings.items():
            if name not in variables:raise ValueError('Undeclared perturbation')
            var=variables[name]
            if var['kind']=='continuous':
                if type(value) not in (int,float) or not np.isfinite(value) or not var['envelope'][0]<=value<=var['envelope'][1]:raise ValueError('Setting outside envelope')
            elif value not in var['levels']:raise ValueError('Undeclared level')
        for block in blocks:
            if any(k not in block for k in context.get('execution',{}).get('required_blocks',[])):raise ValueError('Plan must record the declared object property block')
            for name,value in block.items():
                if name not in variables or variables[name]['kind']!='object_property' or value not in variables[name]['levels']:raise ValueError('Invalid block property')
    budget=context.get('budget_left',spec.get('budget',{}).get('max_new_trials',0))
    budget=min(budget,spec.get('budget',{}).get('max_batch_trials',budget))
    if count>budget:raise ValueError('Plan exceeds remaining trial budget')
    var=args.get('var');definition=variables.get(var,{})
    values=[c['settings'].get(var) for c in plan['cells']] if var else []
    if op=='expand_range':
        amount=args.get('by')
        if type(amount) not in (int,float) or not 0<amount<=definition.get('max_expansion',definition.get('nominal_step',1)):raise ValueError('Expansion step exceeds specification')
        lo,hi=context['current_rule']['variables'][var]['range'];bounds=[lo-amount,hi] if args['side']=='negative' else [lo,hi+amount]
        if bounds[0]<definition['envelope'][0] or bounds[1]>definition['envelope'][1]:raise ValueError('Expansion outside envelope')
        if not any(x is not None and (x<lo or x>hi) for x in values):raise ValueError('Plan must sample the expanded range')
    if op=='densify':
        interval=args.get('interval')
        if interval is None and args.get('region'):
            interval=context['current_rule']['variables'][var]['range']
        if not interval or len(interval)!=2:raise ValueError('Densification interval required')
        observed=context.get('observed_positions',{}).get(var,[])
        fresh={x for x in values if x is not None and interval[0]<=x<=interval[1] and not any(np.isclose(x,old,atol=1e-6) for old in observed)}
        if len(fresh)<3:raise ValueError('Densification requires three new positions')
    if op=='activate_variable':
        levels={x for x in values if x is not None};minimum=3 if definition['kind']=='continuous' else 2
        if len(levels)<minimum or definition.get('reference') not in levels:raise ValueError('Initial variable levels must include the reference')
    if op=='probe_hidden_variable':
        levels=args.get('levels',definition.get('levels',[]));signatures=[]
        for level in levels:
            signatures.append(sorted((str({k:v for k,v in c['settings'].items() if k!=var}),str(c.get('blocks',[])),c['reps']) for c in plan['cells'] if c['settings'].get(var)==level))
        if len(signatures)<2 or not signatures[0] or any(s!=signatures[0] for s in signatures[1:]):raise ValueError('Protocol probes require matched settings at each level')
    if op=='block_by_property':
        for cell in plan['cells']:
            if {b.get(var) for b in cell.get('blocks',[])}!=set(definition['levels']):raise ValueError('Every sampled state must include all declared property blocks')
    if op=='verify_edges' and not any(c.get('source_node_id')==args['source_node_id'] for c in plan['cells']):raise ValueError('Verify the specified inferred edge source')
    if op=='validate_B' and not {'Ready','NeedAction'}<=set(plan.get('cluster_classes',[])):raise ValueError('Validation must cover ready and action states')
    return answer


def update_rule(rule,decision):
    new=deepcopy(rule);args=decision['args'];op=decision['operator'];var=args.get('var')
    if op=='activate_variable':new.setdefault('variables',{}).setdefault(var,{})['active']=True
    elif op=='expand_range':
        bounds=new['variables'][var]['range'];bounds[0 if args['side']=='negative' else 1]+=args['by']*(-1 if args['side']=='negative' else 1)
    new.update(parent_rule_id=rule['rule_id'],reason=decision['reason'],evidence_refs=decision['premises'])
    new['rule_id']='rule_'+digest(new)[:16];return new


def frontier_context(ledger, round_id):
    """Separate operator namespace: no legacy expand_range/densify/queue_plan or campaign retraining."""
    from bottle_scan.max_expert_demonstration_finder.search import offers,report
    r=ledger.round(round_id);attempts=ledger.attempts(round_id);rep=report(r,attempts)
    choices=offers(r,attempts)
    bindings={'round_id':round_id,'model_id':r['manifest']['model_id'],'reference_digest':r['manifest']['reference_digest'],
              'manifest_digest':r['digest'],'remaining_physical':rep['project_physical_budget_left']}
    operations=[]
    if r['stage']=='closed' and any(not x['expert']['stronger'] for x in rep['expert_witnessed_recoverable_set']):
        operations.append(dict(op='retrain_from_frontier',**bindings))
    if r['stage']=='explore' and not ledger.pending() and any(e['controller_role']=='policy' and e['ended_ns'] for e in rep['episodes']):
        operations.append(dict(op='validate_recovery_frontier',requires='predeclare_points_and_n',**bindings))
    return dict(schema='episode_frontier_operator_context_v1',bindings=bindings,facts=rep['facts'],
                collection_choices=choices,lifecycle_choices=operations,expert_requests=rep['expert_requests'])
