"""Guide section 7: facts select applicable symbolic experiment operators.

This module produces plans only. It does not connect to hardware or edit sessions.
"""
from copy import deepcopy
import hashlib
import json
import math

OPS = {'expand_range','densify','split_by_object','probe_hidden_variable',
       'add_axis','verify_edges','train','validate_B','freeze','complete','review_labels','pause'}

def fact_rows(facts):
    return facts.get('facts', []) if isinstance(facts, dict) else facts

def _true(f):
    return f.get('value', f.get('truth', True)) is True

def _has(facts, pred):
    return [f for f in fact_rows(facts) if f.get('pred', f.get('predicate')) == pred and _true(f)]

def _args(f):
    return f.get('args', [])

def _number(x):
    return type(x) in (int,float) and math.isfinite(x)

def applicable_operators(facts, spec, rule, state=None):
    state = state or {}
    out=[]
    def add(op, fs=(), **bound):
        entry={'op':op,'premises_satisfied_by':[f['id'] for f in fs], 'bindings':bound}
        if entry not in out:out.append(entry)
    if state.get('frozen_independent'):
        for f in _has(facts,'indep_B_pass'):add('complete',[f])
        add('pause');return out
    for f in _has(facts,'no_failure'):
        if not _args(f) or _args(f)[0] in (rule.get('rule_id'),'current'):
            for axis in rule.get('active_axes',['a']):
                for side in ('negative','positive'):add('expand_range',[f],axis=axis,side=side)
    # Absence of a predicate is useful only over an observed current-rule side.
    for side in state.get('observed_sides_without_failure',[]):
        fs=[f for f in fact_rows(facts) if f['id'] in side.get('evidence_facts',[]) and _true(f)]
        if fs:add('expand_range',fs,axis=side['axis'],side=side['side'])
    for f in _has(facts,'boundary_gap'):
        a=_args(f)
        if len(a)>=3:add('densify',[f],axis=a[0],interval=a[1],object=a[2])
    for f in _has(facts,'high_disagreement'):
        a=_args(f)
        if len(a)>=2:add('densify',[f],axis=a[0],x=a[1])
    for f in _has(facts,'object_split'):
        a=_args(f)
        if len(a)>=3:add('split_by_object',[f],objects=a[:2],axis=a[2])
    allowed_vars=set(spec.get('recordable_attributes',[]))|set(spec.get('variable_protocol',{}))
    for f in _has(facts,'conflict'):
        for var in sorted(allowed_vars):add('probe_hidden_variable',[f],var=var)
    for axis in spec.get('envelope',{}).get('allowed_axes',['a']):
        if axis in rule.get('active_axes',[]):continue
        fs=[f for f in _has(facts,'unreachable') if f.get('metrics',{}).get('observed_failure') is True]
        evidence={str(g) for f in fs for g in f.get('metrics',{}).get('trial_groups',[])}
        if len(evidence)>=3:add('add_axis',fs,axis=axis)
    for f in _has(facts,'edge_unverified'):
        metrics=f.get('metrics',f.get('values',f.get('basis',{})))
        ratio=metrics.get('path_fraction',metrics.get('unverified_path_fraction',metrics.get('ratio',0)))
        candidates=metrics.get('verification_candidates',[])
        if ratio>.5 and candidates:
            top=max(c['coverage_count'] for c in candidates)
            for c in candidates:
                if c['coverage_count']==top:add('verify_edges',[f],source_node_id=c['source_node_id'],edge_ids=c['edge_ids'])
    coverage={str(_args(f)[0]):f for f in _has(facts,'coverage_ok') if _args(f)}
    if state.get('labels_changed',True) and all(h in coverage for h in ('I','beta','pi')):
        add('train',[coverage[h] for h in ('I','beta','pi')])
    for f in _has(facts,'model_executable'):add('validate_B',[f])
    for f in _has(facts,'dev_B_pass'):add('freeze',[f])
    for f in _has(facts,'label_conflict'):add('review_labels',[f])
    add('pause')
    return out

def validate_decision(answer, context):
    required={'hypotheses_update','operator','args','premises','discriminates','plan','expected','reason'}
    if not isinstance(answer,dict) or not required<=set(answer):raise ValueError('missing symbolic decision fields')
    if set(answer)-required-{'_meta'}:raise ValueError('unknown symbolic decision fields')
    op=answer['operator'];args=answer['args']
    if op not in OPS or not isinstance(args,dict):raise ValueError('unknown operator or invalid args')
    for k in ('reason','expected'):
        if not isinstance(answer[k],str) or not answer[k].strip():raise ValueError(k+' is required')
    if not isinstance(answer['hypotheses_update'],list) or not isinstance(answer['premises'],list):raise ValueError('invalid hypothesis/premise list')
    facts={f['id']:f for f in fact_rows(context['facts'])}
    if any(x not in facts or not _true(facts[x]) for x in answer['premises']):raise ValueError('premise is not a current true fact')
    matches=[]
    for candidate in context['applicable_operators']:
        if candidate['op']!=op:continue
        b=candidate.get('bindings',{})
        if any(k in args and args[k]!=v for k,v in b.items() if k not in ('x','edge_ids')):continue
        if any(k not in args for k in b if k not in ('x','edge_ids')):continue
        if not set(candidate['premises_satisfied_by'])<=set(answer['premises']):continue
        matches.append(candidate)
    if not matches:raise ValueError('operator preconditions/bindings do not hold')
    spec=context['spec'];rule=context['current_rule'];env=spec['envelope']
    if op=='expand_range':
        step=args.get('by_mm',5)
        if not _number(step) or not 0<step<=5:raise ValueError('expansion must be in (0,5] mm')
        axis=args['axis'];lo,hi=rule['range_mm'][axis];elo,ehi=env['range_mm'][axis]
        if args['side']=='negative':lo-=step
        elif args['side']=='positive':hi+=step
        else:raise ValueError('unknown expansion side')
        if lo<elo or hi>ehi:raise ValueError('expansion outside declared envelope')
    if op=='probe_hidden_variable':
        var=args['var'];allowed=set(spec.get('variable_protocol',{}))|set(spec.get('recordable_attributes',[]))
        if var not in allowed:raise ValueError('undeclared variable')
        if not isinstance(args.get('levels'),list) or len(args['levels'])<2:raise ValueError('paired variable levels are required')
        constraints=spec.get('variable_protocol',{}).get(var,{})
        if isinstance(constraints,dict) and 'levels' in constraints and not set(args['levels'])<=set(constraints['levels']):raise ValueError('variable levels outside specification')
    plan=answer.get('plan')
    plan_ops={'expand_range','densify','probe_hidden_variable','add_axis','verify_edges','validate_B'}
    if op in plan_ops:
        if not isinstance(plan,dict) or not isinstance(plan.get('cells'),list) or not plan['cells']:raise ValueError('operator requires nonempty collection plan')
        allowed_cell_keys={'offset_mm','reps','target_symbol','source_node_id'}
        if op=='probe_hidden_variable':allowed_cell_keys.add(args['var'])
        total=0;positions=[]
        axis=args.get('axis','a');limits=env['range_mm'].get(axis)
        if not limits:raise ValueError('axis not declared')
        for cell in plan['cells']:
            if set(cell)-allowed_cell_keys:raise ValueError('plan cannot change undeclared protocol fields')
            x=cell.get('offset_mm');reps=cell.get('reps')
            if not _number(x) or not limits[0]<=x<=limits[1]:raise ValueError('plan offset outside envelope')
            if type(reps) is not int or reps<1:raise ValueError('positive integer repetitions required')
            total+=reps;positions.append(x)
        if total>spec.get('budget_left',0):raise ValueError('plan exceeds remaining trial budget')
        if op=='densify':
            interval=args.get('interval');existing=set(context.get('observed_positions',{}).get(axis,[]))
            if not isinstance(interval,list) or len(interval)!=2 or not interval[0]<interval[1]:raise ValueError('densify interval required')
            if not any('x' not in c['bindings'] or interval[0]<=c['bindings']['x']<=interval[1] for c in matches):raise ValueError('densify interval must contain the cited high-disagreement position')
            new={x for x in positions if interval[0]<=x<=interval[1] and x not in existing}
            if len(new)<3:raise ValueError('densify requires at least three new positions')
        if op=='verify_edges':
            candidates=[c for c in matches if any(cell.get('source_node_id')==c['bindings']['source_node_id'] for cell in plan['cells'])]
            if not candidates:raise ValueError('verify_edges must include a maximum-coverage unverified start node')
        if op=='add_axis':
            if len(set(positions))!=3 or 0 not in positions:raise ValueError('new axis requires three levels including reference')
        if op=='validate_B':
            symbols={cell.get('target_symbol') for cell in plan['cells']}
            if not {'NeedShift','HandoffReady'}<=symbols:raise ValueError('B must cover NeedShift and HandoffReady')
    elif plan is not None:raise ValueError('this operator does not take a collection plan')
    if op in plan_ops:
        compile_collection_plan(answer,context)
    known={h['id'] for h in context.get('hypotheses',[])}
    for h in answer['hypotheses_update']:
        if not isinstance(h,dict) or not isinstance(h.get('id'),str):raise ValueError('hypothesis id required')
        if h.get('new'):
            if h['id'] in known or not h.get('claim'):raise ValueError('new hypothesis requires new id and claim')
            known.add(h['id'])
        elif h['id'] not in known:raise ValueError('unknown hypothesis')
        if not set(h.get('evidence',[]))<=set(facts):raise ValueError('unknown hypothesis evidence')
    if not isinstance(answer['discriminates'],list) or not set(answer['discriminates'])<=known:raise ValueError('unknown discrimination target')
    return deepcopy(answer)

def update_rule(rule, answer):
    result=deepcopy(rule);op=answer['operator'];args=answer['args']
    if op=='expand_range':
        side=0 if args['side']=='negative' else 1
        result['range_mm'][args['axis']][side]+=(-1 if side==0 else 1)*args.get('by_mm',5)
    elif op=='densify':result['densify_interval']={'axis':args['axis'],'interval':args['interval'],'object':args.get('object')}
    elif op=='add_axis':
        result['active_axes'].append(args['axis']);xs=[c['offset_mm'] for c in answer['plan']['cells']];result['range_mm'][args['axis']]=[min(xs),max(xs)]
    elif op=='split_by_object':result['separate_object_targets']=args['objects']
    elif op=='probe_hidden_variable':result['controlled_variable']=deepcopy(args)
    result['parent_rule_id']=rule['rule_id'];result['reason']=answer['reason'];result['evidence_refs']=answer['premises']
    digest=hashlib.sha256(json.dumps(result,sort_keys=True).encode()).hexdigest()[:12]
    result['rule_id']='rule_'+digest
    return result


def compile_collection_plan(answer,context):
    """Compile offline through the existing balance/reference plan validator."""
    from bottle_scan.skill_collection import selection_plan
    spec=context['spec'];cells=answer['plan']['cells'];args=answer['args']
    axis=args.get('axis','a');op=answer['operator']
    cfg=deepcopy(spec.get('collection_config') or {
        'design':{'bottles':['light','heavy'],'zero_every':8},
        'force':{'target':5.,'probe_fraction':.2},
        'motion':{'max_grasp_offset_mm':max(abs(x) for x in spec['envelope']['range_mm'][axis])}})
    if op=='probe_hidden_variable':
        var=args['var'];levels=args['levels']
        grouped={str(level):[] for level in levels}
        for cell in cells:
            if var not in cell or cell[var] not in levels:raise ValueError('each paired cell must record the controlled variable')
            grouped[str(cell[var])].append(cell)
        signatures=[sorted((c['offset_mm'],c['reps']) for c in part) for part in grouped.values()]
        if not signatures[0] or any(x!=signatures[0] for x in signatures):raise ValueError('controlled variable requires matching offsets and repetitions at every level')
    else:grouped={'default':cells}
    if op=='expand_range':
        lo,hi=context['current_rule']['range_mm'][axis];step=args.get('by_mm',5)
        new=(lo-step,lo) if args['side']=='negative' else (hi,hi+step)
        expanded=(lo-step,hi) if args['side']=='negative' else (lo,hi+step)
        if any(c['offset_mm']!=0 and not expanded[0]<=c['offset_mm']<=expanded[1] for c in cells):raise ValueError('plan goes outside the expanded current rule')
        if not any(new[0]<=c['offset_mm']<=new[1] and not lo<=c['offset_mm']<=hi for c in cells):raise ValueError('expansion plan must sample the newly expanded region')
    compiled=[]
    for part in grouped.values():
        selection={'schema':'skill_experiment_selection_v1','role':'development','stage':'a_closed_loop',
            'status':'planned_unexecuted','planned_trial_attempts':sum(c['reps'] for c in part),
            'selected':[{'condition':{'a_mm':c['offset_mm'],'contact_height_l_mm':0,
                'probe_force':cfg['force']['target']*cfg['force']['probe_fraction']},
                'reps':c['reps'],'bottle_blocks':cfg['design']['bottles'],
                'purpose':op,'reason':answer['reason'],'problem_refs':answer['premises']} for c in part]}
        compiled.extend(selection_plan(selection,cfg,'B' if op in ('verify_edges','validate_B') else 'A'))
    return {'rows':compiled,'axis':axis,'execution_authorized':False,
        'requires_axis_executor':axis!='a','source':'existing skill_collection.selection_plan'}
