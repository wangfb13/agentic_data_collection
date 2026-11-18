"""Additional symbolic operators and validators for revision five."""
from copy import deepcopy
import numpy as np
from .common import digest
from .v5_facts import continuous,number

FRONTIER={'probe_recovery_frontier','request_expert_recovery'}
NEW_COLLECTION=FRONTIER|{'recalibrate_reference'}


def revise(out,facts,spec,rule,state):
    cfg=spec.get('facts',{});oc=spec.get('operators',{});cid=spec['active_post_condition']
    bypred={}
    for f in facts:
        if f['value']:bypred.setdefault(f['pred'],[]).append(f)
    get=lambda p:bypred.get(p,[])
    def add(op,fs,**args):
        out.append({'op':op,'bindings':args,'premises_satisfied_by':[f['id'] for f in fs]})
    def remove(op):out[:]=[x for x in out if x['op']!=op]
    if cfg.get('per_side_no_failure'):
        remove('expand_range')
        for f in get('no_failure'):
            if len(f['args'])!=3:continue
            var,side,c=f['args'];decl=spec['perturbation_variables'].get(var,{})
            if c!=cid or decl.get('kind')!='continuous':continue
            v=rule['variables'][var];k=0 if side=='negative' else 1
            if v.get('active',decl.get('active',True)) and v['range'][k]!=decl['envelope'][k]:
                add('expand_range',[f],var=var,side=side)
    for f in get('boundary_within_cluster'):
        cluster,var,c=f['args']
        if c!=cid:continue
        add('densify',[f],var=var,interval=f['metrics']['interval'],cluster=cluster)
    if oc.get('handle_drift_and_unsupported'):
        for f in get('drift'):
            add('recalibrate_reference',[f],session=f['args'][0],post_condition=cid)
        for f in get('recovery_evidence_missing'):
            cluster,c=f['args'];intervals=f['metrics']['intervals']
            if c==cid and intervals:
                add('collect_transitions',[f],cluster=cluster,intervals=intervals,post_condition=c)
    if cfg.get('capability_predicates'):
        remove('activate_variable')
        for f in get('axis_observable'):
            var=f['args'][0];v=spec['perturbation_variables'][var]
            cons=[x for x in get('axis_consequential') if x['args']==[var,cid]]
            if cons and not rule['variables'].get(var,{}).get('active',v.get('active',True)):
                add('activate_variable',[f,*cons],var=var)
        for f in get('action_effect_unidentified'):
            add('collect_transitions',[f],post_condition=cid,action_design='crossed_sign_amplitude')
    if oc.get('decouple_dynamics'):
        remove('train')
        available={f['args'][0]:f for f in get('coverage_ok')}
        if {'V','pi'}<=available.keys():
            add('train',[available['V'],available['pi']])
    if spec.get('campaign',{}).get('integrated_frontier'):
        stop={(tuple(f['args']),digest(f['metrics'].get('condition')),f['metrics'].get('model_id')) for f in get('frontier_direction_closed')}
        for pred,role in [('expert_frontier','expert_recovery'),('policy_frontier','policy_frontier'),('direct_boundary','direct_baseline')]:
            for f in get(pred):
                var,side,c=f['args'];m=f['metrics']
                if c!=cid or (tuple(f['args']),digest(m.get('condition')),m.get('model_id')) in stop:continue
                if role=='policy_frontier' and not get('model_executable'):continue
                x=next_probe(m,spec,var,side)
                if x is not None:
                    add('request_expert_recovery' if role=='expert_recovery' else 'probe_recovery_frontier',[f],
                        role=role,var=var,side=side,start_mm=x,post_condition=cid,condition=m.get('condition') or {})
        for f in get('frontier_gap'):
            var,side,c=f['args'];m=f['metrics']
            if c==cid:
                add('request_expert_recovery',[f],role='expert_recovery',var=var,side=side,
                    interval=m['interval'],post_condition=c,condition=m.get('condition') or {})
    if spec.get('loop',{}).get('rules_via_operators'):
        for f in facts:
            if f['pred'] in ('frontier_gap','recovery_evidence_missing','dev_B_gap'):
                add('set_allocation',[f],post_condition=cid,phase=state.get('phase','plan'))
    if spec.get('observation',{}).get('auto_select'):
        for f in get('representation_ranking'):
            for name in f['metrics'].get('selectable',[]):
                add('select_representation',[f],representation=name)
    phases=[f for f in facts if f['pred'].startswith('phase_')]
    if phases:
        ph=phases[-1];phase=ph['pred'][6:]
        if phase=='plan' and spec.get('loop',{}).get('rounds'):
            out[:]=[o for o in out if o['op'] not in ('validate_B','train','freeze','complete')]
            if not any(o['op']=='collect_outcomes' for o in out):
                add('collect_outcomes',[ph],post_condition=cid)
        elif phase=='dev_validate':
            out[:]=[o for o in out if o['op'] in ('validate_B','pause')]
        elif phase=='update_rule':
            out[:]=[o for o in out if o['op'] in ('set_allocation','activate_variable','select_representation','expand_range','pause')]
        for op in out:
            if ph['id'] not in op['premises_satisfied_by']:op['premises_satisfied_by'].append(ph['id'])
    if any(o['op']!='pause' for o in out) and any(f['pred']=='phase_plan' for f in phases):
        remove('pause')
    return out


def next_probe(metrics,spec,var,side):
    if spec.get('facts',{}).get('rate_based_frontier') and 'bins' in metrics:
        from .frontier_rates import next_bin
        return next_bin(metrics,spec,var,side)
    sign=1 if side=='positive' else -1
    limit=abs(spec['perturbation_variables'][var]['envelope'][1 if sign>0 else 0])
    resolution=metrics['resolution_mm'];points=metrics.get('observed_points',[])
    samples=sorted(set([0.,limit]+[abs(p['x']) for p in points]))
    succeeded={abs(p['x']) for p in points if p['success'] is True}
    failed={abs(p['x']) for p in points if p['success'] is False}
    initial=min(spec['director']['initial_frontier_step_mm'],limit)
    if failed and not succeeded and 0. not in failed:
        return 0.
    if not failed:
        far=max(succeeded,default=0.)
        if far>=limit:return None
        return sign*min(limit,max(initial,2*far))
    pairs=[(a,b) for a,b in zip(samples,samples[1:]) if b-a>=2*resolution and
            (a in succeeded and b in failed or a in failed and b in succeeded)]
    if not pairs:return None
    a,b=max(pairs,key=lambda t:t[1]-t[0])
    return sign*(a+b)/2


def validate(answer,ctx):
    spec=ctx['spec'];op=answer['operator'];a=answer['args'];cells=(answer.get('plan') or {}).get('cells',[])
    if a.get('role')=='policy_frontier' and not any(f['pred']=='model_executable' and f['value'] for f in ctx['facts']):
        raise ValueError('Policy frontier requires a frozen executable candidate')
    if op in FRONTIER and a.get('role') not in ('direct_baseline','policy_frontier','expert_recovery'):
        raise ValueError('Frontier role missing')
    if op=='set_allocation':
        allocation=a.get('allocation');points=a.get('start_points')
        if not isinstance(allocation,dict) or set(allocation)!={'direct','policy','expert'} or not all(type(v) is int and v>=0 for v in allocation.values()) or sum(allocation.values())<1:
            raise ValueError('set_allocation needs finite nonnegative counts for direct, policy and expert')
        if allocation['policy'] and not any(f['pred']=='model_executable' and f['value'] for f in ctx['facts']):
            raise ValueError('Cannot allocate policy trials without an executable model')
        if sum(allocation.values())>ctx['budget_left']:raise ValueError('Allocation exceeds trial budget')
        if not isinstance(points,list) or not points:raise ValueError('set_allocation needs measured start coordinates')
        for point in points:
            if not isinstance(point,dict) or not point:raise ValueError('Empty start point')
            for k,x in point.items():
                if k not in continuous(spec) or not number(x) or not spec['perturbation_variables'][k]['envelope'][0]<=x<=spec['perturbation_variables'][k]['envelope'][1]:
                    raise ValueError('Invalid allocation start coordinate')
        if cells:
            if sum(c['reps']*len(c.get('blocks',[{}])) for c in cells)!=sum(allocation.values()):
                raise ValueError('Role counts must match expanded trial count')
            if any(not any(all(c['settings'].get(k)==x for k,x in pt.items()) for pt in points) for c in cells):
                raise ValueError('Sampled settings differ from allocated start points')
        if a.get('target_consumer') not in {c['id'] for c in spec['post_conditions']}:raise ValueError('Unknown target consumer')
    if op=='densify' and spec.get('facts',{}).get('noise_based_resolution'):
        x=[c['settings'].get(a['var']) for c in cells];interval=a.get('interval') or ctx['current_rule']['variables'][a['var']]['range']
        if any(not number(v) or not interval[0]<v<interval[1] for v in x):raise ValueError('Densify points must be strictly interior')
        distances=np.diff([interval[0],*sorted(set(x)),interval[1]])
        res=next((f['metrics']['resolution_mm'] for f in ctx['facts'] if f['pred']=='coordinate_resolution' and f['args']==[a['var']]),spec['facts']['boundary_resolution_mm'])
        if min(distances)<res:raise ValueError('Densify spacing below coordinate resolution')
    if op in FRONTIER or op=='recalibrate_reference':
        condition=a.get('condition',{})
        for cell in cells:
            if cell.get('actions'):raise ValueError('Role determines actions for this operation')
            settings=cell['settings']
            if 'start_mm' in a and settings.get(a['var'])!=a['start_mm']:raise ValueError('Frontier probe differs from proposed start')
            if 'interval' in a and not a['interval'][0]<=settings.get(a['var'],float('inf'))<=a['interval'][1]:raise ValueError('Expert start outside frontier gap')
            if op=='recalibrate_reference' and any(settings.get(k,0)!=0 for k in continuous(spec)):raise ValueError('Calibration must repeat at placement reference')
            if isinstance(condition,dict):
                for key in spec['perturbation_variables']:
                    if key in condition and any(b.get(key,settings.get(key))!=condition[key] for b in cell.get('blocks',[{}])):
                        raise ValueError('Frontier block differs from measured condition')
    if op=='collect_transitions' and 'intervals' in a:
        for k,(lo,hi) in a['intervals'].items():
            if any(not lo<=c['settings'].get(k,float('inf'))<=hi for c in cells):
                raise ValueError('Transition start must belong to the bound cluster')
    if a.get('action_design')=='crossed_sign_amplitude':
        axes=spec['task']['action_space']['dim']
        actions=[action['u'] for cell in cells for action in cell.get('actions',[])]
        for i in range(axes):
            for sign in (-1,1):
                distinct={abs(u[i]) for u in actions if np.count_nonzero(u)==1 and sign*u[i]>0}
                if len(distinct)<spec['dynamics']['min_amplitude_levels']:raise ValueError('Missing axis/sign/amplitude transition comparisons')


def rule_fields(new,rule,decision,spec):
    op=decision['operator'];args=decision['args'];premises=decision['premises']
    cells=(decision.get('plan') or {}).get('cells',[])
    if op=='set_allocation':
        for k in ('allocation','start_points','target_consumer'):new[k]=deepcopy(args[k])
    elif cells and op!='validate_B':
        role={'direct_baseline':'direct','policy_frontier':'policy','expert_recovery':'expert'}.get(args.get('role'))
        points=[]
        for cell in cells:
            point={k:x for k,x in cell['settings'].items() if k in continuous(spec)}
            if not point:point={'basis':'placement_reference' if spec.get('current_placement') else 'unspecified'}
            if point not in points:points.append(point)
        new['start_points']=points
        totals=dict(direct=0,policy=0,expert=0)
        for cell in cells:
            sampled=role or ('exploration' if cell.get('actions') else 'direct')
            totals[sampled]=totals.get(sampled,0)+cell['reps']*len(cell.get('blocks',[{}]))
        new['allocation']=totals
        if op in ('expand_range','activate_variable','densify','probe_recovery_frontier','request_expert_recovery'):
            var=args.get('var')
            if var in continuous(spec) and any(var in pt for pt in points):
                bounds=new['variables'][var]['range']
                values=[p[var] for p in points if var in p]
                new['variables'][var]['range']=[min(bounds[0],min(values)),max(bounds[1],max(values))]
    if op=='select_representation':new['representation']=args['representation']
    directional={}
    for name,decl in new.get('variables',{}).items():
        if 'range' not in decl:continue
        lo,hi=decl['range'];origin=spec['perturbation_variables'][name].get('reference',0)
        directional[name]={'negative':None if lo>origin else [lo,min(origin,hi)],
                           'positive':None if hi<origin else [max(origin,lo),hi]}
    new['directional_ranges']=directional
    prior=new.setdefault('field_provenance',deepcopy(rule.get('field_provenance',{})))
    def track(key,value,old):
        if key not in prior and value==old and '.' in key:
            parent=key.rsplit('.',1)[0]
            if parent in prior: prior[key]=deepcopy(prior[parent])
        if key not in prior or value!=old:
            prior[key]={'operator':op,'premises':list(premises),'fact_ids':list(premises)}
        if isinstance(value,dict):
            for k,v in value.items(): track(key+'.'+k,v,old.get(k) if isinstance(old,dict) else None)
    for k,value in list(new.items()):
        if k in ('field_provenance','reason','rule_id','parent_rule_id','evidence_refs'):continue
        track(k,value,rule.get(k))
    return new
