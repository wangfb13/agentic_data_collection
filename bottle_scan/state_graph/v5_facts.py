"""Measured spatial evidence and program-derived symbolic predicates."""
from collections import defaultdict
import math
import numpy as np
from scipy.stats import mannwhitneyu
from .common import digest, group_id, independent


def configure(spec):
    cfg=spec.get('facts',{})
    tol=cfg.get('boundary_cross_tolerance_mm')
    if cfg.get('stratified_boundary') and (tol is None or isinstance(tol,dict)):
        specified=tol or {}
        cfg['boundary_cross_tolerance_mm']={k:specified.get(k,float(v.get('nominal_step',cfg['observability_step_mm']))/2)
            for k,v in spec['perturbation_variables'].items() if v['kind']=='continuous'}
    return spec


def _fact(pred,args,rows,metrics=None):
    from .facts import fact
    return fact(pred,args,[r['node_id'] if isinstance(r,dict) else r for r in rows],metrics)


def number(x):
    return type(x) in (float,int) and math.isfinite(x)


def coordinate(row,var,allow_estimate=False):
    """Return value and source. Explicit invalidation overrides all old coordinate labels."""
    from .facts import legacy_coordinate
    p=row.get('provenance') or {}
    schema=p.get('coordinate_schema','')
    value=(row.get('plan_coords') or {}).get(var)
    if p.get('placement_reference_valid') is False or p.get('measurement_valid') is False:
        return None,None
    if schema=='measured_placement_relative':
        if (p.get('placement_reference_id') and p.get('placement_reference_valid') is True
                and p.get('coordinate_source')=='measured' and number(value)):
            return value,'measured'
    elif schema=='fixed_contact_a_v1':
        x=legacy_coordinate(row,var)
        if x is not None:return x,'measured'
    elif schema=='measured_placement_relative_ah_v1' and p.get('measured_pose') is not None and number(value):
        return value,'measured'
    # Archived independent pose labels may be imported with a documented legacy source.
    elif p.get('coordinate_source')=='measured' and p.get('measurement_valid') is True and p.get('evidence_id') and number(value):
        return value,'measured'
    if allow_estimate:
        x=(p.get('coordinate_estimate_mm') or {}).get(var)
        if number(x):return x,'estimate'
    return None,None


def position_related(row,cid,spec):
    post=row.get('post',{}).get(cid,{})
    if not spec.get('consumer',{}).get('fixed_force_failure_causes'):return True
    return post.get('failure_cause',post.get('reason',row.get('provenance',{}).get('failure_cause')))!='closure_floor_reached'


def continuous(spec):
    return [k for k,v in spec['perturbation_variables'].items() if v['kind']=='continuous']


def resolution(rows,spec,var):
    cfg=spec['facts'];samples=defaultdict(dict)
    for r in rows:
        if r.get('window_index')!=0 or independent(r):continue
        x,_=coordinate(r,var)
        key=r.get('repeat_key')
        if x is None or not key or var not in (r.get('repeat_settings') or {}):continue
        # Do not pool across placement witnesses or sensor epochs.
        key=(key,r.get('provenance',{}).get('placement_reference_id'),r.get('session'))
        samples[key][group_id(r)]=x
    ss=df=0
    for group in samples.values():
        v=list(group.values())
        if len(v)>1:
            ss+=sum((x-np.mean(v))**2 for x in v);df+=len(v)-1
    sd=float(np.sqrt(ss/df)) if df else None
    return {'coordinate_source':'measured','noise_mm':sd,'repeat_df':df,
            'resolution_mm':max(cfg['boundary_resolution_mm'],cfg['boundary_noise_multiplier']*(sd or 0.))}


def rank_test(rows,cid,var,source,spec):
    grouped=defaultdict(list)
    for row in rows:
        if independent(row) or not position_related(row,cid,spec):continue
        p=row.get('post',{}).get(cid,{})
        if not row.get('is_terminal') or not p.get('evaluated') or type(p.get('success')) is not bool:continue
        if source=='estimate':
            x=(row.get('provenance',{}).get('coordinate_estimate_mm') or {}).get(var)
        else:x=coordinate(row,var)[0]
        if number(x):grouped[group_id(row)].append((row,x,p['success']))
    # A trial cannot vote for both outcomes in one rank test.
    observations=[(v[0][0]['node_id'],float(np.median([p[1] for p in v])),v[0][2])
                  for v in grouped.values() if len({p[2] for p in v})==1]
    a=[x for _,x,y in observations if y];b=[x for _,x,y in observations if not y]
    if min(len(a),len(b))<spec['facts']['min_axis_trials']:return None
    return {'var':var,'interval':[min(a+b),max(a+b)],'coordinate_source':source,'n_success':len(a),'n_failure':len(b),
            'p_raw':float(mannwhitneyu(a,b,alternative='two-sided').pvalue),'evidence':[i for i,_,_ in observations]}


def boundary_tests(members,cid,spec):
    if not spec.get('facts',{}).get('boundary_before_stochastic'):return []
    tests=[test for var in continuous(spec) for src in ('measured','estimate')
           if (test:=rank_test(members,cid,var,src,spec))]
    for t in tests:t['p_adjusted']=min(1.,len(tests)*t['p_raw'])
    return [t for t in tests if t['p_adjusted']<spec['facts']['alpha']]


def spatial_facts(rows,spec,cid):
    cfg=spec['facts'];result=[]
    variables=continuous(spec)
    actual=[r for r in rows if r.get('is_terminal') and r.get('post',{}).get(cid,{}).get('evaluated') and
            type(r['post'][cid].get('success')) is bool and position_related(r,cid,spec)]
    def position(row,var):
        if cfg.get('measured_placement_coordinates'):return coordinate(row,var)[0]
        from .facts import legacy_coordinate
        return legacy_coordinate(row,var)
    coordinate_source='measured' if cfg.get('measured_placement_coordinates') else 'legacy'

    def compatible(a,b,var):
        if not cfg.get('stratified_boundary'):return True
        for k in variables:
            if k==var or not spec['perturbation_variables'][k].get('active',True):continue
            x=position(a,k);y=position(b,k)
            setting=cfg['boundary_cross_tolerance_mm']
            tol=setting.get(k) if isinstance(setting,dict) else setting
            if tol is None:tol=spec['perturbation_variables'][k].get('nominal_step',cfg['observability_step_mm'])/2
            if x is None or y is None or abs(x-y)>tol:return False
        return True
    for var in variables:
        definition=spec['perturbation_variables'][var]
        data=[(r,x) for r in actual if (x:=position(r,var)) is not None]
        res=resolution(rows,spec,var) if cfg.get('noise_based_resolution') else {'resolution_mm':2*definition.get('nominal_step',1.),'coordinate_source':'measured'}
        res['coordinate_source']=coordinate_source
        result.append(_fact('coordinate_resolution',[var], [r for r,_ in data],res))
        for side in ('positive','negative','reference'):
            ref=definition.get('reference',0);sign={'positive':1,'negative':-1,'reference':0}[side]
            cell=[r for r,x in data if np.sign(x-ref)==sign]
            failed=[r for r in cell if r['post'][cid]['success'] is False]
            if failed:result.append(_fact('failure_present',[var,side,cid],failed,{'coordinate_source':coordinate_source}))
            if cfg.get('per_side_no_failure') and side!='reference' and cell and not failed:
                result.append(_fact('no_failure',[var,side,cid],cell,{'terminal_count':len(cell),'coordinate_source':coordinate_source}))
        # For each point retain the next strictly larger coordinate in its local cross-section.
        data.sort(key=lambda t:t[1])
        emitted=set()
        for i,(a,x) in enumerate(data):
            nearest=None
            for b,y in data[i+1:]:
                if nearest is not None and y>nearest:break
                if y==x or not compatible(a,b,var):continue
                nearest=y
                if y-x<=res['resolution_mm'] or a['post'][cid]['success']==b['post'][cid]['success']:continue
                key=(var,x,y)
                if key not in emitted:
                    result.append(_fact('boundary_gap',[var,[x,y],cid],[a,b],{**res,'gap':y-x}));emitted.add(key)
    return result


def axis_facts(rows,graph,noise,spec,cid):
    from .noise import transform
    cfg=spec['facts'];result=[];byid={r['node_id']:r for r in rows}
    axes=spec['task']['action_space'].get('continuous_displacement_variables') or continuous(spec)[:spec['task']['action_space']['dim']]
    direct=[r for r in rows if r.get('provenance',{}).get('direct_A') or
            r.get('data_role')=='frontier_direct' or r.get('provenance',{}).get('controller_role')=='direct']
    signals=[n for n in cfg['tactile_modalities'] if noise.get('modalities',{}).get(n,{}).get('status')=='ready']
    matrices={}
    for n in signals:
        cov=np.asarray(noise['modalities'][n]['single_covariance'])
        eig,vec=np.linalg.eigh(cov);matrices[n]=vec@np.diag(1/np.sqrt(np.maximum(eig,np.finfo(float).eps)))@vec.T
    test_count=len(axes)
    for i,var in enumerate(axes):
        if var not in spec['perturbation_variables']:continue
        data=[]
        for edge in graph['edges']:
            if edge['kind']!='observed' or edge['source'] not in byid or edge['target'] not in byid or not signals:continue
            receipt=edge.get('receipt') or {}
            if receipt.get('applied') is not True:continue
            measured=receipt.get('net_contact_displacement_mm',byid[edge['target']].get('u_executed'))
            u=np.asarray(measured,float)
            # Pure-axis command identity, with independent measured cross-axis feedback allowed.
            designated=(edge.get('action') or {}).get('u') or edge.get('u')
            command=np.asarray(designated,float)
            if (command.shape!=(len(axes),) or np.count_nonzero(command)!=1 or not command[i] or
                    u.shape!=(len(axes),) or not np.isfinite(u).all() or not u[i]):continue
            a,b=byid[edge['source']],byid[edge['target']]
            delta=np.concatenate([(transform(b,n,noise)-transform(a,n,noise))@matrices[n] for n in signals])
            data.append((group_id(a),u[i],delta,edge['edge_id']))
        groups=sorted({d[0] for d in data})
        if spec.get('graph',{}).get('observable_axes_only'):
            from .axis_resolution import axis_observable
            from .noise import pairwise_distances
            name=spec['task']['action_space']['names'][i]
            stats=graph.get('resolution',{}).get('by_axis',{}).get(name)
            if (stats is None or 'observable' not in stats) and not noise.get('active_modalities'):
                stats={'observable':None}
            if stats is None or 'observable' not in stats:
                obs=[e for e in graph['edges'] if e['kind']=='observed']
                stats=axis_observable(rows,obs,pairwise_distances(rows,noise),spec,i)
            if stats['observable'] is not None:
                result.append(_fact('axis_observable' if stats['observable'] else 'axis_unobservable',[var],
                    [nid for pair in stats['evidence'] for nid in pair], dict(stats,coordinate_source='measured_action')))
        elif len(groups)>=cfg['min_axis_trials']:
            def sensitivity(ds):
                x=np.array([t[1] for t in ds]);z=np.array([t[2] for t in ds])
                return float(np.linalg.norm(x@z/(x@x)))
            # Origin-constrained linear response; bootstrap whole trials rather than individual steps.
            rng=np.random.default_rng(cfg['seed']);boot=[sensitivity([d for g in rng.choice(groups,len(groups)) for d in data if d[0]==g])
                for _ in range(cfg['bootstrap_reps'])]
            value=sensitivity(data);q=(1-cfg['confidence'])/2
            step=spec['perturbation_variables'][var].get('nominal_step',cfg['observability_step_mm'])
            result.append(_fact('axis_observable' if value*step>=1 else 'axis_unobservable',[var],[d[3] for d in data],
                {'sensitivity':value,'confidence_interval':np.quantile(boot,[q,1-q]).tolist(),'step_mm':step,'n_groups':len(groups),
                 'coordinate_source':'measured_action','modalities':signals}))
        test=rank_test(direct,cid,var,'measured',spec)
        if test:
            test['p_adjusted']=min(1.,test_count*test['p_raw'])
            result.append(_fact('axis_consequential' if test['p_adjusted']<cfg['alpha'] else 'axis_inconsequential',
                                [var,cid],test['evidence'],test))
    return result


def revise(base,graph,rows,labels,clusters,noise,spec,models):
    cfg=spec.get('facts',{});ops=spec.get('operators',{})
    strict=cfg.get('measured_placement_coordinates') or cfg.get('per_side_no_failure') or cfg.get('stratified_boundary') or cfg.get('noise_based_resolution')
    skip={'failure_present','boundary_gap'} if strict else set()
    if cfg.get('per_side_no_failure'):skip.add('no_failure')
    if ops.get('handle_drift_and_unsupported'):skip.add('unsupported')
    if spec.get('labels',{}).get('include_inferred') is False:skip.add('inferred_edge_unverified')
    facts=[f for f in base if f['pred'] not in skip];byid={r['node_id']:r for r in rows}
    if cfg.get('axis_balance'):
        from .axis_resolution import balance_facts
        facts.extend(balance_facts(graph,spec))
    for cid,calibration in graph.get('ready_calibration',{}).items():
        if calibration['threshold'] is None:
            facts.append(_fact('ready_threshold_unattainable',[cid],rows,calibration))
    for f in facts:
        if strict and f['pred']=='coverage' and f['args'][0] in continuous(spec):f['metrics']['coordinate_source']='measured' if cfg.get('measured_placement_coordinates') else 'legacy'
    active = spec['active_post_condition']
    if cfg.get('amplitude_coverage') or spec.get('director', {}).get('score_based_axis_choice'):
        from .amplitude_coverage import coverage_facts
        spatial = coverage_facts(rows, graph, spec, active)
        facts.extend(f for f in spatial if cfg.get('amplitude_coverage') or f['pred'] != 'amplitude_gap')
    for condition in spec['post_conditions']:
        cid=condition['id']
        if spec.get('consumer',{}).get('auto_l2'):
            for r in rows:
                result=r.get('post',{}).get(cid,{})
                if result.get('label_conflict') is True:
                    facts.append(_fact('label_conflict',[r['node_id'],cid],[r],{'automatic_L2':result.get('automatic_L2'),'audit':result.get('audit')}))
        if strict:facts.extend(spatial_facts(rows,spec,cid))
        if cfg.get('capability_predicates'):facts.extend(axis_facts(rows,graph,noise,spec,cid))
        for c in clusters['clusters']:
            ids=c['node_ids'];members=[byid[k] for k in ids];cluster=c['cluster_id']
            bounds=boundary_tests(members,cid,spec)
            if bounds:
                facts=[f for f in facts if not (f['pred']=='stochastic_outcome' and f['args']==[cluster,cid])]
                for t in bounds:facts.append(_fact('boundary_within_cluster',[cluster,t['var'],cid],t['evidence'],t))
            if ops.get('handle_drift_and_unsupported'):
                statuses=graph['reachability'][cid]
                if all(statuses[k]['status']=='ready' for k in ids):continue
                if any(e['kind']=='observed' and e['source'] in ids for e in graph['edges']):continue
                # Check borrowable observed moves, even when none reaches a Ready sink.
                outgoing={e['source'] for e in graph['edges'] if e['kind']=='observed' and e.get('executable',True)}
                if spec.get('graph',{}).get('borrow_similar_actions') and spec.get('graph',{}).get('resolution_scaled_radius') and any(
                    e['kind']=='similar' and ((e['source'] in ids and e['target'] in outgoing) or (e['target'] in ids and e['source'] in outgoing)) for e in graph['edges']):continue
                intervals={}
                for var in continuous(spec):
                    xs=[x for r in members if (x:=coordinate(r,var)[0]) is not None]
                    if xs:intervals[var]=[min(xs),max(xs)]
                facts.append(_fact('recovery_evidence_missing',[cluster,cid],members,
                             {'coordinate_source':'measured','intervals':intervals}))
        if spec.get('labels',{}).get('include_inferred') is False:
            inferred={i for state in graph.get('model_reachability',{}).get(cid,{}).values()
                      if state.get('status')=='model_recoverable' for i in state.get('inferred_edge_ids',[])}
            es=[e for e in graph['edges'] if e['edge_id'] in inferred and e['kind']=='inferred']
            if es:
                e=max(es,key=lambda x:x['uncertainty'])
                facts.append(_fact('inferred_edge_unverified',[len(es),cid],[x['source'] for x in es],
                                   {'source_node_id':e['source'],'edge_id':e['edge_id'],'uncertainty':e['uncertainty'],'u':e['u'],'path_fraction':1.}))
    missing=graph.get('resolution',{}).get('missing_axes',[])
    if missing:facts.append(_fact('resolution_scale_missing',missing,[],{'axes':missing}))
    if models:
        dynamics=models['dynamics'];ablation=dynamics.get('action_ablation') or {}
        if ops.get('decouple_dynamics') and dynamics.get('status')!='trained':
            facts.append(_fact('dynamics_unavailable',[],dynamics.get('fit_node_ids',[]),{'status':dynamics.get('status')}))
        gain=ablation.get('relative_gain')
        if cfg.get('capability_predicates') and number(gain) and gain<spec['dynamics']['min_action_gain']:
            facts.append(_fact('action_effect_unidentified',[],dynamics.get('fit_node_ids',[]),{'relative_gain':gain,'min_gain':spec['dynamics']['min_action_gain']}))
    if spec.get('director',{}).get('operator_scoring'):
        cid=spec['active_post_condition'];a,b=next(c.get('p_prior',[1,1]) for c in spec['post_conditions'] if c['id']==cid)
        p=graph['probabilities'][cid]
        variance={k:(v['successes']+a)*(v['n']-v['successes']+b)/((v['n']+a+b)**2*(v['n']+a+b+1)) for k,v in p.items()}
        spatial={r['node_id']:{'coords':{v:x for v in continuous(spec) if (x:=coordinate(r,v)[0]) is not None},
                    'attrs':{k:r.get('attrs',{}).get(k) for k,d in spec['perturbation_variables'].items() if d['kind']=='object_property'},
                    'trial_id':r.get('trial_id')} for r in rows}
        facts.append(_fact('posterior_variance',[cid],list(variance),{'by_node':variance,'locations':spatial,'coordinate_source':'measured'}))
    ranking=spec.get('observation',{}).get('ranking_report')
    if ranking and spec.get('observation',{}).get('auto_select'):
        from pathlib import Path
        import json
        from .representation_eval import director_evidence
        report=json.loads(Path(ranking).read_text()) if isinstance(ranking,str) else ranking
        ranking_facts,_=director_evidence(report,spec)
        facts.extend(ranking_facts)
    return facts



def episode_frontiers(episodes,spec,cid,**kwargs):
    from .frontier_facts import episode_frontiers as calculate
    return calculate(episodes,spec,cid,**kwargs)


def phase_facts(state,spec):
    phase=state.get('phase') or 'plan'
    if state.get('frozen_independent'):phase='independent'
    return [_fact('phase_'+phase,[spec['active_post_condition']],[],{'phase':phase})]
