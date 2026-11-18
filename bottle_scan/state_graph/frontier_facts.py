"""Signed, condition-specific observed frontiers from first-contact episode evidence."""
from collections import defaultdict
import json
from .common import digest
import numpy as np
from .v5_facts import _fact,coordinate,continuous,number,resolution

ROLES={'frontier_direct':'direct','frontier_policy':'policy','frontier_expert':'expert',
       'direct_baseline':'direct','policy_frontier':'policy','expert_recovery':'expert'}
PRED={'direct':'direct_boundary','policy':'policy_frontier','expert':'expert_frontier'}


def records(episodes,spec,cid):
    seen=set()
    for e in episodes:
        if e.get('post_freeze') or e.get('evaluation_only') or e.get('data_role')=='independent_validation' or e.get('role')=='independent':continue
        ident=e.get('attempt_id') or e.get('trial_id')
        if not ident or ident in seen:continue
        seen.add(ident)
        role=e.get('controller_role') or e.get('role')
        role=ROLES.get(role,ROLES.get(e.get('data_role'),role))
        if role not in PRED:continue
        first=e.get('first_contact')
        ev=e.get('initial_coordinate_evidence') or {}
        if first and first.get('window_index')!=0:first=None
        coords={}
        for var in continuous(spec):
            x=coordinate(first,var)[0] if first else None
            if (x is None and ev.get('coordinate_source',ev.get('source'))=='measured' and
                    ev.get('valid',ev.get('placement_reference_valid')) is True and ev.get('evidence_id')
                    and ev.get('stage','first_contact')=='first_contact'):
                x=(e.get('initial_coords') or {}).get(var)
            if number(x):coords[var]=x
        outcome=(e.get('terminal_outcomes') or e.get('outcomes') or {}).get(cid,{})
        if not outcome and e.get('target_consumer')==cid:
            outcome=e.get('outcome') or {'evaluated':e.get('evaluated'),'success':e.get('success')}
        if not isinstance(outcome,dict):continue
        success=outcome.get('success') if outcome.get('evaluated') else None
        cause=outcome.get('failure_cause') or outcome.get('reason') or e.get('failure_cause') or e.get('failure_reason') or e.get('failure_category')
        if e.get('object_moved') or e.get('initial_coordinate_valid') is False:coords={}
        if e.get('recovery_completed') is False:success=False
        if role=='policy' and (e.get('expert_source') or e.get('human_intervention')):continue
        if role=='expert' and success is True:
            receipts=e.get('receipts',[])+[r.get('u_receipt') for r in e.get('nodes',[]) if r.get('u_receipt')]
            if not e.get('expert_source') or not any(r and r.get('applied') for r in receipts):continue
        if cause=='closure_floor_reached' and spec.get('consumer',{}).get('fixed_force_failure_causes'):success=None
        condition=e.get('condition') or e.get('block') or (first or {}).get('attrs',{}).get('mass_class') or {}
        yield ident,role,coords,success,cause,condition,e.get('model_id')


def episode_frontiers(episodes,spec,cid,*,rows=(),endpoint_rows=None,model_id=None):
    if not spec.get('facts',{}).get('capability_predicates'):return []
    grouped=defaultdict(list);conditions={};unlocated_failures=defaultdict(set)
    episodes=list(episodes)
    group_for={e.get('attempt_id') or e.get('trial_id'):e.get('group_id') or e.get('attempt_id') or e.get('trial_id') for e in episodes}
    terminal_based = spec['facts'].get('terminal_based_direct_boundary')
    sources = {}
    if terminal_based:
        from .amplitude_coverage import terminal_samples
        for sample in terminal_samples(rows if endpoint_rows is None else endpoint_rows, spec, cid):
            ident = sample['node_id']
            group_for[ident] = sample['group']
            sources[ident] = sample['source']
            key = json.dumps(sample['condition'], sort_keys=True)
            conditions[key] = sample['condition']
            for var, x in sample['coords'].items():
                for side, sign in (('positive', 1), ('negative', -1)):
                    if sign*x >= 0:
                        grouped[key,var,side,'direct',None].append((ident,x,sample['success'],sample['cause']))
    cumulative=defaultdict(dict)
    for ident,role,coords,success,cause,condition,version in records(episodes,spec,cid):
        if success is False:
            for var,x in coords.items():
                for side,sign in [('positive',1),('negative',-1)]:
                    if sign*x>=0:cumulative[json.dumps(condition,sort_keys=True),var,side][ident]=(x,cause)
        if model_id is not None and role=='policy' and version!=model_id:continue
        k=json.dumps(condition,sort_keys=True)
        conditions[k]=condition
        if success is False:
            unlocated_failures[k].update(v for v in continuous(spec) if v not in coords)
        if terminal_based and role=='direct':
            continue
        for var,x in coords.items():
            for side,sign in (('positive',1),('negative',-1)):
                if x*sign>=0:grouped[k,var,side,role,version if role=='policy' else None].append((ident,x,success,cause))
    facts=[];cfg=spec['facts']
    if not conditions:conditions['{}']={}
    for key,condition in conditions.items():
        for var in continuous(spec):
            definition=spec['perturbation_variables'][var]
            if not definition.get('active',True):continue
            for side,sign in (('positive',1),('negative',-1)):
                envelope=definition['envelope'][1 if sign==1 else 0]
                if sign*envelope<=0:continue
                common={'coordinate_source':'measured','condition':condition,'boundary_definition':'farthest_observed_success',
                        'confidence':cfg['confidence'],'resolution_mm':resolution(rows,spec,var)['resolution_mm']}
                stop=common['resolution_mm']
                versions={t[4] for t in grouped if t[:4]==(key,var,side,'policy')} or {model_id}
                for version in sorted(versions,key=str):
                    limits={};tested={};evidence=[];all_failed=[];actual=[]
                    stats_by_role={}
                    for role,name in PRED.items():
                        cell=grouped[key,var,side,role,version if role=='policy' else None]
                        xs=[r[1] for r in cell if r[2] is True]
                        far=max(xs,key=lambda x:x*sign) if xs else None
                        limits[role]=far
                        observed=[r for r in cell if type(r[2]) is bool]
                        tested[role]=sum(sign*r[1]>0 for r in observed)
                        actual.extend(observed)
                        all_failed.extend(r[3] for r in observed if not r[2])
                        evidence.extend(r[0] for r in cell)
                        ci=None
                        if xs:
                            rng=np.random.default_rng(cfg['seed'])
                            maxima=[max([r[1] for i in indexes if (r:=observed[i])[2]],key=lambda x:x*sign,default=np.nan)
                                    for indexes in rng.integers(len(observed),size=(cfg['bootstrap_reps'],len(observed)))]
                            ci=np.nanquantile(maxima,[(1-cfg['confidence'])/2,(1+cfg['confidence'])/2]).tolist()
                        metrics={**common,'farthest_observed_success':far,'frontier_ci_mm':ci,
                                 'ci_method':'episode bootstrap of observed successes; conditional on a bootstrap success',
                                 'trials':len(cell),'evaluated_trials':len(observed),'n_success':len(xs),'directional_evaluated_trials':tested[role],'model_id':version,
                                 'observed_points':[{'x':r[1],'success':r[2],'failure_cause':r[3],'trial_id':r[0]} for r in cell]}
                        if terminal_based and role=='direct':
                            from .amplitude_coverage import source_counts
                            metrics.update(coordinate_stage='consumer_terminal',
                                terminal_sources=source_counts({'source':sources[r[0]]} for r in observed))
                        if cfg.get('rate_based_frontier'):
                            from .frontier_rates import statistics
                            stats=statistics([(group_for[i],x,ok,cause) for i,x,ok,cause in observed],sign,stop,abs(envelope),cfg)
                            metrics.update(stats);limits[role]=stats['verified_edge_mm'];stats_by_role[role]=stats
                            if role=='policy':
                                k=stats['boundary_bin']
                                stats['adjacent_failures']={i:c for i,(x,c) in cumulative[key,var,side].items() if k is not None and max(0,int(np.ceil(abs(x)/stop))-1) in (k,k+1)}
                                metrics['adjacent_failures']=stats['adjacent_failures']
                        facts.append(_fact(name,[var,side,cid],[r[0] for r in cell],metrics))
                    expert,policy=limits['expert'],limits['policy']
                    physical=var not in unlocated_failures[key] and bool(all_failed) and all(c in ('contact_lost','edge_contact','contact_edge') for c in all_failed)
                    gap=sign*(expert-policy) if None not in (expert,policy) else None
                    if cfg.get('rate_based_frontier'):common['boundary_definition']='connected_wilson_bins'
                    if gap is not None and gap>stop:
                        facts.append(_fact('frontier_gap',[var,side,cid],evidence,
                                 {**common,'model_id':version,'interval':sorted([expert,policy]),'length_mm':gap}))
                    converged=gap is not None and abs(gap)<=stop and tested['expert'] and tested['policy']
                    if cfg.get('rate_based_frontier'):
                        from .frontier_rates import overlap
                        converged=converged and overlap(stats_by_role['policy'],stats_by_role['expert'])
                        near=stats_by_role['policy']['adjacent_failures']
                        physical=len(near)>=cfg['frontier_min_trials'] and sum(c in ('contact_lost','edge_contact','contact_edge') for c in near.values())/len(near)>=cfg['physical_failure_fraction']
                        common['boundary_failures']=near
                    if physical or converged:
                        facts.append(_fact('frontier_direction_closed',[var,side,cid],evidence,
                                  {**common,'reason':'physical_failures_observed' if physical else 'within_resolution','model_id':version}))
    # Conditions and policy versions are separate predicates even for an as-yet unsampled cell.
    for f in facts:
        scope=f['metrics']
        f['id']='F_'+digest([f['pred'],f['args'],scope['condition'],scope.get('model_id'),f['evidence']])[:18]
    return facts

