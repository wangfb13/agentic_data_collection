"""Budget-free candidate screening and optimistic stopping for development B only."""
from math import dist
from .facts import fact
from .v5_facts import coordinate
from .rate_gates import bounds, minimum_counts
from .round_report import success_interval


def screen(rows, graph, starts, spec, model_id):
    cid = spec['active_post_condition']
    reach = graph['reachability'][cid]
    ready = sum(v['status']=='ready' for v in reach.values())
    axes = spec['task']['action_space'].get('continuous_displacement_variables') or [
        k for k,v in spec['perturbation_variables'].items() if v['kind']=='continuous' and v.get('active',True)]
    measured = []
    for row in rows:
        nid = row['node_id']
        if nid not in reach or not row.get('window_valid',True):
            continue
        coords = [coordinate(row,k,allow_estimate=False)[0] for k in axes]
        if axes and all(x is not None for x in coords):
            post=row.get('post',{}).get(cid,{})
            failed=post.get('evaluated') is True and post.get('success') is False
            measured.append((coords, row.get('attrs',{}), nid, reach[nid]['status'], failed))
    checks = []
    for start in starts:
        point = {k:start['settings'].get(k,spec['perturbation_variables'][k].get('reference')) for k in axes}
        choices = [(dist(list(point.values()),coords),nid,status,failed) for coords,attrs,nid,status,failed in measured
                   if None not in point.values() and all(attrs.get(k)==v for k,v in start['block'].items())]
        nearest = min((x[0] for x in choices),default=None)
        neighbors=[(nid,status,failed) for d,nid,status,failed in choices if d==nearest]
        tied=sorted((nid,status) for nid,status,_ in neighbors)
        supported = bool(tied) and all(status in ('ready','recoverable') for _,status in tied)
        checks.append(dict(point=point,condition=start['block'],nearest=tied,distance_mm=nearest,supported=supported,
                           ready_evidence=sum(status=='ready' for _,status in tied),
                           failed_without_recovery=any(failed for _,_,failed in neighbors) and not supported))
    fraction = sum(x['supported'] for x in checks)/len(checks) if checks else 0.
    loop = spec['loop']
    failed = [k for k,met in [('dev_min_ready',ready>=loop['dev_min_ready']),
                               ('dev_min_support_fraction',fraction>=loop['dev_min_support_fraction'])] if not met]
    metrics = dict(model_id=model_id,ready_nodes=ready,support_fraction=fraction,checks=checks,unmet=failed)
    if failed:
        # Preserve an unsupported start and its object condition for targeted collection.
        region = min(checks,key=lambda x:(x['supported'],not x['failed_without_recovery'],x['ready_evidence'])) if checks else {}
        metrics['region'] = region.get('point',{})
        metrics['condition'] = region.get('condition',{})
    return metrics


def capability_fact(gate,cid):
    evidence = sorted({nid for x in gate['checks'] for nid,_ in x['nearest']})
    return fact('candidate_lacks_capability',[cid],evidence,gate)


def interim(records,remaining,spec,model_id):
    from .acceptance import evaluate_B
    ac = spec['acceptance']
    cid = spec['active_post_condition']
    records = [r for r in records if r.get('role')=='development' and r.get('post_condition')==cid and
               r.get('model_id',r.get('frozen_model_id'))==model_id]
    conf = ac['rate_confidence']
    intervals = bounds(records,ac)
    best = {}
    for rate,v in intervals.items():
        # Start classes are measured only on execution. Every remaining trial
        # could be in this denominator and succeed; thus the bound is optimistic.
        upper = success_interval(v['n']+remaining,v['successes']+remaining,conf)['wilson']
        best[rate] = upper[0] if upper else 0.
    impossible = [k for k,x in best.items() if x<ac['development_B']['min_'+k]]
    cause = None
    if ac.get('futility_stop') and impossible:
        cause = 'development_futility_stop'
    elif ac.get('early_success_stop') and records:
        evaluated = [r for r in records if r.get('evaluated') is True]
        ev = bounds(evaluated,ac)
        if (all(ev[k]['n']>=n and ev[k]['wilson'][0]>=ac['development_B']['min_'+k]
                for k,n in minimum_counts(ac,'development').items()) and
                evaluate_B(records,spec,'development',model_id=model_id)['passed']):
            cause = 'development_early_success'
    return dict(reason=cause,remaining=remaining,assessed=len(records),best_case_lower=best,impossible=impossible)


def stop_pending(out,state,spec):
    from pathlib import Path
    from .campaign import read, atomic
    pending=state['pending']
    if pending.get('role')!='development' or state.get('round_stage')!='dev_validate':return False
    ac=spec['acceptance']
    if not (ac.get('futility_stop') or ac.get('early_success_stop')):return False
    ids=set(pending['trial_ids'])
    trials=[t for t in state['trials'] if t['trial_id'] in ids]
    open_trials=[t for t in trials if t['status'] not in ('complete','budget_skipped')]
    if not open_trials or any(t['status']=='started' or (Path(t['directory'])/'result.json').exists() for t in open_trials):
        return False
    completed=[t for t in trials if t['status']=='complete']
    records=[r for t in completed if (r:=read(Path(t['directory'])/'result.json').get('assessment'))]
    check=interim(records,len(open_trials),spec,state['model']['model_id'])
    if not check['reason']:return False
    check.update(used=len(completed),unrun=[t['trial_id'] for t in open_trials])
    state.setdefault('development_stops',{})[str(state['round_index'])]=check
    for t in open_trials:t.update(status='budget_skipped',reason=check['reason'])
    atomic(Path(out)/'state.json',state)
    return True
