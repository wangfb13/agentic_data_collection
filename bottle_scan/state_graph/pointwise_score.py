"""Coordinate-domain H1 Gaussian/Beta estimator for selecting a physical sample."""
from itertools import product
from math import ceil, isfinite, sqrt
from .amplitude_coverage import terminal_samples
from .repeat_evidence import scales
from .ready_kernel import gaussian_weight

PREREQUISITES={'collect_repeat_block','recalibrate_reference'}


def evidence(rows, graph, spec):
    dual=spec.get('consumer',{}).get('dual_paired')
    return dict(widths=scales(rows,spec), consumers={
        c['id']:dict(samples=list(terminal_samples(rows,spec,c['id'])),
            bandwidth=(graph.get('ready_calibration',{}).get(c['id']) or {}).get('h_ready')
                      or spec['sinks']['ready_bandwidths'][0], prior=c.get('p_prior',[1,1]))
        for c in spec['post_conditions'] if dual or c['id']==spec['active_post_condition']})


def point_stats(point, condition, widths, data):
    groups={}
    for s in data['samples']:
        if any(s['condition'].get(k)!=v for k,v in condition.items()) or not point.keys()<=s['coords'].keys():
            continue
        d=sqrt(sum(((x-s['coords'][k])/widths[k])**2 for k,x in point.items()))
        old,ok=groups.get(s['group'],(float('inf'),True))
        groups[s['group']]=(min(d,old),ok and s['success'])
    a,b=data['prior']
    n=0.
    for d,ok in groups.values():
        w=float(gaussian_weight(d,data['bandwidth']))
        n+=w
        if ok:a+=w
        else:b+=w
    variance=a*b/((a+b)**2*(a+b+1))
    return dict(variance=variance,n_eff=n,score=variance/(1+n))


def score(ctx, answer):
    op=answer['operator']
    cells=(answer.get('plan') or {}).get('cells',[])
    n=sum(c['reps']*len(c.get('blocks',[{}])) for c in cells)
    if op in PREREQUISITES or not cells:
        return dict(score=None,prerequisite=op in PREREQUISITES,planned_trials=n)
    data=ctx['point_evidence']
    widths=data['widths']
    axes=list(widths)
    cfg=ctx['spec']['director']
    cost=cfg['trial_costs'][op]
    if not isfinite(cost) or cost<=0:raise ValueError('Invalid per-trial cost for '+op)
    bindings=answer['args']
    ranges=dict(bindings.get('intervals',{}))
    if 'interval' in bindings and bindings.get('var'):ranges[bindings['var']]=bindings['interval']
    best=None
    for cell in cells:
        endpoint={k:cell['settings'].get(k,ctx['spec']['perturbation_variables'][k].get('reference',0.)) for k in axes}
        for act in cell.get('actions',[]):
            for k,u in zip(axes,act.get('u',[])):endpoint[k]+=u
        grids=[]
        for k in axes:
            if k not in ranges:grids.append([endpoint[k]]);continue
            lo,hi=ranges[k]
            grid=[min(hi,lo+i*widths[k]) for i in range(ceil((hi-lo)/widths[k])+1)]
            grids.append(grid)
        for position in product(*grids):
            point=dict(zip(axes,position))
            for condition in cell.get('blocks',[{}]):
                by={cid:point_stats(point,condition,widths,c) for cid,c in data['consumers'].items()}
                score=sum(m['score'] for m in by.values())/cost
                if best is None or score>best['score']:
                    best=dict(score=score,target_point=point,condition=condition,per_consumer=by)
    return dict(**best,per_trial_cost=cost,planned_trials=n)
