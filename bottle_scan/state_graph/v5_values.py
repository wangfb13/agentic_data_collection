"""Finite-action-budget values, witnessed action transfer and calibrated metric radii."""
from collections import defaultdict
import numpy as np


def resolution(rows, observed, D, spec, metric_mm=None):
    cfg=spec['graph']
    indices={r['node_id']:i for i,r in enumerate(rows)}
    by_axis={}
    for k,name in enumerate(spec['task']['action_space']['names']):
        samples=[]
        for edge in observed:
            u=edge.get('u')
            if not edge.get('executable',True) or u is None or np.count_nonzero(u)!=1 or u[k]==0:
                continue
            d=D[indices[edge['source']],indices[edge['target']]]
            if np.isfinite(d):
                m=None if metric_mm is None else metric_mm[indices[edge['source']],indices[edge['target']]]
                samples.append((abs(u[k]),float(d),edge['source'],edge['target'],m))
        smallest=min((s[0] for s in samples),default=None)
        selected=[s for s in samples if s[0]<=smallest*(1+cfg['min_action_bin_relative_tolerance'])]
        enough=len(selected)>=cfg['resolution_min_edges']
        scale=float(np.median([s[1] for s in selected])) if enough else None
        by_axis[name]={'metric_scale_mm':float(np.median([s[4] for s in selected])) if enough and metric_mm is not None else None,'minimum_amplitude_mm':smallest,'n':len(selected),'scale':scale,
                       'edges':[[s[2],s[3]] for s in selected],'status':'measured' if enough else 'missing'}
    # Generic distances have no candidate action axis. Use the tightest resolved
    # axis; use the v4 radius for an axis with insufficient action evidence.
    radii={}
    owners={'similar_radius':'graph','ready_radius':'sinks','radius':'clusters',
            'support_distance':'runtime','endpoint_radius':'dynamics'}
    for key,owner in owners.items():
        fraction=spec[owner][key]
        if not np.isfinite(fraction) or fraction<=0:raise ValueError(owner+'.'+key+' must be positive')
        radii[key]=min([1.]+[fraction*s['scale'] for s in by_axis.values() if s['scale'] is not None])
    return {'by_axis':by_axis,'missing_axes':[k for k,v in by_axis.items() if v['scale'] is None],
            'aggregation':'minimum_available_axis_radius; missing_axis_radius=1'},radii


def solve(ids, edges, probabilities, D, radius, spec, include_inferred):
    idx={x:i for i,x in enumerate(ids)}
    cfg=spec['graph']
    borrow=cfg.get('borrow_similar_actions',False) and cfg.get('resolution_scaled_radius',False)
    discount=1-float(cfg['step_penalty'])
    borrow_discount=1-float(cfg['similar_penalty'])
    if not (0<discount<=1 and 0<borrow_discount<=1 and 0<=cfg['recover_p_min']<=1):
        raise ValueError('Invalid bounded-success discount or minimum')
    # Keep explicit predecessors for each finite horizon. Similarity alone can
    # never hand off, consume a step, or be followed by another similarity.
    outgoing=defaultdict(list)
    for e in sorted(edges,key=lambda e:e['edge_id']):
        if not e.get('executable',True) or e['kind'] not in (('observed','inferred') if include_inferred else ('observed',)):
            continue
        penalty=e['cost']-spec['graph'].get('step_cost',1.)
        outgoing[e['source']].append((e,max(0.,discount-max(0.,penalty))))
    neighbors={x:[ids[j] for j in np.flatnonzero(D[idx[x]]<=radius) if j!=idx[x]] for x in ids} if borrow else {}
    handoff={x:p['p_hat'] if p['ready'] else 0. for x,p in probabilities.items()}
    values=handoff.copy()
    paths={x:[] for x in ids}
    for _ in range(spec['runtime']['max_steps']):
        direct={};measured={}
        for x,options in outgoing.items():
            for edge,factor in options:
                score=factor*values[edge['target']]
                if edge['kind']=='observed' and score>measured.get(x,(0,None))[0]:measured[x]=(score,edge)
                if score>direct.get(x,(0,None))[0]:
                    direct[x]=(score,edge)
        next_values=values.copy()
        next_paths=paths.copy()
        for x in ids:
            own=direct.get(x)
            choices=[(own[0],own[1],False)] if own else []
            choices.extend((measured[n][0]*borrow_discount,measured[n][1],True)
                           for n in neighbors.get(x,[]) if n in measured)
            # A model edge can be the best direct candidate, while only the
            # measured action is eligible to transfer. In the deployment pass
            # only measured edges are in direct.
            for score,edge,borrowed in choices:
                if score<=next_values[x]:continue
                step={'state':x,'demonstrator':edge['source'],'target':edge['target'],'edge_id':edge['edge_id'],
                      'action_source':'borrowed' if borrowed else edge['kind'],'action':edge['action']}
                next_values[x]=score
                next_paths[x]=[step,*paths[edge['target']]]
        if values==next_values:break
        values,paths=next_values,next_paths
    return values,paths
