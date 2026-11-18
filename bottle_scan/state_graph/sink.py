"""Separate empirical post-condition probabilities and zero-cost Ready sinks; direct handoff risk is separate."""
from scipy.stats import beta as beta_distribution
import math
import numpy as np


def sink_id(condition):return '__post__:'+condition


def threshold(n,successes,c,spec):
    a,b=c.get('p_prior',[1,1])
    p=float((a+successes)/(a+b+n))
    settings=spec.get('sinks',{})
    lower=None
    if settings.get('lower_bound_ready'):
        lower=float(beta_distribution.ppf(settings['ready_quantile'],a+successes,b+n-successes))
    return p,lower,n>=c.get('min_ready_trials',3) and (p if lower is None else lower)>=c.get('ready_p',.8)


def estimate_sinks(rows,D,spec,*,ready_radius=1.0,calibrations=None):
    probability={};edges=[]
    for c in spec['post_conditions']:
        cid=c['id'];alpha,beta=c.get('p_prior',[1,1]);values={}
        if spec.get('sinks',{}).get('kernel_ready'):
            from .ready_kernel import estimate
            estimated, meta = estimate(rows, D, c, spec['sinks'])
            if calibrations is not None: calibrations[cid] = meta
            for row, value in zip(rows, estimated):
                nid = row['node_id']; values[nid] = value
                if value['ready']:
                    edges.append(dict(source=nid,target=sink_id(cid),kind='sink',post_condition=cid,cost=0.,p_hat=value['p_hat'],n=value['n'],evidence=value['terminal_ids']))
            probability[cid] = values
            continue
        terminal=[i for i,r in enumerate(rows) if r.get('is_terminal') and r.get('post',{}).get(cid,{}).get('evaluated') is True]
        for i,row in enumerate(rows):
            near=[j for j in terminal if i==j or D[i,j]<=ready_radius]
            # Count independent trial groups, not correlated windows as repeats.
            from .common import group_id
            grouped={}
            for j in near:grouped.setdefault(group_id(rows[j]),[]).append(j)
            votes=[all(rows[j]['post'][cid]['success'] is True for j in ids) for ids in grouped.values()]
            successes=sum(votes);n=len(votes)
            p,lower,ready=threshold(n,successes,c,spec)
            values[row['node_id']]={'ready':ready,'direct_handoff_risk':-math.log(p),'p_hat':float(p),'successes':successes,'n':n,'terminal_ids':[rows[j]['node_id'] for j in near]}
            if lower is not None:values[row['node_id']]['p_lower']=lower
            if ready:
                edges.append({'source':row['node_id'],'target':sink_id(cid),'kind':'sink','post_condition':cid,
                              'cost':0.,'p_hat':p,'n':n,'evidence':values[row['node_id']]['terminal_ids']})
        probability[cid]=values
    return probability,edges


def heldout_probability(test,train,noise,spec,condition):
    from .noise import pairwise_distances
    c=next(c for c in spec['post_conditions'] if c['id']==condition);a,b=c.get('p_prior',[1,1])
    allrows=train+test;D=pairwise_distances(allrows,noise);pred=[]
    if spec.get('sinks',{}).get('kernel_ready'):
        from .ready_kernel import estimate
        values,cal=estimate(train,D[:,:len(train)],c,spec['sinks'])
        return [dict(node_id=r['node_id'],p_hat=v['p_hat'],p_lower=v['p_lower'],ready=v['ready'],h_ready=cal['h_ready'],threshold=cal['threshold'],success=int(r['post'][condition]['success']),n_train_neighbors=v['n']) for r,v in zip(test,values[len(train):]) if r.get('is_terminal') and r['post'][condition]['evaluated']]
    terminal=[i for i,r in enumerate(train) if r.get('is_terminal') and r['post'][condition]['evaluated']]
    radius=1.
    if spec.get('graph',{}).get('resolution_scaled_radius'):
        from .v5_values import resolution
        from .graph import observed_edges
        _,radii=resolution(train,observed_edges(train,spec),D[:len(train),:len(train)],spec)
        radius=radii['ready_radius']
    for j,r in enumerate(test):
        if not r.get('is_terminal') or not r['post'][condition]['evaluated']:continue
        ids=[i for i in terminal if D[len(train)+j,i]<=radius]
        if spec.get('sinks',{}).get('lower_bound_ready'):
            from .common import group_id
            grouped={}
            for i in ids:
                grouped.setdefault(group_id(train[i]),[]).append(train[i]['post'][condition]['success'])
            votes=[all(v) for v in grouped.values()]
            p,lower,ready=threshold(len(votes),sum(votes),c,spec)
            pred.append(dict(node_id=r['node_id'],p_hat=p,p_lower=lower,ready=ready,
                             success=int(r['post'][condition]['success']),n_train_neighbors=len(votes)))
            continue
        if not ids:continue
        successes=sum(train[i]['post'][condition]['success'] for i in ids)
        pred.append({'node_id':r['node_id'],'p_hat':(successes+a)/(len(ids)+a+b),'success':int(r['post'][condition]['success']),'n_train_neighbors':len(ids)})
    return pred
