"""Separate empirical post-condition probabilities and zero-cost Ready sinks; direct handoff risk is separate."""
import math
import numpy as np


def sink_id(condition):return '__post__:'+condition


def estimate_sinks(rows,D,spec):
    probability={};edges=[]
    for c in spec['post_conditions']:
        cid=c['id'];alpha,beta=c.get('p_prior',[1,1]);values={}
        terminal=[i for i,r in enumerate(rows) if r.get('is_terminal') and r.get('post',{}).get(cid,{}).get('evaluated') is True]
        for i,row in enumerate(rows):
            near=[j for j in terminal if i==j or D[i,j]<=1]
            # Count independent trial groups, not correlated windows as repeats.
            from .common import group_id
            grouped={}
            for j in near:grouped.setdefault(group_id(rows[j]),[]).append(j)
            votes=[all(rows[j]['post'][cid]['success'] is True for j in ids) for ids in grouped.values()]
            successes=sum(votes);n=len(votes)
            p=(successes+alpha)/(n+alpha+beta)
            ready=n>=c.get('min_ready_trials',3) and p>=c.get('ready_p',.8)
            values[row['node_id']]={'ready':ready,'direct_handoff_risk':-math.log(p),'p_hat':float(p),'successes':successes,'n':n,'terminal_ids':[rows[j]['node_id'] for j in near]}
            if ready:
                edges.append({'source':row['node_id'],'target':sink_id(cid),'kind':'sink','post_condition':cid,
                              'cost':0.,'p_hat':p,'n':n,'evidence':values[row['node_id']]['terminal_ids']})
        probability[cid]=values
    return probability,edges


def heldout_probability(test,train,noise,spec,condition):
    from .noise import pairwise_distances
    c=next(c for c in spec['post_conditions'] if c['id']==condition);a,b=c.get('p_prior',[1,1])
    allrows=train+test;D=pairwise_distances(allrows,noise);pred=[]
    terminal=[i for i,r in enumerate(train) if r.get('is_terminal') and r['post'][condition]['evaluated']]
    for j,r in enumerate(test):
        if not r.get('is_terminal') or not r['post'][condition]['evaluated']:continue
        ids=[i for i in terminal if D[len(train)+j,i]<=1]
        if not ids:continue
        successes=sum(train[i]['post'][condition]['success'] for i in ids)
        pred.append({'node_id':r['node_id'],'p_hat':(successes+a)/(len(ids)+a+b),'success':int(r['post'][condition]['success']),'n_train_neighbors':len(ids)})
    return pred
