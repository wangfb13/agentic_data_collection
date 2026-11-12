"""Task-gated observation graph with observed, inferred, similar and sink edges."""
import heapq
import math
from collections import defaultdict
import numpy as np
from .common import digest,group_id,independent
from .skill_interface import model_observation
from .noise import pairwise_distances
from .sink import estimate_sinks,sink_id


def gate(a,b,config=None):
    return a.get('window_valid') is True and b.get('window_valid') is True and a.get('task_id') is not None and a.get('task_id')==b.get('task_id')


def validate_graph_rows(rows,spec):
    if len({r['node_id'] for r in rows})!=len(rows):raise ValueError('Duplicate node id')
    if len({r['task_id'] for r in rows})>1:raise ValueError('One task per graph')
    if any(not r.get('window_valid') for r in rows):raise ValueError('Invalid observation window')
    if any(independent(r) for r in rows):raise ValueError('Independent evaluation rows cannot define training topology')
    for name in spec['observation']['modalities']:
        versions={r.get('extractor_versions',{}).get(name) for r in rows}
        if rows and (None in versions or len(versions)!=1):raise ValueError('Mixed/missing extractor version: '+name)
    for row in rows:model_observation(row,spec['observation']['modalities'])


def step_cost(target,spec):
    cfg=spec.get('graph',{});risk=target.get('risk',{});penalty=0.
    for name,setting in spec['safety_envelope'].get('risk_metrics',{}).items():
        value=risk.get(name)
        if value is None:continue
        if setting.get('kind')=='exceeds':value=max(0.,float(value)/float(setting['limit'])-1)
        penalty+=float(setting.get('weight',1))*float(value)
    return float(cfg.get('step_cost',1))+penalty


def observed_edges(rows,spec):
    lookup={r['node_id']:r for r in rows};edges=[]
    dim=spec['task']['action_space']['dim']
    for b in rows:
        a=lookup.get(b.get('previous_node_id'))
        if a is None or group_id(a)!=group_id(b) or not gate(a,b):continue
        if b['window_index']<=a['window_index'] or not b.get('u_receipt'):continue
        u=np.asarray(b.get('u_executed'),float)
        if u.shape!=(dim,) or not np.isfinite(u).all():continue
        if a['phase'] not in ('B_pre','B_post') or b['phase']!='B_post':continue
        edges.append({'source':a['node_id'],'target':b['node_id'],'kind':'observed','u':u.tolist(),'cost':step_cost(b,spec),
                      'receipt':b['u_receipt'],'group_id':group_id(b),'before_obs':a['obs'],'after_obs':b['obs']})
    return edges


def reverse_costs(node_ids,edges,sink):
    reverse=defaultdict(list)
    for e in edges:reverse[e['target']].append((e['source'],e['cost']))
    values={i:math.inf for i in node_ids};values[sink]=0.;queue=[(0.,sink)]
    while queue:
        cost,node=heapq.heappop(queue)
        if cost!=values[node]:continue
        for previous,step in reverse[node]:
            new=cost+step
            if new<values.get(previous,math.inf)-1e-12:
                values[previous]=new;heapq.heappush(queue,(new,previous))
    return {i:float(values[i]) if np.isfinite(values[i]) else None for i in node_ids}


def classify_reachability(rows,edges,R,probabilities,spec):
    byid={r['node_id']:r for r in rows};result={}
    for c in spec['post_conditions']:
        cid=c['id'];states={}
        for row in rows:
            ident=row['node_id'];p=probabilities[cid][ident];cost=R[cid][ident]
            attempts=[e for e in edges if e['kind']=='observed' and ident in (e['source'],e['target'])
                      and byid[e['target']].get('is_terminal') and byid[e['target']]['post'][cid]['evaluated']
                      and byid[e['target']]['post'][cid]['success'] is False]
            if p['n'] and p['p_hat']>=c.get('ready_p',.8):status='ready'
            elif cost is not None:status='recoverable'
            elif attempts:status='unreachable'
            else:status='unsupported'
            states[ident]={'status':status,'trial_groups':sorted({e['group_id'] for e in attempts}),'evidence':[e['edge_id'] for e in attempts]}
        result[cid]=states
    return result


def build_graph(rows,noise,spec,dynamics=None):
    validate_graph_rows(rows,spec);rows=list(rows);D=pairwise_distances(rows,noise);ids=[r['node_id'] for r in rows]
    observed=observed_edges(rows,spec);edges=list(observed);k=spec.get('graph',{}).get('knn',10)
    neighbors=[]
    for i in range(len(rows)):
        candidates=sorted([j for j in range(len(rows)) if j!=i and np.isfinite(D[i,j])],key=lambda j:(D[i,j],ids[j]))
        neighbors.append(set(candidates[:k]))
    for i in range(len(rows)):
        for j in range(i):
            if D[i,j]<=1 and i in neighbors[j] and j in neighbors[i]:
                for a,b in ((i,j),(j,i)):edges.append({'source':ids[a],'target':ids[b],'kind':'similar','cost':0.,'distance':float(D[i,j])})
    inferred=[]
    if dynamics and dynamics.get('status')=='trained':
        from .dynamics import inferred_edges
        inferred=inferred_edges(rows,dynamics,noise,spec,observed);edges.extend(inferred)
    probabilities,sinks=estimate_sinks(rows,D,spec);edges.extend(sinks)
    for e in edges:e['edge_id']='e_'+digest([e['kind'],e['source'],e['target'],e.get('u'),e.get('post_condition')])[:20]
    R={c['id']:reverse_costs(ids,edges,sink_id(c['id'])) for c in spec['post_conditions']}
    reachability=classify_reachability(rows,edges,R,probabilities,spec)
    result={'schema':'state_graph_v3','task_id':rows[0]['task_id'] if rows else None,'node_ids':ids,'edges':edges,'R':R,'probabilities':probabilities,'reachability':reachability,
            'noise_model_id':noise['model_id'],'spec_hash':digest(spec),'active_post_condition':spec['active_post_condition'],
            'counts':{kind:sum(e['kind']==kind for e in edges) for kind in ('observed','inferred','similar','sink')}}
    result['graph_version']=digest(result)[:24]
    return result
