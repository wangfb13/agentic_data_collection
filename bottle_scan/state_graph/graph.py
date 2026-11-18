"""Task-gated observation graph with observed, inferred, similar and sink edges."""
import heapq
import math
from collections import defaultdict
import numpy as np
from .common import digest,group_id,independent
from .skill_interface import model_observation, normalize_action
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
        typed=b.get('action_executed')
        u=np.asarray(b.get('u_executed'),float)
        if typed is None and (u.shape!=(dim,) or not np.isfinite(u).all()):continue
        if not b.get('u_receipt',{}).get('applied',True):continue
        try:
            action=normalize_action(typed or u.tolist(),spec);executable=True
        except ValueError:
            # A measured overshoot remains evidence, but is not a permitted future action.
            action=typed or {'kind':spec['task']['action_space']['id'],'u':u.tolist()};executable=False
        u=action.get('u')
        edges.append({'source':a['node_id'],'target':b['node_id'],'kind':'observed','u':u,'cost':step_cost(b,spec),
                      'action':action,'executable':executable,'receipt':b['u_receipt'],'group_id':group_id(b),'before_obs':a['obs'],'after_obs':b['obs']})
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
            attempts=[e for e in edges if e['kind']=='observed' and ident==e['source']
                      and byid[e['target']].get('is_terminal') and byid[e['target']]['post'][cid]['evaluated']
                      and byid[e['target']]['post'][cid]['success'] is False]
            if p.get('ready',False):status='ready'
            elif cost is not None and cost>0:status='recoverable'
            elif attempts:status='unreachable'
            else:status='unsupported'
            states[ident]={'status':status,'trial_groups':sorted({e['group_id'] for e in attempts}),'evidence':[e['edge_id'] for e in attempts],
                           'meaning':'not_recovered_under_tested_actions' if status=='unreachable' else status}
        result[cid]=states
    return result


def build_graph(rows,noise,spec,dynamics=None):
    validate_graph_rows(rows,spec);rows=list(rows);D=pairwise_distances(rows,noise);ids=[r['node_id'] for r in rows]
    observed=observed_edges(rows,spec);edges=list(observed);k=spec.get('graph',{}).get('knn',10)
    radius={}
    scale=None
    if spec.get('graph',{}).get('resolution_scaled_radius'):
        from .v5_values import resolution
        from .noise import metric_distance_mm
        M=metric_distance_mm(rows,noise) if rows and 'tac_metric' in noise['active_modalities'] else None
        scale,radius=resolution(rows,observed,D,spec,M,metric_axes=[('r','h').index(x) for x in noise['tac_metric']['axes']] if M is not None and spec['observation'].get('metric_per_axis') else None)
        if noise.get('metric_normalized_per_mm'):
            scale['physical_radius_mm']={k:v/noise['metric_normalized_per_mm'] for k,v in radius.items()}
    neighbors=[]
    for i in range(len(rows)):
        candidates=sorted([j for j in range(len(rows)) if j!=i and np.isfinite(D[i,j])],key=lambda j:(D[i,j],ids[j]))
        neighbors.append(set(candidates[:k]))
    for i in range(len(rows)):
        for j in range(i):
            if D[i,j]<=radius.get('similar_radius',1) and i in neighbors[j] and j in neighbors[i]:
                for a,b in ((i,j),(j,i)):edges.append({'source':ids[a],'target':ids[b],'kind':'similar','cost':0.,'distance':float(D[i,j])})
    inferred=[]
    if dynamics and dynamics.get('status')=='trained':
        from .dynamics import inferred_edges
        inferred=inferred_edges(rows,dynamics,noise,spec,observed,endpoint_radius=radius.get('endpoint_radius',1.));edges.extend(inferred)
    calibrations={}
    probabilities,sinks=estimate_sinks(rows,D,spec,**({'calibrations':calibrations} if spec.get('sinks',{}).get('kernel_ready') else {}),**({'ready_radius':radius['ready_radius']} if radius else {}));edges.extend(sinks)
    for e in edges:e['edge_id']='e_'+digest([e['kind'],e['source'],e['target'],e.get('action'),e.get('u'),e.get('post_condition')])[:20]
    action_edges=[e for e in edges if e['kind'] in ('observed','inferred','sink') and e.get('executable',True)]
    include_inferred=spec.get('labels',{}).get('include_inferred',True)
    if not include_inferred:
        action_edges=[e for e in action_edges if e['kind']!='inferred']
    R={c['id']:reverse_costs(ids,action_edges,sink_id(c['id'])) for c in spec['post_conditions']}
    reachability=classify_reachability(rows,edges,R,probabilities,spec)
    result={'schema':'state_graph_v4','recovery_semantics':'action_paths_to_empirical_ready_only','similarity_is_action':False,'task_id':rows[0]['task_id'] if rows else None,'node_ids':ids,'edges':edges,'R':R,'probabilities':probabilities,'reachability':reachability,
            'noise_model_id':noise['model_id'],'spec_hash':digest(spec),'active_post_condition':spec['active_post_condition'],
            'counts':{kind:sum(e['kind']==kind for e in edges) for kind in ('observed','inferred','similar','sink')}}
    if spec.get('graph_schema')=='state_graph_v5':result['schema']='state_graph_v5'
    if 'include_inferred' in spec.get('labels',{}):result['label_inferred']=include_inferred
    if calibrations:result['ready_calibration']=calibrations
    if scale is not None:result.update(resolution=scale,radii=radius)
    if spec.get('graph',{}).get('success_value'):
        from .v5_values import solve
        result.update(S={},best_path={},first_action={},model_reachability={},
                      recovery_semantics='budgeted_discounted_empirical_success',
                      similarity_is_action={'counts_as_step':False,'max_consecutive_borrows':1,
                         'enabled':bool(scale is not None and spec['graph'].get('borrow_similar_actions'))})
        for c in spec['post_conditions']:
            cid=c['id'];prob=probabilities[cid]
            S,paths=solve(ids,edges,prob,D,radius.get('similar_radius',1.),spec,include_inferred)
            MS,mpaths=solve(ids,edges,prob,D,radius.get('similar_radius',1.),spec,True) if inferred and not include_inferred else (S,paths)
            result['S'][cid]=S
            result['best_path'][cid]=paths
            result['first_action'][cid]={i:p[0]['action'] if p else None for i,p in paths.items()}
            result['R'][cid]={i:-math.log(v) if v>0 else None for i,v in S.items()}
            result['model_reachability'][cid]={}
            inf_ids={e['edge_id'] for e in inferred}
            for i in ids:
                entry=result['reachability'][cid][i];p=paths[i]
                if prob[i]['ready']:status='ready'
                elif p and S[i]>=spec['graph']['recover_p_min']:status='recoverable'
                else:status='unreachable' if entry['evidence'] else 'unsupported'
                entry.update(status=status,meaning='not_recovered_under_tested_actions' if status=='unreachable' else status,
                             action_source=p[0]['action_source'] if status=='recoverable' else None)
                hypothesis=[step['edge_id'] for step in mpaths[i] if step['edge_id'] in inf_ids]
                if hypothesis and MS[i]>=spec['graph']['recover_p_min']:
                    result['model_reachability'][cid][i]=dict(status='model_recoverable',S=MS[i],
                                                         inferred_edge_ids=hypothesis,path=mpaths[i])
    if not include_inferred and 'model_reachability' not in result:
        from .labels import shortest_path
        result['model_reachability']={}
        hypothesis_graph={**result,'label_inferred':True}
        for c in spec['post_conditions']:
            cid=c['id'];hypotheses={}
            if inferred:
                model_costs=reverse_costs(ids,[e for e in edges if e['kind'] in ('observed','inferred','sink') and e.get('executable',True)],sink_id(cid))
                for node,cost in model_costs.items():
                    if cost is None or result['reachability'][cid][node]['status'] in ('ready','recoverable'):continue
                    path=shortest_path(hypothesis_graph,node,cid) or []
                    inferred_ids=[e['edge_id'] for e in path if e['kind']=='inferred']
                    if inferred_ids:hypotheses[node]={'status':'model_recoverable','inferred_edge_ids':inferred_ids,'R':cost}
            result['model_reachability'][cid]=hypotheses
    result['graph_version']=digest(result)[:24]
    return result
