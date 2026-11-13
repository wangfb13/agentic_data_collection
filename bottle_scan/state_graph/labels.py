"""Shortest-path value/action supervision, independently for each post condition."""
import heapq
import math
import numpy as np
from .sink import sink_id


def shortest_path(graph,source,condition):
    edges=[e for e in graph['edges'] if e['kind'] in ('observed','inferred','sink') and e.get('executable',True)];adj={}
    for edge in edges:adj.setdefault(edge['source'],[]).append(edge)
    target=sink_id(condition);queue=[(0.,0,source,[])];best={source:(0.,0)}
    while queue:
        cost,hops,node,path=heapq.heappop(queue)
        if best.get(node)!=(cost,hops):continue
        if node==target:return path
        for e in sorted(adj.get(node,[]),key=lambda e:e['edge_id']):
            other=e['target'];key=(cost+e['cost'],hops+1)
            if key<best.get(other,(math.inf,10**9)):
                best[other]=key;heapq.heappush(queue,(*key,other,path+[e]))
    return None


def generate_labels(graph,rows,spec):
    labels=[];advantages=[];dim=spec['task']['action_space']['dim']
    for c in spec['post_conditions']:
        cid=c['id'];R=graph['R'][cid]
        for row in rows:
            ident=row['node_id'];state=graph['reachability'][cid][ident]['status'];path=shortest_path(graph,ident,cid) if R[ident] is not None else None
            steps=[e for e in (path or []) if e['kind'] in ('observed','inferred')]
            action=steps[0].get('u') if steps else ([0.]*dim if state=='ready' else None)
            actions=[e.get('action',{'kind':spec['task']['action_space'].get('id','vector'),'u':e.get('u')}) for e in steps]
            labels.append({'node_id':ident,'post_condition':cid,'status':state,'invocation_label':0 if state=='ready' else (1 if state=='recoverable' else None),
                           'exit_label':state=='ready','value_target':R[ident],'p_target':graph['probabilities'][cid][ident]['p_hat'],'p_evidence_count':graph['probabilities'][cid][ident].get('n',0),
                           'action_target':action,'action_sequence':actions,'first_action':actions[0] if actions else None,'path_edge_ids':[] if path is None else [e['edge_id'] for e in path]})
        for e in graph['edges']:
            if e['kind'] not in ('observed','inferred'):continue
            a,b=R[e['source']],R[e['target']]
            if a is None or b is None:continue
            advantages.append({'edge_id':e['edge_id'],'post_condition':cid,'advantage':a-b-e['cost'],'kind':e['kind']})
    return {'nodes':labels,'transitions':advantages}
