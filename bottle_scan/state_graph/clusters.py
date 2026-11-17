"""Observation-only complete-link clusters and post-condition-specific symbols."""
from collections import defaultdict
import numpy as np
from scipy.cluster.hierarchy import linkage,fcluster
from scipy.spatial.distance import squareform
from .noise import pairwise_distances,measurement_distance
from .common import digest


def cluster_nodes(rows,graph,noise,labels,spec):
    if not rows:return {'clusters':[],'method':'complete_link_observation_distance'}
    D=pairwise_distances(rows,noise)
    finite=np.where(np.isfinite(D),D,1e9);np.fill_diagonal(finite,0.)
    radius=graph.get('radii',{}).get('radius',1.)
    assignments=fcluster(linkage(squareform(finite,checks=False),method='complete'),t=radius,criterion='distance') if len(rows)>1 else [1]
    groups=defaultdict(list)
    for row,k in zip(rows,assignments):groups[int(k)].append(row)
    lookup={(x['post_condition'],x['node_id']):x for x in labels['nodes']};clusters=[]
    for members in groups.values():
        ids=sorted(r['node_id'] for r in members);centroid={**members[0],'obs':{n:np.mean([r['obs'][n] for r in members],axis=0).tolist() for n in spec['observation']['modalities']}}
        symbols={};statistics={}
        for c in spec['post_conditions']:
            cid=c['id'];p=float(np.median([graph['probabilities'][cid][i]['p_hat'] for i in ids]));R=[graph['R'][cid][i] for i in ids]
            terminal=[r for r in members if r['is_terminal'] and r['post'][cid]['evaluated']]
            states=[graph['reachability'][cid][i]['status'] for i in ids]
            directions={tuple(np.sign(lookup[cid,i]['action_target']).astype(int)) for i in ids if lookup[cid,i]['action_target'] is not None and np.linalg.norm(lookup[cid,i]['action_target'])>1e-9}
            boundaries=[]
            if spec.get('facts',{}).get('boundary_before_stochastic'):
                from .v5_facts import boundary_tests
                boundaries=boundary_tests(terminal,cid,spec)
            if states and all(s=='ready' for s in states):symbol='Ready('+cid+')'
            elif boundaries:symbol='Boundary('+cid+')'
            elif .2<=p<=.8 and len(terminal)>=4:symbol='Stochastic('+cid+')'
            elif all(x=='unreachable' for x in states):symbol='Unreachable'
            elif all(x is None for x in R):symbol='Unsupported'
            elif len(directions)==1:symbol='NeedAction('+','.join(map(str,next(iter(directions))))+')'
            else:symbol='Unknown'
            symbols[cid]=symbol;statistics[cid]={'p_hat_median':p,'terminal_count':len(terminal),'terminal_ids':[r['node_id'] for r in terminal],
                 'successes':sum(r['post'][cid]['success'] for r in terminal),'finite_value_count':sum(x is not None for x in R),'directions':[list(d) for d in sorted(directions)]}
        clusters.append({'cluster_id':'C_'+digest(ids)[:16],'region_anchor':ids[0],'node_ids':ids,'prototype':{k:centroid[k] for k in ('task_id','window_valid','obs','extractor_versions')},'symbols':symbols,'statistics':statistics})
    return {**({'radius':radius} if 'radii' in graph else {}),'clusters':clusters,'method':'complete_link_observation_distance','uses_plan_coords':False,'uses_outcomes_for_membership':False}


def ground_cluster(row,clusters,noise):
    candidates=[(measurement_distance(row,c['prototype'],noise),c['cluster_id']) for c in clusters['clusters']]
    if not candidates:return None
    d,c=min(candidates)
    return c if d<=clusters.get('radius',1.) else None


def reconcile_regions(current,previous):
    """Keep a persistent region name for the largest surviving member overlap.

    Splits/merges are recorded, not silently treated as identical regions.
    """
    old=(previous or {}).get('clusters',[]);used=set()
    for c in sorted(current['clusters'],key=lambda c:(-len(c['node_ids']),c['cluster_id'])):
        candidates=sorted(((len(set(c['node_ids'])&set(p['node_ids'])),p['cluster_id'],p) for p in old),key=lambda x:(-x[0],x[1]))
        c['region_parents']=[ident for count,ident,_ in candidates if count]
        for count,ident,p in candidates:
            if count and ident not in used:
                generated=c['cluster_id'];c['cluster_id']=ident;c['region_anchor']=p.get('region_anchor',p['node_ids'][0]);used.add(ident)
                c['membership_version']=generated;break
    return current
