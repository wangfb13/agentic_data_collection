"""Advantage-weighted action regression; actions are vectors in the declared space."""
import numpy as np
from .noise import feature_vector
from .common import group_id
from .value import fit_regression,predict_regression


def policy_samples(rows,graph,labels,spec,condition):
    byid={r['node_id']:r for r in rows};advantages={x['edge_id']:x['advantage'] for x in labels['transitions'] if x['post_condition']==condition}
    result=[];cfg=spec.get('models',{})
    for edge in graph['edges']:
        if edge['kind']!='observed' and not (edge['kind']=='inferred' and cfg.get('include_inferred_policy',False)):continue
        if edge['edge_id'] not in advantages:continue
        weight=np.exp(np.clip(cfg.get('advantage_eta',1)*advantages[edge['edge_id']],-30,30))
        if edge['kind']=='inferred':weight*=max(0,1-edge['p_off'])*np.exp(-edge['uncertainty'])
        result.append({'row':byid[edge['source']],'u':edge['u'],'weight':float(weight),'edge':edge,'advantage':advantages[edge['edge_id']]})
    return result


def fit_policy(rows,graph,labels,noise,spec,condition,seed=0):
    samples=policy_samples(rows,graph,labels,spec,condition);minimum=spec.get('models',{}).get('min_policy_transitions',6)
    if len(samples)<minimum:return {'status':'insufficient_action_labels','n':len(samples),'required':minimum}
    X=np.array([feature_vector(s['row'],noise) for s in samples]);Y=np.array([s['u'] for s in samples]);weights=np.array([s['weight'] for s in samples]);groups=np.array([group_id(s['row']) for s in samples])
    regression=fit_regression(X,Y,groups,weights,ensemble=spec.get('models',{}).get('ensemble',5),seed=seed)
    return {'status':'trained','n':len(samples),'regression':regression,'max_action':spec['safety_envelope']['max_action'],
            'fit_node_ids':[s['row']['node_id'] for s in samples],'fit_groups':sorted(set(groups)),
            'sample_weights':weights.tolist(),'feature_source':'obs only','objective':'weighted Ridge ensemble; weights exp(eta * advantage)',
            'inferred_samples':sum(s['edge']['kind']=='inferred' for s in samples)}


def predict_policy(model,row,noise):
    if model.get('status')!='trained':return {'status':'Unknown','reason':model.get('status','unavailable')}
    mean,std=predict_regression(model['regression'],feature_vector(row,noise));limits=np.asarray(model['max_action'])
    return {'status':'predicted','u':np.clip(mean[0],-limits,limits).tolist(),'std':std[0].tolist()}
