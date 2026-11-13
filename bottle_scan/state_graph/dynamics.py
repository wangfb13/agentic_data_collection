"""Observation/action dynamics ensemble and uncertainty-weighted inferred edges."""
from collections import defaultdict
import warnings
import numpy as np
from scipy.stats import chi2
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LogisticRegression,Ridge
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning
from .skill_interface import model_observation,action_grid
from .common import group_id,group_folds


def transitions(rows,spec):
    from .graph import observed_edges
    lookup={r['node_id']:r for r in rows}
    return [{'source':lookup[e['source']],'target':lookup[e['target']],'u':e['u'],'receipt':e['receipt'],'group_id':e['group_id'],'edge':e} for e in observed_edges(rows,spec) if e.get('u') is not None and e.get('action',{}).get('kind',spec['task']['action_space']['id'])==spec['task']['action_space']['id']]


def target_mapping(rows,spec):
    cfg=spec.get('dynamics',{});mapping=cfg.get('targets')
    if mapping:return {name:list(indices) for name,indices in mapping.items()}
    inputs=cfg.get('inputs',list(spec['observation']['modalities']))
    return {n:list(range(min(len(rows[0]['obs'][n]),12))) for n in inputs}


def input_vector(row,u,inputs):
    obs=model_observation(row,inputs)
    return np.concatenate([*(obs[n] for n in inputs),np.asarray(u,float)])


def target_vector(row,mapping):
    obs=model_observation(row,mapping)
    return np.concatenate([obs[name][indices] for name,indices in mapping.items()])


def _design(X,projectors):
    X=np.atleast_2d(X)
    return np.concatenate([p.transform(X[:,a:b]) if p is not None else X[:,a:b] for a,b,p in projectors],axis=1)


def _fit_ensemble(X,Y,groups,cfg,seed=0):
    projectors=[]
    for a,b in cfg.get('_input_slices',[(0,X.shape[1])]):
        dim=cfg.get('input_pca_dim')
        p=PCA(min(dim,len(X)-1,b-a),svd_solver='full').fit(X[:,a:b]) if dim and b-a>dim else None
        projectors.append((a,b,p))
    X=_design(X,projectors)
    xs=StandardScaler().fit(X);ys=StandardScaler().fit(Y)
    a=xs.transform(X);b=ys.transform(Y);rng=np.random.default_rng(seed);members=[];converged=[]
    unique=np.unique(groups)
    for i in range(cfg.get('ensemble',5)):
        selected=rng.choice(unique,len(unique),replace=True);idx=np.concatenate([np.flatnonzero(groups==g) for g in selected])
        model=MLPRegressor(hidden_layer_sizes=tuple(cfg.get('hidden',[32,32])),activation='tanh',solver='lbfgs',alpha=cfg.get('alpha',.01),
                           max_iter=cfg.get('max_iter',200),random_state=seed+i)
        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter('always',ConvergenceWarning);model.fit(a[idx],b[idx])
        converged.append(not any(issubclass(w.category,ConvergenceWarning) for w in records));members.append(model)
    return {'members':members,'x_scaler':xs,'y_scaler':ys,'converged':converged,'projectors':projectors}


def _predict_ensemble(model,X):
    X=np.atleast_2d(X);a=model['x_scaler'].transform(_design(X,model['projectors']))
    values=[]
    for m in model['members']:
        pred=m.predict(a)
        if pred.ndim==1:pred=pred[:,None]
        values.append(model['y_scaler'].inverse_transform(pred))
    values=np.asarray(values)
    return values.mean(0),values.std(0)


def target_metric(rows,mapping,noise):
    # Projection is in raw observation components, so estimate its repeated noise
    # directly rather than taking indices of a PCA-compressed covariance.
    from sklearn.covariance import LedoitWolf
    result={};offset=0
    for name,indices in mapping.items():
        groups=defaultdict(list);seen=set()
        for r in rows:
            if r.get('window_index',0)!=0 or group_id(r) in seen:continue
            seen.add(group_id(r));groups[r['repeat_key']].append(np.asarray(r['obs'][name])[indices])
        repeated=[(np.asarray(v)-np.mean(v,axis=0))*np.sqrt(len(v)/(len(v)-1)) for v in groups.values() if len(v)>1]
        if not repeated:return None
        cov=LedoitWolf(assume_centered=True).fit(np.concatenate(repeated)).covariance_
        cov+=np.eye(len(cov))*max(float(np.trace(cov)/len(cov))*1e-9,1e-12)
        result[name]={'covariance':2*cov,'slice':(offset,offset+len(indices)),
                      'tau':np.sqrt(chi2.ppf(1-noise['alpha']/len(mapping),len(indices))),
                      'weight':noise['modalities'][name]['weight']}
        offset+=len(indices)
    return result


def prediction_distance(delta,metric,sigma=None):
    result=0.
    for item in metric.values():
        lo,hi=item['slice'];cov=np.asarray(item['covariance'])
        if sigma is not None:cov=cov+np.diag(np.asarray(sigma)[lo:hi]**2)
        d=np.asarray(delta)[lo:hi]
        value=np.sqrt(max(0.,d@np.linalg.solve(cov,d)))/item['tau']*item['weight']
        result=max(result,float(value))
    return result


def fit_dynamics(rows,noise,spec,seed=0):
    cfg=dict(spec.get('dynamics',{}));items=transitions(rows,spec)
    groups=np.array([t['group_id'] for t in items]);minimum=cfg.get('min_transitions',6)
    if len(items)<minimum or len(set(groups))<3:
        return {'status':'insufficient_transitions','n':len(items),'groups':len(set(groups)),'required':minimum,'fit_node_ids':[r['node_id'] for r in rows]}
    inputs=cfg.get('inputs',list(spec['observation']['modalities']));mapping=target_mapping(rows,spec);metric=target_metric(rows,mapping,noise)
    if metric is None:return {'status':'insufficient_repeat_noise','n':len(items)}
    offset=0;slices=[]
    for name in inputs:
        end=offset+len(rows[0]['obs'][name]);slices.append((offset,end));offset=end
    slices.append((offset,offset+spec['task']['action_space']['dim']));cfg['_input_slices']=slices
    X=np.array([input_vector(t['source'],t['u'],inputs) for t in items])
    Y=np.array([target_vector(t['target'],mapping)-target_vector(t['source'],mapping) for t in items])
    # Labels for off-model events are cross-fitted; fitting residuals would give
    # an optimistically small off-event rate.
    from sklearn.model_selection import GroupKFold
    oof=np.zeros_like(Y);oof_sigma=np.zeros_like(Y)
    for k,(tr,te) in enumerate(GroupKFold(min(3,len(set(groups)))).split(X,groups=groups)):
        sub=_fit_ensemble(X[tr],Y[tr],groups[tr],cfg,seed+100*(k+1));oof[te],oof_sigma[te]=_predict_ensemble(sub,X[te])
    errors=np.array([prediction_distance(y-p,metric) for y,p in zip(Y,oof)]);off=(errors>1).astype(int)
    model=_fit_ensemble(X,Y,groups,cfg,seed)
    pmodel=None
    if len(set(off))>1:pmodel=LogisticRegression(C=1,max_iter=1000).fit(model['x_scaler'].transform(_design(X,model['projectors'])),off)
    norms=np.array([prediction_distance(s,metric) for s in oof_sigma])
    return {'status':'trained','n':len(items),'groups':len(set(groups)),'inputs':inputs,'targets':mapping,'metric':metric,'ensemble':model,
            'p_classifier':pmodel,'p_constant':float((off.sum()+1)/(len(off)+2)),'p_off_oof_labels':off.tolist(),
            'mean_delta':Y.mean(0),'linear_baseline':Ridge(alpha=1).fit(model['x_scaler'].transform(_design(X,model['projectors'])),Y),
            'input_projection':'train-fitted per-modality PCA; action and low-dimensional physical inputs retained','p_off_label_definition':'Cross-fitted endpoint residual > 1 in modeled components, without adding predictive variance.',
            'modeled_distance_scope':list(mapping),'unmodeled_components':'Not used for inferred endpoint matching; copied only for optional planning approximation.',
            'uncertainty_q90':float(np.quantile(norms,.9)), 'train_oof_errors':errors.tolist(),
            'fit_node_ids':[r['node_id'] for r in rows],'fit_groups':sorted(set(groups))}


def predict(model,row,u):
    X=input_vector(row,u,model['inputs']);delta,sigma=_predict_ensemble(model['ensemble'],X)
    delta,sigma=delta[0],sigma[0]
    x=model['ensemble']['x_scaler'].transform(_design(X,model['ensemble']['projectors']))
    p_off=float(model['p_classifier'].predict_proba(x)[0,1]) if model['p_classifier'] is not None else model['p_constant']
    return {'mean':target_vector(row,model['targets'])+delta,'sigma':sigma,'p_off':p_off,
            'uncertainty':prediction_distance(sigma,model['metric'])}


def inferred_cost(uncertainty,p_off,spec):
    cfg=spec.get('graph',{})
    return float(cfg.get('step_cost',1))+float(cfg.get('uncertainty_cost',1))*uncertainty-np.log(max(1e-9,1-p_off))


def inferred_edges(rows,model,noise,spec,observed):
    existing={(e['source'],e['target']) for e in observed};best={}
    targets=np.array([target_vector(r,model['targets']) for r in rows]);actions=action_grid(spec)
    limit=spec.get('dynamics',{}).get('max_uncertainty',1.)
    for source in rows:
        for u in actions:
            if np.linalg.norm(u)<1e-12:continue
            prediction=predict(model,source,u)
            if prediction['uncertainty']>limit:continue
            distances=[prediction_distance(prediction['mean']-v,model['metric'],prediction['sigma']) for v in targets]
            candidates=[i for i,r in enumerate(rows) if r['node_id']!=source['node_id'] and (source['node_id'],r['node_id']) not in existing]
            if not candidates:continue
            j=min(candidates,key=lambda i:(distances[i],rows[i]['node_id']))
            if distances[j]>1:continue
            key=(source['node_id'],rows[j]['node_id']);cost=inferred_cost(prediction['uncertainty'],prediction['p_off'],spec)
            edge={'source':key[0],'target':key[1],'kind':'inferred','executable':True,'action':{'kind':spec['task']['action_space']['id'],'u':u.tolist()},'u':u.tolist(),'cost':float(cost),'sigma':prediction['sigma'].tolist(),
                  'uncertainty':prediction['uncertainty'],'p_off':prediction['p_off'],'endpoint_distance':distances[j],
                  'distance_scope':list(model['targets']),'verified':False}
            if key not in best or cost<best[key]['cost']:best[key]=edge
    return list(best.values())


def evaluate_dynamics(model,held_rows,spec):
    if model.get('status')!='trained':return {'status':model.get('status','unavailable'),'samples':[]}
    samples=[]
    for t in transitions(held_rows,spec):
        prediction=predict(model,t['source'],t['u']);actual=target_vector(t['target'],model['targets']);residual=actual-prediction['mean']
        d=prediction_distance(residual,model['metric']);expanded=prediction_distance(residual,model['metric'],prediction['sigma'])
        base=target_vector(t['source'],model['targets']);x=input_vector(t['source'],t['u'],model['inputs'])
        linear=model['linear_baseline'].predict(model['ensemble']['x_scaler'].transform(_design(x,model['ensemble']['projectors'])))[0]
        baseline_errors={name:abs(actual-(base+delta)).tolist() for name,delta in [('identity',0.),('mean_delta',model['mean_delta']),('linear',linear)]}
        samples.append({'baseline_absolute_error':baseline_errors,'source':t['source']['node_id'],'target':t['target']['node_id'],'group_id':t['group_id'],
                        'distance':d,'variance_augmented_distance':expanded,'correct_within_noise':d<=1,
                        'off_label':int(d>1),'p_off':prediction['p_off'],'uncertainty':prediction['uncertainty'],
                        'absolute_error':abs(residual).tolist(),'sigma':prediction['sigma'].tolist(),
                        'within_2sigma':(abs(residual)<=2*prediction['sigma']).tolist()})
    return {'status':'computed' if samples else 'no_heldout_transitions','samples':samples}
