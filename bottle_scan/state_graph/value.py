"""Observation-only ensembles for cost and local post-condition probability."""
import numpy as np
from sklearn.linear_model import Ridge,LogisticRegression
from sklearn.preprocessing import StandardScaler
from .noise import feature_vector
from .common import group_id


def fit_regression(X,Y,groups,weights=None,ensemble=5,seed=0):
    scaler=StandardScaler().fit(X);Z=scaler.transform(X);rng=np.random.default_rng(seed);unique=np.unique(groups);models=[]
    for i in range(ensemble):
        chosen=rng.choice(unique,len(unique),replace=True);ix=np.concatenate([np.flatnonzero(groups==g) for g in chosen])
        model=Ridge(alpha=1.).fit(Z[ix],Y[ix],sample_weight=None if weights is None else np.asarray(weights)[ix]);models.append(model)
    return {'models':models,'scaler':scaler}


def predict_regression(bundle,X):
    predictions=np.array([m.predict(bundle['scaler'].transform(np.atleast_2d(X))) for m in bundle['models']])
    return predictions.mean(0),predictions.std(0)


def fit_value(rows,labels,noise,spec,condition,seed=0):
    labelmap={x['node_id']:x for x in labels['nodes'] if x['post_condition']==condition and x['value_target'] is not None}
    selected=[r for r in rows if r['node_id'] in labelmap];minimum=spec.get('models',{}).get('min_value_nodes',8)
    if len(selected)<minimum:return {'status':'insufficient_value_labels','n':len(selected),'required':minimum}
    X=np.array([feature_vector(r,noise) for r in selected]);Y=np.array([[labelmap[r['node_id']]['value_target'],labelmap[r['node_id']]['p_target']] for r in selected]);groups=np.array([group_id(r) for r in selected])
    fitted=fit_regression(X,Y,groups,ensemble=spec.get('models',{}).get('ensemble',5),seed=seed)
    _,sigma=predict_regression(fitted,X)
    controls={}
    for head,key in [('I','invocation_label'),('beta','exit_label')]:
        idx=[i for i,r in enumerate(selected) if labelmap[r['node_id']][key] is not None]
        y=np.array([int(labelmap[selected[i]['node_id']][key]) for i in idx])
        model=LogisticRegression(C=1,max_iter=1000).fit(fitted['scaler'].transform(X[idx]),y) if len(set(y))>1 else None
        controls[head]={'model':model,'constant':float(y.mean()) if len(y) else None,'counts':{str(c):int(sum(y==c)) for c in (0,1)}}
    return {'status':'trained','n':len(selected),'post_condition':condition,'regression':fitted,'classification_controls':controls,
            'fit_node_ids':[r['node_id'] for r in selected],'fit_groups':sorted(set(groups)),
            'uncertainty_q90':float(np.quantile(sigma[:,0],.9)),'feature_source':'obs only','p_estimate':'auxiliary estimate of local Beta-smoothed outcome probability, separate from exp(-V)'}


def predict_value(model,row,noise):
    if model.get('status')!='trained':return {'status':'Unknown','reason':model.get('status','unavailable')}
    mean,sigma=predict_regression(model['regression'],feature_vector(row,noise));mean,sigma=mean[0],sigma[0]
    return {'status':'predicted','V':float(max(0,mean[0])),'std':float(sigma[0]),'p_hat':float(np.clip(mean[1],0,1)),'p_std':float(sigma[1])}
