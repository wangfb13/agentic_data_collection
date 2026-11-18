"""A group-held-out physical displacement head. No consumer or Ready fields read."""
from collections import defaultdict
import numpy as np
from .common import group_folds,group_id
from .skill_interface import model_observation,repeat_key


def measured_rh(source,target,spec):
    receipt=target.get('u_receipt') or {}
    if not receipt.get('applied',receipt.get('completed',False)):return None
    start=np.asarray(source.get('provenance',{}).get('measured_pose'),float)
    end=np.asarray(target.get('provenance',{}).get('measured_pose'),float)
    axis=spec.get('geometry',{}).get('axes_tcp',spec.get('axes_tcp',{})).get('a')
    if axis is None:axis=spec.get('observation',{}).get('metric',{}).get('a_axis_tcp')
    if start.shape!=(4,4) or end.shape!=(4,4) or axis is None:return None
    r=start[:3,:3]@np.asarray(axis,float);r[2]=0
    if np.linalg.norm(r)<1e-6:return None
    delta=end[:3,3]-start[:3,3]
    return np.array([delta@r/np.linalg.norm(r),delta[2]])


def pairs(rows,spec):
    name=spec['observation']['metric']['input_modality']
    lookup={r['node_id']:r for r in rows};groups=defaultdict(dict);items=[]
    for r in rows:
        if r.get('window_index')==0 and not spec.get('noise',{}).get('relative_repeat_key'):
            groups[(r.get('repeat_key') or repeat_key(r,spec),str(r.get('provenance',{}).get('calibration_epoch',r.get('calibration_epoch'))))].setdefault(group_id(r),r)
        prev=lookup.get(r.get('previous_node_id'))
        if prev and prev.get('window_valid') and r.get('window_valid') and group_id(r)==group_id(prev):
            y=measured_rh(prev,r,spec)
            if y is not None and np.isfinite(y).all():
                items.append((prev,r,y,'action'))
    from .repeat_evidence import indices
    grouped=([[rows[i] for i in ids] for ids in indices(rows,spec)] if spec.get('noise',{}).get('relative_repeat_key')
             else [list(units.values()) for units in groups.values()])
    for v in grouped:
        # Centering all pairs is unnecessary. Adjacent distinct trials supply independent repeat comparisons.
        for i in range(1,len(v)):items.append((v[i-1],v[i],np.zeros(2),'repeat'))
    return [(model_observation(b,[name])[name]-model_observation(a,[name])[name],y,kind,
             [a['node_id'],b['node_id']]) for a,b,y,kind in items]


def fit_head(items,cfg,rank=2):
    if {p[2] for p in items}!={'action','repeat'}:return None
    X=np.array([p[0] for p in items]);Y=np.array([p[1] for p in items])
    if np.linalg.matrix_rank(Y)<rank:return None
    w=np.sqrt([cfg['displacement_weight'] if p[2]=='action' else cfg['repeat_weight'] for p in items])
    X=X*w[:,None];Y=Y*w[:,None];alpha=cfg['ridge']
    if X.shape[1]>len(X):return X.T@np.linalg.solve(X@X.T+alpha*np.eye(len(X)),Y)
    return np.linalg.solve(X.T@X+alpha*np.eye(X.shape[1]),X.T@Y)


def fit_metric(rows,spec):
    cfg=spec['observation']['metric'];name=cfg['input_modality']
    rows=[r for r in rows if name in r.get('obs',{})]
    result={'status':'insufficient_measured_transitions_or_repeats','unit':'mm','axes':['r','h'],
            'input':name,'objects':sorted({str(r.get('attrs',{}).get('object_id') or r.get('task_id')) for r in rows}),
            'fit_groups':sorted(set(map(group_id,rows))),'heldout_samples':[]}
    if spec['observation'].get('metric_per_axis'):
        from .axis_metric import fit_axis_metric
        return fit_axis_metric(rows,spec,result)
    if len(result['fit_groups'])<cfg['min_groups']:return result
    all_pairs=pairs(rows,spec);head=fit_head(all_pairs,cfg)
    if head is None:return result
    folds=group_folds(rows,cfg['folds']);scores=[]
    for train,test in folds:
        train_pairs=pairs([rows[i] for i in train],spec)
        W=fit_head(train_pairs,cfg)
        if W is None:return result
        for x,y,kind,ids in pairs([rows[i] for i in test],spec):
            scores.append({'evidence':ids,'kind':kind,'error_mm':float(np.linalg.norm(x@W-y))})
    errors={k:[p['error_mm'] for p in scores if p['kind']==k] for k in ('action','repeat')}
    if not all(errors.values()):return result
    scale=min(np.linalg.norm(p[1]) for p in all_pairs if p[2]=='action' and np.linalg.norm(p[1])>0)
    ae=float(np.mean(errors['action']));re=float(np.mean(errors['repeat']))
    valid=ae<cfg['max_error_mm'] and re<spec['graph']['similar_radius']*scale
    result.update(status='ready' if valid else 'heldout_validation_failed',action_error_mm=ae,repeat_distance_mm=re,
                  action_resolution_mm=float(scale),heldout_samples=scores)
    if valid:result['coefficients']=head.tolist()
    return result
