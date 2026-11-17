"""Train-only repeated-observation noise with 2Σ and familywise thresholds."""
from collections import defaultdict
import numpy as np
from scipy.stats import chi2
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from .common import digest,independent,group_id
from .skill_interface import model_observation,repeat_key


def observation_vectors(row,spec):return model_observation(row,spec['observation']['modalities'])


def fit_noise(rows,spec):
    rows=[r for r in rows if not independent(r) and r.get('window_valid') is True]
    names=list(spec['observation']['modalities']);cfg=spec.get('noise',{});alpha=float(cfg.get('alpha',.05))
    result={'schema':'noise_v3','fit_node_ids':[r['node_id'] for r in rows],'modalities':{},'alpha':alpha,'covariance_multiplier':2.,'active_modalities':names}
    combine=cfg.get('combine','max')
    if combine not in ('joint','max'):raise ValueError('Unknown noise combination')
    settings=dict(spec['observation']['modalities'])
    if combine=='joint':
        result.update(combine=combine,active_modalities=[])
    if spec.get('observation',{}).get('tac_metric'):
        from .v5_metric import fit_metric
        result['tac_metric']=fit_metric(rows,spec)
        if result['tac_metric']['status']=='ready':settings['tac_metric']={'weight':1.}
    for name,setting in settings.items():
        is_encoder=(name in ('tac_embed','tactile_embedding') or name==spec.get('observation',{}).get('metric',{}).get('input_modality')
                    or setting.get('encoder_features') is True or 'dino' in setting.get('extractor','').lower())
        if combine=='joint' and is_encoder:
            result['modalities'][name]={'status':'projected_only','reason':'encoder noise is estimated in the validated millimetre head'}
            continue
        X=np.array([raw(r,name,result) for r in rows]);item={'status':'insufficient_data','weight':float(setting.get('weight',1))}
        result['modalities'][name]=item
        if len(X)<3:continue
        mean=X.mean(0);components=None
        if X.shape[1]>cfg.get('pca_dim',32):
            fit=PCA(min(cfg.get('pca_dim',32),len(X)-1),svd_solver='full').fit(X)
            components=fit.components_;Y=fit.transform(X);mean=fit.mean_
        else:Y=X-mean
        groups=defaultdict(list);seen=set()
        for i,r in enumerate(rows):
            if int(r.get('window_index',0))!=0 or group_id(r) in seen:continue
            seen.add(group_id(r))
            key=r.get('repeat_key') or repeat_key(r,spec)
            if combine=='joint':
                p=r.get('provenance',{});key=(key,str(p.get('calibration_epoch',r.get('calibration_epoch'))))
            groups[key].append(i)
        repeats=[v for v in groups.values() if len(v)>1]
        item.update(mean=mean.tolist(),components=None if components is None else components.tolist(),input_dim=X.shape[1],dim=Y.shape[1],repeat_sources=[[rows[i]['node_id'] for i in ids] for ids in repeats])
        if not repeats: item['status']='insufficient_repeats';continue
        # Group centering consumes one degree of freedom. Without this scaling,
        # two-repeat groups estimate half the single-observation variance.
        residual=np.concatenate([(Y[ids]-Y[ids].mean(0))*np.sqrt(len(ids)/(len(ids)-1)) for ids in repeats])
        if combine=='joint' and len(residual)<Y.shape[1]*cfg['repeat_dimension_multiplier']:
            item.update(status='insufficient_repeats',repeat_observations=len(residual),repeat_dof=sum(len(ids)-1 for ids in repeats))
            continue
        cov=LedoitWolf(assume_centered=True).fit(residual).covariance_
        floor=max(float(np.trace(cov)/len(cov))*1e-9,1e-12);cov+=np.eye(len(cov))*floor
        item.update(status='ready',single_covariance=cov.tolist(),covariance=(2*cov).tolist(),precision=np.linalg.inv(2*cov).tolist(),
                    tau=float(np.sqrt(chi2.ppf(1-alpha/len(names),Y.shape[1]))),residual_count=len(residual),floor=floor,
                    repeat_mean_correction='sqrt(n/(n-1)) per repeat group before Ledoit-Wolf shrinkage')
        if combine=='joint':result['active_modalities'].append(name)
    if 'tac_metric' in result['active_modalities'] and spec.get('graph',{}).get('resolution_scaled_radius'):
        from .v5_values import resolution
        from .graph import observed_edges
        M=metric_distance_mm(rows,result)
        base=pairwise_distances(rows,result)
        scale,_=resolution(rows,observed_edges(rows,spec),base,spec,M)
        ratios=[a['scale']/a['metric_scale_mm'] for a in scale['by_axis'].values() if a.get('metric_scale_mm') and a['scale']]
        result['metric_resolution']=scale
        if ratios:result['metric_normalized_per_mm']=float(max(ratios))
    result['model_id']=digest(result)
    return result


def raw(row,name,noise):
    if name=='tac_metric':
        head=noise['tac_metric'];return model_observation(row,[head['input']])[head['input']]@np.asarray(head['coefficients'])
    return model_observation(row,[name])[name]


def transform(row,name,noise):
    item=noise['modalities'][name];x=raw(row,name,noise)-np.asarray(item['mean'])
    return x if item['components'] is None else np.asarray(item['components'])@x


def feature_vector(row,noise,names=None):
    names=names or noise['active_modalities'];result=[]
    for name in names:
        item=noise['modalities'][name]
        if item['status']!='ready':raise ValueError('Noise unavailable for '+name)
        x=transform(row,name,noise);scale=np.sqrt(np.maximum(np.diag(item['single_covariance']),1e-12))
        result.extend(x/scale)
    return np.asarray(result)


def metric_distance_mm(rows,noise):
    from scipy.spatial.distance import cdist
    x=np.array([raw(r,'tac_metric',noise) for r in rows])
    return cdist(x,x)


def pairwise_distances(rows,noise):
    from .graph import gate
    n=len(rows);D=np.zeros((n,n));M=len(noise['active_modalities'])
    if n==0:return D
    joint=noise.get('combine')=='joint'
    if not M:
        D.fill(np.inf);np.fill_diagonal(D,0);return D
    for name in noise['active_modalities']:
        item=noise['modalities'][name]
        if item['status']!='ready':return np.full((n,n),np.inf)
        X=np.array([transform(r,name,noise) for r in rows]);P=np.asarray(item['precision'])
        XP=X@P;q=np.einsum('ij,ij->i',XP,X)
        distances=np.sqrt(np.maximum(0,q[:,None]+q[None,:]-2*XP@X.T))
        if joint:
            D+=distances**2
        else:
            distances*=item['weight']/np.sqrt(chi2.ppf(1-noise['alpha']/M,item['dim']))
            D=np.maximum(D,distances)
    if joint:
        df=sum(noise['modalities'][name]['dim'] for name in noise['active_modalities'])
        D=np.sqrt(D/chi2.ppf(1-noise['alpha'],df))
    if noise.get('metric_normalized_per_mm'):
        # Preserve the joint noise test, and also bound the physical projection
        # distance in units of its measured minimum-action scale.
        D=np.maximum(D,metric_distance_mm(rows,noise)*noise['metric_normalized_per_mm'])
    np.fill_diagonal(D,0.)
    for i,a in enumerate(rows):
        for j in range(i+1):
            if not gate(a,rows[j]):D[i,j]=D[j,i]=np.inf
    return D


def measurement_distance(a,b,noise):return float(pairwise_distances([a,b],noise)[0,1])


def drift_report(rows,noise):
    """Reference-window shifts are evidence, never a filter or a graph gate."""
    sessions=defaultdict(list)
    for r in rows:
        if r.get('is_reference'):sessions[r['session']].append(r)
    keys=sorted(sessions);result=[]
    if not keys:return result
    anchor=keys[0]
    for key in keys[1:]:
        a,b=sessions[anchor],sessions[key]
        if len(a)<2 or len(b)<2:continue
        ma={**a[0],'obs':{n:np.median([r['obs'][n] for r in a],axis=0).tolist() for n in a[0]['obs']}}
        mb={**b[0],'obs':{n:np.median([r['obs'][n] for r in b],axis=0).tolist() for n in b[0]['obs']}}
        distance=measurement_distance(ma,mb,noise)
        result.append({'session':key,'reference_session':anchor,'distance':distance,'shift_detected':bool(np.isfinite(distance) and distance>1),
                       'node_ids':[r['node_id'] for r in a+b],'meaning':'Observed reference shift; may include recorded or unrecorded condition differences; no automatic exclusion.'})
    return result
