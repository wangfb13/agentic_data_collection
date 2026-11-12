"""Controller-specific success probability boundaries, not extrema midpoints."""
from collections import defaultdict
import hashlib
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit


def fit_boundary(x,y,bootstrap=300,seed=0):
    x=np.asarray(x,float);y=np.asarray(y,int)
    result={'n':len(x),'successes':int(y.sum()),'range_mm':[float(x.min()),float(x.max())] if len(x) else None,
        'method':'monotone logistic, weak L2=0.001 on standardized slope, trial bootstrap',
        'boundary_probability':.5,'bootstrap_requested':bootstrap}
    if len(x)<3 or len(set(y))<2 or np.ptp(x)<1e-6:
        return {**result,'status':'no_boundary_identified','reason':'only_one_outcome_or_no_position_variation','boundary_mm':None,'bootstrap_boundaries':[]}
    def fit(a,b):
        mean=a.mean();scale=max(a.std(),1e-6);z=(a-mean)/scale
        def loss(w):
            p=w[0]+w[1]*z
            return np.logaddexp(0,p).sum()-b@p+.001*w[1]**2
        def jac(w):
            residual=expit(w[0]+w[1]*z)-b
            return np.array([residual.sum(),residual@z+.002*w[1]])
        opt=minimize(loss,[0.,0.],jac=jac,method='BFGS',options={'gtol':1e-7})
        intercept,slope=opt.x[0]-opt.x[1]*mean/scale,opt.x[1]/scale
        return intercept,slope
    intercept,slope=fit(x,y);boundary=-intercept/slope if abs(slope)>1e-8 else None
    rng=np.random.default_rng(seed);samples=[]
    for _ in range(bootstrap):
        ix=rng.integers(0,len(x),len(x));a,b=x[ix],y[ix]
        if len(set(b))<2 or np.ptp(a)<1e-6:continue
        u,v=fit(a,b)
        if abs(v)>1e-8:samples.append(float(-u/v))
    status='identified_within_sample_range' if boundary is not None and x.min()<=boundary<=x.max() else 'extrapolated_or_flat'
    grid=np.linspace(x.min(),x.max(),101)
    return {**result,'status':status,'intercept':float(intercept),'slope_per_mm':float(slope),'boundary_mm':boundary,
        'bootstrap_valid':len(samples),'boundary_ci95_mm':np.quantile(samples,[.025,.975]).tolist() if samples else None,
        'bootstrap_boundaries':samples,'curve':{'offset_mm':grid.tolist(),'success_probability':expit(intercept+slope*grid).tolist()}}


def success_boundaries(rows,config=None):
    cfg=config or {};groups=defaultdict(list)
    for r in rows:
        if r.get('evaluation_only') or r.get('phase')!='A' or not str(r.get('object_id','')).startswith('white'):continue
        if type(r.get('downstream_success')) is not bool:continue
        key=(r.get('protocol_id'),r.get('time_period'),r.get('T0_id'),r['object_id'])
        groups[key].append(r)
    fits=[];pairs=[]
    for (protocol,period,reference,obj),members in sorted(groups.items(),key=str):
        axis=members[0].get('a_axis_base');T0=members[0].get('T0_mm');values=[];labels=[];used=[]
        for r in members:
            pose=np.asarray(r.get('measured_tcp_mm'),float)
            if axis is None or np.asarray(T0).shape!=(4,4) or pose.shape!=(4,4):continue
            unit=np.asarray(axis,float);unit/=np.linalg.norm(unit)
            values.append(float((pose[:3,3]-np.asarray(T0)[:3,3])@unit));labels.append(int(r['downstream_success']));used.append(r['node_id'])
        fitted=fit_boundary(values,labels,bootstrap=cfg.get('boundary_bootstrap',300),seed=int(hashlib.sha256(str((protocol,period,reference,obj)).encode()).hexdigest()[:8],16)) if values else {'status':'no_boundary_identified','boundary_mm':None,'bootstrap_boundaries':[]}
        fits.append({**fitted,'protocol_id':protocol,'time_period':period,'T0_id':reference,'object_id':obj,
            'closure_floor':members[0].get('closure_floor'),'downstream_controller':members[0].get('downstream_controller'),
            'a_axis_base':axis,'node_ids':used})
    for light in fits:
        if light['object_id']!='white_light' or light['closure_floor']!=.336:continue
        heavy=next((r for r in fits if r['object_id']=='white_heavy' and all(r[k]==light[k] for k in ('protocol_id','time_period','T0_id'))),None)
        if heavy is None:continue
        a,b=light.get('boundary_mm'),heavy.get('boundary_mm');n=min(len(light['bootstrap_boundaries']),len(heavy['bootstrap_boundaries']))
        samples=np.asarray(heavy['bootstrap_boundaries'][:n])-np.asarray(light['bootstrap_boundaries'][:n])
        pairs.append({'protocol_id':light['protocol_id'],'time_period':light['time_period'],
            'downstream_controller':light['downstream_controller'],'heavy_minus_light_boundary_mm':None if a is None or b is None else b-a,
            'difference_ci95_mm':np.quantile(samples,[.025,.975]).tolist() if n else None,
            'interpretable_as_forcepolicy_2mm_test':False,'reason':'Different downstream controller; no ForcePolicy-labelled experiment in this comparison.'})
    return {'status':'computed' if fits else 'unavailable','fits':fits,'comparisons':pairs,
        'sign_convention':'+a is approach from pregrasp toward T0 (forward); -a is retreat (backward), in each recorded T0 base frame.',
        'front_back_scope':'Operational robot approach/retreat convention; not image left/right.',
        'all_success_rule':'No finite boundary inferred from all-success data; no successful-extrema midpoint used.',
        'limitation':'Small observational sample, monotonic logistic model assumption, controller-specific. Bootstrap intervals are descriptive.'}
