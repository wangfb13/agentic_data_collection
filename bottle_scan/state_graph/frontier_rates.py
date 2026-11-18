"""A frontier is a connected chain of sampled bins meeting the success-rate gate."""
from collections import defaultdict
import numpy as np
from scipy.special import ndtri


def lower(successes, totals, confidence):
    n=np.maximum(1,totals); p=successes/n; z=ndtri((1+confidence)/2)
    return (p+z*z/(2*n)-z*np.sqrt(p*(1-p)/n+z*z/(4*n*n)))/(1+z*z/n)


def statistics(observed, sign, width, limit, cfg):
    # Magnitude bins (0,w], (w,2w], ...; the reference zero belongs to bin 0.
    grouped=defaultdict(dict)
    for group,x,ok,cause in observed:
        if type(ok) is not bool or not 0<=sign*x<=limit:continue
        k=max(0,int(np.ceil(abs(x)/width))-1)
        grouped[group][k]=grouped[group].get(k,True) and ok
    bins=int(np.ceil(limit/width))
    groups=sorted(grouped)
    N=np.zeros((len(groups),bins),int);S=np.zeros_like(N)
    for i,g in enumerate(groups):
        for k,v in grouped[g].items(): N[i,k]=1;S[i,k]=int(v)
    def chain(s,n):
        ok=(n>0)&(lower(s,n,cfg['confidence'])>=cfg['frontier_success_target'])
        return int(np.cumprod(ok).sum())
    n=N.sum(0);s=S.sum(0);k=chain(s,n)
    frontier=sign*min(k*width,limit) if k else None
    ci=None
    if groups:
        rng=np.random.default_rng(cfg['seed'])
        draws=rng.multinomial(len(groups),np.full(len(groups),1/len(groups)),size=cfg['bootstrap_reps'])
        # Zero includes resamples with no certified first bin.
        edges=[sign*min(chain(ss,nn)*width,limit) for ss,nn in zip(draws@S,draws@N)]
        alpha=(1-cfg['confidence'])/2
        ci=np.quantile(edges,[alpha,1-alpha]).tolist()
    near=range(max(0,k-2),min(k+1,bins))
    supported=bool(k and all(n[j]>=cfg['frontier_min_trials'] for j in near))
    return dict(boundary_definition='connected_wilson_bins',verified_edge_mm=frontier,frontier_ci_mm=ci,
                ci_method='trial-group bootstrap including empty connected ranges',boundary_bin=k-1 if k else None,
                neighbors_confirmed=supported,bins=[dict(index=j,edge_mm=sign*min((j+1)*width,limit),
                evaluated=int(n[j]),successes=int(s[j]),lower=float(lower(s[j],n[j],cfg['confidence'])) if n[j] else None)
                for j in range(bins)])


def overlap(a,b):
    if not (a.get('neighbors_confirmed') and b.get('neighbors_confirmed')):return False
    x,y=a['frontier_ci_mm'],b['frontier_ci_mm']
    return max(x[0],y[0])<=min(x[1],y[1])


def next_bin(metrics,spec,var,side):
    bins=metrics['bins']; sign=1 if side=='positive' else -1
    if not bins:return None
    k=metrics['boundary_bin']; j=0 if k is None else k+1
    if j>=len(bins):return None
    width=metrics['resolution_mm']
    # Explore outwards while uncertainty is local; never jump over an empty bin.
    outer=[b for b in bins[j:] if b['evaluated'] and b['successes']<b['evaluated']]
    target=min(2*j+1,len(bins)-1) if k is not None else 0
    if outer:target=(j+outer[0]['index'])//2
    unsampled=next((i for i in range(j,target+1) if bins[i]['evaluated']==0),None)
    if unsampled is not None:target=unsampled
    if bins[j]['evaluated'] and bins[j]['lower'] is not None and bins[j]['lower']<spec['facts']['frontier_success_target']:target=j
    return sign*(target*width+abs(bins[target]['edge_mm']))/2
