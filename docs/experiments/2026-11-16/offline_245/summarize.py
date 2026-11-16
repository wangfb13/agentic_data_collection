"""Summarize saved per-window and per-transition held-out predictions."""
from pathlib import Path
import json
from collections import Counter, defaultdict
from statistics import mean
from math import ceil
root=Path(__file__).resolve().parent
def load(name):return [json.loads(l) for l in (root/(name+'.jsonl')).read_text().splitlines()]
def avg(xs):
    xs=list(xs)
    return mean(xs) if xs else None
def quant(xs,p):
    s=sorted(xs);i=(len(s)-1)*p;a=int(i)
    return s[a]+(s[ceil(i)]-s[a])*(i-a) if s else None
def offset(rows):
    result={}
    for key in ('est','diagnostic'):
        z=[r for r in rows if None not in r[key] and r['protocol'] and not r['documented_move']]
        err=[r[key][1]-r[key][0]-r['u'] for r in z]
        result[key]=dict(n=len(z),groups=len(set(r['group'] for r in z)),
             correct_sign=sum((r[key][1]-r[key][0])*r['u']>0 for r in z),plus=sum(r['u']>0 for r in z),
             mae=avg(map(abs,err)),bias=avg(err),p90=quant(list(map(abs,err)),.9),zero=avg(abs(r['u']) for r in z))
    result['all']=len(rows)
    result['reject_windows']=dict(Counter(s for r in rows for s in r['reasons'] if s!='ok'))
    result['by_campaign']={k:[sum(None not in r['est'] for r in rows if r['block'].split('/')[0]==k),sum(r['block'].split('/')[0]==k for r in rows)] for k in set(r['block'].split('/')[0] for r in rows)}
    return result
def decisions(data):
    output={}
    for key,act in [('graph','ag'),('direct','ad'),('neighbor','ad')]:
        end=[r for r in data if r['truth'] is not None];acts=[r for r in data if r['u'] is not None]
        issued=[r for r in acts if r[key]=='recoverable' and r[act] is not None]
        # Tolerance is for similarity to the archived action, not a claim of identical outcomes.
        near=[r for r in issued if r[act]*r['u']>0 and abs(r[act]-r['u'])<=1.]
        available=[r for r in acts if r[act] is not None]
        output[key]=dict(n=len(data),scored=len(end),ready_success=sum(r['truth'] and r[key]=='ready' for r in end),ready_failure=sum(not r['truth'] and r[key]=='ready' for r in end),
          available_success=sum(r['truth'] for r in end),available_fail=sum(not r['truth'] for r in end),
          states=dict(Counter(r[key] for r in data)),n_groups=len(set(r['group'] for r in data)),logged_transitions=len(acts),paired_invoked=len(issued),
          paired_same_sign=sum(r[act]*r['u']>0 for r in issued),paired_closeness_1mm=len(near),recorded_outcome_for_near=dict(Counter(str(r['next_success']) for r in near)),
          action_head_n=len(available),action_head_sign=sum(r[act]*r['u']>0 for r in available),action_head_mae=avg(abs(r[act]-r['u']) for r in available))
    return output
def dynamics(data):
    result={};inds={'centroid_uv_px':[0,1,3,4],'area_px':[2,5],'force_sdk':list(range(6,12))}
    def norm(r,k):return mean(a/b for a,b in zip(r['errors'][k],r['scale']))
    for k in ['graph','static','near_action','linear']:
        folds=defaultdict(list)
        for r in data:folds[r['fold']].append(norm(r,k))
        result[k]=dict(normalized_mae=avg(norm(r,k) for r in data),fold_mae={f:mean(v) for f,v in folds.items()},
                       raw={n:avg(mean(r['errors'][k][i] for i in ix) for r in data) for n,ix in inds.items()})
    result.update(n=len(data),groups=len({r['group'] for r in data}))
    result['wins']= {k:sum(norm(r,'graph')<norm(r,k) for r in data) for k in ['static','near_action','linear']}
    admitted=[r for r in data if r['uncertainty']<=1]
    result['uncertainty_gate']={'n':len(admitted),**{k:avg(norm(r,k) for r in admitted) for k in ['graph','static','near_action','linear']}}
    return result
res={}
for color,num,pairs in [('white',95,22),('green',389,114)]:
    od=load('offset_'+color);assert len(od)==pairs
    res[color]={'offset':offset(od)}
    for split in (['cv','trial','batch6'] if color=='white' else ['cv','trial']):
        rows=load(split+'_'+color);ds=[r for r in rows if r['type']=='decision']
        assert len({r['id'] for r in ds})==len(ds)==(12 if split=='batch6' else num)
        foldset=[r for r in rows if r['type']=='fold']
        assert len(foldset)==(1 if split=='batch6' else 5)
        res[color][split]={'decision':decisions(ds),'folds':foldset}
        if split!='trial':
            for v in ['native','measured']:
                ts=[r for r in rows if r.get('variant')==v];assert len(ts)==(2 if split=='batch6' else pairs)
                res[color][split][v]=dynamics(ts)
    blind={r['target']:r for r in load('blind_'+color)}
    native=[r for r in load('cv_'+color) if r.get('variant')=='measured']
    assert len(blind)==len(native)==pairs
    for r in native:assert blind[r['target']]['fold']==r['fold']
    res[color]['cv']['state_only_mae']=avg(mean(a/b for a,b in zip(blind[r['target']]['errors'],r['scale'])) for r in native)
(root/'summary.json').write_text(json.dumps(res,ensure_ascii=False,indent=2,allow_nan=False))
