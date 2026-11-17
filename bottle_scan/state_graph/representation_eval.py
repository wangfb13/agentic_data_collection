"""Group-held-out tactile representation diagnostics. Inputs are archived node records."""
import argparse
from collections import defaultdict
from copy import deepcopy
import json
from pathlib import Path
import numpy as np
from sklearn.linear_model import Ridge,LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from .common import group_id,group_folds,independent,write,digest
from .noise import fit_noise,pairwise_distances
from .graph import build_graph
from .v5_metric import fit_metric
from .v5_profile import configure
from .skill_interface import load_spec


def coordinates(row,spec):
    from .facts import coordinate_evidence
    return [coordinate_evidence(row,name,spec) for name in ('r_offset_mm','h_offset_mm')]


def scene_ids(row,cfg):
    return set(row.get('representation_scenes',row.get('provenance',{}).get('representation_scenes',[]))) | set(cfg.get('scene_groups',{}).get(group_id(row),[]))


def raw_candidate(row,name):
    data=row.get('representations',{})
    if name in data:return data[name]
    obs=row.get('obs',{})
    if name=='legacy_geometry' and ('px' in str(row.get('extractor_versions',{}).get('tac','')).lower() or len(obs.get('tac',[]))==6):
        return obs.get('tac')
    if name=='calibrated_rh' and str(row.get('extractor_versions',{}).get('tac','')).startswith('calibrated_rh:'):
        return obs.get('tac')
    if name=='force':return obs.get('force')
    if name in ('mean','grid','attention','soft_argmax'):
        evidence=row.get('feature_evidence',{}).get('tac_embed',{}).get('recipe',{})
        pool=evidence.get('pool')
        if pool==name or (name=='mean' and pool=='mean_patch_tokens_excluding_cls_before_force_projector'):return obs.get('tac_embed')
    return None


def fold_vectors(rows,train,name,spec):
    sources=name.split('+');columns=[]
    for source in sources:
        if source=='tac_metric':
            settings=spec['observation'];input_name=settings['metric']['input_modality']
            # Train fold only, preserving the frozen readout used by this evaluation.
            prepared=[{**r,'obs':{**r['obs'],input_name:v}} for r in rows
                      if (v:=raw_candidate(r,settings['tac_embed_readout'])) is not None]
            if len(prepared)!=len(rows):return None
            gids={group_id(rows[i]) for i in train}
            model=fit_metric([r for r in prepared if group_id(r) in gids],spec)
            if model['status']!='ready':return None
            values=[np.asarray(raw_candidate(r,settings['tac_embed_readout']))@model['coefficients'] for r in rows]
        else:
            values=[raw_candidate(r,source) for r in rows]
        if any(v is None for v in values):return None
        columns.append(np.array(values,float))
    return np.concatenate(columns,axis=1)


def support(train,test,X,train_i,test_i,spec):
    local=deepcopy(spec)
    local['observation'].update(modalities={'candidate':{'version':'eval'}},tac_metric=False)
    def attach(records,indices):
        return [{**r,'obs':{'candidate':X[i].tolist()},'extractor_versions':{'candidate':'eval'}} for r,i in zip(records,indices)]
    a,b=attach(train,train_i),attach(test,test_i);noise=fit_noise(a,local)
    graph=build_graph(a,noise,local)
    D=pairwise_distances(a+b,noise)[len(a):,:len(a)]
    r=graph.get('radii',{}).get('support_distance',1.)
    known=np.array([graph['reachability'][spec['active_post_condition']][x['node_id']]['status']!='unsupported' for x in a])
    return [not np.any((d<=r)&known) for d in D]


def records(rows,X,tr,te,spec,candidate):
    scale=StandardScaler().fit(X[tr])
    z=scale.transform(X)
    train=[rows[i] for i in tr];held=[rows[i] for i in te]
    # Encoder noise covariance must be learned in a physically validated head.
    high=any(k in ('grid','mean','attention') for k in candidate.split('+'))
    if high:
        local=deepcopy(spec)
        readout=next(k for k in candidate.split('+') if k in ('grid','mean','attention'))
        local['observation']['tac_embed_readout']=readout
        recipe=candidate.replace(readout,'tac_metric')
        physical=fold_vectors(rows,tr,recipe,local)
        sup=[True]*len(te) if physical is None else support(train,held,physical,tr,te,local)
    else:
        sup=support(train,held,X,tr,te,spec)
    coords=[coordinates(r,spec) for r in rows]
    result={i:{'id':rows[i]['node_id'],'group':group_id(rows[i]),'errors':{},'outcomes':{},'unsupported':s,
               'distance_repeat':[],'distance_action':[],'angle_distance':[]} for i,s in zip(te,sup)}
    for k,var in enumerate(('r','h')):
        valid=[i for i in tr if coords[i][k] is not None]
        if len(set(group_id(rows[i]) for i in valid))<2 or len({coords[i][k] for i in valid})<2:continue
        model=Ridge().fit(z[valid],[coords[i][k] for i in valid])
        pred=model.predict(z[te])
        for i,y in zip(te,pred):
            if coords[i][k] is not None:result[i]['errors'][var]=abs(float(y-coords[i][k]))
    for c in spec['post_conditions']:
        cid=c['id']
        valid=[i for i in tr if rows[i]['is_terminal'] and rows[i]['post'][cid]['evaluated']]
        ys=[rows[i]['post'][cid]['success'] for i in valid]
        if len(set(ys))<2:continue
        model=LogisticRegression(max_iter=500).fit(z[valid],ys)
        for i,y in zip(te,model.predict_proba(z[te])[:,1]):
            if rows[i]['is_terminal'] and rows[i]['post'][cid]['evaluated']:
                result[i]['outcomes'][cid]=[int(rows[i]['post'][cid]['success']),float(y)]
    byid={rows[i]['node_id']:i for i in te}
    actions=[]
    for i in te:
        r=rows[i];prev=byid.get(r.get('previous_node_id'))
        if prev is not None and group_id(r)==group_id(rows[prev]) and (r.get('u_receipt') or {}).get('applied') and r.get('u_executed') is not None:
            u=np.asarray(r['u_executed']);d=np.linalg.norm(u)
            if d and np.count_nonzero(u)==1:actions.append((i,int(np.argmax(abs(u))),d,float(np.linalg.norm(z[i]-z[prev]))))
        for j in tr:
            s=rows[j]
            if r.get('window_index')==s.get('window_index')==0 and r.get('repeat_key') and r['repeat_key']==s.get('repeat_key'):
                result[i]['distance_repeat'].append(float(np.linalg.norm(z[i]-z[j])))
            ar,br=r.get('provenance',{}),s.get('provenance',{})
            if ar.get('angle_group') is not None and br.get('angle_group') is not None and ar['angle_group']!=br['angle_group']:
                if None not in coords[i]+coords[j] and np.linalg.norm(np.subtract(coords[i],coords[j]))<=spec['observation']['representations']['matched_position_tolerance_mm']:
                    result[i]['angle_distance'].append(float(np.linalg.norm(z[i]-z[j])))
    # Compare physical action sizes, never select by observed representation distance.
    for i,axis,mm,d in actions:
        smallest=min(a[2] for a in actions if a[1]==axis)
        if mm<=smallest*(1+spec['graph']['min_action_bin_relative_tolerance']):result[i]['distance_action'].append(d)
    return result


def aggregate(rs,cids):
    def median(xs):return float(np.median(xs)) if xs else None
    def mean(xs):return float(np.mean(xs)) if xs else None
    act=median([v for r in rs for v in r['distance_action']]);rep=median([v for r in rs for v in r['distance_repeat']])
    out={'resolution_noise_ratio':act/rep if act is not None and rep else None,
         'angle_distance':median([v for r in rs for v in r['angle_distance']]),
         'unsupported_fraction':mean([r['unsupported'] for r in rs if r['unsupported'] is not None])}
    for axis in ('r','h'):out['mae_'+axis]=mean([r['errors'][axis] for r in rs if axis in r['errors']])
    for cid in cids:
        pairs=[r['outcomes'][cid] for r in rs if cid in r['outcomes']]
        out['auc_'+cid]=float(roc_auc_score(*zip(*pairs))) if len({p[0] for p in pairs})>1 else None
    return out


def compare(a,b,cids,cfg):
    """Compare common held-out trials by paired group bootstrap on the original metrics."""
    if not a or {r['id'] for r in a}!={r['id'] for r in b}:return {'eligible':False,'reason':'unmatched_or_empty_support'}
    def indexed(records):
        result=defaultdict(list)
        for r in records:result[r['group']].append(r)
        return result
    a,b=indexed(a),indexed(b)
    gids=sorted(a);keys=['resolution_noise_ratio','mae_r','mae_h']+['auc_'+c for c in cids]
    if len(gids)<cfg['min_comparison_groups']:return {'eligible':False,'reason':'insufficient_groups'}
    def signed(x,y,k):
        if x[k] is None or y[k] is None:return None
        return (y[k]-x[k]) if k.startswith('mae_') else (x[k]-y[k])
    rng=np.random.default_rng(cfg['seed']);diffs={k:[] for k in keys}
    for _ in range(cfg['bootstrap_reps']):
        draw=rng.choice(gids,len(gids),replace=True)
        v=aggregate([r for g in draw for r in a[g]],cids)
        w=aggregate([r for g in draw for r in b[g]],cids)
        for k in keys:
            val=signed(v,w,k)
            if val is not None:diffs[k].append(val)
    quantile=(1-cfg['confidence'])/len(keys)
    bounds={k:float(np.quantile(v,quantile)) if len(v)>=cfg['min_valid_bootstrap_fraction']*cfg['bootstrap_reps'] else None for k,v in diffs.items()}
    return {'eligible':all(v is not None and v>=0 for v in bounds.values()),
            'improved':any(v is not None and v>0 for v in bounds.values()),'lower_bounds':bounds,'groups':gids}


def evaluate(rows,spec):
    rows=[r for r in rows if not independent(r) and r['window_valid']]
    ids=[r['node_id'] for r in rows]
    if len(ids)!=len(set(ids)):raise ValueError('Duplicate observation ID')
    cfg=spec['observation']['representations'];cids=[c['id'] for c in spec['post_conditions']]
    scenes=cfg['required_scenes'];names=cfg['candidates'];current=cfg['current']
    folds=group_folds(rows,cfg['folds'])
    report={'schema':'representation_ranking_v5','current':current,'evidence':ids,'scenes':{},'selectable':[],
            'splits':[[group_id(rows[i]) for i in te] for _,te in folds],
            'source_id':digest([(r['node_id'],r['extractor_versions']) for r in rows]),
            'distance_units':'Fold-training standardized Euclidean distance for resolution and invariance. Axis errors are mm.'}
    all_scores={}
    for name in names:
        samples={}
        for tr,te in folds:
            X=fold_vectors(rows,tr,name,spec)
            if X is not None:samples.update(records(rows,X,tr,te,spec,name))
        all_scores[name]={s:[v for i,v in samples.items() if s in scene_ids(rows[i],cfg)] for s in scenes}
        report['scenes'][name]={s:{**aggregate(v,cids),'n':len(v),'groups':len({r['group'] for r in v})} for s,v in all_scores[name].items()}
        report['scenes'][name]['all']=dict(aggregate(list(samples.values()),cids),n=len(samples))
    comparisons={}
    for name in names:
        if name==current:continue
        cs={s:compare(all_scores[name][s],all_scores[current][s],cids,cfg) for s in scenes}
        if cs and all(c['eligible'] for c in cs.values()) and any(c.get('improved') for c in cs.values()):
            report['selectable'].append(name)
        comparisons[name]=cs
    report['comparisons']=comparisons
    return report


def director_evidence(report,spec):
    from .facts import fact
    cfg=spec['observation']['representations'];current=cfg['current']
    if not spec['observation'].get('auto_select') or report.get('schema')!='representation_ranking_v5' or current!=report.get('current'):return [],[]
    selectable=[]
    for name in report['selectable']:
        scores=report.get('comparisons',{}).get(name,{})
        if (name in cfg['candidates'] and set(scores)==set(cfg['required_scenes'])
                and all(v.get('eligible') is True for v in scores.values()) and any(v.get('improved') is True for v in scores.values())):
            selectable.append(name)
    return [fact('representation_ranking',[current],report['evidence'],{'selectable':selectable,'source_id':report['source_id']})],selectable


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--spec',required=True)
    p.add_argument('--nodes',nargs='+',required=True);p.add_argument('--features',help='Optional JSONL keyed by node_id')
    p.add_argument('--out',required=True);a=p.parse_args(argv)
    rows=[json.loads(line) for path in a.nodes for line in Path(path).read_text().splitlines() if line.strip()]
    if a.features:
        features={r['node_id']:r for line in Path(a.features).read_text().splitlines() if (r:=json.loads(line))}
        for r in rows:
            f=features.get(r['node_id'],{});r['representations']=f.get('representations',{})
            if 'representation_scenes' in f:r['representation_scenes']=f['representation_scenes']
    dest=Path(a.out)
    if dest.exists():raise FileExistsError(dest)
    write(dest,evaluate(rows,configure(load_spec(a.spec))))
    return 0


if __name__=='__main__':raise SystemExit(main())
