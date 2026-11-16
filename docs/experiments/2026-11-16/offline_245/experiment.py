"""Read-only 5090 experiment. Run with the repository on sys.path via stdin."""
from pathlib import Path
from functools import lru_cache
from collections import Counter
from copy import deepcopy
import json
import sys
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from bottle_scan.state_graph.common import group_id
from bottle_scan.state_graph.noise import fit_noise,feature_vector,pairwise_distances
from bottle_scan.state_graph.graph import build_graph
from bottle_scan.state_graph.labels import generate_labels
from bottle_scan.state_graph.skill import train_models,SymbolicRecoverySkill
from bottle_scan.state_graph.dynamics import fit_dynamics,transitions,target_vector,predict
from bottle_scan.state_graph.value import fit_regression,predict_regression
from bottle_scan.state_graph.policy import predict_policy

W=Path('outputs/state_graph/campaign_white_a_v4_fixed5_history')
G=Path('outputs/state_graph/20261112_symbolic_v3')
def read(p):return json.loads(Path(p).read_text())
def lines(p):return [json.loads(l) for l in Path(p).read_text().splitlines() if l]
def emit(x): print(json.dumps(x,allow_nan=False,default=lambda v:v.tolist()),flush=True)

def dataset(color):
    if color=='white':
        s=read(W/'revisions/0007/spec.json')
        rr=read(W/'seed_nodes.json')+[r for p in sorted((W/'trials').glob('batch_[1-6]_*/result.json')) for r in read(p)['nodes']]
        cid=s['active_post_condition']
    else:
        rr=[r for r in lines(G/'nodes.jsonl') if r['task_id']=='green_bottle|tcp_translation_a']
        s=read(G/'tasks/352f6945568f/spec.json');cid='handoff_e1e2_squeeze_floor0336'
    s['observation']['modalities']={k:s['observation']['modalities'][k] for k in ('tac','force')}
    s['post_conditions']=[c for c in s['post_conditions'] if c['id']==cid];s['active_post_condition']=cid
    for r in rr:
        for key in ['obs','extractor_versions']:r[key]={m:r[key][m] for m in ('tac','force')}
    assert len(set(r['node_id'] for r in rr))==len(rr)
    return rr,s,cid

def block(r):
    p=Path(r['session'])
    if '/trials/batch_' in str(p):
        i=p.parts.index('trials')
        return p.parts[i-1]+'/'+p.parts[i+1].rsplit('_',1)[0]
    return str(p)

@lru_cache(None)
def events(session):return {e['node_id']:e for e in lines(Path(session)/'events.jsonl') if e['event']=='state_graph_window'}

def net_and_probes(t,color):
    import h5py
    a,b=(t[k] for k in ('source','target'));rc=b['u_receipt']
    if color=='white':
        wa,wb=[events(x['session'])[x['node_id']] for x in (a,b)]
        T1,T2=[np.array(w['measured_tcp_mm']) for w in (wa,wb)]
        probes=[w['probe'] for w in (wa,wb)]
        sess=read(Path(b['provenance']['streams']).parent/'session_meta.json')
        moved=any((w.get('risk') or {}).get('object_moved') is True for w in (wa,wb))
    else:
        pr=b['provenance'];meta=read(pr['meta_path']);sess=read(pr['session_meta_path'])
        with h5py.File(pr['h5_path']) as f:
            stream=f['streams/tcp_measured']
            def pose(key):
                rec=rc[key];idx=rec['index']
                assert int(stream['monotonic_ns'][idx])==rec['timestamp_ns']
                return np.asarray(stream['data'][idx])
            T1,T2=pose('before_tcp_receipt'),pose('after_tcp_receipt')
        probes=[]
        for r in (a,b):
            v=meta
            for k in r['provenance']['probe_pointer'].strip('/').split('/'):
                v=v[int(k) if isinstance(v,list) else k]
            probes.append(v)
        moved=bool(meta.get('object_moved') or meta.get('pushed') or meta.get('label') in ('pushed','displaced'))
    assert not sess.get('simulated',False)
    axis=T1[:3,:3]@sess['config']['axes_tcp']['a'];v=T2[:3,3]-T1[:3,3];u=float(v@axis)
    return u,float(np.linalg.norm(v-u*axis)),probes,sess,moved

def observation_offset(color,rr,s):
    from bottle_scan.contact_policy import ContactPolicy
    from skills.contact_position_a.contact_reference import probe_protocol
    from bottle_scan.contact_policy import feature_vector as contact_x, _predict
    if color=='white':
        m=read('outputs/local_contact_a_v1/offset.json');ref=read('skills/contact_position_a/white_reference.json')
        proto=ref['protocol'];assert ref['reference_digest']==m['contact_reference_digest']
    else:
        m=read('outputs/skill_offline_20261106_v2/bottle_skill/model.json')
        proto=probe_protocol(read(Path(m['source_trials'][0]['session'])/'session_meta.json')['config'])
    assert m['settings']['include_successful_B'] is False and all(o['label_source'].startswith('A_') for o in m['source_trials'])
    digests={r['metadata_sha256'] for r in m['source_trials']}
    limits=m.get('trained_offset_range_mm') or [min(x['measurement']['x_mm'] for x in m['source_trials']),max(x['measurement']['x_mm'] for x in m['source_trials'])]
    for t in transitions(rr,s):
        a,b=(t[k] for k in ('source','target'));truth,off,probes,sess,moved=net_and_probes(t,color)
        assert not any(r['provenance'].get('metadata_sha256') in digests for r in (a,b))
        compatible=probe_protocol(sess['config'])==proto
        pol=ContactPolicy(m,sess['config']['servo'])
        def estimate(p):
            ds=p.get('descriptors') or []
            if not p.get('valid') or len(ds)!=2 or not all(d.get('valid') for d in ds):return None,None,'invalid_probe'
            v=contact_x(ds,dd_pool=m['feature_definition']['dd_pool'])
            v=(v-pol.mean)/pol.scale
            raw=-float(np.mean([_predict(n,v[None])[0]*m['scaling']['y_scale'] for n in m['networks']]))
            if any(d.get('touches_edge') for d in ds):return None,raw,'edge'
            pred=pol.predict(ds)
            if not pred['valid']:return None,raw,pred['reason']
            est=-pred['action_mm']
            if not limits[0]<=est<=limits[1]:return None,raw,'outside_training_range'
            return est,raw,'ok'
        ep=[estimate(p) for p in probes]
        emit(dict(type='offset',color=color,source=a['node_id'],target=b['node_id'],group=group_id(a),block=block(a),protocol=compatible,
                  documented_move=moved,model=m['model_id'],u=truth,receipt=t['u'][0],off=off,est=[e[0] for e in ep],
                  diagnostic=[e[1] for e in ep],reasons=[e[2] for e in ep],limits=limits))

def train_compare(color,fold,tr,te,s,cid,trial_split=False):
    assert not ({group_id(r) for r in tr}&{group_id(r) for r in te})
    if not trial_split:assert not ({block(r) for r in tr}&{block(r) for r in te})
    noise=fit_noise(tr,s);dyn=fit_dynamics(tr,noise,s);g=build_graph(tr,noise,s,dyn);labels=generate_labels(g,tr,s)
    bundle,_=train_models(tr,g,labels,noise,dyn,s);skill=SymbolicRecoverySkill(bundle)
    part=lambda r:r['post'][cid]['success'] if r['post'][cid]['evaluated'] else None
    end=[r for r in tr if part(r) is not None];y=np.array([int(part(r)) for r in end])
    X=lambda r:feature_vector(r,noise)
    logit=make_pipeline(StandardScaler(),LogisticRegression(C=1,max_iter=1000)).fit([X(r) for r in end],y) if len(set(y))==2 else None
    pri=(sum(y)+1)/(len(y)+2)
    edges=transitions(tr,s);held=transitions(te,s);gpi=bundle['models'][cid]['pi'];direct=None
    demo=[e for e in edges if e['edge']['executable']]
    if len(demo)>=6:
        direct=fit_regression(np.array([X(t['source']) for t in demo]),np.array([t['u'] for t in demo]),
                              np.array([t['group_id'] for t in demo]),np.ones(len(demo)),ensemble=5,seed=0)
    D=pairwise_distances(tr+te,noise)[len(tr):,:len(tr)];demids={t['source']['node_id'] for t in demo}
    demidx=[j for j,r in enumerate(tr) if r['node_id'] in demids]
    prefix=dict(color=color,fold=fold)
    emit(dict(type='fold',**prefix,tr=len(tr),te=len(te),end=dict(Counter(str(bool(v)) for v in y)),
              actions=len(demo),blocks_test=sorted({block(r) for r in te}),graph_actions=gpi.get('n'),dynamics=dyn['status']))
    maxu=s['safety_envelope']['max_action'][0];from_feedback={t['source']['node_id']:t for t in held}
    for j,r in enumerate(te):
        pr=skill.infer(r);state=pr['state'];gv=pr['value'].get('p_hat')
        gp=predict_policy(gpi,r,noise);ga=(gp.get('u') or [None])[0]
        dp=float(logit.predict_proba([X(r)])[0,1]) if logit else float(pri)
        da=sd=None
        if direct:
            yy,ss=predict_regression(direct,X(r,)[None]);da=float(np.clip(yy[0,0],-maxu,maxu));sd=ss[0,0]
        supported=np.min(D[j])<=s['runtime'].get('support_distance',1)
        ds='Unknown'
        if supported and dp>=s['post_conditions'][0].get('ready_p',.8):ds='ready'
        elif supported and dp<=.5 and da is not None and demidx and np.min(D[j,demidx])<=s['runtime'].get('support_distance',1) and sd<=s['runtime'].get('max_action_std',4) and abs(da)>1e-6:ds='recoverable'
        # A second graph-free comparator uses only directly measured nearby endpoints.
        # At most one vote per physical trial, at most ten trials, and a Beta(1,1) prior.
        votes={}
        for i in np.argsort(D[j],kind='stable'):
            if D[j,i]>s['runtime'].get('support_distance',1) or len(votes)==10:break
            truth=part(tr[i])
            if truth is not None:votes.setdefault(group_id(tr[i]),truth)
        pk=(1+sum(votes.values()))/(2+len(votes))
        knn='Unknown'
        if votes and pk>=s['post_conditions'][0].get('ready_p',.8):knn='ready'
        elif votes and pk<=.5 and da is not None and demidx and np.min(D[j,demidx])<=s['runtime'].get('support_distance',1) and sd<=s['runtime'].get('max_action_std',4) and abs(da)>1e-6:knn='recoverable'
        nxt=from_feedback.get(r['node_id'])
        emit(dict(type='decision',**prefix,id=r['node_id'],group=group_id(r),phase=r['phase'],truth=part(r),graph=state,
                  direct=ds,neighbor=knn,pk=pk,pg=gv,pd=dp,ag=ga,ad=da,u=nxt['u'][0] if nxt else None,
                  next_success=part(nxt['target']) if nxt else None))
    if not held or trial_split:return
    # Compare changes using training-only per-target scales. Conditional baseline averages
    # up to five closest same-sign training receipts, no test outcomes or test contacts.
    def dynamics_scores(fitted,train_edges,test_edges,variant):
        mapping=fitted['targets'];target=lambda r:target_vector(r,mapping)
        scale= np.std(np.array([target(e['target'])-target(e['source']) for e in train_edges]),axis=0)
        assert np.all(scale>1e-9)
        for e in test_edges:
            u=e['u'][0];pre=target(e['source']);actual=target(e['target'])
            same=[t for t in train_edges if np.sign(t['u'][0])==np.sign(u)] or train_edges
            near=sorted(same,key=lambda t:(abs(t['u'][0]-u),t['source']['node_id']))[:5]
            mean=np.mean([target(t['target'])-target(t['source']) for t in near],axis=0)
            pr=predict(fitted,e['source'],[u]);candidates={'graph':np.asarray(pr['mean']),'static':pre,'near_action':pre+mean}
            # Simple action-conditioned linear baseline trained by the same repository fitter.
            from bottle_scan.state_graph.dynamics import input_vector,_design
            x=input_vector(e['source'],[u],fitted['inputs']);xm=fitted['ensemble']['x_scaler'].transform(_design(x,fitted['ensemble']['projectors']))
            candidates['linear']=pre+fitted['linear_baseline'].predict(xm)[0]
            emit(dict(type='dynamics',variant=variant,**prefix,source=e['source']['node_id'],group=group_id(e['source']),
                      target=e['target']['node_id'],u=u,scale=scale,uncertainty=pr['uncertainty'],errors={k:np.abs(v-actual) for k,v in candidates.items()}))
    if dyn['status']=='trained':dynamics_scores(dyn,edges,held,'native')
    # Refit with contact-to-contact measured displacement (current a axis), for both train/test.
    def netrows(rows,eds):
        changed={}
        for e in eds:
            a=e['source'];b=e['target'];u=net_and_probes(e,color)[0]
            row={**b,'u_executed':[u]}
            if 'action_executed' in row:row['action_executed']={**row['action_executed'],'u':[u]}
            changed[b['node_id']]=row
        return [changed.get(r['node_id'],r) for r in rows]
    tr2=netrows(tr,edges);te2=netrows(te,held)
    dm=fit_dynamics(tr2,noise,s)
    if dm['status']=='trained':dynamics_scores(dm,transitions(tr2,s),transitions(te2,s),'measured')

def state_only(color,fold,tr,te,s):
    # Hold architecture, splits and post-contact targets fixed; remove the action input.
    assert not ({block(r) for r in tr}&{block(r) for r in te})
    def blank(rows):
        adjusted=[]
        for r in rows:
            if r.get('previous_node_id'):
                r={**r,'u_executed':[0.]}
                if 'action_executed' in r:r['action_executed']={**r['action_executed'],'u':[0.]}
            adjusted.append(r)
        return adjusted
    blank_tr,blank_te=blank(tr),blank(te)
    for old,new in [(tr,blank_tr),(te,blank_te)]:
        assert [x['target']['node_id'] for x in transitions(old,s)]==[x['target']['node_id'] for x in transitions(new,s)]
    held=transitions(blank_te,s)
    if not held:return
    dm=fit_dynamics(blank_tr,fit_noise(tr,s),s)
    assert dm['status']=='trained'
    for e in held:
        v=predict(dm,e['source'],[0.])['mean']
        y=target_vector(e['target'],dm['targets'])
        emit(dict(type='state_only',color=color,fold=fold,target=e['target']['node_id'],errors=abs(y-v)))

def main():
    which,color=sys.argv[1:3];rows,s,cid=dataset(color)
    if which=='offset':observation_offset(color,rows,s);return
    if which=='batch6':
        test=[r for r in rows if '/batch_6_91578e00_' in r['session']]
        train=[r for r in rows if r not in test]
        import joblib
        frozen=joblib.load(W/'revisions/0007/models/bundle.joblib');ids={r['node_id'] for r in train}
        assert ids=={r['node_id'] for r in frozen['support']}
        train_compare(color,'batch6',train,test,s,cid);return
    blocks=[group_id(r) if which=='trial' else block(r) for r in rows]
    for k,(tr,te) in enumerate(GroupKFold(5).split(rows,groups=blocks)):
        a,b=[rows[i] for i in tr],[rows[i] for i in te]
        if which=='blind':state_only(color,str(k),a,b,s)
        else:train_compare(color,str(k),a,b,s,cid,trial_split=which=='trial')

if __name__=='__main__':main()
