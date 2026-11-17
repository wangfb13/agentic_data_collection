"""Generic V / policy / dynamics bundle and observe-act-observe runtime."""
from pathlib import Path
import math
import time
import numpy as np
import joblib
from .common import digest,write
from .skill_interface import post_spec,model_observation,action_grid,normalize_action
from .value import fit_value,predict_value
from .policy import fit_policy,predict_policy
from .dynamics import predict as predict_dynamics


def train_models(rows,graph,labels,noise,dynamics,spec,out=None):
    models={}
    for c in spec['post_conditions']:
        models[c['id']]={'V':fit_value(rows,labels,noise,spec,c['id']),
                         'pi':fit_policy(rows,graph,labels,noise,spec,c['id'])}
    support=[{k:r[k] for k in ('node_id','task_id','window_valid','obs','extractor_versions')} for r in rows]
    bundle={'schema':'skill_bundle_v4','support':support,'labels':labels,'spec':spec,'noise':noise,'dynamics':dynamics,'models':models,'graph':graph,
            'graph_version':graph['graph_version'],'model_id':digest([graph['graph_version'],spec,noise['model_id']])[:24]}
    status={'post_conditions':{c:{k:{q:v.get(q) for q in ('status','n','required','inferred_samples')} for k,v in ms.items()} for c,ms in models.items()},
            'dynamics_status':dynamics.get('status'),'active_post_condition':spec['active_post_condition'],
            'active_model_trained':all(m.get('status')=='trained' for m in models[spec['active_post_condition']].values()),
            'trained_does_not_mean_independent_B_passed':True,'robot_started':False}
    if out:
        p=Path(out);p.mkdir(parents=True,exist_ok=True);joblib.dump(bundle,p/'bundle.joblib');write(p/'training_status.json',status)
    return bundle,status


def optional_plan(bundle,row,policy_prediction):
    spec=bundle['spec'];cfg=spec.get('planning',{})
    if not cfg.get('enabled',False) or bundle['dynamics'].get('status')!='trained':return None
    cid=spec['active_post_condition'];candidate=np.asarray(policy_prediction['u']);grid=action_grid(spec,bundle['graph']['edges'])
    radius=cfg.get('radius',float(np.max(spec['safety_envelope']['max_action']))/2)
    choices=[]
    for u in grid:
        if np.linalg.norm(u-candidate)>radius:continue
        pred=predict_dynamics(bundle['dynamics'],row,u)
        if pred['uncertainty']>spec.get('dynamics',{}).get('max_uncertainty',1):continue
        simulated={**row,'obs':{n:list(v) for n,v in row['obs'].items()}};offset=0
        for n,indices in bundle['dynamics']['targets'].items():
            for i in indices:simulated['obs'][n][i]=float(pred['mean'][offset]);offset+=1
        value=predict_value(bundle['models'][cid]['V'],simulated,bundle['noise'])
        if value['status']=='predicted':choices.append((value['V']+spec.get('graph',{}).get('step_cost',1),u.tolist()))
    if not choices:return None
    score,u=min(choices)
    return {'u':u,'score':score,'disagreement':not np.allclose(u,candidate,atol=cfg.get('disagreement_tolerance',.1)),
            'approximation':'Unmodeled observation components remain at their latest measured values.'}


class SymbolicRecoverySkill:
    def __init__(self,bundle,clusters=None):
        self.bundle=bundle;self.spec=bundle['spec'];self.clusters=clusters;self.events=[]

    def infer(self,row,*,readiness_only=False):
        model_observation(row,self.spec['observation']['modalities'])
        cid=self.spec['active_post_condition'];condition=post_spec(self.spec,cid)
        models=self.bundle['models'][cid];v=predict_value(models['V'],row,self.bundle['noise'])
        if v['status']!='predicted':return {'state':'Unknown','value':v,'policy':None}
        from .noise import measurement_distance
        if row.get('task_id')!=self.bundle['graph']['task_id']:
            return {'state':'Unknown','reason':'different_task','value':v,'policy':None}
        support=[]
        for anchor in self.bundle.get('support',[]):
            if anchor['extractor_versions']!=row.get('extractor_versions'):continue
            d=measurement_distance(row,anchor,self.bundle['noise'])
            if d<=self.spec.get('runtime',{}).get('support_distance',1.):
                label=self.bundle['graph']['reachability'][cid][anchor['node_id']]
                support.append((d,label['status'],anchor['node_id']))
        cfg=self.spec.get('runtime',{})
        beta=v.get('exit_probability');invocation=v.get('invocation_probability')
        ready=(any(s[1]=='ready' for s in support) and beta is not None and beta>=cfg.get('exit_classifier_p',.5)
               and v['p_hat']>=condition.get('ready_p',.8))
        if ready:return {'state':'ready','value':v,'policy':None,'plan':None,'support':support}
        if readiness_only:
            return {'state':'not_ready' if support else 'Unknown','value':v,'support':support,
                    'reason':'post_condition_not_confirmed' if support else 'outside_ready_model_support'}
        supported=any(s[1]=='recoverable' for s in support)
        if not supported or invocation is None or invocation<cfg.get('invocation_classifier_p',.5):
            return {'state':'Unknown','reason':'no_supported_recovery','value':v,'policy':None,'support':support}
        pi=predict_policy(models['pi'],row,self.bundle['noise'])
        if pi['status']!='predicted':return {'state':'Unknown','reason':'policy_untrained','value':v,'policy':pi}
        if max(pi['std'])>cfg.get('max_action_std',4.):
            return {'state':'Unknown','reason':'action_uncertainty','value':v,'policy':pi}
        if np.linalg.norm(pi['u'])<cfg.get('min_action_norm',1e-6):
            return {'state':'Unknown','reason':'no_nonzero_recovery_action','value':v,'policy':pi}
        plan=optional_plan(self.bundle,row,pi)
        return {'state':'recoverable','value':v,'policy':pi,'plan':plan,'support':support,
                'action':normalize_action((plan or pi)['u'],self.spec)}

    def execute(self,adapter):
        cid=self.spec['active_post_condition'];maximum=self.spec.get('runtime',{}).get('max_steps',3)
        def emit(event):
            value={'schema':'skill_event_v3','skill_id':self.spec['skill_id'],'post_condition':cid,'graph_version':self.bundle['graph_version'],'t_ns':time.time_ns(),**event}
            self.events.append(value)
            if hasattr(adapter,'emit'):adapter.emit(value)
        row=adapter.observe()
        for step in range(maximum+1):
            result=self.infer(row);emit({'event':'observation_window','step':step,'window':row,'prediction':result})
            if result['state']=='ready':
                cluster=None
                if self.clusters:
                    from .clusters import ground_cluster
                    cluster=ground_cluster(row,self.clusters,self.bundle['noise'])
                handoff={'post_condition':cid,'p_hat':result['value']['p_hat'],'V':result['value']['V'],'cluster':cluster,
                         'graph_version':self.bundle['graph_version'],'latest_observation':row,'readiness_source':'model_prediction; consumer outcome must be logged separately'}
                emit({'event':'skill_handoff','handoff':handoff});return {'completed':True,'handoff':handoff,'steps':step,'events':self.events}
            if result['state']=='Unknown' or step==maximum:
                emit({'event':'skill_incomplete','reason':'unknown' if result['state']=='Unknown' else 'max_steps'})
                return {'completed':False,'reason':'unknown' if result['state']=='Unknown' else 'max_steps','steps':step,'events':self.events}
            u=(result['plan'] or result['policy'])['u'];action=result.get('action') or normalize_action(u,self.spec)
            receipt=adapter.execute(action)
            if not isinstance(receipt,dict) or not (receipt.get('applied') is True or receipt.get('completed') is True):
                emit({'event':'skill_incomplete','reason':'action_not_completed'})
                return {'completed':False,'reason':'action_not_completed','steps':step,'events':self.events}
            emit({'event':'action_executed','step':step,'u':u,'action':action,'receipt':receipt})
            row=adapter.observe()
        raise AssertionError('Unreachable runtime branch')
