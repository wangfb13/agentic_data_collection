"""Generic V / policy / dynamics bundle and observe-act-observe runtime."""
from pathlib import Path
import math
import time
import numpy as np
import joblib
from .common import digest,write
from .skill_interface import post_spec,model_observation,action_grid
from .value import fit_value,predict_value
from .policy import fit_policy,predict_policy
from .dynamics import predict as predict_dynamics


def train_models(rows,graph,labels,noise,dynamics,spec,out=None):
    models={}
    for c in spec['post_conditions']:
        models[c['id']]={'V':fit_value(rows,labels,noise,spec,c['id']),
                         'pi':fit_policy(rows,graph,labels,noise,spec,c['id'])}
    bundle={'schema':'skill_bundle_v3','spec':spec,'noise':noise,'dynamics':dynamics,'models':models,'graph':graph,
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
    cid=spec['active_post_condition'];candidate=np.asarray(policy_prediction['u']);grid=action_grid(spec)
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

    def infer(self,row):
        model_observation(row,self.spec['observation']['modalities'])
        cid=self.spec['active_post_condition'];condition=post_spec(self.spec,cid)
        models=self.bundle['models'][cid];v=predict_value(models['V'],row,self.bundle['noise'])
        if v['status']!='predicted':return {'state':'Unknown','value':v,'policy':None}
        ready=v['V']<=-math.log(condition.get('ready_p',.8)) and v['p_hat']>=condition.get('ready_p',.8)
        pi=None if ready else predict_policy(models['pi'],row,self.bundle['noise'])
        state='ready' if ready else ('recoverable' if pi['status']=='predicted' else 'Unknown')
        return {'state':state,'value':v,'policy':pi,'plan':None if ready or state=='Unknown' else optional_plan(self.bundle,row,pi)}

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
            u=(result['plan'] or result['policy'])['u'];receipt=adapter.execute(u)
            emit({'event':'action_executed','step':step,'u':u,'receipt':receipt})
            row=adapter.observe()
        raise AssertionError('Unreachable runtime branch')
