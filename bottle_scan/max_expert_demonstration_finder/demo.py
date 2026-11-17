"""Deterministic synthetic port. All output is marked simulation. Never uses ROS."""
from copy import deepcopy
from pathlib import Path
import json
import numpy as np
from .evidence import SIGN,label_record,sha
from .ledger import Ledger
from .engine import execute,reconcile
from .episodes import summarize
from .search import offers,report
from .train import training_examples
from .compare import compare
from bottle_scan.state_graph.common import write
from skills.contact_position_a.local_motion import command
from skills.contact_position_a.ready import fixed_consumer,FIXED

def fixture(expert=True,mode='monotone',slope=-1.,support=2.5):
    limits=dict(range=[-5,5],resolution=2.5,cell_tolerance=.7,expansion=2.5,prepare_mm=5,step_mm=5,total_mm=10,
                seconds=30,max_steps=2,physical_attempts=190,project_physical_attempts=1000,max_software_attempts=195)
    exp=dict(id='sim_manual',version='1',source='simulated oracle',observations=['synthetic local sensor'],allowed_actions=['tcp_translation_a'],stronger=False)
    rule={'slope':slope,'support':support,'fixture':mode}
    return dict(environment='simulation',model_id=sha(rule),action_source='graph_policy',graph_model_id='fixture',controller='fixture_linear',
        station='fake',axis_tcp=[0,0,1],protocol={'id':'sim_probe'},target_definition_id='white_fixture',reference_digest='fixture_ref',
        condition=dict(mass_class='light',placement_group='synthetic',approach_group='pose0'),tool_version='sim',sensor_version='sim',
        post_condition={'id':FIXED,'consumer':fixed_consumer(FIXED,None),'evaluator':'skills.contact_position_a.ready:fixed_ready'},
        artifacts={'fixture_weights':rule},reject_rules={'mode':mode},policy_executable=True,limits=limits,execution_caps=deepcopy(limits),
        statistics=dict(screen_n=1,confirmation_n=12,target=.6,confidence=.8),expert=exp if expert else None)

class Fixture:
    environment='simulation'
    def __init__(self,m):
        self.m=m;self.expert_identity=m['expert'];self.identity=m['model_id'];self.mode=m['reject_rules']['mode'];self.i=0;self.t=100;self.x=0.;self.extra={};self.rng=np.random.default_rng(1611)
    def verify(self,m):assert m==self.m
    def stamp(self):self.t+=1;return self.t
    def measurement(self,phase,t):
        return label_record(dict(source='independent_offset_measurement',unit='mm',axis_sign=SIGN,target_definition_id=self.m['target_definition_id'],
           reference_digest=self.m['reference_digest'],phase=phase,timestamp_ns=t,raw_source=self.aid+'/'+phase,
           instrument_id='simulation',calibration_id='fixture',valid=True,object_moved=False,reading_mm=self.x,uncertainty_mm=.01))
    def observe(self,first):
        t=self.stamp();T=np.eye(4);T[2,3]=self.x
        state='no_contact' if first and self.mode=='no_contact' else 'valid_bilateral'
        return dict(id='obs'+str(t),node_id=None if state!='valid_bilateral' else 'node'+str(t),contact_state=state,
            timestamp_ns=t,tcp_mm=T.tolist(),measurement=self.measurement('first_contact',t) if first else None,sensor=self.x) | self.extra
    def decide(self,o,step):
        w=self.m['artifacts']['fixture_weights'];x=o['sensor']
        if abs(x)<.5 or self.mode in ('island','probability'):return {'state':'Ready'}
        if abs(x)>w['support']+.001 or self.mode=='reject':return {'state':'Unknown','reason':'model_ood'}
        return self.act(o,w['slope']*x,'graph_policy')
    def act(self,o,delta,source):
        return dict(state='NeedAction',command=command(delta,dict(observation_id=o['id'],timestamp_ns=o['timestamp_ns']),self.m['target_definition_id'],source))
    def expert_decision(self,o,step,expert):
        return self.act(o,-o['sensor'],'expert') if abs(o['sensor'])>.25 else {'state':'Ready'}
    def perform(self,r):
        self.aid=r['attempt_id'];op=r['operation'];o={'attempt_id':self.aid,'command_id':r['command_id'],'raw_source':'fixture/'+r['command_id'],'physical_interaction':True}
        if op=='prepare':
            self.i+=1;self.x=r['requested_offset_mm'];t=self.stamp();o.update(prepared=True,independent_trial_unit=self.aid,timestamp_ns=t,measurement=self.measurement('preparation',t))
        elif op=='touch':
            if self.mode=='technical':raise OSError('simulated sensor timeout')
            self.last=self.observe(True);o['observation']=self.last
        elif op=='correct_retouch':
            prev=self.last;self.x+=r['command']['u'][0];after=self.observe(False);self.last=after
            o.update(observation=after,receipt=dict(applied=True,observation_id=prev['id'],frame='tcp_at_action_start',reference_id=self.m['target_definition_id'],
                actual_delta_a_mm=self.x-prev['sensor'],source_pose_mm=prev['tcp_mm'],recontact_pose_mm=after['tcp_mm'],
                completed_ns=after['timestamp_ns'],raw_source=o['raw_source']))
        elif op=='post':
            ok=abs(self.x)<.5
            if self.mode=='island':ok=abs(self.x)<.5 or abs(self.x)>4
            if self.mode=='probability':ok=bool(self.rng.random()<max(.15,.95-.14*abs(self.x)))
            o.update(post_condition=self.m['post_condition']['id'],consumer_outcome={**self.m['post_condition']['consumer'],'evidence':[
                dict(event='fixed_squeeze_lift_result',target_reached=ok,lifted=ok,hold_complete=ok,dropped=False,normal_sdk=[5,5])]})
        return o

def run_demo(dest):
    dest=Path(dest);dest.mkdir(parents=True,exist_ok=False);db=Ledger(dest/'simulation.sqlite');m=fixture()
    rid=db.new(m);site=Fixture(m)
    def pick(r,site,role,x):
        req=next(v for v in offers(db.round(r),db.attempts(r)) if v['role']==role and v['x']==x)
        return execute(db,r,req,site,site)
    def explore(r,s):
        for x in (0,-2.5,2.5,-5,5):
            for role in ('policy_frontier','direct_baseline'):pick(r,s,role,x)
    def confirmation(r,s):
        db.confirm(r,[-5,-2.5,0,2.5,5])
        while True:
            opt=offers(db.round(r),db.attempts(r))
            if not opt:break
            execute(db,r,opt[0],s,s)
        db.close(r);write(dest/(r+'.json'),report(db.round(r),db.attempts(r)))
    explore(rid,site)
    for x in (-5,5):pick(rid,site,'expert_recovery',x)
    confirmation(rid,site)
    ids,ex=training_examples(db.attempts(rid),m)
    pairs=[(t['before']['sensor'],t['actual_mm']) for _,e in ex if e['expert_witness'] for t in e['trajectory']]
    X,Y=np.array(pairs).T
    # Fit the new fixture weights using only exploration demonstrations. No held-out outcomes.
    m2=fixture(slope=float(np.dot(X,Y)/np.dot(X,X)),support=float(max(abs(X))))
    r2=db.new(m2,parent=rid);s2=Fixture(m2);explore(r2,s2);confirmation(r2,s2)
    extra={}
    for mode in ('island','probability','no_contact','reject','technical'):
        mx=fixture(False,mode);mx['statistics']['screen_n']=3 if mode=='probability' else 1
        r=db.new(mx);sx=Fixture(mx)
        if mode=='island':explore(r,sx)
        elif mode=='probability':
            for _ in range(mx['statistics']['screen_n']):explore(r,sx)
        elif mode=='reject':pick(r,sx,'policy_frontier',0);pick(r,sx,'policy_frontier',2.5)
        else:
            for _ in range(mx['statistics']['screen_n']):
                aid=pick(r,sx,'policy_frontier',0)
                if db.pending():reconcile(db,aid,[],'synthetic operator checked')
        db.close(r);extra[mode]=report(db.round(r),db.attempts(r))
    result=dict(environment='simulation',real_physical_interactions=0,training_demo_ids=ids,
                learned_parameters=m2['artifacts'],comparison=compare(db,rid,r2),scenarios={k:{'sides':v['policy_recovery_frontier']['sides'],'phenomena':v['phenomena'],'overall':v['policy_recovery_frontier']['overall']} for k,v in extra.items()})
    write(dest/'summary.json',result);return result

