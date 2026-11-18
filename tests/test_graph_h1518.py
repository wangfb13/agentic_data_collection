"""Synthetic action-free fixtures and real round/dispatch paths; never connect hardware."""
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
import json

import numpy as np
import pytest

from bottle_scan.state_graph import campaign as c
from bottle_scan.state_graph import collection_progress as progress
from bottle_scan.state_graph import pointwise_score as points
from bottle_scan.state_graph.common import digest
from bottle_scan.state_graph.director_bridge import context,decide
from bottle_scan.state_graph.facts import fact
from bottle_scan.state_graph.noise import fit_noise
from bottle_scan.state_graph.operators import validate_decision
from bottle_scan.state_graph.repeat_evidence import indices
from bottle_scan.state_graph.v5_profile import merge
from skills.contact_position_a import p3_trial,reference_check
from tests.test_graph_h1214_planning import H,R,CID,measured,sample
from tests.test_v5_geometry_consumers import instance,window,pose,Fake


@pytest.fixture
def spec(tmp_path):
    overrides=json.loads(Path('configs/p3_v5_manual_lift_h1518.json').read_text())
    s=merge(instance(tmp_path),overrides)
    s['director']['mode']='fixed_priority'
    return s


def probe(i,ref,coordinates=(0.,0.)):
    w=window(0,coordinates)
    w.update(node_id=f'n{i}',group_id=f'g{i}',trial_id=f't{i}',task_id='white_bottle|translation_horizontal_r_base_h',
             repeat_settings={R:0,H:0},window_valid=True,is_terminal=False,post={},extractor_versions={'tac':'g','force':'f'},
             obs={'tac':[i*.1]*6,'force':[i*.03]*12})
    w['provenance'].update(placement_reference_id=ref,coordinate_schema='measured_placement_relative')
    return w


def test_partial_noise_continuity_and_placement_covariances(spec):
    rows=[probe(i,'p'+str(i//2)) for i in range(12)]
    noise=fit_noise(rows,spec)
    assert noise['active_modalities']==['tac'] and noise['modalities']['force']['status']=='insufficient_repeats'
    b={'noise':noise,'graph':{'radii':{'similar_radius':.7}}}
    witness=rows[0]
    near=deepcopy(witness)
    near['extractor_versions'].pop('force')
    near['obs'].pop('force')
    result=reference_check.evaluate(witness,near,b,spec['geometry'])
    assert result['passed'] and result['used_modalities']==['tac'] and result['joint_degrees_of_freedom']==6
    near['obs']['tac'][0]+=50
    assert reference_check.evaluate(witness,near,b,spec['geometry'])['status']=='distance_exceeded'
    no_geom=dict(b,noise=dict(noise,active_modalities=[]))
    assert reference_check.evaluate(witness,near,no_geom,spec['geometry'])['reason']=='no_ready_tactile_geometry'
    four=[probe(i,str(i)) for i in range(4)]
    assert indices(four,spec)==[[0,1,2,3]]
    four[0]['attrs']['mass_class']='heavy'
    assert indices(four,spec)==[[1,2,3]]
    audit=noise['modalities']['force']['placement_audit']
    assert (audit['within_placement']['pairs'],audit['across_placements']['pairs'])==(6,60)
    assert np.shape(audit['across_placements']['single_covariance'])==(12,12)
    assert np.trace(audit['within_placement']['single_covariance'])<np.trace(audit['across_placements']['single_covariance'])


def decision(spec,rows,tmp_path,history=(),round_index=1):
    g={'edges':[],'probabilities':{CID:{}}}
    facts=measured(rows,spec)+[fact('repeat_calibration_missing',[CID],[],{'deficit':4,'missing':{'force':{}}})]
    active,repeated=progress.restrictions(history,round_index,spec['loop']['no_progress_cooldown'],rows,g,spec)
    facts.extend(progress.no_progress_fact(r) for r in active)
    ctx=context(spec,{'rule_id':'r','variables':{}},facts,budget_left=1 if not history else 12,
                extras=dict(point_evidence=points.evidence(rows,g,spec),active_cooldowns=active,repeated_no_progress=repeated))
    return ctx,decide(ctx,tmp_path)['answer']


class RepeatIO(Fake):
    placement='P3'
    def __init__(self,spec,req,events,bad):
        super().__init__(spec)
        self.req,self.events,self.bad=req,events,bad
    def prepare(self,req):
        self.events.append('witness')
        self.reference=dict(valid=True,placement_reference_id='pin',pose=pose().tolist())
        c.atomic(Path(req['campaign'])/'placement_references/P3/active.json',self.reference)
    def repeat_pose(self):return pose()
    def repeat_continuity(self):return True
    def consume_for(self,*args):raise AssertionError('Repeat must not hand off or lift')
    def move(self,*args):raise AssertionError('Repeat must not translate')
    @contextmanager
    def contact(self,index):
        self.events.append('probe')
        w=window(index)
        w.update(node_id=self.req['trial_id']+':0',trial_id=self.req['trial_id'],group_id=self.req['trial_id'])
        w['provenance'].update(placement_reference_id='pin',coordinate_schema='measured_placement_relative')
        try:yield None if self.bad else w
        finally:self.events.append('open')


@pytest.mark.parametrize('bad',[False,True])
def test_repeat_batch_dispatch_progress_cooldown_and_next_direction(tmp_path,spec,bad):
    from bottle_scan.state_graph.rounds import dispatch
    rows=[sample(i,i,True) for i in range(1,5)]
    ctx,d=decision(spec,rows,tmp_path/'director')
    assert d['operator']=='collect_repeat_block' and d['args']['probes']==6
    assert set(o['op'] for o in ctx['applicable_operators'])=={'collect_repeat_block'}
    assert next(s for s in ctx['operator_scores'] if s['operator']=='collect_repeat_block')['score'] is None
    folder=tmp_path/'campaign';rd=folder/'rounds/round_0001'
    state=dict(created_ns=901,round_index=1,round_stage='execute',stage_token=None,round_operation=None,
               independent_reserve=24,dispatch_attempts=0,spent=dict(trials=0,actions=0,failures=0,seconds=0.),
               frozen=None,model=None,pending=None,robot_started=False,rule={'rule_id':'r'},
               revision=1,history=[],trials=[])
    c.atomic(folder/'seed_nodes.json',[])
    c.atomic(rd/'plan.json',{'decision':d,'evidence_before':[]})
    events=[]
    def executor(req,path,spec,cfg,**_):
        return p3_trial.run_trial(req,path,spec,cfg,RepeatIO(spec,req,events,bad))
    batches=[dict(role='exploration',plan=d['plan'],operation=d['operator'])]
    assert dispatch(folder,state,spec,{},batches,dict(executor=executor,interactive=False))
    assert events==['witness']+['probe','open']*6
    assert state['spent']['repeat_probes']==6 and state['spent']['trials']==state['dispatch_attempts']==0
    assert dispatch(folder,state,spec,{},batches,dict(executor=executor,interactive=False))
    assert events.count('probe')==6
    records=c.all_rows(folder,state)
    assert len(records)==(0 if bad else 6) and all(not r['is_terminal'] for r in records)
    rec=c.read(rd/'batch_00_progress.json')
    assert rec['actual_increment']==(0 if bad else 6) and rec['expected_increment']==6
    if not bad:
        assert len(indices(records,spec)[0])==6
        return
    assert rec['reason']=='invalid_result'
    next_ctx,next_decision=decision(spec,rows,tmp_path/'next',[rec],2)
    assert next_decision['operator']!='collect_repeat_block'
    assert any(o['op']=='probe_amplitude' and o['bindings']['var']==H and o['bindings']['side']=='positive'
               for o in next_ctx['applicable_operators'])
    assert any(f['pred']=='operator_no_progress' for f in next_ctx['facts'])
    with pytest.raises(ValueError): validate_decision(d,next_ctx)
    _,after=decision(spec,rows,tmp_path/'after',[rec],4)
    assert after['operator']=='collect_repeat_block'
    active,repeated=progress.restrictions([rec,dict(rec,round_index=4)],5,2,rows,{'edges':[]},spec)
    assert len(active)==1 and len(repeated[d['operator']])==2


def test_pointwise_density_cost_and_trial_groups(spec):
    rows=[sample(i,0,True) for i in range(40)]
    data=points.evidence(rows,{},spec);ctx={'spec':spec,'point_evidence':data}
    def calc(x,op='collect_outcomes',interval=None):
        args={'var':H,'interval':interval} if interval else {}
        return points.score(ctx,dict(operator=op,args=args,plan={'cells':[dict(settings={H:x},reps=1)]}))
    dense=calc(0);gap=calc(20)
    assert gap['score']>dense['score']
    assert gap['per_consumer'][CID]['n_eff']==pytest.approx(0,abs=1e-8)
    assert dense['per_consumer'][CID]['n_eff']==pytest.approx(40)
    assert calc(0,interval=[0,20])['score']==gap['score']
    assert calc(20,'probe_amplitude')['score']==pytest.approx(gap['score']/1.25)
    assert points.evidence(rows+[dict(rows[0],evaluation_only=True)],{},spec)==data


@pytest.mark.parametrize('bad',[False,True])
def test_real_round_loop_audits_repeat_then_replans(tmp_path,spec,bad):
    # Seed independent light contacts, enough for 6-D geometry, short of 12-D force.
    path=tmp_path/'seed'
    path.mkdir()
    raw=[]
    for i in range(12):
        w=window(0)
        w.update(node_id='seed'+str(i),group_id='seed'+str(i),trial_id='seed'+str(i))
        w['contacts'][0]['area_px']+=i*.01
        w['provenance'].update(coordinate_schema='measured_placement_relative',placement_reference_id='p'+str(i))
        raw.append(w)
    rows=p3_trial.observe_rows(raw,'seed',path,spec)
    spec['adapters'].pop('campaign_prepare',None)
    spec['budget'].update(max_new_trials=100,max_dispatch_attempts=100)
    spec['acceptance'].update(lower_bound_gates=False,futility_stop=False)
    for role in ('development','independent'):
        spec['acceptance'][role+'_B']['min_trials']=4
    plan={'cluster_classes':['Ready','NeedAction'],'cells':[{'settings':{R:0,H:0},'reps':2,
        'blocks':[{'mass_class':'light'},{'mass_class':'heavy'}],'actions':[]}]}
    c.atomic(tmp_path/'spec.json',spec)
    c.atomic(tmp_path/'config.json',dict(spec=str(tmp_path/'spec.json'),sources=[],
        development_plan=plan,independent_plan=plan))
    out=tmp_path/'rounds_campaign'
    c.initialize(tmp_path/'config.json',out,seed_rows=rows)
    events=[]
    def execute(req,directory,s,cfg,**_):
        assert req['operation']=='collect_repeat_block'
        return p3_trial.run_trial(req,directory,s,cfg,RepeatIO(s,req,events,bad))
    result=c.run(out,executor=execute,interactive=False,max_decisions=6)
    rec=c.read(out/'rounds/round_0001/round_report.json')
    progress=rec['collection_progress']
    assert progress['expected_increment']==14 and progress['actual_increment']==(0 if bad else 14)
    assert rec['batch_progress'][0]['actual_increment']==progress['actual_increment']
    assert result['spent']['repeat_probes']==14 and result['spent']['trials']==0
    assert not result['robot_started'] and result['round_index']==2
    ctx=c.read(out/'director_context.json')
    assert any(f['pred']=='operator_no_progress' for f in ctx['facts'])==bad
    assert not any(o['op']=='collect_repeat_block' for o in ctx['applicable_operators'])
    noise=c.read(out/'revisions/round_0001/noise_model.json')
    assert (noise['modalities']['force']['status']=='ready')!=bad


def test_repeat_retry_does_not_refund_or_consume_b_quota(tmp_path,spec):
    state=dict(created_ns=10,round_index=1,round_stage='execute',round_operation='collect_repeat_block',
               stage_token='1/execute/0',independent_reserve=24,dispatch_attempts=0,
               spent=dict(trials=0,actions=0,failures=0,seconds=0.),frozen=None,model=None,pending=None,
               robot_started=False,rule={'rule_id':'r'},revision=1,history=[],trials=[])
    spec['budget']['repeat_probes']=1
    plan={'cells':[dict(settings={R:0,H:0},blocks=[{'mass_class':'light'}],reps=1,actions=[])]}
    c.queue_plan(tmp_path,state,spec,plan,'exploration')
    calls=[]
    def fail(*args,**kwargs):
        calls.append('probe')
        raise c.RetryTrial('synthetic interrupted probe')
    c.collect_pending(tmp_path,state,spec,{},executor=fail,interactive=False)
    c.collect_pending(tmp_path,state,spec,{},executor=fail,interactive=False)
    assert calls==['probe'] and state['spent']['repeat_probes']==1
    assert state['spent']['trials']==state['dispatch_attempts']==0
    assert state['trials'][0]['status']=='budget_skipped'
