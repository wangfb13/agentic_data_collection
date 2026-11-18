"""Round accounting, immutable validation paths and paired inference with synthetic data."""
from copy import deepcopy
from itertools import product
import pytest

from bottle_scan.state_graph import campaign as c
from bottle_scan.state_graph.common import digest
from bottle_scan.state_graph.round_data import training_rows,bind_request
from bottle_scan.state_graph.round_report import paired_test,paired_report,summary_markdown
from bottle_scan.state_graph.rounds import stop_reason
from bottle_scan.state_graph.v5_profile import configure,protect_output
from tests.fixtures.skill_toy.testing import data
from tests.test_graph_campaign_v4 import make_campaign,synthetic_executor


def test_roles_and_strict_independent_exclusion():
    rows=[dict(node_id=i,data_role=r,round_index=n,expert_success=ok) for i,(r,n,ok) in enumerate([
        ('independent_validation',1,True),('dev_validation',1,True),('dev_validation',2,True),
        ('frontier_expert',1,True),('frontier_expert',1,False),('exploration',1,True)])]
    for dev,expert in product((False,True),repeat=2):
        selected=training_rows(rows,{'loop':dict(dev_to_train=dev,expert_to_train=expert)},2)
        assert {r['node_id'] for r in selected}=={5}|({1} if dev else set())|({3} if expert else set())


def test_paired_mcnemar_and_coordinates():
    pairs=[(True,False)]*6+[(False,True),(True,True)]
    data=paired_test(pairs,.95)
    assert data['rate_difference']==5/8 and data['mcnemar_exact_p']==.125
    assert data['interval'][0]<5/8<data['interval'][1]
    episodes=[dict(paired=True,group_id=str(i),initial_coords={'x':i},terminal_outcomes={
        'a':{'success':p},'b':{'success':q}}) for i,(p,q) in enumerate(pairs)]
    episodes.append(dict(paired=False,group_id='excluded',initial_coords={},terminal_outcomes={}))
    spec={'report':dict(paired_consumers=True,coordinate_bin_mm=4,confidence=.95),'post_conditions':[{'id':'a'},{'id':'b'}]}
    actual=paired_report(episodes,{'reachability':{'a':{'x':{'status':'ready'}},'b':{}}},spec)['a minus b']
    assert actual['all']==data and len(actual['by_initial_coordinate'])==2 and actual['ready_only_left']==['x']
    with pytest.raises(ValueError):paired_report(episodes+[episodes[0]],{'reachability':{}},spec)


def test_summary_paths_and_protection(tmp_path):
    report={'data':{'by/group':[4]}}
    assert '实测' in summary_markdown([dict(text='实测',paths=['/data/by~1group/0'])],report)
    with pytest.raises(ValueError):summary_markdown([dict(text='X',paths=['/invented'])],report)
    (tmp_path/'state.json').write_text('{"legacy":true}')
    with pytest.raises(ValueError):protect_output(tmp_path,kind='analysis')
    assert (tmp_path/'state.json').read_text()=='{"legacy":true}'


def test_convergence_requires_every_condition_and_direction():
    spec={'active_post_condition':'zz','budget':{'max_new_trials':99},'perturbation_variables':{'a':{'envelope':[-4,4]}},
          'facts':{'boundary_resolution_mm':.5}}
    state={'rule':{'variables':{'a':{'active':True,'range':[-4,4]}}},'spent':{'trials':5},'independent_reserve':8}
    def f(kind,side,x,group='g'):return {'pred':kind,'args':['a',side,'zz'],'metrics':{
        'condition':group,'farthest_observed_success':x,'resolution_mm':.5,'directional_evaluated_trials':1}}
    facts=[f(role,side,x) for role,side,x in product(('policy_frontier','expert_frontier'),('positive','negative'),(2.,))]
    assert stop_reason(state,spec,facts,[],False,2)=='frontiers_converged'
    assert stop_reason(state,spec,facts+[f('expert_frontier','positive',4,'other')],[],False,2) is None
    state['spent']['trials']=90
    assert stop_reason(state,spec,[],[],False,2)=='independent_reserve'


def setup_round(tmp_path):
    legacy,raw=make_campaign(tmp_path)
    raw['budget'].update(max_new_trials=26,max_batch_trials=8)
    for role in ('development','independent'):
        raw['acceptance'][role+'_B']={'min_trials':2,'min_verified_recoveries':2}
    spec=configure(raw,{'acceptance':{'lower_bound_gates':False},'director':{'mode':'fixed_priority'},'observation':{'auto_select':False},
                         'consumer':{'dual_paired':False}})
    path=tmp_path/'instance_v5.json';c.atomic(path,spec)
    vplan={'cluster_classes':['Ready','NeedAction'],'cells':[{'settings':{'x':0,'y':0},'reps':2,'actions':[]}]}
    cfg=dict(spec=str(path),sources=[],development_plan=vplan,independent_plan=vplan)
    cfgpath=tmp_path/'config_v5.json';c.atomic(cfgpath,cfg);out=tmp_path/'v5'
    c.initialize(cfgpath,out,seed_rows=deepcopy(data()[0]))
    return out,spec


def test_offline_round_resume_and_budget(tmp_path):
    out,spec=setup_round(tmp_path)
    planned=c.run(out,real=False)
    assert planned['pending'] and planned['spent']['trials']==0
    original=deepcopy(planned['pending']);again=c.run(out,real=False)
    assert again['pending']==original
    def execute(request,directory,spec,cfg,**kwargs):
        return synthetic_executor(request,directory,spec,cfg,**kwargs)
    for _ in range(4):
        done=c.run(out,executor=execute,max_decisions=100)
        if done['phase'] in ('independent_failed','inconclusive','complete'):break
    assert done['phase'] in ('independent_failed','inconclusive','complete')
    assert done['round_index']>=2 and not done['robot_started']
    for index in range(1,done['round_index']+1):
        folder=out/'rounds'/f'round_{index:04d}'
        p=c.read(folder/'plan.json')
        for stage in ('execute','train','dev_validate','summarize','update_rule'):assert (folder/(stage+'.json')).is_file()
        assert p['dev_plan_digest']==digest(p['development_plan'])
        if index>1:assert p['rule']['parent_rule_id']==c.read(out/'rounds'/f'round_{index-1:04d}'/'plan.json')['rule']['rule_id']
    independent=[t for t in done['trials'] if t['request']['data_role']=='independent_validation']
    assert done['spent']['trials']-len(independent)<=spec['budget']['max_new_trials']-done['independent_reserve']
    assert len({t['trial_id'] for t in done['trials']})==len(done['trials'])



def test_technical_retry_keeps_dispatch_cost_and_independent_capacity(tmp_path):
    out,spec=setup_round(tmp_path)
    state=c.read(out/'state.json')
    spec['budget']['max_dispatch_attempts']=state['independent_reserve']+1
    c.queue_plan(out,state,spec,{'cells':[{'settings':{},'actions':[],'reps':1}]},'exploration')
    calls=[]
    def interrupted(*args,**kwargs):
        calls.append(1)
        raise c.RetryTrial('recording interrupted after probe')
    c.collect_pending(out,state,spec,{},executor=interrupted)
    assert state['dispatch_attempts']==1 and state['spent']['trials']==0
    c.collect_pending(out,state,spec,{},executor=interrupted)
    assert calls==[1] and state['pending'] is None
    assert state['trials'][-1]['status']=='budget_skipped'
    assert state['dispatch_attempts']==1 and state['trials'][-1]['dispatch_ordinals']==[1]


def test_unmeasured_legacy_plan_never_creates_paired_coordinate_bin(tmp_path):
    from bottle_scan.state_graph.round_data import episodes
    out,spec=setup_round(tmp_path)
    state=c.read(out/'state.json')
    c.queue_plan(out,state,spec,{'cells':[{'settings':{'x':1},'actions':[],'reps':1}]},'direct_baseline')
    c.collect_pending(out,state,spec,{},executor=synthetic_executor)
    result=list(episodes(state,spec))
    assert result[0]['initial_coords']=={} and result[0]['coordinate_source'] is None
