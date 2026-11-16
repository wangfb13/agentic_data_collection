from copy import deepcopy
from pathlib import Path
import json
import numpy as np
import pytest
from bottle_scan.state_graph.campaign import initialize,run,atomic,read,collect_pending,queue_plan
from bottle_scan.state_graph.skill_interface import load_spec
from tests.fixtures.skill_toy.testing import data


def make_campaign(tmp_path):
    _,spec,_=data();spec=deepcopy(spec)
    spec['budget']['max_new_trials']=50
    spec['models']['ensemble']=2;spec['dynamics'].update(ensemble=2,max_iter=80)
    spec['acceptance']={'requires_ready_and_need_action_coverage':True}
    for role in ('development','independent'):
        spec['acceptance'][role+'_B']={'min_trials':6,'min_normal':2,'min_needs':2,'min_verified_recoveries':2,'max_false_success':0,'max_rejection_rate':.1}
    path=tmp_path/'spec.json';atomic(path,spec);cfg=tmp_path/'config.json';atomic(cfg,{'spec':str(path),'sources':[]})
    out=tmp_path/'campaign';initialize(cfg,out);return out,spec


def synthetic_executor(request,directory,spec,cfg,**kwargs):
    original,_,_=data();rng=np.random.default_rng(int(request['trial_id'].rsplit('_',1)[1])+len(request['role'])*97)
    x=request['settings']['x'];point=np.array([x,0.])+rng.normal(0,.018,2);before=deepcopy(original[0]);gid=request['trial_id']
    before.update(node_id=gid+':0',group_id=gid,trial_id=gid,session='synthetic_campaign',window_index=0,phase='B_pre',previous_node_id=None,u_executed=None,u_receipt=None,is_terminal=False,
                  repeat_key='synthetic_repeat_'+str(x),repeat_settings={'x':x,'y':0},plan_coords={'x':x,'y':0})
    before['obs']={'geometry':point.tolist(),'load':(point*np.array([.3,.4])).tolist(),'image':[float(point.sum()),float(point[0]-point[1])]}
    for p in before['post'].values():p.update(evaluated=False,success=None)
    after=deepcopy(before);after.update(node_id=gid+':1',window_index=1,phase='B_post',previous_node_id=before['node_id'],u_executed=(-.75*point).tolist(),u_receipt={'applied':True,'source':'synthetic_measured'},is_terminal=True)
    target=point*.25+rng.normal(0,.005,2)
    after['obs']={'geometry':target.tolist(),'load':(target*np.array([.3,.4])).tolist(),'image':[float(target.sum()),float(target[0]-target[1])]}
    for p in after['post'].values():p.update(evaluated=True,success=True,evidence=[{'source':'synthetic_consumer'}])
    assessment=None
    if request['role']!='exploration':
        # This fixture verifies orchestration/provenance, not real controller efficacy.
        assessment={'evaluated':True,'success':True,'initial_truth':'Ready' if x==0 else 'NeedAction','invoked':x!=0,
                    'target_truth':True,'exit_declared':True,'rejected':False,'task_success':True,'evidence':['synthetic_consumer']}
    return {'trial_id':gid,'status':'complete','nodes':[before,after],'assessment':assessment,'cost':{'actions':1,'wall_seconds':.2}}


class Director:
    def __init__(self):self.ops=[]
    def symbolic_direct(self,context):
        operators=context['applicable_operators'];available={o['op'] for o in operators}
        if 'complete' in available:op='complete'
        elif 'freeze' in available:op='freeze'
        elif 'validate_B' in available:op='validate_B'
        else:op='collect_seed'
        self.ops.append(op);selected=next(o for o in operators if o['op']==op)
        plan=None
        if op in ('collect_seed','validate_B'):
            plan={'cells':[{'settings':{'x':x,'y':0},'blocks':[{'payload_class':'soft'}],'reps':4} for x in (-1,0,1)],'cluster_classes':['Ready','NeedAction']}
        if op=='validate_B' and context.get('independent_plan'):plan=context['independent_plan']
        return {'operator':op,'args':selected['bindings'],'premises':selected['premises_satisfied_by'],'hypotheses_update':[],
                'discriminates':[],'expected':'合成测试有实际定义的后置结果','reason':'验证持久闭环','plan':plan}


def test_complete_loop_freezes_tested_candidate_and_seals_independent(tmp_path):
    out,spec=make_campaign(tmp_path);director=Director()
    state=run(out,provider=director,executor=synthetic_executor)
    assert state['complete'] and state['spent']['trials']==36
    assert director.ops==['collect_seed','validate_B','freeze','validate_B','complete']
    assert (out/'certificate.json').is_file()
    import joblib
    frozen=joblib.load(state['frozen']['bundle'])
    independent_ids={r['node_id'] for t in state['trials'] if t['request']['role']=='independent' for r in read(Path(t['directory'])/'result.json')['nodes']}
    assert not independent_ids & set(frozen['noise']['fit_node_ids'])
    assert len({t['request']['model_id'] for t in state['trials'] if t['request']['role']=='independent'})==1
    assert state['frozen']['model_id']==next(t['request']['model_id'] for t in state['trials'] if t['request']['role']=='development')


def test_interruption_never_reexecutes_a_started_trial(tmp_path):
    out,spec=make_campaign(tmp_path);state=read(out/'state.json')
    queue_plan(out,state,spec,{'cells':[{'settings':{'x':0},'reps':2}]},'exploration')
    calls=[]
    def crash(*args,**kwargs):calls.append(1);raise RuntimeError('process stopped after motion')
    with pytest.raises(RuntimeError):run(out,executor=crash)
    state=run(out,executor=crash)
    assert len(calls)==1 and state['phase']=='awaiting_result' and state['spent']['trials']==1
    trial=state['trials'][0]
    result=synthetic_executor(trial['request'],Path(trial['directory']),spec,{})
    atomic(Path(trial['directory'])/'result.json',result)
    # Recovery consumes the saved result. Only the other queued request executes.
    def next_one(*args,**kwargs):calls.append(1);return synthetic_executor(*args,**kwargs)
    state=run(out,executor=next_one)
    assert len(calls)==2 and state['spent']['trials']==2 and state['phase']=='awaiting_director'


def test_plan_only_does_not_call_executor_or_count_hardware_attempt(tmp_path):
    out,spec=make_campaign(tmp_path);state=run(out,provider=Director())
    assert state['phase']=='ready_to_collect' and state['spent']['trials']==0 and not state['robot_started']


def test_independent_failure_does_not_retrain_or_change_frozen_candidate(tmp_path):
    out,spec=make_campaign(tmp_path)
    def fail_independent(*args,**kwargs):
        result=synthetic_executor(*args,**kwargs)
        if args[0]['role']=='independent':result['assessment']['success']=False;result['assessment']['target_truth']=False
        return result
    state=run(out,provider=Director(),executor=fail_independent)
    assert state['phase']=='independent_failed' and not state['complete']
    assert not (out/'certificate.json').exists()


def test_small_development_batches_keep_the_candidate_until_coverage_complete(tmp_path):
    out,spec=make_campaign(tmp_path)
    spec=read(out/'spec.json')
    for role in ('development','independent'):spec['acceptance'][role+'_B']['min_trials']=12
    atomic(out/'spec.json',spec)
    class SmallB(Director):
        def symbolic_direct(self,ctx):
            answer=super().symbolic_direct(ctx)
            if answer['operator']=='validate_B' and not ctx.get('independent_plan'):
                for cell in answer['plan']['cells']:cell['reps']=2
            return answer
    state=run(out,provider=SmallB(),executor=synthetic_executor)
    assert state['complete']
    dev=[t for t in state['trials'] if t['request']['role']=='development']
    assert len(dev)==12 and len({t['request']['model_id'] for t in dev})==1
    assert sum(t['request']['role']=='independent' for t in state['trials'])==12


def test_source_pilot_reviews_original_metric_and_queues_followup(tmp_path,monkeypatch):
    from bottle_scan.state_graph import campaign
    out,spec=make_campaign(tmp_path)
    rows,_,noise=data()
    condition=spec['active_post_condition']
    source=next(r for r in rows if r['is_terminal'] and r['post'][condition].get('success') is True)
    atomic(out/'seed_nodes.json',rows)
    atomic(out/'revisions/0001/noise_model.json',noise)
    state=read(out/'state.json');state['revision']=1
    pilot={'cells':[{'settings':{'x':0},'actions':actions,'reps':1} for actions in ([],[[1.,0.]])]}
    queue_plan(out,state,spec,pilot,'exploration',source_review={'revision':1,'source_node_id':source['node_id']})
    original_ids=state['pending']['trial_ids']
    def execute(request,directory,spec,cfg,**kwargs):
        result=synthetic_executor(request,directory,spec,cfg)
        first,last=result['nodes']
        first['obs']=deepcopy(source['obs'])
        if not request['actions']:
            first.update(is_terminal=True,post=deepcopy(source['post']))
            result['nodes']=[first]
            result['cost']['actions']=0
        else:
            last['u_executed']=[.9,0.]
            last['post'][condition].update(evaluated=False,success=None)
        return result
    calls=[]
    def decide(ctx,*args,**kwargs):
        calls.append(ctx['source_comparison'])
        return {'status':'accepted','answer':{'operator':'collect_seed','plan':pilot,'reason':'more evidence'}}
    monkeypatch.setattr(campaign,'decide',decide)
    state=run(out,executor=execute)
    assert state['spent']['trials']==2 and state['phase']=='ready_to_collect'
    assert len(calls)==1 and state['pending']['trial_ids']!=original_ids
    base,moved=calls[0]['arms']
    assert base['source_matched'] and base['distance_to_source']<1e-6 and base['prior_success_groups_nearby']>0
    assert base['consumer_success'] is True and base['measured_actions']==[]
    assert moved['consumer_success'] is None and moved['measured_actions']==[[.9,0.]]
    assert read(out/'reviews'/(state['last_batch']['batch_id']+'.json'))==calls[0]
