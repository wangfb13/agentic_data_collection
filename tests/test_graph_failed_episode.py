"""Failed real episodes must retain evidence without fabricating graph labels."""
import json
from pathlib import Path

import pytest
from bottle_scan.state_graph.extract_nodes import extract_nodes
from bottle_scan.state_graph.skill_interface import load_spec
from skills.contact_position_a.campaign import recover_trial
from skills.contact_position_a.scheduler import lifecycle_event
from skills.contact_position_a.post import forcepolicy_l2

ROOT=Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('filename', ['events.jsonl','saved_scheduler.jsonl'])
def test_episode_without_graph_window_is_not_historical_data(tmp_path,filename):
    events=[{'event':'graph_runtime_selected','model_id':'m'},
            {'event':'fault','reason':'incomplete_contact_window'},
            {'event':'episode_end','state':'FAULT'}]
    path=tmp_path/filename
    path.write_text(''.join(json.dumps(e)+'\n' for e in events))
    spec=load_spec(ROOT/'skills/contact_position_a/spec.campaign_v4.json')
    nodes,report=extract_nodes([str(path)],spec,tmp_path/'extracted')
    assert nodes==[] and report['nodes']==0


def test_completed_failed_process_is_recoverable_without_consumer_labels(tmp_path):
    spec=load_spec(ROOT/'skills/contact_position_a/spec.campaign_v4.json')
    request={'trial_id':'trial3','model_id':'m','role':'exploration',
             'post_condition':spec['active_post_condition']}
    events=[{'event':'graph_runtime_selected','model_id':'m','request':request},
            {'event':'recovery_skill_decision','skill_outputs':{'exit':{'completed':False}}},
            {'event':'episode_end','state':'FAULT','reason':'incomplete_contact_window'}]
    directory=tmp_path/'episodes/episode_1';directory.mkdir(parents=True)
    path=directory/'events.jsonl'
    path.write_text(''.join(json.dumps(e)+'\n' for e in events))
    assert recover_trial(request,tmp_path,spec,{},interactive=False) is None
    (tmp_path/'process.json').write_text(json.dumps({'returncode':1,'wall_seconds':65.17}))
    result=recover_trial(request,tmp_path,spec,{},interactive=False)
    assert result['status']=='failed' and result['nodes']==[] and result['assessment'] is None
    assert result['cost']=={'actions':0,'wall_seconds':65.17}
    assert result['evidence']==[str(path)]


def test_contact_loss_counts_only_after_forcepolicy_owns_gripper():
    spec=load_spec(ROOT/'skills/contact_position_a/spec.campaign_v4.json')
    condition=spec['post_conditions'][0]
    consumer={**condition['consumer'],'version':'actual_checkpoint'}
    window={'node_id':'n','group_id':'trial'}
    event={'event':'handoff_contact_lost_timeout','phase':'probe_control','lost_s':.201,
           'normal_sdk':[.075,1.158],'thresholds_sdk':[.213,.305],'width_normalized':.34}
    assert lifecycle_event(event,window,consumer) is None
    event['phase']='forcepolicy'
    result=lifecycle_event(event,window,consumer)
    outcome=result['consumer_outcome']
    assert outcome['L1'] is False and outcome['L2'] is None and outcome['L3'] is None
    evaluation=forcepolicy_l2({},dict(window,is_terminal=True,consumer_outcome=outcome),condition)
    assert evaluation['evaluated'] and evaluation['success'] is False and evaluation['level']=='L1'
    assert lifecycle_event(event,None,consumer) is None


def test_consumer_failure_keeps_valid_negative_node_and_failed_process(tmp_path,monkeypatch):
    spec=load_spec(ROOT/'skills/contact_position_a/spec.campaign_v4.json')
    request={'trial_id':'trial2','model_id':'m','role':'exploration',
             'post_condition':spec['active_post_condition']}
    events=[{'event':'graph_runtime_selected','model_id':'m','request':request},
            {'event':'state_graph_consumer_result','consumer_outcome':{'L1':False,'L2':None}}]
    directory=tmp_path/'episodes/episode_1';directory.mkdir(parents=True)
    (directory/'events.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in events))
    (tmp_path/'process.json').write_text(json.dumps({'returncode':1,'wall_seconds':69.8}))
    node={'node_id':'n','post':{request['post_condition']:{'evaluated':True,'success':False}}}
    monkeypatch.setattr('skills.contact_position_a.campaign.extract_nodes',lambda *args:([node],{}))
    result=recover_trial(request,tmp_path,spec,{},interactive=False)
    assert result['status']=='failed' and result['nodes']==[node]
    assert result['assessment'] is None and result['unassessed_reason'] is None


def test_incomplete_reprobe_preserves_only_first_window_and_actual_motion(tmp_path):
    from skills.contact_position_a.ready import select
    from skills.contact_position_a.runtime import GraphRecoveryProxy
    spec = load_spec(ROOT/'skills/contact_position_a/spec.campaign_v4.json')
    select(spec, 'fixed5', device_min=.3)
    goal = spec['active_post_condition']
    request = {'mode':'planned', 'model_id':'m', 'trial_id':'test', 'role':'exploration', 'post_condition':goal}
    proxy = GraphRecoveryProxy(None, request, spec)
    proxy.window = {'node_id':'first'}
    receipt = {'applied':True, 'action_mm':-4.48, 'completed_ns':12}
    decision = proxy.advance({'bilateral':False}, applied_action=receipt)
    assert proxy.window is None and decision['exit']['reason']=='incomplete_contact_window'
    features = {'obs':{k:[1.] for k in spec['observation']['modalities']},
                'extractor_versions':{k:v['version'] for k,v in spec['observation']['modalities'].items()}}
    window = {'event':'state_graph_window','session':'s','group_id':'g','trial_id':'i','node_id':'i:0',
              'window_index':0,'phase':'B_pre','t_ns':8,'probe':{},'wrench':[],'attrs':{},
              'preconditions':{'bilateral_contact':True},'raw_evidence':{'wrist':{}},
              'live_features':features,'live_contacts':[],'visual_features':[]}
    events = [{'event':'graph_runtime_selected','model_id':'m','request':request},window,dict(window),
              {'event':'contact_incomplete','corrections':[{'receipt':receipt}],'reason':'incomplete_contact_window'}]
    folder = tmp_path/'episodes'/'episode_1'
    folder.mkdir(parents=True)
    log = folder/'events.jsonl'
    log.write_text(''.join(json.dumps(e)+'\n' for e in events))
    (tmp_path/'process.json').write_text(json.dumps({'returncode':0,'wall_seconds':9}))
    result = recover_trial(request,tmp_path,spec,{},interactive=False)
    assert result['cost']['actions']==1 and result['measured_actions']==[[-4.48]]
    assert result['status']=='failed' and result['attempt_reason']=='incomplete_contact_window'
    node, = result['nodes']
    assert node['node_id']=='i:0' and node['u_executed'] is None and not node['post'][goal]['evaluated']
    events[2]['t_ns']+=1
    log.write_text(''.join(json.dumps(e)+'\n' for e in events))
    with pytest.raises(ValueError,match='Conflicting state_graph_window'):
        extract_nodes([str(log)],spec)
