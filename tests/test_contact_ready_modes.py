from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import json

import pytest
from skills.contact_position_a.campaign import launch_request
from skills.contact_position_a.ready import FIXED_CONSUMER, fixed_ready, handoff_ready, fixed_squeeze_lift, select

ROOT=Path(__file__).resolve().parents[1]


def configured(mode):
    cfg=json.loads((ROOT/'config.state_graph.campaign_v4.json').read_text())
    spec=json.loads((ROOT/'skills/contact_position_a/spec.campaign_v4.json').read_text())
    select(spec,mode,device_min=.300)
    return spec,cfg


def test_handoff_is_l1_and_matched_consumer():
    spec,_=configured('forcepolicy')
    cond=spec['post_conditions'][0]
    cond['consumer']['version']='v1'
    outcome={**deepcopy(cond['consumer']),'L1':True,'L2':False,'L3':False,'evidence':[{'event':'main_resumed'}]}
    window={'is_terminal':True,'consumer_outcome':outcome}
    assert handoff_ready({},window,cond)['success'] is True
    outcome['L2']=None
    assert handoff_ready({},window,cond)['success'] is True
    outcome['L1']=False
    assert handoff_ready({},window,cond)['success'] is None
    outcome['evidence']=[{'event':'handoff_contact_lost_timeout','phase':'forcepolicy'}]
    assert handoff_ready({},window,cond)['success'] is False
    outcome['version']='other'
    assert handoff_ready({},window,cond)['success'] is None


@pytest.mark.parametrize('mode,after',[('forcepolicy','handoff'),('fixed5','stop')])
def test_launch_selects_consumer(tmp_path,mode,after):
    spec,cfg=configured(mode)
    req={'post_condition':spec['active_post_condition'],'trial_id':'x','actions':[],'settings':{'closure_floor':spec['post_conditions'][0]['consumer']['params']['closure_floor']},'block':{'mass_class':'light'}}
    _,env=launch_request(req,tmp_path,spec,cfg)
    assert env['CONTACT_SKILL_AFTER']==after
    assert json.loads((tmp_path/'scheduler_config.json').read_text())['skill_after']==after


@pytest.mark.parametrize('achieved,dropped,expected',[(True,False,True),(False,False,False),(True,True,False)])
def test_fixed_measurement_and_lift(achieved,dropped,expected):
    spec,cfg=configured('fixed5_0330');cond=spec['post_conditions'][0]
    hardware_cfg=json.loads(Path(cfg['execution']['scheduler_config']).read_text())['execution_config']
    events=[];actions=[];current={}
    def regulate(name,target,tol,s):
        assert (name,target,tol,s)==('SQUEEZE',5.,.5,.5);actions.append('squeeze')
        return {'normal_sdk':[4.8,5.2]} if achieved else None
    def lift(target,sample):
        actions.append('lift_hold_lower');current.update(lifted=True,hold_complete=True,dropped=dropped)
    runner=SimpleNamespace(cfg=hardware_cfg,current=current,regulate=regulate,hold_and_lower=lift,
                           hw=SimpleNamespace(snapshot=lambda:{}),release=lambda:actions.append('release'))
    backend=SimpleNamespace(_require_supported=lambda s:None,_emit=events.append,
                            recovery=SimpleNamespace(window={'node_id':'n','group_id':'g'}))
    fixed_squeeze_lift(backend,runner,None)
    assert actions==(['squeeze','lift_hold_lower','release'] if achieved else ['squeeze','release'])
    assert len(events)==1 and events[0]['node_id']=='n'
    w={'is_terminal':True,'consumer_outcome':events[0]['consumer_outcome']}
    assert fixed_ready({},w,cond)['success'] is expected
    w['consumer_outcome']['controller']='existing_forcepolicy'
    assert fixed_ready({},w,cond)['success'] is None


def test_fixed_interruption_or_different_protocol_is_unknown():
    s,_=configured('fixed5_0330');c=s['post_conditions'][0]
    w={'is_terminal':True,'consumer_outcome':{**deepcopy(FIXED_CONSUMER),'evidence':[{'event':'fixed_squeeze_lift_result','target_reached':True,'normal_sdk':[5,5],'lifted':True}]}}
    assert fixed_ready({},w,c)['success'] is None
    w['consumer_outcome']['params']['unit']='N'
    assert fixed_ready({},w,c)['success'] is None


def test_director_can_collect_missing_class_after_fitting():
    from bottle_scan.state_graph.facts import fact
    from bottle_scan.state_graph.operators import applicable_operators
    spec, _ = configured('forcepolicy')
    goal = spec['active_post_condition']
    rule = {'rule_id':'example'}
    fitted = fact('model_executable', [goal], ['n1'])
    missing = fact('development_class_missing', [goal], ['n1'], {'observed_classes':['Ready'], 'missing_classes':['NeedAction']})
    def ops(facts, state=None):
        return {op['op'] for op in applicable_operators(facts, spec, rule, state)}
    assert 'validate_B' in ops([fitted])
    assert {'collect_outcomes','collect_transitions'} <= ops([fitted,missing])
    assert 'validate_B' not in ops([fitted,missing])
    assert 'validate_B' in ops([fitted,missing], {'frozen_independent':True})
    missing['metrics']['observed_classes'].append('Unknown')
    assert 'validate_B' in ops([fitted,missing])


@pytest.mark.parametrize('mode,success,returncode,status', [
    ('fixed5',False,0,'failed'), ('forcepolicy',True,1,'complete')])
def test_exploration_counts_selected_endpoint(tmp_path, monkeypatch, mode, success, returncode, status):
    from skills.contact_position_a.campaign import recover_trial
    spec,_ = configured(mode)
    cid = spec['active_post_condition']
    req = {'trial_id':'t', 'model_id':'m', 'role':'exploration', 'post_condition':cid}
    folder = tmp_path/'episodes'/'episode_1'
    folder.mkdir(parents=True)
    event = {'event':'graph_runtime_selected', 'model_id':'m', 'request':req}
    (folder/'events.jsonl').write_text(json.dumps(event)+'\n')
    (tmp_path/'process.json').write_text(json.dumps({'returncode':returncode,'wall_seconds':1.}))
    terminal = {'node_id':'n','post':{cid:{'evaluated':True,'success':success}}}
    monkeypatch.setattr('skills.contact_position_a.campaign.extract_nodes',lambda *args:([terminal],{}))
    result = recover_trial(req,tmp_path,spec,{},interactive=False)
    assert result['status'] == status


def test_historical_fixed5_uses_matching_final_probe_and_real_session(tmp_path):
    import hashlib
    from skills.contact_position_a.ready import FIXED_CONSUMER
    spec,cfg = configured('fixed5_0330')
    condition = spec['post_conditions'][0]
    hardware = json.loads(Path(cfg['execution']['scheduler_config']).read_text())['execution_config']
    trial = {'complete':True, 'recording_complete':True, 'probe_final':{'valid':True,'timestamp_ns':10},
             'status':'hold_complete','target_reached':True,'lifted':True,'hold_complete':True,'dropped':False,
             'stages':[{'name':'SQUEEZE','start_ns':11,'end_ns':12}, {'name':'LIFT','start_ns':13,'end_ns':14},
                       {'name':'HOLD','start_ns':15,'end_ns':2000000015}]}
    window = {'is_terminal':True, '_legacy':{'downstream_controller':'e1e2_squeeze', 'timestamp_ns':10,
              'closure_floor':FIXED_CONSUMER['params']['closure_floor'], 'provenance':{}}}
    prov = window['_legacy']['provenance']
    def write_record(name, checksum, record):
        f = tmp_path/(name+'.json')
        f.write_text(json.dumps(record))
        prov.update({name:str(f), checksum:hashlib.sha256(f.read_bytes()).hexdigest()})
    write_record('meta_path','metadata_sha256',trial)
    session = {'simulated':False, 'config':hardware}
    write_record('session_meta_path','session_meta_sha256',session)
    assert fixed_ready({},window,condition)['success'] is True
    window['_legacy']['timestamp_ns'] = 9
    assert not fixed_ready({},window,condition)['evaluated']
    window['_legacy']['timestamp_ns'] = 10
    trial['dropped'] = True
    write_record('meta_path','metadata_sha256',trial)
    assert fixed_ready({},window,condition)['success'] is False
    hardware['gripper']['min_normalized'] = .336
    write_record('session_meta_path','session_meta_sha256',session)
    assert not fixed_ready({},window,condition)['evaluated']


def test_initial_source_supplies_start_offset_to_both_comparison_arms():
    from bottle_scan.state_graph.campaign import expand_plan, source_initial_settings
    spec,_ = configured('fixed5')
    plan = {'cells':[{'source_node_id':'n','actions':actions,'reps':1} for actions in
                     ([],[{'kind':'tcp_translation_a','u':[2.5]}])]}
    rows = [{'node_id':'n','previous_node_id':None,'repeat_settings':{'a_offset_mm':-5.}}]
    actual = expand_plan(source_initial_settings(plan,spec,rows),spec)
    assert [x['settings']['a_offset_mm'] for x in actual] == [-5.,-5.]
    assert all('settings' not in c for c in plan['cells'])

def test_single_device_range_request_and_endpoint(tmp_path):
    from skills.contact_position_a.fixed5_once import prepare_once
    from skills.contact_position_a.ready import FIXED_DEVICE
    state, spec, cfg = prepare_once(tmp_path)
    trial, = state['trials']
    req = trial['request']
    assert req['post_condition'] == FIXED_DEVICE and req['mode'] == 'planned'
    assert req['block'] == {'mass_class':'heavy'} and req['actions'] == []
    assert req['settings']['a_offset_mm'] == 0 and req['settings']['closure_floor'] == .3
    folder = Path(trial['directory'])
    _, env = launch_request(req, folder, spec, cfg)
    assert env['CONTACT_SKILL_AFTER'] == 'stop' and float(env['RIGHT_GRIPPER_CLOSE_LIMIT']) == .3
    hardware = json.loads((folder/'scheduler_config.json').read_text())['execution_config']
    assert hardware['gripper']['min_normalized'] == .3
    calls = []
    current = {}
    def lift(target, sample):
        calls.append('lift')
        current.update(lifted=True, hold_complete=True)
    runner = SimpleNamespace(cfg=hardware, hw=SimpleNamespace(snapshot=lambda:{'width':.325}),
                             current=current, hold_and_lower=lift, release=lambda:calls.append('release'))
    events = []
    backend = SimpleNamespace(_require_supported=lambda s:None, _emit=events.append,
                              recovery=SimpleNamespace(window={'node_id':'n','group_id':'g'}))
    for achieved in (False, True):
        calls.clear()
        current.clear()
        def regulate(name, target, tol, duration):
            assert (name, target, tol, duration) == ('SQUEEZE',5.,.5,.5)
            return {'normal_sdk':[4.8,5.2]} if achieved else None
        runner.regulate = regulate
        fixed_squeeze_lift(backend, runner, None, FIXED_DEVICE)
        window = {'is_terminal':True, 'consumer_outcome':events[-1]['consumer_outcome']}
        assert fixed_ready({}, window, spec['post_conditions'][0])['success'] is achieved
        assert not fixed_ready({}, window, configured('fixed5_0330')[0]['post_conditions'][0])['evaluated']
        assert calls == (['lift','release'] if achieved else ['release'])
    req['settings']['closure_floor'] = .33
    with pytest.raises(ValueError, match='floor'):
        launch_request(req, folder, spec, cfg)
