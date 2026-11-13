from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import json
import numpy as np
import pytest
from bottle_scan.state_graph.graph import build_graph
from bottle_scan.state_graph.labels import generate_labels,shortest_path
from bottle_scan.state_graph.policy import fit_policy
from bottle_scan.state_graph.skill import SymbolicRecoverySkill
from bottle_scan.state_graph.skill_interface import normalize_action
from bottle_scan.state_graph.director_bridge import update_hypotheses
from bottle_scan.state_graph.clusters import reconcile_regions
from bottle_scan.state_graph.acceptance import evaluate_B
from tests.fixtures.skill_toy.testing import data,trained


def test_similarity_cannot_teleport_to_a_ready_sink(monkeypatch):
    rows,spec,noise=data();rows=deepcopy(rows[:2]);rows[0]['previous_node_id']=rows[1]['previous_node_id']=None
    probabilities={c['id']:{r['node_id']:{'n':4,'p_hat':.9 if i else .2,'ready':bool(i)} for i,r in enumerate(rows)} for c in spec['post_conditions']}
    sinks=[{'source':rows[1]['node_id'],'target':'__post__:'+c['id'],'kind':'sink','cost':0.} for c in spec['post_conditions']]
    monkeypatch.setattr('bottle_scan.state_graph.graph.pairwise_distances',lambda *args:np.zeros((2,2)))
    monkeypatch.setattr('bottle_scan.state_graph.graph.estimate_sinks',lambda *args:(probabilities,sinks))
    graph=build_graph(rows,noise,spec)
    assert graph['counts']['similar']==2
    for c in spec['post_conditions']:
        assert graph['R'][c['id']][rows[0]['node_id']] is None
        assert graph['R'][c['id']][rows[1]['node_id']]==0
        assert shortest_path(graph,rows[0]['node_id'],c['id']) is None
        assert graph['reachability'][c['id']][rows[0]['node_id']]['status']=='unsupported'


def test_probe_is_not_a_zero_translation_and_path_keeps_typed_order():
    _,spec,_=data();spec=deepcopy(spec)
    spec['capabilities']={'translation_xy':{'kind':'vector','max_action':[1,1]},'probe_load':{'kind':'probe','parameters':{'height':[0,2]}}}
    action=normalize_action({'kind':'probe_load','parameters':{'height':1}},spec)
    assert 'u' not in action
    with pytest.raises(ValueError):normalize_action({'kind':'undeclared','u':[0,0]},spec)
    graph={'edges':[{'edge_id':'p','source':'s','target':'t','kind':'observed','cost':1,'action':action,'u':None},
                    {'edge_id':'g','source':'t','target':'__post__:centered','kind':'sink','cost':0}],
           'R':{'centered':{'s':1,'t':0}},'probabilities':{'centered':{'s':{'p_hat':.1},'t':{'p_hat':.9}}},
           'reachability':{'centered':{'s':{'status':'recoverable'},'t':{'status':'ready'}}}}
    spec['post_conditions']=spec['post_conditions'][:1]
    labels=generate_labels(graph,[{'node_id':'s'},{'node_id':'t'}],spec)
    assert labels['nodes'][0]['first_action']==action and labels['nodes'][0]['action_target'] is None


def test_runtime_does_not_call_a_trained_policy_outside_applicability():
    rows,*_,bundle,status=trained();runtime=SymbolicRecoverySkill(bundle)
    row=deepcopy(rows[0]);row['obs']={k:(np.asarray(v)+1e6).tolist() for k,v in row['obs'].items()}
    result=runtime.infer(row)
    assert result['state']=='Unknown' and result['policy'] is None
    model=bundle['models']['centered']['pi']
    assert np.mean(model['sample_weights'])==pytest.approx(1)


def test_hypotheses_need_new_evidence_not_just_a_new_graph_hash():
    h={'id':'h','status':'open','proposed_at_graph':'g1','proposed_evidence':['n1'],'support_when':{'pred':'gap'},'refute_when':{'pred':'closed'}}
    f={'id':'f','pred':'gap','value':True,'evidence':['n1'],'args':[]}
    assert update_hypotheses([h],[f],'g2')[0]['status']=='open'
    f['evidence'].append('n2')
    assert update_hypotheses([h],[f],'g2')[0]['status']=='supported'
    assert update_hypotheses([h],[f],'g1')[0]['status']=='supported'


def test_region_names_survive_added_members_and_split_is_recorded():
    old={'clusters':[{'cluster_id':'C_old','node_ids':['a','b'],'region_anchor':'a'}]}
    new={'clusters':[{'cluster_id':'C_new','node_ids':['a','b','c']}]}
    out=reconcile_regions(new,old)['clusters'][0]
    assert out['cluster_id']=='C_old' and out['region_anchor']=='a' and out['membership_version']=='C_new'


def test_b_does_not_pool_models_or_hide_rejections():
    _,spec,_=data();spec=deepcopy(spec);spec['acceptance']={'development_B':{'min_trials':3,'max_rejection_rate':.2}}
    records=[{'role':'development','post_condition':'centered','group_id':str(i),'model_id':'m' if i<3 else 'other',
              'evaluated':i!=2,'rejected':i==2} for i in range(6)]
    result=evaluate_B(records,spec,'development',model_id='m')
    assert result['metrics']['trials']==3 and result['metrics']['rejection_rate']==pytest.approx(1/3)
    assert not result['passed']


def test_v4_proxy_uses_graph_decisions_and_existing_action_interface(tmp_path,monkeypatch):
    from skills.contact_position_a.runtime import GraphRecoveryProxy
    rows,spec,_,_,_,_,_,bundle,_=trained();spec=deepcopy(spec)
    spec['task']['action_space'].update(id='tcp_translation_a',dim=1,names=['a']);spec['safety_envelope']['max_action']=[15]
    folder=tmp_path/'invocation_1';folder.mkdir()
    backend=SimpleNamespace(output_dir=tmp_path,_graph_probe={},T0=np.eye(4),config={})
    request={'mode':'planned','actions':[{'kind':'tcp_translation_a','u':[2]}],'model_id':None}
    proxy=GraphRecoveryProxy(backend,request,spec)
    def live(*args):
        index=args[4];row={'node_id':'n'+str(index)}
        return {'node_id':row['node_id'],'attrs':{}},row
    monkeypatch.setattr('skills.contact_position_a.runtime.live_window',live)
    first=proxy.start({'bilateral':True,'force_stable':True})
    assert first['invocation']['invoke'] is True and first['recovery']['action']['delta_mm']==2
    second=proxy.advance({'bilateral':True,'force_stable':True},applied_action={'applied':True,'action_mm':2})
    assert second['exit']['hand_back'] and second['exit']['completed']
    assert second['exit']['target_observed'] is False # exploration is an intervention, not a learned success


def test_runtime_requires_action_receipt_before_observation():
    *_,bundle,_=trained();skill=SymbolicRecoverySkill(bundle);calls=[]
    skill.infer=lambda row:{'state':'recoverable','value':{},'policy':{'u':[.1,0]},'plan':None}
    adapter=SimpleNamespace(observe=lambda:calls.append('observe') or {},execute=lambda action:{'applied':False})
    result=skill.execute(adapter)
    assert result['reason']=='action_not_completed' and calls==['observe']


def test_live_and_offline_use_identical_window_features(tmp_path,monkeypatch):
    from threading import RLock
    from collections import defaultdict
    from skills.contact_position_a.runtime import live_window
    from skills.contact_position_a.scheduler import trials_from_events
    from bottle_scan.state_graph.extract_nodes import extract_nodes
    from bottle_scan.state_graph.skill_interface import load_spec
    spec=load_spec('skills/contact_position_a/spec.campaign_v4.json')
    folder=tmp_path/'invocation_1_123';folder.mkdir()
    baseline=np.zeros((8,8,3),np.float32);frames=[]
    for i in range(5):
        frame=baseline.copy();frame[2:5,1+i%2:4+i%2,0]=.1;frames.append(frame)
    times=list(range(10,60,10))
    class Camera:
        latest_timestamp=1.;latest_frame_id=3;config=SimpleNamespace(color_mode='rgb')
        def read_latest(self,max_age_ms):return np.zeros((8,8,3),np.uint8)
    sample={'pose':np.eye(4),'timestamp_ns':50,'wrench_zeroed':np.ones((2,6)),'normal':np.ones(2),'width':.4}
    buffers=defaultdict(list)
    for side in ('left','right'):buffers['wrench_raw_'+side]=[SimpleNamespace(monotonic_ns=t,data=np.ones(6)*i) for i,t in enumerate(times)]
    hw=SimpleNamespace(lock=RLock(),dd_reference=[baseline,baseline],zero={'contact_thresholds_sdk':[.1,.1],'offset_raw':np.zeros((2,6))},
                       dd={side:[SimpleNamespace(monotonic_ns=t,data=f) for t,f in zip(times,frames)] for side in ('left','right')},snapshot=lambda:sample)
    backend=SimpleNamespace(hw=hw,_graph_buffers=buffers,config={},robot=SimpleNamespace(cameras={'right_wrist':Camera()}),
                            T0=np.eye(4),bounds=SimpleNamespace(a=np.array([1,0,0.])),output_dir=tmp_path,cfg={'gripper':{'min_normalized':.33}},_graph_consumer=None)
    monkeypatch.setattr('skills.contact_position_a.legacy_v2.features.FrozenVision',lambda *a:SimpleNamespace(encode=lambda x:[.1]*384))
    probe={'valid':True,'stable_start_ns':10,'timestamp_ns':50,'descriptors':[{'valid':True},{'valid':True}],
           'window':[{'frame_ns':times,'frame_fids':list(range(5))} for _ in range(2)]}
    event,online=live_window(backend,probe,{'bilateral':True,'force_stable':True},folder,0,None,None,spec)
    path=tmp_path/'events.jsonl';path.write_text(json.dumps(event)+'\n')
    offline,_=extract_nodes([str(path)],spec)
    assert offline[0]['obs']==online['obs'] and offline[0]['extractor_versions']==online['extractor_versions']
    assert offline[0]['obs']['force']==[2.]*12
    assert not (folder/'streams.h5').exists() # the exact recorded live values suffice


def test_embedding_dynamics_projection_is_training_fitted_and_keeps_physical_targets():
    from bottle_scan.state_graph.noise import fit_noise
    from bottle_scan.state_graph.dynamics import fit_dynamics,evaluate_dynamics
    rows,spec,_=data();rows=deepcopy(rows);spec=deepcopy(spec)
    for r in rows:r['obs']['extra']=np.tile(r['obs']['geometry'],20).tolist();r['extractor_versions']['extra']='test'
    spec['observation']['modalities']['extra']={'version':'test'}
    spec['dynamics'].update(inputs=['geometry','load','extra'],input_pca_dim=4,ensemble=2,max_iter=50)
    noise=fit_noise(rows,spec);f=fit_dynamics(rows,noise,spec)
    assert f['targets']==spec['dynamics']['targets']
    assert len(f['ensemble']['projectors'])==4
    assert f['ensemble']['projectors'][2][2].n_components_==4
    sample=evaluate_dynamics(f,rows[:100],spec)['samples'][0]
    assert set(sample['baseline_absolute_error'])=={'identity','mean_delta','linear'}
