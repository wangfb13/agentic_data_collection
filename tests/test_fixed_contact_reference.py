from contextlib import nullcontext
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest

from bottle_scan.state_graph.facts import coordinate_evidence
from skills.contact_position_a.contact_reference import validate_reference,local_training_sessions
from skills.contact_position_a.local_decision import decide_local
from skills.contact_position_a.local_motion import recover_once, command
from skills.contact_position_a.runtime import live_window
from skills.contact_position_a.scheduler import trials_from_events

REF=validate_reference(json.loads((Path(__file__).resolve().parents[1]/'skills/contact_position_a/white_reference.json').read_text()))
ROT=np.array([[0.,0.,1.],[0.,1.,0.],[-1.,0.,0.]])


class Backend(NS):
    @property
    def T0(self):raise AssertionError('unexpected absolute T0')
    @property
    def bounds(self):raise AssertionError('unexpected legacy bounds')
    @property
    def robot(self):raise AssertionError('unexpected camera access')


def make_backend():
    policy=NS(predict=lambda ds:dict(valid=True,action_mm=-5.,uncertainty_mm=1.),
              fit={'trained_offset_range_mm':[-6,6],'model_id':'fixture','settings':{'max_action_mm':5.}})
    cfg={'servo':{'max_regrasps':3,'retreat_mm':10.,'max_delta_mm':3.},'motion':{'pose_tolerance_mm':1.,'max_grasp_offset_mm':20.},
         'gripper':{'min_normalized':.33}}
    hw=NS(now=lambda:1000,lease=NS(is_set=lambda:True),zero={'contact_thresholds_sdk':[.1,.1]})
    return Backend(contact_reference=REF,cfg=cfg,config={},offset_policy=policy,hw=hw,action_source='offset_servo',
        local_settings={'gain':1.,'offset_deadband_mm':1.,'max_steps':2,'observation_max_age_s':.5},
        station={'tcp_min_base_mm':[-100,-100,-100],'tcp_max_base_mm':[100,100,100],'allowed_rotations':[ROT,np.eye(3)],
                 'orientation_tolerance_deg':1.},executing=True,current_observation_ns=1000,current_observation_id='obs',
        _sample=lambda:{'normal':[1.,1.]},_require_supported=lambda s:None)


def observation():
    return dict(observation_id='obs',timestamp_ns=1000,bilateral=True,safety_ok=True,force_stable=True,
                descriptors=REF['standard_sample']['descriptors'])


class Graph:
    def __init__(self,state):self.state=state
    def infer(self,row,*,readiness_only):
        if self.state=='not_ready':assert readiness_only
        return {'state':self.state,'action':{'kind':'tcp_translation_a','u':[2.]}}


def test_offset_gain_clip_and_ready():
    b=make_backend();o=observation()
    s,_,cmd,p=decide_local(b,o,{},Graph('not_ready'),0)
    assert s=='recoverable' and cmd['u']==[-3] and p['offset_estimate']['offset_a_mm']==5
    b.local_settings['gain']=.2
    s,_,cmd,p=decide_local(b,o,{'plan_coords':{'a_offset_mm':99},'T0_mm':'other','placement':'other'},Graph('not_ready'),0)
    assert cmd['u']==[-1] and p['offset_estimate']['offset_a_mm']==5 and p['offset_estimate']['offset_l_mm'] is None
    s,_,cmd,p=decide_local(b,o,{},Graph('ready'),0)
    assert s=='ready' and cmd is None and p['offset_estimate']['offset_a_mm']==5


@pytest.mark.parametrize('fault',['unknown','exhausted','contact_lost','edge','stale','zero_unconfirmed','ood'])
def test_no_false_success(fault):
    b=make_backend();o=deepcopy(observation())
    if fault=='edge':o['descriptors'][0]['touches_edge']=True
    if fault=='stale':o['timestamp_ns']=-1000000000
    if fault=='contact_lost':o['bilateral']=False
    if fault=='zero_unconfirmed':b.offset_policy.predict=lambda ds:dict(valid=True,action_mm=0,uncertainty_mm=1)
    if fault=='ood':b.offset_policy.predict=lambda ds:dict(valid=False,reason='ood')
    s,_,cmd,_=decide_local(b,o,{},Graph('Unknown' if fault=='unknown' else 'not_ready'),2 if fault=='exhausted' else 0)
    assert s=='Unknown' and cmd is None


def test_graph_independent_of_offset_policy():
    b=make_backend();b.action_source='graph_policy';del b.offset_policy
    assert decide_local(b,observation(),{},Graph('Unknown'),0)[2] is None
    s,_,cmd,report=decide_local(b,observation(),{},Graph('recoverable'),0)
    assert cmd['u']==[2] and report['offset_estimate'] is None


@pytest.mark.parametrize('rot',[np.eye(3),ROT])
@pytest.mark.parametrize('touch_ok',[False,True])
def test_measured_contact_to_contact_in_start_frame(rot,touch_ok):
    b=make_backend();pose=np.eye(4);pose[:3,:3]=rot;pose[:3,3]=[19,2,5];start=pose.copy();moves=[]
    b.hw.pose=lambda:pose.copy();b.current_observation_pose=start.copy()
    def move(t):
        moves.append(t.copy());pose[:]=t
    def touch():
        pose[:3,3]+=.2*rot[:,2]
        return 'contact',{'valid':touch_ok}
    b._observation=lambda status,p:dict(force_stable=p['valid'],bilateral=True,safety_ok=True)
    runner=NS(release=lambda:None,stage=lambda _:nullcontext(),move=move,touch=touch,current={'regrasps':0})
    cmd=command(2.,observation(),REF['target_definition_id'],'offset_servo');row={}
    if not touch_ok:
        with pytest.raises(RuntimeError,match='Recontact'):recover_once(b,runner,cmd,row)
        assert row['receipt']['applied'] is False
        return
    after,_,_,row=recover_once(b,runner,cmd,row)
    assert row['receipt']['action_mm']==pytest.approx(2.2)
    assert np.allclose(after[:3,3]-start[:3,3],2.2*rot[:,2])
    assert np.allclose(moves[0][:3,3]-start[:3,3],-10*rot[:,2])


def test_live_and_import_tactile_only_no_T0_no_ground_truth(tmp_path):
    b=make_backend();b.output_dir=tmp_path;folder=tmp_path/'invocation_001_123';folder.mkdir()
    spec=json.loads((Path(__file__).resolve().parents[1]/'skills/contact_position_a/spec.campaign_v4.json').read_text())
    del spec['observation']['modalities']['vis']
    ref=np.zeros((32,32,2));dd=ref.copy();dd[6:24,9:25,0]=.2
    events=[NS(monotonic_ns=i,data=dd) for i in range(1,6)]
    b.hw.dd_reference=[ref,ref];b.hw.dd=dict(left=events,right=events)
    b.hw.zero={'offset_raw':[[0]*6]*2};b.hw.lock=nullcontext()
    b._graph_buffers={'wrench_raw_'+s:[NS(monotonic_ns=3,data=np.ones(6))] for s in ('left','right')}
    b.hw.snapshot=lambda:dict(pose=np.eye(4),wrench_zeroed=np.ones(12),width=.41,timestamp_ns=5)
    probe={'timestamp_ns':5,'stable_start_ns':1,'descriptors':REF['standard_sample']['descriptors'],
           'window':[{'frame_ns':list(range(1,6))}]*2}
    e,row=live_window(b,probe,observation(),folder,0,None,None,spec)
    assert set(row['obs'])=={'tac','force'} and e['supervision']['offset_a_mm'] is None and not e['plan_coords']
    assert 'T0_mm' not in e and not e['raw_evidence']['wrist']['available']
    imported=next(trials_from_events([e],spec))['windows'][0]
    assert not imported['is_reference'] and imported['provenance']['reference_digest']==REF['reference_digest']
    imported['plan_coords']={'a_offset_mm':0}
    assert coordinate_evidence(imported,'a_offset_mm') is None


def test_unwitnessed_A_source_is_not_reidentified():
    manifest={'reference_digest':REF['reference_digest'],'measured_A_sessions':{}}
    with pytest.raises(ValueError,match='Unverified'):
        local_training_sessions([dict(metadata_sha256='new',path='new')],REF,manifest)
