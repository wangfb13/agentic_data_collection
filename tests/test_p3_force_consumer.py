"""Hardware-free tests for lift outcome evidence and fault recovery."""

import numpy as np
import pytest
from bottle_scan.state_graph.campaign import read
from skills.contact_position_a import p3_force


class Arm:
    def __init__(self):
        self.t=0.
        self.p=np.eye(4)
        self.cfg={'motion':dict(max_lift_mm=60.,pose_tolerance_mm=2.,near_speed_mm_s=20.,free_speed_mm_s=20.)}
        self.stage='FORCEPOLICY'
        self.overforce=False
        self.hold_drop_seen=False
        self.moves=[]
        self.stops=0

    def pose(self):return self.p.copy()
    def wait(self,t):self.t+=t
    def move(self,p,speed):
        self.moves.append(p.copy())
        self.p=p.copy()
    def motion_done(self):return True
    def stop_arm(self):self.stops+=1
    def hold(self):pass
    def ownership(self):pass
    def set_context(self,s):self.stage=s


@pytest.mark.parametrize('answer,expected',[('y',True),('n',False),('u',None)])
def test_lift_requires_bottle_evidence(tmp_path,answer,expected):
    hw=Arm()
    def tick(regulate=True):
        return dict(bilateral=True,tcp_motion_ready=True,reference_valid=True,force_ready=True)
    responses=iter(['TABLE',answer])
    o=p3_force.lift_test(hw,dict(version='fake',lift_mm=40.,hold_s=2.,grip_timeout_s=.2),tmp_path,tick,
                         ask=lambda _:next(responses),clock=lambda:hw.t)
    assert o['L1'] and o['L2'] is expected and o['measured_hold_s']>=2
    assert [x[2,3] for x in hw.moves]==[40.,0.]
    assert read(tmp_path/'airborne.json')['possibly_airborne'] is False


def test_stream_loss_in_air_is_technical_stop(tmp_path):
    hw=Arm()
    def tick(regulate=True):
        return dict(bilateral=True,tcp_motion_ready=True,reference_valid=hw.stage!='LIFT',force_ready=True)
    with pytest.raises(RuntimeError,match='reference lost'):
        p3_force.lift_test(hw,dict(version='fake',lift_mm=40.,hold_s=2.,grip_timeout_s=.2),tmp_path,tick,
                           clock=lambda:hw.t)
    assert hw.stops==1 and read(tmp_path/'airborne.json')['possibly_airborne']
    assert read(tmp_path/'consumer.json')['L2'] is None


@pytest.mark.parametrize('valid',[True,False])
def test_no_lift_without_force_ready(tmp_path,valid):
    hw=Arm()
    def tick():
        return dict(bilateral=True,tcp_motion_ready=False,reference_valid=valid,force_ready=False,reason='fixture')
    args=(hw,dict(version='fake',lift_mm=40.,hold_s=2.,grip_timeout_s=.05),tmp_path,tick)
    if valid:
        r=p3_force.lift_test(*args,clock=lambda:hw.t)
        assert r['L1'] is False and r['L2'] is None
    else:
        with pytest.raises(RuntimeError,match='prediction unavailable'):
            p3_force.lift_test(*args,clock=lambda:hw.t)
    assert not hw.moves


def test_independent_failure_seals_evidence_before_new_round(tmp_path):
    from skills.contact_position_a.p3_loop import after_independent_gap
    state=dict(frozen={'model_id':'failed1'},spent={'trials':20},last_batch={'role':'independent'})
    spec=dict(budget={'max_new_trials':60},acceptance={r+'_B':{'min_trials':12} for r in ('development','independent')})
    assert after_independent_gap(tmp_path,state,spec,{'passed':False})
    assert read(tmp_path/'confirmations/failed1/acceptance.json')=={'passed':False}
    assert state['needs_fresh_A'] and state['frozen'] is None and state['model'] is None
    assert state['retired_candidates']==['failed1']
    state.update(frozen={'model_id':'failed2'},spent={'trials':36})
    assert not after_independent_gap(tmp_path,state,spec,{'passed':False})

