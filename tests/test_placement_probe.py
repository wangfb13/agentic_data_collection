"""Offline placement experiments. No observed or synthetic offsets are actuator commands."""
from types import SimpleNamespace
import numpy as np
import pytest

from bottle_scan.session import atomic_json
from skills.contact_position_a import placement_probe as e
from tests.test_e1e2 import FakeHardware, config


def pose(base_x):
    p = np.eye(4)
    p[:3,:3] = [[0,0,1],[0,1,0],[-1,0,0]]
    p[0,3] = base_x
    return p


ref = dict(target_definition_id='t',reference_digest='digest',protocol={'axes_tcp':{'a':[0,0,1]}})
model = dict(model_id=e.MODEL_ID)


def row(run, placement, name, index, x, prediction, *, static=True, skew=0, gate='ok'):
    timestamp = 1_000_000_000
    p = pose(x)
    r = dict(placement=placement, trial=name, schema=e.LABEL, reference_digest='digest', model_id=e.MODEL_ID,
             state='released', status='contact', started_ns=index, before_pose=p,after_pose=p,bottle_static=static,
             window_pose={'T_base_tcp_mm':p,'received_monotonic_ns':timestamp+skew},
             probe={'valid':True,'descriptors':[prediction,gate],'window':[{'frame_ns':[timestamp]}]*2})
    atomic_json(run/placement/name/'observation.json',r)


def mock_replay(_, tactile):
    value, gate = tactile
    return dict(sample_gate=gate,raw_offset_a_mm=value,accepted_offset_a_mm=value if gate=='ok' else None,uncertainty_mm=1)


def test_score_separates_relative_and_absolute(tmp_path):
    for group, origin in [('P1',10),('P2',80)]:
        row(tmp_path,group,'center',0,origin,2)
        row(tmp_path,group,'plus',1,origin+2.7,4.9)
        row(tmp_path,group,'minus',2,origin-3,-1,gate='ood')
    atomic_json(tmp_path/'P2/datum.json',dict(label='P2_standard_witness',arm='right',read_only=True,
                pose_source={'base_frame':'base_link','arms':{'right':{'tcp_frame':'right_gripper_base_link'}}},T_base_tcp_mm=pose(80)))
    (tmp_path/'P2/datum_source.txt').write_text('synthetic fixture for unit test')
    result=e.score(tmp_path,ref,model,predict=mock_replay)
    for group in ['P1','P2']:
        center,plus,minus=[r for r in result['rows'] if r['placement']==group]
        assert plus['measured_change_mm']==pytest.approx(2.7)
        assert plus['change_error_mm']==pytest.approx(.2)
        assert minus['accepted_mm'] is None and minus['measured_change_mm']==-3
        assert result['placements'][group]['accepted']==2
    assert result['placements']['P1']['absolute_MAE_mm'] is None
    assert result['placements']['P2']['absolute_MAE_mm']==pytest.approx(2.1)


def test_moved_bottle_and_stale_tcp(tmp_path):
    row(tmp_path,'P1','center',0,0,2)
    row(tmp_path,'P1','plus',1,3,4,skew=200_000_000)
    row(tmp_path,'P1','minus',2,-3,1,static=False)
    row(tmp_path,'P1','later',3,1,2)
    result=e.score(tmp_path,ref,model,predict=mock_replay)
    assert all(r['measured_change_mm'] is None for r in result['rows'][1:])
    assert result['placements']['P1']['attempts']==4


@pytest.mark.parametrize(("answer", "confirmed"), [("YES", True), ("yes", True), (" Yes ", True), ("no", False), ("", False)])
def test_existing_runner_touches_and_opens_only(tmp_path, answer, confirmed):
    cfg=config()
    hw=FakeHardware(cfg)
    args=SimpleNamespace(run=tmp_path,placement='P2',trial='center',real=True)
    answers=iter(['PROBE',answer])
    e.capture(args,cfg,ref,model,factory=lambda _:hw,ask=lambda _:next(answers))
    record=e.read(tmp_path/'P2/center/observation.json')
    assert record['state']=='released' and record['probe']['valid']
    assert record['bottle_static'] is confirmed
    assert ('recovery_open',) in hw.events and not any(ev[0]=='move' for ev in hw.events)
    assert [ev[1] for ev in hw.events if ev[0]=='stage']==['ZERO','CLOSE','PROBE']
    assert all('T0' not in k for k in record)


def test_fault_records_and_opens(tmp_path):
    cfg=config()
    hw=FakeHardware(cfg)
    def fail():
        raise RuntimeError('fake acquisition fault')
    hw.close_to_floor=fail
    args=SimpleNamespace(run=tmp_path,placement='P1',trial='center',real=True)
    with pytest.raises(RuntimeError, match='fake acquisition fault'):
        e.capture(args,cfg,ref,model,factory=lambda _:hw,ask=lambda _:'PROBE')
    assert ('recovery_open',) in hw.events
    assert e.read(tmp_path/'P1/center/observation.json')['state']=='fault'


def test_outer_pair_even_if_center_rejected(tmp_path):
    row(tmp_path,'P1','center',0,80,1,gate='edge')
    row(tmp_path,'P1','minus',1,77,-1.5)
    row(tmp_path,'P1','plus',2,83,4)
    group=e.score(tmp_path,ref,model,predict=mock_replay)['placements']['P1']
    assert group['relative_pairs']==0
    assert group['plus_minus_span_mm']=={'measured':6.,'predicted':5.5,'error':-.5}


def test_explicit_x5_profile_changes_only_serials():
    from copy import deepcopy
    scheduler = e.read('config.state_graph.v4_scheduler.json')
    cfg = deepcopy(scheduler['execution_config'])
    cfg['hardware_config'] = 'config.ros.yaml'
    configured = e.hardware_for_placement(cfg, scheduler['hardware_config_sha256'], 'configured')
    assert configured['sensor_serials']['left'] == 'X26040505'
    alt = e.hardware_for_placement(cfg, scheduler['hardware_config_sha256'], '0318_0347')
    assert alt['public_ros']['sensors']['serials'] == {'left':'X2620410318','right':'X2620410347'}
    assert alt['sensor_serials'] == alt['public_ros']['sensors']['serials']
    assert cfg['hardware_config'].endswith('/config.ros.x5_0318_0347.yaml')
