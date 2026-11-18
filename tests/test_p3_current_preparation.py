"""P3 preparation never inherits the legacy fixed-P2 orientation gate."""
from copy import deepcopy
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from skills.contact_position_a import p3_prepare,campaign_ah,placement_witness
from bottle_scan.state_graph.campaign import read


def pose(yaw):
    p=np.eye(4);p[:3,:3]=Rotation.from_euler('zy',[-yaw,90],degrees=True).as_matrix()
    p[2,3]=850.
    return p


def spec():
    return {'geometry':{'horizontal_basis':True,'minimum_horizontal_projection':.1,
            'coordinate_residual_mm':1.,'orientation_tolerance_deg':1.},
            'perturbation_variables':{'r_offset_mm':{'envelope':[-10,10]},'h_offset_mm':{'envelope':[-3,45]}},
            'task':{'action_space':{'max_step':[10,45]}}}


def fake(monkeypatch,initial):
    state={'pose':initial.copy(),'moves':[]}
    monkeypatch.setattr(campaign_ah,'current_pose',lambda *args:state['pose'].copy())
    def move(folder,label,arguments,limit,z,invoke):
        xyz=np.asarray(arguments[1:]);state['moves'].append((xyz.copy(),limit,z))
        state['pose'][:3,3]=xyz
        return folder/(label+'.json')
    monkeypatch.setattr(campaign_ah,'move',move)
    def old(*args,**kw):raise AssertionError('Legacy P2 preparation/check must never execute')
    monkeypatch.setattr(campaign_ah,'prepare_position',old)
    monkeypatch.setattr(campaign_ah,'check_measured_pose',old)
    return state


@pytest.mark.parametrize('yaw',[2.725,90.,180.])
def test_zero_start_no_move_no_old_orientation_gate(tmp_path,monkeypatch,yaw):
    ref=pose(0);now=pose(yaw);now[:3,3]+=[80.,-30.,10.]
    state=fake(monkeypatch,now)
    report=p3_prepare.prepare_position(tmp_path,ref,[0,0],{'axes_tcp':{'a':[0,0,1]}},spec(),None)
    assert state['moves']==[]
    assert report['actual_initial_coordinates'] is None
    assert read(tmp_path/'preparation.json')['mode']=='current_pose_no_preparation_motion'
    np.testing.assert_equal(state['pose'],now)


def test_nonzero_uses_current_r_direction_and_base_h(tmp_path,monkeypatch):
    ref=pose(0);now=pose(90)
    state=fake(monkeypatch,now)
    p3_prepare.prepare_position(tmp_path,ref,[5,8],{'axes_tcp':{'a':[0,0,1]}},spec(),None)
    # y=90 points current TCP z along base +Y, instead of old base +X.
    direction=now[:3,:3]@np.array([0.,0.,1.]);direction[2]=0;direction/=np.linalg.norm(direction)
    np.testing.assert_allclose(state['moves'][0][0],now[:3,3]+direction*5,atol=1e-10)
    np.testing.assert_allclose(state['moves'][1][0],now[:3,3]+direction*5+[0,0,8],atol=1e-10)


def test_no_undeclared_lateral_correction(tmp_path,monkeypatch):
    ref=pose(0);now=pose(0);now[1,3]=3
    state=fake(monkeypatch,now)
    p3_prepare.prepare_position(tmp_path,ref,[5,0],{'axes_tcp':{'a':[0,0,1]}},spec(),None)
    assert state['pose'][1,3]==3


def test_world_height_and_range_still_checked(tmp_path,monkeypatch):
    ref=pose(0);state=fake(monkeypatch,ref)
    with pytest.raises(ValueError,match='envelope'):
        p3_prepare.prepare_position(tmp_path,ref,[11,0],{'axes_tcp':{'a':[0,0,1]}},spec(),None)
    with pytest.raises(ValueError,match='高度'):
        p3_prepare.prepare_position(tmp_path,ref,[0,10],{'axes_tcp':{'a':[0,0,1]}},spec(),None,station_z_bounds=[840,855])
    assert not state['moves']


def test_unmatched_reference_does_not_fabricate_zero_label():
    ref={'valid':True,'reference_digest':'fixed','mass_class':'light','pose':pose(0).tolist(),'placement_reference_id':'w1'}
    w={'provenance':{'reference_digest':'fixed','measured_pose':pose(90).tolist()},'attrs':{'mass_class':'light'},'plan_coords':{'r_offset_mm':0}}
    placement_witness.label(w,ref,[0,0,1],spec())
    assert w['plan_coords']=={} and not w['provenance']['placement_reference_valid']


def test_realio_routes_zero_free_start_without_legacy_prepare(tmp_path,monkeypatch):
    from bottle_scan.state_graph.campaign import atomic
    from skills.contact_position_a.p3_trial import RealIO
    now=pose(90);now[:3,3]+=[2,3,4]
    state=fake(monkeypatch,now)
    io=RealIO.__new__(RealIO)
    io.spec=spec();io.spec['facts']={'measured_placement_coordinates':True}
    io.placement='P3';io.mass='light';io.identity=({'reference_digest':'fixed'},None,None)
    io.directory=tmp_path/'trial';io.directory.mkdir()
    io.cfg={'axes_tcp':{'a':[0,0,1]}};io.z_limits=None
    ref={'valid':True,'mass_class':'light','reference_digest':'fixed','pose':pose(0).tolist(),'placement_reference_id':'test'}
    atomic(tmp_path/'placement_references/P3/active.json',ref)
    monkeypatch.setattr('builtins.input',lambda _: 'y')
    io.prepare({'campaign':str(tmp_path),'settings':{'r_offset_mm':0,'h_offset_mm':0},'trial_id':'test'})
    assert not state['moves']
    assert read(io.directory/'preparation.json')['actual_initial_coordinates'] is None
