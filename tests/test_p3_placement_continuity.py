"""Declined placement confirmation never forges labels or reuses stale contact."""
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from bottle_scan.state_graph.campaign import atomic,read
from skills.contact_position_a import p3_trial,p3_trial_v5
from skills.contact_position_a.p3_placement import unlabelled_contact


def window(index=0,t=100):
    return {'node_id':f'g:{index}','group_id':'g','t_ns':t,'plan_coords':{'r_offset_mm':0},
            'is_reference':True,'attrs':{'mass_class':'light'},
            'provenance':{'source_file':'original','reference_digest':'fixed','measured_pose':np.eye(4).tolist(),
                          'coordinate_source':'measured','placement_reference_valid':True,'coordinate_evidence_id':'old'}}


def test_n_reference_is_not_a_new_zero_and_does_not_stop_trial(tmp_path,monkeypatch):
    io=p3_trial.RealIO.__new__(p3_trial.RealIO)
    io.directory=tmp_path/'trial';io.directory.mkdir()
    io.spec={'facts':{'measured_placement_coordinates':True}}
    io.placement='P3';io.mass='light';io.identity=({'reference_digest':'fixed'},None,None)
    io.reference=None;io.reference_windows=[];io.request={'trial_id':'g'}
    @contextmanager
    def contact(index):
        yield unlabelled_contact(window(index),{'answer':'n','unmoved':False},original_observation='original',refreshed=False)
    io.contact=contact
    monkeypatch.setattr('builtins.input',lambda _: 'REFERENCE')
    request={'campaign':str(tmp_path),'trial_id':'g','settings':{'r_offset_mm':0,'h_offset_mm':0}}
    io.prepare(request)
    assert io.reference is None and io.reference_windows==[]
    assert not (tmp_path/'placement_references/P3/active.json').exists()
    assert read(io.directory/'preparation.json')['actual_initial_coordinates'] is None
    assert io.preparation_report['requested_initial_coordinates_executed'] is False


@pytest.mark.parametrize('index,answer,refresh_ok,expected_time',[(-1,'n',True,100),(0,'n',True,200),(0,'?',True,200),(0,'y',True,100),(0,'n',False,None)])
def test_contact_n_refreshes_before_inference(tmp_path,monkeypatch,index,answer,refresh_ok,expected_time):
    from bottle_scan import e1e2,e1e2_hardware
    created=[]
    class HW:
        def __init__(self,cfg):
            self.cfg=cfg;self.overforce=False;self.records=[];self.refreshes=0;created.append(self)
            self.arm=SimpleNamespace(snapshot=lambda:{'base_frame':'base_link'})
        def connect(self):pass
        def begin_record(self,p):self.records.append(p);p.mkdir(parents=True)
        def end_record(self):pass
        def set_context(self,*args):pass
        def zero_unloaded(self):return {'offset_raw':[[0]*6]*2}
        def pose(self):return np.eye(4)
        def hold(self):pass
        def snapshot(self):return {}
        def recovery_open(self):pass
        def close(self):pass
    class Runner:
        def __init__(self,args,cfg,fit,hw):self.hw=hw;self.last_touch_report={}
        def make(self,t):
            self.hw.window_pose={'tcp_frame':'right_gripper_base_link','T_base_tcp_mm':np.eye(4).tolist()}
            self.hw.window_masks=[np.zeros((2,2)),np.zeros((2,2))]
            self.last_touch_report={'width_normalized':.36}
            return {'valid':True,'timestamp_ns':t}
        def touch(self):return 'contact',self.make(100)
        def regulate(self,*args,**kwargs):
            self.hw.refreshes+=1
            assert kwargs['with_dd']
            return self.make(200) if refresh_ok else None
        def note_touch(self,*args):pass
    monkeypatch.setattr(e1e2_hardware,'Hardware',HW)
    monkeypatch.setattr(e1e2,'Runner',Runner)
    def captured(path,identity,center,group,index,nominal,mass,placement,pose,row):
        w=window(index,row['probe']['timestamp_ns']);w['provenance']['source_file']=str(path);return w
    monkeypatch.setattr(p3_trial.ah_evidence,'captured_window',captured)
    monkeypatch.setattr('builtins.input',lambda _: answer)
    io=p3_trial.RealIO.__new__(p3_trial.RealIO)
    io.directory=tmp_path;io.group='g';io.mass='light';io.assets=None;io.placement='P3'
    io.spec={'graph_schema':'state_graph_v5','facts':{'measured_placement_coordinates':False}}
    io.cfg={'force':dict(probe_fraction=.3,target=5,probe_tolerance=.5,probe_stable_s=.2)}
    io.identity=({'reference_digest':'fixed'},None,{'left':'L','right':'R'})
    io.reference={'valid':True};io.reference_path=tmp_path/'reference.json';io.request={'trial_id':'g','settings':{}}
    io.air_file=tmp_path/'airborne.json'
    with io.contact(index) as raw:
        if expected_time is None:assert raw is None
        else:
            assert raw['t_ns']==expected_time
            if answer!='y':
                assert raw['plan_coords']=={} and raw['is_reference'] is False
                assert raw['provenance']['placement_continuity'] is False
                assert 'coordinate_evidence_id' not in raw['provenance']
                if index>=0:assert 'after_placement_confirmation' in raw['provenance']['source_file']
        assert created[0].refreshes==int(index>=0 and answer!='y')
    if answer!='y':assert not read(io.reference_path)['valid']


def test_object_move_is_never_a_measured_recovery_edge(monkeypatch):
    io=p3_trial.RealIO.__new__(p3_trial.RealIO)
    last=window(1);last['provenance']['placement_continuity']=False
    def forbidden(*a,**kw):raise AssertionError('Cannot assign object movement to robot action')
    monkeypatch.setattr(p3_trial.ah_evidence,'link',forbidden)
    io.link(window(),last,'receipt',[0,5])
    assert 'u_receipt' not in last
    assert p3_trial_v5.pair_status(window(),last,None,{})=={'paired':False,'reason':'placement_changed_or_unconfirmed'}


def test_unlabelled_direct_trial_reaches_consumers_without_new_zero(tmp_path):
    from tests.test_v5_geometry_consumers import instance,Fake,request
    from skills.contact_position_a.p3_placement import reference_unavailable
    spec=instance(tmp_path)
    class FreeStart(Fake):
        def prepare(self,req):
            self.directory=tmp_path
            return reference_unavailable(self,req,'placement_not_confirmed')
        def invalidate_reference(self):pass
        @contextmanager
        def contact(self,index):
            with super().contact(index) as raw:
                yield unlabelled_contact(raw,{'answer':'n','unmoved':False},original_observation='synthetic',refreshed=True)
    req=request('direct_baseline');req['settings']={'r_offset_mm':0,'h_offset_mm':0}
    result=p3_trial.run_trial(req,tmp_path,spec,{},FreeStart(spec))
    assert result['status']=='complete'
    assert result['initial_coords']=={}
    assert result['preparation']['status']=='unavailable'
    assert result['consumer_outcomes'] and all(o['L2'] is True for o in result['consumer_outcomes'].values())
    assert not result['paired']
    assert len(result['consumer_outcomes'])==1
    assert result['pairing']['reason']=='second_skipped_placement_changed_or_unconfirmed'
    assert all(n.get('plan_coords')=={} for n in result['nodes'])
