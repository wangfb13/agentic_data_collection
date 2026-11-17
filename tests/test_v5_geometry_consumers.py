"""Controlled placements, receipts and matched downstreams. All data are synthetic."""
from contextlib import contextmanager
from copy import deepcopy
import json
from pathlib import Path
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from skills.contact_position_a import ah_geometry as g, ah_evidence as ev
from skills.contact_position_a import ah_v5_config, campaign_ah, lift_evidence as lift
from skills.contact_position_a import p3_trial, p3_trial_v5, placement_witness as placement
from bottle_scan.state_graph import campaign
from bottle_scan.state_graph.v5_profile import configure


def pose():
    t=np.eye(4)
    t[:3,:3]=Rotation.from_euler('y',55,degrees=True).as_matrix()
    return t


def instance(tmp_path):
    spec,cfg=campaign_ah.configuration(tmp_path,center=pose())
    spec['current_placement']='P3'
    spec=configure(spec,{'observation':{'tac_version':'legacy_pixel'}})
    spec['post_conditions'][0]['consumer']['version']='synthetic_fp'
    return ah_v5_config.configure(spec,cfg)


def window(index,coords=(0,0)):
    p=pose()
    p[0,3],p[2,3]=coords
    return dict(node_id=f'fake:{index}',trial_id='fake',group_id='fake',session='P3',window_index=index,
                t_ns=10+index,phase='PROBE',preconditions={'bilateral_contact':True},attrs={'mass_class':'light'},
                contacts=[{'u_px':8+x,'v_px':6+y,'area_px':12} for x,y in [coords,coords]],
                wrench=[1.]*12,repeat_settings={},plan_coords=dict(zip(g.RH_COORDS,coords)),
                provenance=dict(measured_pose=p.tolist(),coordinate_source='measured',placement_reference_valid=True,
                                placement_reference_id='test',reference_digest='standard',source_file='synthetic'))


def test_axes_and_legacy_reproject_base_vector(tmp_path):
    spec=instance(tmp_path)
    orig=pose()
    for u in ([7,0],[0,8]):
        new=g.target(orig,orig,u,[0,0,1],spec)
        rec,residual=g.measured(orig,new,orig,[0,0,1],spec)
        np.testing.assert_allclose(rec,u,atol=1e-12)
        assert residual<1e-10 and new[2,3]==pytest.approx(u[1])
    with pytest.raises(ValueError):g.basis(np.eye(4),[0,0,1],spec)
    old=window(2)
    old['plan_coords']={'a_offset_mm':5,'h_offset_mm':-2}
    old['provenance']['placement_reference_pose']=orig.tolist()
    old.update(u_receipt={'kind':g.KIND,'applied':True,'start_pose':orig.tolist(),'net_contact_displacement_mm':[5,-2]},
               u_executed=[5,-2])
    row=g.convert_legacy_row(old,spec)
    np.testing.assert_allclose(g.basis(orig,[0,0,1])@[5,-2],g.basis(orig,[0,0,1],spec)@row['u_executed'])
    assert row['provenance']['legacy_plan_coords']==old['plan_coords']
    assert row['plan_coords']['h_offset_mm']==pytest.approx(row['u_executed'][1])


def test_receipt_and_amplitude(tmp_path):
    spec=instance(tmp_path)
    x,y=window(0),window(1,(4,0))
    a=x['provenance']['measured_pose'];b=y['provenance']['measured_pose']
    log=tmp_path/'move.json'
    log.write_text(json.dumps(dict(result='reached_by_measured_feedback',real=True,arm='right',base_frame='base_link',
        tcp_frame='right_gripper_base_link',execution_start_mm=a,end_measured_mm=b,frame_semantics='action_start_v1',
        relative_frame='horizontal_tcp_a')))
    ev.link(x,y,[log],[0,0,1],command=[4,0],spec=spec)
    assert y['previous_node_id']==x['node_id']
    receipt=y['u_receipt']
    assert receipt['kind']==g.RH_KIND and 0<receipt['raw_tcp_a_projection_mm']<4
    assert receipt['out_of_plane_residual_mm']==pytest.approx(0)
    cov=ev.amplitude_coverage([{'receipt':receipt,'edge_id':'real_measured'}],[-10,10],'P3',0,1,spec)
    assert cov['evidence']==['real_measured']


def test_coordinate_witness_and_invalidation(tmp_path):
    spec=instance(tmp_path)
    w=window(1,(3,4))
    ref=placement.witness(window(0),'P3','standard')
    placement.label(w,ref,[0,0,1],spec)
    np.testing.assert_allclose(list(w['plan_coords'].values()),[3,4])
    from bottle_scan.state_graph.facts import coordinate_evidence
    assert coordinate_evidence(w,'r_offset_mm')==3
    placement.label(w,{**ref,'valid':False},[0,0,1],spec)
    assert w['plan_coords']=={} and coordinate_evidence(w,'h_offset_mm') is None


def test_new_validation_multistep_and_old_mode(tmp_path):
    spec=instance(tmp_path)
    action=lambda u:{'kind':g.RH_KIND,'u':u}
    moves=[action([2,0]),action([0,5])]
    check=lambda seq:g.validate_cell({'settings':{'r_offset_mm':1},'actions':seq},spec,'',{})
    check(moves)
    with pytest.raises(ValueError):check(moves+moves)
    with pytest.raises(ValueError):check([action([8,0]),action([2,0])])
    old,_=campaign_ah.configuration(tmp_path,center=pose())
    with pytest.raises(ValueError):g.validate_cell({'actions':[{'kind':g.KIND,'u':[1,0]}]*2},old,'',{})


def test_auto_l2_and_force_saturation(tmp_path):
    cfg={**instance(tmp_path)['consumer'],'expected_weight_n':1.,'normal_axis':2,
         'load_calibration':{'calibration_id':'fixture','upward_tangent_n_per_sdk':[[1,0,0],[1,0,0]]}}
    def sample(f,shift=0,contact=True):
        return dict(bilateral=contact,wrench_zeroed=[[f,0,1,0,0,0]]*2,centroids=[[shift,0],[0,0]])
    pre,up=sample(0),sample(.5)
    auto=lambda hold:lift.automatic(pre,[up],hold,True,True,cfg)
    assert auto([up,up])[:2]==(True,None)
    assert auto([up,sample(.5,contact=False)])[:2]==(False,'contact_lost')
    assert auto([up,sample(.5,7)])[:2]==(False,'slip_during_hold')
    assert auto([sample(.2)])[:2]==(False,'load_weight_mismatch')
    assert lift.automatic(pre,[up],[up],True,True,{**cfg,'load_calibration':None})[0] is None
    band=lift.ForceBand(cfg)
    band.update(0,.33,1,True)
    band.update(0,.33,1,True)
    assert not band.saturated
    band.update(.6,.33,1,True)
    assert band.saturated
    band.update(.8,.33,5.,True)
    band.update(1.5,.33,5.,True)
    assert band.ready and not band.saturated


class Fake:
    def __init__(self,spec,start=(0.,0.),shift=0,drop=False):
        self.pos=np.array(start,float)
        self.spec=spec
        self.seen=[]
        self.shift=shift
        self.drop=drop

    def confirm(self,req):pass
    @contextmanager
    def contact(self,index):
        if self.seen:self.pos[0]+=self.shift
        yield window(index,self.pos.tolist())

    def move(self,a,w):
        self.pos+=a['u']
        return 'feedback'
    def link(self,a,b,receipt,u):
        b.update(previous_node_id=a['node_id'],u_executed=u,u_receipt={'applied':True,'kind':g.RH_KIND})
    def consume_for(self,cid):
        self.seen.append(cid)
        cond=next(x for x in self.spec['post_conditions'] if x['id']==cid)['consumer']
        return dict(controller=cond['controller'],version=cond['version'],params=cond['params'],L1=True,
                    L2=not self.drop,contact_lost=self.drop,evidence=['fake'])


def request(role,actions=[],**kw):
    return dict(trial_id='fake',settings={},role=role,mode='expert' if role=='expert_recovery' else 'planned',
                actions=actions,round_index=2,data_role='exploration',block={'mass_class':'light'},
                target_consumer='forcepolicy_lift',post_condition='forcepolicy_lift',consumer_seed=4,**kw)


def test_two_moves_make_two_sequential_edges_and_two_terminals(tmp_path):
    spec=instance(tmp_path)
    actions=[{'kind':g.RH_KIND,'u':u} for u in ([1,0],[0,3])]
    io=Fake(spec)
    result=p3_trial.run_trial(request('exploration',actions),tmp_path,spec,{},io)
    rows=result['nodes']
    assert [r['previous_node_id'] for r in rows]==[None,'fake:0','fake:1',None]
    assert len(io.seen)==2 and result['consumer_order']==io.seen
    assert result['pairing']['reason']=='no_prior_pair_metric'
    assert [sum(p['evaluated'] for p in r['post'].values()) for r in rows]==[0,0,1,1]
    assert result['terminal_outcomes']['fixed_force_lift']['success']
    assert result['cost']['actions']==2


def test_expert_and_pairing(tmp_path,monkeypatch):
    spec=instance(tmp_path)
    io=Fake(spec,start=(2,5))
    req=request('expert_recovery',expert_source='scripted_coordinate')
    result=p3_trial.run_trial(req,tmp_path,spec,{},io)
    assert [a['u'] for a in result['action_commands']]==[[0,-5],[-2,0]]
    assert all(n['u_receipt']['source']=='expert' for n in result['nodes'] if n.get('u_receipt'))
    free=window(4);free['provenance']['coordinate_source']='estimate'
    with pytest.raises(ValueError):p3_trial_v5.expert_action(free,spec)
    bundle={'graph':{'radii':{'similar_radius':.5}},'noise':{},'model_id':'s'}
    monkeypatch.setattr(p3_trial_v5,'measurement_distance',lambda a,b,n:.3)
    assert p3_trial_v5.pair_status(window(1),window(2),bundle,spec)['paired']
    assert not p3_trial_v5.pair_status(window(1),window(2,(2,0)),bundle,spec)['paired']
    assert p3_trial_v5.consumer_plan(req['post_condition'],spec,req)==result['consumer_order']


def test_lost_contact_skips_second_consumer(tmp_path):
    spec=instance(tmp_path);io=Fake(spec,drop=True)
    res=p3_trial.run_trial(request('direct_baseline'),tmp_path,spec,{},io)
    assert len(io.seen)==1 and res['pairing']['reason']=='second_skipped_object_or_contact_change'


def test_target_selection_uses_named_frozen_model(tmp_path,monkeypatch):
    spec=instance(tmp_path)
    bundle=dict(model_id='frozen',spec=spec,support=[],graph={'radii':{'similar_radius':.4}},
                noise={},models={'forcepolicy_lift':{'status':'recoverable'},'fixed_force_lift':{'status':'ready'}})
    monkeypatch.setattr(p3_trial_v5.joblib,'load',lambda _:bundle)
    monkeypatch.setattr(p3_trial_v5,'measurement_distance',lambda a,b,n:0.)
    class Frozen:
        def __init__(self,package):
            cid=package['spec']['active_post_condition']
            self.state=package['models'][cid]['status']
        def infer(self,row):
            state=self.state if row['window_index']==0 else 'ready'
            return {'state':state,'action':{'kind':g.RH_KIND,'u':[0,4]}}
    monkeypatch.setattr(p3_trial_v5,'SymbolicRecoverySkill',Frozen)
    for i,(target,n) in enumerate([('fixed_force_lift',0),('forcepolicy_lift',1)]):
        path=tmp_path/str(i);path.mkdir()
        req=request('policy_frontier')
        req.update(mode='learned',model_id='frozen',model_bundle='synthetic',
                   target_consumer=target,post_condition=target)
        out=p3_trial.run_trial(req,path,spec,{},Fake(spec),sources=[])
        assert out['cost']['actions']==n and set(out['terminal_outcomes'])==set(bundle['models'])


def test_random_audit_disagreement_preserved(tmp_path):
    from tests.test_p3_force_consumer import Arm
    from skills.contact_position_a import p3_force
    hw=Arm()
    spec=instance(tmp_path)
    params={**spec['consumer'],'version':'fake','audit_selected':True,'normal_axis':2,
            'expected_weight_n':1.,'load_calibration':{
                'calibration_id':'s','upward_tangent_n_per_sdk':[[1,0,0],[1,0,0]]}}
    def tick(regulate=True):
        f=.5 if hw.stage in ('LIFT','HOLD') else 0.
        return dict(bilateral=True,reference_valid=True,tcp_motion_ready=True,force_ready=True,
                    centroids=[[4,5],[6,7]],wrench_zeroed=[[f,0,0,0,0,0]]*2)
    replies=iter(['TABLE','n'])
    o=p3_force.lift_test(hw,params,tmp_path,tick,ask=lambda msg:next(replies),clock=lambda:hw.t)
    assert o['automatic_L2'] is True and o['audit'] is False and o['label_conflict'] and o['L2'] is True


def test_new_entry_preserves_reference_baseline_without_calibration(tmp_path):
    legacy,_=campaign_ah.configuration(tmp_path,center=pose())
    current,cfg=campaign_ah.configuration(tmp_path,center=pose(),version=5)
    assert current['task']['action_space']['id']==g.RH_KIND and current['runtime']['max_steps']==3
    assert current['observation']['tac_version']=='legacy_pixel'
    assert current['observation']['tac_version_requested']=='calibrated_rh'
    assert current==ah_v5_config.configure(deepcopy(current))
    assert legacy['runtime']['max_steps']==1 and 'consumer' not in legacy


def test_representation_digest_and_legal_runtime_target_switch(tmp_path,monkeypatch):
    from bottle_scan.state_graph.common import digest
    spec=instance(tmp_path)
    persisted=tmp_path/'source.json'
    campaign.atomic(persisted,spec)
    effective=deepcopy(spec)
    effective['observation']['modalities'].pop('tac')
    effective['observation']['representations']['current']='force'
    fitted=tmp_path/'fitted.json'
    campaign.atomic(fitted,effective)
    target='fixed_force_lift'
    bundle=dict(model_id='m',spec=effective,support=[],graph={'radii':{'similar_radius':.4}},noise={})
    seen=[]
    class Frozen:
        def __init__(self,b):assert b['spec']['active_post_condition']==target
        def infer(self,row):
            seen.append(set(row['obs']))
            return {'state':'ready'}
    monkeypatch.setattr(p3_trial_v5,'measurement_distance',lambda a,b,n:0.)
    monkeypatch.setattr(p3_trial_v5.joblib,'load',lambda _:bundle)
    monkeypatch.setattr(p3_trial_v5,'SymbolicRecoverySkill',Frozen)
    req=request('policy_frontier')
    req.update(mode='learned',model_bundle='synthetic',model_id='m',target_consumer=target,
               post_condition=target,spec=str(persisted),base_spec_digest=digest(spec),
               effective_spec=str(fitted),effective_spec_digest=digest(effective))
    result=p3_trial.run_trial(req,tmp_path,{**spec,'active_post_condition':target},{},Fake(spec),sources=[])
    assert seen==[{'force'}]
    assert set(result['nodes'][0]['obs'])=={'tac','force'}
    with pytest.raises(ValueError):p3_trial_v5.frozen_recipe(bundle,{**req,'effective_spec_digest':'wrong'},spec)
