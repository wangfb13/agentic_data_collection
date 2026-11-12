from copy import deepcopy
import numpy as np
import pytest
from scipy.stats import chi2
from skills.contact_position_a.legacy_v2.noise import fit_noise, threshold, noise_distance
from skills.contact_position_a.legacy_v2.graph import build_graph,gate,pairwise_distances
from skills.contact_position_a.legacy_v2.provenance import prepare_nodes,same_reference
from skills.contact_position_a.legacy_v2.facts import generate_facts
from skills.contact_position_a.legacy_v2.operators import applicable_operators
from skills.contact_position_a.legacy_v2.boundary import fit_boundary


def observations(n=12):
    rows=[]
    for i in range(n):
        pos={'u_px':30+i*.1,'v_px':20+i*.01,'area_px':100+i,'u_mm':None,'v_mm':None,'area_mm2':None}
        rows.append(dict(node_id=str(i),session='s',trial_id=str(i),group_id=str(i),object_id='green_light',protocol_id='p',phase='A',window_index=0,
            bilateral_contact=True,gripper_state='closing',force_bin='probe',encoder_version='old',edge_class={'left':None,'right':None},
            sensor_contact_left=pos,sensor_contact_right=deepcopy(pos),wrench_left=[1+i*.01,0,1,0,0,0],wrench_right=[1,0,1+i*.01,0,0,0],
            gripper_opening=.4+i*.0001,planned_delta_mm=i//3,time_period='era',acquisition_day='2026-11-10',
            T0_mm=np.eye(4).tolist(),a_axis_base=[1,0,0],tcp_offset_a=float(i),downstream_success=None,
            trial_started_unix_ns=1794278327841243368,protocol={'closure_floor':.336}))
    return rows


def test_px_distance_and_correct_covariance_bonferroni():
    rs=observations();nm=fit_noise(rs)
    assert nm['modalities']['pos']['dimension']==6
    assert np.isfinite(noise_distance(rs[0],rs[1],nm))
    for name in ['pos','force','open']:
        m=nm['modalities'][name]
        np.testing.assert_allclose(m['covariance'],2*np.asarray(m['single_observation_covariance']))
        assert m['tau']==pytest.approx(np.sqrt(chi2.ppf(1-.05/3,m['df'])))
        assert threshold(nm,name,['pos','force','open','tac','vis'])==pytest.approx(np.sqrt(chi2.ppf(1-.05/5,m['df'])))
    dm=pairwise_distances(rs,nm,{'modalities':['pos','force','open']})
    assert dm[0,1]==pytest.approx(noise_distance(rs[0],rs[1],nm),rel=1e-5)


def test_historical_adoption_does_not_unseal_new_independent_data():
    r=observations(3)
    for x in r:x.update(evaluation_only=True,source_role='formal_validation')
    r[1]['frozen_independent']=True
    out=prepare_nodes(r,{'historical_node_ids':['0','1']})
    assert not out[0]['evaluation_only'] and out[0]['original_evaluation_only']
    assert out[0]['source_role']=='historical_development'
    assert out[1]['evaluation_only'] and out[2]['evaluation_only']
    assert '0' in fit_noise(out)['fit_node_ids'] and '1' not in fit_noise(out)['fit_node_ids']


def test_cross_session_kinematic_requires_real_same_reference():
    a,b=observations(2);a['downstream_success']=False;b['downstream_success']=True;b['session']='other'
    assert same_reference(a,b)
    g=build_graph([a,b],fit_noise(observations()),{'modalities':['pos','force','open'],'knn':0})
    assert g['counts']['cross_session_kinematic']==2
    b['T0_mm'][0][3]=2
    assert not same_reference(a,b)
    b['T0_mm']=np.eye(4).tolist();b['acquisition_day']='2026-11-11'
    assert not same_reference(a,b)


def test_time_controller_gating_and_optional_encoder():
    a,b=observations(2);cfg={'modalities':['pos','force','open'],'require_time_period':True}
    b['encoder_version']='another'
    assert gate(a,b,cfg)
    b['time_period']='new_cover'
    assert not gate(a,b,cfg)
    b['time_period']=a['time_period'];b['downstream_controller']='forcepolicy'
    prepared=prepare_nodes([a,b])
    assert prepared[0]['protocol_id']!=prepared[1]['protocol_id']
    assert not gate(*prepared,cfg)


def test_no_edges_is_unsupported_not_reason_to_add_axis():
    r=observations(4)
    for i,x in enumerate(r):x.update(session=str(i),tcp_offset_a=100*i)
    nm=fit_noise(observations());g=build_graph(r,nm,{'knn':0,'modalities':['pos','force','open']})
    f=generate_facts(g,r,[],{'clusters':[]})
    assert sum(x['pred']=='unsupported' for x in f)==4
    spec={'envelope':{'allowed_axes':['a','l']},'recordable_attributes':[]}
    assert not any(x['op']=='add_axis' for x in applicable_operators(f,spec,{'active_axes':['a']}))


def test_add_axis_requires_failed_executed_corrections_from_three_groups():
    r=observations(6)
    for k in range(3):
        a,b=r[2*k:2*k+2];a.update(trial_id=str(k),group_id=str(k),phase='B_pre')
        b.update(trial_id=str(k),group_id=str(k),phase='B_post',window_index=1,previous_node_id=a['node_id'],
            executed_delta_a_mm=1.,execution_receipt={'executed':True},downstream_success=False)
    g=build_graph(r,fit_noise(observations()),{'knn':0,'modalities':['pos','force','open']})
    f=generate_facts(g,r,[],{'clusters':[]});spec={'envelope':{'allowed_axes':['a','l']},'recordable_attributes':[]}
    assert any(x['op']=='add_axis' for x in applicable_operators(f,spec,{'active_axes':['a']}))
    for x in f:
        if x['pred']=='unreachable':x['metrics']['trial_groups']=['same_trial']
    assert not any(x['op']=='add_axis' for x in applicable_operators(f,spec,{'active_axes':['a']}))


def test_all_success_has_no_boundary_and_known_boundary_is_recovered():
    assert fit_boundary([-5,0,5],[1,1,1])['boundary_mm'] is None
    x=np.linspace(-5,8,100);y=(x>2).astype(int)
    result=fit_boundary(x,y,bootstrap=10)
    assert result['boundary_mm']==pytest.approx(2,abs=.2)


def test_explicit_hardware_period_and_forcepolicy_model_identity_persist():
    a,b=observations(2)
    for r in (a,b):r.update(sensor_version='x5_cover_2',time_period='operator_period',downstream_controller='forcepolicy')
    a['downstream_model_id']='model_a';b['downstream_model_id']='model_b'
    a,b=prepare_nodes([a,b])
    assert a['sensor_version']=='x5_cover_2' and a['time_period']=='operator_period'
    assert a['protocol_id']!=b['protocol_id']


def test_ignored_gates_do_not_refute_unknown_arrival():
    from skills.contact_position_a.legacy_v2.graph import _known_gate_mismatches
    a,b=observations(2);b.update(time_period='other',encoder_version='other')
    assert _known_gate_mismatches(a,b,{'ignore_time_period':True,'modalities':['pos','force','open']})==[]


def test_report_counters_are_plain_json_serializable():
    import json
    from skills.contact_position_a.legacy_v2.report import score_ablation
    rows=observations()
    result=score_ablation(rows,['pos','force','open'],fit_noise(rows),folds=3)
    json.dumps(result,allow_nan=False)
    assert type(result['V1']['within_noise_pairs']) is int
