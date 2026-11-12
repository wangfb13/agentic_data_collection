from copy import deepcopy
import pytest
from bottle_scan.state_graph.graph import gate,build_graph
from bottle_scan.state_graph.noise import measurement_distance
from tests.fixtures.skill_toy.testing import data


def test_only_task_and_window_gate():
    rows,spec,noise=data();a=deepcopy(rows[0]);b=deepcopy(a)
    b['attrs'].update(protocol_id='changed',time_period='changed',payload_class='different',sensor_version='different')
    assert gate(a,b) and measurement_distance(a,b,noise)==pytest.approx(0)
    b['task_id']='another_task';assert not gate(a,b)
    b['task_id']=a['task_id'];b['window_valid']=False;assert not gate(a,b)


def test_mixed_versions_and_tasks_rejected_at_graph_level():
    rows,spec,noise=data();a,b=deepcopy(rows[:2]);b['extractor_versions']['geometry']='new'
    with pytest.raises(ValueError,match='version'):build_graph([a,b],noise,spec)
    b=deepcopy(rows[1]);b['task_id']='another'
    with pytest.raises(ValueError,match='One task'):build_graph([a,b],noise,spec)


def test_repeat_covariance_corrects_estimated_group_mean_then_doubles():
    import numpy as np
    from bottle_scan.state_graph.noise import fit_noise
    from bottle_scan.state_graph.dynamics import target_metric
    rows,spec,_=data();spec=deepcopy(spec)
    spec['observation']['modalities']={'geometry':{'version':'test'}}
    small=[]
    for i,value in enumerate((0.,2.,10.,12.)):
        row=deepcopy(rows[0]);row.update(node_id=str(i),group_id=str(i),window_index=0,repeat_key=str(i//2),obs={'geometry':[value]})
        small.append(row)
    noise=fit_noise(small,spec);single=noise['modalities']['geometry']['single_covariance'][0][0]
    assert single==pytest.approx(2.,rel=1e-6)
    assert noise['modalities']['geometry']['covariance'][0][0]==pytest.approx(4.,rel=1e-6)
    metric=target_metric(small,{'geometry':[0]},noise)
    assert metric['geometry']['covariance'][0,0]==pytest.approx(4.,rel=1e-6)
