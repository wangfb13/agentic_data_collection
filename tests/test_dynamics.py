from copy import deepcopy
import numpy as np
from bottle_scan.state_graph.dynamics import predict,inferred_cost,inferred_edges,prediction_distance
from tests.fixtures.skill_toy.testing import trained


def test_dynamics_multidim_inputs_exclude_metadata_and_outcomes():
    rows,spec,noise,f,g,*_=trained();row=deepcopy(rows[0]);a=predict(f,row,[.1,.2])
    row.update(plan_coords={'x':1e8},attrs={'payload_class':'secret'},post={'leaked':{'success':True}},R=-1e9)
    b=predict(f,row,[.1,.2]);np.testing.assert_array_equal(a['mean'],b['mean'])
    assert len(a['mean'])==4 and len(a['sigma'])==4
    assert len(f['p_off_oof_labels'])==f['n']


def test_uncertainty_cost_and_augmented_covariance():
    rows,spec,noise,f,*_=trained()
    assert inferred_cost(.3,.1,spec)>inferred_cost(.1,.1,spec)
    delta=np.ones(4)*.1
    assert prediction_distance(delta,f['metric'],np.ones(4))<prediction_distance(delta,f['metric'])


def test_observed_node_pair_has_no_inferred_duplicate():
    _,_,_,_,g,*_=trained();observed={(e['source'],e['target']) for e in g['edges'] if e['kind']=='observed'}
    inferred=[e for e in g['edges'] if e['kind']=='inferred'];assert inferred
    assert not observed & {(e['source'],e['target']) for e in inferred}
    assert len(inferred)==len({(e['source'],e['target']) for e in inferred})
