import pytest
from skills.contact_position_a import reference_check as check


@pytest.mark.parametrize('noise,distance,expected',[(True,.4,True),(True,.8,False),(False,0,False)])
def test_reference_reuse_checks_frozen_metric(monkeypatch,noise,distance,expected):
    import numpy as np
    monkeypatch.setattr(check,'measurement_distance',lambda *args:distance)
    r=dict(node_id='contact',provenance={'measured_pose':np.eye(4)},extractor_versions={})
    b={'graph':{'radii':{'similar_radius':.5}},'noise':{'active_modalities':['tac'] if noise else []}}
    assert check.evaluate(r,r,b,dict(verification_pose_tolerance_mm=1,orientation_tolerance_deg=1))['passed'] is expected
