import pytest
from bottle_scan.state_graph.calibrations import preflight


def test_real_manual_offline_and_override():
    spec={'consumer':{'auto_l2':True},'observation':{'tac_version':'legacy_pixel','tac_version_requested':'calibrated_rh'}}
    with pytest.raises(ValueError,match='weight_newtons.*load_calibration.*tac_calibration'):preflight(spec,True,False)
    assert len(preflight(spec,False,False)['missing'])==3
    assert preflight(spec,True,True)['missing']==preflight(spec,False,False)['missing']
    assert not preflight({'consumer':{'auto_l2':False},'observation':{'tac_version':'legacy_pixel'}},True,False)['missing']
