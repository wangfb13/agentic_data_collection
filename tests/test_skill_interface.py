from pathlib import Path
import numpy as np
from bottle_scan.state_graph.facts import generate_facts
from bottle_scan.state_graph.operators import applicable_operators
from bottle_scan.state_graph.cli import initial_rule
from tests.fixtures.skill_toy.testing import trained


def test_second_skill_complete_pipeline_and_no_instance_literals():
    rows,spec,noise,f,g,labels,clusters,bundle,status=trained()
    assert spec['task']['action_space']['dim']==2 and len(g['R'])==2
    assert f['status']=='trained' and status['active_model_trained']
    facts=generate_facts(g,rows,labels,clusters,noise,spec,bundle)
    ops=applicable_operators(facts,spec,initial_rule(rows,spec))
    assert facts and any(o['op']=='train' for o in ops)
    for p in Path('bottle_scan/state_graph').rglob('*.py'):
        for token in ['tcp_offset_a','white_','forcepolicy','NeedShift','HandoffReady','kinematic']:
            assert token not in p.read_text(),(p,token)


def test_independent_b_never_defines_training_graph():
    from copy import deepcopy
    from bottle_scan.state_graph.graph import build_graph
    import pytest
    rows,spec,noise,*_=trained();one=deepcopy(rows[0]);one['frozen_independent']=True
    with pytest.raises(ValueError,match='Independent'):build_graph([one],noise,spec)
