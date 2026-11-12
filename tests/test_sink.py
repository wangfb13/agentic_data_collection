from copy import deepcopy
import numpy as np
import pytest
from bottle_scan.state_graph.sink import estimate_sinks,sink_id
from tests.fixtures.skill_toy.testing import data


def test_beta_prior_unassessed_condition_and_pmin():
    rows,spec,_=data();rows=deepcopy(rows[:3])
    for r in rows:
        r['is_terminal']=True;r['post']['centered'].update(evaluated=True,success=True)
        r['post']['in_region'].update(evaluated=False,success=None)
    p,edges=estimate_sinks(rows,np.zeros((3,3)),spec)
    assert p['centered'][rows[0]['node_id']]['p_hat']==pytest.approx(.8)
    assert all(e['post_condition']=='centered' for e in edges)
    single=rows[:1];changed=deepcopy(spec);changed['post_conditions'][0]['p_min']=.8
    _,edges=estimate_sinks(single,np.zeros((1,1)),changed);assert edges==[]


def test_reverse_values_keep_post_conditions_separate():
    from bottle_scan.state_graph.graph import reverse_costs
    edges=[{'source':'x','target':sink_id('c1'),'cost':.2},{'source':'y','target':sink_id('c2'),'cost':1.3}]
    assert reverse_costs(['x','y'],edges,sink_id('c1'))=={'x':.2,'y':None}
