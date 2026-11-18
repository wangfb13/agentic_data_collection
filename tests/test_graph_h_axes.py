import numpy as np
from bottle_scan.state_graph.axis_resolution import axis_observable
from bottle_scan.state_graph.v5_values import resolution
from bottle_scan.state_graph.v5_profile import configure


def test_noisy_h_does_not_shrink_r():
    s=configure({'perturbation_variables':{},'task':{'action_space':{'names':['r','h']}}})
    rows=[dict(node_id=str(i),window_index=0,repeat_key='same' if i<2 else str(i),group_id=str(i)) for i in range(8)]
    D=np.full((8,8),0.2)
    edges=[]
    for i in range(2,8):
        axis=int(i>=5); u=[0.03,0.03];u[axis]=3.
        D[0,i]=1.1 if axis==0 else .19
        edges.append(dict(source='0',target=str(i),u=u))
    assert axis_observable(rows,edges,D,s,1)['observable'] is False
    a,r=resolution(rows,edges,D,s)
    assert a['by_axis']['r']['used'] and not a['by_axis']['h']['used']
    assert r==resolution(rows,edges[:3],D,s)[1]
