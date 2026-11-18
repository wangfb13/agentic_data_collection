import numpy as np
from bottle_scan.state_graph.ready_kernel import estimate
from bottle_scan.state_graph.v5_profile import configure


def test_grouped_kernel_separates_and_ignores_heldout_labels():
    rows=[dict(group_id=str(i),node_id=str(i),is_terminal=True,post={'c':dict(evaluated=True,success=i<12)}) for i in range(24)]
    x=np.repeat([0.,10.],12)
    D=np.abs(x[:,None]-x)
    s=configure({'perturbation_variables':{}},revision_h811=False)['sinks']
    p,m=estimate(rows,D,{'id':'c'},s)
    assert all(v['ready']==(i<12) for i,v in enumerate(p))
    assert m['threshold'] is not None
    assert next(v for v in m['candidates'] if v['threshold']==m['threshold'])['false_release_rate']==0
    _,m2=estimate(rows[:12],D[:,:12],{'id':'c'},s)
    assert m2['threshold'] is None
