import numpy as np
from bottle_scan.state_graph.v5_metric import fit_metric
from bottle_scan.state_graph.v5_profile import configure


def test_axis_without_signal_is_excluded():
    spec=configure({'perturbation_variables':{}});spec['observation']['metric']['a_axis_tcp']=[1,0,0]
    rows=[]
    rng=np.random.default_rng(712)
    for i in range(60):
        r,h=rng.choice([-4,4],2)
        for j in (0,1):
            pose=np.eye(4);pose[:3,3]=[j*r,0,j*h]
            rows.append(dict(group_id=str(i),session='s',task_id='t',node_id=f'{i}/{j}',window_index=j,
                repeat_key='start',window_valid=True,previous_node_id=f'{i}/0' if j else None,
                u_receipt={'applied':True} if j else None,obs={'tac_embed':[float(j*r)]},provenance={'measured_pose':pose.tolist()}))
    head=fit_metric(rows,spec)
    assert head['axes']==['r'] and np.shape(head['coefficients'])==(1,1)
    assert head['axis_reports']['h']['heldout_error_mm']>1.5
