import pytest
from bottle_scan.state_graph.labels import generate_labels


def test_multistep_actions_preserve_order_and_supervise_first_action():
    edges=[{'edge_id':'1','source':'x','target':'y','kind':'observed','cost':1.,'u':[1.,2.]},
           {'edge_id':'2','source':'y','target':'z','kind':'inferred','cost':1.,'u':[-.5,1.]},
           {'edge_id':'3','source':'z','target':'__post__:done','kind':'sink','cost':.1}]
    rows=[{'node_id':i} for i in ['x','y','z']]
    graph={'edges':edges,'R':{'done':{'x':2.1,'y':1.1,'z':.1}},'probabilities':{'done':{i:{'p_hat':.9} for i in ['x','y','z']}},'reachability':{'done':{i:{'status':'ready' if i=='z' else 'recoverable'} for i in ['x','y','z']}}}
    spec={'post_conditions':[{'id':'done'}],'task':{'action_space':{'dim':2}}}
    labels=generate_labels(graph,rows,spec);assert labels['nodes'][0]['action_target']==[1.,2.]
    assert [a['u'] for a in labels['nodes'][0]['action_sequence']]==[[1.,2.],[-.5,1.]]
    assert labels['transitions'][0]['advantage']==pytest.approx(0)
