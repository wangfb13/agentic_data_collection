from bottle_scan.state_graph.frontier_rates import statistics,overlap
from bottle_scan.state_graph.v5_profile import configure


def test_random_far_success_and_empty_gap():
    cfg=configure({'perturbation_variables':{}})['facts']
    data=[(str(i),.5,True,None) for i in range(60)]+[('island',3.5,True,None)]
    a=statistics(data,1,1,5,cfg)
    assert a['verified_edge_mm']==1 and not a['neighbors_confirmed']
    single=statistics([('only',.5,True,None)],1,1,5,cfg)
    assert single['verified_edge_mm'] is None and not overlap(single,single)


def test_stop_uses_cumulative_unique_nearby_attempts():
    from bottle_scan.state_graph.rounds import stop_reason
    spec={'budget':{'max_new_trials':60},'active_post_condition':'lift','perturbation_variables':{'r':{'envelope':[0,5]}},
          'facts':{'rate_based_frontier':True,'frontier_min_trials':5,'physical_failure_fraction':.9}}
    state={'rule':{'variables':{'r':{'active':True,'range':[0,5]}}},'spent':{'trials':5},'independent_reserve':10}
    metrics={'condition':{},'adjacent_failures':{'1':'contact_lost'}}
    facts=[dict(pred='policy_frontier',args=['r','positive','lift'],metrics=metrics)]
    assert stop_reason(state,spec,facts,[],False,2) is None
    metrics['adjacent_failures'].update({str(i):'contact_lost' for i in range(5)})
    assert stop_reason(state,spec,facts,[],False,2)=='remaining_observed_failures_physical'
    metrics['adjacent_failures']['earlier']='model_rejected'
    assert stop_reason(state,spec,facts,[],False,2) is None
