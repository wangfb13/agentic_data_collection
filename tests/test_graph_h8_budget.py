from copy import deepcopy

import pytest

from bottle_scan.state_graph import campaign
from bottle_scan.state_graph.rounds import initialize_rounds, exploration_allowance
from tests.test_state_graph_v5_rounds import setup_round, synthetic_executor


def test_budget_guarantees_two_rounds_and_reports_split(tmp_path):
    out,spec=setup_round(tmp_path)
    spec['loop'].update(budget_check=True,min_rounds=2,exploration_per_round=4)
    state=campaign.read(out/'state.json')
    cfg=campaign.read(out/'config.json')
    spec['budget']['max_new_trials']=13
    with pytest.raises(ValueError,match='minimum=14'):
        initialize_rounds(deepcopy(state),spec,cfg)
    spec['budget']['max_new_trials']=14
    initialize_rounds(state,spec,cfg)
    assert exploration_allowance(12,2,spec,1)==4
    campaign.atomic(out/'spec.json',spec)
    campaign.atomic(out/'state.json',state)
    done=campaign.run(out,executor=synthetic_executor,max_decisions=150)
    assert done['round_index']>=2
    budget=campaign.read(out/'rounds/round_0001/round_report.json')['remaining_budget']
    assert budget['total']==budget['independent_reserved']+budget['for_future_rounds']
    assert budget['independent_reserved']==2
