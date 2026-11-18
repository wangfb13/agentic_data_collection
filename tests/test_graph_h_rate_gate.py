import pytest
from bottle_scan.state_graph.rate_gates import check_plan
from bottle_scan.state_graph.acceptance import evaluate_B


def test_ten_of_twelve_cannot_pass_point_eight():
    gate={'lower_bound_gates':True,'rate_confidence':.95,'primary_rates':['recovery_success_rate']}
    for role in ['development','independent']:gate[role+'_B']={'min_recovery_success_rate':.8}
    spec={'active_post_condition':'lift','acceptance':gate}
    rows=[dict(role='development',post_condition='lift',group_id=i,initial_truth='NeedAction',success=i<10) for i in range(12)]
    r=evaluate_B(rows,spec,'development')
    assert not r['passed'] and r['metrics']['development_point_pass_only']
    with pytest.raises(ValueError,match='16'):check_plan({'planned_denominators':{'NeedAction':12}},12,gate)
    check_plan({'planned_denominators':{'NeedAction':16}},16,gate)
