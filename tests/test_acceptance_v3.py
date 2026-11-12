from copy import deepcopy
from bottle_scan.state_graph.acceptance import evaluate_B
from bottle_scan.state_graph.skill_interface import load_spec


def test_acceptance_inherits_counts_and_post_freeze_independent_requirement():
    spec=load_spec('skills/contact_position_a/spec.json');condition=spec['active_post_condition'];records=[]
    for i in range(12):
        ready=i<6;target=i<9
        records.append({'group_id':str(i),'role':'development','post_condition':condition,'evaluated':True,'initial_truth':'Ready' if ready else 'NeedAction',
            'invoked':not ready,'target_truth':target,'exit_declared':target,'success':True,'rejected':False})
    assert evaluate_B(records,spec,'development')['passed']
    for r in records:r['role']='independent'
    assert not evaluate_B(records,spec,'independent')['passed']
    for r in records:r.update(post_freeze=True,frozen_model_id='frozen_model')
    assert evaluate_B(records,spec,'independent')['passed']


def test_assessed_b_results_feed_director_freeze_and_complete(tmp_path):
    from tests.fixtures.skill_toy.testing import data
    from bottle_scan.state_graph.cli import build,initial_rule
    from bottle_scan.state_graph.operators import applicable_operators
    rows,spec,_=data();spec=deepcopy(spec)
    spec['acceptance']={'development_B':{'min_trials':2},'independent_B':{'min_trials':2}}
    cid=spec['active_post_condition'];assessments=[]
    for role in ('development','independent'):
        assessments += [{'group_id':role+str(i),'role':role,'post_condition':cid,'evaluated':True,
                         'post_freeze':True,'frozen_model_id':'frozen'} for i in range(2)]
    state=build(rows,spec,tmp_path,validate=False,assessments=assessments)
    facts=state['facts'];rule=initial_rule(rows,spec)
    assert 'freeze' in {o['op'] for o in applicable_operators(facts,spec,rule)}
    assert 'complete' in {o['op'] for o in applicable_operators(facts,spec,rule,{'frozen_independent':True})}
