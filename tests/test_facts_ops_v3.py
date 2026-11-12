from copy import deepcopy
import numpy as np
import pytest
from bottle_scan.state_graph.noise import fit_noise
from bottle_scan.state_graph.graph import build_graph
from bottle_scan.state_graph.labels import generate_labels
from bottle_scan.state_graph.clusters import cluster_nodes
from bottle_scan.state_graph.facts import generate_facts
from bottle_scan.state_graph.operators import applicable_operators,validate_decision
from bottle_scan.state_graph.cli import initial_rule
from bottle_scan.state_graph.director_bridge import context,matches
from tests.fixtures.skill_toy.testing import data


def make_case(correlated=True):
    originals,spec,_=data();rows=[]
    for i in range(24):
        r=deepcopy(originals[0]);r.update(node_id=str(i),group_id=str(i),trial_id=str(i),is_terminal=True,repeat_key='same')
        r['attrs'].update(payload_class='soft' if i<12 else 'hard',controller_setting='standard' if i<12 else 'alternative')
        success=i<12 if correlated else i%2==0
        r['post']['centered'].update(evaluated=True,success=success);r['post']['in_region'].update(evaluated=False,success=None);rows.append(r)
    noise=fit_noise(rows,spec);graph=build_graph(rows,noise,spec);labels=generate_labels(graph,rows,spec);clusters=cluster_nodes(rows,graph,noise,labels,spec)
    facts=generate_facts(graph,rows,labels,clusters,noise,spec)
    return rows,spec,facts


def test_fisher_significance_routes_variable_types():
    rows,spec,facts=make_case();ops=applicable_operators(facts,spec,initial_rule(rows,spec));names={o['op'] for o in ops}
    assert {'probe_hidden_variable','block_by_property','request_capability'}<=names
    assert 'split_by_object' not in names and 'add_axis' not in names
    assert any(f['pred']=='stochastic_outcome' for f in facts)
    rows,spec,facts=make_case(False)
    assert not any(f['pred']=='hidden_variable_candidate' for f in facts)
    assert any(o['op']=='repeat_state' for o in applicable_operators(facts,spec,initial_rule(rows,spec)))


def test_hypothesis_partial_argument_and_metric_interval():
    f={'pred':'boundary_gap','args':['x',[0,2],'done'],'metrics':{'gap_mm':2.},'value':True}
    assert matches({'pred':'boundary_gap','args_match':{'0':'x'},'metrics':{'gap_mm':[0,2.5]}},f)
    assert not matches({'pred':'boundary_gap','args_match':{'2':'other'}},f)


def test_request_capability_has_no_collection_and_unknown_operator_rejected():
    rows,spec,facts=make_case();ctx=context(spec,initial_rule(rows,spec),facts)
    op=next(o for o in ctx['applicable_operators'] if o['op']=='request_capability')
    answer={'operator':op['op'],'args':{**op['bindings'],'description':'Need an observable response to distinguish the hidden property.'},'premises':op['premises_satisfied_by'],'plan':None,'hypotheses_update':[],'discriminates':[],'expected':'独立测量区分属性','reason':'未观察到共同就绪区域'}
    assert validate_decision(answer,ctx)==answer
    answer['plan']={'cells':[]}
    with pytest.raises(ValueError):validate_decision(answer,ctx)
