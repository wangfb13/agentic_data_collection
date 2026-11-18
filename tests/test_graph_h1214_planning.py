from copy import deepcopy
from contextlib import contextmanager

import pytest
from bottle_scan.state_graph import campaign
from bottle_scan.state_graph.amplitude_coverage import coverage_facts, terminal_samples
from bottle_scan.state_graph.frontier_facts import episode_frontiers
from bottle_scan.state_graph.director_bridge import context, decide
from bottle_scan.state_graph.operators import validate_decision
from bottle_scan.state_graph.v5_director import fixed_decision
from skills.contact_position_a import p3_trial
from tests.test_v5_geometry_consumers import instance, window, Fake, request

H='h_offset_mm'
R='r_offset_mm'
CID='forcepolicy_lift'


@pytest.fixture
def spec(tmp_path):
    s=instance(tmp_path)
    s['facts'].update(amplitude_coverage=True,min_bin_outcomes=1,terminal_based_direct_boundary=True,
                      boundary_resolution_mm=1,bootstrap_reps=20)
    s['director'].update(score_based_axis_choice=True,mode='fixed_priority')
    s['perturbation_variables'][H].update(envelope=[-3,16],nominal_step=1.)
    s['perturbation_variables'][R].update(envelope=[-4,4],nominal_step=1.)
    return s


def sample(i,h,ok,r=0):
    n=window(1,[r,h])
    n.update(trial_id='trial'+str(i),group_id='g'+str(i),node_id='n'+str(i),is_terminal=True,
             u_receipt={'applied':True},post={CID:{'evaluated':True,'success':ok}})
    n['provenance']['coordinate_schema']='measured_placement_relative'
    return n


def measured(rows,spec):
    g={'probabilities':{CID:{n['node_id']:{'p_lower':(i+1)/(len(rows)+1)} for i,n in enumerate(rows)}}}
    return coverage_facts(rows,g,spec,CID)


def gap(facts,var=H,side='positive'):
    return next(f for f in facts if f['pred']=='amplitude_gap' and f['args'][:2]==[var,side])


def test_outward_gap_doubling_interior_hole_and_invalid_labels(spec):
    rows=[sample(i,i,True) for i in range(1,5)]
    bad=sample('bad',6,True);bad['provenance']['placement_reference_valid']=False
    sealed=sample('sealed',7,True);sealed['data_role']='independent_validation'
    skip=sample('no_consumer',5,True);skip['post'][CID]['evaluated']=False
    f=measured(rows+[bad,sealed,skip],spec)
    assert gap(f)['args'][2]==[4,16] and gap(f)['metrics']['target_mm']==8
    assert gap(f,H,'negative')['metrics']['target_mm']<0
    hole=gap(measured(rows+[sample(8,8,False)],spec))
    assert hole['args'][2]==[4,7] and hole['metrics']['target_mm']==5.5
    spec['facts']['min_bin_outcomes']=2
    assert gap(measured(rows+rows,spec))['metrics']['group_counts'][:4]==[1]*4
    spec['perturbation_variables'][H]['nominal_step']=2.5
    assert gap(measured(rows,spec))['metrics']['bin_width_mm']==2.5


def test_terminal_position_in_h4_does_not_replace_recovery_start(spec):
    rows=[sample(1,6,True),sample(2,6,False),sample(3,1,True)]
    rows[2].update(u_receipt=None,controller_role='direct')
    starts=[]
    for i,n in enumerate(rows):
        first=deepcopy(n); first.update(window_index=0,plan_coords={H:1,R:0})
        starts.append(dict(trial_id='t'+str(i),condition={'mass_class':'light'},model_id='m',
            first_contact=first,controller_role='policy' if i<2 else 'direct',terminal_outcomes=n['post']))
    facts=episode_frontiers(starts,spec,CID,rows=rows,model_id='m')
    get=lambda pred:next(f['metrics'] for f in facts if f['pred']==pred and f['args'][:2]==[H,'positive'])
    direct=get('direct_boundary')
    assert direct['terminal_sources']==dict(after_correction=2,without_correction=1,unknown=0)
    assert (direct['bins'][5]['evaluated'],direct['bins'][5]['successes'])==(2,1)
    assert get('policy_frontier')['bins'][0]['evaluated']==2
    assert not get('policy_frontier')['bins'][5]['evaluated']
    spec['facts']['terminal_based_direct_boundary']=False
    assert next(f['metrics'] for f in episode_frontiers(starts,spec,CID,rows=rows) if f['pred']=='direct_boundary'
                and f['args'][:2]==[H,'positive'])['evaluated_trials']==1


def options(spec,rows):
    facts=measured(rows,spec)
    from bottle_scan.state_graph.facts import fact
    facts += [fact('development_class_missing',[CID,k],[],{'missing_classes':[k]}) for k in ('Ready','NeedAction')]
    return context(spec,{'rule_id':'test_rule','variables':{v:{'active':True,'range':d['envelope']} for v,d in
                                spec['perturbation_variables'].items() if d.get('envelope')}},
                   facts,budget_left=10)


def test_axis_ties_missing_classes_and_validated_schedule(spec,tmp_path):
    rows=[sample(i,i,True) for i in range(1,5)]
    ctx=options(spec,rows)
    candidates=[e for e in ctx['operator_scores'] if e.get('proposal')]
    probes=[e for e in candidates if e['operator']=='probe_amplitude']
    assert len(probes)==4
    # Scores equal when the graph has no posterior/gap variance information.
    decision=fixed_decision(ctx)
    assert decision['args']['var']==H and decision['args']['target_mm']==8
    outcomes=[e for e in candidates if e['operator']=='collect_outcomes']
    assert any(e['bindings'].get('missing_class')=='NeedAction' and
               e['proposal']['plan']['cells'][0]['settings'].get(H)==8 for e in outcomes)
    assert any(e['bindings'].get('missing_class')=='Ready' for e in outcomes)
    assert decision['args']['missing_class']=='NeedAction'
    assert decision['selection_audit']['tie_breaks'][0]['criterion']=='gap_width_mm'
    assert all(e['bindings'].get('target_point')=={R:0,H:4} for e in candidates if e['bindings'].get('missing_class')=='Ready')
    probe=next(e['proposal'] for e in probes if e['bindings']['var']==H and e['bindings']['side']=='positive')
    items=campaign.expand_plan(probe['plan'],spec)
    assert [t['amplitude_target']['mm'] for t in items]==[8,8]
    assert {t['block']['mass_class'] for t in items}=={'light','heavy'}
    bad=deepcopy(probe);bad['plan']['cells'][0]['actions'][0]['u'][1]=11
    with pytest.raises(ValueError):validate_decision(bad,ctx)
    # Reverse offers; scoring and tie resolution must be invariant.
    ctx['operator_scores'].reverse()
    assert fixed_decision(ctx)['args']==decision['args']
    official=decide(ctx,tmp_path/'director')['answer']
    assert official['selection_audit']['candidates'] and official['args']['var']==H


def test_tie_uses_previous_round_then_prior_and_score_beats_both(spec):
    for d in (R,H): spec['perturbation_variables'][d]['envelope']=[0,8]
    ctx=options(spec,[sample(1,1,True,1)])
    ctx['operator_scores']=[e for e in ctx['operator_scores'] if e['operator']=='probe_amplitude']
    assert fixed_decision(ctx)['selection_audit']['prior_decisive']
    prev=fixed_decision(ctx)
    assert prev['args']['var']==H
    ctx['history']=[{'decision':prev}]
    assert fixed_decision(ctx)['args']['var']==R
    next(e for e in ctx['operator_scores'] if e['bindings']['var']==H)['score']=10.
    assert fixed_decision(ctx)['args']['var']==H
    ctx['spec']['director']['use_priors']=False;ctx['history']=[]
    assert not fixed_decision(ctx)['selection_audit']['prior_decisive']


def test_fake_hardware_uses_measured_start_and_delivers_consumer(spec,tmp_path):
    class MeasuredFake(Fake):
        @contextmanager
        def contact(self,index):
            with super().contact(index) as n:
                n['provenance']['coordinate_schema']='measured_placement_relative'
                yield n
    io=MeasuredFake(spec,start=(0.,2.))
    ctx=options(spec,[sample(i,i,True) for i in range(1,5)])
    cell=next(e['proposal']['plan']['cells'][0] for e in ctx['operator_scores'] if e['operator']=='probe_amplitude'
              and e['bindings']['var']==H and e['bindings']['side']=='positive')
    req=request('exploration',cell['actions'],amplitude_target=cell['amplitude_target'],operation='probe_amplitude')
    finished=p3_trial.run_trial(req,tmp_path,spec,{},io)
    assert finished['action_commands'][0]['u']==[0.,6.] and io.pos[1]==8.
    assert finished['terminal_outcomes'][CID]['success'] and len(io.seen)==2
    failed=p3_trial.run_trial(req,tmp_path,spec,{},Fake(spec))
    assert failed['attempt_reason']=='amplitude_start_coordinate_missing' and not failed['action_commands']



def test_all_directions_enumerated_before_scoring(spec):
    from bottle_scan.state_graph.facts import fact
    f=fact('transition_evidence_missing',[CID],[],{})
    ctx=context(spec,{'rule_id':'r','variables':{}},[f],budget_left=10)
    directions={ (e['bindings']['var'],e['bindings']['side']) for e in ctx['operator_scores']
                if e['operator']=='collect_transitions' and e.get('proposal')}
    assert directions=={(v,d) for v in (H,R) for d in ('positive','negative')}


def test_extra_endpoint_evidence_separate_from_policy_training(spec):
    rows=[sample(1,3,False)]
    rows[0].update(data_role='frontier_expert',expert_success=False)
    invalid=sample(2,4,True)
    invalid['provenance']['measurement_valid']=False
    endpoints=rows+[invalid]
    records=list(terminal_samples(endpoints,spec,CID))
    assert len(records)==2 and not records[1]['coords']
    facts=episode_frontiers([],spec,CID,rows=[],endpoint_rows=endpoints)
    direct=next(f for f in facts if f['pred']=='direct_boundary' and f['args'][:2]==[H,'positive'])
    assert direct['metrics']['observed_points'][0]['x']==3
    assert direct['metrics']['n_success']==0
    policy=next(f for f in facts if f['pred']=='policy_frontier' and f['args'][:2]==[H,'positive'])
    assert not policy['metrics']['observed_points']


def test_need_action_does_not_fallback_to_center_without_gap(spec):
    from bottle_scan.state_graph.facts import fact
    f=fact('development_class_missing',[CID,'NeedAction'],[],{'missing_classes':['NeedAction']})
    ctx=context(spec,{'variables':{}},[f],state={'phase':'plan'},budget_left=8)
    assert all(e['op']!='collect_outcomes' for e in ctx['applicable_operators'])
    assert all(f['id'] not in e['premises_satisfied_by'] for e in ctx['applicable_operators'])


def test_continuations_with_identical_plans_get_distinct_trial_ids(spec,tmp_path):
    ctx=options(spec,[sample(i,i,True) for i in range(1,5)])
    cell=next(e['proposal']['plan']['cells'][0] for e in ctx['operator_scores'] if e['operator']=='probe_amplitude')
    ids=[]
    for i in (1,2):
        out=tmp_path/str(i)
        state=dict(created_ns=i,frozen=None,rule={'rule_id':'r'},revision=1,history=[],stage_token='1/execute/0',
                   round_operation='probe_amplitude',spent={'trials':0},trials=[])
        campaign.queue_plan(out,state,spec,{'cells':[cell]},'exploration')
        t=state['trials'][0]
        req=campaign.read(out/'trials'/t['trial_id']/'request.json')
        assert req['mode']=='planned' and req['amplitude_target']==cell['amplitude_target']
        ids.append(t['trial_id'])
    assert ids[0]!=ids[1]
