from bottle_scan.state_graph.development_gate import screen,interim
from bottle_scan.state_graph import campaign
from tests.test_state_graph_v5_rounds import setup_round,synthetic_executor


def test_unsupported_candidate_skips_all_development(tmp_path):
    out,spec=setup_round(tmp_path)
    spec['loop'].update(dev_gate=True,dev_min_ready=100000,dev_min_support_fraction=.5)
    campaign.atomic(out/'spec.json',spec)
    done=campaign.run(out,executor=synthetic_executor,max_decisions=130)
    dev=campaign.read(out/'rounds/round_0001/dev_validate.json')
    assert dev['dev_skipped'] and 'dev_min_ready' in dev['gate']['unmet']
    assert not any(t['request']['role']=='development' for t in done['trials'])
    assert any(f['pred']=='candidate_lacks_capability' for f in dev['acceptance']['facts'])
    assert done['spent']['trials']>0


def test_optimistic_wilson_futility_cannot_stop_independent():
    ac=dict(primary_rates=['recovery_success_rate'],rate_confidence=.95,development_B={'min_recovery_success_rate':.8},futility_stop=True)
    spec=dict(active_post_condition='lift',acceptance=ac)
    record=dict(role='development',model_id='frozen',post_condition='lift',initial_truth='NeedAction',success=False)
    assert interim([],24,spec,'frozen')['reason'] is None
    assert interim([record]*4,20,spec,'frozen')['reason']=='development_futility_stop'


def test_stop_is_idempotent_and_retains_dispatch_cost(tmp_path):
    from bottle_scan.state_graph.development_gate import stop_pending
    records=[]
    for i in range(24):
        ident=str(i);directory=tmp_path/ident
        t=dict(trial_id=ident,status='complete' if i<4 else 'queued',directory=str(directory))
        records.append(t)
        if i<4:
            campaign.atomic(directory/'result.json',dict(assessment=dict(role='development',post_condition='lift',
                model_id='m',group_id=ident,initial_truth='NeedAction',success=False)))
    state=dict(trials=records,round_stage='dev_validate',round_index=2,model={'model_id':'m'},
               spent={'trials':4},dispatch_attempts=5,independent_reserve=24,
               pending={'role':'development','trial_ids':[str(i) for i in range(24)]})
    spec=dict(active_post_condition='lift',acceptance=dict(futility_stop=True,rate_confidence=.95,
              primary_rates=['recovery_success_rate'],development_B={'min_recovery_success_rate':.8}))
    assert stop_pending(tmp_path,state,spec)
    assert state['development_stops']['2']['used']==4 and len(state['development_stops']['2']['unrun'])==20
    assert state['spent']['trials']==4 and state['dispatch_attempts']==5 and state['independent_reserve']==24
    assert not stop_pending(tmp_path,campaign.read(tmp_path/'state.json'),spec)


def test_nearest_start_requires_measured_coordinates_and_mass():
    spec=dict(active_post_condition='lift',loop=dict(dev_min_ready=1,dev_min_support_fraction=.5),
        task={'action_space':{'continuous_displacement_variables':['r']}},
        facts={'measured_placement_coordinates':True},perturbation_variables={'r':{'kind':'continuous','reference':0}})
    def node(n,x,mass,valid):
        return dict(node_id=n,plan_coords={'r':x},attrs={'mass_class':mass},provenance={
            'coordinate_schema':'measured_placement_relative','placement_reference_valid':valid,
            'placement_reference_id':'w','coordinate_source':'measured'})
    rows=[node('valid',2,'light',True),node('invalid',0,'heavy',False)]
    g={'reachability':{'lift':{r['node_id']:{'status':'ready'} for r in rows}}}
    starts=[dict(settings={'r':0},block={'mass_class':m}) for m in ('light','heavy')]
    gate=screen(rows,g,starts,spec,'m')
    assert gate['support_fraction']==.5 and gate['checks'][0]['nearest']==[('valid','ready')]
    assert not gate['checks'][1]['nearest']
    spec['facts']['measured_placement_coordinates']=False
    rows[1]['provenance']={}
    assert not screen(rows,g,starts,spec,'m')['checks'][1]['nearest']

