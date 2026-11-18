from bottle_scan.state_graph.axis_resolution import balance_facts
from skills.contact_position_a.consumer_audit import classify,receipt


def test_only_active_underrepresented_axis_is_requested():
    s={'facts':{'min_axis_trials':3},'task':{'action_space':{'continuous_displacement_variables':['r','h'],'names':['dr','dh']}},
       'perturbation_variables':{'r':{'active':True},'h':{'active':True}}}
    graph={'resolution':{'by_axis':{n:{'n_actions':count,'evidence':[['x','y']]} for n,count in [('dr',1),('dh',4)]}}}
    assert [f['args'] for f in balance_facts(graph,s)]==[['r']]
    s['perturbation_variables']['h']['active']=False
    assert not balance_facts(graph,s)


def test_consumer_statuses_require_receipts():
    cond={'id':'b','consumer':{'controller':'test','version':'v','params':{'closure_floor':.33}}}
    node={'node_id':'n','post':{'b':{'evaluated':True,'success':False}}}
    result={'consumer_order':['a','b'],'consumer_outcomes':{'a':{'L1':True},'b':dict(
        consumer_id='b',controller='test',version='v',params={'closure_floor':.33},L1=True,L2=False)}}
    entries=[classify('b',cond,result,[{},node])]
    result['consumer_outcomes']['b']['version']='wrong'
    entries.append(classify('b',cond,result,[{},node]))
    del result['consumer_outcomes']['b']
    result['pairing']={'reason':'second_skipped_object_or_contact_change'}
    entries.append(classify('b',cond,result,[{}]))
    entries.append(classify('b',cond,{'failure_cause':'model_rejected'},[]))
    log=receipt([dict(trial_id=i,consumers={'b':v}) for i,v in enumerate(entries)])
    assert log['counts']['b']==dict(imported=1,skipped=1,unmatched=1,not_executed=1)
    assert log['warnings']==[dict(trial_id=1,consumer='b',reason='version_mismatch')]
    assert classify('b',cond,{},[node])['status']=='unmatched'


def test_unmatched_import_persists_warning_before_dispatch(tmp_path):
    import pytest
    from bottle_scan.state_graph import campaign
    from tests.test_graph_campaign_v4 import make_campaign
    _,spec=make_campaign(tmp_path)
    spec['import']={'consumer_audit':True}
    campaign.atomic(tmp_path/'spec.json',spec)
    out=tmp_path/'audit'; warning=dict(trial_id='t',consumer='test',reason='version_mismatch')
    campaign.atomic(out/'import_receipt.json',{'consumer_audit':{'warnings':[warning]}})
    with pytest.warns(RuntimeWarning,match='1 unmatched'):
        campaign.initialize(tmp_path/'config.json',out)
    assert campaign.read(out/'state.json')['import_warnings']==[warning]


def test_director_plan_requires_positive_and_negative_on_underrepresented_axis():
    import pytest
    from bottle_scan.state_graph.v5_director import proposal,fixed_decision
    from bottle_scan.state_graph.v5_operators import validate
    spec={'director':{'plan_reps':1},'perturbation_variables':{v:{'kind':'continuous','envelope':[-10,10]} for v in ['r','h']},
        'task':{'action_space':{'id':'rh','dim':2,'continuous_displacement_variables':['r','h']}},
        'budget':{'max_batch_trials':4},'active_post_condition':'lift','safety_envelope':{'max_action':[10,10]}}
    offer={'op':'collect_transitions','bindings':{'action_design':'axis_both_signs','var':'r'},'premises_satisfied_by':[]}
    ctx={'spec':spec,'current_rule':{},'facts':[],'budget_left':4}
    ans=proposal(ctx,offer);validate(ans,ctx)
    assert [c['actions'][0]['u'] for c in ans['plan']['cells']]==[[5.,0.],[-5.,0.]]
    options=[dict(operator='collect_transitions',bindings=offer['bindings'],proposal=ans,premises=[],score=0),
             dict(operator='collect_outcomes',bindings={},proposal={'operator':'collect_outcomes'},premises=[],score=100)]
    assert fixed_decision({'operator_scores':options,'facts':[]})==ans
    ans['plan']['cells'].pop()
    with pytest.raises(ValueError,match='both signs'):validate(ans,ctx)

