"""Synthetic evidence only. These tests never instantiate a device or a cloud provider."""
from copy import deepcopy
import json
import numpy as np
import pytest
from bottle_scan.state_graph.v5_profile import configure
from bottle_scan.state_graph.facts import fact,coordinate_evidence
from bottle_scan.state_graph.cli import initial_rule
from bottle_scan.state_graph.director_bridge import context,decide
from bottle_scan.state_graph.operators import applicable_operators,validate_decision,update_rule
from bottle_scan.state_graph.v5_facts import boundary_tests,spatial_facts,axis_facts,revise,episode_frontiers
from bottle_scan.state_graph.v5_director import adjudicate_predictions,validate_predictions,proposal,fixed_decision,scoring


def spec():
    axes={v:dict(kind='continuous',reference=0,nominal_step=3,active=True,envelope=[-20,20],initial_range=[-5,5],max_expansion=5) for v in ['r','h']}
    return configure(dict(perturbation_variables=axes,active_post_condition='lift',
            task={'action_space':{'id':'rh','dim':2,'names':['r','h'],'continuous_displacement_variables':['r','h']}},
            observation={'modalities':{'tac':{}}},safety_envelope={'max_action':[10,10]},
            post_conditions=[{'id':'lift'}],budget={'max_new_trials':40,'max_batch_trials':16}),{'director':{'mode':'fixed_priority'}})


def row(i,x,y,ok,**kwargs):
    return dict(node_id=str(i),trial_id=str(i),group_id=str(i),window_index=0,session='P',
                obs={'tac':[float(x),float(y)]},attrs={},is_terminal=True,
                repeat_settings={'r':x,'h':y},repeat_key=str((x,y)),
                provenance={'coordinate_schema':'measured_placement_relative','coordinate_source':'measured',
                            'placement_reference_id':'cal-1','placement_reference_valid':True},
                plan_coords={'r':x,'h':y},post={'lift':{'evaluated':True,'success':ok,**kwargs}})


def test_measured_coordinates_and_rank_before_stochastic():
    s=spec();rows=[row(i,i,0,i>8) for i in range(18)];p=rows[0]
    assert coordinate_evidence(p,'r',s)==0
    p['provenance']['placement_reference_valid']=False
    assert coordinate_evidence(p,'r',s) is None
    p['provenance']={'coordinate_estimate_mm':{'r':12}}
    assert coordinate_evidence(p,'r',s) is None
    assert coordinate_evidence(p,'r',s,allow_estimate=True)==12
    test,=boundary_tests(rows,'lift',s)
    assert test['var']=='r' and test['coordinate_source']=='measured'
    for i,r in enumerate(rows):r['post']['lift']['success']=i%2==0
    assert not boundary_tests(rows,'lift',s)


def test_stratified_gaps_signs_noise_and_floor():
    s=spec();rows=[row(1,-1,0,False),row(2,0,0,True),row(3,2,8,False),row(4,3,0,False,failure_cause='closure_floor_reached'),row(5,1,0,True)]
    fs=spatial_facts(rows,s,'lift')
    assert any(f['pred']=='no_failure' and f['args']==['h','negative','lift'] for f in fs)==False
    gaps=[f['args'][1] for f in fs if f['pred']=='boundary_gap' and f['args'][0]=='r']
    assert gaps==[[-1,0]]
    assert not any('4' in f['evidence'] for f in fs if f['pred'] in ('boundary_gap','failure_present'))
    assert all('coordinate_source' in f['metrics'] for f in fs)
    rule=initial_rule([],s)
    ops=applicable_operators([fact('no_failure',['r','positive','lift'],['5'])],s,rule)
    assert [(x['bindings']['var'],x['bindings']['side']) for x in ops if x['op']=='expand_range']==[('r','positive')]
    bad=[row(10,0,0,True),row(11,2,0,False)]
    for r in bad:r.update(repeat_settings={'r':0,'h':0},repeat_key='same')
    res=next(f['metrics'] for f in spatial_facts(bad,s,'lift') if f['pred']=='coordinate_resolution' and f['args']==['r'])
    assert res['noise_mm']==pytest.approx(np.sqrt(2)) and res['resolution_mm']==pytest.approx(2*np.sqrt(2))


def test_capability_uses_single_axis_receipts_and_grouped_outcomes():
    s=spec();rows=[];edges=[]
    for i in range(18):
        x=i-8
        a=row(f'{i}a',x,0,x>=0);a['provenance']['direct_A']=True
        b=row(f'{i}b',x+1,0,x>=0)
        rows.extend((a,b));edges.append(dict(kind='observed',source=a['node_id'],target=b['node_id'],edge_id=str(i),u=[1,0],
                                              receipt={'applied':True,'net_contact_displacement_mm':[1,0]}))
    n={'modalities':{'tac':{'status':'ready','mean':[0,0],'components':None,'single_covariance':np.eye(2).tolist()}}}
    fs=axis_facts(rows,{'edges':edges},n,s,'lift')
    assert any(f['pred']=='axis_observable' and f['args']==['r'] and f['metrics']['sensitivity']==pytest.approx(1.) for f in fs)
    assert any(f['pred']=='axis_consequential' and f['args']==['r','lift'] for f in fs)
    s['perturbation_variables']['r']['active']=False
    ops=applicable_operators(fs,s,initial_rule([],s))
    assert any(o['op']=='activate_variable' and o['bindings']['var']=='r' for o in ops)
    assert not any(o['op']=='activate_variable' for o in applicable_operators(fs[:1],s,initial_rule([],s)))


def test_cluster_missing_evidence_and_dynamics_independent_training():
    s=spec();rs=[row(1,1,0,False)];cid='lift'
    graph={'edges':[],'reachability':{cid:{'1':{'status':'unsupported'}}},'probabilities':{cid:{'1':{'n':1,'successes':0}}}}
    clusters={'clusters':[{'cluster_id':'q','node_ids':['1']}]}
    base=[fact('coverage_ok',[x],['1']) for x in ('V','pi')]+[fact('unsupported',['1','lift'],['1']),fact('drift',['P'],['1'])]
    ms={'dynamics':{'status':'unavailable','action_ablation':{'relative_gain':0}}}
    fs=revise(base,graph,rs,{},clusters,{'modalities':{}},s,ms)
    ops=applicable_operators(fs,s,initial_rule([],s))
    assert {o['op'] for o in ops}>={'train','collect_transitions','recalibrate_reference'}
    assert not any(f['pred']=='unsupported' for f in fs)
    assert any(f['pred']=='recovery_evidence_missing' and f['metrics']['intervals']['r']==[1,1] for f in fs)
    ident=next(o for o in ops if 'action_design' in o['bindings'])
    ctx=context(s,initial_rule([],s),fs,budget_left=16)
    answer=proposal(ctx,ident);assert validate_decision(answer,ctx)
    answer['plan']['cells'].pop()
    with pytest.raises(ValueError,match='sign/amplitude'):validate_decision(answer,ctx)


def ep(i,role,x,ok,model=None,**extra):
    return dict(trial_id=str(i),controller_role=role,condition={'object':'bottle'},model_id=model,
                target_consumer='lift',first_contact=row(str(i)+':0',x,0,ok),terminal_outcomes={'lift':dict(evaluated=True,success=ok)},
                receipts=[{'applied':True}] if role=='expert' else [],expert_source='human' if role=='expert' else None,**extra)


def test_frontier_distinct_roles_versions_gaps_unmeasured_and_stop():
    s=spec();rs=[ep(1,'direct',1,True),ep(2,'policy',3,True,'A'),ep(3,'expert',8,True),
                 ep(4,'policy',18,True,'B'),ep(5,'policy',19,True,'A',post_freeze=True)]
    missing=ep(6,'expert',19,True);missing['first_contact']=None
    rs.append(missing);fs=episode_frontiers(rs,s,'lift',model_id='A')
    f=[f for f in fs if f['args']==['r','positive','lift']]
    vals={q['pred']:q['metrics'].get('farthest_observed_success') for q in f}
    assert [vals[k] for k in ('direct_boundary','policy_frontier','expert_frontier')]==[1,3,8]
    assert next(q for q in f if q['pred']=='frontier_gap')['metrics']['interval']==[3,8]
    op=applicable_operators(f,s,initial_rule([],s))
    assert any(o['op']=='request_expert_recovery' and o['bindings'].get('interval')==[3,8] for o in op)
    assert not any(o['bindings'].get('role')=='policy_frontier' for o in op)
    rs.append(ep(7,'policy',8,True,'A'))
    fs=episode_frontiers(rs,s,'lift',model_id='A')
    assert any(q['pred']=='frontier_direction_closed' and q['args']==['r','positive','lift'] for q in fs)


def test_deterministic_scoring_predictions_rules_and_legacy(tmp_path):
    s=spec();s['priors']={'sampling_preference':'prefer-up'}
    s['director']['use_priors']=False;r=initial_rule([],s)
    hi=fact('recovery_evidence_missing',['high','lift'],['A'],{'intervals':{'r':[1,2]}})
    lo=fact('recovery_evidence_missing',['low','lift'],['B'],{'intervals':{'r':[-2,-1]}})
    var=fact('posterior_variance',['lift'],['A','B'],{'by_node':{'A':.2,'B':.002}})
    ctx=context(s,r,[hi,lo,var],budget_left=10,state={'phase':'plan'})
    assert 'prefer-up' not in json.dumps(ctx)
    choices=[o for o in ctx['operator_scores'] if o['operator']=='collect_transitions']
    assert choices[0]['score']>choices[1]['score']
    a=decide(ctx,tmp_path);b=decide(ctx,tmp_path);assert a['answer']==b['answer'] and a['status']=='accepted'
    for x in a['next_rule']['field_provenance'].values():assert x['premises']
    pred=dict(pred='recovery_evidence_missing',args=['high','lift'],expect='disappears')
    assert adjudicate_predictions([pred],[lo],[hi,lo])[0]['status']=='confirmed'
    with pytest.raises(ValueError):validate_predictions([{**pred,'pred':'I_made_this_up'}],ctx)
    with pytest.raises(ValueError):validate_predictions([{**pred,'expect':'metric','metric':'not_defined','value':0,'comparison':'gt'}],ctx)
    ct2=context(s,r,[lo,var],budget_left=10,state={'phase':'plan'})
    decide(ct2,tmp_path)
    assert json.loads((tmp_path/'hypotheses.json').read_text())['history']
    legacy=deepcopy(s);legacy.pop('graph_schema');legacy.pop('director')
    assert not any(o['op']=='probe_recovery_frontier' for o in applicable_operators([hi],legacy,r))


def test_spacing_and_set_allocation():
    s=spec();r=initial_rule([],s)
    f=fact('boundary_gap',['r',[-3,3],'lift'],['a','b'],{'gap':6})
    ctx=context(s,r,[f],state={'phase':'plan'},budget_left=12)
    a=proposal(ctx,next(o for o in ctx['applicable_operators'] if o['op']=='densify'))
    validate_decision(a,ctx)
    a['plan']['cells'][0]['settings']['r']=-2.95
    with pytest.raises(ValueError,match='spacing'):validate_decision(a,ctx)
    fs=[fact('dev_B_gap',['lift'],[])]
    ctx=context(s,r,fs,state={'phase':'update_rule'},budget_left=5)
    a=proposal(ctx,next(o for o in ctx['applicable_operators'] if o['op']=='set_allocation'))
    new=update_rule(r,validate_decision(a,ctx),s)
    assert new['target_consumer']=='lift' and new['allocation']['policy']==0
    assert new['field_provenance']['allocation.expert']['premises']



def test_measured_region_score_omits_unrelated_same_trial_nodes():
    s=spec();r=initial_rule([],s)
    f=fact('frontier_gap',['r','positive','lift'],['t'],{'condition':{'mass':'light'},'interval':[1,4],'length_mm':3})
    ids={'inside':.10,'outside':.20,'wrong_bottle':.30}
    loc={k:{'coords':{'r':x},'attrs':{'mass':m},'trial_id':'t'} for k,x,m in
         [('inside',2,'light'),('outside',12,'light'),('wrong_bottle',2,'heavy')]}
    stats=fact('posterior_variance',['lift'],ids,{'by_node':ids,'locations':loc})
    ctx=context(s,r,[f,stats],budget_left=16,state={'phase':'plan'})
    score=next(o for o in ctx['operator_scores'] if o['operator']=='request_expert_recovery')
    assert score['target_node_ids']==['inside']
    assert score['score']==pytest.approx((.10+3*s['director']['frontier_weight'])/score['planned_trials'])


def test_expert_gap_select_retrain_and_remeasure_converges(tmp_path):
    """A deliberately learnable synthetic environment; the real training and decision APIs are used."""
    from bottle_scan.state_graph.policy import fit_policy,predict_policy
    s=spec();s['models']={'min_policy_transitions':1,'ensemble':4};s['director']['plan_reps']=1
    # Only one active axis here. Existing synthetic witness at +6 is initially reserved for comparison.
    s['perturbation_variables']['h']['active']=False
    rule=initial_rule([],s);noise={'active_modalities':['tac'],'modalities':{'tac':dict(status='ready',mean=[0,0],components=None,single_covariance=[[1,0],[0,1]])}}
    train=[];edges=[];advantages=[];records=[]
    def learn():
        return fit_policy(train,{'edges':edges},{'transitions':advantages},noise,s,'lift',seed=2)
    def demonstrate(x):
        i=len(train);r=row('training'+str(i),x,0,True);train.append(r)
        edges.append(dict(edge_id=str(i),kind='observed',source=r['node_id'],u=[-x,0],
                          receipt={'applied':True,'source':'expert','net_contact_displacement_mm':[-x,0]}))
        advantages.append(dict(edge_id=str(i),post_condition='lift',advantage=1))
    for _ in range(8):demonstrate(0)
    def check(x,model,version):
        u=predict_policy(model,row('probe',x,0,True),noise)['u'][0]
        ok=abs(x+u)<=1
        records.append(ep('check'+str(len(records)),'policy',x,ok,version))
        return ok
    prior=learn()
    assert check(1,prior,'v0') and not check(6,prior,'v0')
    records.append(ep('witness','expert',6,True))
    version='v0'
    cell=['r','positive','lift']
    for iteration in range(1,13):
        fs=episode_frontiers(records,s,'lift',model_id=version)
        fs=[f for f in fs if f['args']==cell]
        fs.append(fact('model_executable',['lift'],['training0']))
        ctx=context(s,rule,fs,budget_left=16,state={'phase':'plan'})
        chosen=decide(ctx,tmp_path/'director')['answer']
        assert chosen['operator']=='request_expert_recovery' and chosen['args']['interval']==[1,6]
        for cellplan in chosen['plan']['cells']:
            for _ in range(cellplan['reps']):
                x=cellplan['settings']['r']
                records.append(ep('selected'+str(len(records)),'expert',x,True))
                demonstrate(x)
        version='v'+str(iteration)
        new=learn();improved=check(6,new,version)
        assert check(1,new,version)
        if improved:break
    assert improved
    after=episode_frontiers(records,s,'lift',model_id=version)
    assert any(f['pred']=='frontier_direction_closed' and f['args']==cell for f in after)
    assert not any(f['pred']=='frontier_gap' and f['args']==cell for f in after)
    assert not any(o['bindings'].get('side')=='positive' and o['bindings'].get('var')=='r' for o in applicable_operators(after,s,rule))


def test_rule_records_actual_allocation_unknown_starts_and_directional_ranges():
    s=spec();s['current_placement']=True;r=initial_rule([],s)
    f=fact('transition_evidence_missing',['lift'],['real_row'])
    d={'operator':'collect_seed','args':{},'premises':[f['id']],'reason':'repeated starts',
       'plan':{'cells':[{'settings':{},'actions':[],'reps':2,'blocks':[{},{}]},
                         {'settings':{},'actions':[{'kind':'rh','u':[1,0]}],'reps':1,'blocks':[{}]}]}}
    revised=update_rule(r,d,s)
    assert revised['allocation']=={'direct':4,'exploration':1,'expert':0,'policy':0}
    assert revised['start_points']==[{'basis':'placement_reference'}]
    d['operator']='expand_range';d['args']={'side':'positive','var':'r','by':3}
    d['plan']={'cells':[{'settings':{'r':8},'reps':1,'actions':[]}]}
    changed=update_rule(revised,d,s)
    assert changed['variables']['r']['range']==[-5,8] and changed['directional_ranges']['r']['positive']==[0,8]
    assert changed['field_provenance']['directional_ranges.r.positive']['fact_ids']==[f['id']]
