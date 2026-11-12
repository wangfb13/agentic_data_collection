import copy
import tempfile
from skills.contact_position_a.legacy_v2.operators import applicable_operators,validate_decision,update_rule
from skills.contact_position_a.legacy_v2.director_bridge import context,decide,update_hypotheses
import pytest

SPEC={'envelope':{'allowed_axes':['a','l'],'range_mm':{'a':[-25,25],'l':[-10,10]}},
      'budget_left':30,'recordable_attributes':['closure_floor']}
RULE={'rule_id':'R1','active_axes':['a'],'range_mm':{'a':[-5,5]},'objects':['white_light','white_heavy']}
def fact(pred,args,ident='F1',**extra):return dict(id=ident,pred=pred,args=args,evidence=['n1'],**extra)
def answer(op,args,refs=(),plan=None):return {'hypotheses_update':[],'operator':op,'args':args,'premises':list(refs),'discriminates':[],'plan':plan,'expected':'如果新的位置仍无失败则假设被否定','reason':'比较局部边界'}

def test_empty_facts_only_pause():
    assert [x['op'] for x in applicable_operators([],SPEC,RULE)]==['pause']

def test_expand_and_rule_version():
    ctx=context(SPEC,RULE,[fact('no_failure',['R1'])]);a=answer('expand_range',{'axis':'a','side':'negative','by_mm':5},['F1'],{'cells':[{'offset_mm':-10,'reps':2},{'offset_mm':0,'reps':2}]})
    validate_decision(a,ctx);r=update_rule(RULE,a)
    assert r['range_mm']['a']==[-10,5] and RULE['range_mm']['a']==[-5,5] and r['parent_rule_id']=='R1'
    a['args']['by_mm']=6
    with pytest.raises(ValueError):validate_decision(a,ctx)

def test_premises_cannot_be_invented():
    ctx=context(SPEC,RULE,[fact('no_failure',['R1'])])
    with pytest.raises(ValueError):validate_decision(answer('train',{},['invented']),ctx)

def test_conflict_enables_declared_hidden_variable():
    ctx=context(SPEC,RULE,[fact('conflict',['n1','n2','closure_floor'])])
    assert any(x['op']=='probe_hidden_variable' and x['bindings']['var']=='closure_floor' for x in ctx['applicable_operators'])
    a=answer('probe_hidden_variable',{'var':'closure_floor','levels':[.330,.336]},['F1'],{'cells':[{'offset_mm':0,'reps':2,'closure_floor':x} for x in [.330,.336]]})
    validate_decision(a,ctx)
    a['args']['var']='undeclared'
    with pytest.raises(ValueError):validate_decision(a,ctx)

def test_densify_requires_three_new_positions():
    ctx=context(SPEC,RULE,[fact('boundary_gap',['a',[-5,0],'white_heavy'])],observed_positions={'a':[-5,0]})
    a=answer('densify',{'axis':'a','interval':[-5,0],'object':'white_heavy'},['F1'],{'cells':[{'offset_mm':x,'reps':2} for x in [-4,-3,-2,0]]})
    validate_decision(a,ctx)
    a['plan']['cells']=a['plan']['cells'][:2]
    with pytest.raises(ValueError):validate_decision(a,ctx)

def test_frozen_allows_complete_only_with_evidence():
    ops=applicable_operators([fact('no_failure',['R1'])],SPEC,RULE,{'frozen_independent':True})
    assert [x['op'] for x in ops]==['pause']

def test_no_external_without_flag():
    with tempfile.TemporaryDirectory() as d:
        assert decide(context(SPEC,RULE,[]),d)['status']=='not_requested'

def test_two_revisions_then_pause():
    class P:
        n=0
        def direct(self,c):self.n+=1;return answer('complete',{},[])
    p=P()
    with tempfile.TemporaryDirectory() as d:r=decide(context(SPEC,RULE,[]),d,provider=p)
    assert p.n==3 and r['status']=='paused'

def test_llm_status_is_not_a_fact():
    h=[{'id':'H1','status':'open','proposed_at_graph':'G1','pending_status_proposal':'supported','support_when':{'pred':'object_split','args':['a','b','a']}}]
    assert update_hypotheses(h,[],'G2')[0]['status']=='open'
    assert update_hypotheses(h,[fact('object_split',['a','b','a'])],'G2')[0]['status']=='supported'


def test_original_reference_and_balance_validator_is_reused():
    ctx=context(SPEC,RULE,[fact('no_failure',['R1'])])
    a=answer('expand_range',{'axis':'a','side':'negative'},['F1'],{'cells':[{'offset_mm':-10,'reps':2}]})
    with pytest.raises(ValueError,match='reference'):validate_decision(a,ctx)
    a['plan']['cells'].append({'offset_mm':0,'reps':1})
    with pytest.raises(ValueError,match='balance'):validate_decision(a,ctx)


def test_densify_cannot_avoid_cited_disagreement():
    ctx=context(SPEC,RULE,[fact('high_disagreement',['a',4])])
    a=answer('densify',{'axis':'a','interval':[-5,-1]},['F1'],{'cells':[{'offset_mm':x,'reps':2} for x in [-4,-3,-2,0]]})
    with pytest.raises(ValueError,match='contain'):validate_decision(a,ctx)


def test_verify_edges_selects_a_maximum_coverage_start():
    f=fact('edge_unverified',[3],metrics={'path_fraction':1.,'verification_candidates':[
        {'source_node_id':'n1','edge_ids':['e1','e2'],'coverage_count':2,'planned_offset_mm':-5},
        {'source_node_id':'n2','edge_ids':['e3'],'coverage_count':1,'planned_offset_mm':5}]})
    ctx=context(SPEC,RULE,[f]);ops=[x for x in ctx['applicable_operators'] if x['op']=='verify_edges']
    assert len(ops)==1 and ops[0]['bindings']['source_node_id']=='n1'
    a=answer('verify_edges',{'source_node_id':'n1'},['F1'],{'cells':[{'offset_mm':-5,'reps':2,'source_node_id':'n1'},{'offset_mm':0,'reps':2}]})
    validate_decision(a,ctx)
    a['plan']['cells'][0]['source_node_id']='n2'
    with pytest.raises(ValueError,match='maximum-coverage'):validate_decision(a,ctx)


def test_invalid_json_proposal_gets_bounded_repair():
    from bottle_scan.agentic_provider import ProposalError
    class P:
        n=0
        def direct(self,ctx):
            self.n+=1
            if self.n==1:raise ProposalError('invalid JSON')
            return answer('pause',{})
    p=P()
    with tempfile.TemporaryDirectory() as d:r=decide(context(SPEC,RULE,[]),d,provider=p)
    assert r['status']=='accepted' and p.n==2
