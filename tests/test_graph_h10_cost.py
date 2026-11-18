from bottle_scan.state_graph.ready_kernel import select_cost
from bottle_scan.state_graph.ready_comparison import operating_curve,error_analysis


def test_cost_ratio_monotone_and_cap_optional():
    scores=[dict(threshold=t,correct=c,false=f,failures=10,false_release_rate=f/10) for t,c,f in [(.3,10,3),(.6,8,1),(.9,5,0)]]
    def pick(cost,cap=None):
        return select_cost([dict(x) for x in scores],10,dict(false_release_cost=cost,miss_cost=1,max_false_release_rate=cap))['threshold']
    assert [pick(x) for x in [.2,1,4]]==[.3,.6,.9]
    assert pick(.2,0)==.9


def test_comparison_counts_and_hidden_concentration():
    rows=[dict(node_id=str(i),trial_id=str(i),session='one',attrs={'mass_class':'light' if i<10 else 'heavy'}) for i in range(30)]
    preds=[dict(node=str(i),group=str(i),eligible=True,score=.8 if i<10 else .3,success=False,accepted=i<10) for i in range(30)]
    assert operating_curve(preds,[.3,.8])[-1]['false']==10
    analysis=error_analysis({'a':{'heldout':preds},'b':{'heldout':preds}},rows,{'perturbation_variables':{}},
                            dict(error_attributes=['mass_class'],coordinate_bin_mm=2.5,enrichment_alpha=.05))
    assert len(analysis['records'])==10 and analysis['facts'][0]['args']==['mass_class']
