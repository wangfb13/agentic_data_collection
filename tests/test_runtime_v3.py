from copy import deepcopy
import numpy as np
from bottle_scan.state_graph.skill import SymbolicRecoverySkill
from bottle_scan.state_graph.value import predict_value
from bottle_scan.state_graph.policy import predict_policy
from tests.fixtures.skill_toy.testing import trained


def test_models_do_not_use_analysis_fields():
    rows,spec,noise,f,g,labels,clusters,bundle,_=trained();a=deepcopy(rows[0]);b=deepcopy(a)
    b.update(attrs={'secret':True},plan_coords={'x':999},post={'secret':True},R=-999)
    for fn,key in [(predict_value,'V'),(predict_policy,'pi')]:
        assert fn(bundle['models']['centered'][key],a,noise)==fn(bundle['models']['centered'][key],b,noise)


def test_runtime_reobserves_after_action_before_handoff():
    *_,bundle,status=trained();runtime=SymbolicRecoverySkill(bundle);calls=[]
    def infer(row):
        return {'state':'ready','value':{'V':.1,'p_hat':.9},'policy':None,'plan':None} if row['i'] else {'state':'recoverable','value':{'V':2.,'p_hat':.2},'policy':{'u':[.1,.2]},'plan':None}
    runtime.infer=infer
    class Adapter:
        def observe(self):calls.append('observe');return {'i':calls.count('execute')}
        def execute(self,u):calls.append('execute');return {'completed':True}
        def emit(self,event):pass
    result=runtime.execute(Adapter());assert result['completed'] and calls==['observe','execute','observe']
    assert result['handoff']['latest_observation']['i']==1
