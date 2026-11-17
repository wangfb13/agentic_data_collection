"""Offline integration on the installed white graph, no driver or hardware calls."""
from pathlib import Path
import json
import joblib
import pytest
from bottle_scan.max_expert_demonstration_finder.demo import fixture,Fixture
from bottle_scan.max_expert_demonstration_finder.ledger import Ledger
from bottle_scan.max_expert_demonstration_finder.engine import execute
from bottle_scan.max_expert_demonstration_finder.train import training_examples,graph_rows
from bottle_scan.max_expert_demonstration_finder.search import offers
from bottle_scan.state_graph.cli import build
from skills.contact_position_a.frontier import FrozenPolicy,manifest,verify_files
from skills.contact_position_a.local_assets import reference_rows

@pytest.mark.skipif(not Path('outputs/local_contact_a_v1/spec.json').is_file(),reason='requires installed white assets')
def test_frozen_native_policy_and_expert_graph_training(tmp_path):
    cfg=json.loads(Path('skills/contact_position_a/frontier_config.json').read_text());real=manifest(cfg);verify_files(real)
    b=joblib.load(real['artifacts']['bundle']['path']);spec,oldrows=reference_rows(Path('outputs/state_graph/campaign_white_a_v4_forcepolicy'),json.loads(Path(cfg['reference']).read_text()))
    assert len(oldrows)==len(b['support']) and spec==b['spec']
    stamp=101;policy=FrozenPolicy(real,lambda:stamp)
    ref=json.loads(Path(cfg['reference']).read_text())
    obs=dict(tactile={'bilateral':True,'force_stable':True,'safety_ok':True,'descriptors':ref['standard_sample']['descriptors']},
             id='offline',timestamp_ns=stamp,row=b['support'][0])
    assert policy.decide(obs,0)['state'] in ('Ready','NeedAction','Unknown')
    # Synthesized steps deliberately stay in a fixture/test directory. Test the *actual* graph trainer on them.
    m=fixture();m['post_condition']=real['post_condition'];db=Ledger(tmp_path/'fixture');rid=db.new(m);s=Fixture(m);base_obs=s.observe
    def observed(first):
        o=base_obs(first);o['row']=oldrows[0 if first else 1];return o
    s.observe=observed;perform=s.perform
    def endpoint(req):
        v=perform(req)
        if req['operation']=='post':
            v.update(post_condition=m['post_condition']['id'],consumer_outcome={**m['post_condition']['consumer'],'L1':True,'evidence':[{'event':'main_resumed'}]})
        return v
    s.perform=endpoint
    c=next(v for v in offers(db.round(rid),[]) if v['role']=='expert_recovery' and v['x']==2.5)
    execute(db,rid,c,s,s)
    ids,example=training_examples(db.attempts(rid),m);assert len(ids)==1
    add=graph_rows(example,spec,m);res=build(oldrows+add,spec,tmp_path/'fake_training_only',validate=False)
    assert res['bundle']['model_id']!=b['model_id'] and len(res['bundle']['support'])==len(oldrows)+2
    assert any(e['source']==add[0]['node_id'] and e['target']==add[1]['node_id'] and e['kind']=='observed' for e in res['graph']['edges'])
