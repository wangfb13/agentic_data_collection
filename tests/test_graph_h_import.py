from pathlib import Path
from bottle_scan.state_graph.campaign import atomic
from skills.contact_position_a.p3_import import snapshot


def test_copy_restores_only_attested_forcepolicy_and_skips_independent(tmp_path):
    root=tmp_path/'original'; root.mkdir()
    cid='forcepolicy_lift'
    spec={'post_conditions':[{'id':cid,'consumer':{'controller':'existing_forcepolicy','version':'v','params':{'closure_floor':.33}}}],
          'perturbation_variables':{}}
    atomic(root/'seed_nodes.json',[])
    directory=root/'trial';directory.mkdir()
    n={'node_id':'contact','is_terminal':True,'window_index':0,'window_valid':True,'post':{cid:{'evaluated':False}},'provenance':{}}
    outcome=dict(controller='actual',consumer_id=cid,version='v',params={'closure_floor':.33},L1=True,L2=False)
    req=dict(trial_id='t',role='direct_baseline',post_condition=cid,block={},data_role='frontier_direct',round_index=1)
    t=dict(request=req,trial_id='t',directory=str(directory),status='complete')
    result=dict(nodes=[n],status='complete',consumer_order=[cid],consumer_outcomes={cid:outcome})
    atomic(directory/'result.json',result)
    before=(directory/'result.json').read_bytes()
    independent={**t,'directory':str(root/'never-open-independent'),'request':{**req,'role':'independent'}}
    rows,_,episodes,fixed=snapshot(root,{'trials':[t,independent]},spec,{'consumer':outcome})
    assert fixed==['contact'] and rows[0]['post'][cid]['success'] is False
    assert episodes[0]['terminal_outcomes'][cid]['evaluated'] and (directory/'result.json').read_bytes()==before
