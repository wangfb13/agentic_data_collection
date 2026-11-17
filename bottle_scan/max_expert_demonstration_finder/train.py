"""Use measured expert actions for the graph action head, never as offset labels."""
from copy import deepcopy
from pathlib import Path
import json
import joblib
from .episodes import summarize
from .evidence import require,sha
from bottle_scan.state_graph.common import independent,write
from bottle_scan.state_graph.skill_interface import repeat_key,task_id,model_observation
from bottle_scan.state_graph.cli import build

def training_examples(attempts,m):
    out=[];witnesses=[]
    for a in attempts:
        if a['phase']!='explore' or not a['end']:continue
        e=summarize(a,m)
        # Only evaluated independent direct controls, or completed witnessed same-capability expert trajectories.
        if e['label_invalidated'] or not e['consumer_tested']:continue
        if e['controller_role']=='expert':
            if not e['expert_witness'] or e['expert']['stronger']:continue
            witnesses.append(a['id'])
        elif e['controller_role']!='direct' or e['human_intervention']:continue
        out.append((a,e))
    return witnesses,out

def graph_rows(examples,spec,m):
    result=[]
    for a,e in examples:
        first=a['events']['reply/touch']['observation'];obs=[first]+[t['after'] for t in e['trajectory']]
        for i,o in enumerate(obs):
            r=deepcopy(o['row']);require(r.get('window_valid') and not independent(r) and o['contact_state']=='valid_bilateral','invalid training window')
            require(r['task_id']==task_id(spec) and all(r['extractor_versions'].get(k)==v['version'] for k,v in spec['observation']['modalities'].items()),'incompatible expert features')
            model_observation(r,spec['observation']['modalities'])
            group='frontier/'+a['id'];ident=f'{group}:{i}';prev=f'{group}:{i-1}' if i else None
            rec=e['trajectory'][i-1] if i else None
            cid=spec['active_post_condition'];term=i==len(obs)-1
            r.update(node_id=ident,session=group,trial_id=a['id'],group_id=group,window_index=i,previous_node_id=prev,
                t_ns=o['timestamp_ns'],phase='pre' if not i else 'post',is_reference=False,plan_coords={},repeat_settings={},
                evaluation_only=False,evaluation_scope='development',rule_id='frontier_expert_training',risk={},
                post={cid:{'evaluated':term,'success':e['consumer_outcome'] if term else None,'level':cid if term else None,
                           'evidence':e['raw_evidence'] if term else []}},is_terminal=term,
                attrs={**r.get('attrs',{}),**m['condition']},provenance={'coordinate_schema':'fixed_contact_a_v1',
                'reference_digest':m['reference_digest'],'target_definition_id':m['target_definition_id'],'a_tcp':m['axis_tcp'],
                'supervision':first.get('measurement') if i==0 else None,'repeat_context':group},
                u_executed=[rec['actual_mm']] if rec else None,u_receipt=rec['receipt'] if rec else None,
                action_executed={'kind':'tcp_translation_a','u':[rec['actual_mm']]} if rec else None)
            r['repeat_key']=repeat_key(r,spec);result.append(r)
    return result

def retrain(db,rid,dest,config,seed_campaign):
    from skills.contact_position_a.frontier import verify_files,manifest
    from skills.contact_position_a.local_assets import reference_rows
    rr=db.round(rid);m=rr['manifest'];require(rr['stage']=='closed' and not db.pending(),'close sampling before training')
    require(m['environment']=='physical','physical training cannot read fixture evidence')
    verify_files(m);files=m['artifacts'];ref=json.loads(Path(files['reference']['path']).read_text())
    old=joblib.load(files['bundle']['path']);spec=old['spec']
    saved=Path(files['spec']['path']).parent/'training_rows.json'
    if saved.is_file():
        require('training_rows' in files,'unbound parent training archive')
        rows=json.loads(saved.read_text())
    else:
        reconstructed,rows=reference_rows(Path(seed_campaign),ref)
        require(reconstructed==spec,'baseline campaign spec changed')
    # Verify even the old outcomes/edges are reproducible, not merely an equal number of nodes.
    require(not any(independent(r) for r in rows),'independent data in training')
    support={r['node_id']:r for r in old['support']}
    require(len(rows)==len(support) and all(r['node_id'] in support and all(r[k]==v for k,v in support[r['node_id']].items()) for r in rows),'parent support mismatch')
    from bottle_scan.state_graph.graph import build_graph
    parent_graph=build_graph(rows,old['noise'],spec,old['dynamics'])
    require(parent_graph['graph_version']==old['graph_version'] and sha(parent_graph)==sha(old['graph']),'parent outcome/topology changed')
    witnesses,examples=training_examples(db.attempts(rid),m);require(witnesses,'no eligible real expert demonstrations')
    added=[];accepted=[];skipped=[]
    for a,e in examples:
        try:newrows=graph_rows([(a,e)],spec,m)
        except (ValueError,KeyError,TypeError) as exc:
            skipped.append({'attempt_id':a['id'],'reason':str(exc)});continue
        added.extend(newrows)
        if e['expert_witness']:accepted.append(a['id'])
    require(accepted,'witnessed paths lack compatible training observations')
    dest=Path(dest);dest.mkdir(parents=True,exist_ok=False)
    write(dest/'spec.json',spec);write(dest/'training_rows.json',rows+added)
    built=build(rows+added,spec,dest/'graph',validate=False)
    cfg={**config,'assets':str(dest.resolve())}
    if 'offset' in files:cfg['offset_model']=files['offset']['path']
    child=manifest(cfg)
    immutable=('reference_digest','target_definition_id','protocol','condition','limits','statistics','action_source','expert','expert_exchange','station','post_condition','sensor_version','tool_version','reject_rules')
    require(all(child[k]==m[k] for k in immutable),'changed comparison condition')
    new=db.new(child,parent=rid);result={'round_id':new,'parent':rid,'expert_attempts':accepted,'excluded_training_episodes':skipped,'new_rows':len(added),
          'training':built['training'],'verified_expansion':None,'needs_new_independent_confirmation':True}
    write(dest/'lineage.json',result);return result

