"""Read archived artifacts and raw decision/consumer records without starting a scheduler."""
import json
from pathlib import Path
from collections import Counter
import joblib
from .episodes import post_result
from .evidence import sha
from skills.contact_position_a.frontier import filehash

def audit(m,repo):
    base=Path(repo);b=joblib.load(m['artifacts']['bundle']['path']);cid=m['post_condition']['id']
    from bottle_scan.state_graph.skill import SymbolicRecoverySkill
    skill=SymbolicRecoverySkill(b);replayed=Counter(skill.infer(row)['state'] for row in b['support'])
    counts=Counter(v['status'] for v in b['graph']['reachability'][cid].values());others=[];exact=[];same_model_unbound=[];logs=0;unreadable=[]
    paths=set((base/'outputs/state_graph').glob('campaign_white*/trials/*/episodes/*/events.jsonl'))
    paths.update((base/'outputs/local_contact_a_v1').glob('**/events.jsonl'))
    for d in base.parent.glob('contact_scheduler*'):
        paths.update((d/'outputs').glob('**/episode_*/events.jsonl'))
    for p in sorted(paths):
        try:events=[json.loads(l) for l in p.read_text().splitlines() if l.strip()]
        except (OSError,ValueError) as exc:unreadable.append([str(p),str(exc)]);continue
        logs+=1
        header=next((e for e in events if e.get('event')=='graph_runtime_selected'),{})
        req=header.get('request',{});dec=[e for e in events if e.get('event')=='recovery_skill_decision']
        if req.get('post_freeze') or req.get('role')=='independent' or 'offline' in str(p).lower():continue
        if header.get('mode')!='learned' or not dec or dec[0]['skill_outputs']['invocation'].get('invoke') is not True:continue
        final=next((e for e in events if e.get('event')=='contact_completed'),{})
        if not any(c.get('receipt',{}).get('applied') is True for c in final.get('corrections',[]) if c.get('receipt')):continue
        last_window=next((e for e in reversed(events) if e.get('event')=='state_graph_window'),{})
        if not last_window or not last_window.get('previous_node_id'):continue
        outcome=next((e.get('consumer_outcome') for e in events if e.get('event')=='state_graph_consumer_result' and
                      e.get('node_id')==last_window['node_id'] and e.get('group_id')==last_window['group_id']),None)
        if not outcome:continue
        evaluator='fixed_ready' if req.get('post_condition','').startswith('fixed_squeeze') else 'handoff_ready'
        goal={'id':req.get('post_condition'),'consumer':header.get('consumer',{}),'evaluator':'skills.contact_position_a.ready:'+evaluator}
        if not goal['id'] or not goal['consumer']:continue
        if post_result(dict(post_condition=goal['id'],consumer_outcome=outcome),goal) is not True:continue
        row=dict(raw=str(p),model_id=header.get('model_id'),post_condition=goal['id'],initial_model_state='NeedAction',independent_initial_state=None)
        if header.get('model_id')!=b['model_id'] or goal['id']!=cid or goal['consumer']!=m['post_condition']['consumer']:
            others.append(row)
        elif req.get('frontier_model_id')==m['model_id'] and req.get('frontier_manifest_digest')==sha(m):
            exact.append(row)
        else:same_model_unbound.append(row)
    models={}
    for p in [base/'outputs/local_contact_a_v1/offset.json',base/'outputs/white_offset_all_20261116_v1/offset_all.json']:
        if p.is_file():
            v=json.loads(p.read_text());models[p.name]={'model_id':v['model_id'],'usable':v['usable_for_B'],'training_count':v['training_count'],
                'inside_training_label_range_mm':v.get('trained_offset_range_mm'),'blocking_reasons':v['blocking_reasons']}
    prior=base/'outputs/white_offset_all_20261116_v1/windows.json';ranges={}
    if prior.is_file():
        w=json.loads(prior.read_text())
        for tier in {r['tier'] for r in w}:
            values=[r['y'] for r in w if r['tier']==tier]
            ranges[tier]={'windows':len(values),'min':min(values),'max':max(values)}
    return dict(configured_consumer_matches_frozen_version=(m['artifacts'].get('consumer_checkpoint',{}).get('sha256')==m['post_condition']['consumer']['version']) if m['post_condition']['consumer']['controller']=='existing_forcepolicy' else None, model=m['model_id'],graph_id=b['model_id'],support_states=dict(counts),graph_outputs_on_training_support=dict(replayed),pi_status=b['models'][cid]['pi']['status'],
        searched_logs=logs,unreadable=unreadable,matched_recovery_witnesses=exact,same_graph_and_goal_but_incomplete_binding=same_model_unbound,other_learned_goal_passes=others,offset_models=models,
        historical_measured_label_ranges_by_datum=ranges,expert_witnesses=[],frontier_real_attempts=0,physical_interactions_by_this_audit=0,
        next_step='request documented expert recovery seed' if not exact else 'matched direct/policy frontier trials',
        caveat='Other consumer/model/budget results do not define the current frontier. Ranges above are measurements, not success bounds.')

