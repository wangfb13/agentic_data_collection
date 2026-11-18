"""Snapshot committed v5 trials. Keep original roles and replay verified outcomes in memory."""
from copy import deepcopy
from pathlib import Path
from bottle_scan.state_graph.common import independent,digest
from bottle_scan.state_graph.campaign import read
from bottle_scan.state_graph.round_data import episodes
from .post import live_lift


def snapshot(parent,state,spec,source_cfg):
    rows=[r for r in read(parent/'seed_nodes.json') if not independent(r)]
    seen={r['node_id'] for r in rows}
    histories=[];origins=[];corrected=[]
    # Reuse a previous snapshot's episode evidence when chaining campaigns.
    if (parent/'seed_episodes.json').exists():
        histories.extend(e for e in read(parent/'seed_episodes.json') if not independent(e))
    expected=source_cfg['consumer']
    cond=deepcopy(next(c for c in spec['post_conditions'] if c['id']=='forcepolicy_lift'))
    cond['consumer']['controller']=expected['controller']
    for t in state['trials']:
        req=t['request']
        if t['status']!='complete' or req['role']=='independent' or independent(req) or req.get('post_freeze'):continue
        result=read(Path(t['directory'])/'result.json')
        if independent(result):continue
        source_rows=deepcopy(result['nodes'])
        if any(independent(r) for r in source_rows):continue
        cid=cond['id'];outcome=result.get('consumer_outcomes',{}).get(cid)
        terminals=[r for r in source_rows if r.get('is_terminal')]
        order=result.get('consumer_order') or []
        if outcome and cid in order and order.index(cid)<len(terminals):
            target=terminals[order.index(cid)]
            if (outcome.get('consumer_id')==cid and outcome.get('controller')==expected['controller']
                and outcome.get('version')==cond['consumer']['version']==expected['version']):
                post=live_lift({},dict(is_terminal=True,consumer_outcome=outcome),cond)
                if post['evaluated'] and not target['post'][cid]['evaluated']:
                    target['post'][cid]=post
                    corrected.append(target['node_id'])
        role=req['data_role']
        for r in source_rows:
            if r['node_id'] in seen:raise ValueError('Duplicate source node: '+r['node_id'])
            seen.add(r['node_id']);r['data_role']=role
            r.setdefault('provenance',{}).update(import_source=str(parent),source_round=r.get('round_index'))
            # Prior development can train only after its source evaluation has closed.
            r['round_index']=0
            if role=='dev_validation' and not (parent/'rounds'/('round_%04d'%req['round_index'])/'dev_validate.json').exists():
                r['source_dev_evaluation_open']=True
            rows.append(r)
        record=next(episodes({'trials':[t]},spec))
        record['round_index']=0
        record['terminal_outcomes']={c:p for n in terminals for c,p in n['post'].items() if p.get('evaluated')}
        histories.append(record)
        origins.append(dict(trial_id=t['trial_id'],directory=t['directory'],data_role=role,result_digest=digest(result)))
    return rows,origins,histories,corrected
