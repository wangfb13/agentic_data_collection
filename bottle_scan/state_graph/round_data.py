"""Role-aware eligibility, complete episodes (including failed starts), and request binding."""
from collections import Counter
from pathlib import Path
from .common import digest, independent

ROLES = dict(exploration='exploration', direct_baseline='frontier_direct',
             policy_frontier='frontier_policy', expert_recovery='frontier_expert',
             development='dev_validation', independent='independent_validation')


def bind_request(request, state, spec, role):
    target = state['rule'].get('target_consumer', spec['active_post_condition']) if spec.get(
        'consumer', {}).get('target_from_rule') else spec['active_post_condition']
    if target not in {c['id'] for c in spec['post_conditions']}:
        raise ValueError('Unknown target consumer')
    request.update(schema='graph_trial_request_v5', data_role=ROLES[role],
                   operation=state.get('round_operation'),stage_token=state.get('stage_token'), round_index=state.get('round_index', 0), post_condition=target, target_consumer=target,
                   consumer_seed=int(digest([spec['loop']['seed'],request['trial_id']])[:8],16))
    model=state.get('frozen') if role=='independent' else state.get('model')
    if model:
        request.update({k:model[k] for k in ('base_spec_digest','effective_spec','effective_spec_digest') if k in model})
    if role == 'expert_recovery':
        request.update(mode='expert', expert_source=state.get('expert_source','scripted_coordinate'))
    elif role == 'policy_frontier':
        request.update(mode='learned')
    if role == 'direct_baseline' and request['actions']:
        raise ValueError('Direct comparison requires an action-free plan')


def outcomes(result):
    values = {}
    for row in result.get('nodes', []):
        if row.get('is_terminal'):
            for cid, p in row.get('post', {}).items():
                if p.get('evaluated') and type(p.get('success')) is bool:
                    values[cid] = p
    return values


def tag_result(result, request):
    target = request['target_consumer']
    post = outcomes(result).get(target,{})
    for row in result.get('nodes',[]):
        row.update(data_role=request['data_role'], round_index=request['round_index'],
                   target_consumer=target, controller_role=result.get('controller_role',request['role']))
        if request['data_role']=='frontier_expert':
            row['expert_success']=post.get('success') is True and bool(post.get('evidence')) and bool(result.get('expert_source') or request.get('expert_source')) and any(
                (r.get('u_receipt') or {}).get('applied') and r.get('u_executed') is not None for r in result.get('nodes',[]))
            if row['expert_success'] and (row.get('u_receipt') or {}).get('applied'):
                row['u_receipt']['source']='expert'
                row.setdefault('provenance',{})['action_source']='expert'
    result.update(data_role=request['data_role'],round_index=request['round_index'],target_consumer=target)


def training_rows(rows, spec, round_index):
    loop=spec.get('loop',{})
    return [r for r in rows if not independent(r) and
            (r.get('data_role')!='dev_validation' or
             (loop.get('dev_to_train',False) and r['round_index']<round_index)) and
            (r.get('data_role')!='frontier_expert' or
             (loop.get('expert_to_train',False) and r.get('expert_success') is True))]


def contributions(rows):
    return dict(Counter(r.get('data_role','exploration') for r in rows))


def episodes(state, spec, *, include_independent=False):
    from .campaign import read
    from .facts import coordinate_evidence
    for trial in state['trials']:
        req=trial['request']
        if trial['status']!='complete' or (not include_independent and req.get('post_freeze')):
            continue
        result=read(Path(trial['directory'])/'result.json')
        rows=result.get('nodes',[])
        first=next((r for r in rows if r.get('window_index')==0 and not r.get('previous_node_id')),None)
        coords={}
        if first and first.get('window_valid'):
            for key,v in spec['perturbation_variables'].items():
                if v['kind']=='continuous':
                    val=coordinate_evidence(first,key,spec)
                    if val is not None: coords[key]=val
        role={'direct_baseline':'direct','policy_frontier':'policy','development':'policy',
              'independent':'policy','expert_recovery':'expert'}.get(req['role'],'exploration')
        if role=='exploration' and result.get('controller_role')=='direct':role='direct'
        terminal=outcomes(result)
        target=req['post_condition']
        classification=result.get('failure_cause') or terminal.get(target,{}).get('reason') or result.get('attempt_reason')
        yield dict(trial_id=req['trial_id'],group_id=req['trial_id'], round_index=req.get('round_index'),
                   first_contact=first,role=role,controller_role=role,data_role=req.get('data_role'),
                   model_id=req.get('model_id'),target_consumer=target, mass=req['block'].get('mass_class'),
                   condition=req['block'],initial_coords=coords,coordinate_source='measured' if coords else None,
                   initial_coordinate_evidence=result.get('initial_coordinate_evidence') or (first or {}).get('provenance'), terminal_outcomes=terminal,
                   failure_cause=classification, expert_source=result.get('expert_source',req.get('expert_source')),
                   receipts=[r['u_receipt'] for r in rows if (r.get('u_receipt') or {}).get('applied')],
                   paired=result.get('paired') is True,consumer_order=result.get('consumer_order'),
                   status=result['status'],assessment=result.get('assessment'),cost=result.get('cost',{}))

