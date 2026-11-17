"""Reduce an attempt, including attempts which never produced a graph node."""
from .evidence import CONTACTS,ROLES,measured,checked_receipt

def post_result(raw,condition):
    from skills.contact_position_a.ready import fixed_ready,handoff_ready
    if not isinstance(raw,dict) or raw.get('physical_interaction') is False or raw.get('post_condition')!=condition['id']:return None
    outcome=raw.get('consumer_outcome') or {}
    expected=condition['consumer']
    if any(outcome.get(k)!=expected[k] for k in ('controller','version')):return None
    params=dict(outcome.get('params') or {})
    if expected['controller']=='existing_forcepolicy' and 'variant' not in expected['params'] and params.get('variant') is None:
        params.pop('variant',None)
    if params!=expected['params']:return None
    evaluate={'skills.contact_position_a.ready:fixed_ready':fixed_ready,
              'skills.contact_position_a.ready:handoff_ready':handoff_ready}.get(condition['evaluator'])
    if not evaluate:return None
    res=evaluate({},dict(is_terminal=True,consumer_outcome=outcome),condition)
    return res['success'] if res['evaluated'] else None

def summarize(a,m):
    e=a['events'];prep=e.get('reply/prepare',{});first=e.get('reply/touch',{}).get('observation',{}) if e.get('reply/touch',{}).get('evidence_verified') is not False else {}
    reason=a['reason'];role=ROLES[a['role']];commands=[];receipts=[];trajectory=[]
    observations=[first] if first else [];before=first;raw_cost=sum(k.startswith('io/') for k in e)
    phenomena=list(dict.fromkeys(e.get('phenomena',[])))
    moved=any(v.get('object_moved') is True or v.get('observation',{}).get('object_moved') is True
              for v in e.values() if isinstance(v,dict))
    def coord(label,phase,stamp):
        return None if moved or not stamp else measured(label,m['reference_digest'],m['target_definition_id'],phase=phase,stamp=stamp,axis=m['axis_tcp'])
    x=coord(first.get('measurement'),'first_contact',first.get('timestamp_ns'))
    xp=coord(prep.get('measurement'),'preparation',prep.get('timestamp_ns')) if prep.get('evidence_verified') is not False else None
    state=first.get('contact_state','not_observed')
    if state not in CONTACTS:state='not_observed'
    for k,v in e.items():
        if not k.startswith('io/action_'):continue
        cmd=v['command'];commands.append(cmd)
        after=e.get('reply/'+k[3:],{});obs=after.get('observation',{});receipt=after.get('receipt')
        if receipt:
            try:
                actual=checked_receipt({'command':cmd,'receipt':receipt},before,obs,m)
            except (KeyError,ValueError,TypeError):phenomena.append('invalid_receipt')
            else:
                receipts.append(receipt);trajectory.append({'before':before,'command':cmd,'actual_mm':actual,'receipt':receipt,'after':obs})
        if obs:observations.append(obs)
        before=obs
    human=any(isinstance(v,dict) and (v.get('human_intervention') is True or v.get('observation',{}).get('human_intervention') is True) for v in e.values())
    post=e.get('reply/post',{}); consumer=post_result(post,m['post_condition']) if post.get('evidence_verified') is not False else None
    if any(o.get('rejected_measurement') for o in [prep,*observations]):phenomena.append('coordinate_evidence_invalid')
    if 'technical_fault' in e or 'preflight_fault' in e:phenomena.append('technical_fault')
    if state!='valid_bilateral':phenomena.append(state)
    if any(o.get('edge') is True for o in observations):phenomena.append('edge')
    if moved:phenomena.append('object_moved')
    if human:phenomena.append('human_intervention')
    if consumer is False:phenomena.append('consumer_failed')
    if reason and reason!='finished':phenomena.append(reason)
    recovered=('recovery_end' in e and len(commands)==len(receipts) and (state=='valid_bilateral' or (role=='expert' and state in (m.get('expert') or {}).get('permitted_start_contacts',[]))) and not moved)
    success=(reason=='finished' and consumer is True and recovered and not (human and role!='expert')
             and (role!='expert' or bool(trajectory)))
    initial=e.get('decision/0',{});truth=first.get('independent_initial_evidence') or {}
    supported_truth=truth.get('state') if truth.get('source') in ('matched_direct_trial','independent_review') and truth.get('raw_source') else None
    witness=bool(success and role=='expert' and m.get('expert') and trajectory)
    return dict(episode_id=a['id'],attempt_id=a['id'],frontier_round_id=a['rid'],phase=a['phase'],controller_role=role,
        model_id=m['model_id'],action_source=m['action_source'] if role=='policy' else role,
        controller=m['controller'],condition=m['condition'],reference_digest=m['reference_digest'],target_definition_id=m['target_definition_id'],
        protocol=m['protocol'],post_condition=m['post_condition'],requested_offset_mm=a['x'],preparation_offset_mm=xp,first_contact_offset_mm=x,
        preparation_measurement=prep.get('measurement'),first_contact_measurement=first.get('measurement'),label_invalidated=moved,
        independent_trial_unit=prep.get('independent_trial_unit'),initial_model_offset_mm=initial.get('offset_a_mm') if role=='policy' else None,initial_model_state=initial.get('state') if role=='policy' else None,
        initial_controller_decision=initial.get('state'),first_contact_label_valid=x is not None,preparation_label_valid=xp is not None,initial_independent_state=supported_truth,
        independent_initial_evidence=truth,first_contact_state=state,first_contact_observation_id=first.get('id'),
        first_valid_node=next((o.get('node_id') for o in observations if o.get('node_id') and o.get('contact_state')=='valid_bilateral'),None),
        terminal_observation_id=before.get('id'),steps_attempted=len(commands),steps_measured=len(receipts),trajectory=trajectory,
        recovery_completed=recovered,consumer_tested=consumer is not None,consumer_outcome=consumer,success=success,
        task_outcome=post.get('consumer_outcome',{}).get('L3') if post.get('task_evidence') else None,
        autonomous_recovery_witness=bool(success and role=='policy' and initial.get('state')=='NeedAction' and trajectory),
        expert_witness=witness,expert=m.get('expert') if role=='expert' else None,demonstration_id=a['id'] if witness else None,
        recovery_seconds=e.get('recovery_end',{}).get('recovery_seconds'),technical_error=e.get('technical_fault') or e.get('preflight_fault'),human_intervention=human,expert_requested=role=='expert',termination_reason=reason,phenomena=sorted(set(phenomena)),
        verified_causes=[],raw_evidence=[v['raw_source'] for v in e.values() if isinstance(v,dict) and v.get('raw_source')],
        started_ns=a['start'],ended_ns=a['end'],charged_dispatches=raw_cost)

