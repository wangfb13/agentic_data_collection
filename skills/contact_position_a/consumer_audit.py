"""Classify source consumer evidence without equating a planned order with execution."""
from .post import live_lift

STATES=('imported','not_executed','skipped','unmatched')


def classify(cid,condition,result,terminals):
    order=result.get('consumer_order') or []
    outcomes=result.get('consumer_outcomes') or {}
    actual=outcomes.get(cid)
    existing=[n for n in terminals if n.get('post',{}).get(cid,{}).get('evaluated')]
    evidence=dict(consumer=cid,slot=order.index(cid) if cid in order else None)
    def entry(status,reason,**fields):
        return dict(evidence,status=status,reason=reason,**fields)
    if actual is None:
        if existing:
            for node in existing:
                post=node['post'][cid]
                for evidence in post.get('evidence',[]):
                    if not isinstance(evidence,dict) or evidence.get('consumer_id')!=cid:continue
                    checked=live_lift({},dict(is_terminal=True,consumer_outcome=evidence),condition)
                    if checked.get('evaluated') and checked['success'] is post.get('success'):
                        return entry('imported','verified_terminal_evidence',node_ids=[node['node_id']],success=checked['success'])
            return entry('unmatched','terminal_without_verifiable_consumer_receipt')
        reason=(result.get('pairing') or {}).get('reason','')
        if cid in order[1:] and outcomes.get(order[0]) and (reason.startswith('second_skipped') or reason=='second_contact_invalid'):
            return entry('skipped',reason)
        return entry('not_executed',result.get('failure_cause') or result.get('attempt_reason') or
                     ('not_scheduled' if cid not in order else 'no_consumer_execution_record'))
    if not isinstance(actual,dict):return entry('unmatched','malformed_consumer_record')
    if actual.get('skipped') is True or actual.get('status')=='skipped':
        return entry('skipped',actual.get('reason') or 'explicit_consumer_skip')
    if actual.get('consumer_id')!=cid:
        return entry('unmatched','consumer_id_mismatch')
    if cid not in order or order.count(cid)!=1:return entry('unmatched','missing_or_ambiguous_order')
    slot=order.index(cid)
    if slot>=len(terminals):return entry('unmatched','missing_terminal_window')
    cond=condition['consumer']
    if actual.get('controller')!=cond.get('controller'):return entry('unmatched','controller_mismatch')
    if not actual.get('version') or actual['version']!=cond.get('version'):return entry('unmatched','version_mismatch')
    checked=live_lift({},dict(is_terminal=True,consumer_outcome=actual),condition)
    if not checked['evaluated']:return entry('unmatched',checked['evidence'][0]['reason'])
    recorded=terminals[slot].get('post',{}).get(cid,{})
    if recorded.get('evaluated') and recorded.get('success') is checked['success']:
        return entry('imported','matched_evaluated_terminal',node_ids=[terminals[slot]['node_id']],success=checked['success'])
    return entry('unmatched','terminal_post_missing_or_inconsistent')


def receipt(records):
    consumers=sorted({cid for t in records for cid in t['consumers']})
    counts={cid:dict.fromkeys(STATES,0) for cid in consumers}
    warnings=[]
    for t in records:
        for cid,entry in t['consumers'].items():
            counts[cid][entry['status']]+=1
            if entry['status']=='unmatched':
                warnings.append(dict(trial_id=t['trial_id'],consumer=cid,reason=entry['reason']))
    return dict(trials=records,counts=counts,warnings=warnings)
