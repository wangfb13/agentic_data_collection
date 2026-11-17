"""Run exactly one controlled attempt. The director selects cells, never commands."""
import time
from .evidence import CONTACTS,checked_receipt,num,require
from .episodes import post_result
from .search import validate_choice

def execute(db,rid,choice,station,policy,*,clock=time.monotonic):
    rr=db.round(rid);m=rr['manifest'];validate_choice(choice,rr,db.attempts(rid))
    aid=db.begin(rid,choice['role'],choice['x'],op=choice['op'],proposal=choice)
    a=db.get(aid);role=a['role'];lim=m['limits']
    def finish(reason):
        db.finish(aid,reason);return aid
    def call(key,operation,**payload):
        req=dict(attempt_id=aid,command_id=aid+'/'+key,operation=operation,**payload)
        db.dispatch(aid,key,req);result=station.perform(req);db.reply(aid,key,result)
        recorded=db.get(aid)['events']['reply/'+key]
        require(recorded.get('evidence_verified') is not False,'unverifiable device evidence')
        if operation in ('touch','correct_retouch','post'):require(recorded['physical_interaction'],'device reports no interaction')
        return recorded
    try:
        station.verify(m)
        require(station.environment==m['environment'],'station/environment mismatch')
        if role=='policy_frontier':require(policy and policy.identity==m['model_id'] and m['policy_executable'],'frozen policy unavailable')
        if role=='expert_recovery':require(m.get('expert') and station.expert_identity==m['expert'],'expert unavailable')
    except Exception as exc:
        db.event(aid,'preflight_fault',{'detail':str(exc)});return finish('technical_preflight')
    try:
        p=call('prepare','prepare',requested_offset_mm=a['x'],max_prepare_mm=lim['prepare_mm'],condition=m['condition'])
        if m['environment']=='physical' and p.get('condition')!=m['condition']:return finish('preparation_condition_mismatch')
        if p.get('prepared') is not True:return finish(p.get('reason') or 'preparation_failed')
        start=clock();deadline=start+lim['seconds'];used=0.;step=0
        obs=call('touch','touch',phase='first_contact',deadline_seconds=lim['seconds'])['observation']
        while True:
            require(obs.get('id') and obs['contact_state'] in CONTACTS and type(obs['timestamp_ns']) is int,'unidentified contact')
            if clock()>deadline:return finish('time_exhausted')
            if obs.get('object_moved'):return finish('object_moved')
            if obs.get('human_intervention') and role!='expert_recovery':return finish('human_intervention')
            if obs['contact_state']!='valid_bilateral' and (role!='expert_recovery' or obs['contact_state'] not in m['expert'].get('permitted_start_contacts',[])):
                return finish(obs['contact_state'])
            if role=='direct_baseline':decision={'state':'Ready','source':'direct_test_without_position_action'}
            elif role=='policy_frontier':decision=policy.decide(obs,step)
            else:decision=station.expert_decision(obs,step,m['expert'])
            db.event(aid,'decision/'+str(step),decision)
            if decision['state']=='Ready':break
            if decision['state']!='NeedAction':return finish(decision.get('reason') or 'model_rejection')
            if step>=lim['max_steps']:return finish('steps_exhausted')
            cmd=decision['command'];delta=cmd['u'][0]
            require(len(cmd['u'])==1 and num(delta) and 0<abs(delta)<=lim['step_mm'] and
                    cmd['kind']=='tcp_translation_a' and cmd['frame']=='tcp_at_action_start' and
                    cmd.get('frame_semantics_version')==1 and cmd['reference_id']==m['target_definition_id'] and
                    cmd['observation_id']==obs['id'] and cmd['observation_ns']==obs['timestamp_ns'] and
                    cmd['action_source']==(m['action_source'] if role=='policy_frontier' else 'expert'),'invalid controller command')
            if used+abs(delta)>lim['total_mm']:return finish('cumulative_displacement_exhausted')
            remain=deadline-clock()
            if remain<=0:return finish('time_exhausted')
            reply=call('action_'+str(step),'correct_retouch',command=cmd,remaining_seconds=remain,
                       cumulative_remaining_mm=lim['total_mm']-used)
            if not reply.get('receipt',{}).get('applied'):return finish(reply.get('reason') or 'action_or_retouch_failed')
            after=reply['observation'];net=checked_receipt({'command':cmd,'receipt':reply['receipt']},obs,after,m)
            used+=max(abs(delta),abs(net));step+=1;obs=after
            if used>lim['total_mm'] or abs(net)>lim['step_mm']:return finish('measured_motion_limit_exceeded')
        if clock()>deadline:return finish('time_exhausted')
        db.event(aid,'recovery_end',{'observation_id':obs['id'],'steps':step,'recovery_seconds':clock()-start})
        post=call('post','post',observation_id=obs['id'],post_condition=m['post_condition'])
        result=post_result(post,m['post_condition'])
        return finish('finished' if result is True else 'consumer_failed' if result is False else 'consumer_not_verified')
    except Exception as exc:
        db.event(aid,'technical_fault',{'detail':str(exc),'type':type(exc).__name__})
        e=db.get(aid)['events']
        if any(k.startswith('io/') and 'reply/'+k[3:] not in e for k in e):return aid
        return finish('technical_execution')

def reconcile(db,aid,station_records,reason):
    """Never dispatch. Accept only replies to existing intents. Keep cost even with missing receipts."""
    require(reason.strip(),'record reconciliation evidence or inspection reason')
    for item in station_records:db.reply(aid,item['command_id'].split('/',1)[1],item)
    a=db.get(aid);m=db.round(a['rid'])['manifest'];result=post_result(a['events'].get('reply/post'),m['post_condition']) if all(v.get('evidence_verified') is not False for k,v in a['events'].items() if k.startswith('reply/')) else None
    final='finished' if 'recovery_end' in a['events'] and result is True else 'consumer_failed' if result is False else 'interrupted:'+reason
    db.finish(aid,final);return aid

