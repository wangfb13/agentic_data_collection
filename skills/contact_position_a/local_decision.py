"""Inference at natural contact. Geometric offset and controller output stay distinct."""
import numpy as np
from .local_motion import command


def decide_local(backend, observation, row, graph, step):
    cfg=backend.local_settings
    source=backend.action_source
    report={'action_source':source,'offset_estimate':None}
    def unknown(reason):
        return 'Unknown',reason,None,report
    age=(backend.hw.now()-observation['timestamp_ns'])*1e-9
    if (not observation.get('safety_ok') or not observation.get('bilateral') or not observation.get('force_stable')
            or not 0<=age<=cfg['observation_max_age_s']):
        return unknown('invalid_or_stale_probe')
    ds=observation.get('descriptors') or []
    if len(ds)!=2 or any(not d.get('valid') or d.get('touches_edge') for d in ds):
        return unknown('missing_or_edge_contact')
    graph_result=graph.infer(row,readiness_only=(source=='offset_servo'))
    report['post_condition_prediction']=graph_result
    if source=='offset_servo':
        pred=backend.offset_policy.predict(ds)
        if not pred['valid']:
            report['offset_diagnostic']=pred
            return unknown(pred['reason'])
        offset=-pred['action_mm']
        lo,hi=backend.offset_policy.fit['trained_offset_range_mm']
        if not lo<=offset<=hi:return unknown('outside_observed_offset_range')
        report['offset_estimate']={'offset_a_mm':offset,'offset_c_mm':None,'offset_l_mm':None,
            'uncertainty_mm':pred['uncertainty_mm'],'model_id':backend.offset_policy.fit['model_id']}
    elif source!='graph_policy':
        raise ValueError('Unknown action source')
    state=graph_result['state']
    if state=='ready':return 'ready','observed_post_condition',None,report
    if state=='Unknown':return unknown(graph_result.get('reason','unknown_post_condition'))
    if step>=min(cfg['max_steps'],backend.cfg['servo']['max_regrasps']):return unknown('max_steps')
    limit=backend.cfg['servo']['max_delta_mm']
    if source=='offset_servo':
        limit=min(limit,backend.offset_policy.fit['settings']['max_action_mm'])
        if abs(offset)<=cfg['offset_deadband_mm']:return unknown('no_correction_and_goal_unconfirmed')
        delta=float(np.clip(-cfg['gain']*offset,-limit,limit))
    else:
        action=graph_result.get('action') or {}
        if state!='recoverable' or action.get('kind')!='tcp_translation_a':return unknown('graph_action_unavailable')
        delta=float(action['u'][0])
        if not np.isfinite(delta) or not 0<abs(delta)<=limit:return unknown('graph_action_outside_limits')
    cmd=command(delta,observation,backend.contact_reference['target_definition_id'],source)
    return 'recoverable',source,cmd,report
