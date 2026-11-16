"""Translate in one measured TCP frame. The surveyed TCP box must clear the full open tool."""
import math
import numpy as np
from bottle_scan.agentic_learning import _pose, _rotation_deg
from .contact_reference import FRAME


def validate_station(station):
    lo, hi = (np.asarray(station[k], float) for k in ('tcp_min_base_mm','tcp_max_base_mm'))
    if (not station.get('survey_id') or not station.get('full_open_tool_clearance_verified')
            or lo.shape != (3,) or hi.shape != (3,) or not np.isfinite([lo,hi]).all() or np.any(lo >= hi)):
        raise ValueError('A surveyed collision-free TCP box for the complete open gripper is required')
    if type(station.get('full_open_tool_clearance_verified')) is not bool:
        raise ValueError('Full open tool clearance needs explicit confirmation')
    if not station['allowed_rotations'] or not 0 < station['orientation_tolerance_deg'] <= 2:
        raise ValueError('Missing surveyed approach orientation range')
    for r in station['allowed_rotations']:
        T=np.eye(4); T[:3,:3]=r
        if _pose(T) is None:raise ValueError('Invalid surveyed approach orientation')
    return station


def support_error(pose, station):
    T=_pose(pose)
    if T is None:return 'invalid_tcp'
    p=T[:3,3]
    if np.any(p < station['tcp_min_base_mm']) or np.any(p > station['tcp_max_base_mm']):
        return 'outside_surveyed_tcp_box'
    if min(_rotation_deg(T[:3,:3],np.asarray(r)) for r in station['allowed_rotations']) > station['orientation_tolerance_deg']:
        return 'outside_surveyed_approach'
    return None


def command(delta, obs, reference_id, source):
    return {'kind':'tcp_translation_a','axis':'a','interface':'open_retreat_translate_reapproach_probe',
            'frame':FRAME,'frame_semantics_version':1,'u':[float(delta)],'delta_mm':float(delta),
            'action_delta_a_mm':float(delta),'observation_id':obs['observation_id'],
            'observation_ns':obs['timestamp_ns'],'reference_id':reference_id,'action_source':source}


def recover_once(backend,runner,action,row):
    ref=backend.contact_reference
    now=int(backend.hw.now())
    if (action['frame']!=FRAME or action['frame_semantics_version']!=1 or action['kind']!='tcp_translation_a'
            or action['reference_id']!=ref['target_definition_id'] or action['action_source']!=backend.action_source
            or action['observation_id']!=backend.current_observation_id
            or action['observation_ns']!=backend.current_observation_ns
            or not 0<=now-action['observation_ns']<=backend.local_settings['observation_max_age_s']*1e9):
        raise ValueError('Expired or incompatible local action')
    if not backend.executing or not backend.hw.lease.is_set():raise RuntimeError('No exclusive contact lease')
    delta=float(action['u'][0])
    if (len(action['u'])!=1 or delta!=action['action_delta_a_mm'] or not math.isfinite(delta)
            or abs(delta)>backend.cfg['servo']['max_delta_mm']):
        raise ValueError('Out-of-range local correction')
    sample=backend._sample()
    backend._require_supported(sample)
    if not np.all(np.asarray(sample['normal']) >= backend.hw.zero['contact_thresholds_sdk']):
        raise RuntimeError('Contact lost before correction')
    before=_pose(backend.hw.pose())
    if before is None:raise ValueError('Invalid starting TCP')
    recorded=backend.current_observation_pose
    if (np.linalg.norm(recorded[:3,3]-before[:3,3])>backend.cfg['motion']['pose_tolerance_mm']
            or not np.allclose(recorded[:3,:3],before[:3,:3],atol=.01)):
        raise RuntimeError('TCP changed since decision observation')
    axis=before[:3,:3] @ np.asarray(ref['protocol']['axes_tcp']['a'])
    retreat=backend.cfg['servo']['retreat_mm']
    def translated(mm):
        target=before.copy();target[:3,3]+=axis*mm
        return target
    path=[translated(d) for d in (-retreat,delta-retreat,delta)]
    if not hasattr(runner,'first_contact_position'):
        runner.first_contact_position=before[:3,3].copy()
    if np.linalg.norm(path[-1][:3,3]-runner.first_contact_position)>backend.cfg['motion']['max_grasp_offset_mm']:
        raise ValueError('Cumulative correction outside invocation motion envelope')
    for target in [before,*path]:
        reason=support_error(target,backend.station)
        if reason:raise ValueError(reason)
    # Convex surveyed volume, same rotation throughout all three straight segments.
    runner.action_start_pose=before
    row.update(before_pose=before.tolist(),axis_base=axis.tolist(),frame=FRAME,attempt_started_ns=now,
               commanded_target_pose=path[-1].tolist(),receipt={'applied':False})
    runner.release()
    for stage,target in zip(('REGRASP_RETREAT','REGRASP_CORRECT','REGRASP_APPROACH'),path):
        with runner.stage(stage):
            runner.move(target)
    status,probe=runner.touch()
    after=_pose(backend.hw.pose())
    if after is None:raise ValueError('Missing measured post-contact TCP')
    if (not np.allclose(after[:3,:3],before[:3,:3],atol=.01)
            or np.linalg.norm(after[:3,3]-path[-1][:3,3])>backend.cfg['motion']['pose_tolerance_mm']):
        raise RuntimeError('Measured recontact endpoint differs from local correction')
    # A completed approach alone cannot be used as a measured regrasp edge.
    post_obs=backend._observation(status,probe)
    if not post_obs['bilateral'] or not post_obs['force_stable'] or not post_obs['safety_ok']:
        raise RuntimeError('Recontact was not confirmed')
    actual=float((after[:3,3]-before[:3,3])@axis)
    row.update(after_pose=after.tolist(),applied_delta_mm=actual,receipt={
        'applied':True,'action_mm':actual,'actual_delta_a_mm':actual,'completed_ns':int(backend.hw.now()),
        'frame':FRAME,'reference_id':action['reference_id'],'observation_id':action['observation_id'],
        'source_pose_mm':before.tolist(),'recontact_pose_mm':after.tolist()})
    runner.current['regrasps']+=1
    backend.pending_recontact=(probe,post_obs)
    return after,status,probe,row
