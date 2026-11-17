"""Terminal sensor evidence and failure labels; SDK wrench is never implicitly measured in N."""
import numpy as np


def masks(dd,background,tactile):
    depth=tactile['depth_channel']
    output=[]
    for current,zero in zip(dd,background):
        delta=np.asarray(current,float)-np.asarray(zero,float)
        output.append((delta[...,depth] if delta.ndim==3 else delta)>tactile['dd_threshold'])
    return output


def centroids(dd,background,tactile):
    points=[]
    for mask in masks(dd,background,tactile):
        y,x=np.nonzero(mask)
        if len(x)<tactile['min_area_px']:
            return None
        points.append([float(x.mean()),float(y.mean())])
    return points


def automatic(pre,lift,hold,height_ok,held,params):
    samples=[*lift,*hold]
    if not samples:
        return None,'missing_lift_trace',{}
    if any(not s['bilateral'] for s in samples):
        return False,'contact_lost',{}
    if not height_ok or not held:
        return False,'lift_or_hold_incomplete',{}
    pts=[s.get('centroids') for s in hold]
    if not pts or any(p is None for p in pts):
        return None,'missing_hold_centroid',{}
    trace=np.asarray(pts,float)
    drift=float(np.max(np.linalg.norm(trace-trace[0],axis=2)))
    metrics={'max_finger_centroid_drift_px':drift}
    if drift>params['slip_threshold']:
        return False,'slip_during_hold',metrics
    calibration=params.get('load_calibration') or {}
    weight=params.get('expected_weight_n')
    if weight is None or not calibration.get('calibration_id'):
        return None,'missing_weight_or_load_calibration',metrics
    if 'finger_forces_to_tcp_newtons' in calibration:
        # Each 3x3 map includes the installed finger's sign, mounting rotation and
        # SDK-to-newton calibration. Project world vertical at this measured posture.
        maps=np.asarray(calibration['finger_forces_to_tcp_newtons'],float)
        rotation=np.asarray(params.get('rotation_base_tcp'),float)
        if maps.shape!=(2,3,3) or rotation.shape!=(3,3) or not np.isfinite(maps).all():
            return None,'missing_directional_load_calibration',metrics
        coeff=np.einsum('i,fij->fj',rotation[2,:],maps)
        coeff[:,params['normal_axis']]=0.
    else:
        coeff=np.asarray(calibration.get('upward_tangent_n_per_sdk'),float)
        if 'rotation_base_tcp' in params:
            calibrated=np.asarray(calibration.get('calibration_rotation_base_tcp'),float)
            if calibrated.shape!=(3,3) or 'orientation_tolerance_deg' not in calibration:
                return None,'missing_load_calibration_posture',metrics
            relative=calibrated.T@np.asarray(params['rotation_base_tcp'])
            angle=float(np.rad2deg(np.arccos(np.clip((relative.trace()-1)/2,-1,1))))
            if angle>calibration['orientation_tolerance_deg']:
                return None,'outside_calibrated_load_posture',metrics
    if coeff.shape!=(2,3) or not np.isfinite(coeff).all() or not (np.linalg.norm(coeff,axis=1)>0).all():
        raise ValueError('Load calibration requires two signed tangential projection rows in N/SDK')
    normal=params['normal_axis']
    if not np.allclose(coeff[:,normal],0.) or not np.isfinite(weight) or weight<=0:
        raise ValueError('Tangential load or bottle weight is invalid')
    baseline=np.asarray(pre['wrench_zeroed'])[:,:3]
    loads=[float(np.sum((np.asarray(s['wrench_zeroed'])[:,:3]-baseline)*coeff)) for s in hold]
    metrics.update(expected_weight_n=weight,calibration_id=calibration['calibration_id'],
                   min_supported_load_n=min(loads),max_supported_load_n=max(loads))
    if max(abs(v-weight) for v in loads)>params['load_tolerance']:
        return False,'load_weight_mismatch',metrics
    return True,None,metrics


class ForceBand:
    """Durations use distinct measured force frames. Paused samples do not extend a run."""
    def __init__(self,params):
        self.cfg=params
        self.entered=self.at_floor=None
        self.last=None
        self.ready=self.saturated=False

    def update(self,stamp,opening,normal,bilateral):
        if stamp==self.last:return
        if self.last is not None and stamp<self.last:raise ValueError('Nonmonotonic force frame')
        self.last=stamp
        c=self.cfg
        target,tol=c['fixed_force_sdk'],c['force_tolerance_sdk']
        in_band=bilateral and abs(normal-target)<=tol
        at_floor=opening<=c['fixed_closure_floor']+c['floor_tolerance'] and normal<target-tol
        self.entered=(self.entered if self.entered is not None else stamp) if in_band else None
        self.at_floor=(self.at_floor if self.at_floor is not None else stamp) if at_floor else None
        self.ready=in_band and stamp-self.entered>=c['stable_force_s']
        self.saturated=at_floor and stamp-self.at_floor>=c['floor_saturation_s']
