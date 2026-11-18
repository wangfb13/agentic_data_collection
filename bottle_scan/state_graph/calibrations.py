"""Declare missing calibration evidence before any real campaign dispatch."""
import numpy as np


def matrix(value,shape):
    try:a=np.asarray(value,float)
    except (TypeError,ValueError):return False
    return a.shape==shape and np.isfinite(a).all()


def missing(spec):
    c=spec.get('consumer',{}); obs=spec.get('observation',{}); items=[]
    if c.get('auto_l2'):
        weights=c.get('weight_newtons') or {}
        masses=spec.get('perturbation_variables',{}).get('mass_class',{}).get('levels',[]) or list(weights)
        if not masses or any(not isinstance(weights.get(k),(float,int)) or not np.isfinite(weights[k]) or weights[k]<=0 for k in masses):
            items.append('consumer.weight_newtons')
        cal=c.get('load_calibration') or {}
        if not (cal.get('calibration_id') and (matrix(cal.get('finger_forces_to_tcp_newtons'),(2,3,3)) or
                (matrix(cal.get('upward_tangent_n_per_sdk'),(2,3)) and matrix(cal.get('calibration_rotation_base_tcp'),(3,3)) and
                 isinstance(cal.get('orientation_tolerance_deg'),(float,int)) and cal['orientation_tolerance_deg']>=0))):
            items.append('consumer.load_calibration')
    if 'calibrated_rh' in (obs.get('tac_version'),obs.get('tac_version_requested')):
        cal=obs.get('tac_calibration') or {}
        fingers=cal.get('fingers') or {}
        if not (cal.get('calibration_id') and cal.get('sensor_version') and matrix(cal.get('axes_tcp',{}).get('a'),(3,)) and
                all(matrix(fingers.get(s,{}).get('tcp_mm_per_px'),(3,2)) and matrix(fingers.get(s,{}).get('origin_tcp_mm'),(3,)) for s in ('left','right'))):
            items.append('observation.tac_calibration')
    return items


def preflight(spec,real,allow):
    issues=missing(spec)
    if issues and real and not allow:
        raise ValueError('Missing calibration: '+', '.join(issues)+'; --allow-degraded explicitly permits degraded acquisition')
    return dict(missing=issues,real=bool(real),allow_degraded=bool(allow))
