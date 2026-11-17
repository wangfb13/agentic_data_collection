"""Content-addressed measurement records. No estimator, plan or terminal-window fallback."""
import hashlib
import json
import math
from numbers import Real
import numpy as np

SIGN = 'hand_displacement_from_standard_along_positive_tool_a_mm'
SOURCES = {'independent_offset_measurement', 'measured_local_standard_tcp'}
CONTACTS = {'no_contact','unilateral','unstable_bilateral','valid_bilateral'}
ROLES = {'direct_baseline':'direct','policy_frontier':'policy','expert_recovery':'expert'}

def encode(x):
    return json.dumps(x, sort_keys=True, separators=(',',':'), ensure_ascii=False, allow_nan=False)

def sha(x):
    return hashlib.sha256(encode(x).encode()).hexdigest()

def num(x):
    return isinstance(x,Real) and not isinstance(x,bool) and math.isfinite(x)

def require(ok, msg):
    if not ok: raise ValueError(msg)

def pose(raw):
    t=np.asarray(raw,dtype=float)
    require(t.shape==(4,4) and np.isfinite(t).all() and np.allclose(t[3],[0,0,0,1]) and
            np.allclose(t[:3,:3].T@t[:3,:3],np.eye(3),atol=1e-5) and np.linalg.det(t[:3,:3])>0,'invalid measured TCP')
    return t

def measured(label, reference, target, *, phase=None, stamp=None, axis=None, invalid=()):
    """Record contains raw measurement/standard witness, rather than just a free-form ID."""
    if not isinstance(label,dict):return None
    r=label.get('record')
    try:
        if not isinstance(r,dict) or label['evidence_id']!=sha(r):return None
        if not (reference and target and r['reference_digest']==reference and r['target_definition_id']==target
                and r['source'] in SOURCES and r['unit']=='mm' and r['axis_sign']==SIGN
                and r['valid'] is True and r['object_moved'] is False and r['raw_source']
                and r['phase'] in ('preparation','first_contact') and type(r['timestamp_ns']) is int and r['timestamp_ns']>0
                and num(r['uncertainty_mm']) and r['uncertainty_mm']>=0
                and (phase is None or phase==r['phase']) and (stamp is None or stamp==r['timestamp_ns'])):return None
        if r.get('basis_id') in invalid:return None
        if r['source']=='independent_offset_measurement':
            if not r['instrument_id'] or not r['calibration_id']:return None
            val=r['reading_mm']
        else:
            w=r['standard_witness']
            if not (r['basis_id']==sha(w) and w['reference_digest']==reference and w['accepted_standard'] is True
                    and w['raw_source'] and w['timestamp_ns']<=r['timestamp_ns'] and w['placement_id']==r['placement_id']):return None
            a=np.asarray(w['a_tcp'],float)
            if axis is None or not np.allclose(a,axis) or not np.isclose(np.linalg.norm(a),1.):return None
            t0=pose(w['tcp_mm']); t=pose(r['tcp_mm'])
            if not np.allclose(t[:3,:3],t0[:3,:3],atol=.01):return None
            delta=t[:3,3]-t0[:3,3];direction=t0[:3,:3]@a;val=float(delta@direction)
            if np.linalg.norm(delta-val*direction)>2.:return None
        return float(val) if num(val) else None
    except (KeyError,ValueError,TypeError,OverflowError):return None

def label_record(record):
    return {'record':record,'evidence_id':sha(record)}

def checked_receipt(step,before,after,m):
    q=step['command'];r=step['receipt']; ref=m['target_definition_id']
    require(q['kind']=='tcp_translation_a' and q['frame']=='tcp_at_action_start' and q['reference_id']==ref and
            q['observation_id']==before['id'] and r.get('observation_id')==before['id'] and
            r.get('frame')==q['frame'] and r.get('reference_id')==ref and r['applied'] is True,'unmatched action receipt')
    a,b=pose(r['source_pose_mm']),pose(r['recontact_pose_mm'])
    require(np.linalg.norm(a[:3,3]-pose(before['tcp_mm'])[:3,3])<=m.get('pose_tolerance_mm',.05) and np.linalg.norm(b[:3,3]-pose(after['tcp_mm'])[:3,3])<=m.get('pose_tolerance_mm',.05)
            and np.allclose(a[:3,:3],b[:3,:3],atol=.01),'receipt disagrees with observation')
    actual=float((b[:3,3]-a[:3,3])@(a[:3,:3]@np.array(m['axis_tcp'])))
    require(num(r['actual_delta_a_mm']) and abs(actual-r['actual_delta_a_mm'])<.05 and
            before['timestamp_ns']<min(r['completed_ns'],after['timestamp_ns']) and abs(r['completed_ns']-after['timestamp_ns'])<=1_000_000_000 and r['raw_source'],'unmeasured action')
    return actual

