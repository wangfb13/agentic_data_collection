"""Explicit v2 reinterpretation of historical evidence; raw sessions stay read-only."""
from copy import deepcopy
from datetime import datetime, timezone, timedelta
import hashlib
import json
from pathlib import Path
import numpy as np


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()


def frozen_independent(row):
    return bool(row.get('frozen_independent') or row.get('evaluation_scope')=='post_freeze_independent' or
                row.get('campaign_frozen_at_ns') is not None and row.get('independent_B'))


def prepare_nodes(rows,config=None):
    cfg=config or {};out=[];cache={}
    allowed=set(cfg.get('historical_node_ids',[]))
    manifest=cfg.get('historical_manifest')
    if manifest:
        data=json.loads(Path(manifest).read_text());allowed.update(data['node_ids'])
    periods=cfg.get('periods',[])
    for raw in rows:
        row=deepcopy(raw)
        if row.get('provenance_schema')=='symbolic_node_provenance_v2':out.append(row);continue
        row['original_evaluation_only']=row.get('evaluation_only',False)
        row['original_source_role']=row.get('source_role')
        if row['node_id'] in allowed and not frozen_independent(row):
            row.update(evaluation_only=False,source_role='historical_development',evaluation_scope='historical_group_cv',historical_adoption=True)
        elif frozen_independent(row):row.update(evaluation_only=True,evaluation_scope='post_freeze_independent')
        source=row.get('provenance',{}).get('session_meta_path')
        if source not in cache:
            cache[source]=json.loads(Path(source).read_text()) if source and Path(source).exists() else {}
        meta=cache[source];c=meta.get('config') or {}
        stamp=row.get('trial_started_unix_ns') or row.get('session_created_unix_ns')
        day=datetime.fromtimestamp(stamp/1e9,timezone(timedelta(hours=8))).strftime('%Y-%m-%d') if stamp else row.get('acquisition_day','unknown')
        row['acquisition_day']=day
        selected=next((p for p in periods if p.get('session_contains') and p['session_contains'] in row.get('session','')),None)
        if selected is None:selected=next((p for p in periods if p.get('date_from','9999')<=day<=p.get('date_to','0000')),None)
        row['time_period']=selected['id'] if selected else (row.get('time_period') or day)
        row['period_evidence']=selected.get('evidence') if selected else (row.get('period_evidence') or 'Explicit node period or calendar day from recorded UTC start; physical sensor revision unrecorded.')
        row['sensor_version']=row.get('sensor_version') or meta.get('sensor_version') or (meta.get('hardware_identity') or {}).get('sensor_version')
        row['sensor_version_known']=row['sensor_version'] is not None
        row['T0_mm']=meta.get('T0',row.get('T0_mm'))
        axes=c.get('axes_tcp') or {};T=np.asarray(row['T0_mm'],dtype=float)
        row['a_axis_base']=(T[:3,:3]@np.asarray(axes['a'],float)).tolist() if T.shape==(4,4) and axes.get('a') is not None else row.get('a_axis_base')
        row['axes_tcp']=axes or row.get('axes_tcp')
        row['T0_id']=digest({'T0':row['T0_mm'],'a_axis_base':row['a_axis_base']})
        row['a_sign_convention']='+a = recorded T0 rotation times axes_tcp.a, approach from pregrasp to T0; -a = retreat; not image left/right'
        protocol=deepcopy(row.get('protocol') or {})
        controller=row.get('downstream_controller') or protocol.get('downstream_controller') or 'e1e2_squeeze'
        protocol['downstream_controller']=controller
        if row.get('downstream_model_id'):protocol['downstream_model_id']=row['downstream_model_id']
        row['downstream_controller']=controller
        row['downstream_controller_evidence']='explicit node/controller field' if controller!='e1e2_squeeze' else 'Historical E1/E2 A/B collector SQUEEZE stages; not deployment ForcePolicy'
        row['original_protocol_id']=row.get('protocol_id');row['protocol']=protocol;row['protocol_id']=digest(protocol)[:20]
        row['provenance_schema']='symbolic_node_provenance_v2'
        out.append(row)
    return out


def same_reference(a,b,config=None):
    """Cross-session kinematics require a common physical reference, not only labels."""
    cfg=config or {}
    if any(a.get(k)!=b.get(k) for k in ('object_id','protocol_id','time_period')):return False
    if a.get('sensor_version') and b.get('sensor_version') and a['sensor_version']!=b['sensor_version']:return False
    if a.get('session')==b.get('session'):return True
    if not cfg.get('cross_session_kinematic',True):return False
    if not a.get('acquisition_day') or a.get('acquisition_day')!=b.get('acquisition_day'):return False
    A=np.asarray(a.get('T0_mm'),float);B=np.asarray(b.get('T0_mm'),float)
    if A.shape!=(4,4) or B.shape!=(4,4) or not np.isfinite(A).all() or not np.isfinite(B).all():return False
    if np.linalg.norm(A[:3,3]-B[:3,3])>cfg.get('T0_position_tolerance_mm',.1):return False
    cosine=np.clip((np.trace(A[:3,:3].T@B[:3,:3])-1)/2,-1,1)
    if np.degrees(np.arccos(cosine))>cfg.get('T0_rotation_tolerance_deg',.1):return False
    aa=np.asarray(a.get('a_axis_base'),float);bb=np.asarray(b.get('a_axis_base'),float)
    return aa.shape==(3,) and bb.shape==(3,) and np.allclose(aa,bb,atol=1e-4,rtol=0)
