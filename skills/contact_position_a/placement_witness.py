"""A witnessed placement is a source of supervised coordinates, independent of policy output."""
import numpy as np
from bottle_scan.geometry import validate_transform,pose_error
from bottle_scan.state_graph.common import digest
from .ah_geometry import measured,coord_names


def witness(window,placement,reference_digest):
    if not window or not window.get('preconditions',{}).get('bilateral_contact'):
        raise ValueError('A valid standard bilateral contact must be witnessed')
    p=window['provenance']
    record=dict(placement=placement,reference_digest=reference_digest,
                mass_class=window['attrs']['mass_class'],source_window=window['node_id'],
                source_file=p['source_file'],timestamp_ns=window['t_ns'],
                pose=validate_transform(p['measured_pose']).tolist(),valid=True)
    return {**record,'placement_reference_id':digest(record)}


def label(window,ref,axis,spec):
    p=window['provenance']
    p.update(coordinate_schema='measured_placement_relative',coordinate_source=None,placement_reference_valid=False,
             placement_reference_id=(ref or {}).get('placement_reference_id'))
    window['plan_coords']={}
    if not ref or not ref['valid'] or ref['reference_digest']!=p.get('reference_digest') or ref['mass_class']!=window['attrs']['mass_class']:
        return
    pose=p['measured_pose']
    u,residual=measured(ref['pose'],pose,ref['pose'],axis,spec)
    p['placement_residual_mm']=residual
    geom=spec['geometry']
    if residual>geom['coordinate_residual_mm'] or pose_error(ref['pose'],pose)[1]>geom['orientation_tolerance_deg']:
        return
    window['plan_coords']=dict(zip(coord_names(spec),map(float,u)))
    p.update(coordinate_source='measured',placement_reference_valid=True,placement_reference_pose=ref['pose'],calibration_epoch=ref['placement_reference_id'],
             repeat_context=[ref['placement_reference_id'],ref['reference_digest']],coordinate_evidence_id=window['node_id'])
