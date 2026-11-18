"""Check continuity against the frozen contact metric at the witnessed TCP pose."""
import numpy as np
from bottle_scan.geometry import pose_error
from bottle_scan.state_graph.noise import measurement_distance


def evaluate(reference_row, current_row, bundle, geometry):
    radius=(bundle or {}).get('graph',{}).get('radii',{}).get('similar_radius')
    noise=(bundle or {}).get('noise') or {}
    report=dict(passed=False,status='metric_unavailable',similar_radius=radius,
                model_id=(bundle or {}).get('model_id'),noise_model_id=noise.get('model_id'))
    if not noise.get('active_modalities') or radius is None or not np.isfinite(radius):return report
    if reference_row is None or current_row is None:return dict(report,status='window_unavailable')
    p,a=pose_error(reference_row['provenance']['measured_pose'],current_row['provenance']['measured_pose'])
    report.update(position_error_mm=p,orientation_error_deg=a)
    if p>geometry['verification_pose_tolerance_mm'] or a>geometry['orientation_tolerance_deg']:
        return dict(report,status='pose_mismatch')
    if reference_row['extractor_versions']!=current_row['extractor_versions']:
        return dict(report,status='feature_mismatch')
    d=measurement_distance(reference_row,current_row,noise)
    return dict(report,status='passed' if d<=radius else 'distance_exceeded',passed=bool(d<=radius),
                distance=float(d) if np.isfinite(d) else None,window_id=current_row['node_id'])


def verify(io, answer):
    from . import campaign_ah
    from .p3_prepare import prepare_position
    from .p3_trial_v5 import model_row
    from bottle_scan.state_graph.campaign import atomic
    folder=io.directory/'reference_check'
    folder.mkdir(exist_ok=True)
    bundle=io.reference_bundle
    result=evaluate(None,None,bundle,io.spec['geometry'])
    old=io.reference.get('raw_witness')
    if old and result['status']!='metric_unavailable':
        target=np.asarray(io.reference['pose'])
        from .ah_geometry import measured,position_limits
        now=campaign_ah.current_pose(folder,'verification_start',io.cfg,campaign_ah.child)
        geom=io.spec['geometry']
        projected,residual=measured(target,now,now,io.cfg['axes_tcp']['a'],io.spec)
        limits=position_limits(io.spec)
        reachable=np.all(abs(projected)<=limits[:,1]-limits[:,0]+2*campaign_ah.POSITION_TOLERANCE_MM)
        if (not reachable or pose_error(target,now)[1]>geom['orientation_tolerance_deg'] or residual>geom['verification_pose_tolerance_mm']):
            result.update(status='witness_pose_not_reachable_by_translation')
        else:
            prepare_position(folder,target,[0,0],io.cfg,io.spec,campaign_ah.child,
                             station_z_bounds=io.z_limits,return_to_witness=True)
            index=-(max(geom['reference_probe_repetitions'],geom['recalibration_probe_repetitions'])+1)
            with io.contact(index) as raw:
                if raw and raw['provenance'].get('placement_continuity') is not False:
                    spec=bundle['spec']
                    rows=[]
                    for name,window in [('reference',old),('verification',raw)]:
                        dest=folder/name;dest.mkdir(exist_ok=True)
                        rows.append(model_row(window,window.get('trial_id',io.request['trial_id']),dest,spec))
                    result=evaluate(*rows,bundle,geom)
                    result['window_file']=str(dest/'windows.json')
    result.update(operator_answer=answer,reference_id=io.reference['placement_reference_id'],rewitnessed=False)
    io.reference_check=result
    atomic(folder/'check.json',result)
    return result['passed']
