"""P3 preparation: zero requests stay here; measured perturbations use current axes.

The placement witness is only a measurement origin. Its orientation cannot gate
execution or orient a movement. Legacy fixed-P2 scans retain their own adapter.
"""
from pathlib import Path
import numpy as np
from bottle_scan.geometry import validate_transform
from bottle_scan.state_graph.campaign import atomic
from . import campaign_ah
from .ah_geometry import basis,coord_names,measured,position_allowed,position_limits


def prepare_position(folder,reference_pose,start_u,hw,spec,invoke,*,station_z_bounds=None):
    folder=Path(folder)
    requested=np.asarray(start_u,dtype=float)
    if not position_allowed(requested,spec):raise ValueError('Requested preparation outside declared envelope')
    reference=validate_transform(reference_pose)
    now=campaign_ah.current_pose(folder,'prepare_pose_0',hw,invoke)
    report={'schema':'p3_current_axes_preparation_v1','requested_coordinates_mm':dict(zip(coord_names(spec),requested.tolist())),
            'start_pose':now.tolist(),'old_reference_pose_used_for_execution_orientation':False,
            'reference_pose_role':'measurement_origin_only_not_deployment_target','movement_receipts':[],'command_frames':[]}
    path=folder/'preparation.json'
    # A direct baseline starts from the operator's current contact approach.
    # Neither a cached pose nor a requested zero is measured coordinate truth.
    if not np.any(requested):
        report.update(mode='current_pose_no_preparation_motion',actual_initial_coordinates=None,
                      note='Requested zero means no preparation motion. Contact coordinates are measured separately, or missing.')
        atomic(path,report)
        return report
    # Nonzero controlled perturbations refer to the current placement witness,
    # projected onto the axes at this command start. Preserve out-of-plane
    # position: never introduce an undeclared lateral or rotation command.
    axis=hw['axes_tcp']['a'];frame=now.copy();M=basis(frame,axis,spec)
    current,residual=measured(reference,now,frame,axis,spec)
    delta=requested-current
    limits=np.asarray(spec['task']['action_space']['max_step'],dtype=float)
    bounds=position_limits(spec)
    maximum_travel=(bounds[:,1]-bounds[:,0])+2*campaign_ah.POSITION_TOLERANCE_MM
    if np.any(np.abs(delta)>maximum_travel):
        raise ValueError('当前起点距本次测量参考过远；请确认瓶位或重新见证测量参考，禁止自动长距离返回')
    safe_z=(reference[2,3]+bounds[1]).tolist()
    if station_z_bounds is not None:
        safe_z=[max(safe_z[0],station_z_bounds[0]),min(safe_z[1],station_z_bounds[1])]
    report.update(mode='witnessed_target_in_current_axes',axes_base=M.tolist(),
                  initial_projection_mm=current.tolist(),out_of_plane_residual_mm=residual,
                  planned_delta_mm=delta.tolist(),measurement_label_not_assumed=True)
    atomic(path,report)
    leg=0
    for k in range(2):
        remaining=float(delta[k])
        while abs(remaining)>.25:
            amount=float(np.clip(remaining,-limits[k],limits[k]))
            command_basis=basis(now,axis,spec)
            goal=now[:3,3]+command_basis[:,k]*amount
            report['command_frames'].append({'pose':now.tolist(),'axes_base':command_basis.tolist(),'axis_index':k,'delta_mm':amount})
            if not safe_z[0]<=goal[2]<=safe_z[1]:raise ValueError('准备目标超出独立高度边界')
            receipt=campaign_ah.move(folder,'prepare_move_'+str(leg),['--xyz-mm',*goal],
                                      limits[k],safe_z,invoke)
            report['movement_receipts'].append(str(receipt));atomic(path,report)
            leg+=1
            now=campaign_ah.current_pose(folder,'prepare_pose_'+str(leg),hw,invoke)
            remaining-=amount
    report['end_pose']=now.tolist();atomic(path,report)
    return report
