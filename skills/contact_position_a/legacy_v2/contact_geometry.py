"""Contact geometry from recorded DD; no hardware imports or invented calibration."""
from __future__ import annotations
import math
import numpy as np
from bottle_scan.sensing import dd_descriptors

def contact_geometry(dd, reference, *, threshold=.022, depth_channel=0,
                     pixel_pitch_mm=None, root_edge=None, edge_width_px=3):
    """Weighted thresholded contact geometry. mm coordinates use image origin."""
    result = dd_descriptors(dd, reference, depth_channel=depth_channel,
        contact_threshold=threshold, pixel_pitch_mm=pixel_pitch_mm, edge_width_px=edge_width_px)
    raw=np.asarray(dd,float); ref=np.asarray(reference,float)
    depth=raw[...,depth_channel] if raw.ndim==3 else raw
    baseline=ref[...,depth_channel] if ref.ndim==3 else ref
    mask=(depth-baseline)>threshold
    h,w=mask.shape; yy,xx=np.nonzero(mask)
    distances=({"left":float(xx.min()),"right":float(w-1-xx.max()),
                "top":float(yy.min()),"bottom":float(h-1-yy.max())} if len(xx) else {})
    opposite={"left":"right","right":"left","top":"bottom","bottom":"top"}
    pitch=None
    if result["pixel_pitch_available"]:
        pitch=np.broadcast_to(np.asarray(pixel_pitch_mm,float),(2,))
    result.update(shape_hw=[h,w], bbox_xyxy_px=([int(xx.min()),int(yy.min()),int(xx.max()),int(yy.max())] if len(xx) else None),
        edge_distances_px=distances, root_edge=root_edge, root_edge_calibrated=root_edge in opposite,
        edge_dist_root_px=distances.get(root_edge), edge_dist_tip_px=distances.get(opposite.get(root_edge)),
        edge_dist_root_mm=None,edge_dist_tip_mm=None,principal_angle_deg=None,
        valid=bool(result["contact"]),u_px=None,v_px=None,u_mm=None,v_mm=None,
        coordinate_convention="u right, v down; sensor mm origin at image pixel (0,0)")
    if result["centroid_px"] is not None:result["u_px"],result["v_px"]=result["centroid_px"]
    if result["centroid_mm"] is not None:result["u_mm"],result["v_mm"]=result["centroid_mm"]
    if root_edge in distances:
        at_root=distances[root_edge]<edge_width_px;at_tip=distances[opposite[root_edge]]<edge_width_px
        result["edge_class"]="root_and_tip" if at_root and at_tip else "root" if at_root else "tip" if at_tip else "none"
        if pitch is not None:
            scale=pitch[0 if root_edge in ("left","right") else 1]
            result["edge_dist_root_mm"]=result["edge_dist_root_px"]*float(scale)
            result["edge_dist_tip_mm"]=result["edge_dist_tip_px"]*float(scale)
    else:result["edge_class"]=None
    moments=result.get("second_moment_px2")
    if moments is not None:
        eig,vec=np.linalg.eigh(np.asarray(moments,float))
        if eig[-1]>1e-15 and not np.isclose(eig[0],eig[1],rtol=1e-8,atol=1e-15):
            result["principal_angle_deg"]=(math.degrees(math.atan2(vec[1,-1],vec[0,-1]))+90)%180-90
    return result

def window_contact_geometry(frames,reference,**kwargs):
    """Median descriptor across independent frames, never reuse planned offsets."""
    desc=[contact_geometry(frame,reference,**kwargs) for frame in frames]
    if not desc or not all(d["valid"] for d in desc):
        raise ValueError("window_contains_empty_contact")
    result=dict(desc[len(desc)//2])
    for key in ("centroid_px","centroid_mm","u_px","v_px","u_mm","v_mm","area_px","area_mm2",
                "edge_dist_root_px","edge_dist_tip_px","edge_dist_root_mm","edge_dist_tip_mm",
                "bbox_xyxy_px","depth_weight_sum","edge_fraction"):
        vals=[d.get(key) for d in desc]
        result[key]=np.median(np.asarray(vals,float),axis=0).tolist() if all(v is not None for v in vals) else None
    # Conservative edge membership across the selected contact window.
    edge=[d.get("edge_class") for d in desc]
    result["edge_class"]=edge[0] if len(set(edge))==1 else "mixed"
    result["window_frames"]=len(desc)
    return result

def contact_mean_diff(left,right):
    keys=("u_px","v_px","area_px","edge_dist_root_px","edge_dist_tip_px",
          "u_mm","v_mm","area_mm2","edge_dist_root_mm","edge_dist_tip_mm")
    mean={};diff={}
    for key in keys:
        a,b=left.get(key),right.get(key)
        mean[key]=None if a is None or b is None else (a+b)/2
        diff[key]=None if a is None or b is None else a-b
    return mean,diff

def object_contact_position(sensor_contact,measured_tcp_mm,calibration):
    """Project to a calibrated bottle frame. T0 alone is not a bottle pose."""
    required=("T_tcp_sensor","T_base_object")
    if not calibration or any(k not in calibration for k in required):
        return {"h_mm":None,"theta_rad":None,"reason":"missing_tcp_sensor_or_object_pose"}
    if sensor_contact.get("u_mm") is None:
        return {"h_mm":None,"theta_rad":None,"reason":"missing_physical_pixel_pitch"}
    tcp=np.asarray(measured_tcp_mm,float);sensor=np.asarray(calibration["T_tcp_sensor"],float)
    obj=np.asarray(calibration["T_base_object"],float)
    if any(t.shape!=(4,4) or not np.isfinite(t).all() for t in (tcp,sensor,obj)):
        raise ValueError("invalid_contact_calibration_transform")
    # Sensor x/y coincide with calibrated image u/v axes, z is contact plane.
    p=np.array([sensor_contact["u_mm"],sensor_contact["v_mm"],calibration.get("plane_z_mm",0.),1.])
    po=np.linalg.solve(obj,tcp@sensor@p)
    return {"h_mm":float(po[2]),"theta_rad":float(np.arctan2(po[1],po[0])),
            "point_object_mm":po[:3].tolist(),"reason":"calibrated"}
