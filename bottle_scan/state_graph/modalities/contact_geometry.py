"""Contact-region geometry with explicit pixel and calibrated physical versions."""
import numpy as np


def calibrated_geometry(window, calibration):
    if not calibration or not calibration.get('calibration_id'):
        raise ValueError('Calibrated tactile geometry unavailable: finger-plane calibration required')
    meta=window.get('provenance',{})
    if calibration.get('sensor_version') != meta.get('sensor_version') or not calibration.get('sensor_version'):
        raise ValueError('Finger-plane calibration does not match the observation sensor version')
    pose=np.asarray(meta.get('measured_pose'),float)
    if pose.shape!=(4,4) or not np.isfinite(pose).all():
        raise ValueError('Calibrated tactile geometry requires the observed TCP pose')
    rot=pose[:3,:3]
    radial=rot@np.asarray(calibration['axes_tcp']['a'],float)
    radial[2]=0.
    if np.linalg.norm(radial)<1e-6:raise ValueError('Undefined horizontal approach direction')
    axes=np.stack((radial/np.linalg.norm(radial),[0.,0.,1.]))
    values=[]
    for side in ('left','right'):
        m=np.asarray(window['finger_masks'][side],bool)
        if m.ndim!=2 or np.count_nonzero(m)<3:raise ValueError('Insufficient contact pixels for '+side)
        cfg=calibration['fingers'][side]
        matrix=axes@rot@np.asarray(cfg['tcp_mm_per_px'],float)
        if abs(np.linalg.det(matrix))<1e-9:raise ValueError('Degenerate contact-plane projection')
        y,x=np.nonzero(m)
        cloud=np.stack((x,y),1)@matrix.T+axes@rot@np.asarray(cfg['origin_tcp_mm'],float)
        center=np.mean(cloud,axis=0)
        eig,dirs=np.linalg.eigh(np.cov(cloud.T,bias=True))
        angle=(np.arctan2(dirs[0,-1],dirs[1,-1])+np.pi/2)%np.pi-np.pi/2
        border=bool(np.any(m[0]) or np.any(m[-1]) or np.any(m[:,0]) or np.any(m[:,-1]))
        values.extend([*center,len(x)*abs(np.linalg.det(matrix)),angle,
                       4*np.sqrt(max(eig[-1],0)),4*np.sqrt(max(eig[0],0)),float(border)])
    return values


def contact_geometry(window,spec=None):
    obs=(spec or {}).get('observation',{})
    if obs.get('tac_version')=='calibrated_rh':
        from ..common import digest
        cal=obs.get('tac_calibration')
        values=calibrated_geometry(window,cal)
        return {'values':values,'version':'calibrated_rh:'+digest(cal),'evidence':{'calibration_id':cal['calibration_id'],
                'sensor_version':cal['sensor_version'],'order':['left','right'],
                'fields':['r_mm','h_mm','area_mm2','angle_rad','major_mm','minor_mm','edge']}}
    contacts=window['contacts']
    vectors=[np.asarray([side[k] for k in ('u_px','v_px','area_px')],float) for side in contacts]
    if len(vectors)!=2:raise ValueError('This modality requires two contact surfaces')
    return np.concatenate([(vectors[0]+vectors[1])/2,vectors[0]-vectors[1]]).tolist()
