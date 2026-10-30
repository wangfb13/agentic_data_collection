"""Calibration calculations and simulated rehearsal orchestration.

No ROS imports. Every commanded leg uses Runner; physical wizard is disabled.
The fitted response gives equivalent TCP a/l mm, not measured gel millimetres.
"""
from __future__ import annotations
import copy
from pathlib import Path
import numpy as np
from .geometry import pose_offset, grasp_pose, translate_local
from .runtime import Acquisition, SafetyFault
from .session import atomic_json, plain
from .sensing import TemplateGuard
from .window_descriptors import enrich_trial_windows

SIDES = ('left', 'right')


def fit_response(records):
    xy = np.array([r['actual_al_mm'] for r in records], float)
    X = np.column_stack([np.ones(len(xy)), xy])
    if X.shape[0] < 5 or np.linalg.matrix_rank(X) != 3 or not np.isfinite(X).all():
        raise ValueError('方向标定需要 T0、a/l ±5 mm 的有效测量')
    models = {}
    for side in SIDES:
        uv = np.asarray([r['centroids_px'][side] for r in records], float)
        if uv.shape != (len(X), 2) or not np.isfinite(uv).all():
            raise ValueError(side+': 无有效双指质心，不能推断方向')
        coeff, *_ = np.linalg.lstsq(X, uv, rcond=None)
        J = coeff[1:].T
        singular = np.linalg.svd(J, compute_uv=False)
        error = float(np.sqrt(np.mean((X @ coeff-uv)**2)))
        if singular[-1] < .05 or singular[0]/singular[-1] > 20 or error > .5:
            raise ValueError(side+': 质心响应过弱、病态或非线性，转人工处理')
        models[side] = {'J_px_per_tcp_mm':J.tolist(), 'intercept_px':coeff[0].tolist(),
            'inverse_tcp_mm_per_px':np.linalg.inv(J).tolist(), 'rms_px':error,
            'condition':float(singular[0]/singular[-1]),
            'unit':'equivalent TCP a/l mm; not a measured gel pixel pitch'}
    return models


def center_correction(centroids, image_centers, models, max_step_mm=3., tolerance_mm=1.):
    if not 0 < max_step_mm <= 3 or not 0 < tolerance_mm <= 1:
        raise ValueError('居中每步不得超过 3 mm，停止容差不得超过 1 mm')
    matrices, residuals, mm = [], [], {}
    for side in SIDES:
        J = np.asarray(models[side]['J_px_per_tcp_mm'], float)
        residual = np.asarray(centroids[side], float)-np.asarray(image_centers[side], float)
        if J.shape != (2,2) or residual.shape != (2,) or not np.isfinite(residual).all():
            raise ValueError('有效质心及响应矩阵缺失')
        mm[side] = np.linalg.solve(J,residual).tolist()
        matrices.append(J);residuals.append(residual)
    delta, *_ = np.linalg.lstsq(np.vstack(matrices),-np.concatenate(residuals),rcond=None)
    norm = np.linalg.norm(delta)
    if norm > max_step_mm: delta *= max_step_mm/norm
    return {'delta_al_mm':delta.tolist(),'residual_equivalent_al_mm':mm,
            'converged':all(np.linalg.norm(mm[s]) <= tolerance_mm for s in SIDES)}


def probe_at(runner, target, directory):
    """Release before moving, then CLOSE/PROBE/RELEASE with full native recording."""
    directory = Path(directory);directory.mkdir(parents=True,exist_ok=False)
    old_meta = runner.meta
    meta = {'trial_id':directory.name,'task':'preparation','stages':[],
        'T_target':plain(target),'auto':{'contact_times_ns':[None,None],
        'contact_widths_mm':[None,None],'exceptions':[]},'operator':{},'status':'running','simulated':bool(runner.backend.simulated)}
    runner.meta = meta
    runner.backend.set_trial({'T_target':plain(target),'delta_acl_mm':[0,0,0]})
    runner.backend.drain_samples()
    runner.recorder = Acquisition(runner.backend,directory)
    try:
        with runner.phase('RELEASE'):
            runner._command_force(0.);runner.ensure_open()
        with runner.phase('APPROACH'):
            runner.move(target,noncontact=True)
        if not runner.close_contact() or not runner.probe():
            raise SafetyFault('轻触没有稳定双侧接触；不能标定/居中')
        actual = np.asarray(runner.backend.snapshot()['tcp']).copy()
        with runner.phase('RELEASE'):
            runner._command_force(0.);runner.ensure_open()
        meta['status']='completed'
    except BaseException:
        meta['status']='aborted';raise
    finally:
        try:
            runner.recorder.close()
        finally:
            runner.recorder=None
            atomic_json(directory/'meta.json',meta)
            runner.meta=old_meta
    enrich_trial_windows(meta,directory/'streams.h5',runner.cfg,runner.reference)
    centers={}
    for side in SIDES:
        window=meta['descriptor_windows']['PROBE'].get(side,{})
        median=window.get('median',{})
        if window.get('selected_frames',0)<3 or window.get('probe_not_settled'):
            raise SafetyFault('稳定 PROBE 窗口不足；未伪造质心')
        centers[side]=[median.get('centroid_u_px'),median.get('centroid_v_px')]
        if not np.isfinite(np.asarray(centers[side],float)).all():
            raise SafetyFault('PROBE 质心无效')
    atomic_json(directory/'meta.json',meta)
    return {'actual_tcp':actual.tolist(),'centroids_px':centers,'raw_directory':str(directory)}


def calibrate_and_center(runner, session, ui):
    if runner.backend.simulated:runner.backend.calibrating=True
    origin=session.T0.copy()
    records=[];base=session.path/'calibration';base.mkdir(exist_ok=True)
    try:
        for i,delta in enumerate(([0,0],[5,0],[-5,0],[0,5],[0,-5])):
            target=pose_offset(origin,[delta[0],0,delta[1]],axes=runner.cfg['axes'])
            record=probe_at(runner,target,base/f'direction_{i}')
            local=origin[:3,:3].T@(np.asarray(record['actual_tcp'])[:3,3]-origin[:3,3])
            basis=np.column_stack([runner.cfg['axes'][a] for a in ('a','l')])
            record['actual_al_mm']=(basis.T@local).tolist();records.append(record)
        models=fit_response(records)
        atomic_json(base/'sensor_response.json',{'models':models,'samples':records,'simulated':bool(runner.backend.simulated)})
        ui.say('  方向标定：T0、a±5、l±5 mm 完成；双指响应矩阵有效。')
        target=origin.copy();history=[]
        while True:
            for index in range(5):
                measured=probe_at(runner,target,base/f'center_{len(history):02d}')
                shape=runner.cfg['tactile']['dd_shape']
                image_centers={s:[(shape[1]-1)/2,(shape[0]-1)/2] for s in SIDES}
                correction=center_correction(measured['centroids_px'],image_centers,models)
                history.append({**measured,**correction,'round':len(history)+1})
                atomic_json(base/'centering.json',{'models':models,'history':history,'simulated':bool(runner.backend.simulated)})
                if correction['converged']:break
                if index<4:
                    da,dl=correction['delta_al_mm']
                    target=pose_offset(np.asarray(measured['actual_tcp']),[da,0,dl],axes=runner.cfg['axes'])
            if correction['converged']:break
            ui.say('  自动居中达到 5 轮仍未达标；夹爪已松开、未提瓶。')
            key=ui.ask('请手动调整，或输入 a+/a-/l+/l- 作 1 mm 开爪微调；回车重新检查，q 退出：')
            if key=='q':raise SafetyFault('居中失败，操作员退出；停在未提瓶状态')
            target=np.asarray(runner.backend.snapshot()['tcp']).copy()
            if key in ('a+','a-','l+','l-'):
                delta=[0.,0.,0.];delta[0 if key[0]=='a' else 2]=1. if key[1]=='+' else -1.
                target=pose_offset(target,delta,axes=runner.cfg['axes'])
        new=np.asarray(measured['actual_tcp'])
        fixed=origin[:3,3]+origin[:3,:3]@np.asarray(runner.cfg['geometry']['bottle']['center_offset_mm'])
        runner.cfg['geometry']['bottle']['center_offset_mm']=(new[:3,:3].T@(fixed-new[:3,3])).tolist()
        session.T0=new;session.cfg['T0']=new.tolist()
        session.meta.update(T0=new.tolist(),B=grasp_pose(new,runner.cfg).tolist(),B_legacy=new.tolist(),
            wizard_sensor_response=models,wizard_centering=history[-1],
            tune_geometry_reexpression={'old_T0':origin.tolist(),'bottle_axis_origin_base_mm':fixed.tolist()})
        session.save_meta()
        return correction
    finally:
        if runner.backend.simulated:runner.backend.calibrating=False


def calibrate_right_camera(runner, images, roi, directory):
    """References remain fixed; calibrate camera sensitivity with +2 mm and returns."""
    guard=TemplateGuard(images,{'right_wrist':roi},checked_cameras=['right_wrist'],
        min_score=runner.cfg['check']['min_score'],search_radius_px=runner.cfg['check']['search_radius_px'])
    initial=np.asarray(runner.backend.snapshot()['tcp']).copy()
    records={};directory=Path(directory);directory.mkdir(parents=True,exist_ok=False)
    previous_meta=runner.meta
    runner.meta={'trial_id':'camera_calibration','task':'preparation','stages':[],'auto':{'exceptions':[]},'simulated':bool(runner.backend.simulated),'status':'running'}
    recorder=Acquisition(runner.backend,directory);runner.recorder=recorder
    writer=recorder.writer
    try:
        def capture():
            samples=runner.backend.image_samples()
            for name,sample in samples.items():
                from .park_guard import image_pose_metadata
                attrs=dict(sample.metadata or {})
                attrs.update(image_pose_metadata(runner.backend.snapshot(),runner.session,camera_stamp_ns=sample.monotonic_ns))
                writer.jpg.append(name,sample.data,sample.monotonic_ns,sample.device_timestamp,attrs=attrs)
            result=guard._match('right_wrist',samples['right_wrist'].data)
            result['measured_tcp_mm']=np.asarray(runner.backend.snapshot()['tcp']).tolist()
            return result
        with runner.phase('CHECK'):
            noise=[]
            for _ in range(3):
                runner.wait(.2,noncontact=True);noise.append(capture())
        if any(not n['valid'] for n in noise):raise ValueError('参考 ROI 匹配质量低，请重新框选')
        noise_max=max(n['offset_px'] for n in noise)
        for axis in ('a','l','c'):
            with runner.phase('CHECK'):
                runner.ensure_open()
                runner.move(translate_local(initial,2.,axis,runner.cfg['axes']),noncontact=True)
                runner.wait(.3,noncontact=True);moved=capture()
                runner.move(initial,noncontact=True);runner.wait(.3,noncontact=True);returned=capture()
            valid=moved['valid'] and returned['valid'] and returned['offset_px']<=max(1.,3*noise_max)
            threshold=moved.get('offset_px',0)*.5
            observable=valid and moved.get('offset_px',0)>=2 and threshold>=max(.5,3*noise_max)
            records[axis]={'moved':moved,'returned':returned,'observable':bool(observable),
                'threshold_px':threshold if observable else None,'commanded_tcp_mm':2.}
        valid=[v['threshold_px'] for v in records.values() if v['observable']]
        if not valid:raise ValueError('所有方向像素位移都不可可靠观测；禁止自动设零阈值')
        guard.thresholds_px={'right_wrist':min(valid)}
        guard.calibration={'method':'robot_camera_motion_2mm_each_axis_and_return','directions':records,
            'stationary_noise_px':noise_max,'threshold_policy':'minimum reliable directional threshold',
            'unobservable_directions':[k for k,v in records.items() if not v['observable']],
            'limitation':'camera translation sensitivity is not a complete 3D bottle displacement calibration'}
        writer.json('calibration.json',guard.calibration)
        runner.meta['status']='completed'
    except BaseException as exc:
        runner.meta.update(status='aborted',error=str(exc))
        raise
    finally:
        try:recorder.close()
        finally:
            runner.recorder=None
            atomic_json(directory/'meta.json',runner.meta)
            runner.meta=previous_meta
    return guard
