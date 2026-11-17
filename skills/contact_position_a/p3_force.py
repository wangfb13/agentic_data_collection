"""Drive the deployed force predictor + its right-hand controller, with a measured lift test."""
import ast
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import time
from types import SimpleNamespace

import numpy as np

from . import lift_evidence
from bottle_scan.hardware_arm import pose_error
from bottle_scan.state_graph.campaign import atomic
from bottle_scan.state_graph.common import clean, digest

STAR = Path('/home/dm/DM_das/starvla-feat-sparky_x5_starvla')
LAUNCHER = STAR/'shfile/infer_261011_right10_forcepolicy_impl.sh'
SLOTS = ('right_left','right_right')


def controller_source():
    # Read the import-free class only; never source the launcher or run its record loop.
    shell = LAUNCHER.read_text()
    lines = shell[shell.index('class RightNormalForceController:'):].splitlines()
    end = next(i for i,line in enumerate(lines[1:],1) if line and not line.startswith((' ','#')))
    source = '\n'.join(lines[:end])
    tree = ast.parse(source)
    if len(tree.body)!=1 or not isinstance(tree.body[0],ast.ClassDef):
        raise ValueError('Unexpected ForcePolicy controller format')
    defaults = dict(re.findall(r'^([A-Z_0-9]+)="\$\{[A-Z_0-9]+:-([\w.+-]+)\}"$',shell,re.M))
    defaults.update(RIGHT_FORCE_CONTROL='true',RIGHT_FORCE_POLICY_MODE='control',RIGHT_FORCE_POSITION_MIN='0.330')
    return source, {k:v for k,v in defaults.items() if k in source}


def identity(checkpoint):
    source,settings=controller_source()
    def checksum(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    impl={n:checksum(STAR/'src/forcepolicy'/f'{n}.py') for n in ('runtime','inference','models','control_bridge')}
    info=dict(checkpoint=str(Path(checkpoint).resolve()), weights=checksum(checkpoint),
              controller=digest(source), settings=settings, implementation=impl,
              protocol='measured_contact_lift40_hold2_v1', lift_mm=40., hold_s=2., grip_timeout_s=60.,
              force_input='software_zeroed_x5_wrench',runtime_contact_threshold=.20)
    return {**info,'version':'P3_FP_'+digest(info)[:20]}


def load(expected):
    if identity(expected['checkpoint']) != expected:
        raise ValueError('Current consumer differs from this campaign version')
    if str(STAR/'src') not in sys.path:
        sys.path.insert(0,str(STAR/'src'))
    from forcepolicy.inference import ForcePolicyPredictor
    from forcepolicy.control_bridge import ForceTargetLimiter
    source,settings=controller_source()
    namespace=dict(math=math,time=time,os=SimpleNamespace(environ=settings),
                   model_close_handoff=True,startup_legacy_force_gate=False,wrench_max_age_ms=200,
                   ForceTargetLimiter=ForceTargetLimiter)
    exec(compile(source,str(LAUNCHER),'exec'),namespace)
    return ForcePolicyPredictor(expected['checkpoint'],device='cpu'),namespace['RightNormalForceController']()


def samples(f):
    return {slot:SimpleNamespace(fid=f['sequence'][i],received_monotonic=f['stamps'][i]*1e-9,
                                 source_timestamp=None,wrench=f['wrench_zeroed'][i])
            for i,slot in enumerate(SLOTS)}


def sampling_worker(hw,expected,predictor):
    from forcepolicy.runtime import RuntimeConfig,ForcePolicyRuntime
    return ForcePolicyRuntime(predictor,lambda:samples(hw.force_sample()),
                              RuntimeConfig(contact_threshold=expected['runtime_contact_threshold']))


def consume(hw,expected,directory,controller,worker,ask=input,options=None):
    controller.reset()
    directory=Path(directory)
    hw.hold()
    hw.set_context('FORCEPOLICY')
    hw.begin_record(directory/'force_streams')
    try:
        with (directory/'force_trace.jsonl').open('x',buffering=1) as trace:
            def tick(regulate=True):
                hw.check()
                f=hw.force_sample()
                if any(v is None for v in f['sequence']):
                    raise RuntimeError('Missing X5 source frame identity')
                g=hw.grip
                pair=samples(f)
                if abs(f['stamps'][0]-f['stamps'][1])>50_000_000:
                    raise RuntimeError('Raw finger wrench time skew')
                ref=worker.latest_reference()
                with g.control_lock:
                    hw.check()
                    actual=g.fresh()['normalized']
                    command,info=controller.update(latch={'latched':True,'command_before_safety_limits':actual},
                            raw_observation={'main_gripper_right':actual},samples=pair,force_reference=ref)
                    if regulate:
                        if g.mode!='hold' or not g.armed or g.overforce:
                            raise RuntimeError('Gripper control ownership lost')
                        if command<.330-1e-9 or not math.isfinite(command):
                            raise RuntimeError('ForcePolicy command outside closure limits')
                        g._publish(command,'p3_forcepolicy')
                info.update(force_sequences=f['sequence'],wrench_zeroed=f['wrench_zeroed'])
                info['bilateral']=all(n>t for n,t in zip(f['normal'],f['contact_thresholds']))
                if options and options.get('auto_l2'):
                    info['centroids']=lift_evidence.centroids(f['raw']['dd'],hw.dd_reference,hw.cfg['tactile'])
                trace.write(json.dumps(clean({'t_ns':time.monotonic_ns(),'stage':hw.stage,**info}),allow_nan=False)+'\n')
                return info
            return lift_test(hw,{**expected,**(options or {})},directory,tick,ask=ask)
    finally:
        hw.end_record()


def lift_test(hw,params,directory,tick,*,ask=input,clock=time.monotonic):
    """Keep the consumer active in flight. AIR receipt prevents unconditional release on faults."""
    path=directory/'consumer.json'
    air=directory/'airborne.json'
    out=dict(controller=params.get('controller','existing_forcepolicy'),version=params['version'],params={'closure_floor':params.get('closure_floor',.33)},
             L1=None,L2=None,evidence=[str(directory/'force_trace.jsonl')],lift_mm=params['lift_mm'],hold_s=params['hold_s'])
    def save(reason):
        out['reason']=reason
        atomic(path,out)
        return out
    until=clock()+params['grip_timeout_s']
    while True:
        info=tick()
        if info['reference_valid'] and info['tcp_motion_ready'] and info['force_ready'] and info['bilateral']:
            break
        if info.get('floor_saturated'):
            out.update(L1=False,failure_cause='closure_floor_reached',contact_lost=not info['bilateral'])
            return save('closure_floor_reached')
        if clock()>=until:
            if not info['reference_valid']:
                raise RuntimeError('ForcePolicy prediction unavailable at handoff: '+str(info['reason']))
            out['L1']=False
            if params.get('auto_l2') or params.get('dual_paired'):
                out['contact_lost']=not info['bilateral']
            if params.get('fixed_force_failure_causes') and out['controller']=='fixed_force':
                out['failure_cause']='target_force_not_reached_other'
            return save('force_not_ready_by_deadline')
        hw.wait(.02)
    pre=info
    lift_samples,hold_samples=[],[]
    auto=params.get("auto_l2",False)
    monitor=None
    axes=[i for i in range(3) if i!=hw.cfg['force']['normal_axis']] if 'lower' in hw.cfg else []
    def tangential(info):
        return np.asarray(info['wrench_zeroed'])[:,:3][:,axes].reshape(-1)
    before=tangential(info) if axes else None
    base=hw.pose()
    goal=base.copy()
    goal[2,3]+=params['lift_mm']
    m=hw.cfg['motion']
    if not 0<params['lift_mm']<=m['max_lift_mm']:
        raise ValueError('Lift exceeds hardware envelope')
    speed=min(m['near_speed_mm_s'],m['free_speed_mm_s'],10.)
    out.update(L1=True,start_pose=base.tolist())
    save('force_ready')
    atomic(air,{'possibly_airborne':True,'start_pose':base.tolist()})
    def segment(target,up,contact_ok):
        deadline=clock()+max(10.,params['lift_mm']/speed*3)
        hw.move(target,speed)
        while True:
            info=tick(regulate=contact_ok)
            if up and auto:lift_samples.append(info)
            if contact_ok and not info['reference_valid']:
                raise RuntimeError('ForcePolicy reference lost during lift/return')
            if contact_ok and not (info['bilateral'] and info['tcp_motion_ready']):
                hw.hold()
                contact_ok=False
                if up:
                    hw.stop_arm()
                    return False
            if not up and monitor and monitor({**info,'pose':hw.pose()}):
                hw.stop_arm()
                out['unload_detected']=True
                return contact_ok
            if hw.motion_done():
                translation,angle=pose_error(target,hw.pose())
                if translation<=m['pose_tolerance_mm'] and angle<=1:
                    return contact_ok
            if clock()>=deadline:
                raise TimeoutError('Lift/return motion feedback timeout')
            hw.wait(.02)
    try:
        hw.ownership()
        hw.set_context('LIFT')
        ok=segment(goal,True,True)
        out['lift_pose']=hw.pose().tolist()
        height_ok=abs(out['lift_pose'][2][3]-goal[2,3])<=m['pose_tolerance_mm']
        hw.set_context('HOLD')
        lifted=tangential(tick())-before if ok and axes else None
        loads=[]
        previous=None
        start=clock()
        while ok and clock()-start<params['hold_s']:
            info=tick()
            if not info['reference_valid']:
                raise RuntimeError('ForcePolicy reference lost during hold')
            if auto:hold_samples.append(info)
            ok=info['bilateral'] and info['tcp_motion_ready'] and not hw.hold_drop_seen
            if ok and axes and tuple(info['force_sequences'])!=previous:
                loads.append(tangential(info)-before)
                previous=tuple(info['force_sequences'])
            hw.wait(.02)
        out['measured_hold_s']=clock()-start
        if not ok:
            hw.hold()
        if ok and axes:
            from bottle_scan.e1e2 import unload_monitor
            out['unload_proxy'],monitor=unload_monitor(hw.cfg['lower'],axes,before,lifted,loads,False,hw.pose()[2,3])
        hw.set_context('LOWER')
        return_ok=segment(base,False,ok)
        out['return_pose']=hw.pose().tolist()
        supported=bool(auto and out.get('unload_detected') and return_ok and
                       abs(out['return_pose'][2][3]-base[2,3])<=m['pose_tolerance_mm'])
        if not supported and ask('下降已停止。确认瓶底已由桌面支撑，输入 TABLE 才松爪：').strip().upper()!='TABLE':
            raise RuntimeError('Bottle support unconfirmed; holding for operator')
        atomic(air,{'possibly_airborne':False,'return_pose':out['return_pose']})
        # TCP motion alone cannot prove that the bottle rose with the hand.
        if auto:
            observed,cause,metrics=lift_evidence.automatic(pre,lift_samples,hold_samples,height_ok,
                    ok and out['measured_hold_s']>=params['hold_s'],params)
            classify=out['controller']!='fixed_force' or params.get('fixed_force_failure_causes')
            out.update(L2=observed,automatic_L2=observed,failure_cause=cause if classify else None,metrics=metrics,return_contact=return_ok,
                       contact_lost=any(not x['bilateral'] for x in lift_samples+hold_samples),
                       object_moved=cause=='slip_during_hold',audit=None,label_conflict=False)
            if params.get('audit_selected'):
                answer=ask('本次为随机抽查。瓶底确实离桌并保持到结束？y / n / u：').strip().lower()
                audit={'y':True,'n':False}.get(answer)
                out.update(audit=audit,label_conflict=observed is not None and audit is not None and observed!=audit)
            return save(cause or 'automatic_lift_hold')
        seen=ask('刚才瓶底是否确实离桌并保持满 2 秒？y / n / u（看不清）：').strip().lower()
        out.update(observed_lift_hold={'y':True,'n':False}.get(seen),return_contact=return_ok)
        out['L2']=False if not ok or seen=='n' else True if seen=='y' else None
        return save('lift_hold_reviewed')
    except BaseException:
        try:
            hw.stop_arm()
        finally:
            if not hw.overforce:
                hw.hold()
        raise


def consume_fixed(hw,condition,params,directory,ask=input):
    """Use the existing gripper force loop and the same lift/hold/return protocol."""
    directory=Path(directory)
    if hw.cfg['gripper']['min_normalized']!=params['fixed_closure_floor']:
        raise ValueError('Fixed-force floor differs from active hardware controller')
    band=lift_evidence.ForceBand(params)
    hw.set_context('FIXED_FORCE')
    hw.begin_record(directory/'force_streams')
    try:
        with (directory/'force_trace.jsonl').open('x',buffering=1) as trace:
            hw.force(params['fixed_force_sdk'],params['force_tolerance_sdk'])
            def tick(regulate=True):
                hw.check()
                f=hw.force_sample()
                opening=hw.grip.fresh()['normalized']
                bilateral=bool(np.all(np.asarray(f['normal'])>f['contact_thresholds']))
                band.update(min(f['stamps'])*1.e-9,opening,float(np.mean(f['normal'])),bilateral)
                info=dict(reference_valid=True,force_ready=band.ready,tcp_motion_ready=bilateral,
                          floor_saturated=band.saturated and params.get('fixed_force_failure_causes',False),
                          force_sequences=f['sequence'],wrench_zeroed=f['wrench_zeroed'],bilateral=bilateral,
                          opening=opening,normal=f['normal'],stamps_ns=f['stamps'])
                if params.get('auto_l2'):
                    info['centroids']=lift_evidence.centroids(f['raw']['dd'],hw.dd_reference,hw.cfg['tactile'])
                trace.write(json.dumps(clean({'stage':hw.stage,**info}),allow_nan=False)+'\n')
                return info
            consumer=condition['consumer']
            return lift_test(hw,{**params,**consumer['params'],'controller':consumer['controller'],
                                 'version':consumer['version']},directory,tick,ask=ask)
    finally:
        hw.end_record()
