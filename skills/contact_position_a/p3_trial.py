"""Single P3 trial. The same observe/decide/consume sequence is used with real and fake I/O."""
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
import time

import joblib
import numpy as np

from bottle_scan.hardware_arm import pose_error
from bottle_scan.state_graph import campaign
from bottle_scan.state_graph.extract_nodes import extract_nodes
from bottle_scan.state_graph.noise import measurement_distance
from bottle_scan.state_graph.skill import SymbolicRecoverySkill
from . import ah_evidence, campaign_ah, p3_force
from .ah_geometry import KIND, validate_cell


def observe_rows(windows,trial,directory,spec):
    path=directory/'windows.json'
    campaign.atomic(path,[dict(trial_id=trial,windows=windows)])
    rows,_=extract_nodes([path],spec)
    if len(rows)!=len(windows):
        raise ValueError('Contact window did not pass the configured P3 extractors')
    return rows


def direct_evidence(row,bundle,source_rows,mass):
    """Independent measured no-action comparators in this candidate's training snapshot."""
    cid=bundle['spec']['active_post_condition']
    ids={anchor['node_id'] for anchor in bundle['support']}
    matched=[]
    for old in source_rows:
        if old['node_id'] not in ids or old.get('evaluation_only') or not old['provenance'].get('direct_A'):
            continue
        post=old['post'][cid]
        if (old['attrs']['mass_class']!=mass or old['extractor_versions']!=row['extractor_versions']
                or not post['evaluated'] or old['group_id']==row['group_id']):
            continue
        distance=measurement_distance(row,old,bundle['noise'])
        if distance<=min(1.,bundle['spec']['runtime']['support_distance']):
            matched.append((old['group_id'],post['success'],float(distance)))
    signs={m[1] for m in matched}
    label=None
    if len({m[0] for m in matched})>=2 and len(signs)==1:
        label='Ready' if signs=={True} else 'NeedAction'
    return {'initial_truth':label,'kind':'unanimous_matched_direct_A','comparators':matched}


def learned_action(inference,bundle,spec):
    """Constrain the ridge suggestion to a witnessed, executable one-axis alternative."""
    prediction=np.asarray(inference['policy']['u'],float)
    sources={s[2] for s in inference['support'] if s[1]=='recoverable'}
    options=[]
    for edge in bundle['graph']['edges']:
        if edge['kind']!='observed' or edge['source'] not in sources or not edge.get('executable',True):
            continue
        if bundle['graph']['reachability'][spec['active_post_condition']][edge['target']]['status']!='ready':
            continue
        u=edge.get('u')
        if u is None:
            continue
        action={'kind':KIND,'u':u}
        try:
            validate_cell({'actions':[action]},spec,'',{})
        except ValueError:
            continue
        options.append((float(np.linalg.norm(np.asarray(u)-prediction)),edge['edge_id'],action))
    if not options:
        return None
    _,edge_id,action=min(options,key=lambda item:item[:2])
    return {**action,'edge_id':edge_id}


class RealIO:
    def __init__(self,req,directory,spec,cfg):
        self.directory=directory
        self.spec=spec
        self.cfg,self.identity=campaign_ah.check_assets()
        self.consumer=cfg['consumer']
        self.assets=p3_force.load(self.consumer)
        self.mass=req['block']['mass_class']
        self.group='P3_FP_'+req['trial_id']
        self.z_limits=cfg['execution'].get('table_tcp_z_mm')
        self.hw=None

    def confirm(self,request):
        print(f"{request['trial_id']} {request['role']}，请放{'轻' if self.mass=='light' else '重'}白瓶。"
              f" {request['mode']}，计划动作 {request['actions']}。",flush=True)
        if input('瓶已放稳，右爪全开空载并对准瓶身、遥操已停。确认修正路径、向上 40 mm 及返回路径畅通，输入 MOVE：').strip()!='MOVE':
            raise campaign.RetryTrial('未授权本次动作')

    @contextmanager
    def contact(self,index):
        from bottle_scan.e1e2 import Runner
        from bottle_scan.e1e2_hardware import Hardware
        from bottle_scan.geometry import validate_transform

        class Capture(Hardware):
            def probe_window(self,since):
                value=super().probe_window(since)
                if value:
                    self.window_pose=self.arm.snapshot()['arms']['right']
                return value

        folder=self.directory/('initial' if index==0 else 'reprobe')
        folder.mkdir()
        path=folder/'observation.json'
        hw=self.hw=Capture(self.cfg)
        active=False
        worker=None
        try:
            hw.connect()
            hw.begin_record(folder/'record')
            hw.set_context('ZERO',self.group)
            zero=hw.zero_unloaded()
            worker=self.worker=p3_force.sampling_worker(hw,self.consumer,self.assets[0])
            worker.start()
            before=hw.pose()
            runner=Runner(SimpleNamespace(out=folder,cond='A',step=False),self.cfg,None,hw)
            active=True
            status,p=runner.touch()
            hw.hold()
            row=dict(status=status,probe=p,unloaded_zero=zero,window_pose=getattr(hw,'window_pose',None),
                     touch_report=runner.last_touch_report,before_pose=before,after_pose=hw.pose(),
                     reference_digest=self.identity[0]['reference_digest'],finger_serials=self.identity[2])
            campaign.atomic(path,row)
            hw.end_record()
            raw=None
            if status=='contact' and p and p.get('valid'):
                fb=row['window_pose']
                if fb['tcp_frame']!='right_gripper_base_link' or hw.arm.snapshot()['base_frame']!='base_link':
                    raise ValueError('Unexpected tool/base frames')
                pose=validate_transform(fb['T_base_tcp_mm'])
                raw=ah_evidence.captured_window(path,self.identity,None,self.group,index,{},self.mass,'P3',pose,row)
                # Only an unmoved bottle supports a contact-to-contact transition or direct reference.
                if input('从本次首触至现在，瓶一直在原处直立、未被推移？y / n：').strip().lower() not in ('y','yes'):
                    raw=None
            yield raw
        finally:
            try:
                air=self.directory/'airborne.json'
                if active and (hw.overforce or not (air.exists() and campaign.read(air)['possibly_airborne'])):
                    hw.recovery_open()
            finally:
                try:
                    if worker:
                        worker.stop()
                finally:
                    hw.close()
                    self.hw=None

    def move(self,action,first):
        u=action['u']
        k=0 if u[0] else 1
        now=campaign_ah.current_pose(self.directory,'action_pose',self.cfg,campaign_ah.child)
        err,angle=pose_error(first['provenance']['measured_pose'],now)
        if err>1 or angle>1:
            raise RuntimeError('首触和开爪后实测 TCP 不一致')
        return campaign_ah.move(self.directory,'action_move',['--relative-at-confirm','--frame',('tcp','base')[k],
            '--direction','z+' if u[k]>0 else 'z-','--distance-mm',abs(u[k])],
            self.spec['task']['action_space']['max_step'][k],self.z_limits,campaign_ah.child)

    def link(self,first,last,receipt,u):
        ah_evidence.link(first,last,[receipt],self.cfg['axes_tcp']['a'],command=u)

    def consume(self):
        return p3_force.consume(self.hw,self.consumer,self.directory,self.assets[1],self.worker)


def run_trial(request,directory,spec,cfg,io,*,sources=None):
    directory=Path(directory)
    trial=request['trial_id']
    role=request['role']
    learned=role in ('development','independent')
    if (role not in ('exploration','development','independent') or request['post_condition']!=spec['active_post_condition']
            or request['mode']!=('learned' if learned else 'planned') or learned and request['actions']):
        raise ValueError('Wrong P3 execution contract')
    validate_cell({'actions':request['actions'],'settings':request['settings'],'blocks':[request['block']]},spec,'',{})
    skill=None
    truth={'initial_truth':None}
    if learned:
        bundle=joblib.load(request['model_bundle'])
        if (bundle['model_id']!=request['model_id'] or bundle['spec']!=spec
                or role=='independent' and request['frozen_model_id']!=request['model_id']):
            raise ValueError('Candidate or frozen specification mismatch')
        skill=SymbolicRecoverySkill(bundle)
        if sources is None:
            base=Path(request['campaign'])
            sources=campaign.all_rows(base,campaign.read(base/'state.json'))
    start=time.monotonic()
    windows=[]
    decisions=[]
    action=receipt=outcome=None
    exit_declared=False
    reason=None
    io.confirm(request)
    for index in range(2):
        with io.contact(index) as raw:
            if raw is None:
                reason='invalid_initial_contact' if index==0 else 'invalid_recontact'
                break
            windows.append(raw)
            if index:
                io.link(windows[0],raw,receipt,action['u'])
                if not raw.get('u_receipt'):
                    reason='unmatched_motion_receipt'
                    break
            row=observe_rows(windows,trial,directory,spec)[-1]
            if learned:
                if index==0:
                    truth=direct_evidence(row,bundle,sources,request['block']['mass_class'])
                pred=skill.infer(row,readiness_only=bool(index))
                decisions.append(pred)
                campaign.atomic(directory/'decisions.json',dict(model_id=bundle['model_id'],predictions=decisions,**truth))
                if pred['state']=='ready':
                    exit_declared=True
                elif pred['state']=='recoverable' and index==0:
                    action=learned_action(pred,bundle,spec)
                    if not action:
                        reason='no_supported_single_axis_action'
                        break
                else:
                    reason=pred.get('reason') or 'no_confirmed_exit'
                    break
            elif index==0:
                action=next(iter(request['actions']),None)
            if exit_declared or index or not action:
                raw['is_terminal']=True
                raw['provenance']['direct_A']=not learned and not action
                outcome=io.consume()
                raw['consumer_outcome']=outcome
                break
        if index==0 and action:
            campaign.atomic(directory/'action_command.json',action)
            receipt=io.move(action,windows[0])
    rows=observe_rows(windows,trial,directory,spec)
    post=rows[-1]['post'][spec['active_post_condition']] if rows and outcome else {}
    success=post.get('success')
    assessment=None
    if learned:
        assessment=dict(**truth,evaluated=bool(post.get('evaluated')),success=success,
                        target_truth=success,exit_declared=exit_declared,invoked=receipt is not None,
                        rejected=not exit_declared or success is None,evidence=post.get('evidence',[]),task_success=None)
    result=dict(trial_id=trial,status='failed' if reason or success is False else 'complete',nodes=rows,
                attempt_reason=reason or (outcome or {}).get('reason'),assessment=assessment,consumer_outcome=outcome,
                cost={'actions':int(receipt is not None),'wall_seconds':time.monotonic()-start})
    campaign.atomic(directory/'completed_capture.json',result)
    return result


def execute_trial(request,directory,spec,cfg,*,real,interactive):
    if not real or not interactive:
        raise ValueError('P3 physical execution requires an interactive --real session')
    return run_trial(request,Path(directory),spec,cfg,RealIO(request,Path(directory),spec,cfg))

