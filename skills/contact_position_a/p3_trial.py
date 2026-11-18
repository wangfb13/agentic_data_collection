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
from .ah_geometry import KIND, validate_cell, horizontal, action_kind, coord_names


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
    anchors={anchor['node_id']:anchor for anchor in bundle['support']}
    ids=anchors.keys()
    matched=[]
    for old in source_rows:
        if old['node_id'] not in ids or old.get('evaluation_only') or not old['provenance'].get('direct_A'):
            continue
        post=old.get('post',{}).get(cid,{})
        old={**old,**anchors[old['node_id']]}
        if (old['attrs']['mass_class']!=mass or old['extractor_versions']!=row['extractor_versions']
                or not post.get('evaluated') or old['group_id']==row['group_id']):
            continue
        distance=measurement_distance(row,old,bundle['noise'])
        if distance<=min(bundle['graph'].get('radii',{}).get('similar_radius',1.),bundle['spec']['runtime']['support_distance']):
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
        action={'kind':action_kind(spec),'u':u}
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
        self.consumer=None if req.get('operation')=='collect_repeat_block' else cfg.get('consumer')
        self.assets=p3_force.load(self.consumer) if self.consumer else None
        self.mass=req['block']['mass_class']
        self.group='P3_FP_'+req['trial_id']
        self.z_limits=cfg['execution'].get('table_tcp_z_mm')
        self.hw=None
        self.request=req
        self.sequence=0
        self.reference=None
        self.reference_windows=[]
        self.air_file=self.directory/'airborne.json'
        self.contact_failure=None
        self.repeat_contacts={'measurement':0,'setup':0}
        self.placement=spec.get('current_placement') or 'P3'

    def confirm(self,request):
        print(f"{request['trial_id']} {request['role']}，请放{'轻' if self.mass=='light' else '重'}白瓶。"
              f" {request['mode']}，计划动作 {request['actions']}。",flush=True)
        if request.get('operation')=='collect_repeat_block':
            print('同一放置重复轻触；每次松爪，在同一 TCP 采下一次。请保持瓶子和夹爪位置。',flush=True)
        scope='确认修正路径畅通。' if self.consumer is None else '确认修正、提起及返回路径畅通。'
        if input('瓶已放稳，右爪全开空载并对准瓶身、遥操已停。'+scope+'输入 MOVE：').strip()!='MOVE':
            raise campaign.RetryTrial('未授权本次动作')

    @contextmanager
    def contact(self,index):
        from bottle_scan.e1e2 import Runner
        from bottle_scan.e1e2_hardware import Hardware
        from bottle_scan.geometry import validate_transform

        capture_masks=self.spec.get('graph_schema')=='state_graph_v5'
        class Capture(Hardware):
            def probe_window(self,since):
                value=super().probe_window(since)
                if value:
                    self.window_pose=self.arm.snapshot()['arms']['right']
                    if capture_masks:
                        from .lift_evidence import masks
                        self.window_masks=masks([v['median_dd'] for v in value],self.dd_reference,self.cfg['tactile'])
                return value

        folder=self.directory/(('placement_reference' if index==-1 else 'placement_reference_'+str(-index)) if index<0 else 'initial' if index==0 else 'reprobe' if index==1 else f'reprobe_{index}')
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
            if self.assets is not None:
                worker=self.worker=p3_force.sampling_worker(hw,self.consumer,self.assets[0])
                worker.start()
            runner=Runner(SimpleNamespace(out=folder,cond='A',step=False),self.cfg,None,hw)
            def finish_capture(record_path,before,status,p):
                hw.hold()
                row=dict(status=status,probe=p,unloaded_zero=zero,window_pose=getattr(hw,'window_pose',None),
                         touch_report=runner.last_touch_report,before_pose=before,after_pose=hw.pose(),
                         reference_digest=self.identity[0]['reference_digest'],finger_serials=self.identity[2])
                campaign.atomic(record_path,row)
                hw.end_record()
                raw=None
                if status=='contact' and p and p.get('valid'):
                    fb=row['window_pose']
                    if fb['tcp_frame']!='right_gripper_base_link' or hw.arm.snapshot()['base_frame']!='base_link':
                        raise ValueError('Unexpected tool/base frames')
                    pose=validate_transform(fb['T_base_tcp_mm'])
                    raw=ah_evidence.captured_window(record_path,self.identity,None,self.group,index,{},self.mass,'P3',pose,row)
                    if capture_masks:
                        from .placement_witness import label
                        raw['finger_masks']=dict(zip(('left','right'),[m.astype(int).tolist() for m in hw.window_masks]))
                        raw['provenance']['sensor_version']=self.identity[2]
                        raw['session']=self.placement
                        if index>=0 and self.spec.get('facts',{}).get('measured_placement_coordinates'):
                            label(raw,self.reference,self.cfg['axes_tcp']['a'],self.spec)
                            raw['repeat_settings']=self.request['settings'] if index==0 else {}
                else:
                    self.contact_failure='contact_lost'
                return raw,row
            self.contact_failure=None
            before=hw.pose()
            active=True
            if self.request.get('operation')=='collect_repeat_block':
                self.repeat_contacts['measurement' if index>=0 else 'setup']+=1
            status,p=runner.touch()
            raw,row=finish_capture(path,before,status,p)
            if raw is not None:
                answer=input('仅看刚才这次轻触：瓶子未被推移？y / n（n 继续采集，但不使用旧位置标签）：').strip().lower()
                confirmation={'answer':answer,'unmoved':True if answer in ('y','yes') else False if answer in ('n','no') else None}
                row['placement_confirmation']=confirmation
                campaign.atomic(path,row)
                if confirmation['unmoved'] is not True:
                    from .p3_placement import unlabelled_contact
                    self.invalidate_reference()
                    refresh=index>=0 and self.request.get('operation')!='collect_repeat_block'
                    if refresh:
                        # The pre-confirmation window may predate the movement.
                        # Acquire a new stable window at the same PROBE protocol;
                        # no arm motion and no reused position label.
                        refreshed=folder/'after_placement_confirmation'
                        refreshed.mkdir()
                        self.probe_refresh_count=getattr(self,'probe_refresh_count',0)+1
                        hw.begin_record(refreshed/'record')
                        before=hw.pose()
                        f=self.cfg['force']
                        p=runner.regulate('PROBE',f['probe_fraction']*f['target'],f['probe_tolerance'],f['probe_stable_s'],with_dd=True)
                        status='contact' if p and p.get('valid') else 'probe_unstable'
                        runner.note_touch(status,hw.snapshot(),p)
                        raw,new_row=finish_capture(refreshed/'observation.json',before,status,p)
                        new_row.update(placement_confirmation=confirmation,original_observation=str(path))
                        campaign.atomic(refreshed/'observation.json',new_row)
                    raw=unlabelled_contact(raw,confirmation,original_observation=path,refreshed=refresh)
            yield raw
        finally:
            try:
                air=self.air_file
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
        now=campaign_ah.current_pose(self.directory,'action_pose_'+str(self.sequence),self.cfg,campaign_ah.child)
        err,angle=pose_error(first['provenance']['measured_pose'],now)
        if err>1 or angle>1:
            raise RuntimeError('首触和开爪后实测 TCP 不一致')
        seq=self.sequence
        self.sequence+=1
        frame=(('horizontal_tcp_a' if horizontal(self.spec) else 'tcp'),'base')[k]
        opts=['--min-horizontal-projection',self.spec['geometry']['minimum_horizontal_projection']] if frame=='horizontal_tcp_a' else []
        return campaign_ah.move(self.directory,'action_move'+('_'+str(seq) if seq else ''),['--relative-at-confirm',*opts,'--frame',frame,
            '--direction','z+' if u[k]>0 else 'z-','--distance-mm',abs(u[k])],
            self.spec['task']['action_space']['max_step'][k],self.z_limits,campaign_ah.child)

    def link(self,first,last,receipt,u):
        if last['provenance'].get('placement_continuity') is False:
            last['provenance']['transition_excluded_reason']='object_placement_changed_or_unconfirmed'
            return
        ah_evidence.link(first,last,[receipt],self.cfg['axes_tcp']['a'],command=u,spec=self.spec)

    def consume(self):
        return p3_force.consume(self.hw,self.consumer,self.directory,self.assets[1],self.worker)


    def invalidate_reference(self):
        if self.reference is not None:
            self.reference.update(valid=False,invalidated_by_trial=self.request['trial_id'])
            campaign.atomic(self.reference_path,self.reference)

    def prepare(self,request):
        """Reference belongs to this uninterrupted placement, not to a mass class or a target TCP."""
        if not self.spec.get('facts',{}).get('measured_placement_coordinates'):return
        from .placement_witness import witness,label
        self.reference_path=Path(request['campaign'])/'placement_references'/self.placement/'active.json'
        if self.reference_path.exists():self.reference=campaign.read(self.reference_path)
        automatic=self.spec.get('geometry',{}).get('automatic_reference_check',False)
        reuse=False
        if self.reference and self.reference.get('valid') and self.reference['mass_class']==self.mass and self.reference['reference_digest']==self.identity[0]['reference_digest']:
            answer=input('自上次参考轻触以来，同一瓶始终处于同一放置、未被移动或重新放置？y / n：').strip().lower()
            if automatic:
                from .reference_check import verify
                reuse=verify(self,answer)
            else:reuse=answer in ('yes','y')
        recalibrate=bool(request.get('recalibrate_reference') or request.get('operation')=='recalibrate_reference')
        if recalibrate:reuse=False
        if not reuse:
            self.invalidate_reference()
            if input('请以已确认标准夹法对准现在的瓶子、全开并结束遥操。输入 REFERENCE 见证本次放置参考：').strip()!='REFERENCE':
                if automatic:
                    from .p3_placement import reference_unavailable
                    return reference_unavailable(self,request,'reference_witness_declined')
                raise campaign.RetryTrial('未见证放置参考')
            with self.contact(-1) as raw:
                if not raw:
                    reason={'object_moved':'参考轻触有效，但瓶位确认回答为否，需重新放稳',
                            'placement_unconfirmed':'参考轻触有效，但未确认瓶位（请输入 y 或 n）'}.get(self.contact_failure,'未获得参考双侧接触')
                    if automatic:
                        from .p3_placement import reference_unavailable
                        return reference_unavailable(self,request,reason)
                    raise campaign.RetryTrial(reason)
                if raw['provenance'].get('placement_continuity') is False:
                    from .p3_placement import reference_unavailable
                    return reference_unavailable(self,request,'placement_not_confirmed',raw)
                r=witness(raw,self.placement,self.identity[0]['reference_digest'])
                if automatic:
                    from copy import deepcopy
                    r['raw_witness']=deepcopy(raw)
                self.reference=r
                self.reference_path.parent.mkdir(parents=True,exist_ok=True)
                campaign.atomic(self.reference_path.with_name(r['placement_reference_id']+'.json'),r)
                campaign.atomic(self.reference_path,r)
                raw.update(is_reference=True,session=self.placement+'/'+r['placement_reference_id'])
                label(raw,r,self.cfg['axes_tcp']['a'],self.spec)
                self.reference_windows.append(raw)
            geom=self.spec['geometry']
            repeat=geom['recalibration_probe_repetitions' if recalibrate else 'reference_probe_repetitions']
            if type(repeat) is not int or repeat<1:raise ValueError('Invalid reference repeat count')
            for i in range(1,repeat):
                with self.contact(-(1+i)) as repeat_window:
                    if repeat_window is None:
                        self.invalidate_reference()
                        reason={'object_moved':'参考重复轻触有效，但瓶位确认回答为否',
                                'placement_unconfirmed':'参考重复轻触有效，但未确认瓶位（请输入 y 或 n）'}.get(self.contact_failure,'参考重复未获得有效双侧接触')
                        raise campaign.RetryTrial(reason)
                    if repeat_window['provenance'].get('placement_continuity') is False:
                        from .p3_placement import reference_unavailable
                        return reference_unavailable(self,request,'reference_repeat_placement_not_confirmed',repeat_window)
                    label(repeat_window,r,self.cfg['axes_tcp']['a'],self.spec)
                    if not repeat_window['provenance']['placement_reference_valid']:
                        from .p3_placement import reference_unavailable
                        return reference_unavailable(self,request,'reference_repeat_not_coordinate_compatible',repeat_window)
                    repeat_window.update(is_reference=True,session=self.placement+'/'+r['placement_reference_id'])
                    self.reference_windows.append(repeat_window)
        if automatic and not reuse and getattr(self,'reference_check',None):
            self.reference_check['rewitnessed']=True
            campaign.atomic(self.directory/'reference_check'/'check.json',self.reference_check)
        if self.reference.get('valid') and any(k in request['settings'] for k in coord_names(self.spec)):
            u=[request['settings'].get(k,0) for k in coord_names(self.spec)]
            from .p3_prepare import prepare_position
            prepare_position(self.directory,np.array(self.reference['pose']),u,self.cfg,self.spec,campaign_ah.child,
                             station_z_bounds=self.z_limits)

    def repeat_pose(self):
        return campaign_ah.current_pose(self.directory,'repeat_tcp',self.cfg,campaign_ah.child)

    def repeat_continuity(self):
        return input('重复块进行中：同一只瓶始终未被移动，仍在同一放置？y / n：').strip().lower() in ('y','yes')

    def expert_action(self,raw):
        """Human expert specifies one signed movement through the same owned executor."""
        text=input('专家：输入 r 或 h 和有符号毫米数（例如 h -2），或 HANDOFF 结束：').strip()
        if text=='HANDOFF':return None
        axis,amount=text.split()
        index=[k.split('_')[0] for k in coord_names(self.spec)].index(axis)
        u=[0.,0.]
        u[index]=float(amount)
        return {'kind':action_kind(self.spec),'u':u}

    def consume_for(self,cid):
        import random
        from bottle_scan.state_graph.skill_interface import post_spec
        condition=post_spec(self.spec,cid)
        options=self.spec['consumer']
        root=self.directory/'consumers'/cid
        root.mkdir(parents=True)
        self.air_file=root/'airborne.json'
        rng=random.Random(str(self.request['consumer_seed'])+'/'+cid)
        params={**options,'expected_weight_n':options.get('weight_newtons',{}).get(self.mass),
                'normal_axis':self.cfg['force']['normal_axis'],'rotation_base_tcp':self.hw.pose()[:3,:3].tolist(),'audit_selected':rng.random()<options['audit_rate']}
        if condition['consumer']['controller']=='fixed_force':
            outcome=p3_force.consume_fixed(self.hw,condition,params,root)
        else:
            outcome=p3_force.consume(self.hw,self.consumer,root,self.assets[1],self.worker,options=params)
        # A lift followed by a setdown is a new placement unless independently verified.
        # The next fresh probe is used to assess pairing; labels need a new witness.
        if outcome.get('object_moved') or outcome.get('contact_lost') or outcome.get('return_contact') is False:
            self.invalidate_reference()
        return outcome


def run_trial(request,directory,spec,cfg,io,*,sources=None):
    if spec.get('graph_schema')=='state_graph_v5' or spec['runtime'].get('max_steps',1)>1:
        from .p3_trial_v5 import run_trial as multistep
        return multistep(request,directory,spec,cfg,io,sources=sources)
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

