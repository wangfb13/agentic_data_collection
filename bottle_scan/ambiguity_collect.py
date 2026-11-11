"""Explicit opt-in PROBE-only pilot; no hardware at import or plan preview."""
from __future__ import annotations
import argparse
import csv
import fcntl
import json
from pathlib import Path
import signal
import threading
import time
import traceback
from types import SimpleNamespace
import numpy as np
from .ambiguity_plan import make_design, validate_design, ProbeBounds, plan_report
from .e1e2 import Runner, UserExit, resolve_config
from .e1e2_core import load_settings, validate, config_identity
from .geometry import pose_error, validate_transform
from .session import atomic_json


def parser():
    p = argparse.ArgumentParser(description='观测歧义验证：离线计划 / 独立轻触采集')
    sub = p.add_subparsers(dest='command', required=True)
    for name in ('plan', 'collect'):
        s = sub.add_parser(name)
        s.add_argument('--config', type=Path, default=Path(__file__).resolve().parents[1]/'config.e1e2.yaml')
        s.add_argument('--out', type=Path, required=True)
        s.add_argument('--design-file', type=Path, help='Validated immutable skill-extension design')
        s.add_argument('--step-mm', type=float)
        s.add_argument('--tilt-deg', type=float, help='绕局部 c，TCP 为旋转中心；0 为仅 a/l')
        s.add_argument('--repeats', type=int)
        s.add_argument('--seed', type=int)
        s.add_argument('--bottle', help='同 session 只用同一物体')
        if name == 'plan':
            s.add_argument('--pose-file', type=Path)
        else:
            s.add_argument('--real', action='store_true')
            s.add_argument('--step', action='store_true', help='所有阶段前回车；默认前两次逐阶段')
            s.add_argument('--limit', type=int)
            s.add_argument('--min-normalized', type=float)
            s.add_argument('--target', type=float, help='沿用 E1/E2 F_target，实际只执行其 PROBE 比例')
            s.add_argument('--f-max', type=float)
    return p


def design_from_args(args, existing=None):
    if getattr(args, 'design_file', None):
        design = validate_design(json.loads(args.design_file.read_text()))
        if any(getattr(args, key, None) is not None for key in ('step_mm', 'tilt_deg', 'repeats', 'seed', 'bottle')):
            raise ValueError('design-file cannot be combined with design overrides')
        if existing is not None and design != existing:
            raise ValueError('resume cannot change extension design')
        return design
    if existing is not None and existing.get('schema') == 'skill_extension_probe_design_v1':
        raise ValueError('extension resume requires original design-file')
    options = {'step_mm': 4., 'tilt_deg': 3., 'repeats': 3, 'seed': 20261105, 'bottle': 'bottle'}
    if existing is not None:
        validate_design(existing)
    for key, default in list(options.items()):
        value = getattr(args, key, None)
        if existing is not None and value is not None and value != existing[key]:
            raise ValueError('续跑不能更换 '+key+'；请新建目录')
        options[key] = existing[key] if existing is not None else (default if value is None else value)
    return make_design(**options)


def summary(root):
    rows = []
    for p in sorted((root/'trials').glob('*/meta.json')):
        m = json.loads(p.read_text())
        rows.append({**{k: m.get(k) for k in ('id','state_id','repeat','bottle','status','label',
                                            'note','complete','recording_complete','skipped_reason')},
                     **m.get('perturbation', {}), 'probe_valid': bool(m.get('probe_initial', {}).get('valid'))})
    fields = ['id','state_id','repeat','bottle','a_mm','l_mm','tilt_deg','status','probe_valid',
              'label','note','complete','recording_complete','skipped_reason']
    tmp = root/'summary.csv.tmp'
    with tmp.open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    tmp.replace(root/'summary.csv')
    return rows


class ProbeRunner(Runner):
    def __init__(self, args, cfg, design, hw, **kwargs):
        args.cond = 'A'
        super().__init__(args, cfg, None, hw, **kwargs)
        self.design = design
        self.last_pre = None
        self.last_commanded_pose = None

    def move(self, target, role='grasp', anchor=None, **kwargs):
        error = self.bounds.check(target, role, anchor)
        if error:
            raise ValueError(error)
        snapshot = self.hw.snapshot()
        if abs(snapshot['width']-self.cfg['gripper']['open_normalized']) > self.cfg['gripper']['open_tolerance']:
            raise RuntimeError('轻触实验只允许全开状态移动/旋转')
        distance, angle = pose_error(self.hw.pose(), target)
        # A failed or externally disturbed pose is never silently rotated near
        # the bottle. Only retracted-to-retracted segments permit rotation.
        expected = self.last_commanded_pose if self.last_commanded_pose is not None else self.T0
        angular_error = pose_error(expected,self.hw.pose())[1]
        if angular_error > self.cfg['motion']['pose_tolerance_deg']:
            raise RuntimeError('接近段姿态偏差过大；请现场处理')
        if not np.allclose(target[:3,:3],expected[:3,:3],atol=1e-8,rtol=0):
            if role != 'retreat':
                raise RuntimeError('目标姿态变化只允许发生在退让位')
            if self.last_pre is None:
                raise RuntimeError('旋转只能从已到达的退让位开始')
            mm, deg = pose_error(self.last_pre, self.hw.pose())
            if mm > self.cfg['motion']['pose_tolerance_mm'] or deg > self.cfg['motion']['pose_tolerance_deg']:
                raise RuntimeError('旋转前实测未在退让位')
        speed = min(5., self.cfg['motion']['near_speed_mm_s'])
        self.engaged = True
        self.hw.move(target, speed)  # Existing Hardware uses 1 degree/s.
        deadline = time.monotonic()+max(distance/speed, angle)+self.cfg['motion']['timeout_margin_s']
        while True:
            self.hw.snapshot()
            if self.hw.motion_done():
                self.last_commanded_pose = target.copy()
                return 'reached'
            if time.monotonic() > deadline:
                raise RuntimeError('轻触实验移动超时')
            self.hw.wait(.01)

    def initial(self):
        if self.progress.get('unsafe_recovery') or self.progress.get('airborne'):
            raise RuntimeError('上次恢复未完成；请现场处理并使用新目录，旧数据保留')
        if self.meta:
            self.T0 = validate_transform(self.meta['T0']).copy()
            self.say('原 T0 xyz(mm)='+str(self.T0[:3,3].tolist()))
            self.say('原 T0 旋转矩阵='+str(self.T0[:3,:3].tolist()))
            self.read('同一物体复位、夹爪全开，恢复原 T0 和朝向，停止遥操后按回车。')
            mm, deg = pose_error(self.T0, self.hw.pose())
            if mm > self.cfg['motion']['pose_tolerance_mm'] or deg > self.cfg['motion']['pose_tolerance_deg']:
                raise RuntimeError('实测未恢复原 T0；不会自动运动到 T0')
        else:
            self.read('瓶子保持桌面支撑、胶面未接触；夹爪全开移到理想抓取位，停止遥操后按回车记录 T0。')
            self.T0 = validate_transform(self.hw.pose()).copy()
            self.meta = {'schema':'ambiguity_session_v1','cond':'A','simulated':False,
                         'protocol':'probe_only_a_l_tilt','config':self.cfg,
                         'config_identity':config_identity(self.cfg,'A',None),
                         'ambiguity_design':self.design,'plan':self.design['plan'],
                         'T0':self.T0.tolist(),'created_unix_ns':time.time_ns(),'zero_history':[],
                         'training_eligible':False,'recovery_action_ground_truth':False}
            self.save()
        self.bounds = ProbeBounds(self.cfg, self.T0, self.design)
        self.last_commanded_pose = self.T0.copy()
        neutral_error = self.bounds.check(self.bounds.pre(self.T0),'retreat',self.T0)
        if neutral_error:
            raise ValueError('初始退让位不满足限制：'+neutral_error)
        preview = plan_report(self.design, self.cfg, self.T0)
        atomic_json(self.root/'plan_preview.json', preview)
        for row in preview['trials']:
            self.say(f"{row['id']} {row['perturbation']} Δbase={np.round(row['delta_base_xyz_mm'],3).tolist()}"
                     + (' 跳过：'+row['skip_reason'] if row['skip_reason'] else ''))
        self.say('瓶子须始终由桌面支撑；只 PROBE，不执行 SQUEEZE/LIFT。倾斜绕 TCP，胶面会随旋转移动。')
        self.say('全开后先退让，再在退让位变换姿态/位置，然后直线接近；没有工具包络或碰撞检测。')
        self.read('确认全开空载，按回车扣零；此步不发运动。')
        self.baseline_folder = self.root/'baseline'/str(time.time_ns())
        self.hw.begin_record(self.baseline_folder)
        error = None
        try:
            self.hw.set_context('ZERO', 'baseline')
            zero = self.hw.zero_unloaded()
            np.savez_compressed(self.baseline_folder/'dd_reference.npz',left=self.hw.dd_reference[0],right=self.hw.dd_reference[1])
            zero['dd_reference_file'] = str((self.baseline_folder/'dd_reference.npz').relative_to(self.root))
            self.meta['zero_history'].append(zero)
            self.meta['force_controller'] = getattr(self.hw,'legacy_evidence',None)
            self.save()
        except BaseException as exc:
            error = str(exc) or type(exc).__name__; raise
        finally:
            self.hw.end_record(error=error)
        self.say('逐指接触阈值 SDK='+str(zero['contact_thresholds_sdk']))
        self.authorize('已打印全部目标。确认退让/旋转净空和瓶子支撑，输入 MOVE 开始。')
        self.hw.ownership()

    def record_probe(self, probe, target):
        if not probe:
            return {}
        snapshot = self.hw.snapshot()
        out = dict(probe)
        out.update(measured_tcp_mm=self.hw.pose().tolist(),commanded_tcp_mm=target.tolist(),
                   measurement_timestamp_ns=snapshot['timestamp_ns'],
                   wrench_zeroed=np.asarray(snapshot['wrench_zeroed']).tolist(),
                   stable_probe=bool(probe.get('valid') and self.bilateral(snapshot)),
                   wrench_timing='post_PROBE_snapshot_not_exact_DD_timestamp',rotation_pivot='tcp')
        window = getattr(self.hw,'last_probe_window',None)
        if window and [x['frame_ns'] for x in window] == [x['frame_ns'] for x in probe['window']]:
            path = self.folder/'probe_observation.npz'
            np.savez_compressed(path,dd_left=window[0]['median_dd'],dd_right=window[1]['median_dd'],
                                reference_left=self.hw.dd_reference[0],reference_right=self.hw.dd_reference[1],
                                wrench_zeroed=snapshot['wrench_zeroed'])
            out['observation_file'] = path.name
        else:
            out['observation_export_missing'] = True
        return out

    def label(self):
        self.say('检查瓶子在标记处且全程桌面支撑；这是采集质量标签，不是抓取成功。')
        while True:
            value = self.read('回车/1=未见移动，2=瓶子移动需复位，0=作废，n=备注，q=标完本次后退出：').lower()
            if value == 'n':
                self.current['note'] = self.read('备注：'); self.save(); continue
            if value == 'q':
                self.quit = True; self.say('已请求退出，请先用 1/2/0 标记本次。'); continue
            if value in ('','1','2','0'):
                self.current['label'] = {'':'normal','1':'normal','2':'pushed','0':'invalid'}[value]
                self.current['bottle_moved'] = value == '2'
                self.current['operator_checked_support'] = True
                if value == '2' and not self.quit:
                    self.read('请复位瓶子，确认张开且停在退让位，再按回车。')
                return

    def trial(self, item):
        self.folder = self.root/'trials'/item['id']
        if self.folder.exists():
            archive = self.root/'interrupted'/f"{item['id']}_{time.time_ns()}"
            archive.parent.mkdir(parents=True,exist_ok=True); self.folder.rename(archive)
        self.folder.mkdir(parents=True)
        self.stage_step = self.args.step or self.physical_completed < 2
        self.current = {**item,'complete':False,'recording_complete':False,'status':'started',
                        'protocol':'probe_only_a_l_tilt','zero':self.hw.zero,'note':'',
                        'started_unix_ns':time.time_ns(),'training_eligible':False,
                        'label_semantics':'operator observed bottle stability'}
        self.progress['active'] = item['id']; self.save()
        target = self.bounds.target(item); pre = self.bounds.pre(target)
        reason = self.bounds.check(target) or self.bounds.check(pre,'retreat',target)
        if reason:
            self.current.update(status='skipped_bound',skipped_reason=reason,label='invalid',complete=True)
        else:
            self.hw.begin_record(self.folder); error = None
            try:
                self.release()
                if self.last_pre is None:
                    with self.stage('INITIAL_RETREAT'):
                        neutral_pre = self.bounds.pre(self.T0)
                        self.move(neutral_pre,'retreat',self.T0); self.last_pre = neutral_pre.copy()
                with self.stage('MOVE_PRE'):
                    self.move(pre,'retreat',target); self.last_pre = pre.copy()
                with self.stage('APPROACH'):
                    self.move(target)
                status, probe = self.touch()
                initial_status = status
                if hasattr(self, 'after_initial_probe'):
                    self.current['probe_initial'] = self.record_probe(probe, target)
                    target, status, probe = self.after_initial_probe(target, status, probe)
                    pre = self.bounds.pre(target)
                    self.current['probe_final'] = self.record_probe(probe, target)
                self.current.update(status=status,initial_status=initial_status,
                                    probe_initial=self.current.get('probe_initial', self.record_probe(probe,target)),
                                    bilateral=self.bilateral(self.hw.snapshot()),
                                    target_tcp_mm=target.tolist(),lifted=False)
                self.save(); self.release()
                with self.stage('RETREAT'):
                    self.move(pre,'retreat',target); self.last_pre = pre.copy()
            except BaseException as exc:
                error = str(exc) or type(exc).__name__
                self.current.update(status='interrupted',error=error)
                self.recover(exc); raise
            finally:
                self.save()
                try:
                    self.hw.end_record(error=error)
                finally:
                    path = self.folder/'recording_status.json'
                    self.current['recording_complete'] = bool(path.exists() and json.loads(path.read_text()).get('complete'))
                    self.save()
            if not self.current['recording_complete']:
                raise RuntimeError('记录不完整；停止，不能把本次计为完成')
            self.current['physical_finished'] = True
            self.save()
            self.hw.set_context('LABEL',item['id'])
            self.say(f"轻触 {item['id']}/{len(self.design['plan'])} {item['perturbation']} 状态={status} PROBE有效={bool(probe and probe.get('valid'))}")
            self.label(); self.current['complete'] = True; self.physical_completed += 1
        self.current['finished_unix_ns'] = time.time_ns()
        self.progress['completed'].append(item['id']); self.progress['active'] = None
        self.save(); summary(self.root)

    def run(self):
        rows = summary(self.root); done = {r['id'] for r in rows if r['complete']}
        self.progress['completed'] = sorted(done)
        self.physical_completed = sum(bool(r['complete']) and r['status'] != 'skipped_bound' for r in rows)
        self.initial(); count = 0
        for item in self.design['plan']:
            if item['id'] in done:
                continue
            if self.args.limit is not None and count >= self.args.limit:
                break
            self.trial(item); count += 1
            if self.quit:
                break
        self.say(f'本次完成 {count}；累计 {len(self.progress["completed"])}/{len(self.design["plan"])}。夹爪张开，保持最后退让位。')
        self.say('离线诊断：bash run_on_5090.sh analyze_ambiguity '+str(self.root)+' --read-h5 --out '+str(self.root/'analysis_ambiguity'))
        return 0

    def recover(self, exc):
        try:
            super().recover(exc)
        finally:
            # Parent recovery writes its one-dimensional summary; restore the
            # diagnostic columns even when a recording/motion fault interrupted.
            summary(self.root)


def hardware_factory(cfg):
    from .e1e2_hardware import Hardware  # Only after collect --real.
    class ProbeHardware(Hardware):
        def probe_window(self, since):
            result = super().probe_window(since)
            self.last_probe_window = result
            return result
    return ProbeHardware(cfg)


def finish_pending_labels(root, ask=input, say=print):
    """No hardware even if interrupted after RETREAT but before label commit."""
    for path in sorted((root/'trials').glob('*/meta.json')):
        item = json.loads(path.read_text())
        if item.get('complete'):
            continue
        stages = item.get('stages',[])
        retired = any(s.get('name') == 'RETREAT' and s.get('end_ns') and not s.get('error') for s in stages)
        record_path = path.parent/'recording_status.json'
        recorded = bool(record_path.exists() and json.loads(record_path.read_text()).get('complete'))
        if not (recorded and (item.get('physical_finished') or retired)):
            continue
        say('试验 '+str(item['id'])+' 已完成实际采集，仅补标签，不重复运动。')
        while True:
            value = ask('根据当时观察选择：1=未见瓶子移动，2=移动，0=无法确认/作废，q=退出：\n> ').strip().lower()
            if value == 'q':
                raise UserExit()
            if value in ('1','2','0'):
                item.update(label={'1':'normal','2':'pushed','0':'invalid'}[value],
                            bottle_moved=value=='2',recording_complete=True,physical_finished=True,
                            complete=True,label_recovered_unix_ns=time.time_ns(),
                            finished_unix_ns=time.time_ns())
                atomic_json(path,item)
                break
    summary(root)


def main(argv=None, *, make_hardware=None, input_fn=input, output=print, runner_factory=None):
    p = parser(); args = p.parse_args(argv)
    if args.command == 'collect' and not args.real:
        p.error('采集必须显式 --real；机器人关闭时用 plan 或 analyze_ambiguity')
    if args.command == 'collect' and args.limit is not None and args.limit < 1:
        p.error('--limit 必须为正整数')
    hw = runner = lock = None; handlers = {}
    try:
        if args.command == 'plan':
            cfg = load_settings(args.config); validate(cfg, require_resolved=False)
            design = design_from_args(args); T0 = None
            if args.pose_file:
                value = json.loads(args.pose_file.read_text())
                T0 = value.get('T_base_tcp_mm',value.get('T0',value.get('pose_mm'))) if isinstance(value,dict) else value
                if T0 is None:
                    raise ValueError('pose-file 需含 T_base_tcp_mm 或 T0')
                validate_transform(T0)
            atomic_json(args.out,plan_report(design,cfg,T0))
            output(f"仅生成计划：{design['states']} 个状态 × {design['repeats']} 轮 = {len(design['plan'])} 次。")
            output('a/l=±'+str(design['step_mm'])+' mm；绕 TCP 局部 c=±'+str(design['tilt_deg'])+'°；只有 PROBE。')
            output('未连接硬件；'+('含实测 T0 基座目标。' if T0 is not None else '无实测 T0，基座目标将在现场生成。'))
            output(str(args.out.resolve())); return 0
        args.out = args.out.expanduser().resolve(); args.out.mkdir(parents=True,exist_ok=True)
        lock = (args.out/'.ambiguity.lock').open('a'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        path = args.out/'session_meta.json'
        if path.exists():
            meta = json.loads(path.read_text())
            if meta.get('schema') != 'ambiguity_session_v1':
                raise ValueError('目录不是观测歧义实验；不能复用 E1/E2 或 RSI session')
            design = design_from_args(args,meta['ambiguity_design']); cfg = meta['config']; validate(cfg)
            for attr, field in [('min_normalized',('gripper','min_normalized')),('target',('force','target')),('f_max',('force','max'))]:
                if getattr(args,attr) is not None and getattr(args,attr) != cfg[field[0]][field[1]]:
                    raise ValueError('续跑不能更换 '+attr)
            finish_pending_labels(args.out,input_fn,output)
            done = {r['id'] for r in summary(args.out) if r['complete']}
            if all(row['id'] in done for row in design['plan']):
                output('计划全部完成；不连接硬件、不发动作。'); return 0
        else:
            if any(x.name != '.ambiguity.lock' for x in args.out.iterdir()):
                raise ValueError('新采集请使用空目录')
            design = design_from_args(args)
            base_args = SimpleNamespace(**vars(args))
            base_args.cond = 'A'; base_args.offsets = base_args.reps = base_args.fit = None
            base_args.seed = None; base_args.trial_run = False
            cfg, _ = resolve_config(base_args,input_fn,output)
        hw = (make_hardware or hardware_factory)(cfg)
        runner = (runner_factory or ProbeRunner)(args,cfg,design,hw,ask=input_fn,say=output)
        if threading.current_thread() is threading.main_thread():
            def interrupted(signum, frame):
                raise KeyboardInterrupt('收到信号 '+str(signum))
            for sig in (signal.SIGINT,signal.SIGTERM):
                handlers[sig] = signal.signal(sig,interrupted)
        hw.connect(); return runner.run()
    except (Exception,KeyboardInterrupt) as exc:
        if runner:
            try:
                runner.recover(exc)
            except Exception as recovery:
                output('恢复失败，请现场接管：'+str(recovery))
        if args.command == 'collect':
            atomic_json(args.out/'errors'/f'{time.time_ns()}.json',{'error':str(exc) or type(exc).__name__,'traceback':traceback.format_exc()})
        output('观测歧义实验停止：'+(str(exc) or type(exc).__name__))
        return 130 if isinstance(exc,(KeyboardInterrupt,UserExit)) else 2
    finally:
        if hw:
            hw.close()
        for sig, old in handlers.items():
            signal.signal(sig,old)
        if lock:
            lock.close()
