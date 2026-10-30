"""Bounded batch adapter for the existing E1/E2 state machine.

Importing this module never imports ROS.  The agent only supplies a frozen batch
of targets; Runner still owns every physical stage, force limit and recovery.
"""
from __future__ import annotations

import copy
import fcntl
import json
from pathlib import Path
import signal
import threading
import time
from types import SimpleNamespace

import numpy as np

from .e1e2 import Runner, UserExit, rebuild_summary
from .e1e2_core import Correction, MotionBounds, config_identity, validate
from .session import atomic_json, fingerprint, plain


def _correction(fit, settings):
    if not isinstance(fit, dict):
        raise ValueError('条件 B 需要有效且已冻结的触觉模型')
    if fit and fit.get('schema') == 'agentic_fit_v1':
        from .agentic_learning import AgenticCorrection
        return AgenticCorrection(fit, settings)
    return Correction(fit, settings)


def validate_plan(plan, cfg, fit):
    """Validate identifiers and configuration before importing hardware."""
    validate(cfg)
    if not isinstance(plan, list) or not plan:
        raise ValueError('agentic batch 需要非空计划')
    seen = set()
    for row in plan:
        name = row.get('id')
        if (not isinstance(name, str) or not name or len(name) > 128 or
                name in ('.', '..') or '/' in name or '\\' in name or
                any(ord(c) < 32 for c in name) or name in seen):
            raise ValueError('试验 id 需唯一且不能含路径')
        seen.add(name)
        if row.get('cond') not in ('A', 'B'):
            raise ValueError('每项计划必须显式指定 A/B')
        if type(row.get('index')) is not int or row['index'] < 1:
            raise ValueError('每项计划 index 需要为正整数')
        if row.get('bottle') not in cfg['design']['bottles']:
            raise ValueError('未知瓶子分块')
        delta = row.get('delta_mm')
        if isinstance(delta, bool) or not isinstance(delta, (int, float)) or not np.isfinite(delta) or abs(delta) > cfg['motion']['max_grasp_offset_mm']:
            raise ValueError('偏移超出固定抓取范围')
        if row.get('kind') not in ('scan', 'zero_monitor'):
            raise ValueError('未知试验类型')
        if row['kind'] == 'zero_monitor' and delta != 0:
            raise ValueError('零点监测必须使用零偏移')
    if any(r['cond'] == 'B' for r in plan):
        _correction(fit, cfg['servo'])
        if fit.get('axes_tcp') is not None and not all(np.allclose(fit['axes_tcp'][k], cfg['axes_tcp'][k]) for k in ('a', 'c', 'l')):
            raise ValueError('拟合轴定义与执行配置不同')
        if fit.get('normal_signs') is not None and fit['normal_signs'] != cfg['force']['normal_signs']:
            raise ValueError('拟合法向力符号与执行配置不同')
        if fit.get('normal_axis') is not None and fit['normal_axis'] != cfg['force']['normal_axis']:
            raise ValueError('拟合法向力轴与执行配置不同')


def recover_context(root, fallback=None):
    """Read the last committed context, including an interrupted first T0."""
    context = copy.deepcopy(fallback or {})
    path = Path(root) / 'session_meta.json'
    if path.exists():
        meta = json.loads(path.read_text())
        context.update(meta.get('agentic_context') or {})
        for key in ('T0', 'initial_centroids', 'T0_accepted_unix_ns'):
            if meta.get(key) is not None:
                context[key] = meta[key]
    progress = Path(root) / 'progress.json'
    if progress.exists():
        state = json.loads(progress.read_text())
        if state.get('airborne') or state.get('unsafe_recovery'):
            context['unsafe_recovery'] = True
            context['airborne'] = bool(state.get('airborne'))
    # Trial metadata is committed before the context checkpoint.  Recover the
    # final labelled zero in that narrow crash window without counting it twice.
    samples = context.setdefault('zero_samples', [])
    known = {s['trial_path'] for s in samples}
    for trial_path in sorted((Path(root) / 'trials').glob('*/meta.json')):
        row = json.loads(trial_path.read_text())
        key = str(trial_path.parent.resolve())
        if (key not in known and row.get('complete') and row.get('recording_complete') and
                row.get('label') != 'invalid' and row.get('zero_centroids_px')):
            samples.append({'trial_path': key, 'centroids_px': row['zero_centroids_px']})
            known.add(key)
    if len(samples) >= 3:
        context.setdefault('zero_baseline_px', np.mean([s['centroids_px'] for s in samples[:3]], axis=0).tolist())
    # These authorizations belong to the live caller, not a disk checkpoint.
    # In particular, resuming must not resurrect an old continuous authorization
    # or replace a newly computed remaining-time deadline with a stale one.
    for key in ('continuous_campaign', 'deadline_unix_ns'):
        if fallback is not None and key in fallback:
            context[key] = fallback[key]
    return plain(context)


def deadline_reached(context):
    deadline = context.get('deadline_unix_ns')
    return deadline is not None and time.time_ns() >= int(deadline)


def _validate_context(context):
    deadline = context.get('deadline_unix_ns')
    if deadline is not None and (isinstance(deadline, bool) or not isinstance(deadline, int) or deadline <= 0):
        raise ValueError('deadline_unix_ns 必须为正整数时间戳')
    for key in ('T0', 'last_safe_pose', 'last_safe_target', 'last_grasp_target'):
        if context.get(key) is None:
            continue
        pose = np.asarray(context[key], float)
        if (pose.shape != (4, 4) or not np.isfinite(pose).all() or
                not np.allclose(pose[3], [0, 0, 0, 1]) or
                not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-4) or
                np.linalg.det(pose[:3, :3]) < .99):
            raise ValueError('campaign 中的 ' + key + ' 不是合法刚体位姿')


def _metadata(root, plan, cfg, fit, context, simulated):
    conds = sorted({r['cond'] for r in plan})
    return {
        'schema': 'e1e2_session_v1', 'cond': conds[0] if len(conds) == 1 else 'mixed',
        'config': plain(cfg), 'fit': plain(fit), 'plan': plain(plan),
        'config_identity': config_identity(cfg, conds, fit), 'created_unix_ns': time.time_ns(),
        'simulated': bool(simulated), 'T0': context.get('T0'),
        'initial_centroids': context.get('initial_centroids'),
        'T0_accepted_unix_ns': context.get('T0_accepted_unix_ns'), 'zero_history': [],
        'agentic': {**cfg.get('agentic', {}), 'role': context.get('batch_role', cfg.get('agentic', {}).get('role', 'exploration'))},
        'agentic_context': plain(context), 'batch_plan_sha256': fingerprint(plan),
        'delta_visibility': 'Only executor sees planned delta; correction receives tactile/status/bilateral only',
        'position_truth': 'measured TCP projection relative to fixed T0; bottle displacement is not measured',
    }


class BatchRunner(Runner):
    """Use Runner.trial unchanged; replace only campaign-level orchestration."""
    def __init__(self, args, cfg, fit, hw, plan, context, *, ask=input, say=print, authorize=True):
        self.batch_plan = copy.deepcopy(plan)
        self.context = copy.deepcopy(context)
        self.needs_authorization = bool(authorize)
        self.motion_authorized = False
        self.initial_phase = True
        # Manual Runner accepts only its legacy fit in __init__.  Instantiate A
        # then choose the model for each actual trial, including mixed holdout.
        original_cond = args.cond
        args.cond = 'A'
        super().__init__(args, cfg, fit, hw, ask=ask, say=say)
        args.cond = original_cond
        self.correction = None

    def save(self):
        if self.meta is not None:
            self.meta['plan'] = self.batch_plan
            self.meta['agentic_context'] = plain(self.export_context())
            self.meta['agentic'] = {**self.cfg.get('agentic', {}), 'role': self.context.get('batch_role', self.cfg.get('agentic', {}).get('role', 'exploration'))}
            self.meta['batch_plan_sha256'] = fingerprint(self.batch_plan)
        super().save()

    def export_context(self):
        out = copy.deepcopy(self.context)
        if self.meta is not None:
            for key in ('T0', 'initial_centroids', 'T0_accepted_unix_ns'):
                if self.meta.get(key) is not None:
                    out[key] = plain(self.meta[key])
        out['unsafe_recovery'] = bool(self.progress.get('unsafe_recovery'))
        out['airborne'] = bool(self.progress.get('airborne'))
        return plain(out)

    def _permit_motion(self, message):
        if self.motion_authorized:
            return
        if self.needs_authorization or not self.context.get('continuous_campaign'):
            self.authorize(message)
        self.hw.ownership()
        self.motion_authorized = True

    def touch(self, prefix=''):
        if self.initial_phase:
            if not self.motion_authorized:
                self.print_plan(self.completed())
            self._permit_motion('将先在当前 T0 执行轻触与回开；确认瓶位和场地，输入 MOVE。')
        return super().touch(prefix)

    def regulate(self, name, target, tolerance, stable_s, *, with_dd=False):
        result = super().regulate(name, target, tolerance, stable_s, with_dd=with_dd)
        if result is not None and with_dd:
            measured = self.hw.pose()
            result['measured_tcp_mm'] = measured.tolist()
            result['actual_delta_mm'] = float((measured[:3, 3] - self.T0[:3, 3]) @ self.bounds.a)
            result['position_reference'] = 'TCP projection relative to fixed campaign T0; not bottle-relative truth'
            result['tcp_sample_timestamp_ns'] = self.hw.now()
        return result

    def _open_readback(self):
        # Hardware.snapshot requires a force zero; checking width before a fresh
        # zero must therefore read the public gripper state directly.
        grip = getattr(self.hw, 'grip', None)
        width = grip.fresh()['normalized'] if grip is not None else self.hw.snapshot()['width']
        if not np.isfinite(width) or abs(float(width) - self.cfg['gripper']['open_normalized']) > self.cfg['gripper']['open_tolerance'] + 1e-9:
            raise RuntimeError('批次开始需要夹爪全开；请现场释放后恢复，程序未移动机械臂')

    def initial(self):
        if self.context.get('unsafe_recovery') or self.context.get('airborne') or self.progress.get('unsafe_recovery') or self.progress.get('airborne'):
            raise RuntimeError('上次中断可能仍夹瓶或在空中；禁止自动续跑，请先现场处理并执行恢复确认')
        accepted = self.meta and self.meta.get('T0') is not None and self.meta.get('T0_accepted_unix_ns')
        # Existing seed sessions contain an accepted T0 but some old versions
        # have no timestamp.  The caller may explicitly confirm that source.
        accepted = accepted or (self.meta and self.meta.get('T0') is not None and self.context.get('T0_source_accepted'))
        if not accepted:
            super().initial()
            self.initial_phase = False
            self.context['last_safe_pose'] = self.hw.pose().tolist()
            self.context['last_safe_target'] = self.T0.tolist()
            self.context['last_grasp_target'] = self.T0.tolist()
            self.save()
            return
        self.T0 = np.asarray(self.meta['T0'], float)
        self.bounds = MotionBounds(self.cfg, self.T0)
        self.print_plan(self.completed())
        if self.needs_authorization or not self.context.get('continuous_campaign'):
            self.read('固定使用 campaign 原 T0。请确认瓶子在原标记、夹爪全开；允许从上次记录的安全退让位返回，或先恢复到 T0。停止遥操后按回车。')
        self._open_readback()
        now = self.hw.pose()
        if not np.allclose(now[:3, :3], self.T0[:3, :3], atol=.01):
            raise RuntimeError('当前朝向与固定 T0 不同；本流程不会通过旋转恢复，请现场恢复')
        tolerance = self.cfg['motion']['pose_tolerance_mm']
        at_t0 = np.linalg.norm(now[:3, 3] - self.T0[:3, 3]) <= tolerance
        prior = self.context.get('last_safe_pose')
        at_prior = prior is not None and np.linalg.norm(now[:3, 3] - np.asarray(prior)[:3, 3]) <= tolerance
        if not at_t0 and not at_prior:
            raise RuntimeError('当前 TCP 不在原 T0 或已记录的开爪退让位；请恢复原 T0 后续跑，不重新定义零点')
        self._permit_motion('已打印本批完整计划；确认固定 T0 与直线路径，输入 MOVE 开始。')
        self.baseline_folder = self.root / 'baseline' / str(time.time_ns())
        self.hw.begin_record(self.baseline_folder)
        try:
            self.hw.set_context('ZERO', 'batch_baseline')
            zero = self.hw.zero_unloaded()
            np.savez(self.baseline_folder / 'dd_reference.npz', left=self.hw.dd_reference[0], right=self.hw.dd_reference[1])
            zero['dd_reference_file'] = str(self.baseline_folder / 'dd_reference.npz')
            self.meta.setdefault('zero_history', []).append(zero)
            if not at_t0:
                anchor = self.context.get('last_grasp_target')
                expected = (self.context.get('last_safe_target') or
                            (self.bounds.pre(np.asarray(anchor)).tolist() if anchor is not None else None))
                if (anchor is None or expected is None or
                        self.bounds.check(np.asarray(expected), 'retreat', np.asarray(anchor)) or
                        np.linalg.norm(now[:3, 3] - np.asarray(expected)[:3, 3]) > tolerance):
                    raise RuntimeError('已记录退让位缺少合法抓取锚点，禁止自动返回')
                with self.stage('BATCH_RETURN_T0'):
                    self.move(self.T0)
            self.hw.keyframes('batch_T0_ready')
            self.context['last_safe_pose'] = self.hw.pose().tolist()
            self.context['last_safe_target'] = self.T0.tolist()
            self.context['last_grasp_target'] = self.T0.tolist()
            self.save()
        except BaseException as exc:
            self.recover(exc)
            try:
                self.hw.end_record(error=str(exc) or type(exc).__name__)
            except Exception:
                pass
            raise
        else:
            self.hw.end_record()
        self.initial_phase = False
        self.say('固定 T0 已核对，空载扣零完成；复用 campaign 零点漂移基准。')

    def zero_drift(self, probe):
        if not probe or not probe.get('valid') or self.current['delta_mm'] != 0:
            return
        values = [d['centroid_px'] for d in probe['descriptors']]
        self.current['zero_centroids_px'] = values
        samples = self.context.setdefault('zero_samples', [])
        if len(samples) < 3:
            self.current['zero_drift'] = {'checked': False, 'reason': 'collecting_first_three_campaign_zeros', 'reference_count': len(samples)}
            return
        baseline = self.context.setdefault('zero_baseline_px', np.mean([s['centroids_px'] for s in samples[:3]], axis=0).tolist())
        errors = None
        if self.fit and self.fit.get('usable_for_B'):
            if self.fit.get('schema') == 'agentic_fit_v1':
                from .agentic_learning import finger_estimate
                before = [finger_estimate(f, p) for f, p in zip(self.fit['fingers'], baseline)]
                after = [finger_estimate(f, p) for f, p in zip(self.fit['fingers'], values)]
                if all(v is not None for v in before + after):
                    errors = [float(b-a) for a, b in zip(before, after)]
            else:
                errors = [float((values[i][0 if f['coordinate'] == 'u' else 1] - baseline[i][0 if f['coordinate'] == 'u' else 1]) / f['slope_px_per_mm']) for i, f in enumerate(self.fit['fingers'])]
        if errors is None:
            self.current['zero_drift'] = {'checked': False, 'reason': 'no_valid_mm_mapping_for_baseline_and_current', 'baseline_centroids_px': baseline, 'current_centroids_px': values}
            return
        event = {'checked': True, 'errors_equivalent_mm': errors, 'baseline_centroids_px': baseline,
                 'slope_source': 'frozen_batch_model_inverse; TCP equivalent, not measured bottle displacement',
                 'alarm': max(abs(e) for e in errors) > self.cfg['design']['drift_mm']}
        self.current['zero_drift'] = event
        if event['alarm']:
            event['operator_response'] = self.read('零偏移质心超过 campaign 漂移阈值，请检查瓶子标记并复位；回车继续，q 暂停。')
            self.quit = event['operator_response'].lower() == 'q'
        self.save()

    def trial(self, item):
        self.args.cond = item['cond']
        self.correction = _correction(self.fit, self.cfg['servo']) if item['cond'] == 'B' else None
        super().trial(item)
        if self.current.get('status') != 'skipped_bound':
            self.context['last_safe_pose'] = self.hw.pose().tolist()
            self.context['last_grasp_target'] = self.current['final_tcp_target_mm']
            self.context['last_safe_target'] = self.bounds.pre(np.asarray(self.current['final_tcp_target_mm'])).tolist()
            self.context['last_bottle'] = item['bottle']
        if self.current.get('complete') and self.current.get('recording_complete') and self.current.get('label') != 'invalid' and self.current.get('zero_centroids_px'):
            samples = self.context.setdefault('zero_samples', [])
            key = str(self.folder.resolve())
            if not any(s['trial_path'] == key for s in samples):
                samples.append({'trial_path': key, 'centroids_px': self.current['zero_centroids_px']})
            if len(samples) >= 3:
                self.context.setdefault('zero_baseline_px', np.mean([s['centroids_px'] for s in samples[:3]], axis=0).tolist())
        self.save()

    def run_batch(self):
        done = self.completed()
        self.initial()
        self.print_plan(done)
        self._permit_motion('已打印本批完整偏移和基座变化，输入 MOVE 开始自动试验。')
        last_bottle = self.context.get('last_bottle')
        count = 0
        timed_out = False
        for item in self.batch_plan:
            if item['id'] in done:
                continue
            if deadline_reached(self.context):
                timed_out = True
                self.quit = True
                self.say('时间预算已耗尽；当前已完成试验保留，不再启动下一次运动。')
                break
            if item['bottle'] != last_bottle:
                self.read('请放置 ' + item['bottle'] + ' 瓶（同一标记），确认已开爪，按回车开始此块。')
                last_bottle = item['bottle']
            if deadline_reached(self.context):
                timed_out = True
                self.quit = True
                self.say('等待操作员期间时间预算已耗尽；本次试验尚未开始。')
                break
            self.trial(item)
            count += 1
            if self.quit:
                break
        return {'complete': len(self.completed()) == len(self.batch_plan), 'quit': self.quit,
                'context': self.export_context(), 'completed_count': count,
                'total_completed': len(self.progress['completed']), 'root': str(self.root),
                'deadline_reached': timed_out or deadline_reached(self.context)}


def execute_batch(root, plan, cfg, fit, context, *, simulate=False, ask=input, say=print,
                  authorize=True, hardware_factory=None):
    """Execute one frozen batch.  Completed batches never connect to hardware.

    ``authorize=False`` is effective only with ``continuous_campaign=True`` in
    the context issued by the still-running, preauthorized campaign process.
    A caller resuming after exit must pass authorize=True again.
    """
    root = Path(root).expanduser().resolve()
    validate_plan(plan, cfg, fit)
    context = copy.deepcopy(context or {})
    _validate_context(context)
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / '.agentic_batch.lock').open('a')
    hw = runner = None
    handlers = {}
    try:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('此批次已有执行进程') from exc
        path = root / 'session_meta.json'
        if path.exists():
            meta = json.loads(path.read_text())
            if meta.get('plan') != plain(plan) or meta.get('config') != plain(cfg) or meta.get('fit') != plain(fit) or meta.get('simulated') != bool(simulate):
                raise ValueError('续跑不能更换本批计划、配置、拟合或实机/模拟模式')
            if context.get('T0') is not None and meta.get('T0') is not None and not np.allclose(context['T0'], meta['T0'], atol=1e-8):
                raise ValueError('本批 T0 与 campaign 不一致')
            context = recover_context(root, context)
        else:
            atomic_json(path, _metadata(root, plan, cfg, fit, context, simulate))
        _validate_context(context)
        rows = rebuild_summary(root)
        done = {row['id'] for row in rows if row.get('complete')}
        if all(row['id'] in done for row in plan):
            say('本批已完成，跳过；不连接硬件、不发动作。')
            return {'complete': True, 'quit': False, 'context': recover_context(root, context),
                    'completed_count': 0, 'total_completed': len(done), 'root': str(root)}
        if context.get('unsafe_recovery') or context.get('airborne'):
            raise RuntimeError('campaign 有未解决的空中/恢复状态，禁止自动连接执行')
        if deadline_reached(context):
            say('时间预算已耗尽；不连接硬件、不发动作。')
            return {'complete': False, 'quit': True, 'context': context,
                    'completed_count': 0, 'total_completed': len(done),
                    'root': str(root), 'deadline_reached': True}
        if simulate:
            if hardware_factory is not None:
                raise ValueError('模拟模式不能注入实机工厂')
            from .agentic_sim import execute_simulated_batch
            return execute_simulated_batch(root, plan, cfg, fit, context, ask=ask, say=say)
        if hardware_factory is None:
            from .e1e2_hardware import Hardware
            hardware_factory = Hardware
        args = SimpleNamespace(out=root, cond=plan[0]['cond'], step=bool(cfg.get('agentic', {}).get('step', False)), limit=None)
        hw = hardware_factory(cfg)
        runner = BatchRunner(args, cfg, fit, hw, plan, context, ask=ask, say=say, authorize=authorize)
        if threading.current_thread() is threading.main_thread():
            def interrupted(signum, frame):
                raise KeyboardInterrupt('收到信号 ' + str(signum))
            for sig in (signal.SIGINT, signal.SIGTERM):
                handlers[sig] = signal.signal(sig, interrupted)
        hw.connect()
        return runner.run_batch()
    except BaseException as exc:
        if runner is not None:
            try:
                runner.recover(exc)
            finally:
                runner.save()
        atomic_json(root / 'errors' / f'{time.time_ns()}.json', {'error': str(exc) or type(exc).__name__, 'context': recover_context(root, context)})
        raise
    finally:
        if hw is not None:
            hw.close()
        for sig, old in handlers.items():
            signal.signal(sig, old)
        lock.close()
