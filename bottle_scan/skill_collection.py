"""Opt-in complete skill on existing E1/E2 guards. No ROS import."""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import time
from types import SimpleNamespace
import numpy as np
import yaml
from .agentic_executor import BatchRunner, execute_batch, deadline_reached
from .e1e2 import resolve_config, rebuild_summary
from .e1e2_core import load_settings, plan_trials
from .recovery_skill import RecoverySkill
from .session import atomic_json, plain

class SkillBatchRunner(BatchRunner):
    def observation(self, status, probe):
        p = probe or {}
        s = self.hw.snapshot()
        timestamp = p.get('timestamp_ns', s['timestamp_ns'])
        start = p.get('stable_start_ns', timestamp)
        settings = self.cfg['recovery_skill']
        return {
            'first_touch_complete': True, 'safety_ok': True,
            'timestamp_ns': int(timestamp), 'now_ns': int(self.hw.now()),
            'status': status, 'descriptors': copy.deepcopy(p.get('descriptors', [])),
            'bilateral': self.bilateral(s), 'force_stable': bool(
                p.get('valid') and status == 'contact' and
                np.all(np.isfinite(s['normal'])) and self.bilateral(s) and
                abs(float(np.mean(s['normal'])) - self.cfg['force']['target']*self.cfg['force']['probe_fraction'])
                <= self.cfg['force']['probe_tolerance']),
            'stable_duration_s': max(0., (timestamp-start)/1e9),
            'context': {**settings.get('context', {}),
                        **({'target_definition_id': settings['target_definition']['id']}
                           if settings.get('target_definition') else {}),
                        'probe_target_sdk': self.cfg['force']['target']*self.cfg['force']['probe_fraction']},
        }

    def record_observation(self, phase, observation, result=None):
        if self.current is None:
            return
        event = {'phase': phase, 'observation': plain(observation),
                 'skill_outputs': plain(result), 'raw_evidence': str(self.folder/'meta.json')}
        self.current.setdefault('skill_observations', []).append(event)
        self.save()

    def touch(self, prefix=''):
        status, probe = super().touch(prefix)
        if self.current is not None:
            self.record_observation('first_touch' if not self.current.get('skill_observations') else 'recovery_observation',
                                    self.observation(status, probe))
        if self.current is not None and status == 'probe_contact_failed' and self.correction:
            # Original Runner intentionally skips apply_B on failed first probe.
            # Preserve an explicit incomplete handoff for the skill consumer.
            spec = self.cfg['recovery_skill']
            skill = RecoverySkill(self.fit, self.cfg['servo'], spec.get('runtime'),
                                  spec.get('certificate'), mode=spec.get('mode', 'deployment'))
            result = skill.start(self.observation(status, probe))
            if not result['exit']['hand_back']:
                result = skill.abort('first_probe_contact_failed')
            self.current['skill_outputs'] = result
            self.current['skill_handoff'] = result['exit']
            self.current['servo_exhausted'] = result['exit']['reason'] == 'recovery_budget_exhausted'
            self.current['skill_incomplete'] = True
            self.current['skill_failure_reason'] = result['exit']['reason']
            self.record_observation('failed_first_touch', self.observation(status, probe), result)
        return status, probe

    def apply_B(self, target, status, probe):
        spec = self.cfg['recovery_skill']
        skill = RecoverySkill(self.fit, self.cfg['servo'], spec.get('runtime'),
                              spec.get('certificate'), mode=spec.get('mode', 'deployment'))
        observation = self.observation(status, probe)
        result = skill.start(observation)
        history = []
        observations_seen = 1
        while True:
            if not result['exit']['hand_back'] and (observations_seen >= 12 or deadline_reached(self.context)):
                result = skill.abort('observation_budget_exhausted' if observations_seen >= 12 else 'campaign_deadline')
            self.record_observation('skill_decision', observation, result)
            self.current['skill_outputs'] = result
            self.current['correction_history'] = history
            self.save()
            if result['exit']['hand_back']:
                history.append({'attempt': len(history), 'accepted': result['exit']['completed'],
                                'reason': result['exit']['reason'], 'timestamp_ns': self.hw.now(),
                                'terminal': True, 'skill_decision': result})
                self.current['correction_history'] = history
                self.current['skill_runtime_events'] = skill.events
                self.current['skill_handoff'] = result['exit']
                self.current['servo_exhausted'] = result['exit']['reason'] == 'recovery_budget_exhausted'
                self.current['skill_incomplete'] = not result['exit']['completed']
                self.current['skill_failure_reason'] = None if result['exit']['completed'] else result['exit']['reason']
                return (target, status, probe) if result['exit']['completed'] else (target, 'skill_incomplete', None)
            action = result['recovery']['action']
            if action is None:
                probe = self.regulate('PROBE', self.cfg['force']['target']*self.cfg['force']['probe_fraction'],
                                     self.cfg['force']['probe_tolerance'], self.cfg['force']['probe_stable_s'], with_dd=True)
                status = 'contact' if probe else 'probe_unstable'
                self.current.setdefault('probe_confirmations', []).append({'status': status, 'probe': probe})
                observations_seen += 1
                observation = self.observation(status, probe)
                result = skill.advance(observation)
                continue
            if action['axis'] not in getattr(self, 'allowed_recovery_axes', ('a',)):
                result = skill.abort('unsupported_axis_in_a_executor')
                continue
            delta = action['delta_mm']
            proposed = target.copy()
            proposed[:3,3] += getattr(self.bounds, action['axis'])*delta
            retreat = self.bounds.pre(target, self.cfg['servo']['retreat_mm'])
            corrected_pre = self.bounds.pre(proposed, self.cfg['servo']['retreat_mm'])
            error = (self.bounds.check(proposed) or self.bounds.check(retreat, 'retreat', target)
                     or self.bounds.check(corrected_pre, 'retreat', proposed))
            row = {'attempt': len(history), 'delta_mm': delta, 'accepted': False,
                   'timestamp_ns': self.hw.now(), 'skill_decision': result}
            history.append(row)
            if error:
                row.update(blocked=True, blocked_reason=error, applied_delta_mm=0.)
                result = skill.abort('motion_bounds:' + error)
                continue
            before = self.hw.pose()
            self.release()
            with self.stage('REGRASP_RETREAT'): self.move(retreat, 'retreat', target)
            with self.stage('REGRASP_CORRECT'): self.move(corrected_pre, 'retreat', proposed)
            with self.stage('REGRASP_APPROACH'): self.move(proposed)
            applied = float((self.hw.pose()[:3,3]-before[:3,3]) @ getattr(self.bounds, action['axis']))
            receipt = {'applied': True, 'action_mm': applied, 'completed_ns': int(self.hw.now())}
            row['applied_delta_mm'] = applied
            target = proposed
            status, probe = self.touch()
            self.current['regrasps'] += 1
            self.current.setdefault('probe_regrasps', []).append(
                {'status': status, 'probe': probe, 'target_tcp_mm': target.tolist(), 'receipt': receipt})
            observation = self.observation(status, probe)
            observations_seen += 1
            result = skill.advance(observation, applied_action=receipt)
            row['accepted'] = result['exit']['completed']

    def read(self, message):
        start = time.monotonic()
        try:
            if getattr(self, '_offline_label_review', False):
                return self.ask(message + '\n> ').strip()
            return super().read(message)
        finally:
            if self.current is not None and '复位' in message:
                self.current.setdefault('operator_waits', []).append(
                    {'reason': 'reset_and_confirmation', 'seconds': time.monotonic()-start,
                     'reset_only_seconds': None})

    def label(self):
        # Runner has released/retreated and ended recording before entering LABEL.
        # These prompts only read answers; they never issue robot commands.
        if not getattr(self, '_offline_label_review', False):
            if self.current.get('recording_complete') and not self.airborne:
                self.current['skill_label_pending'] = True
                self.current['skill_physical_finished'] = True
                self.context['last_safe_pose'] = self.hw.pose().tolist()
                self.context['last_grasp_target'] = self.current['final_tcp_target_mm']
                self.context['last_safe_target'] = self.bounds.pre(np.asarray(self.current['final_tcp_target_mm'])).tolist()
                self.context['last_bottle'] = self.current['bottle']
                # Release and retreat already completed. Interrupting review must
                # not emit recovery commands or replay the physical trial.
                self.engaged = False
                self.save()
        super().label()
        definition = self.cfg['recovery_skill'].get('target_definition')
        self.say('以下独立标签用于验证调用与退出；不确定时回车，不能根据计划偏移代填。')
        if definition:
            self.say('目标定义 ' + definition['id'] + '：' + definition['description'])
            self.say('无需恢复：' + definition['normal'])
            self.say('需要恢复：' + definition.get('needs_recovery', definition.get('needs', '')))
            self.say('达到目标：' + definition['completion'])
            self.say('仅仅夹在触觉传感器上、有双侧接触，不等于位置已经正常；请按目标区域与接触形态判断。')
            self.say('当前仅做记录回看和标签登记；不会为了判断标签重新轻触或移动。看不清请留未知。')
        initial = self.read('回看首次轻触：1需要恢复，2已经正常，3暂不适用，回车未知：')
        final = self.read('交回控制权前是否达到稳定双侧内部目标接触？1是，2否，回车未知：')
        self.current['skill_ground_truth'] = {
            'source': 'operator_review_of_observed_contact',
            'initial_class': {'1':'needs_recovery','2':'already_normal','3':'not_applicable'}.get(initial),
            'final_target_reached': {'1': True, '2': False}.get(final),
            'independent_of_experimental_offset': True}
        if definition:
            self.current['skill_ground_truth']['target_definition_id'] = definition['id']
            unique = {}
            for event in self.current.get('skill_observations', []):
                observation = event.get('observation') or {}
                ts = observation.get('timestamp_ns')
                if type(ts) is int and ts not in unique:
                    unique[ts] = event
            self.current['skill_window_labels'] = []
            for ts, event in sorted(unique.items()):
                observation = event['observation']
                centroids = [d.get('centroid_px') for d in observation.get('descriptors', []) if isinstance(d, dict)]
                self.say('回看窗口 ' + str(ts) + '（' + str(event.get('phase')) + '），记录：' +
                         str(event.get('raw_evidence')) + '，双指质心=' + str(centroids) +
                         '，力稳定=' + str(observation.get('force_stable')))
                value = self.read('只判断这个时间窗口是否达到上述目标接触（不要按模型输出或偏移推断）：1是，2否，回车未知：')
                self.current['skill_window_labels'].append({
                    'timestamp_ns': ts, 'target_reached': {'1': True, '2': False}.get(value),
                    'target_definition_id': definition['id'],
                    'source': 'operator_review_of_observed_contact',
                    'independent_of_experimental_offset': True})
        self.save()

    def trial(self, item):
        started = time.monotonic()
        try:
            super().trial(item)
        finally:
            if self.current is not None:
                if self.current.get('complete'):
                    self.current['skill_label_pending'] = False
                stages = self.current.get('stages', [])
                role = self.context.get('batch_role', 'exploration')
                self.current['skill_cost'] = {
                    'schema': 'skill_trial_cost_v1', 'wall_seconds': time.monotonic()-started,
                    'robot_stage_seconds': sum(max(0., (s['end_ns']-s['start_ns'])/1e9)
                                               for s in stages if s.get('end_ns')),
                    'operator_reset_confirmation_seconds': sum(x['seconds'] for x in self.current.get('operator_waits', [])),
                    'reset_only_seconds': None, 'attempts': 1,
                    'A_exploration_attempts': int(item['cond']=='A'),
                    'B_development_attempts': int(item['cond']=='B' and role!='final_validation'),
                    'B_independent_attempts': int(item['cond']=='B' and role=='final_validation'),
                    'failed_attempts': int(not self.current.get('complete') or self.current.get('label') != 'normal' or not self.current.get('hold_complete')),
                    'regrasp_actions': self.current.get('regrasps', 0),
                    'source_task_cost': False, 'role': role,
                    'missing_time_is_not_zero': True}
                self.save()



def finish_pending_labels(root, plan, cfg, fit, context, *, ask=input, say=print):
    """Resume a proven post-retreat review without connecting any hardware.

    A pending review is an explicit new-format marker, never inferred from a
    recording alone. This invocation only finishes labels; further motion needs
    another explicit collect invocation.
    """
    root = Path(root)
    pending = []
    for item in plan:
        path = root/'trials'/item['id']/'meta.json'
        if not path.exists():
            continue
        row = json.loads(path.read_text())
        if row.get('complete') or row.get('skill_label_pending') is not True:
            continue
        if (row.get('skill_physical_finished') is not True or
                row.get('recording_complete') is not True or row.get('error') or
                row.get('final_tcp_target_mm') is None):
            raise ValueError('pending skill label lacks a safely completed recorded trial')
        pending.append((path, row))
    if not pending:
        return None
    args = SimpleNamespace(out=root, cond=plan[0]['cond'], step=False, limit=None)
    reviewer = SkillBatchRunner(args, cfg, fit, None, plan, context, ask=ask, say=say)
    reviewer._offline_label_review = True
    for path, row in pending:
        reviewer.current = row
        reviewer.folder = path.parent
        reviewer.args.cond = row['cond']
        say('试验 ' + row['id'] + ' 已完成动作；此次只补齐标签，不连接硬件，不重新采集。')
        reviewer.label()
        row.update(complete=True, skill_label_pending=False, finished_unix_ns=time.time_ns())
        if row.get('skill_cost'):
            row['skill_cost']['failed_attempts'] = int(row.get('label') != 'normal' or not row.get('hold_complete'))
        reviewer.progress['active'] = None
        if row['id'] not in reviewer.progress['completed']:
            reviewer.progress['completed'].append(row['id'])
        reviewer.save()
    done = reviewer.completed()
    return {'complete': len(done) == len(plan), 'quit': reviewer.quit,
            'context': reviewer.export_context(), 'completed_count': 0,
            'labelled_existing_count': len(pending), 'labels_only': True,
            'total_completed': len(done), 'root': str(root)}


def selection_plan(selection, cfg, cond):
    """Compile a frozen development recommendation; no truth enters runtime."""
    if (selection.get('schema') != 'skill_experiment_selection_v1' or
        selection.get('role') != 'development' or selection.get('stage') != 'a_closed_loop' or
        selection.get('status') != 'planned_unexecuted'):
        raise ValueError('only unexecuted a-axis development selections can collect')
    rows = []
    for choice in selection.get('selected', []):
        condition = choice['condition']
        if condition.get('contact_height_l_mm') != 0 or not np.isclose(
                condition.get('probe_force', float('nan')),
                cfg['force']['target']*cfg['force']['probe_fraction']):
            raise ValueError('use skill-extension for changed height/probe force')
        reps = choice.get('reps')
        if type(reps) is not int or reps < 2 or reps % 2:
            raise ValueError('selection repetitions must balance bottle blocks')
        if choice.get('bottle_blocks') != cfg['design']['bottles']:
            raise ValueError('selection bottle blocks differ from protocol')
        delta = condition.get('a_mm')
        if isinstance(delta,bool) or not isinstance(delta,(int,float)) or not np.isfinite(delta) or abs(delta)>cfg['motion']['max_grasp_offset_mm']:
            raise ValueError('selection exceeds a bounds')
        for repeat in range(reps//2):
            for bottle in cfg['design']['bottles']:
                rows.append({'id':f'{len(rows)+1:05d}', 'index':len(rows)+1,
                    'cond':cond, 'kind':'scan', 'bottle':bottle, 'repeat':repeat,
                    'delta_mm':float(delta),'purpose':choice['purpose'],
                    'selection_reason':choice['reason'],'evidence_refs':choice.get('problem_refs',[])})
    if not rows or len(rows) != selection.get('planned_trial_attempts'):
        raise ValueError('selection count differs from frozen interaction budget')
    # Explicit reference repeats are part of the budget, never hidden extra trials.
    if not any(r['delta_mm'] == 0 for r in rows):
        raise ValueError('selection must include an explicit budgeted reference repeat')
    every = cfg['design']['zero_every']
    if every:
        since_reference=0
        for row in rows:
            since_reference = 0 if row['delta_mm']==0 else since_reference+1
            if since_reference>every:
                raise ValueError('selection reference gap exceeds original monitoring interval')
    return rows

def load_skill_config(path):
    spec = yaml.safe_load(Path(path).read_text())
    if spec.get('schema') != 'bottle_skill_config_v1':
        raise ValueError('expected bottle_skill_config_v1')
    definition = spec.get('target_definition')
    if isinstance(definition, dict) and 'needs_recovery' not in definition and 'needs' in definition:
        definition['needs_recovery'] = definition.pop('needs')
    if definition is not None or spec.get('learned_heads'):
        if not isinstance(definition, dict) or any(
                not isinstance(definition.get(k), str) or not definition[k].strip()
                for k in ('id', 'description', 'normal', 'needs_recovery', 'completion')):
            raise ValueError('learned target_definition requires id, description, normal, needs_recovery, completion')
        declared = spec.get('learning', {}).get('target_definition_id')
        if declared is not None and declared != definition['id']:
            raise ValueError('learning target_definition_id differs from annotation protocol')
    return spec

def main(argv=None):
    parser = argparse.ArgumentParser(description='药瓶完整 skill：离线计划 / 显式实机 A/B 采集')
    parser.add_argument('command', choices=['plan','collect','train'])
    parser.add_argument('--config', type=Path, default=Path('config.skill.bottle.yaml'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--cond', choices=['A','B'], default='A')
    parser.add_argument('--role', choices=['exploration','development','final_validation'], default='development')
    parser.add_argument('--fit', type=Path)
    parser.add_argument('--session', type=Path, action='append', help='A/development evidence for training; final validation excluded')
    parser.add_argument('--selection', type=Path, help='frozen development selection.json; includes reference cost')
    parser.add_argument('--reference-session', type=Path, help='reuse original accepted T0 and sensing protocol for new A/B')
    parser.add_argument('--certificate', type=Path)
    parser.add_argument('--mode', choices=['deployment','development'], default='deployment')
    parser.add_argument('--real', action='store_true')
    parser.add_argument('--min-normalized', type=float)
    parser.add_argument('--learning-mode', choices=['legacy', 'joint'], default='legacy')
    parser.add_argument('--limit', type=int, help='maximum newly completed trial attempts in this invocation; resume keeps frozen plan')
    parser.add_argument('--step', action='store_true', default=None, help='confirm each existing guarded motion stage')
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        parser.error('--limit must be a positive integer')
    if args.command == 'collect' and not args.real:
        parser.error('collect requires --real; plan never opens hardware')
    spec = load_skill_config(args.config)
    if args.command == 'train':
        from .contact_policy import train_contact_policy
        if not args.session:
            parser.error('train requires at least one --session')
        if args.out.exists():
            raise ValueError('model versions cannot be overwritten')
        if args.learning_mode == 'joint':
            from .learned_skill import train_learned_skill
            learning = copy.deepcopy(spec.get('learning', {}))
            if not spec.get('target_definition'):
                raise ValueError('joint learning requires an explicit target_definition annotation protocol')
            learning['target_definition_id'] = spec['target_definition']['id']
            model = train_learned_skill(args.session, learning, out=args.out)
        else:
            model = train_contact_policy(args.session, spec.get('learning', {}), out=args.out)
        print(json.dumps({'out':str(args.out), 'usable_for_B':model['usable_for_B'],
                          'blocking_reasons':model['blocking_reasons']}, ensure_ascii=False))
        return 0 if model['usable_for_B'] else 2
    base = args.config.resolve().parent/spec['base_config']
    cfg = load_settings(base)
    cfg['design'].update(spec['collection'])
    selection = json.loads(args.selection.read_text()) if args.selection else None
    if selection and args.role == 'final_validation':
        raise ValueError('feedback selection cannot define independent evaluation conditions')
    if args.command == 'plan':
        atomic_json(args.out, {'schema':'skill_batch_preview_v1','plan':selection_plan(selection,cfg,args.cond) if selection else plan_trials(cfg,args.cond),
                              'role':args.role,'mode':args.mode,'runtime':spec['runtime'],
                              'hardware_connected':False,'fit_required_for_B':True})
        print(args.out)
        return 0
    fit = json.loads(args.fit.read_text()) if args.fit else None
    if args.cond == 'B' and fit is None:
        parser.error('B requires --fit frozen ContactPolicy/Agentic model')
    existing = args.out/'session_meta.json'
    settings = {'runtime':spec['runtime'], 'context':spec['context'], 'mode':args.mode,
                'certificate':json.loads(args.certificate.read_text()) if args.certificate else None,
                'certification':spec.get('certification', {}), 'collection_selection': selection}
    if spec.get('target_definition'):
        settings['target_definition'] = copy.deepcopy(spec['target_definition'])
    if existing.exists():
        saved = json.loads(existing.read_text())
        if saved['config'].get('recovery_skill') != settings or saved.get('fit') != fit:
            raise ValueError('resume cannot change skill, mode, context, certificate, or model')
        if saved.get('agentic',{}).get('role') != args.role:
            raise ValueError('resume cannot change development/independent role')
        cfg = saved['config']
        if any(cfg['design'][key] != value for key,value in spec['collection'].items()):
            raise ValueError('resume cannot change plan')
        plan = saved['plan']
        context = saved.get('agentic_context', {})
    else:
        resolve_args = SimpleNamespace(out=args.out, config=base, cond='A',
            offsets=cfg['design']['offsets_mm'], reps=cfg['design']['reps'], seed=cfg['design']['seed'],
            min_normalized=args.min_normalized, target=None, f_max=None, fit=None, trial_run=False)
        cfg, _ = resolve_config(resolve_args)
        cfg['recovery_skill'] = settings
        plan = selection_plan(selection,cfg,args.cond) if selection else plan_trials(cfg, args.cond)
        context = {'batch_role':args.role}
        if fit is not None:
            if fit.get('reference_T0') is None:
                raise ValueError('frozen model requires its measured reference_T0')
            context.update(T0=fit['reference_T0'], T0_source_accepted=True)
    if args.reference_session:
        from .agentic_learning import _hash, _signature
        ref = json.loads((args.reference_session/'session_meta.json').read_text())
        if ref.get('T0') is None or _hash(_signature(ref)) != _hash(_signature({'config':cfg})):
            raise ValueError('reference session missing T0 or incompatible sensing protocol')
        if context.get('T0') is not None and not np.allclose(context['T0'],ref['T0'],atol=1e-8):
            raise ValueError('reference T0 differs from frozen model/session')
        context.update(T0=ref['T0'], T0_source_accepted=True)
    print(json.dumps(execute_batch(args.out, plan, cfg, fit, context, limit=args.limit, step=args.step), ensure_ascii=False))
    return 0
