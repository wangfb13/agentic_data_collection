"""Persistent method experiments over the existing, bounded E1/E2 executor.

Training, method generation, evaluation and promotion have separate provenance.
This module never publishes ROS commands. Only explicit run/resume --real can
call the existing executor; reports, memory and training are entirely offline.
"""
from __future__ import annotations

import argparse
import copy
import fcntl
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import yaml

from .session import atomic_json, fingerprint, plain

ROOT = Path(__file__).resolve().parents[1]
TERMINAL = {'complete', 'budget_exhausted'}


class PolicyTrainingUnavailable(ValueError):
    def __init__(self, model):
        self.model_id = model['model_id']
        self.reasons = model.get('blocking_reasons', [])
        super().__init__('ContactPolicy 尚未达到离线验证门禁：' + str(self.reasons) +
                         '；验证=' + str(model.get('validation', {})))


def load_settings(path):
    path = Path(path).expanduser().resolve()
    s = yaml.safe_load(path.read_text())
    if not isinstance(s, dict) or s.get('schema') != 'rsi_config_v1':
        raise ValueError('需要 rsi_config_v1 配置')
    for container, key in ((s, 'base_config'), (s['memory'], 'path')):
        value = Path(container[key]).expanduser()
        container[key] = str(value if value.is_absolute() else path.parent / value)
    s['seed_sessions'] = [str(Path(p).expanduser().resolve()) for p in s.get('seed_sessions', [])]
    validate_settings(s)
    return s


def validate_settings(s):
    from .agentic import finite
    lo, hi = (finite(s['scope'][k], k) for k in ('min_mm', 'max_mm'))
    if not -25 <= lo < hi <= 25 or not lo <= 0 <= hi:
        raise ValueError('RSI 比较范围必须包含零点且在 [-25,25] mm 内')
    if finite(s['scope']['step_mm'], 'step_mm') < .5:
        raise ValueError('名义采样间隔不得小于 0.5 mm；不代表实测精度')
    for section in ('bootstrap', 'comparison', 'final_validation'):
        v = s[section]
        if type(v['reps']) is not int or v['reps'] < 2 or v['reps'] % 2:
            raise ValueError(section + '.reps 必须为正偶数，两瓶平衡')
        points = v['offsets_mm']
        if not points or len(set(points)) != len(points):
            raise ValueError(section + ' 测点为空或重复')
        for point in points:
            if not lo <= finite(point, 'offset') <= hi:
                raise ValueError(section + ' 测点超出固定比较范围')
    for name in ('collection_trials_per_branch', 'batch_scan_trials'):
        v = s['comparison'][name]
        if type(v) is not int or v < 2 or v % 2:
            raise ValueError(name + ' 必须为正偶数')
    checkpoints = s['comparison'].get('checkpoints', [s['comparison']['collection_trials_per_branch']])
    if (not checkpoints or any(type(n) is not int or n < 2 or n % 2 for n in checkpoints)
            or sorted(set(checkpoints)) != checkpoints or checkpoints[-1] != s['comparison']['collection_trials_per_branch']):
        raise ValueError('checkpoints 必须为递增正偶数，最后一个等于每分支总采集数')
    for key in ('max_trials', 'max_cycles', 'max_seconds'):
        if type(s['budget'][key]) is not int or s['budget'][key] < 1:
            raise ValueError('budget.' + key + ' 必须为正整数')
    if s['memory']['mode'] not in ('recursive', 'fixed', 'none'):
        raise ValueError('memory.mode 必须为 recursive/fixed/none')
    if s.get('approval', 'campaign') not in ('campaign', 'batch'):
        raise ValueError('approval 必须为 campaign/batch')
    if s['provider']['backend'] not in ('chatgpt', 'api', 'deterministic', 'mock'):
        raise ValueError('未知 Agent provider')
    if type(s.get('seed')) is not int:
        raise ValueError('seed 必须为整数')
    if s.get('problem') is not None and (not isinstance(s['problem'], str) or not s['problem'].strip() or len(s['problem']) > 1000):
        raise ValueError('problem 必须为非空问题描述，最长1000字符')
    if type(s['comparison']['min_trials']) is not int or s['comparison']['min_trials'] < 20:
        raise ValueError('min_trials 至少为 20；少量成功不自动晋升方法')
    if not 0 <= finite(s['comparison']['min_effect'], 'min_effect') < 1:
        raise ValueError('min_effect 必须位于 [0,1)')


def plan_for_cells(cells, cfg, *, cond, seed, monitor_start=0):
    """Paired bottles with explicit global monitor accounting across chunks."""
    units = []
    rng = random.Random(seed)
    for cell in cells:
        if type(cell['reps']) is not int or cell['reps'] < 2 or cell['reps'] % 2:
            raise ValueError('采集方法必须产生每档正偶数次数')
        for repeat in range(cell['reps'] // 2):
            pair = [{'cond': cond, 'bottle': b, 'delta_mm': float(cell['offset_mm']),
                     'repeat': repeat, 'kind': 'scan'} for b in cfg['design']['bottles']]
            rng.shuffle(pair)
            units.append(pair)
    rng.shuffle(units)
    result = []
    count = monitor_start
    every = cfg['design']['zero_every']
    for pair in units:
        for row in pair:
            result.append(row)
            count += 1
            if every and count % every == 0:
                result.append({**row, 'delta_mm': 0., 'kind': 'zero_monitor', 'repeat': None})
    for i, row in enumerate(result, 1):
        row.update(id=f'{i:05d}', index=i)
    return result


def check_targets(plan, cfg, T0):
    if T0 is None:
        return  # Executor rechecks every target after operator records T0.
    from .e1e2_core import MotionBounds
    bounds = MotionBounds(cfg, np.asarray(T0, dtype=float))
    for row in plan:
        target = bounds.grasp(row['delta_mm'])
        error = (bounds.check(target) or bounds.check(bounds.pre(target), 'retreat', target)
                 or bounds.check(bounds.lift(target), 'lift', target))
        if error:
            raise ValueError(f"计划 δ={row['delta_mm']:g} 被现有限幅否决：{error}")


def trial_rows(roots):
    rows = []
    for root in roots:
        for path in sorted((Path(root) / 'trials').glob('*/meta.json')):
            row = json.loads(path.read_text())
            status = path.parent / 'recording_status.json'
            row['verified_recording'] = bool(status.exists() and json.loads(status.read_text()).get('complete'))
            rows.append(row)
    return rows


def summarize(roots, cells):
    rows = [r for r in trial_rows(roots) if r.get('kind') == 'scan']
    grouped = []
    # A restarted evaluation is not a fresh replacement for a failed attempt.
    complete = not any(list((Path(root) / 'interrupted').glob('*/meta.json')) for root in roots)
    for cell in cells:
        group = [r for r in rows if r.get('bottle') == cell['bottle'] and
                 abs(float(r['delta_mm']) - cell['offset_mm']) < 1e-8]
        valid = [r for r in group if r.get('complete') and r.get('recording_complete')
                 and r['verified_recording'] and r.get('label') != 'invalid']
        successes = sum(bool(r.get('hold_complete') and r.get('lifted') and not r.get('dropped')
                             and r.get('label') == 'normal') for r in valid)
        complete &= len(group) == cell['trials'] == len(valid)
        grouped.append({**cell, 'trials': len(valid), 'successes': successes})
    complete &= len(rows) == sum(c['trials'] for c in cells)
    return {'successes': sum(c['successes'] for c in grouped),
            'trials': sum(c['trials'] for c in grouped), 'evaluation_cells': grouped,
            'records_complete': bool(complete),
            'mean_regrasps': float(np.mean([r.get('regrasps', 0) for r in rows])) if rows else None}


class RSICampaign:
    def __init__(self, root, *, settings=None, ask=input, say=print, executor=None, proposer=None):
        from .agentic import resolve_base, source_manifest, Campaign
        from .rsi_methods import MethodStore, baseline_method
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / 'rsi.json'
        self.ask, self.say, self.executor, self.proposer = ask, say, executor, proposer
        self.first_motion = True
        self.started = None
        if self.path.exists():
            self.state = json.loads(self.path.read_text())
            if self.state.get('schema') != 'rsi_campaign_v1':
                raise ValueError('未知 RSI 会话格式')
            if settings is not None and fingerprint(settings) != fingerprint(self.state['settings']):
                raise ValueError('续跑不能修改 RSI 设置；请新建目录')
            if source_manifest(self.state['seed_sessions']) != self.state['seed_manifest']:
                raise ValueError('原始种子数据被修改，禁止静默续跑')
        else:
            if settings is None:
                raise ValueError('RSI 会话不存在；请使用 run 新建')
            if (self.root / 'campaign.json').exists() or (self.root / 'session_meta.json').exists():
                raise ValueError('请为 RSI 使用新的输出目录；旧实验可作为 seed-session')
            validate_settings(settings)
            cfg = resolve_base(settings, self.root, ask, say)
            seeds = settings.get('seed_sessions', [])
            context = {}
            for seed in seeds:
                meta = json.loads((Path(seed) / 'session_meta.json').read_text())
                if bool(meta.get('simulated')) != bool(settings.get('simulate')):
                    raise ValueError('模拟和实机数据不能混用')
                if not context:
                    context = {k: meta[k] for k in ('T0', 'initial_centroids', 'T0_accepted_unix_ns') if k in meta}
                    context['source_T0_session'] = seed
            hardware = Campaign.hardware_manifest(cfg, settings)
            # Protocol identity excludes paths and experiment conditions.
            from .agentic_learning import _signature
            signature = fingerprint({'hardware': sorted(hardware.values()),
                                     'protocol': _signature({'config': cfg, 'simulated': settings.get('simulate')})})
            self.state = {'schema': 'rsi_campaign_v1', 'settings': plain(settings), 'base_config': plain(cfg),
                          'task_id': settings.get('task_id') or fingerprint(str(self.root))[:20],
                          'status': 'ready', 'phase': 'bootstrap' if not seeds else 'initial_train',
                          'context': context, 'seed_sessions': seeds, 'seed_manifest': source_manifest(seeds),
                          'base_sessions': seeds[:], 'batches': [], 'cycles': [], 'active_batch': None,
                          'elapsed_s': 0., 'created_unix_ns': time.time_ns(),
                          'hardware_manifest': hardware, 'hardware_signature': signature,
                          'initial_policy_id': None, 'current_policy_id': None, 'models': {}}
            atomic_json(self.path, self.state)
        self.settings, self.cfg = self.state['settings'], self.state['base_config']
        self.store = MethodStore(self.settings['memory']['path'])
        if 'incumbent_method_id' not in self.state:
            self.state['incumbent_method_id'] = self.store.register(baseline_method())
            self.state['memory_snapshot'] = (self.store.retrieve(self.memory_context(), limit=self.settings['memory']['retrieval_limit'])
                                             if self.settings['memory']['mode'] != 'none' else [])
            # Only independently evaluated methods may become an incumbent.
            self.state['recommended_methods'] = [entry['method']['method_id'] for entry in self.state['memory_snapshot']
                                                 if entry.get('status') in ('current_task_valid', 'cross_task_valid')]
            self.state['fixed_method_id'] = next(iter(self.state['recommended_methods']), self.state['incumbent_method_id'])
            self.save()

    def save(self):
        atomic_json(self.path, self.state)

    def memory_context(self):
        return {'task_id': self.state['task_id'], 'simulated': bool(self.settings.get('simulate')),
                'hardware_signature': self.state['hardware_signature'],
                'scope': self.settings['scope'], 'problem': self.settings.get('problem') or 'a-axis tactile recovery acquisition efficiency',
                'explicit_problem': bool(self.settings.get('problem')),
                'problem_signature': fingerprint({'problem': self.settings['problem'], 'scope': self.settings['scope']}) if self.settings.get('problem') else None}

    def spent(self, batch_ids=None):
        result = 0
        for b in self.state['batches']:
            if batch_ids is not None and b['id'] not in batch_ids:
                continue
            root = self.root / b['path']
            result += len(list(root.glob('trials/*/meta.json'))) + len(list(root.glob('interrupted/*/meta.json')))
        return result

    def elapsed(self):
        return self.state['elapsed_s'] + (time.monotonic() - self.started if self.started is not None else 0.)

    def model(self, identifier):
        if identifier is None:
            return None
        result = json.loads((self.root / self.state['models'][identifier]).read_text())
        if result.get('model_id') != identifier:
            raise ValueError('模型版本与注册信息不一致')
        if result.get('usable_for_B'):
            from .contact_policy import ContactPolicy
            ContactPolicy(result, self.cfg['servo'])
        return result

    def train(self, roots):
        from .contact_policy import train_contact_policy
        training_settings = copy.deepcopy(self.settings)
        if self.state.get('frozen_validation_groups'):
            training_settings.setdefault('contact_policy', {})['frozen_validation_groups'] = self.state['frozen_validation_groups'][:]
        model = train_contact_policy(roots, training_settings)
        path = f"models/{model['model_id']}.json"
        atomic_json(self.root / path, model)
        self.state['models'][model['model_id']] = path
        self.save()
        if not model.get('usable_for_B'):
            raise PolicyTrainingUnavailable(model)
        return model['model_id']

    def analysis(self, roots):
        from .agentic_learning import analyze_sessions
        result = analyze_sessions(roots, self.settings)
        issues = result.get('quality', {}).get('blocking_reasons', [])
        if issues:
            raise ValueError('采集证据不兼容：' + '; '.join(map(str, issues)))
        # Only a PREVIOUS completed development comparison may inform a later
        # acquisition. Its B data remain excluded from policy training. The
        # current comparison and final held-out trials are never peeked at.
        for cycle in self.state['cycles']:
            if cycle.get('phase') != 'complete':
                continue
            for key, summary in cycle.get('comparison', {}).items():
                if key not in ('old', 'new'):
                    continue
                if summary.get('model_id') != self.state['current_policy_id']:
                    continue
                for cell in summary.get('evaluation_cells', []):
                    result['diagnostics'].append({'code': 'B_offset_feedback', 'offset_mm': cell['offset_mm'],
                        'n': cell['trials'], 'failures': cell['trials'] - cell['successes'],
                        'cycle': cycle['id'], 'branch': key, 'source': 'completed_previous_development_evaluation'})
        return result

    def check_cycle_origin(self, cycle):
        from .agentic import source_manifest
        if fingerprint(source_manifest(cycle['base_sessions'])) != cycle['protocol']['initial_dataset_hash']:
            raise ValueError('方法对照的共同数据起点已被修改，拒绝继续训练或比较')
        if fingerprint(self.model(cycle['base_policy_id'])) != cycle['protocol']['initial_policy_hash']:
            raise ValueError('方法对照的共同基准策略被修改')
        if self.settings.get('contact_policy', {}) != cycle['protocol']['training_budget']:
            raise ValueError('方法对照的预注册训练预算被修改')
        if fingerprint(self.state.get('frozen_validation_groups', [])) != cycle['protocol'].get('offline_validation_groups_hash', fingerprint([])):
            raise ValueError('方法对照的离线留出分组被修改')

    def batch(self, plan, *, phase, branch=None, cycle=None, model_id=None):
        from .agentic_executor import validate_plan
        validate_plan(plan, self.cfg, self.model(model_id))
        check_targets(plan, self.cfg, self.state['context'].get('T0'))
        item = {'id': f'{len(self.state["batches"])+1:05d}', 'phase': phase, 'branch': branch,
                'cycle': cycle, 'plan': plain(plan), 'model_id': model_id, 'status': 'planned'}
        item['path'] = 'batches/' + item['id']
        item['plan_sha256'] = fingerprint(plan)
        self.state['batches'].append(item)
        atomic_json(self.root / item['path'] / 'batch_plan.json', item)
        return item

    def roots(self, ids):
        return [str(self.root / b['path']) for b in self.state['batches'] if b['id'] in ids]

    def execute(self, b):
        from .agentic import Campaign
        from .agentic_executor import execute_batch, recover_context
        if b['status'] == 'complete':
            return True
        if fingerprint(b['plan']) != b['plan_sha256']:
            raise ValueError('已冻结批次计划被修改')
        if Campaign.hardware_manifest(self.cfg, self.settings) != self.state['hardware_manifest']:
            raise ValueError('硬件配置变化，需新建任务并检查兼容性')
        fit = self.model(b['model_id'])
        if fit:
            from .agentic_learning import _signature, _hash, _pose, _rotation_deg
            if fit.get('signature') != _hash(_signature({'config': self.cfg, 'simulated': self.settings.get('simulate')})):
                raise ValueError('策略与当前力/DD/机械臂协议不兼容')
            if bool(fit.get('simulated')) != bool(self.settings.get('simulate')):
                raise ValueError('禁止模拟策略用于实机或反向混用')
            original, current = _pose(fit.get('reference_T0')), _pose(self.state['context'].get('T0'))
            if original is None or current is None or np.linalg.norm(original[:3, 3]-current[:3, 3]) > 1e-7 or _rotation_deg(original[:3, :3], current[:3, :3]) > 1e-5:
                raise ValueError('策略的固定 T0 与会话不一致')
        context = recover_context(self.root / b['path'], self.state['context'])
        if context.get('airborne') or context.get('unsafe_recovery'):
            raise ValueError('存在未处理的空中/恢复状态；先现场处理，再运行 rsi recover')
        context.update(continuous_campaign=not self.first_motion, campaign_root=str(self.root),
                       batch_role='validation' if b['phase'] in ('evaluation', 'final_validation') else 'exploration',
                       deadline_unix_ns=time.time_ns() + int(max(0, self.settings['budget']['max_seconds'] - self.elapsed())*1e9))
        remaining = sum(r['id'] not in {v['id'] for v in trial_rows([self.root / b['path']]) if v.get('complete')} for r in b['plan'])
        if self.spent() + remaining > self.settings['budget']['max_trials']:
            raise ValueError('中断尝试也计入预算；剩余预算不足以完成已冻结批次')
        cfg = copy.deepcopy(self.cfg)
        cfg['agentic'] = {**cfg.get('agentic', {}), 'rsi': True, 'task_id': self.state['task_id'],
                          'cycle': b['cycle'], 'branch': b['branch'], 'role': context['batch_role']}
        self.state.update(status='running', active_batch=b['id'])
        b['status'] = 'running'
        self.save()
        self.say(f"RSI {b['phase']} / {b['branch'] or 'shared'}：{len(b['plan'])} 次；数据 {b['path']}")
        try:
            result = (self.executor or execute_batch)(self.root / b['path'], b['plan'], cfg,
                         self.model(b['model_id']), context, simulate=bool(self.settings.get('simulate')),
                         ask=self.ask, say=self.say,
                         authorize=self.first_motion or self.settings.get('approval') == 'batch')
        finally:
            self.state['context'] = recover_context(self.root / b['path'], context)
            self.state['context'].pop('continuous_campaign', None)
            self.save()
        self.first_motion = False
        self.state['context'] = plain(result.get('context', self.state['context']))
        self.state['context'].pop('continuous_campaign', None)
        b['status'] = 'complete' if result.get('complete') else 'paused'
        self.state['active_batch'] = None if result.get('complete') else b['id']
        if result.get('quit') or not result.get('complete'):
            self.state['status'] = 'paused'
        self.save()
        return bool(result.get('complete') and not result.get('quit'))

    def cells(self, section):
        value = self.settings[section]
        return [{'offset_mm': float(d), 'bottle': b, 'trials': value['reps']//2}
                for d in value['offsets_mm'] for b in self.cfg['design']['bottles']]

    def scheduled_count(self, scans):
        every = self.cfg['design']['zero_every']
        return scans + (scans // every if every else 0)

    def final_cost(self):
        v = self.settings['final_validation']
        return self.scheduled_count(len(v['offsets_mm']) * v['reps'] * 2)

    def cycle_cost(self):
        v = self.settings['comparison']
        return 2 * (self.scheduled_count(v['collection_trials_per_branch']) +
                    self.scheduled_count(len(v['offsets_mm']) * v['reps']) * len(self.checkpoints()))

    def checkpoints(self):
        return self.settings['comparison'].get('checkpoints', [self.settings['comparison']['collection_trials_per_branch']])

    def checkpoint_target(self, cycle):
        return self.checkpoints()[cycle.get('checkpoint_index', 0)]

    def propose(self):
        from .rsi_methods import cell_features
        from .rsi_proposer import propose_method
        report = self.analysis(self.state['base_sessions'])
        memory = self.store.retrieve(self.memory_context(), limit=self.settings['memory']['retrieval_limit']) if self.settings['memory']['mode'] == 'recursive' else self.state['memory_snapshot']
        if self.settings['memory']['mode'] == 'none':
            memory = []
        # Keep the durable store complete; present bounded evidence summaries
        # rather than indefinitely growing raw transcripts to the model.
        memory = copy.deepcopy(memory)
        for entry in memory:
            entry['past_experiences'] = [{k: e.get(k) for k in
                ('experience_id', 'task_id', 'context', 'diagnosis', 'notes', 'evidence_refs', 'evaluation', 'training_failure')}
                for e in entry.get('past_experiences', [])[-6:]]
        library = list({m['method_id']: m for m in
                        [self.store.get(self.state['incumbent_method_id'])] + [m['method'] for m in memory]}.values())
        references = {'metrics': report['metrics'], 'diagnostics': report['diagnostics'],
                      'acquisition_features': cell_features(report, scope=self.settings['scope'], step_mm=self.settings['scope']['step_mm'])}
        for entry in memory:
            references['method:' + entry['method']['method_id']] = entry
        context = {**self.memory_context(), 'references': references, 'valid_evidence_refs': sorted(references),
                   'available_features': ['failure_rate', 'uncertainty', 'novelty', 'action_conflict', 'coverage_gap'],
                   'method_library': library, 'experience_memory': memory,
                   'baseline_method_id': self.state['incumbent_method_id'],
                   'reusable_method_ids': [x['method_id'] for x in library],
                   'scope': self.settings['scope'], 'comparison_protocol': self.settings['comparison'],
                   'available_observations': 'two-finger tactile descriptors; processed DD pooling where available',
                   'unavailable_capabilities': ['new sensing modality', 'new motion primitive', 'new force or safety limits'],
                   'label_caveat': 'action conflict uses measured-TCP geometric supervision; not proven optimal actions'}
        number = len(self.state['cycles']) + 1
        # A failed attempt does not overwrite an earlier prompt buffer.
        stamp = f'{number:03d}_{time.time_ns()}'
        atomic_json(self.root / 'prompt_buffer' / (stamp + '_context.json'), context)
        if self.settings['memory']['mode'] == 'fixed':
            decision = {'action': 'reuse_method', 'method_id': self.state['fixed_method_id'],
                        'method': None, 'reason': '固定方法记忆消融；不更新采集方法'}
        else:
            try:
                provider_settings = {**self.settings['provider'], 'max_proposal_attempts': self.settings.get('planning', {}).get('max_attempts', 3)}
                decision = (self.proposer or propose_method)(provider_settings, context)
            except Exception as exc:
                if getattr(exc, 'audit', None) is not None:
                    atomic_json(self.root / 'prompt_buffer' / (stamp + '_audit.json'), exc.audit)
                raise
        atomic_json(self.root / 'prompt_buffer' / (stamp + '_decision.json'), decision)
        if decision['action'] == 'pause':
            self.state.update(status='paused', last_error=decision['reason'])
            self.save()
            return False
        new_id = (self.store.register(decision['method']) if decision['action'] == 'propose_method' else decision['method_id'])
        if new_id not in [m['method_id'] for m in library] and decision['action'] != 'propose_method':
            raise ValueError('Agent 引用未提供的方法')
        from .agentic import source_manifest
        count = self.settings['comparison']['collection_trials_per_branch']
        protocol = {'task_id': self.state['task_id'],
                    'problem_signature': self.memory_context()['problem_signature'],
                    'initial_dataset_hash': fingerprint(source_manifest(self.state['base_sessions'])),
                    'initial_policy_hash': fingerprint(self.model(self.state['current_policy_id'])),
                    'training_budget': copy.deepcopy(self.settings.get('contact_policy', {})),
                    'offline_validation_groups_hash': fingerprint(self.state.get('frozen_validation_groups', [])),
                    'training_initialization': 'fresh_fixed_seed_same_algorithm; initial_policy_hash is shared reference, not a warm-start',
                    'collection_budget': self.scheduled_count(count),
                    'evaluation_cells': self.cells('comparison'), 'simulated': bool(self.settings.get('simulate')),
                    'min_trials': self.settings['comparison']['min_trials'],
                    'min_effect': self.settings['comparison']['min_effect'],
                    'checkpoints': self.checkpoints(),
                    'preregistered': True, 'evaluation_independent': True}
        cycle = {'id': number, 'created_unix_ns': time.time_ns(), 'phase': 'collect', 'base_sessions': self.state['base_sessions'][:],
                 'base_policy_id': self.state['current_policy_id'], 'protocol': protocol,
                 'protocol_sha256': fingerprint(protocol), 'decision': decision,
                 'branches': {key: {'method_id': mid, 'collect': [], 'evaluation': [], 'scan_count': 0, 'model_id': None}
                              for key, mid in [('old', self.state['incumbent_method_id']), ('new', new_id)]},
                 'schedule': [], 'evaluation_schedule': [], 'checkpoint_index': 0, 'checkpoints': [],
                 'same_method_replication': new_id == self.state['incumbent_method_id']}
        self.state['cycles'].append(cycle)
        self.state['phase'] = 'comparison'
        self.save()
        atomic_json(self.root / f'cycles/{number:03d}/protocol.json', protocol)
        return True

    def collection_round(self, cycle):
        from .rsi_methods import select_cells
        self.check_cycle_origin(cycle)
        # Finish an already planned side before adding another round. This
        # makes interruption between the old/new schedule commits idempotent.
        total = self.checkpoint_target(cycle)
        next_target = min(total,
                          min(b['scan_count'] for b in cycle['branches'].values()) +
                          self.settings['comparison']['batch_scan_trials'])
        for key, branch in cycle['branches'].items():
            if branch['scan_count'] >= total or branch['scan_count'] >= next_target:
                continue
            n = min(self.settings['comparison']['batch_scan_trials'], total - branch['scan_count'])
            roots = cycle['base_sessions'] + self.roots(branch['collect'])
            report = self.analysis(roots)
            try:
                cells = select_cells(self.store.get(branch['method_id']), report,
                                     scope=self.settings['scope'], trial_budget=n,
                                     seed=self.settings['seed'] + cycle['id'] * 100 + branch['scan_count'])
                if sum(c['reps'] for c in cells) != n:
                    raise ValueError('采集方法无法分配完整等预算批次；不静默裁切或补随机点')
                plan = plan_for_cells(cells, self.cfg, cond='A', seed=self.settings['seed'] + branch['scan_count'],
                                      monitor_start=branch['scan_count'])
                b = self.batch(plan, phase='collection', branch=key, cycle=cycle['id'])
            except ValueError as exc:
                cycle['method_failure'] = {'branch': key, 'reason': str(exc)}
                cycle['experience_id'] = self.store.record_experience({
                    'method_id': branch['method_id'], 'task_id': self.state['task_id'],
                    'created_unix_ns': cycle.get('created_unix_ns', self.state['created_unix_ns']),
                    'simulated': bool(self.settings.get('simulate')), 'context': self.memory_context(),
                    'diagnosis': '方法无法在既定约束与预算内形成可执行计划',
                    'method_failure': cycle['method_failure'], 'notes': 'No method superiority claim; keep previous policy',
                    'evidence_refs': [str(self.root / f'cycles/{cycle["id"]:03d}/protocol.json')]})
                cycle['phase'] = 'method_failed'
                self.state['phase'] = 'propose'
                self.say('方法计划未通过，已保存原因并交回 Agent；原策略和限幅保持。')
                self.save()
                return
            branch['collect'].append(b['id'])
            branch['scan_count'] += n
            cycle['schedule'].append(b['id'])
        # Alternate branch order to reduce confounding from time/drift.
        recent = cycle['schedule'][-2:]
        if len(recent) == 2 and (len(cycle['schedule'])//2 + self.settings['seed']) % 2:
            cycle['schedule'][-2:] = recent[::-1]
        self.save()

    def prepare_evaluation(self, cycle):
        self.check_cycle_origin(cycle)
        for branch in cycle['branches'].values():
            if branch['model_id'] is None:
                self.check_cycle_origin(cycle)
                from .agentic import source_manifest
                branch['training_manifest'] = source_manifest(cycle['base_sessions'] + self.roots(branch['collect']))
                try:
                    branch['model_id'] = self.train(cycle['base_sessions'] + self.roots(branch['collect']))
                except PolicyTrainingUnavailable as exc:
                    branch['unusable_model_id'] = exc.model_id
                    cycle['training_failure'] = {'method_id': branch['method_id'], 'reasons': exc.reasons,
                                                  'model_id': exc.model_id}
                    cycle['experience_id'] = self.store.record_experience({
                        'method_id': branch['method_id'], 'task_id': self.state['task_id'],
                        'created_unix_ns': cycle.get('created_unix_ns', self.state['created_unix_ns']),
                        'simulated': bool(self.settings.get('simulate')), 'context': self.memory_context(),
                        'diagnosis': '本次采集后策略未通过预注册离线门禁；未执行该策略的实机评价',
                        'training_failure': cycle['training_failure'],
                        'evidence_refs': [str(self.root / self.state['models'][exc.model_id])],
                        'notes': '方法效果未验证；不伪造实机成功率为零；保留原策略和共同起点'})
                    cycle['phase'] = 'training_failed'
                    self.state['phase'] = 'propose'
                    self.say('本轮训练未通过，保存方法反例并保留原策略，交回 Agent 更新方法。')
                    self.save()
                    return
                self.save()
        v = self.settings['comparison']
        count = 0
        for offset in v['offsets_mm']:
            for repeat in range(v['reps']//2):
                order = ['old', 'new']
                random.Random(self.settings['seed'] + cycle['id']*1000 + count).shuffle(order)
                for key in order:
                    branch = cycle['branches'][key]
                    slot = f"{cycle.get('checkpoint_index', 0)}:{key}:{count}"
                    existing = next((b for b in self.state['batches'] if b.get('evaluation_slot') == slot
                                     and b.get('cycle') == cycle['id']), None)
                    if existing:
                        if existing['id'] not in branch['evaluation']: branch['evaluation'].append(existing['id'])
                        if existing['id'] not in cycle['evaluation_schedule']: cycle['evaluation_schedule'].append(existing['id'])
                        continue
                    plan = plan_for_cells([{'offset_mm': offset, 'reps': 2}], self.cfg, cond='B',
                                          seed=self.settings['seed'] + count, monitor_start=count)
                    b = self.batch(plan, phase='evaluation', branch=key, cycle=cycle['id'], model_id=branch['model_id'])
                    b['evaluation_slot'] = slot
                    branch['evaluation'].append(b['id'])
                    cycle['evaluation_schedule'].append(b['id'])
                count += 2
        cycle['phase'] = 'evaluate'
        self.save()

    def finish_checkpoint(self, cycle):
        index = cycle.get('checkpoint_index', 0)
        existing = next((p for p in cycle.get('checkpoints', []) if p['index'] == index), None)
        if existing is None:
            point = {'index': index, 'scan_budget': self.checkpoint_target(cycle),
                     'spent_trials': self.scheduled_count(self.checkpoint_target(cycle)), 'branches': {}}
            for key, branch in cycle['branches'].items():
                result = summarize(self.roots(branch['evaluation']), cycle['protocol']['evaluation_cells'])
                if not result['records_complete']:
                    raise ValueError('开发评价存在作废、记录缺失或中断重试；禁止静默替换评价样本')
                point['branches'][key] = {**result, 'model_id': branch['model_id'],
                                           'batches': branch['evaluation'][:]}
            cycle.setdefault('checkpoints', []).append(point)
            self.save()
        if index + 1 < len(self.checkpoints()):
            for branch in cycle['branches'].values():
                branch['evaluation'] = []
                branch['model_id'] = None
            cycle['checkpoint_index'] = index + 1
            cycle['evaluation_schedule'] = []
            cycle['phase'] = 'collect'
            self.save()
            return False
        return True

    def write_curve(self, cycle):
        from .rsi_methods import learning_curve
        curves = {}
        for key in ('old', 'new'):
            points = [{'spent_trials': p['spent_trials'],
                       'success_rate': p['branches'][key]['successes']/p['branches'][key]['trials']}
                      for p in cycle.get('checkpoints', []) if p['branches'][key]['trials']]
            curves[key] = learning_curve(points) if len(points) >= 2 else {'points': points, 'normalized_auc': None}
            curves[key]['first_observed_budget_at_80_percent'] = next((p['spent_trials'] for p in points if p['success_rate'] >= .8), None)
        curves['caveat'] = 'Observed checkpoint rates, not a confidence guarantee or minimum possible sample requirement. Only preregistered last checkpoint decides promotion.'
        folder = self.root / f'cycles/{cycle["id"]:03d}'
        atomic_json(folder / 'learning_curve.json', curves)
        try:
            import matplotlib
            matplotlib.use('Agg')
            from matplotlib import pyplot as plt
            fig, ax = plt.subplots()
            for key in ('old', 'new'):
                points = curves[key]['points']
                ax.plot([p['spent_trials'] for p in points], [p['success_rate'] for p in points], 'o-', label=key)
            ax.set(xlabel='Acquisition attempts (including monitors)', ylabel='Observed recovery success', ylim=(0, 1.05))
            ax.legend()
            fig.tight_layout()
            fig.savefig(folder / 'learning_curve.png', dpi=160)
            plt.close(fig)
        except ImportError:
            curves['plot_unavailable'] = 'matplotlib not installed; exact curve points remain in JSON'
            atomic_json(folder / 'learning_curve.json', curves)
        cycle['learning_curve'] = curves

    def finish_cycle(self, cycle):
        from .rsi_methods import evaluate_comparison
        from .agentic import source_manifest
        self.check_cycle_origin(cycle)
        if fingerprint(cycle['protocol']) != cycle['protocol_sha256']:
            raise ValueError('比较标准在实验后被修改')
        summaries = {}
        for key, branch in cycle['branches'].items():
            if source_manifest(cycle['base_sessions'] + self.roots(branch['collect'])) != branch['training_manifest']:
                raise ValueError('分支训练数据在训练后被修改')
            summary = summarize(self.roots(branch['evaluation']), cycle['protocol']['evaluation_cells'])
            summaries[key] = {**cycle['protocol'], **summary, 'method_id': branch['method_id'],
                              'spent_trials': self.spent(branch['collect']), 'model_id': branch['model_id']}
        comparison = {'old': summaries['old'], 'new': summaries['new'], 'protocol': cycle['protocol']}
        result = evaluate_comparison(**comparison)
        if cycle.get('same_method_replication'):
            if result['issues'] == ['cannot compare a method with itself']:
                result.update(outcome='same_method_replication', winner=None,
                              interpretation='固定方法的等预算重复；不能用于证明方法变好或自动晋升')
        cycle['comparison'], cycle['result'] = comparison, result
        evaluated = [b for point in cycle.get('checkpoints', []) for branch in point['branches'].values() for b in branch['batches']]
        cycle['record_manifest'] = source_manifest(self.roots(cycle['schedule'] + evaluated + cycle['evaluation_schedule']))
        self.write_curve(cycle)
        record = {'method_id': cycle['branches']['new']['method_id'], 'task_id': self.state['task_id'],
                  'created_unix_ns': cycle.get('created_unix_ns', self.state.get('created_unix_ns', 1)),
                  'simulated': bool(self.settings.get('simulate')), 'context': self.memory_context(),
                  'diagnosis': cycle['decision']['reason'], 'notes': 'Geometric-supervision ContactPolicy; real and synthetic evidence separated',
                  'evidence_refs': [str(self.root / f'cycles/{cycle["id"]:03d}/comparison.json')]}
        if not cycle.get('same_method_replication'):
            record['comparison'] = comparison
        else:
            record['replication'] = comparison
        atomic_json(self.root / f'cycles/{cycle["id"]:03d}/comparison.json', {'comparison': comparison, 'result': result})
        cycle['experience_id'] = self.store.record_experience(record)
        if result.get('outcome') == 'invalid':
            self.save()
            raise ValueError('方法对照记录无效，已保留经验但未更新方法/策略：' + '; '.join(result.get('issues', [])))
        # A new method needs the independent evaluator's win; ties keep old.
        winner = 'new' if result.get('winner') == cycle['branches']['new']['method_id'] else 'old'
        selected = cycle['branches'][winner]
        self.state['incumbent_method_id'] = selected['method_id']
        self.state['current_policy_id'] = selected['model_id']
        self.state['base_sessions'] = cycle['base_sessions'] + self.roots(selected['collect'])
        cycle['phase'] = 'complete'
        self.state['phase'] = 'propose'
        self.say('方法比较：' + json.dumps(result, ensure_ascii=False))
        self.save()

    def prepare_final(self):
        from .agentic import make_plan
        v = self.settings['final_validation']
        plan = make_plan({'cond': 'mixed', 'cells': [{'offset_mm': d, 'reps': v['reps']} for d in v['offsets_mm']]},
                         self.cfg, seed=self.settings['seed'], validation=True)
        b = self.batch(plan, phase='final_validation', model_id=self.state['current_policy_id'])
        self.state['final_batch'] = b['id']
        self.state['frozen'] = {'model_id': self.state['current_policy_id'],
                                'method_id': self.state['incumbent_method_id'], 'cells': self.cells('final_validation'),
                                'policy_sha256': fingerprint(self.model(self.state['current_policy_id'])),
                                'statement': 'Final A/B data never enter training, method choice or experience promotion'}
        self.state['phase'] = 'final_validation'
        self.save()

    def report(self):
        s = self.state
        lines = ['# RSI 实验报告', '', f"状态：{s['status']}；阶段：{s['phase']}",
                 f"数据：{'模拟，不能证明实机能力' if self.settings.get('simulate') else '实机'}",
                 f"任务：{s['task_id']}；记忆方式：{self.settings['memory']['mode']}",
                 f"新增尝试（包含中断、监测）：{self.spent()}/{self.settings['budget']['max_trials']}",
                 f"当前方法：{s['incumbent_method_id']}；当前策略：{s['current_policy_id']}",
                 '', '训练目标：实测 TCP 相对固定 T0 的几何纠正，不等于已证明的最优恢复动作。',
                 '比较数据属于方法开发验证；最终 A/B 独立验证不反馈给方法库。',
                 '名义点距不等于实际空间分辨率；瓶位移动会使几何监督失真。', '',
                 '| 轮次 | 旧方法 | 新方法 | 评估 |', '|---|---|---|---|']
        for c in s['cycles']:
            lines.append(f"| {c['id']} | {c['branches']['old']['method_id']} | {c['branches']['new']['method_id']} | {json.dumps(c.get('result', {'status': c['phase']}), ensure_ascii=False)} |")
        if s.get('final_result'):
            lines += ['', '最终独立验证：', '```json', json.dumps(s['final_result'], ensure_ascii=False, indent=2), '```']
        if s.get('last_error'):
            lines += ['', '停止原因：' + s['last_error']]
        lines += ['', '文件：rsi.json（断点），prompt_buffer/（实际提示与提案），cycles/（比较证据），',
                  'models/（ContactPolicy），batches/*/trials/*/streams.h5（原始服务数据），方法库：' + self.settings['memory']['path']]
        (self.root / 'REPORT.md').write_text('\n'.join(lines) + '\n')

    def run(self):
        if self.state['status'] in TERMINAL:
            self.say('RSI 会话已结束，不连接硬件。报告：' + str(self.root / 'REPORT.md'))
            return 0
        if self.state.get('active_run_unix_ns'):
            self.state['elapsed_s'] += max(0., (time.time_ns() - self.state['active_run_unix_ns']) / 1e9)
        self.started = time.monotonic()
        self.state['active_run_unix_ns'] = time.time_ns()
        self.state['status'] = 'running'
        self.save()
        self.say('RSI：固定范围内比较采集方法；标签、换瓶和复位仍由现场操作员完成。')
        self.say('方法记忆和 ContactPolicy 分开保存；样本不足不会自动晋升方法。')
        try:
            while self.state['status'] == 'running':
                if self.elapsed() >= self.settings['budget']['max_seconds']:
                    self.state.update(status='budget_exhausted', last_error='时间预算耗尽；不代表没有值得补充的数据')
                    break
                phase = self.state['phase']
                if phase == 'bootstrap':
                    if not self.state.get('bootstrap_batch'):
                        v = self.settings['bootstrap']
                        p = plan_for_cells([{'offset_mm': d, 'reps': v['reps']} for d in v['offsets_mm']], self.cfg,
                                           cond='A', seed=self.settings['seed'])
                        if len(p) + self.final_cost() + self.cycle_cost() > self.settings['budget']['max_trials']:
                            raise ValueError('预算不足以完成共同起点采集、一轮方法比较及最终验证')
                        b = self.batch(p, phase='bootstrap')
                        self.state['bootstrap_batch'] = b['id']
                        self.save()
                    b = next(b for b in self.state['batches'] if b['id'] == self.state['bootstrap_batch'])
                    if not self.execute(b): break
                    self.state['base_sessions'] = self.roots([b['id']])
                    self.state['phase'] = 'initial_train'
                elif phase == 'initial_train':
                    identifier = self.train(self.state['base_sessions'])
                    model = self.model(identifier)
                    self.state['frozen_validation_groups'] = sorted({model['source_trials'][i]['group'] for i in model['split']['validation_indices']})
                    self.state.update(initial_policy_id=identifier, current_policy_id=identifier, phase='propose')
                elif phase == 'propose':
                    if len(self.state['cycles']) >= self.settings['budget']['max_cycles'] or self.spent() + self.cycle_cost() + self.final_cost() > self.settings['budget']['max_trials']:
                        if self.spent() + self.final_cost() > self.settings['budget']['max_trials']:
                            self.state.update(status='budget_exhausted', last_error='预算不足以完成最终验证，未宣称收敛')
                            break
                        self.prepare_final()
                    elif not self.propose(): break
                elif phase == 'freeze':
                    if self.spent() + self.final_cost() > self.settings['budget']['max_trials']:
                        self.state.update(status='budget_exhausted', last_error='预算不足以完成最终验证')
                        break
                    self.prepare_final()
                elif phase == 'comparison':
                    cycle = self.state['cycles'][-1]
                    if cycle['phase'] == 'collect':
                        pending = [b for i in cycle['schedule'] for b in self.state['batches'] if b['id'] == i and b['status'] != 'complete']
                        if pending:
                            if not self.execute(pending[0]): break
                        elif all(b['scan_count'] >= self.checkpoint_target(cycle) for b in cycle['branches'].values()):
                            self.prepare_evaluation(cycle)
                        else:
                            self.collection_round(cycle)
                    elif cycle['phase'] == 'evaluate':
                        pending = [b for i in cycle['evaluation_schedule'] for b in self.state['batches'] if b['id'] == i and b['status'] != 'complete']
                        if pending:
                            if not self.execute(pending[0]): break
                        else:
                            if self.finish_checkpoint(cycle):
                                self.finish_cycle(cycle)
                elif phase == 'final_validation':
                    b = next(b for b in self.state['batches'] if b['id'] == self.state['final_batch'])
                    if fingerprint(self.model(b['model_id'])) != self.state['frozen']['policy_sha256']:
                        raise ValueError('最终验证模型已被修改')
                    if not self.execute(b): break
                    rows = trial_rows(self.roots([b['id']]))
                    from .agentic_learning import _rate
                    self.state['final_result'] = {c: _rate([r for r in rows if r['cond'] == c and r.get('kind') == 'scan'
                                                           and r.get('complete') and r.get('recording_complete') and r['verified_recording']
                                                           and r.get('label') != 'invalid']) for c in ('A', 'B')}
                    expected = sum(c['trials'] for c in self.state['frozen']['cells'])
                    retried = bool(list((self.root / b['path'] / 'interrupted').glob('*/meta.json')))
                    self.state['final_result']['records_complete'] = not retried and all(self.state['final_result'][c]['n'] == expected for c in ('A', 'B'))
                    self.state['final_result']['outcome'] = 'descriptive_completed' if self.state['final_result']['records_complete'] else 'invalid_or_incomplete'
                    self.state['final_result']['scope'] = 'Measured discrete offsets only; no claim of continuous coverage or recursive superiority'
                    self.state.update(status='complete', phase='complete')
                self.save()
                self.report()
            self.save()
            return 0
        except (Exception, KeyboardInterrupt) as exc:
            self.state.update(status='paused', last_error=str(exc) or type(exc).__name__)
            atomic_json(self.root / 'errors' / f'{time.time_ns()}.json', {'error': self.state['last_error'], 'type': type(exc).__name__})
            self.say('RSI 已暂停：' + self.state['last_error'])
            return 130 if isinstance(exc, KeyboardInterrupt) else 2
        finally:
            self.state['elapsed_s'] = self.elapsed()
            self.state.pop('active_run_unix_ns', None)
            self.started = None
            self.save()
            self.report()
            self.say('RSI 记录与报告：' + str(self.root / 'REPORT.md'))

    def acknowledge_recovery(self):
        # Reuse the existing manual recovery acknowledgement without motion.
        from .agentic import Campaign
        if not self.state.get('active_batch'):
            raise ValueError('没有待恢复的活动批次')
        adapter = object.__new__(Campaign)
        adapter.root, adapter.state, adapter.ask, adapter.say = self.root, self.state, self.ask, self.say
        adapter.save = self.save
        adapter.acknowledge_recovery()


def parser():
    p = argparse.ArgumentParser(description='方法经验 → 等预算采集 → ContactPolicy → 对照评估 → 方法版本更新')
    sub = p.add_subparsers(dest='command', required=True)
    for name in ('run', 'resume'):
        q = sub.add_parser(name)
        q.add_argument('--out', type=Path, required=True)
        mode = q.add_mutually_exclusive_group(required=True)
        mode.add_argument('--real', action='store_true')
        mode.add_argument('--simulate', action='store_true')
        q.add_argument('--config', type=Path, default=ROOT / 'config.rsi.yaml')
        q.add_argument('--seed-session', type=Path, action='append')
        q.add_argument('--provider', choices=['chatgpt', 'api', 'deterministic', 'mock'])
        q.add_argument('--memory-mode', choices=['recursive', 'fixed', 'none'])
        q.add_argument('--task-id')
        q.add_argument('--problem', help='明确的新问题/失败场景描述；仅换输出目录不会证明跨问题迁移')
    for name in ('status', 'report', 'recover'):
        sub.add_parser(name).add_argument('--out', type=Path, required=True)
    memory = sub.add_parser('memory')
    memory.add_argument('--config', type=Path, default=ROOT / 'config.rsi.yaml')
    train = sub.add_parser('train')
    train.add_argument('--session', action='append', type=Path, required=True)
    train.add_argument('--out', type=Path, required=True)
    train.add_argument('--config', type=Path, default=ROOT / 'config.rsi.yaml')
    train.add_argument('--simulate', action='store_true')
    return p


def main(argv=None, *, input_fn=input, output=print, executor=None, proposer=None):
    args = parser().parse_args(argv)
    lock = None
    try:
        if args.command == 'memory':
            from .rsi_methods import MethodStore
            output(json.dumps(MethodStore(load_settings(args.config)['memory']['path']).list_methods(), ensure_ascii=False, indent=2))
            return 0
        if args.command == 'train':
            from .contact_policy import train_contact_policy
            settings = load_settings(args.config)
            settings['simulate'] = args.simulate
            model = train_contact_policy(args.session, settings, out=args.out)
            output(json.dumps({k: model.get(k) for k in ('model_id', 'usable_for_B', 'validation', 'reason')}, ensure_ascii=False, indent=2))
            return 0 if model.get('usable_for_B') else 2
        root = args.out.expanduser().resolve()
        if args.command == 'status':
            data = json.loads((root / 'rsi.json').read_text())
            output(json.dumps({k: data.get(k) for k in ('status', 'phase', 'task_id', 'active_batch', 'incumbent_method_id', 'current_policy_id', 'last_error')}, ensure_ascii=False, indent=2))
            return 0
        root.mkdir(parents=True, exist_ok=True)
        lock = (root / '.rsi.lock').open('a')
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError('此 RSI 会话已有运行进程') from None
        s = None
        if args.command in ('run', 'resume'):
            exists = (root / 'rsi.json').exists()
            if args.command == 'resume' and not exists:
                raise ValueError('会话不存在，不能续跑')
            if exists:
                old = json.loads((root / 'rsi.json').read_text())['settings']
                if bool(args.simulate) != bool(old.get('simulate')):
                    raise ValueError('续跑不能转换模拟/实机模式')
                if any(getattr(args, k) is not None for k in ('seed_session', 'provider', 'memory_mode', 'task_id', 'problem')):
                    raise ValueError('续跑只需 --out 和 --real/--simulate，不能覆盖设置')
            else:
                s = load_settings(args.config)
                s['simulate'] = bool(args.simulate)
                if args.seed_session: s['seed_sessions'] = [str(v.expanduser().resolve()) for v in args.seed_session]
                if args.provider: s['provider']['backend'] = args.provider
                if args.memory_mode: s['memory']['mode'] = args.memory_mode
                if args.task_id: s['task_id'] = args.task_id
                if args.problem: s['problem'] = args.problem
        campaign = RSICampaign(root, settings=s, ask=input_fn, say=output, executor=executor, proposer=proposer)
        if args.command == 'report':
            campaign.report()
            output(str(root / 'REPORT.md'))
            return 0
        if args.command == 'recover':
            campaign.acknowledge_recovery()
            return 0
        return campaign.run()
    except (Exception, KeyboardInterrupt) as exc:
        output('RSI 停止：' + (str(exc) or type(exc).__name__))
        return 130 if isinstance(exc, KeyboardInterrupt) else 2
    finally:
        if lock: lock.close()
