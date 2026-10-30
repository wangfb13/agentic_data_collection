"""Durable, bounded active experiment campaigns around the independent E1/E2 runner.

The LLM proposes bounded experimental batches, or selects legacy candidates.
It cannot supply actuator commands or relax the fixed execution limits.
"""
from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
import traceback

import numpy as np
import yaml

from .session import atomic_json, fingerprint, plain

ROOT = Path(__file__).resolve().parents[1]
TERMINAL = {'complete', 'budget_exhausted', 'no_usable_model'}


def load_settings(path):
    path = Path(path).expanduser().resolve()
    result = yaml.safe_load(path.read_text())
    if not isinstance(result, dict) or result.get('schema') != 'agentic_campaign_config_v1':
        raise ValueError('不是有效的 agentic 配置')
    result = copy.deepcopy(result)
    base = Path(result.get('base_config', 'config.e1e2.yaml')).expanduser()
    result['base_config'] = str(base if base.is_absolute() else path.parent / base)
    result['seed_sessions'] = [str(Path(p).expanduser().resolve()) for p in result.get('seed_sessions', [])]
    validate_settings(result)
    return result


def finite(value, name, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or (positive and value <= 0):
        raise ValueError(name + ' 必须是有限' + ('正数' if positive else '数'))
    return float(value)


def validate_settings(s):
    planning = s.get('planning', {})
    if planning.get('mode', 'candidate') not in ('agent', 'candidate'):
        raise ValueError('planning.mode 必须为 agent 或 candidate')
    attempts = planning.get('max_attempts', 3)
    if isinstance(attempts, bool) or not isinstance(attempts, int) or not 1 <= attempts <= 5:
        raise ValueError('planning.max_attempts 必须为1至5的整数')
    if planning.get('mode') == 'agent' and not s['scope'].get('adaptive', False):
        raise ValueError('Agent 自拟计划需要 scope.adaptive；旧 campaign 保留 candidate 模式')
    scope = s['scope']
    lo, hi = finite(scope['min_mm'], 'min_mm'), finite(scope['max_mm'], 'max_mm')
    if not -25 <= lo < hi <= 25 or not lo <= 0 <= hi:
        raise ValueError('探索范围须包含零点并位于 [-25,25] mm')
    finite(scope['step_mm'], 'step_mm', True)
    if scope.get('adaptive', False):
        for key in ('adaptive', 'allow_expand'):
            if key in scope and not isinstance(scope[key], bool): raise ValueError('scope.' + key + ' 必须是布尔值')
        a = finite(scope.get('initial_min_mm', lo), 'initial_min_mm')
        b = finite(scope.get('initial_max_mm', hi), 'initial_max_mm')
        if not lo <= a < b <= hi or not a <= 0 <= b:
            raise ValueError('初始范围必须包含0且位于固定硬边界内')
        minimum = finite(scope.get('min_step_mm', .5), 'min_step_mm', True)
        if minimum > scope['step_mm']: raise ValueError('min_step_mm 不能大于初始步长')
        if finite(scope.get('expansion_step_mm', 5.), 'expansion_step_mm', True) > 5.:
            raise ValueError('单轮扩展不能超过5 mm')
    for name in ('max_trials', 'max_batches', 'batch_trials'):
        v = s['budget'][name]
        if isinstance(v, bool) or not isinstance(v, int) or v < 1:
            raise ValueError('budget.' + name + ' 必须为正整数')
    finite(s['budget']['max_seconds'], 'max_seconds', True)
    v = s['validation']
    if not isinstance(v['reps'], int) or isinstance(v['reps'], bool) or v['reps'] < 2 or v['reps'] % 2:
        raise ValueError('validation.reps 必须为正偶数，轻重瓶各一半')
    if v.get('adaptive_scope', False) and not scope.get('adaptive', False):
        raise ValueError('动态验证范围需要 scope.adaptive')
    if v.get('adaptive_scope', False):
        points = v.get('points', 5)
        if isinstance(points, bool) or not isinstance(points, int) or not 3 <= points <= 25:
            raise ValueError('validation.points 必须为3至25的整数')
    if not v.get('offsets_mm') or len(set(v['offsets_mm'])) != len(v['offsets_mm']):
        raise ValueError('验证偏移列表为空或重复')
    for d in v['offsets_mm']:
        if not lo <= finite(d, 'validation offset') <= hi:
            raise ValueError('验证偏移超出授权范围')
    if not 0 < finite(v['success_target'], 'success_target') < 1:
        raise ValueError('success_target 必须位于 (0,1)')
    for k in ('target_mae_mm', 'target_p90_mm', 'min_gain_mm'):
        finite(s['learning'][k], 'learning.' + k, True)
    for k in ('min_levels', 'min_per_level', 'plateau_batches', 'min_B_batches', 'min_B_per_cell'):
        v = s['learning'].get(k, 1)
        if isinstance(v, bool) or not isinstance(v, int) or v < 1:
            raise ValueError('learning.' + k + ' 必须为正整数')
    gains = s.get('policy', {}).get('gains', [1.0])
    if not gains or any(not .5 <= finite(g, 'gain') <= 1 for g in gains):
        raise ValueError('允许的纠正增益必须位于 [0.5,1]，不增加物理修正上限')
    if s.get('approval', 'campaign') not in ('campaign', 'batch'):
        raise ValueError('approval 必须为 campaign 或 batch')
    if s.get('provider', {}).get('backend') not in ('chatgpt', 'api', 'deterministic', 'mock'):
        raise ValueError('未知 agent provider')


def source_manifest(paths):
    """Keep raw data in place, but reject silent label/model edits on resume."""
    result = {}
    for root in paths:
        p = Path(root).resolve()
        meta = p / 'session_meta.json'
        if not meta.is_file():
            raise ValueError('找不到种子 session：' + str(p))
        digest = hashlib.sha256()
        files = [meta] + sorted(p.glob('trials/*/meta.json')) + sorted(p.glob('trials/*/recording_status.json'))
        for f in files:
            digest.update(str(f.relative_to(p)).encode()); digest.update(f.read_bytes())
        # Legacy sessions supply measured PROBE coordinates from this dataset.
        # Pin the consumed numeric evidence without hashing multi-GB images/DD.
        for f in sorted(p.glob('trials/*/streams.h5')):
            try:
                import h5py
                with h5py.File(f, 'r') as h:
                    if 'streams/tcp_measured' not in h: continue
                    digest.update(str(f.relative_to(p)).encode())
                    group = h['streams/tcp_measured']
                    for name in ('monotonic_ns', 'data'):
                        array = np.asarray(group[name])
                        digest.update(name.encode()); digest.update(str(array.shape).encode())
                        digest.update(str(array.dtype).encode()); digest.update(array.tobytes())
            except ImportError:
                raise ValueError('读取种子实测 TCP 需要 h5py；请使用项目 Python 环境') from None
        result[str(p)] = digest.hexdigest()
    return result


def make_plan(candidate, cfg, *, seed, validation=False):
    rng = random.Random(seed)
    bottles = cfg['design']['bottles']
    if len(bottles) != 2:
        raise ValueError('此实验的平衡计划要求两种瓶子')
    result = []
    for bottle in bottles:
        units = []
        for cell in candidate['cells']:
            for repeat in range(cell['reps'] // 2):
                pair = ['A', 'B'] if validation else [candidate['cond']]
                rng.shuffle(pair)
                units.append([{'cond': cond, 'bottle': bottle, 'delta_mm': float(cell['offset_mm']),
                               'repeat': repeat, 'kind': 'scan'} for cond in pair])
        rng.shuffle(units)
        for group in units:
            result.extend(group)
    # Monitors belong to a condition but are not validation-success denominators.
    every = cfg['design']['zero_every']
    with_monitors = []
    for i, item in enumerate(result, 1):
        with_monitors.append(item)
        if every and i % every == 0:
            with_monitors.append({**item, 'delta_mm': 0., 'kind': 'zero_monitor', 'repeat': None})
    for i, item in enumerate(with_monitors):
        item.update(id=f'{i+1:05d}', index=i+1)
    return with_monitors


def validation_offsets(settings, design=None):
    if not settings['validation'].get('adaptive_scope', False):
        return list(settings['validation']['offsets_mm'])
    scope = settings['scope']
    design = design or {'min_mm': scope.get('initial_min_mm', scope['min_mm']),
                        'max_mm': scope.get('initial_max_mm', scope['max_mm'])}
    lo, hi = float(design['min_mm']), float(design['max_mm'])
    n = settings['validation'].get('points', 5)
    points = sorted(set([round(float(x), 8) for x in np.linspace(lo, hi, n - 1)] + [0.]))
    while len(points) < n:
        a, b = max(zip(points, points[1:]), key=lambda pair: pair[1] - pair[0])
        midpoint = round((a + b) / 2, 8)
        if midpoint in points: raise ValueError('范围过窄，无法生成互异验证点')
        points = sorted(points + [midpoint])
    return points


def validation_candidate(settings, design=None, offsets=None):
    return {'id': 'frozen-validation', 'cond': 'mixed', 'cells': [
        {'offset_mm': d, 'reps': settings['validation']['reps']} for d in (offsets if offsets is not None else validation_offsets(settings, design))],
        'reason': '固定独立验证；每个瓶子和偏移内 A/B 成对，随机先后顺序'}


def validate_design_transition(candidate, settings, current):
    """The supervisor selects a checked design; it cannot edit numeric limits."""
    action = candidate.get('design_action', 'sample')
    nxt = candidate.get('design_next')
    if not settings['scope'].get('adaptive', False):
        if nxt is not None or action != 'sample': raise ValueError('固定范围 campaign 不能改变设计')
        return None
    if current is None: raise ValueError('自适应设计尚未初始化')
    if action not in ('sample', 'repeat', 'refine', 'expand_left', 'expand_right'):
        raise ValueError('未知设计动作')
    if nxt is None:
        if action not in ('sample', 'repeat'): raise ValueError('设计动作缺少 design_next')
        nxt = {k: current[k] for k in ('min_mm', 'max_mm', 'step_mm')}
    if not isinstance(nxt, dict) or set(nxt) != {'min_mm', 'max_mm', 'step_mm'}:
        raise ValueError('design_next 只能包含范围和步长')
    nxt = {k: finite(v, 'design_next.' + k, k == 'step_mm') for k, v in nxt.items()}
    lo, hi, step = (nxt[k] for k in ('min_mm', 'max_mm', 'step_mm'))
    oldlo, oldhi, oldstep = (float(current[k]) for k in ('min_mm', 'max_mm', 'step_mm'))
    scope = settings['scope']; minimum = scope.get('min_step_mm', .5)
    if not scope['min_mm'] <= lo < hi <= scope['max_mm'] or not lo <= 0 <= hi or step < minimum:
        raise ValueError('新设计超出硬边界或最小步长')
    same = lambda a, b: abs(a - b) < 1e-8
    if lo > oldlo + 1e-8 or hi < oldhi - 1e-8: raise ValueError('不能收缩范围以丢弃失败点')
    if action == 'refine':
        if not same(lo, oldlo) or not same(hi, oldhi) or not same(step, max(minimum, oldstep / 2)) or step >= oldstep:
            raise ValueError('细化只能在原范围内减半步长，到最小步长为止')
    elif action.startswith('expand_'):
        if not scope.get('allow_expand', False) or not same(step, oldstep): raise ValueError('本轮不允许此扩展')
        growth = oldlo - lo if action == 'expand_left' else hi - oldhi
        other_same = same(hi, oldhi) if action == 'expand_left' else same(lo, oldlo)
        if not other_same or not 0 < growth <= min(5., scope.get('expansion_step_mm', 5.)) + 1e-8:
            raise ValueError('一次只能按上限向单侧扩展')
        if not any((cell['offset_mm'] < oldlo if action == 'expand_left' else cell['offset_mm'] > oldhi) for cell in candidate['cells']):
            raise ValueError('扩展计划没有新增范围内的测点')
    elif not all(same(nxt[k], current[k]) for k in nxt):
        raise ValueError('普通补采不能隐式改变范围或步长')
    if any(not lo <= finite(c['offset_mm'], '设计点') <= hi for c in candidate['cells']):
        raise ValueError('候选点超出本批设计范围')
    return nxt


def validate_candidate(candidate, settings, cfg, remaining, approved_models):
    if candidate.get('cond') not in ('A', 'B'):
        raise ValueError('候选实验只能是 A 或 B')
    if not candidate.get('id') or not candidate.get('cells'):
        raise ValueError('候选实验缺少 ID 或点位')
    lo, hi = settings['scope']['min_mm'], settings['scope']['max_mm']
    seen = set()
    for cell in candidate['cells']:
        d = finite(cell['offset_mm'], '候选偏移')
        r = cell['reps']
        if not lo <= d <= hi or d in seen:
            raise ValueError('候选点重复或超出授权探索范围')
        if isinstance(r, bool) or not isinstance(r, int) or r < 2 or r % 2:
            raise ValueError('每档重复数必须是正偶数')
        seen.add(d)
    if candidate['cond'] == 'B' and candidate.get('model_id') not in approved_models:
        raise ValueError('B 候选没有经过验证的模型')
    policy = candidate.get('policy') or {'gain': 1.0}
    gain = policy.get('gain', 1.)
    if (set(policy) - {'gain'} or isinstance(gain, bool) or
            (candidate['cond'] == 'B' and gain not in settings['policy']['gains']) or
            (candidate['cond'] == 'A' and gain != 1.)):
        raise ValueError('候选包含未授权的策略字段或增益')
    plan = make_plan(candidate, cfg, seed=settings['seed'])
    if len(plan) > remaining:
        raise ValueError('候选批次超过剩余预算（已包含零点监测）')
    return plan


def validate_decision(decision, candidates, stop):
    if not isinstance(decision, dict) or decision.get('action') not in ('select_batch', 'freeze', 'pause'):
        raise ValueError('agent 返回未知动作')
    if not isinstance(decision.get('reason'), str) or not decision['reason'].strip():
        raise ValueError('agent 必须说明选择理由')
    # The provider can return diagnostic metadata but never numerical execution changes.
    if set(decision) - {'action', 'candidate_id', 'reason', 'model_id', '_meta'}:
        raise ValueError('agent 返回了未授权的执行字段')
    if decision['action'] == 'freeze' and not stop['eligible']:
        raise ValueError('当前数值证据未达到停止条件，拒绝 agent 提前宣告完成')
    if decision['action'] == 'select_batch':
        chosen = next((c for c in candidates if c['id'] == decision.get('candidate_id')), None)
        if chosen is None:
            raise ValueError('agent 只能选择本地验证后的候选 ID')
        if decision.get('model_id') not in (None, chosen.get('model_id')):
            raise ValueError('agent 不能替换候选绑定的模型')
        return chosen
    return None


def resolve_base(settings, out, ask, say):
    from .e1e2 import parser, resolve_config
    argv = ['--cond', 'A', '--config', settings['base_config'], '--out', str(out / '_config_only')]
    if settings.get('simulate'):
        argv += ['--min-normalized', '.336']
    cfg, _ = resolve_config(parser().parse_args(argv), ask, say)
    return cfg


class Campaign:
    def __init__(self, out, *, settings=None, ask=input, say=print, executor=None, provider=None):
        self.root = Path(out).expanduser().resolve(); self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / 'campaign.json'; self.ask = ask; self.say = say
        self.executor = executor; self.provider = provider; self.first_batch_this_process = True
        if self.path.exists():
            self.state = json.loads(self.path.read_text())
            if self.state.get('schema') != 'agentic_campaign_v1': raise ValueError('未知 campaign 格式')
            if settings is not None and fingerprint(settings) != fingerprint(self.state['settings']):
                raise ValueError('续跑不能更改范围、模型服务、预算或评价标准；请创建新 campaign')
            if source_manifest(self.state['seeds']) != self.state['seed_manifest']:
                raise ValueError('种子数据已被更改；为保证溯源，请创建新的 campaign')
        else:
            if settings is None: raise ValueError('新 campaign 需要配置')
            cfg = resolve_base(settings, self.root, ask, say)
            reserve = len(make_plan(validation_candidate(settings), cfg, seed=settings['seed'], validation=True))
            if reserve >= settings['budget']['max_trials']:
                raise ValueError(f'总预算需大于独立验证预留的 {reserve} 次（含监测）')
            seeds = settings.get('seed_sessions', [])
            context = {}
            if seeds:
                seed_meta = json.loads((Path(seeds[0]) / 'session_meta.json').read_text())
                if bool(seed_meta.get('simulated')) != bool(settings.get('simulate')):
                    raise ValueError('模拟和实机种子不能混用')
                if seed_meta.get('T0'):
                    context['T0'] = seed_meta['T0']; context['source_T0_session'] = seeds[0]
                    # A seed is a fixed pose reference; first use still asks the
                    # operator to restore it and performs a fresh unloaded zero.
                    if seed_meta.get('T0_accepted_unix_ns'):
                        context['T0_accepted_unix_ns'] = seed_meta['T0_accepted_unix_ns']
                    context['initial_centroids'] = seed_meta.get('initial_centroids')
            self.state = {'schema': 'agentic_campaign_v1', 'status': 'ready', 'phase': 'explore',
                'settings': plain(settings), 'base_config': plain(cfg), 'context': context,
                'seeds': seeds, 'seed_manifest': source_manifest(seeds), 'batches': [], 'models': {},
                'active_batch': None, 'created_unix_ns': time.time_ns(), 'elapsed_s': 0.,
                'validation_reserved_trials': reserve, 'history': [], 'decisions': [], 'frozen': None,
                'design': None, 'design_history': [],
                'hardware_config_manifest': self.hardware_manifest(cfg, settings)}
            self.save()
        self.settings = self.state['settings']; self.cfg = self.state['base_config']

    def save(self): atomic_json(self.path, self.state)

    def agent_planning(self):
        # Missing mode belongs to an already-created legacy campaign.
        return self.settings.get('planning', {}).get('mode', 'candidate') == 'agent'

    def next_decision_number(self):
        # A crash may leave audit files before state.decisions is committed.
        # Never reuse their numbers on resume.
        numbers = [int(d['number']) for d in self.state['decisions'] if isinstance(d.get('number'), int)]
        for path in (self.root / 'decisions').glob('*_*.json'):
            prefix = path.name.split('_', 1)[0]
            if prefix.isdigit(): numbers.append(int(prefix))
        return max(numbers, default=0) + 1

    @staticmethod
    def hardware_manifest(cfg, settings):
        if settings.get('simulate'): return {}
        path = Path(cfg['hardware_config']).expanduser()
        if not path.is_absolute(): path = Path(settings['base_config']).parent / path
        path = path.resolve()
        return {str(path): hashlib.sha256(path.read_bytes()).hexdigest()}

    def _spent(self):
        total = 0
        for b in self.state['batches']:
            p = self.root / b['path']
            total += len(list(p.glob('trials/*/meta.json'))) + len(list(p.glob('interrupted/*/meta.json')))
        return total

    def training_sessions(self):
        return self.state['seeds'] + [str(self.root / b['path']) for b in self.state['batches']
            if b['phase'] == 'explore' and (self.root / b['path'] / 'session_meta.json').exists()]

    def effective_settings(self):
        from .agentic_design import enabled, effective_settings
        if enabled(self.settings) and self.state.get('design'):
            return effective_settings(self.settings, self.state['design'])
        return self.settings

    def frozen_validation_candidate(self):
        frozen = self.state.get('frozen') or {}
        return validation_candidate(self.settings, frozen.get('design'), frozen.get('validation_offsets_mm'))

    def analyze(self):
        from .agentic_learning import analyze_sessions, write_plots
        from .agentic_design import enabled, initialize
        report = analyze_sessions(self.training_sessions(), self.effective_settings())
        if enabled(self.settings) and self.state.get('design') is None:
            self.state['design'] = initialize(self.settings, report)
            self.state.setdefault('design_history', []).append({'action': 'initialize', 'after': copy.deepcopy(self.state['design']),
                'reason': self.state['design'].get('origin'), 'created_unix_ns': time.time_ns()})
            self.save()
            report = analyze_sessions(self.training_sessions(), self.effective_settings())
        for model in report.get('models', []):
            if not model.get('usable_for_B'): continue
            mid = model['model_id']
            path = self.root / 'models' / (mid + '.json')
            if path.exists() and json.loads(path.read_text()) != plain(model):
                raise ValueError('不可覆盖已存在的模型版本 ' + mid)
            atomic_json(path, model)
            self.state['models'][mid] = str(path.relative_to(self.root))
        report['plots'] = write_plots(report, self.root / 'analysis' / 'plots')
        report['design'] = self.state.get('design')
        report['hard_scope_mm'] = [self.settings['scope']['min_mm'], self.settings['scope']['max_mm']]
        atomic_json(self.root / 'analysis' / 'latest.json', report)
        self.state['latest_metrics'] = report.get('metrics', {})
        self.save()
        return report

    def model(self, mid):
        return json.loads((self.root / self.state['models'][mid]).read_text()) if mid else None

    def stop_evidence(self, report, candidates=None):
        if self.agent_planning():
            from .agentic_evaluation import prospective_B_evidence
            evidence = prospective_B_evidence(report, self.state['batches'], self.root,
                                              self.settings, self.state.get('design'))
            eligible = bool(report.get('stopping_eligible') and evidence['eligible'])
            return {'eligible': eligible, 'mapping_eligible': bool(report.get('stopping_eligible')),
                    'B_evidence': evidence,
                    'required_B_cells': evidence['required_cells'],
                    'B_batches_on_current_model': evidence['B_batches_on_current_model'],
                    'B_exploratory_success_rate': evidence['success_rate'],
                    'unresolved_design_candidates': [],
                    'advisory_design_candidates': [c['id'] for c in (candidates or []) if c.get('unresolved')],
                    'reason': '映射精度及当前模型同一增益的必测点 B 覆盖均满足要求' if eligible else
                              '仍需满足映射精度，以及当前模型同一增益在必测点和两种瓶子的 B 验证'}
        best = report.get('best_model')
        tested = [b for b in self.state['batches'] if b['phase'] == 'explore' and b['status'] == 'complete'
                  and b['candidate'].get('cond') == 'B' and best and b.get('model_id') == best['model_id']]
        from .e1e2 import rebuild_summary
        prospective = [r for b in tested for r in rebuild_summary(self.root / b['path'])
                       if r['kind'] == 'scan' and r['complete'] and r['recording_complete'] and r['label'] != 'invalid']
        b_rate = sum(r['success'] for r in prospective) / len(prospective) if prospective else None
        b_ok = b_rate is not None and b_rate >= self.settings['validation']['success_target']
        eligible = bool(report.get('stopping_eligible') and b_ok and len(tested) >= self.settings['learning'].get('min_B_batches', 1))
        unresolved = [c['id'] for c in (candidates or []) if c.get('unresolved')]
        eligible = eligible and not unresolved
        return {'eligible': eligible, 'mapping_eligible': bool(report.get('stopping_eligible')),
                'unresolved_design_candidates': unresolved,
                'B_batches_on_current_model': len(tested),
                'B_exploratory_success_rate': b_rate,
                'reason': '范围覆盖、留出误差和当前模型的 B 试验满足要求' if eligible else '需继续补充映射或当前版本 B 反馈'}

    def candidates(self, report):
        from .agentic_learning import propose_candidates
        from .agentic_design import enabled, propose_adaptive
        if self.state['phase'] != 'explore': return []
        remaining = self.settings['budget']['max_trials'] - self._spent() - self.state['validation_reserved_trials']
        options = propose_candidates(report, self.effective_settings(), remaining, len(self.state['batches']))
        if enabled(self.settings):
            options += propose_adaptive(report, self.settings, self.state['design'], remaining, len(self.state['batches']))
        valid = []; rejected = []
        for c in options:
            try:
                validate_candidate(c, self.settings, self.cfg, remaining, self.state['models'])
                validate_design_transition(c, self.settings, self.state.get('design'))
                valid.append(c)
            except ValueError as exc:
                rejected.append({'id': c.get('id'), 'reason': str(exc)})
        atomic_json(self.root / 'analysis' / 'candidate_checks.json', {'accepted': valid, 'rejected': rejected,
            'design': self.state.get('design'), 'remaining_exploration_trials': remaining})
        # Compare unresolved design questions before re-testing a model that an
        # impending range/resolution update may replace. Scores inside each
        # class rank equivalent research choices rather than batch-size sums.
        return sorted(valid, key=lambda c: (not bool(c.get('unresolved')), -c.get('score', 0.)))

    def decide(self, report, candidates, stop):
        if self.agent_planning():
            return self.decide_plan(report, candidates, stop)
        context = {'ranked_candidates': candidates,
            'approved_models': [{'model_id': k} for k in self.state['models']],
            'analysis': {k: report.get(k) for k in ('metrics', 'diagnostics', 'quality')},
            'stop': stop, 'history': self.state['history'][-6:],
            'constraints': {'scope': self.settings['scope'], 'budget': self.settings['budget'],
                            'current_design': self.state.get('design'),
                            'remaining_trials': self.settings['budget']['max_trials'] - self._spent(),
                            'fixed_execution_config_sha256': fingerprint(self.cfg)}}
        number = self.next_decision_number()
        atomic_json(self.root / 'decisions' / f'{number:04d}_input.json', context)
        backend = self.settings['provider']['backend']
        if backend in ('deterministic', 'mock'):
            if stop['eligible']: decision = {'action': 'freeze', 'reason': stop['reason']}
            elif candidates: decision = {'action': 'select_batch', 'candidate_id': candidates[0]['id'], 'reason': candidates[0]['reason']}
            else: decision = {'action': 'pause', 'reason': '没有满足当前预算和约束的候选实验'}
            decision['_meta'] = {'provider': 'explicit_deterministic', 'llm_used': False}
        else:
            if self.provider is None:
                from .agentic_provider import AgentProvider
                self.provider = AgentProvider(self.settings['provider'])
            decision = self.provider.propose(context)
        chosen = validate_decision(decision, candidates, stop)
        if chosen is not None: validate_design_transition(chosen, self.settings, self.state.get('design'))
        atomic_json(self.root / 'decisions' / f'{number:04d}_decision.json', decision)
        self.state['decisions'].append({'number': number, 'action': decision['action'], 'reason': decision['reason'], 'candidate_id': decision.get('candidate_id')})
        self.state.pop('last_error', None)
        self.save(); self.say('Agent：' + decision['reason'])
        return decision, chosen

    def planning_context(self, report, suggestions, stop):
        from .agentic_planning import build_context
        remaining = self.settings['budget']['max_trials'] - self._spent() - self.state['validation_reserved_trials']
        metrics_by_batch = {h['batch']: h.get('metrics', {}) for h in self.state['history'] if 'batch' in h}
        history = []
        for batch in self.state['batches']:
            candidate = batch['candidate']
            plan = (candidate.get('proposal') or {}).get('plan') or {}
            history.append({'id': batch['id'], 'batch': batch['id'], 'phase': batch['phase'],
                'round_index': candidate.get('round_index'), 'cond': candidate['cond'],
                'model_id': batch.get('model_id'), 'cells': candidate['cells'], 'policy': candidate.get('policy'),
                'status': batch['status'], 'reason': candidate.get('reason'),
                'hypothesis': plan.get('hypothesis'), 'expected_outcome': plan.get('expected_outcome'),
                'design': batch.get('design_after'), 'metrics': metrics_by_batch.get(batch['id'], {})})
        context = build_context(report, self.settings, self.cfg, self.state.get('design'),
                                history, {k: self.model(k) for k in self.state['models']},
                                remaining, stop, suggestions, len(self.state['batches']))
        context['fixed_execution_config_sha256'] = fingerprint(self.cfg)
        T0 = self.state['context'].get('T0')
        if T0 is not None:
            from .e1e2_core import MotionBounds
            bounds = MotionBounds(self.cfg, T0)
            # All three nominal targets have a linear z constraint in delta.
            lo, hi = self.settings['scope']['min_mm'], self.settings['scope']['max_mm']
            az = float(bounds.a[2]); zmin = -float(self.cfg['motion']['z_below_T0_mm'])
            if abs(az) > 1e-12:
                for extra in (0., -az * self.cfg['motion']['pre_mm'], self.cfg['motion']['lift_mm']):
                    cut = (zmin - extra) / az
                    if az > 0: lo = max(lo, cut)
                    else: hi = min(hi, cut)
            context['motion_evidence'] = {'a_base': bounds.a.tolist(),
                'T0_xyz_mm': np.asarray(T0)[:3, 3].tolist(),
                'nominal_height_feasible_offset_interval_mm': [lo, hi] if lo <= hi else None,
                'note': '仅当前 T0 的名义抓取/退让/提起高度计算；非碰撞保证，执行器继续逐目标检查'}
        return context

    def decide_plan(self, report, suggestions, stop):
        from .agentic_planning import validate_plan_response
        from .agentic_provider import AgentProvider, ProposalError
        context = self.planning_context(report, suggestions, stop)
        remaining = self.settings['budget']['max_trials'] - self._spent() - self.state['validation_reserved_trials']
        number = self.next_decision_number()
        prefix = self.root / 'decisions' / f'{number:04d}'
        atomic_json(str(prefix) + '_input.json', context)
        backend = self.settings['provider']['backend']
        deterministic = backend in ('deterministic', 'mock')
        if not deterministic and self.provider is None:
            self.provider = AgentProvider(self.settings['provider'])
        attempts = self.settings.get('planning', {}).get('max_attempts', 3)
        for attempt in range(1, attempts + 1):
            attempt_prefix = str(prefix) + f'_attempt_{attempt:02d}'
            atomic_json(attempt_prefix + '_input.json', context)
            decision = None
            try:
                if deterministic:
                    # This path is an explicitly selected offline baseline, never
                    # a fallback for failed ChatGPT planning.
                    if stop['eligible']:
                        decision = {'action': 'freeze', 'reason': stop['reason'], 'plan': None}
                    elif suggestions:
                        suggestion = suggestions[(attempt - 1) % len(suggestions)]
                        decision = {'action': 'propose_batch', 'reason': '显式 deterministic 规则基线：' + suggestion['reason'],
                            'plan': {'cond': suggestion['cond'], 'cells': copy.deepcopy(suggestion['cells']),
                                     'model_id': suggestion.get('model_id') if suggestion['cond'] == 'B' else None,
                                     'gain': suggestion.get('policy', {}).get('gain', 1.) if suggestion['cond'] == 'B' else 1.,
                                     'hypothesis': '显式规则基线用于离线比较，不是 Agent 自拟计划',
                                     'evidence_refs': ['metrics'], 'expected_outcome': '通过新增真实或模拟测量检验当前映射与纠正表现'}}
                    else:
                        decision = {'action': 'pause', 'reason': '显式规则基线没有候选；未调用 Agent', 'plan': None}
                    decision['_meta'] = {'provider': 'explicit_deterministic', 'llm_used': False}
                else:
                    decision = self.provider.propose(context)
                payload = {k: v for k, v in decision.items() if k != '_meta'} if isinstance(decision, dict) else decision
                # Write only strict finite data; malformed provider output is
                # represented by its rejection below, never executed.
                json.dumps(payload, allow_nan=False)
                atomic_json(attempt_prefix + '_proposal.json', decision)
                chosen = validate_plan_response(payload, settings=self.settings, cfg=self.cfg,
                    design=self.state.get('design'), approved_models={k: self.model(k) for k in self.state['models']},
                    remaining_trials=remaining, context=context, round_index=len(self.state['batches']),
                    T0=self.state['context'].get('T0'))
                if decision['action'] == 'freeze' and not stop['eligible']:
                    raise ValueError('停止条件未满足；请查看 stop 中的映射精度和缺失 B 覆盖，不能提前宣告结束')
                atomic_json(attempt_prefix + '_check.json', {'accepted': True,
                    'candidate_id': chosen['id'] if chosen else None,
                    'warnings': chosen.get('warnings', []) if chosen else []})
            except (ValueError, TypeError, ProposalError) as exc:
                error = str(exc)[:2000] or type(exc).__name__
                code = getattr(exc, 'code', 'invalid_proposal')
                atomic_json(attempt_prefix + '_check.json', {'accepted': False, 'code': code, 'reason': error})
                self.say(f'Agent 计划未通过检查（{attempt}/{attempts}）：{error}')
                context['revision'] = {'attempt': attempt + 1, 'previous_rejection': {'code': code, 'reason': error},
                    'instruction': '根据具体拒绝原因重新提交完整计划；不能修改约束，无法解决则 pause。'}
                # A valid-shaped rejected plan is useful for targeted repair;
                # provider parser failures deliberately do not expose raw text.
                if isinstance(decision, dict):
                    try:
                        clean = {k: v for k, v in decision.items() if k != '_meta'}
                        json.dumps(clean, allow_nan=False)
                        context['revision']['previous_proposal'] = clean
                    except (TypeError, ValueError):
                        pass
                if attempt < attempts:
                    continue
                self.state['decisions'].append({'number': number, 'action': 'rejected', 'reason': error,
                                               'attempts': attempt, 'candidate_id': None})
                self.save()
                raise ValueError(f'Agent 连续 {attempts} 次计划未通过检查，已暂停；未切换到规则方案。最后原因：{error}') from None
            except Exception as exc:
                # Transport/auth failures are not proposal mistakes: pause now.
                atomic_json(attempt_prefix + '_check.json', {'accepted': False, 'code': 'provider_unavailable',
                                                            'reason': str(exc)[:2000]})
                self.state['decisions'].append({'number': number, 'action': 'provider_error',
                                               'reason': str(exc)[:2000], 'attempts': attempt, 'candidate_id': None})
                self.save()
                raise
            atomic_json(str(prefix) + '_decision.json', decision)
            if chosen:
                chosen['decision_number'] = number
                atomic_json(str(prefix) + '_validated_plan.json', chosen)
                self.say('Agent 自拟计划：' + chosen['cond'] + '；' +
                         '，'.join(f"{c['offset_mm']:+g} mm × {c['reps']}" for c in chosen['cells']))
            self.state['decisions'].append({'number': number, 'action': decision['action'],
                'reason': decision['reason'], 'candidate_id': chosen['id'] if chosen else None, 'attempts': attempt})
            self.state.pop('last_error', None)
            self.save(); self.say('Agent：' + decision['reason'])
            return decision, chosen

    def freeze(self, report, reason, trigger='agent'):
        model = report.get('best_model')
        if not model or not model.get('usable_for_B'):
            self.state.update(status='no_usable_model', stop_reason=reason + '；没有可部署的触觉模型')
            self.save(); return
        chosen_gain = 1.; selection = 'default; no comparable prospective gain evidence'
        if not self.agent_planning():
            # A gain is selected only from completed prospective B batches on this model.
            gain_scores = {}
            from .e1e2 import rebuild_summary
            for b in self.state['batches']:
                if b['status'] != 'complete' or b['phase'] != 'explore' or b.get('model_id') != model['model_id'] or b['candidate'].get('cond') != 'B': continue
                gain = b['candidate'].get('policy', {}).get('gain', 1.)
                rows = [r for r in rebuild_summary(self.root / b['path']) if r['kind'] == 'scan' and r['recording_complete'] and r['label'] != 'invalid']
                # Compare only matching bottle/offset cells across gains below.
                gain_scores.setdefault(gain, []).extend(rows)
            chosen_gain = 1.; selection = 'default; no comparable prospective gain evidence'
            if gain_scores:
                common = set.intersection(*[{(r['bottle'], r['delta_mm']) for r in rows} for rows in gain_scores.values()])
                if common:
                    def score(g):
                        rows = [r for r in gain_scores[g] if (r['bottle'], r['delta_mm']) in common]
                        cells = [[r for r in rows if (r['bottle'], r['delta_mm']) == key] for key in common]
                        return (float(np.mean([np.mean([r['success'] for r in cell]) for cell in cells])),
                                -float(np.mean([np.mean([r['regrasps'] for r in cell]) for cell in cells])))
                    chosen_gain = max(gain_scores, key=score); selection = 'prospective B: equal-weight matched bottle/offset cells; success then fewer regrasps'
        stop_snapshot = None
        if self.agent_planning():
            stop_snapshot = self.stop_evidence(report)
            evidence = stop_snapshot['B_evidence']
            chosen_gain = evidence['chosen_gain']
            if chosen_gain is None:
                allowed = self.settings['policy']['gains']
                chosen_gain = 1. if 1. in allowed else allowed[-1]
                selection = 'provisional default: no gain has complete prospective B coverage'
            else:
                selection = ('current-model same-gain fixed-offset balanced prospective B; ' +
                             ('criteria met' if evidence['eligible'] else 'provisional; criteria not met'))
            if trigger == 'agent' and not stop_snapshot['eligible']:
                raise ValueError('Agent 请求冻结但数值停止条件未满足')
        self.state['frozen'] = {'model_id': model['model_id'], 'policy': {'gain': chosen_gain},
            'reason': reason, 'mapping_eligible': bool(report.get('stopping_eligible')), 'selection': selection,
            'freeze_trigger': trigger,
            'exploration_eligible': stop_snapshot['eligible'] if stop_snapshot else None,
            'stop_evidence': stop_snapshot,
            'design': copy.deepcopy(self.state.get('design')),
            'validation_offsets_mm': validation_offsets(self.settings, self.state.get('design')),
            'training_sessions': self.training_sessions(), 'created_unix_ns': time.time_ns()}
        atomic_json(self.root / 'frozen.json', self.state['frozen'])
        self.state.update(phase='validation', status='running'); self.save()

    def recheck_agent_plan(self, candidate, remaining):
        from .agentic_planning import validate_plan_response
        number = candidate.get('decision_number')
        if isinstance(number, bool) or not isinstance(number, int) or number < 1:
            raise ValueError('Agent 计划缺少已保存的决策记录')
        prefix = self.root / 'decisions' / f'{number:04d}'
        context = json.loads(Path(str(prefix) + '_input.json').read_text())
        saved = json.loads(Path(str(prefix) + '_validated_plan.json').read_text())
        if candidate != saved:
            raise ValueError('Agent 计划在校验后被修改；拒绝执行')
        checked = validate_plan_response(candidate['proposal'], settings=self.settings, cfg=self.cfg,
            design=self.state.get('design'), approved_models={k: self.model(k) for k in self.state['models']},
            remaining_trials=remaining, context=context, round_index=len(self.state['batches']),
            T0=self.state['context'].get('T0'))
        if checked is None or checked != {k: v for k, v in candidate.items() if k != 'decision_number'}:
            raise ValueError('Agent 计划与当前状态不一致，需重新制定计划')
        return checked['design_next']

    def prepare_batch(self, candidate, validation=False):
        if not validation and self.state['phase'] != 'explore':
            raise ValueError('模型已冻结，不能创建新的探索批次或改变设计')
        index = len(self.state['batches']) + 1
        fit_id = self.state['frozen']['model_id'] if validation else candidate.get('model_id')
        candidate = copy.deepcopy(candidate)
        before = copy.deepcopy(self.state.get('design'))
        after = before
        if validation:
            expected = self.frozen_validation_candidate()
            if candidate['cells'] != expected['cells']: raise ValueError('验证点已冻结，不能更换')
            candidate['policy'] = self.state['frozen']['policy']
        else:
            remaining = self.settings['budget']['max_trials'] - self._spent() - self.state['validation_reserved_trials']
            validate_candidate(candidate, self.settings, self.cfg, remaining, self.state['models'])
            if candidate.get('origin') == 'agent_proposal':
                if not self.agent_planning(): raise ValueError('旧 campaign 不能静默改为 Agent 自拟模式')
                transition = self.recheck_agent_plan(candidate, remaining)
            else:
                if self.agent_planning(): raise ValueError('Agent 自拟模式只能执行经过校验并保存的完整提案')
                transition = validate_design_transition(candidate, self.settings, before)
            if transition is not None: after = {**(before or {}), **transition}
        plan = make_plan(candidate, self.cfg, seed=self.settings['seed'] + index, validation=validation)
        if self._spent() + len(plan) > self.settings['budget']['max_trials']:
            raise ValueError('剩余总预算不足以执行完整批次，不能裁切独立验证计划')
        batch = {'id': f'{index:04d}', 'path': f'batches/{index:04d}', 'phase': 'validation' if validation else 'explore',
                 'candidate': candidate, 'plan': plan, 'model_id': fit_id, 'status': 'planned',
                 'design_before': before, 'design_after': copy.deepcopy(after)}
        atomic_json(self.root / batch['path'] / 'batch_plan.json', batch)
        if after != before:
            self.state['design'] = after
            self.state.setdefault('design_history', []).append({'batch': batch['id'], 'candidate_id': candidate['id'],
                'action': candidate.get('design_action'), 'before': before, 'after': copy.deepcopy(after),
                'reason': candidate['reason'], 'created_unix_ns': time.time_ns()})
            self.say(f"探索设计更新：{before} → {after}；{candidate.get('design_action')}")
        self.state['batches'].append(batch); self.state['active_batch'] = batch['id']; self.save()
        return batch

    def execute(self, batch):
        expected = self.state.get('hardware_config_manifest', {})
        if expected != self.hardware_manifest(self.cfg, self.settings):
            raise ValueError('硬件配置在 campaign 开始后发生变化；拒绝执行，请新建 campaign')
        if self.executor is None:
            from .agentic_executor import execute_batch
            self.executor = execute_batch
        cfg = copy.deepcopy(self.cfg)
        cfg['servo']['gain'] = batch['candidate'].get('policy', {}).get('gain', 1.)
        context = copy.deepcopy(self.state['context'])
        context['continuous_campaign'] = not self.first_batch_this_process
        context['campaign_root'] = str(self.root)
        context['batch_role'] = 'validation' if batch['phase'] == 'validation' else 'exploration'
        elapsed = self.state['elapsed_s'] + max(0., time.time() - (self.state.get('active_run_unix_ns') or time.time_ns()) / 1e9)
        context['deadline_unix_ns'] = time.time_ns() + int(max(0., self.settings['budget']['max_seconds'] - elapsed) * 1e9)
        from .e1e2 import rebuild_summary
        completed = {r['id'] for r in rebuild_summary(self.root / batch['path']) if r['complete']}
        pending = sum(item['id'] not in completed for item in batch['plan'])
        ceiling = self.settings['budget']['max_trials']
        if batch['phase'] != 'validation': ceiling -= self.state['validation_reserved_trials']
        if self._spent() + pending > ceiling:
            raise ValueError('中断尝试也计入预算；剩余预算不足以完成原批次，已停止，保留全部记录')
        self.say(f"批次 {batch['id']}：{batch['phase']}，{len(batch['plan'])} 次，{batch['candidate']['reason']}")
        batch['status'] = 'running'; self.state['status'] = 'running'; self.save()
        result = self.executor(self.root / batch['path'], batch['plan'], cfg, self.model(batch['model_id']), context,
            simulate=bool(self.settings.get('simulate')), ask=self.ask, say=self.say,
            authorize=self.first_batch_this_process or self.settings.get('approval') == 'batch')
        self.first_batch_this_process = False
        self.state['context'] = plain(result.get('context', context)); batch['result'] = plain(result)
        batch['status'] = 'complete' if result.get('complete') else 'paused'
        if result.get('complete'): self.state['active_batch'] = None
        if result.get('quit') or not result.get('complete'): self.state['status'] = 'paused'
        if result.get('deadline_reached'):
            self.state.update(status='budget_exhausted', stop_reason='时间预算耗尽；已安全完成当前试验，未完成的验证不作成功结论')
        self.save()
        return result

    def recover_context(self):
        """Recover T0 committed by the runner even if the process died mid-batch."""
        active = self.state.get('active_batch')
        if not active: return
        batch = next(b for b in self.state['batches'] if b['id'] == active)
        p = self.root / batch['path']; m = p / 'session_meta.json'
        if m.exists():
            meta = json.loads(m.read_text())
            if meta.get('T0'):
                old = self.state['context'].get('T0')
                if old is not None and not np.allclose(old, meta['T0'], atol=1e-7):
                    raise ValueError('批次 T0 与 campaign T0 不一致，拒绝自动恢复')
                from .agentic_executor import recover_context
                self.state['context'] = recover_context(p, self.state['context']); self.save()

    def finish_validation(self):
        from .agentic_learning import frozen_evaluation, write_plots
        roots = [str(self.root / b['path']) for b in self.state['batches'] if b['phase'] == 'validation' and b['status'] == 'complete']
        model = self.model(self.state['frozen']['model_id'])
        settings = copy.deepcopy(self.settings)
        settings['validation']['offsets_mm'] = self.state['frozen'].get('validation_offsets_mm', settings['validation']['offsets_mm'])
        settings['validation']['frozen_gain'] = self.state['frozen']['policy']['gain']
        report = frozen_evaluation(roots, roots, model, settings)
        report['plots'] = write_plots(report, self.root / 'validation' / 'plots')
        atomic_json(self.root / 'validation' / 'report.json', report)
        self.state.update(status='complete', validation=report, stop_reason=self.state['frozen']['reason'])
        self.state.pop('last_error', None)
        self.save(); self.write_report()

    def write_report(self):
        s = self.state
        metrics = s.get('latest_metrics', {})
        validation = s.get('validation') or {}
        outcome = {'passed': '通过配置的独立验证标准', 'not_met': '未通过独立验证标准',
                   'inconclusive': '验证证据不足，不能宣称通过', 'invalid': '验证记录不满足协议，不能用于验收'}.get(validation.get('outcome'), '独立验证尚未完成')
        lines = ['# Agentic 实验报告', '', f"状态：{s['status']}",
                 f"模式：{'模拟（不能作为实机结论）' if self.settings.get('simulate') else '实机'}",
                 f"计划方式：{'Agent 提交完整计划，本地校验' if self.agent_planning() else '旧版候选选择'}",
                 f"固定硬边界：{self.settings['scope']['min_mm']} 至 {self.settings['scope']['max_mm']} mm",
                 f"当前探索设计：{s.get('design') or self.settings['scope']}",
                 f"新增尝试/预算（含监测、中断和跳过）：{self._spent()}/{self.settings['budget']['max_trials']}",
                 f"停止原因：{s.get('stop_reason', s.get('last_error', '尚未结束'))}", '',
                 f"冻结触发：{(s.get('frozen') or {}).get('freeze_trigger', '尚未冻结')}；探索停止条件满足：{(s.get('frozen') or {}).get('exploration_eligible')}",
                 f"结论：{outcome}。complete 只表示流程结束。",
                 f"留出映射：MAE={metrics.get('mae_mm')} mm；P90={metrics.get('p90_mm')} mm；范围覆盖={metrics.get('scope_covered')}。", '',
                 '## 批次与更新', '', '| 批次 | 阶段 | 条件 | 状态 | 理由 |', '|---|---|---|---|---|']
        for b in s['batches']:
            lines.append(f"| {b['id']} | {b['phase']} | {b['candidate']['cond']} | {b['status']} | {b['candidate']['reason'].replace('|','/')} |")
        lines += ['', '## 范围与步长更新记录', '', '```json',
                  json.dumps(s.get('design_history', []), ensure_ascii=False, indent=2), '```',
                  '步长是计划间隔，实际分辨率由实测 TCP 和重复噪声判断。固定验证只支持所测点位的结论，不证明整个连续范围。']
        lines += ['', '## 留出映射误差', '', '```json', json.dumps(s.get('latest_metrics', {}), ensure_ascii=False, indent=2), '```',
                  '', '## 冻结后的独立验证', '', '探索期样本用于学习；验证结果不反馈到本轮模型。预算终止不等于达到精度目标。',
                  '', '```json', json.dumps(s.get('validation', {'status': '尚未完成'}), ensure_ascii=False, indent=2), '```',
                  '', '模型版本在 models/，原始逐次数据在 batches/，每次决策的输入和结果在 decisions/。',
                  '实测 TCP 位移不是瓶子发生移动时的真实相对偏移；推动/倾倒标签不用于映射训练。']
        plots = sorted((self.root / 'analysis' / 'plots').glob('*.png')) + sorted((self.root / 'validation' / 'plots').glob('*.png'))
        if plots:
            lines += ['', '## 结果图', '']
            for plot in plots:
                relative = plot.relative_to(self.root).as_posix()
                lines += [f'![{plot.stem}]({relative})', '']
        (self.root / 'REPORT.md').write_text('\n'.join(lines) + '\n')

    def acknowledge_recovery(self):
        active = self.state.get('active_batch')
        if not active: raise ValueError('没有需要恢复的活动批次')
        batch = next(b for b in self.state['batches'] if b['id'] == active)
        path = self.root / batch['path']
        self.recover_context()
        if not self.state['context'].get('T0'): raise ValueError('尚无固定 T0，请创建新的 campaign')
        self.say('本命令不连接机器人。请现场把瓶子安全放回、夹爪完全张开，并恢复到原 T0。')
        self.say('原 T0 基座 xyz(mm)：' + str(np.asarray(self.state['context']['T0'])[:3, 3].tolist()))
        if self.ask('确认上述现场处理已完成，输入 RECOVERED；其它输入取消：\n> ').strip() != 'RECOVERED':
            self.say('取消；保留原恢复状态。'); return
        files = [path / 'progress.json', path / 'session_meta.json']
        before = {str(p.relative_to(self.root)): json.loads(p.read_text()) for p in files if p.exists()}
        atomic_json(self.root / 'recovery_acknowledgements' / f'{time.time_ns()}.json', {
            'operator_assertion': 'RECOVERED: bottle supported, gripper fully open, original T0 restored',
            'before': before, 'campaign_context_before': self.state['context'], 'motion_sent': False})
        context = self.state['context']
        context.update(airborne=False, unsafe_recovery=False, continuous_campaign=False,
            last_safe_pose=context['T0'], last_safe_target=context['T0'], last_grasp_target=context['T0'])
        for p in files:
            if not p.exists(): continue
            data = json.loads(p.read_text())
            if p.name == 'progress.json': data.update(airborne=False, unsafe_recovery=False)
            else: data['agentic_context'] = copy.deepcopy(context)
            atomic_json(p, data)
        self.state['status'] = 'paused'; self.save()
        self.say('已记录人工恢复；续跑仍会读取实测开度和位姿，首段动作需要 MOVE。')

    def run(self):
        if self.state['status'] in TERMINAL:
            self.say('此 campaign 已结束；不连接硬件。报告：' + str(self.root / 'REPORT.md')); return 0
        self.recover_context()
        started = time.monotonic()
        if self.state.get('active_run_unix_ns'):
            self.state['elapsed_s'] += max(0., (time.time_ns() - self.state['active_run_unix_ns']) / 1e9)
        self.state['active_run_unix_ns'] = time.time_ns(); self.save()
        self.say('授权探索范围 ' + str(self.settings['scope']) + '；固定限幅和力控不由 agent 修改。')
        self.say('总预算包含独立验证预留 ' + str(self.state['validation_reserved_trials']) + ' 次；标签与换瓶仍由现场操作员完成。')
        self.say('计划方式：' + ('Agent 自拟点位、次数和 A/B；本地校验，不受候选菜单限制。' if self.agent_planning() else '旧版：从本地候选中选择。'))
        try:
            while True:
                if self.state.get('active_batch'):
                    b = next(b for b in self.state['batches'] if b['id'] == self.state['active_batch'])
                    result = self.execute(b)
                    if self.state['status'] == 'paused' or self.state['status'] in TERMINAL: break
                    if b['phase'] == 'validation': self.finish_validation(); break
                    report = self.analyze()
                    self.state['history'].append({'batch': b['id'], 'cond': b['candidate']['cond'], 'model_id': b.get('model_id'),
                        'design': copy.deepcopy(self.state.get('design')), 'metrics': report.get('metrics', {})})
                    self.save()
                    continue
                if self.state['phase'] == 'validation':
                    if any(b['phase'] == 'validation' and b['status'] == 'complete' for b in self.state['batches']):
                        self.finish_validation(); break
                    self.prepare_batch(self.frozen_validation_candidate(), validation=True); continue
                report = self.analyze()
                blockers = report.get('quality', {}).get('blocking_reasons', [])
                if blockers:
                    self.state.update(status='paused', last_error='数据质量阻止自动采集：' + '; '.join(map(str, blockers)))
                    self.save(); self.say(self.state['last_error']); break
                options = self.candidates(report); stop = self.stop_evidence(report, options)
                spent = self._spent(); budget = self.settings['budget']
                limit = len([b for b in self.state['batches'] if b['phase'] == 'explore']) >= budget['max_batches']
                minimum_batch = len(make_plan({'cond': 'A', 'cells': [{'offset_mm': 0., 'reps': 2}]},
                                              self.cfg, seed=self.settings['seed']))
                limit = limit or budget['max_trials'] - self.state['validation_reserved_trials'] - spent < minimum_batch
                limit = limit or self.state['elapsed_s'] + time.monotonic() - started >= budget['max_seconds']
                # Only distinct A updates count as learning plateaus. Repeated
                # B-only evaluations cannot manufacture mapping convergence.
                updates = [h['metrics'].get('mae_mm') for h in self.state['history'] if h['cond'] == 'A'
                           and h.get('design') == self.state.get('design')]
                n = self.settings['learning']['plateau_batches']
                recent = updates[-(n+1):]
                plateau = (len(recent) == n+1 and all(v is not None for v in recent) and
                           all(abs(a-b) < self.settings['learning']['min_gain_mm'] for a,b in zip(recent,recent[1:])))
                plateau = plateau and not any(c.get('unresolved') for c in options)
                if self.state['elapsed_s'] + time.monotonic() - started >= budget['max_seconds']:
                    self.state.update(status='budget_exhausted', stop_reason='总时间预算耗尽，未完成的独立验证不作成功结论')
                    self.save(); break
                if limit or (not self.agent_planning() and (not options or plateau)):
                    reason = '连续 A 更新收益低，转入固定验证；不代表误差目标已经达到' if plateau else '探索预算已到或没有可执行的补采候选；不代表精度已经达到'
                    self.freeze(report, reason, trigger='budget' if limit else ('plateau' if plateau else 'no_candidates'))
                    if self.state['status'] in TERMINAL: break
                    continue
                decision, chosen = self.decide(report, options, stop)
                if self.state['elapsed_s'] + time.monotonic() - started >= budget['max_seconds']:
                    self.state.update(status='budget_exhausted', stop_reason='制定计划期间总时间预算耗尽；未执行新计划')
                    self.save(); break
                if decision['action'] == 'pause':
                    self.state.update(status='paused', last_error=decision['reason']); self.save(); break
                if decision['action'] == 'freeze': self.freeze(report, decision['reason']); continue
                self.prepare_batch(chosen)
        except (Exception, KeyboardInterrupt) as exc:
            self.state.update(status='paused', last_error=str(exc) or type(exc).__name__)
            atomic_json(self.root / 'errors' / f'{time.time_ns()}.json', {'error': self.state['last_error'], 'type': type(exc).__name__})
            self.save(); self.say('Agentic 已暂停：' + self.state['last_error'])
            return 130 if isinstance(exc, KeyboardInterrupt) else 2
        finally:
            self.state['elapsed_s'] += time.monotonic() - started
            self.state['active_run_unix_ns'] = None
            if self.provider is not None and hasattr(self.provider, 'close'): self.provider.close()
            self.save(); self.write_report()
        self.say('Campaign 状态：' + self.state['status'] + '；报告：' + str(self.root / 'REPORT.md'))
        return 0


def parser():
    p = argparse.ArgumentParser(description='完整主动实验循环；help/status/analyze/auth 不连接机器人')
    sub = p.add_subparsers(dest='command', required=True)
    for name in ('run', 'resume'):
        q = sub.add_parser(name); q.add_argument('--out', type=Path, required=True)
        q.add_argument('--config', type=Path, default=ROOT / 'config.agentic.yaml')
        q.add_argument('--seed-session', action='append', type=Path)
        mode = q.add_mutually_exclusive_group(required=True)
        mode.add_argument('--real', action='store_true'); mode.add_argument('--simulate', action='store_true')
        q.add_argument('--provider', choices=['chatgpt', 'api', 'deterministic'])
        q.add_argument('--planning', choices=['agent', 'candidate'], help='仅新 campaign：Agent 自拟完整计划或旧版候选选择')
        q.add_argument('--max-trials', type=int); q.add_argument('--max-batches', type=int)
        q.add_argument('--scope', nargs=2, type=float, metavar=('MIN_MM', 'MAX_MM'))
        q.add_argument('--approval', choices=['campaign', 'batch'])
    for name in ('status', 'analyze', 'plan', 'recover'):
        q = sub.add_parser(name); q.add_argument('--out', type=Path, required=True)
    auth = sub.add_parser('auth'); auth.add_argument('operation', choices=['login', 'status', 'logout', 'models', 'test'])
    auth.add_argument('--config', type=Path, default=ROOT / 'config.agentic.yaml')
    auth.add_argument('--proxy'); auth.add_argument('--identity-dir'); auth.add_argument('--auth-dir')
    auth.add_argument('--model'); auth.add_argument('--open-browser', action='store_true')
    return p


def main(argv=None, *, input_fn=input, output=print, executor=None, provider=None):
    args = parser().parse_args(argv)
    lock = None
    try:
        if args.command == 'auth':
            settings = load_settings(args.config)['provider']
            for k in ('proxy', 'identity_dir', 'auth_dir', 'model'):
                if getattr(args, k) is not None: settings[k] = getattr(args, k)
            if args.open_browser: settings['open_browser'] = True
            from . import agentic_auth as auth
            from .agentic_provider import list_models, AgentProvider
            result = {'login': lambda: auth.login(settings, say=output),
                      'status': lambda: auth.auth_status(settings), 'logout': lambda: auth.logout(settings),
                      'models': lambda: list_models(settings), 'test': lambda: AgentProvider(settings).test_connection()}[args.operation]()
            output(json.dumps(plain(result), ensure_ascii=False, indent=2)); return 0
        root = args.out.expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        lock = (root / '.campaign.lock').open('a')
        try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError: raise ValueError('同一 campaign 已有进程，请勿并行控制')
        if args.command in ('status', 'analyze', 'plan', 'recover'):
            campaign = Campaign(root, ask=input_fn, say=output)
            if args.command == 'status':
                output(json.dumps({'planning_mode': 'agent' if campaign.agent_planning() else 'candidate',
                    **{k: campaign.state.get(k) for k in ('status', 'phase', 'active_batch', 'design', 'design_history', 'latest_metrics', 'frozen', 'last_error')}}, ensure_ascii=False, indent=2))
            elif args.command == 'recover': campaign.acknowledge_recovery()
            else:
                report = campaign.analyze()
                if args.command == 'analyze' and campaign.state.get('frozen') and any(
                        b['phase'] == 'validation' and b['status'] == 'complete' for b in campaign.state['batches']):
                    campaign.finish_validation()
                else:
                    campaign.write_report()
                if args.command == 'plan':
                    suggestions = campaign.candidates(report)
                    view = (campaign.planning_context(report, suggestions, campaign.stop_evidence(report, suggestions))
                            if campaign.agent_planning() else suggestions)
                    output(json.dumps(view, ensure_ascii=False, indent=2))
                else: output('报告：' + str(root / 'REPORT.md'))
            return 0
        if (root / 'campaign.json').exists():
            old = json.loads((root / 'campaign.json').read_text())['settings']
            if bool(args.simulate) != bool(old.get('simulate')): raise ValueError('不能将模拟 campaign 续跑为实机，或反向转换')
            if any(getattr(args, k) is not None for k in ('seed_session', 'provider', 'planning', 'max_trials', 'max_batches', 'scope', 'approval')):
                raise ValueError('续跑只需 --out 和 --real/--simulate；不能覆盖已冻结的 campaign 设置')
            campaign = Campaign(root, ask=input_fn, say=output, executor=executor, provider=provider)
        else:
            s = load_settings(args.config); s['simulate'] = bool(args.simulate)
            if args.seed_session: s['seed_sessions'] = [str(x.expanduser().resolve()) for x in args.seed_session]
            if args.provider: s['provider']['backend'] = args.provider
            if args.planning: s.setdefault('planning', {})['mode'] = args.planning
            if args.max_trials is not None: s['budget']['max_trials'] = args.max_trials
            if args.max_batches is not None: s['budget']['max_batches'] = args.max_batches
            if args.scope:
                s['scope'].update(min_mm=args.scope[0], max_mm=args.scope[1])
                if s['scope'].get('adaptive', False):
                    s['scope']['initial_min_mm'] = max(args.scope[0], s['scope'].get('initial_min_mm', args.scope[0]))
                    s['scope']['initial_max_mm'] = min(args.scope[1], s['scope'].get('initial_max_mm', args.scope[1]))
                    s['validation']['offsets_mm'] = validation_offsets(s)
                else:
                    s['validation']['offsets_mm'] = sorted(set([args.scope[0], 0., args.scope[1]]))
            if args.approval: s['approval'] = args.approval
            validate_settings(s)
            campaign = Campaign(root, settings=s, ask=input_fn, say=output, executor=executor, provider=provider)
        return campaign.run()
    except (Exception, KeyboardInterrupt) as exc:
        output('Agentic 停止：' + (str(exc) or type(exc).__name__))
        return 130 if isinstance(exc, KeyboardInterrupt) else 2
    finally:
        if lock: lock.close()
