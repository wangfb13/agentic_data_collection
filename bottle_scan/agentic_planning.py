"""Convert an agent's declarative experiment plan into a locally checked batch.

No arbitrary code, shell command, controller parameter or device action is
accepted. Historical failures and heuristic rankings are evidence for the agent,
not extra motion prohibitions. Only explicit protocol/budget/physical constraints
are enforced here. Hardware is never initialized by this module.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
import math

import numpy as np


class PlanningError(ValueError):
    """Safe structured feedback; code/reason never contain raw provider output."""
    def __init__(self, code, reason):
        self.code, self.reason = code, reason
        super().__init__(reason)


def _fail(code, reason):
    raise PlanningError(code, reason)


def enabled(settings):
    return settings.get('planning', {}).get('mode', 'candidate') == 'agent'


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _text(value, limit):
    return isinstance(value, str) and bool(value.strip()) and len(value) <= limit and '\x00' not in value


def validate_response(value):
    """Validate only the closed JSON response shape, before contextual checks."""
    if not isinstance(value, dict) or set(value) != {'action', 'reason', 'plan'}:
        _fail('response_keys', '响应只能包含 action、reason、plan 三个字段。')
    if value['action'] not in ('propose_batch', 'freeze', 'pause'):
        _fail('action', 'action 必须为 propose_batch、freeze 或 pause。')
    if not _text(value['reason'], 4000):
        _fail('reason', 'reason 不能为空，且不能超过 4000 字符。')
    plan = value['plan']
    if value['action'] in ('freeze', 'pause'):
        if plan is not None:
            _fail('nonbatch_plan', '冻结或暂停时 plan 必须为 null。')
        return deepcopy(value)
    keys = {'cond', 'cells', 'model_id', 'gain', 'hypothesis', 'evidence_refs', 'expected_outcome'}
    if not isinstance(plan, dict) or set(plan) != keys:
        _fail('plan_keys', 'plan 只能包含 cond、cells、model_id、gain、hypothesis、evidence_refs、expected_outcome。')
    if plan['cond'] not in ('A', 'B'):
        _fail('condition', '实验条件必须为 A 或 B。')
    if plan['model_id'] is not None and not _text(plan['model_id'], 128):
        _fail('model_id', 'model_id 必须为 null 或不超过 128 字符的非空文本。')
    if not _finite(plan['gain']):
        _fail('gain_number', 'gain 必须为有限数值。')
    for key in ('hypothesis', 'expected_outcome'):
        if not _text(plan[key], 2000):
            _fail('plan_explanation', '实验假设和预期结果均不能为空，各不超过 2000 字符。')
    refs = plan['evidence_refs']
    if (not isinstance(refs, list) or not 1 <= len(refs) <= 128 or
            any(not _text(ref, 128) for ref in refs) or len(set(refs)) != len(refs)):
        _fail('evidence_shape', '证据引用必须为非空列表，不能重复，最多 128 项。')
    cells = plan['cells']
    if not isinstance(cells, list) or not 1 <= len(cells) <= 64:
        _fail('cells_shape', 'cells 必须为非空列表，最多包含 64 个点位。')
    for cell in cells:
        if not isinstance(cell, dict) or set(cell) != {'offset_mm', 'reps'}:
            _fail('cell_keys', '每个点位只能包含 offset_mm 和 reps。')
        if not _finite(cell['offset_mm']):
            _fail('offset_number', '偏移必须为有限数值。')
        if type(cell['reps']) is not int or not 1 <= cell['reps'] <= 10000:
            _fail('reps_shape', '重复数必须为正整数，最多 10000。')
    return deepcopy(value)


def _compact(value, *, depth=0):
    """Keep diagnostics readable without shipping frames, arrays or raw files."""
    if depth > 7:
        return '[nested detail omitted]'
    if value is None or isinstance(value, (str, bool, int)):
        return value[:2000] if isinstance(value, str) else value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        forbidden = {'median_dd', 'dd', 'raw_dd', 'image', 'pixels', 'session_meta', 'config', 'observations', 'validation_predictions', 'source_trials'}
        return {str(k): _compact(v, depth=depth + 1) for k, v in list(value.items())[:96] if k not in forbidden}
    if isinstance(value, (list, tuple)):
        return [_compact(v, depth=depth + 1) for v in value[:128]]
    return '[unsupported data omitted]'


def _ref_offset(value):
    return format(0. if float(value) == 0 else float(value), '.10g')


def _failure_summary(report, settings):
    # The numerical report excludes invalid/no-contact rows from mapping. Read
    # their compact outcome metadata so an agent can still design informative A.
    from .agentic_design import _load_outcomes
    groups = defaultdict(list)
    for row in _load_outcomes(report, settings):
        groups[row['offset_mm']].append(row)
    return [{'offset_mm': offset, 'n': len(rows),
             'successful_A': sum(r['passed'] for r in rows),
             'contact_force_failures': sum(r['contact_force_failure'] for r in rows),
             'execution_blocked': sum(r['blocked'] for r in rows),
             'complete_recordings': sum(r['complete'] and r['recording'] for r in rows),
             'reasons': dict(Counter(str(r['reason']) for r in rows))}
            for offset, rows in sorted(groups.items())]


def build_context(report, settings, cfg, design, history, approved_models, remaining_trials, stop, suggestions, round_index):
    """Supply statistics and reference ids, not raw DD or executable tools."""
    references = {'metrics': _compact(report.get('metrics', {})), 'stop': _compact(stop)}
    levels = _compact(report.get('evidence', {}).get('level_stats', []))
    for level in levels:
        if isinstance(level, dict) and _finite(level.get('offset_mm')):
            references['level:' + _ref_offset(level['offset_mm'])] = level
    diagnostics = [_compact(d) for d in report.get('diagnostics', [])]
    for i, diagnostic in enumerate(diagnostics):
        references['diagnostic:' + str(i)] = diagnostic
    failures = _failure_summary(report, settings)
    for outcome in failures:
        references['outcome:' + _ref_offset(outcome['offset_mm'])] = outcome
    if not levels and not failures:
        references['no_data'] = {'status': 'no physical mapping observations yet',
                                 'purpose': 'propose a bounded initial A experiment'}
    model_summaries = []
    for model_id, fit in (approved_models.items() if isinstance(approved_models, dict) else [(x, {}) for x in approved_models]):
        fit = fit if isinstance(fit, dict) else {}
        summary = {'model_id': model_id, 'schema': fit.get('schema'), 'family': fit.get('family'),
                   'usable_for_B': fit.get('usable_for_B'),
                   'support_mm': fit.get('common_range_mm', fit.get('common_linear_range_mm')),
                   'validation': _compact(fit.get('validation', {})),
                   'inverse_policy': fit.get('inverse_policy', 'existing correction policy'),
                   'fingers': [_compact({k: f.get(k) for k in ('finger', 'coordinate', 'direction_sign', 'slope_px_per_mm',
                                                              'intercept_px', 'linear_range_mm', 'zero_anchor')})
                               for f in fit.get('fingers', [])]}
        model_summaries.append(summary)
        references['model:' + str(model_id)] = summary
    recent = []
    for row in list(history or [])[-20:]:
        if isinstance(row, dict):
            recent.append(_compact({k: row.get(k) for k in ('id', 'round_index', 'cond', 'cells', 'model_id', 'policy',
                                                           'status', 'completed', 'reason', 'metrics', 'design',
                                                           'hypothesis', 'expected_outcome')}))
    # Candidate examples have no authority over this new planner. In particular,
    # a heuristic score never becomes a required condition for a valid proposal.
    examples = [_compact({k: c.get(k) for k in ('id', 'cond', 'cells', 'reason', 'model_id', 'policy', 'design_action')})
                for c in list(suggestions or [])[:12]]
    return {'schema': 'agentic_planning_context_v1', 'planning_mode': 'agent',
            'round_index': int(round_index), 'remaining_trials_including_monitors': int(remaining_trials),
            'design': _compact(design), 'scope': _compact(settings.get('scope', {})),
            'level_stats': levels, 'diagnostics': diagnostics, 'outcome_summary': failures,
            'quality': _compact(report.get('quality', {})), 'metrics': references['metrics'],
            'stop': references['stop'], 'approved_models': model_summaries,
            'references': references, 'valid_evidence_refs': sorted(references), 'history': recent,
            'optional_examples': examples,
            'constraints': {'hard_scope_mm': [settings['scope']['min_mm'], settings['scope']['max_mm']],
                            'min_distinct_spacing_mm': settings['scope'].get('min_step_mm', .5),
                            'max_one_sided_expansion_mm': min(5., settings['scope'].get('expansion_step_mm', 5.)),
                            'allow_expand': settings['scope'].get('allow_expand', False),
                            'batch_scan_limit': settings['budget']['batch_trials'],
                            'repetitions': 'even total per offset; split equally between the two bottles',
                            'allowed_gains': settings.get('policy', {}).get('gains', [1.]),
                            'monitor_every': cfg['design']['zero_every'],
                            'motion': _compact(cfg.get('motion', {})), 'axes_tcp': _compact(cfg.get('axes_tcp', {})),
                            'force_protocol': _compact(cfg.get('force', {})), 'gripper_limits': {
                                k: cfg.get('gripper', {}).get(k) for k in ('min_normalized', 'open_normalized')},
                            'B_truth_rule': 'The online B correction never receives experiment delta or TCP truth.',
                            'proposal_rule': 'Only closed plan JSON is accepted; code, commands and control changes are forbidden.',
                            'failed_samples': 'Informative evidence, not automatic prohibition. Physical execution constraints remain fixed.',
                            'model_domain': 'B may test outside learned support within current design; controller still refuses extrapolation.'}}


def _design(settings, design):
    scope = settings['scope']
    hard_lo, hard_hi = scope['min_mm'], scope['max_mm']
    if not all(_finite(x) for x in (hard_lo, hard_hi)) or not -25 <= hard_lo < hard_hi <= 25 or not hard_lo <= 0 <= hard_hi:
        _fail('hard_scope', '固定研究范围必须包含零点且位于 [-25,25] mm 内。')
    if design is None:
        design = {'min_mm': scope.get('initial_min_mm', hard_lo), 'max_mm': scope.get('initial_max_mm', hard_hi),
                  'step_mm': scope['step_mm']}
    if not isinstance(design, dict) or any(not _finite(design.get(k)) for k in ('min_mm', 'max_mm', 'step_mm')):
        _fail('design', '当前设计的边界必须有限，名义步长必须为正数。')
    lo, hi, step = (float(design[k]) for k in ('min_mm', 'max_mm', 'step_mm'))
    if not hard_lo <= lo < hi <= hard_hi or not lo <= 0 <= hi or step <= 0:
        _fail('design', '当前设计必须位于固定研究范围内并包含零点。')
    return lo, hi, step


def validate_plan_response(value, *, settings, cfg, design, approved_models, remaining_trials, context, round_index, T0=None):
    """Validate a freely proposed batch against explicit immutable constraints.

    Freeze eligibility and pause handling belong to the campaign's state machine.
    This function returns None for both, after validating their response shape.
    """
    response = validate_response(value)
    if not enabled(settings):
        _fail('planning_mode', '自由提案需要 planning.mode=agent；当前为旧候选选择模式。')
    if response['action'] != 'propose_batch':
        return None
    p = response['plan']; cond = p['cond']
    refs = context.get('references', {})
    valid_refs = set(refs) | set(context.get('valid_evidence_refs', []))
    if not set(p['evidence_refs']) <= valid_refs:
        _fail('unknown_evidence', '所有证据引用必须来自本轮提供的上下文。')
    if cond == 'A':
        if p['model_id'] is not None or p['gain'] != 1.:
            _fail('A_model_gain', '条件 A 必须使用 model_id=null、gain=1。')
    else:
        if p['model_id'] not in approved_models:
            _fail('unapproved_model', '条件 B 必须使用已批准的模型 ID。')
        if p['gain'] not in settings.get('policy', {}).get('gains', [1.]):
            _fail('unapproved_gain', '条件 B 的增益必须来自配置允许的列表。')
        if not .5 <= p['gain'] <= 1.:
            _fail('gain_bounds', '纠正增益必须位于 [0.5,1.0]。')
    left, right, old_step = _design(settings, design)
    scope = settings['scope']; minimum = scope.get('min_step_mm', .5)
    if not _finite(minimum) or minimum <= 0:
        _fail('minimum_spacing', '最小点间距必须为有限正数。')
    offsets = sorted(float(cell['offset_mm']) for cell in p['cells'])
    if len(set(offsets)) != len(offsets):
        _fail('duplicate_offsets', '同一偏移不能重复列出；请通过 reps 指定重复次数。')
    if any(b - a < minimum - 1e-9 for a, b in zip(offsets, offsets[1:])):
        _fail('spacing', '本批不同点位的间距小于 min_step_mm。')
    if any(not scope['min_mm'] <= d <= scope['max_mm'] for d in offsets):
        _fail('hard_scope', '提案中的偏移超出了固定研究范围。')
    if any(cell['reps'] < 2 or cell['reps'] % 2 for cell in p['cells']):
        _fail('balanced_repetitions', '每档重复数必须为正偶数，以便轻重瓶各执行一半。')
    if len(cfg['design']['bottles']) != 2 or len(set(cfg['design']['bottles'])) != 2:
        _fail('bottles', '执行协议要求两种不同的瓶子分块。')
    scan_count = sum(cell['reps'] for cell in p['cells'])
    if scan_count > settings['budget']['batch_trials']:
        _fail('scan_budget', '本批正式扫描次数超过 budget.batch_trials。')
    new_left, new_right = min(left, offsets[0]), max(right, offsets[-1])
    growth_left, growth_right = left - new_left, new_right - right
    if cond == 'B' and (growth_left > 1e-9 or growth_right > 1e-9):
        _fail('B_expansion', '只有条件 A 可以扩大当前探索范围。')
    if growth_left > 1e-9 or growth_right > 1e-9:
        if not scope.get('allow_expand', False):
            _fail('expansion_disabled', '本 campaign 的配置未允许扩大当前探索范围。')
        maximum = min(5., scope.get('expansion_step_mm', 5.))
        if growth_left > 1e-9 and growth_right > 1e-9:
            _fail('two_sided_expansion', '每批只能向当前范围的一侧扩展。')
        if max(growth_left, growth_right) > maximum + 1e-9:
            _fail('expansion_size', '本次扩展超过配置的单侧上限，且单次最多 5 mm。')
    batch_spacing = min(b - a for a, b in zip(offsets, offsets[1:])) if len(offsets) > 1 else old_step
    # A sparse later batch must not erase previous sampling detail or loosen
    # endpoint coverage checks, which use this stored design spacing.
    nominal = min(old_step, batch_spacing)
    nxt = {'min_mm': new_left, 'max_mm': new_right, 'step_mm': float(nominal)}
    candidate = {'origin': 'agent_proposal', 'proposal': response, 'cond': cond,
                 'cells': deepcopy(p['cells']), 'model_id': p['model_id'], 'policy': {'gain': float(p['gain'])},
                 'reason': response['reason'], 'hypothesis': p['hypothesis'],
                 'evidence_refs': deepcopy(p['evidence_refs']), 'expected_outcome': p['expected_outcome'],
                 'design_action': 'agent_plan', 'design_next': nxt, 'scope_mm': [new_left, new_right],
                 'round_index': int(round_index), 'trial_count': scan_count, 'scan_count': scan_count,
                 'score': 0., 'requires_prospective_evaluation': cond == 'B', 'warnings': []}
    # Lazy import avoids a module cycle when the campaign imports this validator.
    # The exact execution planner includes its actual monitor insertion policy.
    from .agentic import make_plan
    plan = make_plan(candidate, cfg, seed=int(cfg['design'].get('seed', 0)) + int(round_index))
    candidate['planned_total'] = candidate['plan_count'] = len(plan)
    candidate['monitor_count'] = sum(row['kind'] == 'zero_monitor' for row in plan)
    if type(remaining_trials) is not int or remaining_trials < 0 or len(plan) > remaining_trials:
        _fail('total_budget', '正式扫描加上自动插入的零点监测，超过剩余试验预算。')
    if cond == 'B':
        fit = approved_models[p['model_id']] if isinstance(approved_models, dict) else None
        if isinstance(fit, dict):
            if fit.get('usable_for_B') is False or fit.get('model_id', p['model_id']) != p['model_id']:
                _fail('unusable_model', '批准模型不可用于 B，或模型 ID 与批准记录不一致。')
            support = fit.get('common_range_mm', fit.get('common_linear_range_mm'))
            if isinstance(support, (list, tuple)) and len(support) == 2 and all(_finite(v) for v in support):
                if any(d < support[0] or d > support[1] for d in offsets):
                    candidate['warnings'].append('部分 B 初始点在模型支持范围外；现有纠正模块继续禁止外推。')
            else:
                candidate['warnings'].append('上下文中缺少模型支持范围；执行时仍检查模型。')
        else:
            candidate['warnings'].append('目前只有模型批准标识；执行时仍检查模型。')
    if T0 is not None:
        try:
            pose = np.asarray(T0, float)
        except (ValueError, TypeError):
            _fail('T0', 'T0 必须是有限且合法的 4×4 刚体变换。')
        if (pose.shape != (4, 4) or not np.isfinite(pose).all() or not np.allclose(pose[3], [0, 0, 0, 1]) or
                not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-4) or np.linalg.det(pose[:3, :3]) < .99):
            _fail('T0', 'T0 必须是有限且合法的 4×4 刚体变换。')
        from .e1e2_core import MotionBounds
        bounds = MotionBounds(cfg, pose)
        for offset in sorted(set(row['delta_mm'] for row in plan)):
            grasp = bounds.grasp(offset)
            checks = [('抓取', bounds.check(grasp)),
                      ('退让', bounds.check(bounds.pre(grasp), 'retreat', grasp)),
                      ('提起', bounds.check(bounds.lift(grasp), 'lift', grasp))]
            for role, error in checks:
                if error:
                    _fail('motion_bounds', f'偏移 {offset:+g} mm 的{role}目标违反固定限幅：{error}；整批拒绝，不裁剪点位。')
        candidate['motion_bounds_checked'] = True
    else:
        candidate['motion_bounds_checked'] = False
        candidate['warnings'].append('尚未记录 T0；执行器读取初始位姿后仍须检查固定运动限幅。')
    identity = {'proposal': response, 'round_index': int(round_index), 'design_next': nxt, 'planned_total': len(plan)}
    candidate['id'] = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()[:16]
    return candidate
