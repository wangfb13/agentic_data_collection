"""Read-only adaptive experimental resolution and bounded range proposals.

The hard scope remains immutable. A design is only the currently explored span
and nominal sampling interval; nominal interval is never claimed as achieved
TCP resolution. No hardware imports, publishing or motion occur here.
"""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def enabled(settings):
    return settings.get('scope', {}).get('adaptive', False) is True


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _limits(settings):
    scope = settings['scope']
    lo, hi, step = scope['min_mm'], scope['max_mm'], scope['step_mm']
    if not all(_number(v) for v in (lo, hi, step)) or not -25 <= lo < hi <= 25 or not lo <= 0 <= hi or step <= 0:
        raise ValueError('Adaptive design requires finite hard scope within [-25,25], including zero, and positive step')
    minimum = scope.get('min_step_mm', .5)
    expansion = scope.get('expansion_step_mm', min(step, 5.))
    if not all(_number(v) and v > 0 for v in (minimum, expansion)):
        raise ValueError('min_step_mm and expansion_step_mm must be finite and positive')
    if expansion > 5.:
        raise ValueError('Each adaptive expansion is limited to 5 mm')
    return float(lo), float(hi), float(step), float(minimum), float(expansion)


def _levels(report):
    return {float(s['offset_mm']): s for s in report.get('evidence', {}).get('level_stats', [])
            if _number(s.get('offset_mm')) and s.get('n', 0) > 0}


def initialize(settings, report):
    """Initialize from valid measured A levels, otherwise the declared initial span."""
    lo, hi, step, minimum, _ = _limits(settings)
    scope = settings['scope']
    stats = _levels(report)
    observations = report.get('evidence', {}).get('observations', [])
    if observations:
        clean_levels = {float(o['planned_delta_mm']) for o in observations
                        if _number(o.get('planned_delta_mm')) and
                        o.get('measurement', {}).get('source', '').startswith('measured_tcp') and
                        len(o.get('descriptors', [])) == 2 and
                        all(d.get('valid') and not d.get('touches_edge') for d in o['descriptors'])}
        stats = {x: s for x, s in stats.items() if x in clean_levels}
    levels = sorted(x for x, s in stats.items() if lo <= x <= hi and s.get('measured_n', 0) > 0)
    if not levels:
        levels = sorted(set(float(o['planned_delta_mm']) for o in report.get('evidence', {}).get('observations', [])
                            if _number(o.get('planned_delta_mm')) and lo <= o['planned_delta_mm'] <= hi and
                            o.get('measurement', {}).get('source', '').startswith('measured_tcp') and
                            all(d.get('valid') and not d.get('touches_edge') for d in o.get('descriptors', [])) and len(o.get('descriptors', [])) == 2))
    if len(levels) >= 2:
        left, right = min(0., levels[0]), max(0., levels[-1])
        spacings = np.diff(levels)
        spacing = float(np.median(spacings[spacings > 1e-9]))
        step = max(minimum, spacing)
        origin = 'measured_A_seed_levels'
    else:
        left, right = scope.get('initial_min_mm', lo), scope.get('initial_max_mm', hi)
        if not _number(left) or not _number(right):
            raise ValueError('Initial scope must be finite')
        left, right = max(lo, min(0., float(left))), min(hi, max(0., float(right)))
        if left >= right:
            raise ValueError('Initial span must have positive width')
        step = max(minimum, step)
        origin = 'configured_initial_span'
    return {'min_mm': float(max(lo, left)), 'max_mm': float(min(hi, right)),
            'step_mm': step, 'origin': origin}


def _validate_design(settings, design):
    lo, hi, _, minimum, expansion = _limits(settings)
    values = [design.get(k) for k in ('min_mm', 'max_mm', 'step_mm')]
    if not all(_number(v) for v in values):
        raise ValueError('Design bounds/step must be finite')
    left, right, step = map(float, values)
    if not lo <= left < right <= hi or not left <= 0 <= right or step < minimum - 1e-9:
        raise ValueError('Design exceeds hard scope or resolution setting')
    return left, right, step, minimum, expansion


def effective_settings(settings, design):
    """Return current active scope without allowing old initial limits to clip it."""
    left, right, step, _, _ = _validate_design(settings, design)
    result = deepcopy(settings)
    result['scope'].update(min_mm=left, max_mm=right, step_mm=step,
                           initial_min_mm=left, initial_max_mm=right)
    return result


def _load_outcomes(report, settings):
    """Read outcome JSON solely for deciding whether exploration may expand.

    Force/no-contact/height failures stay visible even though mapping training
    correctly excludes them. Missing data or operator-invalid trials never count
    as positive evidence authorizing expansion.
    """
    result, seen = [], set()
    for s in report.get('evidence', {}).get('sessions', []):
        raw_path = s.get('path') if isinstance(s, dict) else s
        if not raw_path:
            continue
        root = Path(raw_path).expanduser().resolve()
        if str(root) in seen:
            continue
        seen.add(str(root))
        try:
            meta = json.loads((root / 'session_meta.json').read_text())
        except (OSError, ValueError):
            continue
        role = meta.get('agentic', {}).get('role', meta.get('agentic_role', 'exploration'))
        if role in ('validation', 'frozen_validation', 'final_validation') or bool(meta.get('simulated', False)) != bool(settings.get('simulate', False)):
            continue
        for path in sorted((root / 'trials').glob('*/meta.json')):
            try:
                row = json.loads(path.read_text())
                if row.get('cond') != 'A' or not _number(row.get('delta_mm')):
                    continue
                status_path = path.parent / 'recording_status.json'
                recording = bool(row.get('recording_complete') and status_path.exists() and
                                 json.loads(status_path.read_text()).get('complete'))
            except (OSError, ValueError):
                continue
            probe = row.get('probe_initial', {})
            ds = probe.get('descriptors', [])
            valid = bool(probe.get('valid') and len(ds) == 2 and all(d.get('valid') and not d.get('touches_edge') for d in ds))
            blocked = bool(row.get('status') == 'skipped_bound' or row.get('skipped_reason') or
                           any(h.get('blocked_reason') for h in row.get('correction_history', [])))
            passed = bool(row.get('complete') and recording and row.get('label') == 'normal' and
                          valid and row.get('bilateral') and row.get('target_reached') and row.get('lifted') and
                          row.get('hold_complete') and not row.get('dropped') and not blocked)
            # An operator-invalid sample or a missing recording says nothing
            # about the physical contact boundary. Explicit, completely recorded
            # contact/force failures remain evidence even if labelled invalid.
            contact_force_failure = bool(row.get('complete') and recording and not blocked and
                row.get('status') in ('no_contact', 'stall_no_contact', 'force_not_reached',
                                      'probe_contact_failed', 'probe_unstable'))
            result.append({'offset_mm': float(row['delta_mm']), 'session': str(root), 'id': row.get('id'),
                           'bottle': row.get('bottle'), 'passed': passed, 'valid_probe': valid,
                           'complete': bool(row.get('complete')), 'blocked': blocked,
                           'contact_force_failure': contact_force_failure,
                           'reason': row.get('skipped_reason') or row.get('status'),
                           'recording': recording})
    return result


def _noise(stats):
    return max([float(f.get('sd_equivalent_mm') or 0.) for f in stats.get('fingers', [])] + [0.])


def _curvature(report):
    """Dimensionless local slope changes, never a claim of precision."""
    by = defaultdict(list)
    model = report.get('best_model') or {}
    coord = (model.get('fingers') or [{'coordinate': 'u'}])[0].get('coordinate', 'u')
    idx = 0 if coord == 'u' else 1
    for o in report.get('evidence', {}).get('observations', []):
        try:
            if o['descriptors'][0].get('valid') and not o['descriptors'][0].get('touches_edge'):
                by[float(o['planned_delta_mm'])].append(float(o['descriptors'][0]['centroid_px'][idx]))
        except (KeyError, TypeError, IndexError, ValueError):
            continue
    x = sorted(by)
    scores = {}
    for a, b, c in zip(x, x[1:], x[2:]):
        left = (float(np.median(by[b])) - float(np.median(by[a]))) / (b - a)
        right = (float(np.median(by[c])) - float(np.median(by[b]))) / (c - b)
        value = abs(left - right) / max(abs(left), abs(right), .05)
        scores[(a, b)] = max(scores.get((a, b), 0.), value)
        scores[(b, c)] = max(scores.get((b, c), 0.), value)
    return scores


def propose_adaptive(report, settings, design, remaining_trials, round_index):
    """Bounded A-only refinement/expansion alternatives with stable audit IDs."""
    if not enabled(settings) or report.get('quality', {}).get('blocking_reasons'):
        return []
    left, right, step, minimum, expansion = _validate_design(settings, design)
    hard_left, hard_right, _, _, _ = _limits(settings)
    scope = settings['scope']; learning = settings.get('learning', {})
    batch = min(int(remaining_trials), int(settings.get('budget', {}).get('batch_trials', 12)))
    batch -= batch % 2
    if batch < 2:
        return []
    stats = _levels(report)
    outcomes = _load_outcomes(report, settings)
    by = defaultdict(list)
    for o in outcomes:
        by[o['offset_mm']].append(o)
    levels = sorted(set([x for x in stats if left <= x <= right] +
                        [x for x in by if left <= x <= right]))
    observed = {x: s for x, s in stats.items() if left <= x <= right}
    target_error = float(learning.get('target_mae_mm', 1.))
    curvature = _curvature(report)
    refined = max(minimum, step / 2.)
    max_repeats = int(scope.get('noise_repeat_cap', 12))
    boundary_min = int(scope.get('boundary_min_trials', max(2, learning.get('min_per_level', 2))))
    choices = []

    def add(action, points, next_design, reason, score, details):
        unique = list(dict.fromkeys(round(float(x), 8) for x in points))
        unique = [x for x in unique if hard_left <= x <= hard_right]
        if not unique:
            return
        unique = unique[:batch // 2]
        cells = [{'offset_mm': x, 'reps': 2} for x in unique]
        # Test new positions with a small balanced sample, then learn before
        # spending more. Only explicit noise repeats consume extra repetitions.
        spare = batch - 2 * len(cells)
        i = 0
        while spare >= 2 and cells and details.get('mode') == 'repeat_before_refine':
            available = [cell for cell in cells if details.get('mode') != 'repeat_before_refine' or
                         stats.get(cell['offset_mm'], {}).get('n', 0) + cell['reps'] + 2 <= max_repeats]
            if not available:
                break
            available[i % len(available)]['reps'] += 2; spare -= 2; i += 1
        candidate = {'cond': 'A', 'cells': cells, 'trial_count': sum(c['reps'] for c in cells),
                     'model_id': (report.get('best_model') or {}).get('model_id'), 'policy': {'gain': 1.},
                     'round_index': int(round_index), 'reason': reason, 'score': float(score),
                     'design_action': action, 'design_next': next_design,
                     'scope_mm': [next_design['min_mm'], next_design['max_mm']],
                     'unresolved': True, 'requires_resolution': True,
                     'evidence': details, 'requires_prospective_evaluation': False}
        candidate['id'] = hashlib.sha256(json.dumps(candidate, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()[:16]
        choices.append(candidate)

    intervals, repeats = [], {}
    for a, b in zip(levels, levels[1:]):
        gap = b - a
        if gap < 2 * minimum - 1e-9:
            continue
        sa, sb = stats.get(a, {}), stats.get(b, {})
        oa, ob = by.get(a, []), by.get(b, [])
        blocked_a, blocked_b = any(x['blocked'] for x in oa), any(x['blocked'] for x in ob)
        # A height/workspace skip is not a contact boundary. Do not bisect it as
        # if sensor data could remove the immutable movement constraint.
        if blocked_a or blocked_b:
            continue
        pass_a = bool(oa and all(x['passed'] for x in oa))
        pass_b = bool(ob and all(x['passed'] for x in ob))
        failed_a = bool(oa and any(x['contact_force_failure'] for x in oa))
        failed_b = bool(ob and any(x['contact_force_failure'] for x in ob))
        transition = (pass_a and failed_b) or (pass_b and failed_a)
        bias = max(abs(sa.get('cv_bias_mm') or 0.), abs(sb.get('cv_bias_mm') or 0.))
        cv = max(sa.get('cv_mae_mm') or 0., sb.get('cv_mae_mm') or 0.)
        curved = curvature.get((a, b), 0.)
        feedback = 0.
        for d in report.get('diagnostics', []):
            if d.get('code') == 'B_offset_feedback' and a - minimum <= d['offset_mm'] <= b + minimum:
                nonbound = max(0, d.get('failures', 0) - d.get('execution_blocked', 0)) / max(1, d.get('n', 0))
                exhausted = max(0, d.get('servo_exhausted', 0) - d.get('execution_blocked', 0)) / max(1, d.get('n', 0))
                excess_error = max(0., abs(d.get('initial_mae_mm') or 0.) - target_error)
                feedback = max(feedback, nonbound * 3 + exhausted * 2 + min(3., excess_error))
        need = transition or bias > target_error or cv > target_error or curved > float(scope.get('curvature_threshold', .2)) or feedback > 0 or not report.get('best_model')
        if not need:
            continue
        score = (8. if transition else 0.) + bias + cv + curved * 3. + feedback + gap / max(step, minimum)
        midpoint = (a + b) / 2.
        if any(abs(midpoint - x) < minimum - 1e-9 for x in levels):
            continue
        actual_gap = abs(float(sb.get('actual_mean_mm', b)) - float(sa.get('actual_mean_mm', a)))
        actual_sd = max(float(sa.get('actual_sd_mm') or 0.), float(sb.get('actual_sd_mm') or 0.))
        noise = max(_noise(sa), _noise(sb))
        n = min(max(1, sa.get('n', 0)), max(1, sb.get('n', 0)))
        # Frame counts are irrelevant here: physical repeat spread and achieved
        # TCP separation decide whether halving nominal spacing has information.
        position_unresolved = actual_sd * 2 >= gap / 2. or actual_gap < max(gap / 2., actual_sd * 2.)
        noise_unresolved = noise > target_error and noise / math.sqrt(n) > target_error / 2.
        if position_unresolved or (not transition and noise_unresolved):
            for x, s in ((a, sa), (b, sb)):
                if s.get('n', 0) + 2 <= max_repeats:
                    repeats[x] = max(repeats.get(x, 0.), score)
            continue
        intervals.append((score, midpoint, {'interval_mm': [a, b], 'contact_transition': transition,
                                           'cv_bias_mm': bias, 'curvature_score': curved, 'B_feedback_score': feedback,
                                           'actual_endpoint_separation_mm': actual_gap, 'actual_sd_mm': actual_sd,
                                           'nominal_spacing_not_resolution': True}))
    if repeats:
        ordered = sorted(repeats, key=lambda x: (-repeats[x], x))
        add('repeat', ordered, {'min_mm': left, 'max_mm': right, 'step_mm': step},
            'Repeat noisy or poorly separated measured positions before reducing nominal spacing; preserve pose tolerance and scope',
            max(repeats.values()) + 2., {'mode': 'repeat_before_refine', 'physical_repeat_cap': max_repeats,
                                      'nominal_spacing_not_resolution': True})
    if intervals:
        intervals.sort(key=lambda v: (-v[0], v[1]))
        add('refine' if refined < step - 1e-9 else 'sample', [x[1] for x in intervals],
            {'min_mm': left, 'max_mm': right, 'step_mm': refined},
            'Add local midpoints where held-out error, curvature, B feedback or a measured contact transition remains unresolved; preserve failed boundary',
            max(x[0] for x in intervals), {'mode': 'midpoint', 'intervals': [x[2] for x in intervals[:batch // 2]],
                                         'nominal_spacing_not_resolution': True})

    # Separate one-sided options. Historical no-contact/force-limit/height
    # failures veto extrapolation; a fitted line alone is never positive proof.
    if scope.get('allow_expand', False):
        for direction, action, boundary, hard in [(-1, 'expand_left', left, hard_left), (1, 'expand_right', right, hard_right)]:
            if abs(boundary - hard) < 1e-9:
                continue
            s = stats.get(boundary, {})
            evidence = by.get(boundary, [])
            measured_n = int(s.get('measured_n', 0))
            if measured_n < boundary_min or len(evidence) < boundary_min or not all(o['passed'] for o in evidence):
                continue
            if len({o['bottle'] for o in evidence if o['passed']}) < 2:
                continue
            target = round(max(hard, boundary - expansion) if direction < 0 else min(hard, boundary + expansion), 8)
            known = [o for o in outcomes if direction * (o['offset_mm'] - boundary) > 0 and direction * (target - o['offset_mm']) >= -1e-9 and not o['passed']]
            if known:
                continue
            # Do not proceed wider while endpoint TCP/descriptor noise itself
            # is unresolved; a repeat proposal remains the appropriate next test.
            noise = _noise(s)
            if noise > target_error and noise / math.sqrt(max(1, s.get('n', 0))) > target_error / 2.:
                continue
            next_design = {'min_mm': target if direction < 0 else left,
                           'max_mm': target if direction > 0 else right, 'step_mm': step}
            add(action, [target, boundary], next_design,
                'Extend one side within immutable hard scope after measured, bilateral, target-force and successful A evidence; repeat old boundary as control',
                5. + abs(target - boundary) / max(step, minimum),
                {'boundary_mm': boundary, 'measured_trials': measured_n, 'successful_trials': len(evidence),
                 'max_expansion_mm': expansion, 'unexplored_position_is_not_assumed_successful': True})
    return sorted(choices, key=lambda c: (-c['score'], c['id']))
