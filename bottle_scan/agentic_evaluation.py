"""Prospective B evidence for stopping and freezing; never controls hardware.

The evaluation cells are defined independently of an agent's next local batch.
Evidence from different fit versions or correction gains is never pooled.
"""
from __future__ import annotations

import math
from pathlib import Path


OFFSET_TOLERANCE_MM = 1e-8


def _finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(name + ' must be a positive integer')
    return value


def _mean(values):
    return sum(values) / len(values) if values else None


def prospective_B_evidence(report, batches, root, settings, design):
    """Evaluate one current-model gain over all predeclared bottle/offset cells.

    A complete but unsuccessful gain can be selected for a budget-triggered
    frozen evaluation; ``eligible`` remains false. When any gain passes the
    process gate, selection is restricted to passing gains. With no complete
    coverage, ``chosen_gain`` and ``success_rate`` are both None. These observed
    success rates are an exploration gate, not a confidence-bound acceptance.
    """
    # Lazy imports avoid a campaign/evaluation import cycle and any ROS startup.
    from .agentic import validation_offsets
    from .e1e2 import LABELS, rebuild_summary

    offsets = [float(value) for value in validation_offsets(settings, design)]
    if not offsets or not all(math.isfinite(value) for value in offsets):
        raise ValueError('evaluation offsets must be finite and nonempty')
    if any(math.isclose(a, b, abs_tol=OFFSET_TOLERANCE_MM, rel_tol=0.)
           for i, a in enumerate(offsets) for b in offsets[i + 1:]):
        raise ValueError('evaluation offsets are not distinguishable at the numeric matching tolerance')
    bottles = list(settings.get('validation', {}).get('bottles', ['light', 'heavy']))
    if len(bottles) != 2 or len(set(bottles)) != 2:
        raise ValueError('prospective evaluation requires two distinct bottle types')
    learning = settings.get('learning', {})
    minimum = _positive_integer(learning.get('min_B_per_cell', 1), 'learning.min_B_per_cell')
    minimum_batches = _positive_integer(learning.get('min_B_batches', 1), 'learning.min_B_batches')
    target = settings.get('validation', {}).get('success_target', .8)
    if not _finite_number(target) or not 0 <= target <= 1:
        raise ValueError('validation.success_target must be in [0, 1]')
    allowed = settings.get('policy', {}).get('gains', [.5, .75, 1.])
    allowed = sorted(set(float(gain) for gain in allowed if _finite_number(gain) and .5 <= gain <= 1.))
    best = report.get('best_model') or {}
    model_id = best.get('model_id')
    required = [{'offset_mm': offset, 'bottle': bottle, 'required_n': minimum}
                for offset in offsets for bottle in bottles]
    collected = {}
    seen_paths = set()
    excluded = {'unknown_gain_batches': 0, 'ineligible_rows': 0, 'outside_required_cells': 0}
    valid_labels = set(LABELS.values()) - {'invalid'}
    for batch in batches:
        candidate = batch.get('candidate') or {}
        if (not model_id or batch.get('phase') != 'explore' or batch.get('status') != 'complete'
                or candidate.get('cond') != 'B' or batch.get('model_id') != model_id):
            continue
        gain = (candidate.get('policy') or {}).get('gain')
        if not _finite_number(gain) or float(gain) not in allowed:
            excluded['unknown_gain_batches'] += 1
            continue
        if not batch.get('path'):
            continue
        path = (Path(root) / batch['path']).resolve()
        if path in seen_paths:
            continue
        seen_paths.add(path)
        gain = float(gain)
        group = collected.setdefault(gain, {'cells': {}, 'batches': []})
        contributed = False
        for row in rebuild_summary(path):
            if (row.get('cond') != 'B' or row.get('kind') != 'scan'
                    or not row.get('complete') or not row.get('recording_complete')
                    or row.get('label') not in valid_labels):
                excluded['ineligible_rows'] += 1
                continue
            delta = row.get('delta_mm')
            matched = next((offset for offset in offsets if _finite_number(delta)
                            and math.isclose(float(delta), offset,
                                             abs_tol=OFFSET_TOLERANCE_MM, rel_tol=0.)), None)
            if matched is None or row.get('bottle') not in bottles:
                excluded['outside_required_cells'] += 1
                continue
            group['cells'].setdefault((matched, row['bottle']), []).append(row)
            contributed = True
        if contributed:
            group['batches'].append(batch.get('id', str(path)))

    gains = []
    for gain, group in sorted(collected.items()):
        cells, missing = [], []
        for requirement in required:
            key = (requirement['offset_mm'], requirement['bottle'])
            rows = group['cells'].get(key, [])
            successes = sum(bool(row.get('success')) for row in rows)
            regrasps = [float(row['regrasps']) for row in rows
                        if _finite_number(row.get('regrasps')) and row['regrasps'] >= 0]
            cell = {**requirement, 'n': len(rows), 'successes': successes,
                    'success_rate': successes / len(rows) if rows else None,
                    'mean_regrasps': _mean(regrasps) if len(regrasps) == len(rows) else None}
            cells.append(cell)
            if len(rows) < minimum:
                missing.append({**requirement, 'n': len(rows), 'missing_n': minimum - len(rows)})
        coverage = not missing
        per_offset = []
        for offset in offsets:
            here = [cell for cell in cells if cell['offset_mm'] == offset]
            complete = all(cell['n'] >= minimum for cell in here)
            rate = _mean([cell['success_rate'] for cell in here]) if complete else None
            regrasp = (_mean([cell['mean_regrasps'] for cell in here])
                       if complete and all(cell['mean_regrasps'] is not None for cell in here) else None)
            per_offset.append({'offset_mm': offset, 'coverage': complete,
                               'n': sum(cell['n'] for cell in here),
                               'successes': sum(cell['successes'] for cell in here),
                               'success_rate': rate, 'mean_regrasps': regrasp,
                               'success_met': bool(complete and rate >= target), 'cells': here})
        success_met = coverage and all(item['success_met'] for item in per_offset)
        enough_batches = len(group['batches']) >= minimum_batches
        gains.append({'gain': gain, 'coverage': coverage, 'missing_cells': missing,
                      'cells': cells, 'per_offset': per_offset,
                      'n': sum(cell['n'] for cell in cells),
                      'successes': sum(cell['successes'] for cell in cells),
                      'success_rate': _mean([item['success_rate'] for item in per_offset]) if coverage else None,
                      'mean_regrasps': (_mean([item['mean_regrasps'] for item in per_offset])
                                        if coverage and all(item['mean_regrasps'] is not None for item in per_offset) else None),
                      'batch_ids': group['batches'], 'batch_count': len(group['batches']),
                      'success_met': bool(success_met), 'eligible': bool(success_met and enough_batches)})
    eligible = [item for item in gains if item['eligible']]
    choices = eligible or [item for item in gains if item['coverage']]
    chosen = max(choices, key=lambda item: (item['success_rate'],
                 -item['mean_regrasps'] if item['mean_regrasps'] is not None else -math.inf,
                 -item['gain'])) if choices else None
    return {'eligible': bool(chosen and chosen['eligible']),
            'chosen_gain': chosen['gain'] if chosen else None, 'model_id': model_id,
            'required_offsets_mm': offsets, 'required_cells': required,
            'min_B_per_cell': minimum, 'min_B_batches': minimum_batches,
            'success_target': target, 'gains': gains,
            'B_batches_on_current_model': sum(item['batch_count'] for item in gains),
            'success_rate': chosen['success_rate'] if chosen else None,
            'excluded': excluded, 'offset_match_tolerance_mm': OFFSET_TOLERANCE_MM,
            'criterion': 'Each required offset uses equal bottle weights; all offsets must reach the observed success target on one current-model gain. This is not statistical acceptance.'}
