"""Calibrate handoff evidence on independent terminal groups of the training set."""
from collections import defaultdict

import numpy as np
from scipy.stats import beta

from .common import group_id


def estimate(train, distances, condition, settings):
    cid = condition['id']
    groups = defaultdict(list)
    for i, row in enumerate(train):
        post = row.get('post', {}).get(cid, {})
        if row.get('is_terminal') and post.get('evaluated') and type(post.get('success')) is bool:
            groups[group_id(row)].append(i)
    members = list(groups.values())
    labels = np.array([all(train[i]['post'][cid]['success'] for i in ids) for ids in members], dtype=bool)
    count = distances.shape[0]
    d = np.column_stack([distances[:, ids].min(axis=1) for ids in members]) if members else np.empty((count, 0))
    a, b = condition.get('p_prior', [1, 1])
    minimum = condition.get('min_ready_trials', 3)
    eps = np.finfo(float).eps
    if settings['ready_kernel'] != 'gaussian':
        raise ValueError('Unknown Ready kernel')
    # One out-of-group vote per terminal trial. For repeated terminal windows,
    # score the closest evidence supporting release on a failed trial conservatively.
    bandwidths = settings['ready_bandwidths']
    if not bandwidths or any(not np.isfinite(h) or h <= 0 for h in bandwidths):
        raise ValueError('Ready bandwidths must be positive')
    if not 0 <= settings['max_false_release_rate'] <= 1 or not settings['ready_thresholds'] or any(not 0 <= t <= 1 for t in settings['ready_thresholds']):
        raise ValueError('Ready error cap and threshold candidates must be probabilities')
    best = None
    for h in bandwidths:
        weights = np.exp(-0.5 * np.square(d / h))
        cv = weights[:len(train)].copy()
        for j, ids in enumerate(members):
            cv[ids, j] = 0
        n = cv.sum(axis=1)
        s = cv @ labels.astype(float)
        means = (a + s) / (a + b + n)
        lowers = beta.ppf(settings['ready_quantile'], a + s, b + np.maximum(0, n - s))
        probabilities = [np.mean(means[ids]) for ids in members]
        ll = sum(np.log(np.clip(p if y else 1-p, eps, 1)) for p, y in zip(probabilities, labels))
        if best is None or ll > best[0]:
            best = ll, h, weights, lowers, n
    _, h, weights, lowers, cv_n = best
    scores = []
    for t in settings['ready_thresholds']:
        votes = np.array([np.all((lowers[ids] >= t) & (cv_n[ids] >= minimum)) if y else
                          np.any((lowers[ids] >= t) & (cv_n[ids] >= minimum))
                          for ids, y in zip(members, labels)], dtype=bool)
        false = int((votes & ~labels).sum())
        failures = int((~labels).sum())
        scores.append(dict(threshold=t, correct=int((votes & labels).sum()), false=false,
                           failures=failures, false_release_rate=false/failures if failures else None))
    feasible = [v for v in scores if v['correct'] and v['false_release_rate'] is not None
                and v['false_release_rate'] <= settings['max_false_release_rate']]
    chosen = max(feasible, key=lambda v: (v['correct'], -v['false'], v['threshold'])) if feasible else None
    cutoff = chosen['threshold'] if chosen else None
    meta = dict(h_ready=h, threshold=cutoff, candidates=scores, kernel=settings['ready_kernel'],
                training_groups=len(members), log_likelihood=best[0],
                terminal_groups=[dict(nodes=[train[i]['node_id'] for i in ids],success=bool(y)) for ids,y in zip(members,labels)])
    n = weights.sum(axis=1)
    s = weights @ labels.astype(float)
    lower = beta.ppf(settings['ready_quantile'], a + s, b + np.maximum(0, n - s))
    values = []
    for i in range(count):
        p = float((a+s[i])/(a+b+n[i]))
        values.append(dict(ready=bool(cutoff is not None and n[i]>=minimum and lower[i]>=cutoff),
                           p_hat=p, p_lower=float(lower[i]), n=float(n[i]), successes=float(s[i]),
                           direct_handoff_risk=float(-np.log(p)),
                           terminal_ids=[train[j]['node_id'] for k, ids in enumerate(members) if weights[i,k]>0 for j in ids]))
    return values, meta
