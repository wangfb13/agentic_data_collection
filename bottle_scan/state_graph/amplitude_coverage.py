"""Coverage of measured consumer handoff coordinates, independent of recovery success."""
from collections import Counter, defaultdict
from math import ceil, isfinite

from .common import group_id, independent
from .v5_facts import _fact, continuous, coordinate, resolution


def terminal_samples(rows, spec, cid):
    eligible = [r for r in rows if not independent(r) and not r.get('post_freeze')
                and r.get('role') != 'independent' and not r.get('source_dev_evaluation_open')]
    first_move = {}
    for r in eligible:
        if (r.get('u_receipt') or {}).get('applied') and r.get('window_index') is not None:
            g = group_id(r)
            first_move[g] = min(first_move.get(g, r['window_index']), r['window_index'])
    direct_start = {group_id(r) for r in eligible if r.get('window_index')==0 and r.get('is_terminal')}
    seen = set()
    for r in eligible:
        post = r.get('post', {}).get(cid, {})
        if not r.get('is_terminal') or not post.get('evaluated') or type(post.get('success')) is not bool:
            continue
        if r['node_id'] in seen:
            continue
        seen.add(r['node_id'])
        group = group_id(r)
        if (r.get('u_receipt') or {}).get('applied') or r.get('previous_node_id') or (
                group in first_move and r.get('window_index', -1) >= first_move[group]):
            origin = 'after_correction'
        elif (r.get('controller_role') in ('direct', 'direct_baseline') or r.get('data_role') == 'frontier_direct'
              or r.get('provenance', {}).get('direct_A') is True):
            origin = 'without_correction'
        elif r.get('controller_role') in ('policy', 'expert') and group in direct_start:
            origin = 'without_correction'
        else:
            origin = 'unknown'
        coords = {v: x for v in continuous(spec) if (x := coordinate(r, v)[0]) is not None}
        condition = {k:r['attrs'][k] for k, d in spec['perturbation_variables'].items()
                     if d['kind'] == 'object_property' and r.get('attrs', {}).get(k) is not None}
        yield dict(node_id=r['node_id'], group=group, coords=coords, success=post['success'],
                   source=origin, condition=condition, cause=post.get('failure_cause') or post.get('reason'))


def bin_index(magnitude, width):
    # Same convention as H4. Zero and the first outer edge both belong to bin zero.
    return max(0, ceil(magnitude / width) - 1)


def coverage_facts(rows, graph, spec, cid):
    from .v5_operators import next_probe
    samples = list(terminal_samples(rows, spec, cid))
    axes = [v for v in continuous(spec) if spec['perturbation_variables'][v].get('active', True)]
    minimum = spec['facts'].get('min_bin_outcomes', 2)
    if type(minimum) is not int or minimum < 1:
        raise ValueError('min_bin_outcomes must be a positive count of independent trials')
    facts, covered_nodes = [], {}
    for var in axes:
        decl = spec['perturbation_variables'][var]
        origin = decl.get('reference', 0.)
        width = max(resolution(rows, spec, var)['resolution_mm'], decl.get('nominal_step', 0.))
        if not isfinite(width) or width <= 0:
            raise ValueError('Amplitude bin width must be finite and positive')
        covered_nodes[var] = set()
        for side, sign in (('positive', 1), ('negative', -1)):
            limit = sign * (decl['envelope'][int(sign > 0)] - origin)
            if limit <= 0:
                continue
            bins = [defaultdict(list) for _ in range(ceil(limit / width))]
            for s in samples:
                if var not in s['coords']:
                    continue
                d = sign * (s['coords'][var] - origin)
                if 0 <= d <= limit:
                    bins[bin_index(d, width)][s['group']].append(s['node_id'])
            enough = [len(groups) >= minimum for groups in bins]
            for groups, ok in zip(bins, enough):
                if ok:
                    covered_nodes[var].update(n for ids in groups.values() for n in ids)
            if all(enough):
                continue
            first = enough.index(False)
            end = next((i for i in range(first + 1, len(bins)) if enough[i]), len(bins))
            inner, outer = first * width, min(end * width, limit)
            interval = sorted([origin + sign * inner, origin + sign * outer])
            ids = [n for groups in bins for ns in groups.values() for n in ns]
            metrics = dict(var=var, consumer=cid, side=side, interval=interval,
                           reference_mm=origin, inner_mm=inner, outer_mm=outer,
                           covered_beyond=end < len(bins), bin_width_mm=width, resolution_mm=width,
                           min_bin_outcomes=minimum, group_counts=[len(g) for g in bins])
            metrics['target_mm'] = next_probe(metrics, spec, var, side)
            facts.append(_fact('amplitude_gap', [var, side, interval], ids, metrics))
    if not axes:
        return facts
    # Use a single observed joint coordinate. Combining per-axis maxima can
    # invent a joint starting point that has never been measured.
    covered = set.intersection(*(covered_nodes[v] for v in axes))
    probs = graph.get('probabilities', {}).get(cid, {})
    eligible = [s for s in samples if s['node_id'] in covered and
                probs.get(s['node_id'], {}).get('p_lower') is not None]
    if eligible:
        best = min(eligible, key=lambda s: (-probs[s['node_id']]['p_lower'], s['node_id']))
        facts.append(_fact('best_covered_ready_region', [cid], [best['node_id']],
                           dict(target_point={v:best['coords'][v] for v in axes},
                                p_lower=probs[best['node_id']]['p_lower'], condition=best['condition'])))
    return facts


def source_counts(samples):
    counts = Counter(s['source'] for s in samples)
    return {key: counts[key] for key in ('without_correction', 'after_correction', 'unknown')}
