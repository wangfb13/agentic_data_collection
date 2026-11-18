"""Observability at the scale of executed movements relative to repeat noise."""
from collections import defaultdict
from itertools import combinations
import numpy as np
from .common import group_id
from .skill_interface import repeat_key


def single_axis(edge, spec):
    u = np.asarray(edge.get('u'), float)
    if u.ndim != 1 or not np.isfinite(u).all() or not np.any(u) or not edge.get('executable',True):
        return None
    axis = int(np.argmax(abs(u)))
    residual = np.linalg.norm(np.delete(u,axis))
    cfg = spec['graph']
    name = (edge.get('receipt') or {}).get('commanded_axis')
    variables = spec['task']['action_space'].get('continuous_displacement_variables',[])
    if name is not None and variables and name != variables[axis].split('_')[0]:
        return None
    if residual > min(cfg['axis_residual_mm'], cfg['axis_residual_fraction']*abs(u[axis])):
        return None
    return axis


def axis_observable(rows, observed, D, spec, axis):
    index = {r['node_id']: i for i,r in enumerate(rows)}
    moves = [e for e in observed if single_axis(e,spec)==axis and
             np.isfinite(D[index[e['source']],index[e['target']]])]
    change = [D[index[e['source']], index[e['target']]] for e in moves]
    repeats = defaultdict(dict)
    for i,r in enumerate(rows):
        if r.get('window_index') != 0 or not r.get('window_valid',True):continue
        key = r.get('repeat_key') or repeat_key(r,spec)
        epoch = r.get('provenance',{}).get('calibration_epoch')
        repeats[key,epoch].setdefault(group_id(r), i)
    noise = [D[i,j] for ids in repeats.values() for i,j in combinations(ids.values(),2) if np.isfinite(D[i,j])]
    signal = float(np.median(change)) if change else None
    background = float(np.median(noise)) if noise else None
    known = len(change)>=spec['graph']['resolution_min_edges'] and bool(noise)
    visible = bool(signal>0 and signal>=background*spec['graph']['observable_ratio']) if known else None
    return dict(observable=visible,action_median=signal,repeat_median=background,n_actions=len(change),
                n_repeat_pairs=len(noise),reason=('insufficient_actions_or_repeats' if not known else
                'below_noise_ratio' if not visible else None), evidence=[[e['source'],e['target']] for e in moves])
