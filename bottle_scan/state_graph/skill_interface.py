"""Specification and instance plug-in interfaces for local observation skills.

Instances provide trial iteration, observation windows, measured action receipts,
post-condition evaluators and execution adapters. Framework models receive obs only.
"""
from copy import deepcopy
import importlib
import itertools
import json
from pathlib import Path
from typing import Protocol, TypedDict, Any
import numpy as np


class PostResult(TypedDict):
    evaluated: bool
    success: bool | None
    level: str | None
    evidence: list


class RuntimeAdapter(Protocol):
    def observe(self) -> dict: ...
    def execute(self, action: list[float]) -> dict: ...
    def emit(self, event: dict) -> None: ...


def load_callable(name):
    module,attribute=name.split(':',1)
    return getattr(importlib.import_module(module),attribute)


def validate_spec(value):
    spec=deepcopy(value)
    if spec.get('schema')!='skill_spec_v3': raise ValueError('Expected skill_spec_v3')
    space=spec['task']['action_space'];dim=space['dim']
    if not isinstance(dim,int) or dim<1 or len(space['names'])!=dim: raise ValueError('Action dimension mismatch')
    limits=np.asarray(spec['safety_envelope']['max_action'],float)
    if limits.shape!=(dim,) or not np.isfinite(limits).all() or np.any(limits<=0): raise ValueError('Invalid action envelope')
    modalities=spec['observation']['modalities']
    if not modalities: raise ValueError('No observation modalities')
    for name,m in modalities.items():
        if not name or float(m.get('weight',1))<=0: raise ValueError('Invalid modality weight')
    posts={p['id']:p for p in spec['post_conditions']}
    if len(posts)!=len(spec['post_conditions']) or spec['active_post_condition'] not in posts: raise ValueError('Invalid post conditions')
    for p in posts.values():
        if len(p.get('p_prior',[1,1]))!=2 or min(p.get('p_prior',[1,1]))<=0: raise ValueError('Invalid Beta prior')
        if not 0<p.get('p_min',.05)<=p.get('ready_p',.8)<1: raise ValueError('Invalid post probability thresholds')
    spec.setdefault('dynamics',{})
    spec.setdefault('models',{})
    spec.setdefault('graph',{})
    return spec


def load_spec(path): return validate_spec(json.loads(Path(path).read_text()))

def task_id(spec):
    return spec['task']['object_category']+'|'+spec['task']['action_space']['id']


def post_spec(spec,condition=None):
    condition=condition or spec['active_post_condition']
    return next(p for p in spec['post_conditions'] if p['id']==condition)


def action_grid(spec, observed=()):
    space=spec['task']['action_space'];limits=np.asarray(spec['safety_envelope']['max_action'],float)
    continuous = space.get('continuous_displacement_variables')
    if continuous:
        # The director chooses freely. Offline model queries reuse legal measured actions
        # instead of inventing a Cartesian product of fixed levels.
        bounds = np.asarray([spec['perturbation_variables'][k]['envelope'] for k in continuous])
        values = set()
        for edge in observed:
            receipt = edge.get('receipt') or {}
            if edge['kind'] != 'observed' or receipt.get('applied') is not True:
                continue
            u = np.asarray(receipt.get('net_contact_displacement_mm'), dtype=float)
            if (u.shape == limits.shape and np.count_nonzero(u) == 1 and np.isfinite(u).all()
                    and np.all(u >= bounds[:, 0]) and np.all(u <= bounds[:, 1])):
                values.add(tuple(u))
        return np.asarray(sorted(values), dtype=float).reshape(-1, space['dim'])
    values=space.get('candidate_actions')
    if values is None:
        levels=space.get('candidate_levels') or [[-float(x),0.,float(x)] for x in limits]
        values=list(itertools.product(*levels))
    actions=np.asarray(values,float)
    if actions.ndim!=2 or actions.shape[1]!=space['dim'] or not np.isfinite(actions).all(): raise ValueError('Invalid action grid')
    if np.any(abs(actions)>limits+1e-9): raise ValueError('Action grid exceeds specification')
    return actions


def model_observation(row,names):
    """Only this mapping enters model design matrices. Metadata is never flattened."""
    obs=row.get('obs')
    if not isinstance(obs,dict): raise ValueError('Missing observation mapping')
    result={}
    for name in names:
        x=np.asarray(obs.get(name),float)
        if x.ndim!=1 or not x.size or not np.isfinite(x).all(): raise ValueError('Missing/nonfinite observation: '+name)
        result[name]=x
    return result


def repeat_key(row,spec):
    if spec.get('noise',{}).get('relative_repeat_key'):
        from .repeat_evidence import relative_key
        widths=spec.get('repeat_resolution_mm') or {k:spec['facts']['boundary_resolution_mm'] for k in spec['task']['action_space']['continuous_displacement_variables']}
        return relative_key(row,spec,widths)
    settings=row.get('repeat_settings',{})
    variables=spec.get('perturbation_variables',{})
    active={name:settings.get(name) for name,v in variables.items() if v.get('active',True) and v['kind']!='object_property'}
    blocks={name:row.get('attrs',{}).get(name) for name,v in variables.items() if v['kind']=='object_property'}
    return json.dumps([row.get('provenance',{}).get('repeat_context',row['session']),row['task_id'],active,blocks],sort_keys=True,separators=(',',':'))


def normalize_action(action,spec):
    """Typed executable action. A sensing action is never a zero translation."""
    kind=spec['task']['action_space']['id']
    value={'kind':kind,'u':list(action)} if not isinstance(action,dict) else deepcopy(action)
    capabilities=spec.get('capabilities',{kind:{'kind':'vector','dim':spec['task']['action_space']['dim'],
                                                'max_action':spec['safety_envelope']['max_action']}})
    if value.get('kind') not in capabilities:raise ValueError('Undeclared action capability: '+str(value.get('kind')))
    cap=capabilities[value['kind']]
    if cap.get('kind','vector')=='vector':
        u=np.asarray(value.get('u'),float);limits=np.asarray(cap.get('max_action',spec['safety_envelope']['max_action']),float)
        if u.shape!=limits.shape or not np.isfinite(u).all() or np.any(abs(u)>limits+1e-9):raise ValueError('Action outside capability envelope')
        value['u']=u.tolist()
    else:
        params=value.get('parameters',{})
        if set(params)!=set(cap.get('parameters',{})):raise ValueError('Capability parameters differ from specification')
        for name,bounds in cap.get('parameters',{}).items():
            x=params[name]
            if type(x) not in (float,int) or not np.isfinite(x) or not bounds[0]<=x<=bounds[1]:raise ValueError('Capability parameter outside envelope')
    return value
