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


def action_grid(spec):
    space=spec['task']['action_space'];limits=np.asarray(spec['safety_envelope']['max_action'],float)
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
    settings=row.get('repeat_settings',{})
    variables=spec.get('perturbation_variables',{})
    active={name:settings.get(name) for name,v in variables.items() if v.get('active',True) and v['kind']!='object_property'}
    blocks={name:row.get('attrs',{}).get(name) for name,v in variables.items() if v['kind']=='object_property'}
    return json.dumps([row['session'],row['task_id'],active,blocks],sort_keys=True,separators=(',',':'))
