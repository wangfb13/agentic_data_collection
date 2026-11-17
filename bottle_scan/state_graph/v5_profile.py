"""Materialize a new graph profile. Never upgrade persisted campaigns in place."""
from copy import deepcopy
import json
from pathlib import Path

PROFILES=Path(__file__).with_name('profiles')


def merge(base,changes):
    result=deepcopy(base)
    for key,value in changes.items():
        if isinstance(value,dict) and isinstance(result.get(key),dict):
            result[key]=merge(result[key],value)
        else:
            result[key]=deepcopy(value)
    return result


def configure(spec,overrides=None,*,version=5):
    if version==4:
        if spec.get('graph_schema')=='state_graph_v5':
            raise ValueError('Select the original v4 spec for an exact legacy replay')
        return merge(spec,overrides or {})
    if version!=5:
        raise ValueError('Unknown graph version')
    defaults={}
    for path in sorted(PROFILES.glob('*.json')):
        defaults=merge(defaults,json.loads(path.read_text()))
    configured=(merge(defaults,spec) if spec.get('graph_schema')=='state_graph_v5' else merge(spec,defaults))
    configured['graph_schema']='state_graph_v5'
    from .v5_facts import configure as configure_facts
    result=merge(configured,overrides or {})
    configure_facts(result)
    return result


def protect_output(directory, *, kind):
    """Only reuse output explicitly created by this version and entry point."""
    directory=Path(directory)
    marker=directory/'v5_output.json'
    identity={'schema':'state_graph_v5','kind':kind}
    directory.mkdir(parents=True,exist_ok=True)
    if marker.exists():
        if json.loads(marker.read_text())!=identity:raise ValueError('Output belongs to a different operation')
    elif any(p.name!='campaign.lock' for p in directory.iterdir()):
        raise ValueError('Choose a new v5 directory; existing artifacts will be preserved')
    else:
        with marker.open('x') as handle:json.dump(identity,handle)
