"""Independent probes at a common relative command, retaining placement as evidence."""
from collections import defaultdict
from itertools import combinations
import json
import numpy as np
from .common import group_id
from .v5_facts import coordinate, resolution


def scales(rows, spec):
    return {v: resolution(rows,spec,v)['resolution_mm']
            for v in spec['task']['action_space']['continuous_displacement_variables']}


def relative_key(row, spec, widths):
    p=row.get('provenance',{})
    attrs=row.get('attrs',{})
    settings=row.get('repeat_settings') or {}
    if not p.get('placement_reference_id') or p.get('placement_reference_valid') is not True:
        return None
    mass=attrs.get('mass_class')
    if mass is None or p.get('placement_continuity') is False:
        return None
    quantized={}
    for axis,width in widths.items():
        measured,_=coordinate(row,axis)
        command=settings.get(axis)
        if measured is None:return None
        # An uncommanded free start is usable only at the measured witness origin.
        if command is None:
            if abs(measured)>width:return None
            command=0.
        if not np.isfinite(command):return None
        quantized[axis]=int(np.rint(command/width))
    setup={k:settings.get(k,attrs.get(k)) for k,d in spec['perturbation_variables'].items()
           if d.get('active',True) and k not in widths and d['kind']!='object_property'}
    condition=[p.get('reference_digest'),attrs.get('protocol_id'),
               p.get('sensor_version') or attrs.get('sensor_version'),row.get('extractor_versions')]
    return json.dumps([row['task_id'],quantized,setup,mass,condition],sort_keys=True,separators=(',',':'))


def indices(rows,spec,joint=True,widths=None):
    from .skill_interface import repeat_key
    relative=spec.get('noise',{}).get('relative_repeat_key')
    if widths is None:widths=scales(rows,spec) if relative else {}
    groups=defaultdict(dict)
    seen=set()
    for i,r in enumerate(rows):
        if r.get('window_index',0)!=0 or not r.get('window_valid',True):continue
        if relative:
            key=relative_key(r,spec,widths)
            if key is None:continue
        else:
            key=r.get('repeat_key') or repeat_key(r,spec)
            if joint:key=(key,str(r.get('provenance',{}).get('calibration_epoch',r.get('calibration_epoch'))))
        gid=group_id(r)
        if gid not in seen:
            groups[key][gid]=i
            seen.add(gid)
    return [list(d.values()) for d in groups.values() if len(d)>1]


def placement_audit(rows, groups, values):
    # Pair differences estimate single-observation covariance via E[ddᵀ]/2.
    audit={}
    for within,label in ((True,'within_placement'),(False,'across_placements')):
        moment=np.zeros((values.shape[1],values.shape[1]))
        pairs=0
        sources=set()
        for ids in groups:
            for i,j in combinations(ids,2):
                same=rows[i]['provenance']['placement_reference_id']==rows[j]['provenance']['placement_reference_id']
                if same!=within:continue
                d=values[i]-values[j]
                moment+=np.outer(d,d)
                pairs+=1
                sources.update((i,j))
        audit[label]=dict(pairs=pairs,independent_probes=len(sources),
                          single_covariance=(moment/(2*pairs)).tolist() if pairs else None)
    return audit
