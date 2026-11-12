"""Supported contact-state targets, learned offline from explicit window labels.

The set is a union of small neighborhoods around observed acceptable contacts.
Its geometric anchors are measured A examples, not labels inferred from offset.
No convex hull, global mean target, runtime TCP or experimental offset is used.
"""
from copy import deepcopy
import hashlib
import json
import numpy as np

SCHEMA = 'acceptable_contact_set_v1'
DEFAULTS = {
    'min_anchor_groups': 2, 'min_anchor_trials': 2,
    'min_support_radius': .15, 'max_support_radius': .75,
    'support_radius_multiplier': 1.5, 'negative_margin': .45,
    'min_positive_coverage': .5, 'max_negative_acceptance': 0.,
}


def settings(whole):
    mode = whole.get('target_mode', 'reference')
    if mode not in ('reference', 'acceptable_set'):
        raise ValueError('target_mode must be reference or acceptable_set')
    cfg = {**DEFAULTS, **whole.get('acceptable_set', {})}
    if set(cfg) != set(DEFAULTS):
        raise ValueError('unknown acceptable_set settings')
    for key in ('min_anchor_groups', 'min_anchor_trials'):
        if type(cfg[key]) is not int or cfg[key] < 2:
            raise ValueError('acceptable_set.'+key+' requires >=2 independent examples')
    for key in set(cfg)-{'min_anchor_groups','min_anchor_trials'}:
        if type(cfg[key]) not in (int,float) or not np.isfinite(cfg[key]):
            raise ValueError('invalid acceptable_set.'+key)
    if (not 0 < cfg['min_support_radius'] <= cfg['max_support_radius'] or
            cfg['support_radius_multiplier'] < 1 or not 0 < cfg['negative_margin'] < .5 or
            not 0 < cfg['min_positive_coverage'] <= 1 or cfg['max_negative_acceptance'] != 0):
        raise ValueError('invalid acceptable_set support/validation thresholds')
    return mode, cfg


def _hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def _distance(a, b):
    return np.sqrt(np.mean((np.asarray(a)-np.asarray(b))**2,axis=-1))


def _id(artifact):
    return _hash({k:v for k,v in artifact.items() if k!='set_id'})[:20]


def _coverage(samples, artifact):
    # Each physical trial has equal weight despite repeated observation windows.
    grouped = {}
    for sample in samples:
        match = match_features(artifact, sample['features'])
        grouped.setdefault((sample['trial_path'],sample['label']), []).append(match['supported'])
    positives=[all(v) for (_,label),v in grouped.items() if label]
    negatives=[any(v) for (_,label),v in grouped.items() if not label]
    return {'positive_trials':len(positives),'negative_trials':len(negatives),
            'positive_coverage':float(np.mean(positives)) if positives else None,
            'negative_acceptance':float(np.mean(negatives)) if negatives else None}


def train(samples, head, recovery, whole, invocation_head=None):
    """Use head-training groups only; calibration/validation only audit coverage.

    Anchors also exclude geometry-model validation trials. This prevents its
    residual report from being repurposed as data to choose recovery endpoints.
    """
    _, cfg = settings(whole)
    artifact = {'schema':SCHEMA,'usable':False,'target_definition_id':whole['target_definition_id'],
        'settings':cfg,'anchors':[],'blocking_reasons':[], 'validation':{},
        'semantics':'union of observed positive contact neighborhoods; T0 is a coordinate origin only',
        'training_source':'explicit true first-touch A windows; no offset-derived labels',
        'action_semantics':'nearest measured acceptable anchor using tactile-only inferred a coordinate; reobserve to establish success',
        'caveats':['Neighborhood support is empirical, not a guarantee of task success.',
                   'Unobserved gaps are not accepted by interpolation or a convex hull.',
                   'Successful B windows update state heads; new recovery anchors need measured labelled A windows.']}
    if not head.get('scaling') or not recovery.get('split'):
        artifact['blocking_reasons'].append('state/geometry training split unavailable')
        artifact['set_id']=_id(artifact);return artifact
    artifact['scaling']=deepcopy(head['scaling'])
    groups=head['split']['groups']; train_groups=set(groups['train'])
    geometry_train={recovery['source_trials'][i]['trial_path'] for i in recovery['split']['training_indices']}
    selected=[s for s in samples if s['group'] in train_groups]
    mu=np.asarray(artifact['scaling']['mean']);scale=np.asarray(artifact['scaling']['scale'])
    normalized=lambda s:(np.asarray(s['features'])-mu)/scale
    positives=[s for s in selected if s['label'] is True]
    negatives=[s for s in selected if s['label'] is False]
    candidates=[s for s in positives if s.get('anchor_measurement',{}).get('source','').startswith('measured_tcp')
                and s['trial_path'] in geometry_train]
    zpositive=np.asarray([normalized(s) for s in positives])
    znegative=np.asarray([normalized(s) for s in negatives])
    # Keep real exemplars. Deduplicate identical feature/position anchors only,
    # without averaging distinct components or their measured coordinates.
    def confident(head,sample,positive):
        if not head or not head.get('weights'):return False
        z=(np.asarray(sample['features'])-head['scaling']['mean'])/head['scaling']['scale']
        score=float(1/(1+np.exp(-np.clip(np.r_[1.,z,z*z]@np.asarray(head['weights']),-40,40))))
        return score>=head['thresholds']['positive_min'] if positive else score<=head['thresholds']['negative_max']
    keys=set()
    for sample in candidates:
        if not confident(head,sample,True) or not confident(invocation_head,sample,False):continue
        z=normalized(sample)
        cross=[float(_distance(normalized(s),z)) for s in positives if s['group']!=sample['group']]
        if not cross or not len(znegative):continue
        radius=min(cfg['max_support_radius'],max(cfg['min_support_radius'],min(cross)*cfg['support_radius_multiplier']))
        # A training negative cannot be swallowed by a positive neighborhood.
        radius=min(radius,float(np.min(_distance(znegative,z)))*cfg['negative_margin'])
        if radius<=1e-12:continue
        local=[s for s,d in zip(positives,_distance(zpositive,z)) if d<=radius]
        support_groups=sorted({s['group'] for s in local})
        support_trials=sorted({s['trial_path'] for s in local})
        if len(support_groups)<cfg['min_anchor_groups'] or len(support_trials)<cfg['min_anchor_trials']:continue
        a=float(sample['anchor_measurement']['x_mm'])
        if not np.isfinite(a):continue
        key=tuple(np.round(np.r_[sample['features'],a],8))
        if key in keys:continue
        keys.add(key)
        source={k:sample[k] for k in ('session','trial_path','metadata_sha256','group','timestamp_ns')}
        source['measurement']=deepcopy(sample['anchor_measurement'])
        anchor={'features':list(sample['features']),'a_mm':a,'radius':radius,
                'source':source,'support_groups':support_groups,'support_trials':support_trials}
        anchor['id']=_hash(anchor)[:16];artifact['anchors'].append(anchor)
    if not artifact['anchors']:
        artifact['blocking_reasons'].append('no independently supported measured acceptable A anchors in training partitions')
    artifact['anchor_count']=len(artifact['anchors'])
    artifact['anchor_positions_mm']=sorted({a['a_mm'] for a in artifact['anchors']})
    for part in ('calibration','validation'):
        part_samples=[s for s in samples if s['group'] in set(groups[part])]
        result=_coverage(part_samples,artifact)
        artifact['validation'][part]=result
        if (not result['positive_trials'] or not result['negative_trials'] or
                result['positive_coverage']<cfg['min_positive_coverage'] or
                result['negative_acceptance']>cfg['max_negative_acceptance']):
            artifact['blocking_reasons'].append(part+' acceptable-set support coverage/separation failed')
        # Every anchor must have evidence of a positive neighborhood in held-out
        # groups; unused/uncovered target islands are not silently certified.
        uncovered=[]
        for anchor in artifact['anchors']:
            za=(np.asarray(anchor['features'])-mu)/scale
            if not any(s['label'] is True and float(_distance(normalized(s),za))<=anchor['radius'] for s in part_samples):
                uncovered.append(anchor['id'])
        result['uncovered_anchor_ids']=uncovered
        if uncovered:artifact['blocking_reasons'].append(part+' lacks observed positive support for some target anchors')
    artifact['usable']=not artifact['blocking_reasons'];artifact['set_id']=_id(artifact)
    return artifact


def validate(artifact, target_id):
    if (artifact.get('schema')!=SCHEMA or artifact.get('set_id')!=_id(artifact) or
            artifact.get('target_definition_id')!=target_id or artifact.get('usable') is not True or
            artifact.get('blocking_reasons') or not artifact.get('anchors')):
        raise ValueError('invalid or unsupported acceptable contact set')
    _,cfg=settings({'target_mode':'acceptable_set','acceptable_set':artifact.get('settings',{})})
    mu=np.asarray(artifact.get('scaling',{}).get('mean'),float)
    scale=np.asarray(artifact.get('scaling',{}).get('scale'),float)
    if mu.shape!=(4,) or scale.shape!=(4,) or not np.isfinite(mu).all() or not np.isfinite(scale).all() or np.any(scale<=0):
        raise ValueError('invalid acceptable contact set scaling')
    for anchor in artifact['anchors']:
        feature=np.asarray(anchor.get('features'),float)
        if (feature.shape!=(4,) or not np.isfinite(feature).all() or
                type(anchor.get('a_mm')) not in (int,float) or not np.isfinite(anchor['a_mm']) or
                type(anchor.get('radius')) not in (int,float) or not np.isfinite(anchor['radius']) or
                not 0<anchor['radius']<=cfg['max_support_radius'] or
                len(set(anchor.get('support_groups',[])))<cfg['min_anchor_groups'] or
                len(set(anchor.get('support_trials',[])))<cfg['min_anchor_trials'] or
                not anchor.get('source',{}).get('measurement',{}).get('source','').startswith('measured_tcp')):
            raise ValueError('invalid acceptable contact set anchor')


def match_features(artifact, features):
    anchors=artifact.get('anchors',[])
    result={'supported':False,'set_id':artifact.get('set_id'),'anchor_ids':[]}
    if not anchors:return result
    z=(np.asarray(features,float)-np.asarray(artifact['scaling']['mean']))/np.asarray(artifact['scaling']['scale'])
    distances=[float(_distance((np.asarray(a['features'])-artifact['scaling']['mean'])/artifact['scaling']['scale'],z)) for a in anchors]
    result.update(anchor_ids=[a['id'] for a,d in zip(anchors,distances) if d<=a['radius']],
                  nearest_distance=min(distances))
    result['supported']=bool(result['anchor_ids']);return result


def choose_action(artifact, prediction):
    """Reinterpret a geometry-only negative-a estimate as a set-directed action."""
    if prediction.get('valid') is not True:return deepcopy(prediction)
    value=prediction.get('action_mm')
    if type(value) not in (int,float) or not np.isfinite(value):
        return {'valid':False,'reason':'invalid_recovery_observation'}
    if not artifact.get('anchors'):
        return {'valid':False,'reason':'acceptable_set_has_no_recovery_anchor'}
    current=-float(value)
    anchor=min(artifact['anchors'],key=lambda a:(abs(a['a_mm']-current),a['id']))
    return {**deepcopy(prediction),'action_mm':float(anchor['a_mm']-current),
            'geometry_return_to_reference_mm':float(value),'estimated_a_mm':current,
            'target_a_mm':anchor['a_mm'],'target_anchor_id':anchor['id'],
            'target_set_id':artifact['set_id'],'reason':'nearest_supported_contact_target'}
