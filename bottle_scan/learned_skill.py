"""Versioned a-axis skill learning. Pure offline NumPy; never opens hardware.

Invocation and completion are learned from independent contact-state annotations.
The action learner reuses measured geometry; acceptable_set mode directs it
toward labelled supported contact targets instead of the reference origin. Safety and
temporal guards remain the responsibility of RecoverySkill and its adapter.
"""
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np

from .agentic_learning import _load, _compatibility, _hash
from .contact_policy import train_contact_policy, _digest as _recovery_digest

SCHEMA = 'learned_a_skill_v1'
SET_SCHEMA = 'learned_a_skill_set_v1'
FINAL_ROLES = {'validation', 'independent', 'final_validation', 'frozen_validation'}
LABEL_SOURCE = 'operator_review_of_observed_contact'
DEFAULTS = {
    'seed': 20261109, 'train_fraction': .6, 'calibration_fraction': .2,
    'min_train_per_class': 4, 'min_calibration_per_class': 2,
    'min_validation_per_class': 2, 'epochs': 800, 'learning_rate': .05,
    'l2': .02, 'min_coverage': .5, 'max_false_positive_rate': .1,
    'max_false_negative_rate': .2, 'ood_min_radius': .5,
    'ood_radius_multiplier': 1.5, 'score_margin': .02,
}


def _digest(model):
    return _hash({k: v for k, v in model.items() if k not in ('integrity_sha256', 'model_id')})


def _seal(model):
    model['integrity_sha256'] = _digest(model)
    model['model_id'] = model['integrity_sha256'][:20]
    return model


def _settings(settings):
    whole = deepcopy(settings or {})
    target = whole.get('target_definition_id')
    if not isinstance(target, str) or not target.strip():
        raise ValueError('target_definition_id is required; define the contact target before learning')
    from .acceptable_contact_set import settings as target_settings
    target_settings(whole)
    cfg = {**DEFAULTS, **whole.get('state_learning', {})}
    if set(cfg) - set(DEFAULTS):
        raise ValueError('unknown state_learning settings: ' + str(sorted(set(cfg)-set(DEFAULTS))))
    for k in ('seed', 'min_train_per_class', 'min_calibration_per_class', 'min_validation_per_class', 'epochs'):
        if type(cfg[k]) is not int or cfg[k] < (0 if k == 'seed' else 1):
            raise ValueError('invalid state_learning.' + k)
    if cfg['epochs'] > 10000:
        raise ValueError('state_learning epochs exceeds offline budget')
    for k in set(cfg) - {'seed','min_train_per_class','min_calibration_per_class','min_validation_per_class','epochs'}:
        if type(cfg[k]) not in (int,float) or not np.isfinite(cfg[k]) or cfg[k] <= 0:
            raise ValueError('invalid state_learning.' + k)
    if not .4 <= cfg['train_fraction'] <= .8 or not .1 <= cfg['calibration_fraction'] <= .3 or cfg['train_fraction'] + cfg['calibration_fraction'] > .9:
        raise ValueError('train/calibration fractions must leave at least 10% validation groups')
    if not 0 < cfg['min_coverage'] <= 1 or not 0 < cfg['max_false_positive_rate'] < 1 or not 0 < cfg['max_false_negative_rate'] < 1 or cfg['score_margin'] >= .25:
        raise ValueError('invalid state score/validation thresholds')
    return whole, cfg


def _features(observation):
    """Only four tactile centroid coordinates; extras cannot leak experiment truth."""
    ds = observation.get('descriptors') or []
    if len(ds) != 2 or any(not isinstance(d, dict) or d.get('valid') is not True for d in ds):
        raise ValueError('two valid tactile descriptors required')
    a = np.asarray([d.get('centroid_px') for d in ds], dtype=float)
    if a.shape != (2,2) or not np.isfinite(a).all():
        raise ValueError('finite two-finger centroids required')
    return a.reshape(4)


def _events(row):
    result = {}
    for event in row.get('skill_observations') or []:
        o = event.get('observation') if isinstance(event, dict) else None
        if not isinstance(o, dict) or type(o.get('timestamp_ns')) is not int:
            continue
        ts = o['timestamp_ns']
        try:
            feature = _features(o).tolist()
        except (ValueError, TypeError):
            continue
        if ts in result and result[ts]['features'] != feature:
            raise ValueError('conflicting observations share a timestamp: ' + row['trial_path'])
        result[ts] = {'features': feature, 'observation': o}
    return result


def _group(row):
    # Repeated windows and offset scans within one physical repeat block stay together.
    repeat = row.get('repeat')
    if repeat is None:
        return row.get('canonical_trial_path',row['trial_path'])
    return f"{row.get('canonical_session',row['session'])}|{row.get('bottle','unknown')}|{repeat}"


def _collect(roots, whole):
    sessions, rows, issues = _load(roots, whole)
    sessions = [s for s in sessions if s['role'] not in FINAL_ROLES]
    compatible, compatibility = _compatibility(sessions, whole)
    issues += compatible
    # Legacy compatibility omits opening settings. For learned state boundaries
    # these settings must be explicitly checked rather than silently pooled.
    gripper_keys = ('min_normalized','open_normalized','close_speed_normalized_s',
                    'fast_close_until_normalized','fast_close_speed_normalized_s')
    protocols = {_hash({k:s['meta'].get('config',{}).get('gripper',{}).get(k) for k in gripper_keys}) for s in sessions}
    if len(protocols) > 1:
        issues.append('gripper contact protocol differs; changed closing floor needs a separate evidence set')
    target = whole['target_definition_id']
    samples = {'invocation': [], 'exit': []}
    excluded, legacy_initial, legacy_final = Counter(), Counter(), Counter()
    source_trials = []
    for row in rows:
        if row['session_role'] in FINAL_ROLES:
            excluded['independent_evidence_never_trains'] += 1
            continue
        if row['session_role'] not in ('exploration','development'):
            issues.append('unsupported training source role: ' + row['session_role'])
            continue
        if row.get('cond') == 'B' and row['session_role'] != 'development':
            excluded['B_requires_development_role'] += 1
            continue
        if row.get('cond') not in ('A','B') or not row.get('complete') or not row.get('recording_verified') or row.get('archived_attempt'):
            excluded['incomplete_or_ineligible_trial'] += 1
            continue
        if row.get('label') in ('invalid','pushed','tipped','tilted'):
            excluded['invalid_or_object_moved'] += 1
            continue
        gt = row.get('skill_ground_truth') or {}
        legacy_initial[str(gt.get('initial_class'))] += 1
        legacy_final[str(gt.get('final_target_reached'))] += 1
        events = _events(row)
        source = {k:row[k] for k in ('session','trial_path','metadata_sha256')}
        source['group'] = _group(row)
        source_trials.append(source)
        def add(head, ts, label):
            sample={**source, 'timestamp_ns':ts, 'label':label, 'features':events[ts]['features']}
            if (whole.get('target_mode')=='acceptable_set' and head=='exit' and label is True
                    and row.get('cond')=='A' and ts==first and sessions):
                # Position is a training-only coordinate attached to an explicit
                # positive observation; it never determines the contact label.
                from .agentic_learning import _measured_probe
                try:
                    same=np.array_equal(_features(row.get('probe_initial',{})),events[ts]['features'])
                except (ValueError,TypeError):same=False
                if same:sample['anchor_measurement']=_measured_probe(row,sessions[0]['meta'],whole)
            samples[head].append(sample)
        explicit_id = gt.get('target_definition_id')
        first = None
        if explicit_id is not None and explicit_id != target:
            issues.append('mixed target definitions: ' + row['trial_path'])
        if (explicit_id == target and gt.get('source') == LABEL_SOURCE and
                gt.get('independent_of_experimental_offset') is True):
            label = {'already_normal':False,'needs_recovery':True}.get(gt.get('initial_class'))
            if label is not None and events:
                # Explicit first-contact observation in recording order; not a later success.
                first = next((e.get('observation',{}).get('timestamp_ns') for e in row.get('skill_observations',[])
                              if e.get('phase') == 'first_touch' and e.get('observation',{}).get('timestamp_ns') in events),None)
                if first is not None:
                    add('invocation',first,label)
                else:
                    excluded['initial_label_without_first_touch_window'] += 1
            else:
                excluded['initial_unknown_or_not_applicable'] += 1
        else:
            excluded['initial_label_missing_target_definition_or_provenance'] += 1
        seen_labels = {}
        for label in row.get('skill_window_labels') or []:
            if not isinstance(label,dict):
                excluded['invalid_window_annotation'] += 1
                continue
            if label.get('target_definition_id') not in (None,target):
                issues.append('mixed target definitions in window labels: ' + row['trial_path'])
                continue
            ts, value = label.get('timestamp_ns'),label.get('target_reached')
            if (label.get('source') != LABEL_SOURCE or label.get('target_definition_id') != target or
                    label.get('independent_of_experimental_offset') is not True or type(value) is not bool or type(ts) is not int or ts not in events):
                excluded['window_label_unknown_unmatched_or_unverified'] += 1
                continue
            if ts in seen_labels:
                if seen_labels[ts] != value:
                    issues.append('conflicting completion labels: ' + row['trial_path'])
                continue
            seen_labels[ts] = value
            add('exit',ts,value)
        if first is not None and first in seen_labels:
            initial = gt.get('initial_class')
            if ((initial == 'needs_recovery' and seen_labels[first] is True) or
                    (initial == 'already_normal' and seen_labels[first] is False)):
                issues.append('contradictory_contact_labels: initial class and first-touch target label disagree: ' + row['trial_path'])
    return sessions, samples, list(dict.fromkeys(issues)), compatibility, {
        'legacy_initial_classes':dict(legacy_initial),'legacy_final_targets':dict(legacy_final),
        'excluded':dict(excluded),'source_trials':source_trials,
        'note':'Legacy contact labels without an explicit target definition are diagnostic only; never silently relabelled.'}


def _partition(samples, cfg):
    groups = sorted({s['group'] for s in samples})
    # Sort by seeded digest. Features and labels never select the split.
    groups.sort(key=lambda g: hashlib.sha256(f"{cfg['seed']}|{g}".encode()).hexdigest())
    n = len(groups)
    ntrain = min(max(1,int(n*cfg['train_fraction'])),max(n-2,0))
    ncal = min(max(1,int(n*cfg['calibration_fraction'])),max(n-ntrain-1,0))
    parts = {'train':groups[:ntrain], 'calibration':groups[ntrain:ntrain+ncal], 'validation':groups[ntrain+ncal:]}
    indices = {k:np.array([i for i,s in enumerate(samples) if s['group'] in gs],dtype=int) for k,gs in parts.items()}
    return parts, indices


def _counts(samples):
    return {'positive':sum(s['label'] is True for s in samples),
            'negative':sum(s['label'] is False for s in samples),
            'physical_trials':len({s['trial_path'] for s in samples}),
            'groups':len({s['group'] for s in samples})}


def diagnose_training_data(roots, settings=None):
    whole,cfg = _settings(settings)
    roots = [roots] if isinstance(roots,(str,Path)) else list(roots)
    sessions,samples,issues,compatibility,diagnostics = _collect(roots,whole)
    return {'schema':'learned_skill_data_diagnosis_v1','target_definition_id':whole['target_definition_id'],
            'class_counts':{k:_counts(v) for k,v in samples.items()},'blocking_reasons':issues,
            'data_gaps':_data_gaps(samples,cfg),'diagnostics':diagnostics,'compatibility':compatibility,
            'source_sessions':[s['path'] for s in sessions],'hardware_connected':False}


def _data_gaps(samples,cfg):
    result=[]
    for name,ss in samples.items():
        parts,indices=_partition(ss,cfg)
        for part,ii in indices.items():
            # Count independent physical trials, not frames from one episode.
            for positive in (False,True):
                n=len({ss[i]['trial_path'] for i in ii if ss[i]['label'] is positive})
                minimum=cfg['min_'+part+'_per_class']
                if n<minimum:
                    result.append({'component':name,'partition':part,'class':'positive' if positive else 'negative',
                        'physical_trials':n,'minimum':minimum,'missing':minimum-n,
                        'reason':'Need explicit independent contact-state labels and whole-group coverage; repeated frames are not new trials.'})
    return result


def _design(X,mean,scale):
    z=(X-mean)/scale
    return np.column_stack([np.ones(len(X)),z,z*z])


def _sigmoid(x):
    return 1./(1.+np.exp(-np.clip(x,-40,40)))


def _metrics(y,p,valid,low,high):
    yes=valid & (p>=high); no=valid & (p<=low); covered=yes|no
    tp=int(np.sum(yes & y)); tn=int(np.sum(no & ~y)); fp=int(np.sum(yes & ~y)); fn=int(np.sum(no & y))
    positives=int(np.sum(y));negatives=len(y)-positives
    return {'n':len(y),'positives':positives,'negatives':negatives,'true_positive':tp,'true_negative':tn,
        'false_positive':fp,'false_negative':fn,'abstained':int(np.sum(~covered)),
        'coverage':float(np.mean(covered)) if len(y) else 0.,
        'false_positive_rate':fp/negatives if negatives else None,
        'false_negative_rate':fn/positives if positives else None,
        'selective_accuracy':(tp+tn)/int(np.sum(covered)) if np.any(covered) else None}


def _train_head(samples,cfg,name):
    groups,ii=_partition(samples,cfg)
    result={'family':'regularized_quadratic_logistic_centroids','usable':False,'class_counts':_counts(samples),
        'split':{'method':'whole_physical_repeat_groups','groups':groups,
                 'selection':'seeded group digest; no feature or label access'},
        'source_samples':[{k:v for k,v in s.items() if k!='features'} for s in samples],
        'feature_names':['left.centroid_u_px','left.centroid_v_px','right.centroid_u_px','right.centroid_v_px'],
        'blocking_reasons':[], 'validation':{},
        'score_meaning':'Logistic score with empirical calibration-set abstention thresholds; not a calibrated probability or a deployment certificate.'}
    gaps=_data_gaps({name:samples},cfg)
    if gaps:
        result['blocking_reasons'].append('insufficient independently labelled positive/negative trials across train/calibration/validation')
        return result
    X=np.array([s['features'] for s in samples]);y=np.array([s['label'] for s in samples],dtype=bool)
    tr,ca,va=ii['train'],ii['calibration'],ii['validation']
    mu=X[tr].mean(0);scale=X[tr].std(0);scale=np.where(scale>1e-6,scale,1.)
    Z=_design(X,mu,scale);w=np.zeros(Z.shape[1])
    # Give each physical trial equal weight even if it has many exit windows.
    trialcounts=Counter(samples[i]['trial_path'] for i in tr)
    weight=np.array([1./trialcounts[samples[i]['trial_path']] for i in tr]);weight/=weight.sum()
    for _ in range(cfg['epochs']):
        penalty=cfg['l2']*w;penalty[0]=0
        w-=cfg['learning_rate']*(Z[tr].T@(weight*(_sigmoid(Z[tr]@w)-y[tr]))+penalty)
    normalized=(X-mu)/scale
    neighbors=[]
    for i in tr:
        other=[j for j in tr if samples[j]['group']!=samples[i]['group']]
        if other: neighbors.append(float(np.min(np.sqrt(np.mean((normalized[other]-normalized[i])**2,axis=1)))))
    radius=max(cfg['ood_min_radius'],float(np.quantile(neighbors,.95))*cfg['ood_radius_multiplier'] if neighbors else cfg['ood_min_radius'])
    distance=np.min(np.sqrt(np.mean((normalized[:,None,:]-normalized[tr][None,:,:])**2,axis=2)),axis=1)
    p=_sigmoid(Z@w);supported=distance<=radius
    positive=p[ca][y[ca]];negative=p[ca][~y[ca]]
    high=max(.6,float(negative.max())+cfg['score_margin'])
    low=min(.4,float(positive.min())-cfg['score_margin'])
    result.update(weights=w.tolist(),scaling={'mean':mu.tolist(),'scale':scale.tolist()},
        ood={'support':normalized[tr].tolist(),'radius':radius,'meaning':'train-only cross-group nearest-neighbor empirical support'},
        thresholds={'negative_max':low,'positive_min':high},
        calibration=_metrics(y[ca],p[ca],supported[ca],low,high),
        validation=_metrics(y[va],p[va],supported[va],low,high))
    for part in ('calibration','validation'):
        met=result[part]
        if (not 0<=low<.5<high<=1 or met['coverage']<cfg['min_coverage'] or
                not met['true_positive'] or not met['true_negative'] or
                met['false_positive_rate']>cfg['max_false_positive_rate'] or
                met['false_negative_rate']>cfg['max_false_negative_rate']):
            result['blocking_reasons'].append(part+' contact-state discrimination/coverage gate failed')
    result['usable']=not result['blocking_reasons']
    return result


def train_learned_skill(roots, settings=None, *, out=None):
    whole,cfg=_settings(settings)
    roots=[roots] if isinstance(roots,(str,Path)) else list(roots)
    if out is not None and Path(out).exists():
        raise FileExistsError('skill versions are immutable; choose a new output')
    sessions,samples,issues,compatibility,diagnostics=_collect(roots,whole)
    # Never even pass independent sessions to the action training interface.
    action_settings=deepcopy(whole)
    if whole.get('target_mode')=='acceptable_set':
        # In this mode the predictor is a coordinate estimator. Successful B
        # action deltas may end at different targets and cannot train that map.
        action_settings['contact_policy']={**whole.get('contact_policy',{}),'include_successful_B':False}
    recovery=train_contact_policy([s['path'] for s in sessions],action_settings)
    heads={name:_train_head(samples[name],cfg,name) for name in ('invocation','exit')}
    reasons=list(issues)
    if not recovery.get('usable_for_B'):
        reasons.extend('recovery: '+x for x in recovery.get('blocking_reasons',['unusable action model']))
    for name,head in heads.items():
        reasons.extend(name+': '+x for x in head['blocking_reasons'])
    target_set=None
    if whole.get('target_mode')=='acceptable_set':
        from .acceptable_contact_set import train as train_target_set
        target_set=train_target_set(samples['exit'],heads['exit'],recovery,whole,heads['invocation'])
        reasons.extend('acceptable_set: '+x for x in target_set['blocking_reasons'])
    model={'schema':SCHEMA,'usable_for_B':not reasons,'target_definition_id':whole['target_definition_id'],
        'recovery_action_axis':'a','recovery_model':recovery,
        'invocation_head':heads['invocation'],'exit_head':heads['exit'],
        'state_learning':cfg,'blocking_reasons':list(dict.fromkeys(reasons)),
        'data_gaps':_data_gaps(samples,cfg),'diagnostics':diagnostics,'compatibility':compatibility,
        'source_sessions':[s['path'] for s in sessions],
        'source_snapshot_sha256':_hash({'samples':samples,'session_hashes':[s['metadata_sha256'] for s in sessions]}),
        'simulated':bool(whole.get('simulate',False)),
        'three_components_updated_together':True,'certified':False,
        'caveats':['Action supervision remains measured geometric return to reference, not optimal recovery.',
                   'State scores do not prove an action is physically recoverable; actual B transitions are required.',
                   'Safety, freshness, action receipts and temporal completion guards are not learned or relaxed.',
                   'Development/calibration metrics are not independent final validation.']}
    if target_set is not None:
        model.update(schema=SET_SCHEMA,target_mode='acceptable_set',acceptable_set=target_set)
        model['data_gaps'].extend({'component':'acceptable_set','reason':reason,
            'required_evidence':'Explicit positive and negative window labels, repeated measured A anchors across independent groups, and held-out support for each target neighborhood'}
            for reason in target_set['blocking_reasons'])
        model['caveats'][0]='Measured geometry estimates current a; actions target an observed acceptable contact set, not T0. Actual B recovery transitions still required.'
    for key in ('reference_T0','axes_tcp','normal_axis','normal_signs','signature','fingers','common_range_mm','common_linear_range_mm'):
        if key in recovery:model[key]=deepcopy(recovery[key])
    _seal(model)
    if out is not None:
        p=Path(out);p.parent.mkdir(parents=True,exist_ok=True)
        with p.open('x',encoding='utf-8') as f:json.dump(model,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
    return model


class LearnedSkillModel:
    """Validated immutable learned state heads; no control or truth inputs."""
    def __init__(self,model):
        if model.get('schema') not in (SCHEMA,SET_SCHEMA) or model.get('integrity_sha256')!=_digest(model) or model.get('model_id')!=model.get('integrity_sha256','')[:20]:
            raise ValueError('invalid learned skill schema or integrity')
        if model.get('usable_for_B') is not True or model.get('blocking_reasons') or not model.get('target_definition_id'):
            raise ValueError('learned skill is not ready: '+str(model.get('blocking_reasons')))
        recovery=model.get('recovery_model',{})
        if (recovery.get('usable_for_B') is not True or recovery.get('schema')!='contact_policy_v1' or
                recovery.get('integrity_sha256')!=_recovery_digest(recovery) or
                recovery.get('model_id')!=recovery.get('integrity_sha256','')[:20]):
            raise ValueError('unusable recovery model')
        self.model=deepcopy(model)
        if (model.get('schema')==SET_SCHEMA)!=(model.get('target_mode')=='acceptable_set'):
            raise ValueError('learned target mode and schema are inconsistent')
        if model.get('target_mode') not in (None,'reference','acceptable_set'):
            raise ValueError('unsupported learned target mode')
        if model.get('target_mode')=='acceptable_set':
            from .acceptable_contact_set import validate as validate_target_set
            validate_target_set(model.get('acceptable_set',{}),model['target_definition_id'])
            if recovery.get('settings',{}).get('include_successful_B') is not False:
                raise ValueError('acceptable-set recovery requires geometry-only action supervision')
        for name in ('invocation','exit'):
            h=self.model.get(name+'_head') or {}
            if h.get('usable') is not True or h.get('blocking_reasons'):
                raise ValueError('unusable '+name+' head')
            w=np.asarray(h.get('weights'),float);mu=np.asarray(h.get('scaling',{}).get('mean'),float);scale=np.asarray(h.get('scaling',{}).get('scale'),float)
            support=np.asarray(h.get('ood',{}).get('support'),float)
            if w.shape!=(9,) or mu.shape!=(4,) or scale.shape!=(4,) or support.ndim!=2 or support.shape[1]!=4 or not len(support) or not np.isfinite(w).all() or not np.isfinite(mu).all() or not np.isfinite(scale).all() or not np.isfinite(support).all() or np.any(scale<=0):
                raise ValueError('invalid '+name+' numerical artifact')
            t=h.get('thresholds',{});radius=h['ood'].get('radius')
            if not 0<=t.get('negative_max',-1)<.5<t.get('positive_min',2)<=1 or type(radius) not in (int,float) or not np.isfinite(radius) or radius<=0:
                raise ValueError('invalid learned thresholds/domain')

    def _predict(self,name,observation):
        base={'valid':False,'positive':None,'probability':None}
        if (observation.get('context') or {}).get('target_definition_id')!=self.model['target_definition_id']:
            return {**base,'reason':'target_definition_mismatch'}
        try:x=_features(observation)[None,:]
        except (ValueError,TypeError):return {**base,'reason':name+'_missing_features'}
        h=self.model[name+'_head'];mu=np.array(h['scaling']['mean']);scale=np.array(h['scaling']['scale'])
        z=(x-mu)/scale;distance=float(np.min(np.sqrt(np.mean((np.array(h['ood']['support'])-z[0])**2,axis=1))))
        if distance>h['ood']['radius']:
            return {**base,'reason':name+'_out_of_distribution','ood_distance':distance}
        score=float(_sigmoid(_design(x,mu,scale)@np.array(h['weights']))[0])
        t=h['thresholds']; positive=True if score>=t['positive_min'] else False if score<=t['negative_max'] else None
        return {'valid':positive is not None,'positive':positive,'probability':score,
                'reason':name+'_supported' if positive is not None else name+'_uncertain','ood_distance':distance}

    def invocation(self,observation):return self._predict('invocation',observation)

    def exit(self,observation):
        prediction=self._predict('exit',observation)
        if self.model.get('target_mode')=='acceptable_set' and prediction.get('valid') is True:
            from .acceptable_contact_set import match_features
            match=match_features(self.model['acceptable_set'],_features(observation))
            prediction['acceptable_set_match']=match
            if prediction.get('positive') is True and not match['supported']:
                prediction.update(valid=False,positive=None,reason='exit_outside_supported_acceptable_set')
        return prediction

    def recovery_prediction(self,prediction):
        if self.model.get('target_mode')=='acceptable_set':
            from .acceptable_contact_set import choose_action
            return choose_action(self.model['acceptable_set'],prediction)
        return prediction
