"""Declarative improvement methods, evidence memory and fixed-budget comparison.

Offline only: no ROS imports, motion, shell execution, pickle or Python eval.
Method text is a hypothesis; only recomputed comparison evidence changes status.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from contextlib import contextmanager
from copy import deepcopy
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import tempfile

FEATURES = frozenset(('failure_rate', 'uncertainty', 'novelty', 'action_conflict', 'coverage_gap'))
SCHEMA = 'rsi_method_v1'
CONFLICT_CAVEAT = ('action_conflict compares measured TCP geometric supervision (-a offset) '
    'for similar two-finger centroids; it is not evidence of different optimal recovery actions. '
    'Bottle motion, omitted DD shape/history and label noise can explain apparent conflict.')


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _num(value, lo=-math.inf, hi=math.inf):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and lo <= value <= hi


def _text(value, limit=4000):
    return isinstance(value, str) and bool(value.strip()) and len(value) <= limit and '\x00' not in value


def _ast(node, depth=0, count=None):
    count = [0] if count is None else count
    count[0] += 1
    if depth > 8 or count[0] > 128 or not isinstance(node, dict):
        raise ValueError('score AST exceeds depth 8/nodes 128 or is not an object')
    op = node.get('op')
    if op == 'feature':
        if set(node) != {'op', 'name'} or not isinstance(node['name'], str) or node['name'] not in FEATURES:
            raise ValueError('Unknown score feature or extra AST fields')
        return dict(node)
    if op == 'constant':
        if set(node) != {'op', 'value'} or not _num(node['value'], -10, 10):
            raise ValueError('score constants must be finite in [-10,10]')
        return {'op': op, 'value': float(node['value'])}
    if op not in ('add', 'mul', 'min', 'max') or set(node) != {'op', 'args'}:
        raise ValueError('Only feature/constant/add/mul/min/max AST operations are allowed')
    if not isinstance(node['args'], list) or not 2 <= len(node['args']) <= 8:
        raise ValueError('score operator args must contain 2..8 nodes')
    return {'op': op, 'args': [_ast(n, depth + 1, count) for n in node['args']]}


def validate_method(method, *, known_refs=None, parent_ids=None):
    required = {'name', 'parent_id', 'hypothesis', 'applicability', 'score', 'sampling', 'evidence_refs', 'expected_outcome'}
    if not isinstance(method, dict) or not required <= set(method) or set(method) - required - {'schema', 'method_id'}:
        raise ValueError('method has missing or unknown fields; controller/safety changes are forbidden')
    for name in ('name', 'hypothesis', 'applicability', 'expected_outcome'):
        if not _text(method[name], 160 if name == 'name' else 4000):
            raise ValueError('method ' + name + ' must be bounded nonempty text')
    parent = method['parent_id']
    if parent is not None and (not _text(parent, 128) or (parent_ids is not None and parent not in parent_ids)):
        raise ValueError('Unknown or malformed parent_id')
    refs = method['evidence_refs']
    if (not isinstance(refs, list) or not 1 <= len(refs) <= 128 or any(not _text(x, 128) for x in refs)
            or len(set(refs)) != len(refs) or (known_refs is not None and any(x not in known_refs for x in refs))):
        raise ValueError('evidence_refs must be unique known references, 1..128 items')
    sample = method['sampling']
    if not isinstance(sample, dict) or set(sample) != {'step_mm', 'extra_offsets_mm', 'max_reps_per_cell', 'diversity_mm'}:
        raise ValueError('sampling fields must be step_mm,extra_offsets_mm,max_reps_per_cell,diversity_mm')
    if not _num(sample['step_mm'], .5, 50) or not _num(sample['diversity_mm'], 0, 50):
        raise ValueError('step_mm must be 0.5..50; diversity_mm must be 0..50')
    cap = sample['max_reps_per_cell']
    if type(cap) is not int or not 2 <= cap <= 1000 or cap % 2:
        raise ValueError('max_reps_per_cell must be positive even integer <=1000')
    extra = sample['extra_offsets_mm']
    if (not isinstance(extra, list) or len(extra) > 101 or any(not _num(x, -25, 25) for x in extra)
            or len(set(extra)) != len(extra)):
        raise ValueError('extra_offsets_mm must be unique finite positions in [-25,25], at most 101')
    if any(b-a < .5-1e-9 for a,b in zip(sorted(extra), sorted(extra)[1:])):
        raise ValueError('extra_offsets_mm distinct spacing must be >=0.5 mm')
    result = {k: deepcopy(method[k]) for k in required}
    result['score'] = _ast(method['score'])
    result['sampling'] = {'step_mm': float(sample['step_mm']), 'extra_offsets_mm': sorted(map(float, extra)),
                          'max_reps_per_cell': cap, 'diversity_mm': float(sample['diversity_mm'])}
    result['schema'] = SCHEMA
    identity = _hash(result)
    if method.get('schema', SCHEMA) != SCHEMA or method.get('method_id', identity) != identity:
        raise ValueError('method schema/content hash mismatch')
    result['method_id'] = identity
    return result


def baseline_method():
    return validate_method({'name': 'Failure rate with coverage floor', 'parent_id': None,
        'hypothesis': 'A fixed failure-first rule is a reproducible comparison baseline.',
        'applicability': 'One-dimensional bottle contact collection under the unchanged experiment envelope.',
        'score': {'op':'add','args':[{'op':'feature','name':'failure_rate'},
                  {'op':'mul','args':[{'op':'constant','value':.2},{'op':'feature','name':'coverage_gap'}]}]},
        'sampling': {'step_mm': 5., 'extra_offsets_mm': [], 'max_reps_per_cell': 20, 'diversity_mm': 0.},
        'evidence_refs': ['baseline_protocol'], 'expected_outcome': 'Reference data-efficiency curve; no claim of superiority.'})


def _score(ast, features):
    op = ast['op']
    if op == 'feature':
        return features[ast['name']]
    if op == 'constant':
        return ast['value']
    values = [_score(child, features) for child in ast['args']]
    result = {'add': lambda: sum(values), 'mul': lambda: math.prod(values),
              'min': lambda: min(values), 'max': lambda: max(values)}[op]()
    return max(-1e6, min(1e6, result))


def _positions(scope, step, extra=()):
    if not isinstance(scope, dict) or not all(_num(scope.get(k), -25, 25) for k in ('min_mm','max_mm')):
        raise ValueError('scope must provide finite min_mm/max_mm within [-25,25]')
    lo, hi = float(scope['min_mm']), float(scope['max_mm'])
    if lo >= hi or not lo <= 0 <= hi or not _num(step, .5, 50):
        raise ValueError('scope must include zero and have positive width; nominal step >=0.5 mm')
    if any(not lo <= x <= hi for x in extra):
        raise ValueError('method extra offsets exceed the immutable scope; no silent clipping')
    # Explicit extra points take priority over grid points within 0.5 mm.
    positions = list(extra)
    for point in [0., lo, hi] + [round(lo + i*step, 10) for i in range(int((hi-lo)/step)+1)]:
        if all(abs(point-existing) >= .5-1e-9 for existing in positions):
            positions.append(float(point))
    return sorted(positions)


def _outcomes(report):
    """Include real complete A failures absent from regression observations."""
    counts = defaultdict(lambda: [0, 0])
    seen = set()
    for item in report.get('evidence', {}).get('sessions', []):
        root = Path(item['path']) if isinstance(item, dict) and item.get('path') else None
        if root is None or str(root.resolve()) in seen or item.get('role') in ('validation','frozen_validation','final_validation'):
            continue
        seen.add(str(root.resolve()))
        for file in sorted((root / 'trials').glob('*/meta.json')):
            try:
                row = json.loads(file.read_text())
                status = json.loads((file.parent / 'recording_status.json').read_text())
            except (OSError, ValueError):
                continue
            if (row.get('cond') != 'A' or not row.get('complete') or not row.get('recording_complete') or not status.get('complete')
                    or row.get('label') == 'invalid' or row.get('status') == 'skipped_bound' or not _num(row.get('delta_mm'))):
                continue
            success = bool(row.get('lifted') and row.get('hold_complete') and not row.get('dropped') and row.get('label') == 'normal')
            counts[float(row['delta_mm'])][0] += int(not success)
            counts[float(row['delta_mm'])][1] += 1
    for d in report.get('diagnostics', []):
        if d.get('code') == 'B_offset_feedback' and _num(d.get('offset_mm')) and _num(d.get('n'), 1) and _num(d.get('failures'), 0, d['n']):
            counts[float(d['offset_mm'])][0] += d['failures']
            counts[float(d['offset_mm'])][1] += d['n']
    return counts


def cell_features(report, *, scope, step_mm=1., extra_offsets_mm=()):
    """Observable conflict is a *geometric proxy*, never an optimal action label.

    Only measured, independent A trial groups enter the conflict calculation.
    Centroid distance <=2 px and measured geometric target difference >=2 mm
    identify ambiguous observations. Repeated frames/trial identities add no votes.
    """
    positions = _positions(scope, step_mm, extra_offsets_mm)
    observations, identities = [], set()
    for o in report.get('evidence', {}).get('observations', []):
        identity = (o.get('session'), o.get('id'))
        if identity in identities or not _num(o.get('planned_delta_mm')) or not _num(o.get('x_mm')):
            continue
        desc = o.get('descriptors', [])
        if len(desc) != 2 or any(not d.get('valid') or not isinstance(d.get('centroid_px'), list) or
            len(d['centroid_px']) != 2 or any(not _num(v) for v in d['centroid_px']) for d in desc):
            continue
        identities.add(identity)
        observations.append(o)
    conflict = defaultdict(float)
    pairs = []
    # Spatial bins avoid comparing distant observations. Every retained pair is
    # still checked by its actual 2px RMS distance; no optimal-action claim.
    buckets = defaultdict(list)
    eligible = [o for o in observations if o.get('measurement', {}).get('source','').startswith('measured_tcp')]
    for o in eligible:
        vector = [v for d in o['descriptors'] for v in d['centroid_px']]
        key = tuple(int(math.floor(v/4.)) for v in vector)
        candidates = []
        import itertools
        for delta in itertools.product((-1,0,1), repeat=4):
            candidates.extend(buckets.get(tuple(a+b for a,b in zip(key,delta)), ()))
        for other, oldvec in candidates:
            if o.get('group') and o.get('group') == other.get('group'):
                continue
            distance = math.sqrt(sum((a-b)**2 for a,b in zip(vector, oldvec))/4.)
            action_gap = abs(o['x_mm']-other['x_mm'])
            if distance > 2. or action_gap < 2.:
                continue
            strength = math.exp(-.5*(distance/2.)**2) * min(1., action_gap/6.)
            for item in (o, other):
                conflict[float(item['planned_delta_mm'])] = max(conflict[float(item['planned_delta_mm'])], strength)
            if len(pairs) < 32:
                pairs.append({'left': f"{other.get('session')}|{other.get('id')}",
                    'right': f"{o.get('session')}|{o.get('id')}", 'centroid_rms_px': distance,
                    'geometric_target_gap_mm': action_gap, 'strength': strength})
        buckets[key].append((o, vector))
    levels = report.get('evidence', {}).get('level_stats', [])
    outcomes = _outcomes(report)
    result = []
    radius = max(.25, step_mm/2.)
    span = float(scope['max_mm']-scope['min_mm'])
    for x in positions:
        nearby = [o for o in observations if abs(o['planned_delta_mm']-x) <= radius]
        bottle_count = Counter(o.get('bottle') for o in nearby)
        failure = sum(v[0] for loc,v in outcomes.items() if abs(loc-x) <= radius)
        total = sum(v[1] for loc,v in outcomes.items() if abs(loc-x) <= radius)
        errors = [float(l['cv_mae_mm']) for l in levels if _num(l.get('offset_mm')) and abs(l['offset_mm']-x)<=radius
                  and _num(l.get('cv_mae_mm'),0)]
        count = len(nearby)
        gap = 1./(1.+min(bottle_count.get('light',0),bottle_count.get('heavy',0)))
        uncertainty = max(1./math.sqrt(1+count), min(1.,max(errors,default=0.)/3.))
        distance = min((abs(x-o['planned_delta_mm']) for o in observations), default=span)
        feature = {'failure_rate': failure/total if total else .5, 'uncertainty': uncertainty,
                   'novelty': min(1.,distance/max(.5,span/4.)), 'coverage_gap': gap,
                   'action_conflict': max((v for pos,v in conflict.items() if abs(pos-x)<=radius),default=0.)}
        result.append({'offset_mm':x, 'features':feature, 'observations':count, 'outcome_trials':total})
    return {'cells':result, 'conflict_examples':pairs, 'conflict_definition':CONFLICT_CAVEAT,
            'excluded_unmeasured_from_conflict':len(observations)-len(eligible),
            'nominal_spacing_caveat':'Requested offsets are not a claim of achieved TCP precision.'}


def select_cells(method, report, *, scope, trial_budget, seed=0):
    method = validate_method(method)
    if type(trial_budget) is not int or trial_budget < 2 or trial_budget % 2:
        raise ValueError('trial_budget must be a positive even number; light/heavy are balanced')
    if report.get('quality', {}).get('blocking_reasons'):
        raise ValueError('Cannot select cells from incompatible/blocked evidence: '+str(report['quality']['blocking_reasons']))
    sample = method['sampling']
    data = cell_features(report, scope=scope, step_mm=sample['step_mm'], extra_offsets_mm=sample['extra_offsets_mm'])
    rng = random.Random(seed)
    ranked = [{'offset_mm':c['offset_mm'], 'score':_score(method['score'],c['features']), 'tie':rng.random()}
              for c in data['cells']]
    selected = Counter()
    for _ in range(trial_budget//2):
        available = [c for c in ranked if selected[c['offset_mm']] < sample['max_reps_per_cell'] and
            (selected[c['offset_mm']] or all(abs(c['offset_mm']-p)>=sample['diversity_mm']-1e-9 for p in selected))]
        if not available:
            raise ValueError('Method diversity/repetition constraints cannot fill equal trial budget; revise the method explicitly')
        # Repeated visits are penalized regardless of score sign. This prevents
        # arbitrary negative scores accidentally favoring duplicate acquisition.
        chosen = max(available, key=lambda c:(c['score']-math.log1p(selected[c['offset_mm']]/2.), c['tie']))
        selected[chosen['offset_mm']] += 2
    return [{'offset_mm':x,'reps':n} for x,n in sorted(selected.items())]


def _wilson(k,n):
    if n == 0:
        return [0.,1.]
    z=1.959963984540054
    denominator=1+z*z/n
    center=(k/n+z*z/(2*n))/denominator
    half=z*math.sqrt(k/n*(1-k/n)/n+z*z/(4*n*n))/denominator
    return [max(0.,center-half),min(1.,center+half)]


def _cell_map(cells, outcome=False):
    if not isinstance(cells,list) or not cells:
        raise ValueError('evaluation_cells must be a nonempty fixed plan')
    result={}
    for cell in cells:
        if not isinstance(cell,dict) or not _num(cell.get('offset_mm'),-25,25) or not _text(cell.get('bottle'),128):
            raise ValueError('evaluation cell needs finite offset and bottle')
        if type(cell.get('trials')) is not int or cell['trials']<1:
            raise ValueError('evaluation cell trials must be positive integer')
        key=(float(cell['offset_mm']),cell['bottle'])
        if key in result:
            raise ValueError('duplicate evaluation cells')
        if outcome and (type(cell.get('successes')) is not int or not 0<=cell['successes']<=cell['trials']):
            raise ValueError('evaluation cell successes must be integer in [0,trials]')
        result[key]=cell
    return result


def evaluate_comparison(old, new, protocol):
    """Fixed-sample independent branch comparison, with conservative uncertainty.

    Summaries must be built by the executor from all records, including failed
    attempts. This validates protocol consistency; it cannot authenticate arbitrary
    externally fabricated summaries. Method status is never accepted as input.
    """
    issues=[]
    if not all(isinstance(x,dict) for x in (old,new,protocol)):
        raise ValueError('comparison needs old/new/protocol JSON objects')
    required=('task_id','initial_dataset_hash','initial_policy_hash','training_budget','collection_budget','simulated')
    for field in required:
        if field not in protocol or field not in old or field not in new:
            issues.append('missing comparison identity/budget: '+field)
        elif old[field]!=new[field] or old[field]!=protocol[field]:
            issues.append('different starting conditions or budget: '+field)
    for field in ('task_id','initial_dataset_hash','initial_policy_hash'):
        if not _text(protocol.get(field),256):
            issues.append('invalid protocol identity: '+field)
    if type(protocol.get('simulated')) is not bool:
        issues.append('explicit simulated flag required')
    if not protocol.get('preregistered') is True or not protocol.get('evaluation_independent') is True:
        issues.append('evaluation must be preregistered and independent of training/method selection')
    if type(protocol.get('collection_budget')) is not int or protocol['collection_budget']<1:
        issues.append('positive collection budget required')
    if not _num(protocol.get('training_budget'),1) and not (isinstance(protocol.get('training_budget'),dict) and protocol['training_budget']):
        issues.append('training budget must be a positive number or fixed budget specification')
    min_n=protocol.get('min_trials',20)
    effect=protocol.get('min_effect',.05)
    if type(min_n) is not int or min_n<20 or not _num(effect,0,1):
        issues.append('min_trials must be >=20; min_effect must be finite in [0,1]')
        min_n,effect=20,.05
    maps=[]
    try:
        expected=_cell_map(protocol.get('evaluation_cells'))
    except ValueError as exc:
        expected={}; issues.append(str(exc))
    for name,group in (('old',old),('new',new)):
        if not _text(group.get('method_id'),128):
            issues.append(name+': missing method_id')
        if group.get('records_complete') is not True:
            issues.append(name+': incomplete raw records/attempt accounting')
        if type(group.get('spent_trials')) is not int or group['spent_trials']!=protocol.get('collection_budget'):
            issues.append(name+': spent_trials differs from fixed collection budget (include all attempts)')
        k,n=group.get('successes'),group.get('trials')
        if type(k) is not int or type(n) is not int or n<1 or not 0<=k<=n:
            issues.append(name+': invalid successes/trials')
        try:
            cells=_cell_map(group.get('evaluation_cells'),outcome=True)
            if set(cells)!=set(expected) or any(cells[key]['trials']!=expected[key]['trials'] for key in set(cells)&set(expected)):
                issues.append(name+': evaluation coverage/count differs from fixed plan')
            if sum(c['trials'] for c in cells.values())!=n or sum(c['successes'] for c in cells.values())!=k:
                issues.append(name+': aggregate counts do not match cell records')
        except ValueError as exc:
            cells={}; issues.append(name+': '+str(exc))
        maps.append(cells)
    if old.get('method_id')==new.get('method_id'):
        issues.append('cannot compare a method with itself')
    rates=[]
    for group in (old,new):
        k,n=group.get('successes'),group.get('trials')
        valid=type(k) is int and type(n) is int and n>0 and 0<=k<=n
        rates.append({'success_rate':k/n if valid else None,'wilson_95':_wilson(k,n) if valid else [0.,1.],
                      'successes':k,'trials':n})
    lower=rates[1]['wilson_95'][0]-rates[0]['wilson_95'][1]
    upper=rates[1]['wilson_95'][1]-rates[0]['wilson_95'][0]
    outcome='inconclusive'
    if issues:
        outcome='invalid'
    elif min(old['trials'],new['trials'])<min_n:
        issues.append('insufficient fixed evaluation sample; do not promote from a small sample')
    elif lower>effect:
        outcome='new_better'
    elif upper < -effect:
        outcome='old_better'
    if not issues and outcome=='inconclusive':
        issues.append('conservative success-rate intervals overlap or effect is too small')
    result={'schema':'rsi_comparison_v1','outcome':outcome,'task_id':protocol.get('task_id'),
            'simulated':protocol.get('simulated'), 'old_method_id':old.get('method_id'),'new_method_id':new.get('method_id'),
            'old':rates[0],'new':rates[1],'difference_interval_conservative': [lower,upper],
            'issues':issues,'winner': new.get('method_id') if outcome=='new_better' else old.get('method_id') if outcome=='old_better' else None,
            'protocol_hash':_hash(protocol),'evidence_hash':_hash({'old':old,'new':new,'protocol':protocol}),
            'interpretation':'Evidence about fixed-budget acquisition efficiency for this task and scope; no proof of optimal actions or general RSI.'}
    result['comparison_id']=_hash(result)
    return result


def learning_curve(points):
    """Report descriptive data-efficiency AUC, never a replacement for fixed test."""
    if not isinstance(points,list) or len(points)<2:
        raise ValueError('learning curve needs at least two independent checkpoint summaries')
    ordered=sorted(points,key=lambda p:p.get('spent_trials',-1))
    for i,p in enumerate(ordered):
        if type(p.get('spent_trials')) is not int or p['spent_trials']<0 or not _num(p.get('success_rate'),0,1):
            raise ValueError('invalid learning curve checkpoint')
        if i and p['spent_trials']<=ordered[i-1]['spent_trials']:
            raise ValueError('learning curve budgets must be strictly increasing')
    width=ordered[-1]['spent_trials']-ordered[0]['spent_trials']
    auc=sum((b['spent_trials']-a['spent_trials'])*(a['success_rate']+b['success_rate'])/2 for a,b in zip(ordered,ordered[1:]))/width
    return {'points':deepcopy(ordered),'normalized_auc':auc,'budget_range':[ordered[0]['spent_trials'],ordered[-1]['spent_trials']],
            'interpretation':'Descriptive learning curve; repeated checkpoint peeking is not independent final validation.'}


class MethodStore:
    """Immutable content-addressed files and append-only derived experience.

    Locking prevents lost updates among local processes. It is not protection
    against a user who can rewrite the filesystem; evidence authenticity comes
    from the trusted executor and its original raw-record hashes.
    """
    def __init__(self,root):
        self.root=Path(root).expanduser().resolve()
        self.root.mkdir(parents=True,exist_ok=True)
        for name in ('methods','experiences'):
            (self.root/name).mkdir(exist_ok=True)

    @contextmanager
    def _lock(self):
        with (self.root/'.lock').open('a') as lock:
            fcntl.flock(lock.fileno(),fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock.fileno(),fcntl.LOCK_UN)

    def _write(self,directory,identity,value):
        path=self.root/directory/(identity+'.json')
        payload=json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+'\n'
        if path.exists():
            if json.loads(path.read_text())!=value:
                raise ValueError('immutable memory collision/corruption')
            return
        fd,name=tempfile.mkstemp(prefix='.pending-',dir=str(path.parent))
        try:
            with os.fdopen(fd,'w') as f:
                f.write(payload); f.flush(); os.fsync(f.fileno())
            os.replace(name,path)
            descriptor=os.open(path.parent,os.O_RDONLY)
            try: os.fsync(descriptor)
            finally: os.close(descriptor)
        finally:
            if os.path.exists(name): os.unlink(name)

    def register(self,method):
        with self._lock():
            parents={p.stem for p in (self.root/'methods').glob('*.json')}
            value=validate_method(method,parent_ids=parents)
            self._write('methods',value['method_id'],value)
            return value['method_id']

    def get(self,identity):
        if not isinstance(identity,str) or not re.fullmatch('[0-9a-f]{64}',identity):
            raise ValueError('invalid method identity')
        value=json.loads((self.root/'methods'/(identity+'.json')).read_text())
        value=validate_method(value)
        if value['method_id']!=identity:
            raise ValueError('method file content identity mismatch')
        return value

    def record_experience(self,record):
        if not isinstance(record,dict):
            raise ValueError('experience must be a JSON object')
        forbidden={'status','promoted','winner','outcome','evaluation','experience_id'}
        if forbidden & set(record):
            raise ValueError('experience status/promotion is evaluator-derived; do not supply claims')
        if not _text(record.get('task_id'),256) or type(record.get('simulated')) is not bool or not isinstance(record.get('context'),dict):
            raise ValueError('experience needs task_id, explicit simulated, and context')
        if 'created_unix_ns' in record and (type(record['created_unix_ns']) is not int or record['created_unix_ns'] <= 0):
            raise ValueError('experience created_unix_ns must be a positive integer when supplied')
        method=self.get(record.get('method_id'))
        context=record['context']
        if not record['simulated'] and not _text(context.get('hardware_signature'),256):
            raise ValueError('hardware experience needs a verified hardware_signature')
        if context.get('simulated',record['simulated'])!=record['simulated']:
            raise ValueError('experience context simulation mismatch')
        value=deepcopy(record)
        comparison=record.get('comparison')
        if comparison is not None:
            if not isinstance(comparison,dict) or set(comparison)!={'old','new','protocol'}:
                raise ValueError('comparison must contain only old,new,protocol summaries')
            evaluation=evaluate_comparison(**comparison)
            if evaluation['task_id']!=record['task_id'] or evaluation['simulated']!=record['simulated']:
                raise ValueError('experience/comparison task or simulation mismatch')
            if method['method_id'] not in (evaluation['old_method_id'],evaluation['new_method_id']):
                raise ValueError('experience method is not part of comparison')
            self.get(evaluation['old_method_id']); self.get(evaluation['new_method_id'])
            value['evaluation']=evaluation
        else:
            value['evaluation']=None
        value['schema']='rsi_experience_v1'
        value['experience_id']=_hash(value)
        with self._lock():
            self._write('experiences',value['experience_id'],value)
        return value['experience_id']

    def _experiences(self):
        result=[]
        for path in sorted((self.root/'experiences').glob('*.json')):
            value=json.loads(path.read_text())
            check=deepcopy(value); identity=check.pop('experience_id',None)
            if identity!=path.stem or _hash(check)!=identity:
                raise ValueError('experience memory hash mismatch')
            if value.get('comparison') is not None:
                if evaluate_comparison(**value['comparison'])!=value['evaluation']:
                    raise ValueError('experience evaluation differs from trusted recomputation')
            elif value.get('evaluation') is not None:
                raise ValueError('experience contains evaluation without original evidence')
            result.append(value)
        # Creation order is supplied by the campaign once, not generated here:
        # retrying an experience remains content-addressed and idempotent. Older
        # records have no timestamp and retain a deterministic hash ordering.
        return sorted(result,key=lambda value:(value.get('created_unix_ns',0),value['experience_id']))

    @staticmethod
    def _status(experiences,identity):
        wins=set(); problems=set(); losses=[]
        for item in experiences:
            result=item.get('evaluation') or {}
            if result.get('winner')==identity:
                wins.add(item['task_id'])
                context=item.get('context') or {}
                # Output directories/campaign IDs identify repetitions, not a
                # new research problem. Legacy records remain task-local until
                # an explicit problem description supplies a distinct identity.
                problem=context.get('problem_signature')
                if context.get('explicit_problem') is True and _text(problem,256):
                    problems.add(problem)
            elif result.get('winner'):
                losses.append(item['experience_id'])
        status=('contested' if wins and losses else 'cross_task_valid' if len(problems)>=2 else
                'current_task_valid' if wins else 'not_supported' if losses else 'pending')
        return {'status':status,'winning_task_ids':sorted(wins),'winning_problem_ids':sorted(problems),
                'counterexample_ids':losses,
                'interpretation':'Past validity is scope-qualified; cross-task status requires distinct explicitly described problems, not renamed campaigns. Conflicting evidence must be inspected before reuse.'}

    def list_methods(self):
        experiences=self._experiences()
        result=[]
        for path in sorted((self.root/'methods').glob('*.json')):
            method=self.get(path.stem)
            relevant=[x for x in experiences if x['method_id']==method['method_id'] or
                      method['method_id'] in ((x.get('evaluation') or {}).get('old_method_id'),(x.get('evaluation') or {}).get('new_method_id'))]
            groups=defaultdict(list)
            for item in relevant:
                groups[(item['simulated'],item['context'].get('hardware_signature'))].append(item)
            method['lifecycle']={'experience_count':len(relevant),'by_scope':[
                {'simulated':key[0],'hardware_signature':key[1],**self._status(items,method['method_id'])}
                for key,items in groups.items()]}
            result.append(method)
        return result

    def retrieve(self,context,limit=5):
        if not isinstance(context,dict) or type(context.get('simulated')) is not bool:
            raise ValueError('retrieval context needs explicit simulated flag')
        if not context['simulated'] and not _text(context.get('hardware_signature'),256):
            raise ValueError('real retrieval needs hardware signature; cannot silently transfer simulation evidence')
        if type(limit) is not int or not 1<=limit<=100:
            raise ValueError('retrieval limit must be integer 1..100')
        query=set(re.findall(r'[\w]+',json.dumps(context,ensure_ascii=False).lower()))
        evidence=[x for x in self._experiences() if x['simulated']==context['simulated'] and
                  x['context'].get('hardware_signature')==context.get('hardware_signature')]
        results=[]
        for row in self.list_methods():
            method=self.get(row['method_id'])
            related=[x for x in evidence if x['method_id']==method['method_id'] or
                     method['method_id'] in ((x.get('evaluation') or {}).get('old_method_id'),(x.get('evaluation') or {}).get('new_method_id'))]
            words=set(re.findall(r'[\w]+',json.dumps({'method':method,'contexts':[x['context'] for x in related]},ensure_ascii=False).lower()))
            relevance=len(words&query)/max(1,len(query))
            status=self._status(related,method['method_id'])
            results.append({'method':method,**status,'relevance':relevance,'past_experiences':deepcopy(related),
                            'scope':{'simulated':context['simulated'],'hardware_signature':context.get('hardware_signature')},
                            'applicability_requires_check':True})
        return sorted(results,key=lambda r:(-r['relevance'],-len(r['past_experiences']),r['method']['method_id']))[:limit]
