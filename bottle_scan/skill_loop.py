"""Versioned a-axis joint skill campaign. Offline until explicit run --real.

The planner has no actuator API. All physical batches go through skill_collection,
with the original human setup/labels, ownership, motion and force checks intact.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import fcntl
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import yaml
from .session import atomic_json, plain

SCHEMA = 'a_skill_loop_v1'
TERMINAL = {'complete', 'budget_exhausted', 'independent_failed'}


def digest(value):
    return hashlib.sha256(json.dumps(plain(value),sort_keys=True,allow_nan=False,separators=(',',':')).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def immutable(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f: json.dump(plain(value),f,ensure_ascii=False,indent=2,allow_nan=False)


def raw_tcp_snapshot(path):
    # The action learner reads this raw stream as well as meta.json. Do not
    # hash unrelated camera payloads at every safe boundary.
    path=Path(path)
    if not path.exists():return None
    try:
        import h5py
        with h5py.File(path,'r') as h:
            if 'streams/tcp_measured' not in h:return {'tcp_measured':'absent'}
            g=h['streams/tcp_measured'];out={}
            for key in ('monotonic_ns','data'):
                a=np.asarray(g[key]);out[key]={'shape':list(a.shape),'dtype':str(a.dtype),'hash':hashlib.sha256(a.tobytes()).hexdigest()}
            return out
    except (OSError,KeyError) as exc:
        return {'unreadable':type(exc).__name__}


def code_snapshot():
    root=Path(__file__).resolve().parents[1]
    paths=list((root/'bottle_scan').glob('*.py'))+list(root.glob('*.py'))+[root/'run_on_5090.sh']
    return {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}


def source_snapshot(root):
    root=Path(root).resolve(); meta=read(root/'session_meta.json')
    trials=[]
    for p in sorted((root/'trials').glob('*/meta.json')):
        if meta.get('skill_label_review'):
            from .skill_label_review import verify_view_row
            verify_view_row(root,meta,read(p),p)
        trials.append({'path':str(p),'hash':digest(read(p)),
                       'raw_tcp':raw_tcp_snapshot(p.parent/'streams.h5'),
                       'recording':digest(read(p.parent/'recording_status.json')) if (p.parent/'recording_status.json').exists() else None})
    return {'path':str(root),'meta_hash':digest(meta),'trials':trials}


def complete_session(root):
    root=Path(root)
    if not (root/'session_meta.json').exists(): return False
    meta=read(root/'session_meta.json'); plan=meta.get('plan',[])
    progress=read(root/'progress.json') if (root/'progress.json').exists() else {}
    if progress.get('active') or progress.get('airborne') or progress.get('unsafe_recovery'): return False
    return bool(plan) and all((root/'trials'/p['id']/'meta.json').exists() and
                             read(root/'trials'/p['id']/'meta.json').get('complete') is True for p in plan)


def attempts(root):
    # Partial/interrupted physical records count, even if the trial was retried.
    root=Path(root)
    return len(list((root/'trials').glob('*/meta.json')))+len(list((root/'interrupted').glob('*/meta.json')))


def protocol(meta):
    from .agentic_learning import _signature
    return {'signature':_signature(meta),'T0':meta.get('T0'),
            'gripper':meta.get('config',{}).get('gripper')}


def _execution_overrides(value):
    """Only the operator-declared closure floor may differ from the old reference.

    This is a new frozen execution protocol, never a mutation of reference
    evidence or a permission to mix historical contact labels across protocols.
    """
    if not isinstance(value, dict) or set(value)-{'gripper'}:
        raise ValueError('execution_overrides permits only gripper.min_normalized')
    if not value:
        return {}
    gripper=value.get('gripper')
    if not isinstance(gripper,dict) or set(gripper)!={'min_normalized'}:
        raise ValueError('execution_overrides permits only gripper.min_normalized')
    floor=gripper['min_normalized']
    if type(floor) not in (int,float) or not np.isfinite(floor) or not 0<=floor<1:
        raise ValueError('execution_overrides.gripper.min_normalized must be finite and in [0,1)')
    return deepcopy(value)


def validate_config(c):
    if c.get('schema')!=SCHEMA: raise ValueError('expected a_skill_loop_v1')
    _execution_overrides(c.get('execution_overrides',{}))
    target=c.get('target_definition',{})
    if not all(isinstance(target.get(k),str) and target[k].strip() for k in ('id','description','normal','needs_recovery','completion')):
        raise ValueError('explicit task-specific target definition and label instructions required')
    for k in ('max_rounds','max_new_attempts','max_batch_attempts'):
        if type(c.get('budget',{}).get(k)) is not int or c['budget'][k]<1:raise ValueError('invalid budget '+k)
    if c.get('planner',{}).get('mode') not in ('llm','agent_evidence'):raise ValueError('choose llm or explicitly named evidence heuristic')
    authority=c.get('planner',{}).get('authority','legacy')
    if authority not in ('legacy','director'):raise ValueError('invalid experiment authority')
    if authority=='director':
        if c['planner']['mode']!='llm':raise ValueError('director authority requires an explicit LLM provider')
        limit=c['budget'].get('max_agent_decisions',64)
        if type(limit) is not int or limit<1:raise ValueError('invalid director decision budget')
    if c.get('action_axis','a')!='a':raise ValueError('only a-axis supported')
    from .skill_loop_feedback import acceptance_criteria
    from .learned_skill import _settings as state_settings
    from .contact_policy import _settings as action_settings
    c['acceptance']=acceptance_criteria(c.get('acceptance'))
    settings={**c.get('learning',{}),'target_definition_id':target['id']}
    state_settings(settings);action_settings(settings)
    if type(c.get('seed')) is not int or c['seed']<0:raise ValueError('invalid campaign seed')
    return c


def init_campaign(config_path,out,*,seed_sessions=(),reference_session=None):
    from .skill_collection import load_skill_config
    from .e1e2_core import load_settings, plan_trials, validate
    from .skill_exploration import validate_exploration_config
    config_path=Path(config_path).resolve();c=validate_config(yaml.safe_load(config_path.read_text()))
    parent=config_path.parent
    spec=load_skill_config(parent/c['skill_config']); base=load_settings((parent/c['skill_config']).parent/spec['base_config'])
    if spec['runtime'].get('action_axis','a')!='a':raise ValueError('only a-axis supported')
    ref=Path(reference_session).resolve() if reference_session else (parent/c['reference_session']).resolve()
    refmeta=read(ref/'session_meta.json')
    # An explicit existing reference supplies measured T0 and protocol, not proof of safety.
    if refmeta.get('simulated') or refmeta.get('T0') is None:raise ValueError('real measured reference session required')
    if refmeta.get('config',{}).get('axes_tcp')!=base.get('axes_tcp'):raise ValueError('reference axes mismatch')
    base=deepcopy(refmeta['config']);base.pop('recovery_skill',None)
    base['hardware_config']=str((parent/base['hardware_config']).resolve())
    floor=base.get('gripper',{}).get('min_normalized')
    if type(floor) not in (int,float) or not 0<=floor<base['gripper']['open_normalized']:raise ValueError('reference lacks an explicit closure floor')
    overrides=_execution_overrides(c.get('execution_overrides',{}))
    base['gripper'].update(overrides.get('gripper',{}))
    if base['gripper']['min_normalized']>=base['gripper']['open_normalized']:
        raise ValueError('execution closure floor must be below the reference open_normalized')
    validate(base)
    expected_protocol=protocol({**refmeta,'config':base})
    exploration=validate_exploration_config(yaml.safe_load((parent/c['exploration_config']).read_text()))
    if not np.isclose(exploration['variables']['probe_force']['reference'],base['force']['target']*base['force']['probe_fraction']):raise ValueError('probe force must remain fixed')
    bounds=exploration['variables']['a_mm']['range']
    if max(abs(v) for v in bounds)>base['motion']['max_grasp_offset_mm']:raise ValueError('exploration exceeds existing envelope')
    for name in ('initial_A','development_B','independent_B'):
        design=c[name]
        if any(not bounds[0]<=v<=bounds[1] for v in design['offsets_mm']) or 0 not in design['offsets_mm']:raise ValueError('plans require reference zero within fixed a envelope')
        candidate=deepcopy(base);candidate['design'].update(design);validate(candidate)
        if len(plan_trials(candidate,'A' if name=='initial_A' else 'B'))>c['budget']['max_batch_attempts']:raise ValueError(name+' exceeds max batch attempts')
    seeds=[]
    for root in seed_sessions:
        meta=read(Path(root)/'session_meta.json')
        role=meta.get('agentic',{}).get('role')
        if role not in ('exploration','development') or meta.get('cond') not in ('A','B'):raise ValueError('training seed must be exploration/development')
        if not complete_session(root):raise ValueError('seed session incomplete')
        if digest(protocol(meta))!=digest(expected_protocol):raise ValueError('seed protocol/T0/opening differs from execution protocol')
        seeds.append(source_snapshot(root))
    if len({s['path'] for s in seeds})!=len(seeds):raise ValueError('duplicate seed session')
    root=Path(out).resolve();root.mkdir(parents=True,exist_ok=False)
    c.update(code_snapshot=code_snapshot(),reference_session=str(ref),reference_snapshot=source_snapshot(ref),
             reference_protocol=protocol(refmeta),execution_protocol=expected_protocol,hardware_config_hash=digest(yaml.safe_load(Path(base['hardware_config']).read_text())),base_config=base,skill_spec=spec,exploration=exploration)
    c['learning']={**c.get('learning',{}),'target_definition_id':c['target_definition']['id']}
    immutable(root/'config.json',c)
    state={'schema':SCHEMA,'config_hash':digest(c),'round':1,'phase':'train' if seeds else 'collect_A',
           'training_sources':seeds,'sealed_evaluation_sources':[],'sealed_reports':[],'rounds':[],'new_attempts':0,'batch':None,'model':None,
           'created_unix_ns':time.time_ns(),'last_error':None}
    from .skill_director import enabled
    if enabled(c):
        state.update(phase='director',director_decisions=[],director_batch_seq=0,
                     training_revision=0,frozen_independent=False,pending_director_decision=None)
    atomic_json(root/'state.json',state)
    if not seeds and not enabled(c):_prepare_batch(root,state,c,'A')
    return state


def _save(root,state):atomic_json(root/'state.json',state)


def _load(root):
    root=Path(root).resolve();c=read(root/'config.json');s=read(root/'state.json')
    if digest(c)!=s['config_hash']:raise ValueError('frozen campaign config changed')
    if code_snapshot()!=c['code_snapshot']:raise ValueError('campaign implementation changed; start a new versioned campaign')
    if digest(yaml.safe_load(Path(c['base_config']['hardware_config']).read_text()))!=c['hardware_config_hash']:raise ValueError('hardware configuration changed')
    for src in s['training_sources']+s.get('sealed_evaluation_sources',[]):
        if source_snapshot(src['path'])!=src:raise ValueError('previous training evidence changed: '+src['path'])
    for artifact in s.get('sealed_reports',[]):
        if digest(read(artifact['path']))!=artifact['hash']:raise ValueError('sealed evaluation report changed')
    if source_snapshot(c['reference_session'])!=c['reference_snapshot']:raise ValueError('reference evidence changed')
    if s.get('model') and digest(read(root/s['model']['path']))!=s['model']['hash']:raise ValueError('frozen model artifact changed')
    return root,s,c


def _prepare_batch(root,s,c,kind,selection=None):
    from .e1e2_core import plan_trials
    from .skill_collection import selection_plan
    from .skill_director import enabled
    directory=(root/'batches'/f'{s["director_batch_seq"]:03d}') if enabled(c) else root/f'round_{s["round"]:03d}'
    directory.mkdir(parents=True,exist_ok=True)
    spec=deepcopy(c['skill_spec']);spec['base_config']='base.yaml';spec['target_definition']=deepcopy(c['target_definition'])
    spec['learning']=deepcopy(c['learning']);spec['learning_mode']='joint'
    spec['collection']=deepcopy(c[{'A':'initial_A','B':'development_B','independent':'independent_B'}[kind]])
    base=deepcopy(c['base_config']); base['design'].update(spec['collection'])
    plan=selection_plan(selection,base,'A' if kind=='A' else 'B') if selection else plan_trials(base,'A' if kind=='A' else 'B')
    if s['new_attempts']+len(plan)>c['budget']['max_new_attempts']:
        s['phase']='budget_exhausted';s['stop_reason']='insufficient remaining attempts for complete batch';_save(root,s);return
    if len(plan)>c['budget']['max_batch_attempts']:raise ValueError('batch exceeds frozen budget')
    spec_path=directory/(kind+'_skill.yaml'); base_path=directory/'base.yaml'
    if not base_path.exists():base_path.write_text(yaml.safe_dump(c['base_config'],sort_keys=False))
    if spec_path.exists():
        if yaml.safe_load(spec_path.read_text())!=spec:raise ValueError('batch specification changed during resume')
    else:
        with spec_path.open('x') as f:yaml.safe_dump(spec,f,sort_keys=False,allow_unicode=True)
    selection_path=None
    if selection is not None:
        selection_path=directory/(kind+'_selection.json')
        if selection_path.exists():
            if read(selection_path)!=selection:raise ValueError('selection changed during resume')
        else:immutable(selection_path,selection)
    s['batch']={'kind':kind,'session':str(directory/kind),'spec':str(spec_path),
                'spec_hash':digest(spec),'base_hash':digest(c['base_config']),'plan':plan,
                'selection':str(selection_path) if selection_path else None,
                'selection_hash':digest(selection) if selection else None,'counted_attempts':0}
    s['phase']='collect_'+kind;_save(root,s)


def _check_batch(s,c):
    b=s['batch'];spec=yaml.safe_load(Path(b['spec']).read_text())
    if digest(spec)!=b['spec_hash'] or digest(yaml.safe_load((Path(b['spec']).parent/'base.yaml').read_text()))!=b['base_hash']:raise ValueError('frozen batch configuration changed')
    if b['selection'] and digest(read(b['selection']))!=b['selection_hash']:raise ValueError('frozen collection selection changed')


def _consume(root,s,c):
    b=s['batch'];_check_batch(s,c)
    n=attempts(b['session']);s['new_attempts']+=max(0,n-b['counted_attempts']);b['counted_attempts']=n;_save(root,s)
    if not complete_session(b['session']):
        if s['new_attempts']>=c['budget']['max_new_attempts']:
            s['phase']='budget_exhausted';s['stop_reason']='physical trial attempt budget exhausted';_save(root,s)
        return False
    meta=read(Path(b['session'])/'session_meta.json')
    if meta.get('plan')!=b['plan']:raise ValueError('collected plan differs from frozen batch')
    if digest(protocol(meta))!=digest(c.get('execution_protocol',c['reference_protocol'])):raise ValueError('collected protocol/T0/opening changed')
    expected={'A':'exploration','B':'development','independent':'final_validation'}[b['kind']]
    if meta.get('agentic',{}).get('role')!=expected:raise ValueError('batch role mismatch')
    if b['kind']!='A' and digest(meta.get('fit'))!=s['model']['hash']:raise ValueError('B did not execute this frozen skill version')
    if b['kind']!='independent':s['training_sources'].append(source_snapshot(b['session']))
    else:s['sealed_evaluation_sources'].append(source_snapshot(b['session']))
    if b['kind']=='A':
        from .skill_director import enabled
        s['phase']='director' if enabled(c) else 'train'
    else:s['phase']='evaluate'
    _save(root,s);return True


def _verified_set_terminal(row,detail,target_id,expected_model_id,min_duration=.2):
    """Bind a successful set handoff to the actual saved terminal observation."""
    terminal=detail.get('terminal_exit') or {}
    if terminal.get('completed') is not True:return False
    achieved=terminal.get('achieved_state') or {}
    ts=achieved.get('timestamp_ns')
    try:
        features=np.asarray(achieved.get('contact_features'),dtype=float)
        if (type(ts) is not int or features.shape!=(4,) or not np.isfinite(features).all()
                or achieved.get('target_definition_id')!=target_id):return False
        for event in row.get('skill_observations',[]):
            observation=event.get('observation') or {};output=event.get('skill_outputs') or {}
            if observation.get('timestamp_ns')!=ts or output.get('model_id')!=expected_model_id:continue
            if output.get('exit')!=terminal or terminal.get('hand_back') is not True:continue
            ds=observation.get('descriptors') or []
            if len(ds)!=2 or any(not isinstance(d,dict) or d.get('valid') is not True or d.get('touches_edge',False) for d in ds):continue
            measured=np.asarray([d.get('centroid_px') for d in ds],dtype=float)
            if measured.shape!=(2,2) or not np.isfinite(measured).all():continue
            if (observation.get('bilateral') is True and observation.get('force_stable') is True and
                    observation.get('status')=='contact' and
                    type(observation.get('stable_duration_s')) in (int,float) and
                    np.isfinite(observation['stable_duration_s']) and
                    observation['stable_duration_s']>=min_duration and
                    (observation.get('context') or {}).get('target_definition_id')==target_id and
                    np.array_equal(measured.reshape(4),features)):
                return True
    except (TypeError,ValueError):
        return False
    return False


def evaluate_batch(session,*,role,expected_model_id):
    from .skill_offline import evaluate_skill_sessions
    report=evaluate_skill_sessions([session],role=role)
    report['session_complete']=complete_session(session)
    report['task_outcomes']=[]; counts={k:0 for k in ('TP','TN','FP','FN','unknown','target_count','not_target_count','n')}
    meta=read(Path(session)/'session_meta.json');target_id=meta['config'].get('recovery_skill',{}).get('target_definition',{}).get('id')
    report['target_definition_id']=target_id
    detail_by_path={d['trial_path']:d for d in report['details']}
    verified_labels=True
    for p in sorted((Path(session)/'trials').glob('*/meta.json')):
        row=read(p);trial_path=str(p.parent.resolve())
        if trial_path not in detail_by_path:continue
        truth=row.get('skill_ground_truth') or {}
        detail=detail_by_path[trial_path]
        if (meta.get('fit') or {}).get('schema')=='learned_a_skill_set_v1':
            min_duration=(meta.get('config',{}).get('recovery_skill',{}).get('runtime',{})
                          .get('target',{}).get('min_stable_duration_s',.2))
            detail['set_terminal_observation_verified']=_verified_set_terminal(
                row,detail,target_id,expected_model_id,min_duration)
        if not target_id or truth.get('target_definition_id')!=target_id:
            detail['label_source_verified']=False
            detail['initial_class']=None;detail['final_target_reached']=None
        report['task_outcomes'].append({**{k:row.get(k) for k in ('id','status','target_reached','lifted','hold_complete','dropped','label')},
            'trial_path':trial_path,'success':row.get('lifted') is True and row.get('hold_complete') is True and row.get('label')=='normal' and row.get('dropped') is not True})
        events={}
        for e in row.get('skill_observations',[]):
            output=e.get('skill_outputs') or {};obs=e.get('observation') or {}
            if output.get('model_id')==expected_model_id:
                ts=obs.get('timestamp_ns')
                if ts in events and events[ts].get('learned_conditions')!=output.get('learned_conditions'):raise ValueError('ambiguous per-window learned predictions')
                events[ts]=output
        seen={}
        for label in row.get('skill_window_labels',[]):
            ts=label.get('timestamp_ns');truth=label.get('target_reached');out=events.get(ts)
            if type(ts) is not int:verified_labels=False;continue
            if ts in seen:
                if seen[ts]!=label:raise ValueError('conflicting duplicate completion annotation')
                continue
            seen[ts]=label
            if (type(truth) is not bool or label.get('source')!='operator_review_of_observed_contact'
                    or label.get('independent_of_experimental_offset') is not True
                    or not target_id or label.get('target_definition_id')!=target_id):
                verified_labels=False;continue
            pred=((out or {}).get('learned_conditions') or {}).get('exit') or {}
            value=pred.get('positive') if pred.get('valid') is True else None
            counts['n']+=1;counts['target_count' if truth else 'not_target_count']+=1
            counts['unknown' if type(value) is not bool else ('T' if value==truth else 'F')+('P' if value else 'N')]+=1
        if set(events)-set(seen):verified_labels=False
    counts['label_source_verified']=verified_labels and counts['n']>0
    report['completion_validation']=counts
    return report


def _planner(root,s,c,report,model,*,allow_agent=False,provider=None):
    from .skill_loop_feedback import summarize_round,propose_collection
    from .skill_collection import selection_plan
    diagnosis=summarize_round(report,model)
    remaining=c['budget']['max_new_attempts']-s['new_attempts']
    directory=root/f'round_{s["round"]:03d}'
    if c['planner']['mode']=='agent_evidence':
        selection=propose_collection(report,c['exploration'],model=model,seed=c['seed']+s['round'],remaining_trials=min(remaining,c['budget']['max_batch_attempts']))
        audit={'backend':'deterministic_evidence_heuristic','diagnosis':diagnosis,'selection':selection}
    else:
        if not allow_agent:return None
        from .agentic_provider import AgentProvider
        supplied=provider is not None; provider=provider or AgentProvider(c['planner']['provider'])
        refs=report.get('evidence_refs') or [str(root/'state.json')]
        context={'planning_mode':'agent','task':'a-axis joint invocation/recovery/exit learning',
            'target_definition':c['target_definition'],'diagnosis':diagnosis,
            'training_data_gaps':model.get('data_gaps',[]),'stop':{'eligible':False},
            'valid_evidence_refs':refs,'references':{x:'recorded experiment evidence' for x in refs},
            'scope':c['exploration']['variables']['a_mm'],'remaining_trials':min(remaining,c['budget']['max_batch_attempts']),
            'bottle_blocks':['light','heavy'],'reference_monitor_interval':c['base_config']['design']['zero_every'],
            'constraints':'Only nonzero offsets must be unique; repeat zero cells where needed. Arrange cells so no more than reference_monitor_interval consecutive nonzero trials occur. A only; even reps >=2 per cell; include a=0; fixed l=0 and probe force; no safety/label/criteria changes',
            'approved_models':[]}
        try:answer=provider.propose(context)
        finally:
            if not supplied:provider.close()
        audit={'backend':'llm','context':context,'answer':answer}
        immutable(directory/f'agent_{time.time_ns()}.json',audit)
        if answer.get('action')=='pause':
            s['phase']='awaiting_agent';s['stop_reason']=answer.get('reason');_save(root,s);return None
        if answer.get('action')!='propose_batch':raise ValueError('agent cannot freeze or bypass numerical acceptance')
        from .agentic_planning import validate_response
        validate_response(answer)
        p=answer['plan']
        if p.get('cond')!='A' or p.get('gain')!=1 or p.get('model_id') is not None:raise ValueError('agent may only propose bounded A acquisition')
        if not p.get('evidence_refs') or not set(p['evidence_refs'])<=set(refs):raise ValueError('agent invented evidence references')
        lo,hi=c['exploration']['variables']['a_mm']['range']; selected=[]; seen=set()
        for cell in p['cells']:
            delta=cell['offset_mm']; reps=cell['reps']
            if type(delta) not in (int,float) or not np.isfinite(delta) or not lo<=delta<=hi or (delta!=0 and delta in seen):raise ValueError('invalid proposed a offset')
            if type(reps) is not int or reps<2 or reps%2:raise ValueError('invalid proposed repetition count')
            seen.add(delta);selected.append({'condition':{'a_mm':float(delta),'contact_height_l_mm':0.,'probe_force':c['base_config']['force']['target']*c['base_config']['force']['probe_fraction']},'reps':reps,'bottle_blocks':['light','heavy'],'purpose':'reference_repeat' if delta==0 else 'agent_hypothesis','reason':answer['reason'],'problem_refs':p['evidence_refs']})
        total=sum(x['reps'] for x in selected)
        if 0 not in seen or total>min(remaining,c['budget']['max_batch_attempts']):raise ValueError('agent proposal lacks reference or exceeds budget')
        selection={'schema':'skill_experiment_selection_v1','role':'development','stage':'a_closed_loop','status':'planned_unexecuted','selector':'llm','selected':selected,'planned_trial_attempts':total,'evidence_refs':refs,'hypothesis':p['hypothesis'],'expected_outcome':p['expected_outcome']}
        audit['selection']=selection
    if not selection.get('selected'):s['phase']='budget_exhausted';s['stop_reason']='no complete bounded reference-containing batch fits budget';_save(root,s);return None
    # Compile before round advance or writing the next batch. Invalid proposals
    # cannot strand the campaign in a partly created next round.
    checked_base=deepcopy(c['base_config']);checked_base['design'].update(c['initial_A'])
    selection_plan(selection,checked_base,'A')
    immutable(directory/f'feedback_{time.time_ns()}.json',audit)
    return selection


def advance(root,*,allow_agent=False,provider=None,trainer=None,evaluator=None):
    root,s,c=_load(root)
    if s['phase'] in TERMINAL:return s
    from . import skill_director
    if skill_director.enabled(c):
        return skill_director.advance(root,s,c,allow_agent=allow_agent,provider=provider,trainer=trainer,evaluator=evaluator)
    if s['phase'].startswith('collect_'):
        if not _consume(root,s,c):return s
    if s['phase']=='train':
        if trainer is None:
            from .learned_skill import train_learned_skill
            trainer=train_learned_skill
        directory=root/f'round_{s["round"]:03d}';directory.mkdir(exist_ok=True)
        path=directory/'model.json'
        # A crash after immutable model write can resume, but cannot substitute a model.
        request={'sources':s['training_sources'],'settings':c['learning']}
        request_path=directory/'training_request.json'
        if request_path.exists() and read(request_path)!=request:raise ValueError('training request changed')
        if not request_path.exists():immutable(request_path,request)
        if path.exists():model=read(path)
        else:
            model=trainer([x['path'] for x in s['training_sources']],c['learning'])
            immutable(path,model)
        s['model']={'path':str(path.relative_to(root)),'hash':digest(model),'model_id':model['model_id']}
        s['rounds'].append({'round':s['round'],'model_id':model['model_id'],'usable_for_B':model['usable_for_B'],'data_gaps':model.get('data_gaps',[]),'blocking_reasons':model.get('blocking_reasons',[])})
        if model['usable_for_B']:
            _prepare_batch(root,s,c,'B');return s
        s['phase']='feedback';s['feedback_report']={'role':'development','feedback_source':'training_data','evidence_refs':[str(path)],'gaps':model.get('blocking_reasons',[]),'data_gaps':model.get('data_gaps',[])};_save(root,s)
    if s['phase']=='evaluate':
        model=read(root/s['model']['path']);independent=s['batch']['kind']=='independent'
        report=(evaluator or evaluate_batch)(s['batch']['session'],role='independent' if independent else 'development',expected_model_id=model['model_id'])
        from .skill_loop_feedback import assess_acceptance
        result=assess_acceptance(report,{**c['acceptance'],'expected_role':'independent' if independent else 'development'},expected_model_id=model['model_id'],model=model)
        report_path=root/f'round_{s["round"]:03d}'/('independent_report.json' if independent else 'development_report.json')
        if report_path.exists():
            if read(report_path)!=report:raise ValueError('evaluation evidence changed')
        else:immutable(report_path,report)
        s['sealed_reports'].append({'path':str(report_path),'hash':digest(report)})
        s['rounds'][-1]['independent' if independent else 'development']=result
        if independent:
            s['phase']='complete' if result['passed'] else 'independent_failed'
            s['stop_reason']='frozen numerical criteria met on independent batch' if result['passed'] else 'independent evaluation failed; sealed evidence cannot guide this campaign';_save(root,s);return s
        if result['passed']:
            _prepare_batch(root,s,c,'independent');return s
        s['phase']='feedback';s['feedback_report']=report;_save(root,s)
    if s['phase'] in ('feedback','awaiting_agent'):
        if s['round']>=c['budget']['max_rounds'] or s['new_attempts']>=c['budget']['max_new_attempts']:
            s['phase']='budget_exhausted';s['stop_reason']='budget reached without verified completion';_save(root,s);return s
        model=read(root/s['model']['path'])
        selection=_planner(root,s,c,s['feedback_report'],model,allow_agent=allow_agent,provider=provider)
        if selection is not None:
            s['round']+=1;_prepare_batch(root,s,c,'A',selection)
        elif s['phase'] not in TERMINAL:s['phase']='awaiting_agent';_save(root,s)
    return s


def run(root,*,real=False,allow_agent=False,step=True,collector=None,**kwargs):
    if not real:raise ValueError('run requires explicit --real; inspect/status/advance never open hardware')
    root,s,c=_load(root)
    if c.get('physical_setup_confirmed') is not True:raise ValueError('confirm target definition, reference scene and white-bottle closure floor in a new campaign config before real run')
    if collector is None:
        from .skill_collection import main as collector
    while True:
        s=advance(root,allow_agent=allow_agent,**kwargs)
        from .skill_director import enabled
        if enabled(c) and s['phase']=='director':continue
        if not s['phase'].startswith('collect_'):return s
        _check_batch(s,c);b=s['batch']; n=attempts(b['session']);remaining=c['budget']['max_new_attempts']-s['new_attempts']-max(0,n-b['counted_attempts'])
        if remaining<=0:s['phase']='budget_exhausted';_save(root,s);return s
        args=['collect','--real','--config',b['spec'],'--cond','A' if b['kind']=='A' else 'B','--role',{'A':'exploration','B':'development','independent':'final_validation'}[b['kind']], '--mode','development','--reference-session',c['reference_session'],'--min-normalized',str(c['base_config']['gripper']['min_normalized']),'--out',b['session'],'--limit',str(remaining)]
        if step:args.append('--step')
        if b['selection']:args.extend(['--selection',b['selection']])
        if b['kind']!='A':args.extend(['--fit',str(root/s['model']['path'])])
        pending_labels=any(read(p).get('skill_label_pending') is True for p in (Path(b['session'])/'trials').glob('*/meta.json'))
        result=collector(args)
        if result not in (None,0):raise RuntimeError('collection stopped or failed; inspect saved recovery before resume')
        if pending_labels or not complete_session(b['session']):
            _consume(root,s,c);return s


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['init','status','advance','run']);p.add_argument('--out',type=Path,required=True);p.add_argument('--config',type=Path);p.add_argument('--seed-session',action='append',default=[]);p.add_argument('--reference-session');p.add_argument('--real',action='store_true');p.add_argument('--allow-agent',action='store_true');p.add_argument('--auto-stages',action='store_true',help='omit per-stage prompts; initial authorization and labels still required')
    a=p.parse_args(argv)
    if a.command!='run' and a.real:p.error('--real only permitted for run')
    if a.command=='run' and not a.real:p.error('run requires --real')
    if a.command=='init':
        if not a.config:p.error('init requires --config')
        s=init_campaign(a.config,a.out,seed_sessions=a.seed_session,reference_session=a.reference_session)
    else:
        with (a.out/'.loop.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            if a.command=='status':_,s,_=_load(a.out)
            else:
                try:s=run(a.out,real=True,allow_agent=a.allow_agent,step=not a.auto_stages) if a.command=='run' else advance(a.out,allow_agent=a.allow_agent)
                except Exception as e:
                    state=read(a.out/'state.json');state['last_error']={'type':type(e).__name__,'message':str(e),'time_ns':time.time_ns()};_save(a.out,state);raise
    print(json.dumps(s,ensure_ascii=False,indent=2));return 0
