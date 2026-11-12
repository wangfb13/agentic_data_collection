"""Export a completed a-axis campaign for the existing primary-policy scheduler.

Pure file validation and packaging. No hardware, subprocess, account or network
API is imported. A pilot pass is not a deployment reliability certificate.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np
import yaml

from . import skill_loop as loop
from .learned_skill import LearnedSkillModel
from .recovery_skill import RecoverySkill, load_skill_bundle
from .skill_loop_feedback import assess_acceptance

SCHEMA = 'a_skill_primary_export_v1'
DEFAULT_TEMPLATE = Path('/home/dm/DM_das/contact_scheduler_20261104_experiment_b_v4/contact_scheduler.skill_a.right_x5_20261107.json')


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _json(path, value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')


def _in_root(root, value):
    path=Path(value).resolve()
    if not path.is_relative_to(root.resolve()):raise ValueError('export artifact escaped its version directory')
    return path


def _check_session(session, config, model, *, role):
    session=Path(session).resolve();meta=loop.read(session/'session_meta.json')
    if not loop.complete_session(session) or meta.get('simulated') is not False or meta.get('cond')!='B':
        raise ValueError('completed real B evidence required for primary export')
    expected='final_validation' if role=='independent' else 'development'
    if meta.get('agentic',{}).get('role')!=expected:raise ValueError('B evidence role mismatch')
    if loop.digest(meta.get('fit'))!=loop.digest(model):raise ValueError('B did not execute the exact exported model')
    if loop.digest(loop.protocol(meta))!=loop.digest(config.get('execution_protocol',config['reference_protocol'])):
        raise ValueError('B protocol differs from frozen campaign execution protocol')
    definition=meta.get('config',{}).get('recovery_skill',{}).get('target_definition')
    if definition!=config['target_definition']:raise ValueError('B target definition differs from campaign')
    return meta


def _campaign_evaluation_layout(root, state, config, model):
    """Resolve frozen evidence without assuming the director used fixed folders."""
    directory=root/f'round_{state["round"]:03d}'
    if config.get('planner',{}).get('authority')!='director':
        if _in_root(root,root/state['model']['path'])!=directory/'model.json':
            raise ValueError('model is not the final frozen round version')
        return {role:{'session':str(directory/folder),
                      'report_path':str(directory/(role+'_report.json')),
                      'result':state['rounds'][-1].get(role)}
                for role,folder in (('development','B'),('independent','independent'))}
    completion=state.get('completion_decision') or {}
    reference={'path':completion.get('path'),'hash':completion.get('hash')}
    if reference not in state.get('director_decisions',[]):
        raise ValueError('director completion lacks a sealed decision')
    path=_in_root(root,reference['path']);audit=loop.read(path)
    if (loop.digest(audit)!=reference['hash'] or
            audit.get('schema')!='skill_director_decision_v1' or
            audit.get('answer',{}).get('action')!='complete' or
            audit.get('validation',{}).get('accepted') is not True or
            any(value!=model['model_id'] for value in (audit.get('model_id'),completion.get('model_id'))) or
            any(value!=state['model']['hash'] for value in (audit.get('model_hash'),completion.get('model_hash')))):
        raise ValueError('director completion decision differs from the frozen model')
    layout={'development':state.get('current_development') or {},
            'independent':state.get('independent_evaluation') or {}}
    for role,evidence in layout.items():
        if (evidence.get('model_id')!=model['model_id'] or
                evidence.get('model_hash')!=state['model']['hash'] or
                not evidence.get('session') or not evidence.get('report_path')):
            raise ValueError('director '+role+' evidence differs from the frozen model')
        _in_root(root,evidence['session']);_in_root(root,evidence['report_path'])
    return layout


def _target_state_contract(model):
    """Publish target semantics together with the exact immutable model."""
    if model.get('target_mode')!='acceptable_set':
        return None
    target=model.get('acceptable_set') or {}
    if (target.get('schema')!='acceptable_contact_set_v1' or target.get('usable') is not True or
            target.get('target_definition_id')!=model.get('target_definition_id') or
            not target.get('set_id') or not target.get('anchors')):
        raise ValueError('acceptable contact set is missing or not usable')
    return {'mode':'acceptable_set','target_definition_id':model['target_definition_id'],
            'set_id':target['set_id'],'anchor_count':len(target['anchors']),
            'completion':'fresh stable observed contact supported by the learned acceptable set',
            'reference_T0_role':'coordinate reference and motion bounds; not the sole completion state',
            'primary_task_validated':False}


def _completed_evidence(root):
    root,state,config=loop._load(root)
    if state.get('phase')!='complete':raise ValueError('primary export requires a completed independent B campaign')
    if config.get('physical_setup_confirmed') is not True:raise ValueError('campaign physical setup was not confirmed')
    if not state.get('model') or not state.get('rounds'):raise ValueError('completed state lacks frozen model and round evidence')
    model_path=_in_root(root,root/state['model']['path'])
    model=loop.read(model_path);LearnedSkillModel(model)
    if model.get('simulated') is not False:raise ValueError('simulated skill cannot be exported for the real primary policy')
    if model.get('target_definition_id')!=config['target_definition']['id']:
        raise ValueError('model target definition differs from campaign')
    layout=_campaign_evaluation_layout(root,state,config,model)
    loop._check_batch(state,config)
    if (state['batch']['kind']!='independent' or
            Path(state['batch']['session']).resolve()!=Path(layout['independent']['session']).resolve()):
        raise ValueError('completed campaign does not end in its frozen independent batch')
    training_paths={x['path'] for x in state['training_sources']}
    final_paths={x['path'] for x in state.get('sealed_evaluation_sources',[])}
    if training_paths & final_paths:raise ValueError('independent evidence leaked into training sources')
    if layout['development']['session'] not in training_paths or layout['independent']['session'] not in final_paths:
        raise ValueError('missing sealed development/independent source evidence')
    round_record=state['rounds'][-1]
    if round_record.get('model_id')!=model['model_id'] or round_record.get('round')!=state['round']:
        raise ValueError('final round model identity mismatch')
    reports={};acceptances={}
    sealed={x['path']:x['hash'] for x in state.get('sealed_reports',[])}
    for role,evidence in layout.items():
        session=Path(evidence['session']);meta=_check_session(session,config,model,role=role)
        if role=='independent' and meta.get('plan')!=state['batch']['plan']:
            raise ValueError('independent plan changed')
        path=Path(evidence['report_path'])
        report=loop.evaluate_batch(session,role=role,expected_model_id=model['model_id'])
        if sealed.get(str(path))!=loop.digest(report) or loop.read(path)!=report:
            raise ValueError('sealed '+role+' report differs from raw evidence')
        acceptance=assess_acceptance(report,{**config['acceptance'],'expected_role':role},
                                     expected_model_id=model['model_id'],model=model)
        if (acceptance['passed'] is not True or round_record.get(role)!=acceptance or
                evidence.get('result')!=acceptance):
            raise ValueError(role+' acceptance is not independently reproducible from raw evidence')
        reports[role]=report;acceptances[role]=acceptance
    # The deployed reference must prove the trained execution protocol as well
    # as geometry. The original T0 reference can have an older closure floor;
    # its immutable provenance remains in config['reference_snapshot'].
    source=None;source_path=None
    for path in model.get('source_sessions',[]):
        if path not in training_paths:continue
        meta=loop.read(Path(path)/'session_meta.json')
        if meta.get('simulated') is False and loop.digest(loop.protocol(meta))==loop.digest(config.get('execution_protocol',config['reference_protocol'])):
            source=meta;source_path=path;break
    if source is None:raise ValueError('no eligible training source proves the exported reference geometry')
    if not np.allclose(model.get('reference_T0'),source.get('T0'),atol=1e-7,rtol=0):
        raise ValueError('learned skill reference differs from measured source')
    return root,state,config,model,source,source_path,reports,acceptances


def export_primary_skill(campaign,out,*,scheduler_template=DEFAULT_TEMPLATE):
    out=Path(out).expanduser().resolve()
    if out.exists():raise FileExistsError('primary skill exports are immutable; choose a fresh directory')
    root,state,config,model,source,source_path,reports,acceptances=_completed_evidence(campaign)
    template=loop.read(Path(scheduler_template).expanduser().resolve())
    if template.get('schema')!='starvla_contact_scheduler_v1':raise ValueError('unsupported scheduler template')
    execution=deepcopy(config['base_config']);execution.pop('recovery_skill',None)
    runtime=deepcopy(config['skill_spec'].get('runtime',{}))
    required={'task':'bottle','contact_protocol':'fixed_orientation_a',
              'probe_target_sdk':execution['force']['target']*execution['force']['probe_fraction']}
    if runtime.get('required_context',required)!=required:raise ValueError('unsupported primary contact context')
    runtime['required_context']=required
    skill=RecoverySkill(model,execution['servo'],runtime,mode='development')
    if skill.config['action_axis']!='a':raise ValueError('primary export currently supports a axis only')
    hw_path=Path(execution['hardware_config'])
    hw_bytes=hw_path.read_bytes()
    if loop.digest(yaml.safe_load(hw_bytes))!=config['hardware_config_hash']:
        raise ValueError('campaign hardware configuration changed during export')
    execution['hardware_config']=str(out/'hardware.yaml')
    template.pop('hardware_rebind',None)
    target_state=_target_state_contract(model)
    template.pop('target_state_contract',None)
    if target_state is not None:template['target_state_contract']=target_state
    template.update(model_path=str(out/'skill/model.json'),recovery_skill_bundle=str(out/'skill'),
        reference_session_meta=str(out/'reference_session_meta.json'),
        reference_session_original_path=source_path,
        execution_config=execution,target_definition=deepcopy(config['target_definition']),
        require_primary_resume_contract=True,recovery_skill_mode='development',
        skill_after='handoff',skill_recovery='enabled',
        max_regrasps=skill.config['max_regrasps'],source_campaign=str(root),source_policy_id=model['model_id'],
        hardware_config_sha256=hashlib.sha256(hw_bytes).hexdigest(),
        primary_skill_export={'schema':SCHEMA,'manifest_file':str(out/'primary_export.json'),
            'model_id':model['model_id'],'target_definition_id':model['target_definition_id'],
            'independent_pilot_passed':True,'deployment_certificate_issued':False})
    out.parent.mkdir(parents=True,exist_ok=True)
    stage=Path(tempfile.mkdtemp(prefix='.'+out.name+'.',dir=out.parent))
    try:
        skill.save_bundle(stage/'skill')
        _json(stage/'reference_session_meta.json',source)
        template['reference_session_meta_sha256']=file_hash(stage/'reference_session_meta.json')
        (stage/'hardware.yaml').write_bytes(hw_bytes)
        for role,report in reports.items():_json(stage/(role+'_report.json'),report)
        _json(stage/'scheduler.json',template)
        files={str(p.relative_to(stage)):file_hash(p) for p in sorted(stage.rglob('*')) if p.is_file()}
        manifest={'schema':SCHEMA,'model_id':model['model_id'],'target_definition_id':model['target_definition_id'],
            'source_campaign':str(root),'campaign_config_hash':state['config_hash'],
            'campaign_reference':config['reference_snapshot'],'deployed_reference_source':source_path,
            'source_model_hash':state['model']['hash'],'acceptance':acceptances,
            'files_sha256':files,'runtime_mode':'development','requires_explicit_real_run':True,
            'certificate_issued':False,'hardware_started':False,
            'interpretation':'Frozen independent pilot evidence; primary full-task success still requires prospective observation.'}
        if target_state is not None:manifest['target_state_contract']=target_state
        _json(stage/'primary_export.json',manifest)
        if out.exists():raise FileExistsError('export destination appeared during packaging')
        os.rename(stage,out)
    except BaseException:
        shutil.rmtree(stage,ignore_errors=True);raise
    return validate_export_config(out/'scheduler.json')


def validate_export_config(path):
    """Read-only validation before spawning the existing scheduler launcher."""
    path=Path(path).expanduser().resolve();root=path.parent;config=loop.read(path)
    marker=config.get('primary_skill_export') or {}
    if config.get('schema')!='starvla_contact_scheduler_v1' or marker.get('schema')!=SCHEMA:
        raise ValueError('--skill-config requires an explicit completed joint skill export')
    manifest_path=_in_root(root,marker['manifest_file'])
    if manifest_path!=root/'primary_export.json':raise ValueError('unexpected primary export manifest location')
    manifest=loop.read(manifest_path)
    if manifest.get('schema')!=SCHEMA or manifest.get('runtime_mode')!='development':
        raise ValueError('invalid primary export manifest')
    files=manifest.get('files_sha256') or {}
    required={'scheduler.json','skill/model.json','skill/skill.json','reference_session_meta.json',
              'hardware.yaml','development_report.json','independent_report.json'}
    if set(files)!=required or path.name!='scheduler.json':raise ValueError('primary export file set is incomplete')
    for name,expected in files.items():
        artifact=_in_root(root,root/name)
        if file_hash(artifact)!=expected:raise ValueError('primary export artifact changed: '+name)
    expected_paths={'model_path':'skill/model.json','recovery_skill_bundle':'skill','reference_session_meta':'reference_session_meta.json'}
    for key,name in expected_paths.items():
        if _in_root(root,config[key])!=root/name:raise ValueError('primary export '+key+' points to another version')
    if _in_root(root,config['execution_config']['hardware_config'])!=root/'hardware.yaml':
        raise ValueError('primary export hardware path points to another version')
    model=loop.read(root/'skill/model.json');LearnedSkillModel(model)
    if model.get('simulated') is not False or loop.digest(model)!=manifest.get('source_model_hash'):
        raise ValueError('primary model is not the exact real validated artifact')
    definition=config.get('target_definition') or {}
    target_state=_target_state_contract(model)
    if (config.get('target_state_contract')!=target_state or
            manifest.get('target_state_contract')!=target_state):
        raise ValueError('primary acceptable-state contract differs from model')
    if any(v!=model['model_id'] for v in (manifest.get('model_id'),marker.get('model_id'),config.get('source_policy_id'))):
        raise ValueError('primary model identity mismatch')
    if any(v!=model['target_definition_id'] for v in (manifest.get('target_definition_id'),marker.get('target_definition_id'),definition.get('id'))):
        raise ValueError('primary target definition mismatch')
    if config.get('require_primary_resume_contract') is not True or config.get('recovery_skill_mode')!='development':
        raise ValueError('primary resume contract or explicit runtime mode missing')
    for role in ('development','independent'):
        report=loop.read(root/(role+'_report.json'));record=manifest.get('acceptance',{}).get(role) or {}
        current=assess_acceptance(report,record.get('criteria'),expected_model_id=model['model_id'],model=model)
        if current.get('passed') is not True or current!=record or report.get('role')!=role:
            raise ValueError('exported '+role+' acceptance invalid')
    recovery=load_skill_bundle(root/'skill',mode='development')
    if recovery.model!=model or recovery.servo!=config['execution_config']['servo']:
        raise ValueError('scheduler and skill bundle differ')
    source=loop.read(root/'reference_session_meta.json')
    if source.get('simulated') is not False or not np.allclose(model['reference_T0'],source.get('T0'),atol=1e-7,rtol=0):
        raise ValueError('primary reference mismatch')
    return {'schema':SCHEMA,'scheduler_config':str(path),'model_id':model['model_id'],
            'target_definition_id':model['target_definition_id'],'independent_pilot_passed':True,
            'runtime_mode':'development','primary_task_validated':False,'hardware_started':False,
            'target_state_contract':target_state}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('export');p.add_argument('--campaign',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--scheduler-template',type=Path,default=DEFAULT_TEMPLATE)
    p=sub.add_parser('check');p.add_argument('--config',type=Path,required=True)
    args=parser.parse_args(argv)
    result=export_primary_skill(args.campaign,args.out,scheduler_template=args.scheduler_template) if args.command=='export' else validate_export_config(args.config)
    print(json.dumps(result,ensure_ascii=False,indent=2));return 0
