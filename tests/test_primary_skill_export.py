"""Offline learned-campaign export and launch contracts; synthetic evidence only."""
from copy import deepcopy
import json
import os
import subprocess
import shutil
import sys
from pathlib import Path
from unittest.mock import patch

import yaml

from bottle_scan import skill_loop as loop
from bottle_scan.learned_skill import train_learned_skill
from bottle_scan.primary_skill_export import export_primary_skill, validate_export_config, _check_session
from bottle_scan.session import atomic_json
from tests.test_agentic_executor import cfg
from tests.test_learned_skill import LearnedSkillFixture, learned_settings, TARGET_ID, SOURCE
from tests.test_skill_loop_feedback import exploration_config


class PrimaryExportTests(LearnedSkillFixture):
    def campaign(self, *, state_set=False, closure_floor=None):
        base=cfg();base['force']['probe_stable_s']=.2
        base['axes_tcp']={'a':[1.,0.,0.],'l':[0.,0.,1.],'c':[0.,1.,0.]}
        target={'id':TARGET_ID,'description':'Synthetic observed target',
                'normal':'In target','needs_recovery':'Observed off target','completion':'Target reached'}
        if closure_floor is not None:
            self.make_learned('old_reference',simulated=False,patch_cfg=base)
            base=deepcopy(base);base['gripper']['min_normalized']=closure_floor
        A=self.make_learned('A',simulated=False,patch_cfg=base)
        meta=loop.read(A/'session_meta.json')
        meta['plan']=[{k:loop.read(p)[k] for k in ('id','cond','kind','bottle','repeat','delta_mm')}
                      for p in sorted((A/'trials').glob('*/meta.json'))]
        meta['config']['recovery_skill']={'target_definition':target}
        atomic_json(A/'session_meta.json',meta)
        learning=learned_settings();learning['simulate']=False
        if state_set:learning['target_mode']='acceptable_set'
        model=train_learned_skill([A],learning)
        self.assertTrue(model['usable_for_B'],model['blocking_reasons'])
        repo=Path(__file__).resolve().parents[1]
        (self.root/'config.ros.yaml').write_text((repo/'config.ros.yaml').read_text())
        (self.root/'base.yaml').write_text(yaml.safe_dump(base))
        design={'offsets_mm':[-5.,0.,5.],'reps':4,'seed':11}
        runtime={'action_axis':'a','required_context':{'task':'bottle','contact_protocol':'fixed_orientation_a','probe_target_sdk':1.5}}
        spec={'schema':'bottle_skill_config_v1','base_config':'base.yaml','runtime':runtime,'collection':design,'target_definition':target}
        (self.root/'skill.yaml').write_text(yaml.safe_dump(spec))
        (self.root/'exploration.yaml').write_text(yaml.safe_dump(exploration_config()))
        c={'schema':loop.SCHEMA,'skill_config':'skill.yaml','reference_session':'A','exploration_config':'exploration.yaml',
           'action_axis':'a','target_definition':target,'physical_setup_confirmed':True,'learning':learning,
           'planner':{'mode':'agent_evidence'},'budget':{'max_rounds':3,'max_new_attempts':100,'max_batch_attempts':32},
           'initial_A':design,'development_B':design,'independent_B':design,'acceptance':{},'seed':11}
        if closure_floor is not None:
            c['reference_session']='old_reference'
            c['execution_overrides']={'gripper':{'min_normalized':closure_floor}}
        (self.root/'campaign.yaml').write_text(yaml.safe_dump(c));campaign=self.root/'campaign'
        loop.init_campaign(self.root/'campaign.yaml',campaign,seed_sessions=[A])
        state=loop.advance(campaign,trainer=lambda sources,settings:model)
        self.assertEqual(state['phase'],'collect_B')
        for kind in ('B','independent'):
            self.fill_batch(campaign,target,model)
            state=loop.advance(campaign)
            self.assertEqual(state['phase'],'collect_independent' if kind=='B' else 'complete',state)
        template=self.root/'scheduler_template.json'
        atomic_json(template,{'schema':'starvla_contact_scheduler_v1','support_height_tolerance_mm':2.,
                             'support_lateral_tolerance_mm':2.,'support_angle_tolerance_deg':1.,'max_skill_s':90.})
        return campaign,template,model

    def fill_batch(self,campaign,target,model):
        state=loop.read(campaign/'state.json');c=loop.read(campaign/'config.json');batch=state['batch']
        session=Path(batch['session']);meta=loop.read(Path(c['reference_session'])/'session_meta.json')
        meta.update(cond='B',plan=batch['plan'],fit=model,config=deepcopy(c['base_config']),
                    agentic={'role':'development' if batch['kind']=='B' else 'final_validation'})
        meta['config']['recovery_skill']={'target_definition':target}
        atomic_json(session/'session_meta.json',meta)
        scan_index=0
        for item in batch['plan']:
            need=scan_index>=6
            if item['kind']=='scan':scan_index+=1
            invocation={'needs_recovery':need,'invoke':need,'permitted':True,'applicable':True,'candidate_supported':True}
            end={'hand_back':True,'completed':True,'status':'success','reason':'stable_target_contact_confirmed'}
            first={'model_id':model['model_id'],'invocation':invocation,
                   'exit':{'hand_back':False,'completed':False},
                   'learned_conditions':{'exit':{'positive':not need,'valid':True}},
                   'recovery':{'decision':{},'action':{'axis':'a','delta_mm':-2.} if need else None}}
            final={**deepcopy(first),'exit':end,'learned_conditions':{'exit':{'positive':True,'valid':True}}}
            shared={'source':SOURCE,'target_definition_id':target['id'],'independent_of_experimental_offset':True}
            row={**item,'complete':True,'recording_complete':True,'label':'normal','status':'hold_complete',
                 'lifted':True,'hold_complete':True,'dropped':False,'bilateral':True,
                 'skill_ground_truth':{**shared,'initial_class':'needs_recovery' if need else 'already_normal','final_target_reached':True},
                 'skill_handoff':end,'skill_outputs':final,'skill_observations':[
                    {'phase':'first_touch','observation':{'timestamp_ns':100},'skill_outputs':first},
                    {'phase':'completion','observation':{'timestamp_ns':200},'skill_outputs':final}],
                 'skill_window_labels':[{**shared,'timestamp_ns':100,'target_reached':not need},
                                        {**shared,'timestamp_ns':200,'target_reached':True}],
                 'regrasps':int(need),'correction_history':[],'probe_regrasps':[]}
            if model.get('target_mode')=='acceptable_set':
                from bottle_scan.acceptable_contact_set import match_features
                anchor=model['acceptable_set']['anchors'][(scan_index-1)%len(model['acceptable_set']['anchors'])]
                features=anchor['features']
                achieved={**match_features(model['acceptable_set'],features),
                          'target_definition_id':target['id'],'timestamp_ns':200,'contact_features':features}
                end['achieved_state']=achieved
                row['skill_observations'][-1]['observation'].update(
                    descriptors=[{'valid':True,'touches_edge':False,'centroid_px':features[:2]},
                                 {'valid':True,'touches_edge':False,'centroid_px':features[2:]}],
                    bilateral=True,force_stable=True,stable_duration_s=.2,status='contact',
                    context={'target_definition_id':target['id']})
            if need:
                row['correction_history']=[{'delta_mm':-2.,'applied_delta_mm':-2.,'blocked':False,'skill_decision':first}]
                row['probe_regrasps']=[{'receipt':{'applied':True,'action_mm':-2.,'completed_ns':150},'probe':{'timestamp_ns':200}}]
            atomic_json(session/'trials'/item['id']/'meta.json',row)
            atomic_json(session/'trials'/item['id']/'recording_status.json',{'complete':True})

    def test_complete_campaign_exports_exact_three_head_model_and_primary_contract(self):
        campaign,template,model=self.campaign();out=self.root/'export'
        with patch('bottle_scan.agentic_executor.execute_batch',side_effect=AssertionError('hardware forbidden')):
            result=export_primary_skill(campaign,out,scheduler_template=template)
        self.assertFalse(result['hardware_started']);self.assertFalse(result['primary_task_validated'])
        self.assertEqual(result['model_id'],model['model_id'])
        config=loop.read(out/'scheduler.json')
        self.assertEqual(config['target_definition']['id'],TARGET_ID)
        self.assertTrue(config['require_primary_resume_contract'])
        self.assertEqual(loop.read(out/'skill/model.json'),model)
        self.assertEqual(config['reference_session_original_path'],str(self.root/'A'))
        self.assertEqual(config['execution_config']['gripper']['min_normalized'],.336)
        with self.assertRaises(FileExistsError):export_primary_skill(campaign,out,scheduler_template=template)

    def test_new_closure_floor_exports_execution_source_and_keeps_original_reference(self):
        campaign,template,model=self.campaign(closure_floor=.330)
        config=loop.read(campaign/'config.json')
        self.assertEqual(config['reference_protocol']['gripper']['min_normalized'],.336)
        self.assertEqual(config['execution_protocol']['gripper']['min_normalized'],.330)
        out=self.root/'new-floor-export'
        result=export_primary_skill(campaign,out,scheduler_template=template)
        self.assertEqual(result['model_id'],model['model_id'])
        scheduler=loop.read(out/'scheduler.json')
        self.assertEqual(scheduler['execution_config']['gripper']['min_normalized'],.330)
        self.assertEqual(scheduler['reference_session_original_path'],str(self.root/'A'))
        source=loop.read(out/'reference_session_meta.json')
        self.assertEqual(source['config']['gripper']['min_normalized'],.330)
        manifest=loop.read(out/'primary_export.json')
        self.assertEqual(manifest['campaign_reference']['path'],str(self.root/'old_reference'))
        self.assertEqual(loop.read(self.root/'old_reference/session_meta.json')['config']['gripper']['min_normalized'],.336)

    def test_old_floor_B_cannot_validate_new_execution_protocol(self):
        campaign,_,model=self.campaign(closure_floor=.330)
        config=loop.read(campaign/'config.json');session=campaign/'round_001/B'
        self.assertEqual(_check_session(session,config,model,role='development')['config']['gripper']['min_normalized'],.330)
        path=session/'session_meta.json';meta=loop.read(path)
        meta['config']['gripper']['min_normalized']=.336;atomic_json(path,meta)
        with self.assertRaisesRegex(ValueError,'execution protocol'):
            _check_session(session,config,model,role='development')

    def test_acceptable_set_export_preserves_contract_and_rejects_tampered_identity(self):
        campaign,template,model=self.campaign(state_set=True);out=self.root/'export'
        result=export_primary_skill(campaign,out,scheduler_template=template)
        contract=result['target_state_contract']
        self.assertEqual(contract['set_id'],model['acceptable_set']['set_id'])
        self.assertEqual(contract['anchor_count'],len(model['acceptable_set']['anchors']))
        self.assertIn('not the sole completion state',contract['reference_T0_role'])
        self.assertEqual(loop.read(out/'scheduler.json')['target_state_contract'],contract)
        manifest=loop.read(out/'primary_export.json')
        manifest['target_state_contract']['set_id']='forged-set'
        atomic_json(out/'primary_export.json',manifest)
        with self.assertRaisesRegex(ValueError,'acceptable-state contract'):
            validate_export_config(out/'scheduler.json')

    def test_export_loads_in_actual_separate_scheduler_runtime_offline(self):
        repo=Path(__file__).resolve().parents[1]
        candidates=[repo.parent/'agent_director_scheduler_20261110',repo.parent/'scheduler',repo.parent/'contact_scheduler_20261104_experiment_b_v4']
        scheduler=next((p for p in candidates if (p/'runtime/bottle_scan/contact_recovery_backend.py').is_file()),None)
        if scheduler is None:self.skipTest('separate scheduler checkout unavailable')
        campaign,template,model=self.campaign();out=self.root/'export'
        export_primary_skill(campaign,out,scheduler_template=template)
        env={k:v for k,v in os.environ.items() if not k.startswith('CONTACT_')}
        env['PYTHONPATH']=str(scheduler/'runtime')
        code=('from bottle_scan.contact_scheduler import load_config; '
              'from bottle_scan.contact_skill_backend import validate_config; '
              'import sys,json; print(json.dumps(validate_config(load_config(sys.argv[1]))))')
        result=subprocess.run([sys.executable,'-c',code,str(out/'scheduler.json')],cwd=scheduler,
                              env=env,text=True,capture_output=True,timeout=30)
        self.assertEqual(result.returncode,0,result.stderr)
        report=json.loads(result.stdout)
        self.assertEqual(report['model_id'],model['model_id'])
        self.assertEqual(report['conditions_source'],'learned_invocation_and_exit_heads')
        self.assertTrue(report['requires_primary_resume_contract'])

    def director_campaign(self):
        """Repackage independently evaluated fixture into director artifact layout."""
        from bottle_scan.skill_loop_feedback import assess_acceptance
        campaign,template,model=self.campaign()
        config=loop.read(campaign/'config.json');state=loop.read(campaign/'state.json')
        config['planner']={'mode':'llm','authority':'director'}
        state['config_hash']=loop.digest(config);atomic_json(campaign/'config.json',config)
        destination=campaign/'models/001/model.json';destination.parent.mkdir(parents=True)
        shutil.move(campaign/state['model']['path'],destination)
        state['model']['path']=str(destination.relative_to(campaign))
        state['sealed_reports']=[]
        for index,(role,folder,key) in enumerate((('development','B','current_development'),
                                                ('independent','independent','independent_evaluation')),1):
            old=campaign/'round_001'/folder;session=campaign/'batches'/str(index)/folder
            session.parent.mkdir(parents=True);shutil.move(old,session)
            collection='training_sources' if role=='development' else 'sealed_evaluation_sources'
            state[collection]=[x for x in state[collection] if x['path']!=str(old)]
            state[collection].append(loop.source_snapshot(session))
            report=loop.evaluate_batch(session,role=role,expected_model_id=model['model_id'])
            path=campaign/'reports'/str(index)/(role+'_report.json');atomic_json(path,report)
            result=assess_acceptance(report,{**config['acceptance'],'expected_role':role},
                                     expected_model_id=model['model_id'],model=model)
            self.assertTrue(result['passed'],result)
            state['sealed_reports'].append({'path':str(path),'hash':loop.digest(report)})
            state['rounds'][-1][role]=result
            state[key]={'session':str(session),'report_path':str(path),'model_id':model['model_id'],
                        'model_hash':state['model']['hash'],'result':result}
            if role=='independent':state['batch']['session']=str(session)
        audit={'schema':'skill_director_decision_v1','index':1,'context':{},
               'answer':{'action':'complete'},'validation':{'accepted':True},
               'model_id':model['model_id'],'model_hash':state['model']['hash']}
        path=campaign/'decisions/0001.json';atomic_json(path,audit)
        reference={'path':str(path),'hash':loop.digest(audit)}
        state['director_decisions']=[reference];state['sealed_reports'].append(reference)
        state['completion_decision']={**reference,'model_id':model['model_id'],'model_hash':state['model']['hash']}
        atomic_json(campaign/'state.json',state)
        return campaign,template,model

    def test_director_layout_exports_frozen_evidence_and_requires_agent_completion(self):
        campaign,template,model=self.director_campaign()
        result=export_primary_skill(campaign,self.root/'export',scheduler_template=template)
        self.assertEqual(result['model_id'],model['model_id'])
        state=loop.read(campaign/'state.json');state.pop('completion_decision')
        atomic_json(campaign/'state.json',state)
        with self.assertRaisesRegex(ValueError,'sealed decision'):
            export_primary_skill(campaign,self.root/'missing-decision-export',scheduler_template=template)

    def test_director_export_rejects_evidence_from_an_older_model(self):
        campaign,template,_=self.director_campaign();state=loop.read(campaign/'state.json')
        state['current_development']['model_id']='previous-model'
        atomic_json(campaign/'state.json',state)
        with self.assertRaisesRegex(ValueError,'differs from the frozen model'):
            export_primary_skill(campaign,self.root/'export',scheduler_template=template)

    def test_forged_complete_state_cannot_replace_real_accepted_reports(self):
        campaign,template,_=self.campaign()
        state=loop.read(campaign/'state.json');state['rounds'][-1]['independent']['metrics']['verified_recoveries']=999
        atomic_json(campaign/'state.json',state)
        with self.assertRaisesRegex(ValueError,'reproducible'):
            export_primary_skill(campaign,self.root/'export',scheduler_template=template)
        self.assertFalse((self.root/'export').exists())

    def test_mutated_source_evidence_or_model_cannot_export(self):
        campaign,template,_=self.campaign()
        p=campaign/'round_001/independent/trials/00001/meta.json'
        row=loop.read(p);row['skill_ground_truth']['final_target_reached']=False;atomic_json(p,row)
        with self.assertRaisesRegex(ValueError,'evidence changed'):
            export_primary_skill(campaign,self.root/'export',scheduler_template=template)

    def test_incomplete_campaign_and_post_export_artifact_changes_are_rejected(self):
        campaign,template,_=self.campaign();state=loop.read(campaign/'state.json')
        state['phase']='awaiting_agent';atomic_json(campaign/'state.json',state)
        with self.assertRaisesRegex(ValueError,'completed independent'):
            export_primary_skill(campaign,self.root/'export',scheduler_template=template)
        state['phase']='complete';atomic_json(campaign/'state.json',state)
        export_primary_skill(campaign,self.root/'export',scheduler_template=template)
        p=self.root/'export/hardware.yaml';p.write_text(p.read_text()+'\nchanged: true\n')
        with self.assertRaisesRegex(ValueError,'artifact changed'):
            validate_export_config(self.root/'export/scheduler.json')

    def test_export_manifest_cannot_promote_failure_by_only_flipping_passed(self):
        campaign,template,_=self.campaign();out=self.root/'export'
        export_primary_skill(campaign,out,scheduler_template=template)
        manifest=loop.read(out/'primary_export.json')
        manifest['acceptance']['independent']['metrics']['verified_recoveries']=999
        atomic_json(out/'primary_export.json',manifest)
        with self.assertRaisesRegex(ValueError,'acceptance invalid'):validate_export_config(out/'scheduler.json')
