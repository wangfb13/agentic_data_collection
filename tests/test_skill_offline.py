import json
from pathlib import Path
import tempfile
import unittest

from bottle_scan.skill_offline import (split_observations, choose_batch,
    observation_from_probe, cost_record, stage_training, run)


class OfflineTests(unittest.TestCase):
    def samples(self):
        return [{'session': f's{s}', 'group': f's{s}|b|{r}', 'trial_path': f's{s}/{r}/{i}',
                 'planned_delta_mm': float(i), 'x_mm': float(i)}
                for s in range(5) for r in range(4) for i in range(3)]

    def test_split_no_frame_or_group_leakage(self):
        obs=self.samples()
        roles,method=split_observations(obs, 7)
        self.assertEqual(roles, split_observations(obs,7)[0])
        for key in {o['group'] for o in obs}:
            self.assertEqual(len({roles[o['trial_path']] for o in obs if o['group']==key}),1)
        self.assertEqual(set(roles.values()), {'pool','development','independent'})

    def test_fixed_ignores_feedback_and_models(self):
        pool=self.samples()
        a,_=choose_batch(pool,pool[:3],10,'fixed',[],None,3,2.)
        b,_=choose_batch(pool,pool[:3],10,'fixed',[{'kind':'scan','delta_mm':99}],{},3,2.)
        self.assertEqual(a,b)
        self.assertEqual(len({o['trial_path'] for o in a}),10)

    def test_three_modes_same_budget_unique_trials(self):
        pool=self.samples()
        for mode in ('fixed','simple_feedback','agent_evidence'):
            chosen,_=choose_batch(pool,pool[:3],10,mode,[],None,3,2.)
            self.assertEqual(len(chosen),10)
            self.assertTrue(not ({o['trial_path'] for o in chosen} & {o['trial_path'] for o in pool[:3]}))

    def test_runtime_observation_excludes_hidden_truth(self):
        row={'delta_mm':88,'bottle':'secret','initial_status':'contact'}
        p={'valid':True,'timestamp_ns':30,'stable_start_ns':10,
           'descriptors':[{'valid':True,'contact':True},{'valid':True,'contact':True}],
           'measured_tcp_mm':[[42]],'normal_sdk':[1,1]}
        o=observation_from_probe(row,p)
        self.assertNotIn('delta_mm',o)
        self.assertNotIn('measured_tcp_mm',o)
        self.assertNotIn('bottle',o)
        self.assertIsNone(o['force_stable'])

    def test_unknown_cost_is_not_zero_and_no_lift_is_failed(self):
        row={'trial_path':'p','session':'s','metadata_sha256':'abc','kind':'scan',
             'label':'normal','complete':True,'lifted':False}
        c=cost_record(row)
        self.assertTrue(c['failed'])
        self.assertIsNone(c['robot_motion_seconds'])
        self.assertIsNone(c['reset_seconds'])
        self.assertIsNone(c['recorded_stage_seconds'])

    def test_no_overwrite_frozen_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp,'existing.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'fresh output'):
                run([],tmp)

    def test_stage_uses_raw_symlinks_and_frozen_group_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            original=Path(tmp,'original'); original.mkdir()
            (original/'meta.json').write_text('{"unchanged": true}')
            o={'trial_path':str(original)}
            row={'trial_path':str(original),'session':'canonical','id':'001',
                 'bottle':'light','repeat':2,'session_meta':{'schema':'e1e2_session_v1'}}
            roots,held=stage_training(Path(tmp,'view'),[],[o],{str(original):row})
            self.assertTrue(Path(roots[0],'trials','001').is_symlink())
            self.assertEqual(held,[roots[0]+'|light|2'])
            self.assertEqual(json.loads((original/'meta.json').read_text()), {'unchanged':True})


class ObservedSkillEvaluationTests(unittest.TestCase):
    def make_session(self, tmp, *, role='development', label='needs_recovery',
                     reached=True, complete=True, receipt=True, command=2., measured=1.8):
        from bottle_scan.skill_offline import write_json
        root=Path(tmp,'session')
        root.mkdir()
        write_json(root/'session_meta.json', {'schema':'e1e2_session_v1','simulated':False,
            'agentic':{'role':role},'config':{'force':{'target':3.,'probe_fraction':.5}}})
        first={'model_id':'frozen_one','invocation':{'needs_recovery':True,'candidate_supported':True,
               'applicable':False,'verified':False,'invoke':True},'recovery':{'action':{'delta_mm':command}},
               'exit':{'completed':False,'hand_back':False}}
        handoff={'completed':complete,'hand_back':True}
        row={'id':'0001','cond':'B','kind':'scan','recording_complete':True,'complete':True,
             'delta_mm':999., 'skill_ground_truth':{'source':'operator_review_of_observed_contact',
                 'initial_class':label,'final_target_reached':reached,'independent_of_experimental_offset':True},
             'skill_observations':[{'phase':'first_touch','observation':{},'skill_outputs':None},
                                   {'phase':'skill_decision','observation':{},'skill_outputs':first}],
             'skill_outputs':{'model_id':'frozen_one','exit':handoff},'skill_handoff':handoff,
             'correction_history':[{'delta_mm':command,'applied_delta_mm':measured}],
             'probe_regrasps':[{'probe':{'timestamp_ns':101},
                 'receipt':{'applied':True,'action_mm':measured,'completed_ns':100}}] if receipt else []}
        folder=root/'trials'/'0001'
        folder.mkdir(parents=True)
        write_json(folder/'meta.json',row)
        write_json(folder/'recording_status.json',{'complete':True})
        return root

    def test_actual_labels_actions_and_exit(self):
        from bottle_scan.skill_offline import evaluate_skill_sessions
        with tempfile.TemporaryDirectory() as tmp:
            root=self.make_session(tmp)
            r=evaluate_skill_sessions([root],role='development')
            self.assertEqual(r['invocation']['TP'],1)
            self.assertEqual(r['recovery']['successes'],1)
            self.assertAlmostEqual(r['recovery']['tracking_mae_mm'],.2)
            self.assertTrue(r['actions'][0]['fresh_post_action_observation'])
            self.assertEqual(r['exit']['TP'],1)
            self.assertFalse(r['certificate_issued'])
            self.assertEqual(r['cells'][0]['conditions']['a_mm'],999.)
            # Extreme planned offset did not replace the independent normal/target label.

    def test_legacy_rejection_diagnosis_preserves_original_record(self):
        from bottle_scan.skill_offline import evaluate_skill_sessions, write_json
        with tempfile.TemporaryDirectory() as tmp:
            root=self.make_session(tmp, reached=False, complete=False)
            path=root/'trials'/'0001'/'meta.json'
            row=json.loads(path.read_text())
            first=row['skill_observations'][1]['skill_outputs']
            first['invocation'].update(invoke=False, permitted=False, reasons=[])
            first['recovery']['decision']={'rejected':True, 'reason':'policy_uncertain', 'uncertainty_mm':5.}
            row['skill_handoff']['reason']='missing_or_mismatched_validation_certificate'
            write_json(path,row)
            before=path.read_bytes()
            r=evaluate_skill_sessions([root],role='development')
            detail=r['details'][0]
            self.assertEqual(detail['diagnostic_terminal_reason'],'policy_uncertain')
            self.assertEqual(detail['terminal_exit']['reason'],'missing_or_mismatched_validation_certificate')
            self.assertEqual(r['rejection_diagnostics']['terminal_reasons'],{'policy_uncertain':1})
            self.assertEqual(r['rejection_diagnostics']['legacy_reason_corrections'],1)
            self.assertEqual(path.read_bytes(),before)
            first['recovery']['decision']={'rejected':False, 'reason':'target_reached'}
            write_json(path,row)
            r=evaluate_skill_sessions([root],role='development')
            self.assertEqual(r['details'][0]['diagnostic_terminal_reason'], 'missing_or_mismatched_validation_certificate')
            self.assertEqual(r['rejection_diagnostics']['legacy_reason_corrections'],0)

    def test_no_labels_stay_unknown_not_geometric_truth(self):
        from bottle_scan.skill_offline import evaluate_skill_sessions
        with tempfile.TemporaryDirectory() as tmp:
            root=self.make_session(tmp,label=None,reached=None)
            r=evaluate_skill_sessions([root],role='development')
            self.assertEqual(r['invocation']['unknown'],1)
            self.assertEqual(r['exit']['unknown'],1)
            self.assertEqual(r['coverage']['initial_unlabelled'],1)

    def test_inappropriate_invocation_and_false_exit(self):
        from bottle_scan.skill_offline import evaluate_skill_sessions
        with tempfile.TemporaryDirectory() as tmp:
            root=self.make_session(tmp,label='not_applicable',reached=False)
            r=evaluate_skill_sessions([root],role='development')
            self.assertEqual(r['invocation']['unsafe_or_inapplicable_invocation'],1)
            self.assertEqual(r['candidate_support']['FP'],1)
            self.assertEqual(r['exit']['false_success'],1)

    def test_independent_source_cannot_become_development(self):
        from bottle_scan.skill_offline import evaluate_skill_sessions
        with tempfile.TemporaryDirectory() as tmp:
            root=self.make_session(tmp,role='final_validation')
            with self.assertRaisesRegex(ValueError,'cannot enter'):
                evaluate_skill_sessions([root],role='development')
            r=evaluate_skill_sessions([root],role='independent')
            self.assertEqual(r['role'],'independent')

    def test_missing_action_receipt_is_unknown(self):
        from bottle_scan.skill_offline import evaluate_skill_sessions
        with tempfile.TemporaryDirectory() as tmp:
            root=self.make_session(tmp,receipt=False)
            r=evaluate_skill_sessions([root],role='development')
            self.assertEqual(r['recovery']['missing_receipts'],1)
            self.assertIsNone(r['recovery']['tracking_mae_mm'])

    def test_feedback_revalidates_sources_and_saves_pending_memory(self):
        from bottle_scan.skill_offline import evaluate_skill_sessions, feedback_cli
        import yaml
        with tempfile.TemporaryDirectory() as tmp:
            root=self.make_session(tmp)
            report=Path(tmp,'dev.json')
            evaluate_skill_sessions([root],role='development',out=report)
            base=Path(tmp,'base.yaml')
            base.write_text(yaml.safe_dump({'force':{'target':3.,'probe_fraction':.5},'servo':{'max_delta_mm':15.}}))
            cfg=Path(tmp,'config.yaml')
            cfg.write_text(yaml.safe_dump({'base_config':'base.yaml',
                'collection':{'offsets_mm':[-10.,-5.,0.,5.,10.]},
                'memory':{'path':str(Path(tmp,'memory'))}}))
            code=feedback_cli(['feedback','--report',str(report),'--config',str(cfg),
                               '--out',str(Path(tmp,'feedback')),'--budget','6'])
            self.assertEqual(code,0)
            result=json.loads(Path(tmp,'feedback','selection.json').read_text())
            self.assertEqual(result['status'],'planned_unexecuted')
            self.assertLessEqual(result['planned_trial_attempts'],6)
            self.assertTrue(Path(tmp,'feedback','memory_record.json').exists())
            data=json.loads(report.read_text()); data['invocation']['FP']=100
            report.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError,'no longer matches'):
                feedback_cli(['feedback','--report',str(report),'--config',str(cfg),
                              '--out',str(Path(tmp,'forged'))])

    def test_feedback_rejects_independent_report(self):
        from bottle_scan.skill_offline import evaluate_skill_sessions,feedback_cli
        with tempfile.TemporaryDirectory() as tmp:
            root=self.make_session(tmp,role='final_validation')
            report=Path(tmp,'independent.json')
            evaluate_skill_sessions([root],role='independent',out=report)
            with self.assertRaisesRegex(ValueError,'development-only'):
                feedback_cli(['feedback','--report',str(report),'--out',str(Path(tmp,'feedback'))])


class EfficiencyAndCertificateTests(unittest.TestCase):
    def test_first_checkpoint_does_not_assume_monotonicity(self):
        from bottle_scan.skill_offline import sample_efficiency_summary
        curve=[{'mode':'x','selected_A_trials':n,'independent_A':{'mae_mm':m,'p90_mm':p}}
               for n,m,p in [(24,.5,1.),(48,3.,5.),(72,.8,1.5)]]
        report=sample_efficiency_summary(curve,source_attempts=310,shared_development_trials=45)
        self.assertEqual(report['branches']['x']['first_observed_A_budget'],24)
        self.assertEqual(report['branches']['x']['selected_plus_shared_development_trials'],69)
        self.assertEqual(report['branches']['x']['full_skill_equivalent_capability'],'not_established')
        self.assertEqual(report['new_physical_interactions'],0)

    def test_no_geometric_target_reached_is_censored(self):
        from bottle_scan.skill_offline import sample_efficiency_summary
        r=sample_efficiency_summary([{'mode':'x','selected_A_trials':24,
             'independent_A':{'mae_mm':3.,'p90_mm':5.}}],source_attempts=310,shared_development_trials=45)
        self.assertIsNone(r['branches']['x']['first_observed_A_budget'])
        self.assertTrue(r['branches']['x']['censored'])

    def test_development_cannot_certify(self):
        from bottle_scan.skill_offline import build_skill_certificate
        with self.assertRaisesRegex(ValueError,'independent'):
            build_skill_certificate({'role':'development'}, {}, {})

    def certificate_fixture(self,tmp,n,stable=True):
        from bottle_scan.skill_offline import write_json, _hash
        from types import SimpleNamespace
        root=ObservedSkillEvaluationTests().make_session(tmp,role='final_validation')
        model={'fixture':'no hardware','model_id':'frozen_one'}
        meta=json.loads((root/'session_meta.json').read_text())
        meta.update(fit=model)
        meta['config'].update(servo={},recovery_skill={'runtime':{}})
        write_json(root/'session_meta.json',meta)
        row=json.loads((root/'trials'/'0001'/'meta.json').read_text())
        observation={'status':'contact','context':{'task':'bottle'},'timestamp_ns':1,
                     'bilateral':True,'force_stable':stable,'stable_duration_s':.3,
                     'descriptors':[{'centroid_px':[1.,2.],'valid':True,'touches_edge':False},
                                    {'centroid_px':[3.,4.],'valid':True,'touches_edge':False}]}
        first=row['skill_observations'][1]['skill_outputs']
        terminal=row['skill_outputs']
        row['skill_observations']=[
            {'phase':'skill_decision','observation':observation,'skill_outputs':first},
            {'phase':'skill_decision','observation':{**observation,'timestamp_ns':300000100,'now_ns':300000100},'skill_outputs':terminal},
            {'phase':'skill_decision','observation':{**observation,'timestamp_ns':600000100,'now_ns':600000100},'skill_outputs':terminal}]
        for i in range(1,n+1):
            folder=root/'trials'/f'{i:04d}'
            folder.mkdir(exist_ok=True)
            write_json(folder/'meta.json',{**row,'id':f'{i:04d}'})
            write_json(folder/'recording_status.json',{'complete':True})
        skill=SimpleNamespace(model_id='frozen_one',model_sha256=_hash(model),config_sha256='runtime',
            config={'required_context':{'task':'bottle'},'tracking_tolerance_mm':2.,
                    'target':{'min_stable_duration_s':.2}})
        return root,model,skill

    def test_sparse_independent_cell_yields_explicit_empty_certificate(self):
        from bottle_scan.skill_offline import evaluate_skill_sessions,build_skill_certificate
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as tmp:
            root,model,skill=self.certificate_fixture(tmp,1)
            report=evaluate_skill_sessions([root],role='independent')
            with patch('bottle_scan.recovery_skill.RecoverySkill',return_value=skill):
                cert=build_skill_certificate(report,model,{})
            self.assertEqual(cert['observed_cells'],[])
            self.assertEqual(cert['status'],'insufficient_evidence')

    def test_repeated_exact_cell_requires_actual_stable_windows(self):
        from bottle_scan.skill_offline import evaluate_skill_sessions,build_skill_certificate
        from unittest.mock import patch
        for stable in (False,True):
            with tempfile.TemporaryDirectory() as tmp:
                root,model,skill=self.certificate_fixture(tmp,30,stable=stable)
                report=evaluate_skill_sessions([root],role='independent')
                with patch('bottle_scan.recovery_skill.RecoverySkill',return_value=skill):
                    cert=build_skill_certificate(report,model,{})
                self.assertEqual(len(cert['observed_cells']),1 if stable else 0)
                if stable:
                    cell=cert['observed_cells'][0]
                    self.assertEqual(cell['centroid_bounds_px'],[[[1.,1.],[2.,2.]],[[3.,3.],[4.,4.]]])
                    self.assertGreater(cell['wilson_lower'],.8)
                    self.assertEqual(cell['n'],30)


class PreregisteredCertificateTests(unittest.TestCase):
    certificate_fixture = EfficiencyAndCertificateTests.certificate_fixture
    def specification(self):
        return {'observation_cells':[{'cell_id':'reference_contact','status':'contact',
            'context':{'task':'bottle'},'centroid_bounds_px':[[[.5,1.5],[1.5,2.5]],[[2.5,3.5],[3.5,4.5]]]}]}

    def test_preregistered_cell_supports_varying_observations_and_rejects_changed_bounds(self):
        from bottle_scan.skill_offline import evaluate_skill_sessions,build_skill_certificate,write_json
        from unittest.mock import patch
        from copy import deepcopy
        with tempfile.TemporaryDirectory() as tmp:
            root,model,skill=self.certificate_fixture(tmp,30)
            spec=self.specification()
            meta=json.loads((root/'session_meta.json').read_text())
            meta['config']['recovery_skill']['certification']=spec
            write_json(root/'session_meta.json',meta)
            for index, path in enumerate(sorted((root/'trials').glob('*/meta.json'))):
                row=json.loads(path.read_text())
                row['skill_observations'][0]['observation']['descriptors'][0]['centroid_px'][0]=.8+index*.01
                write_json(path,row)
            report=evaluate_skill_sessions([root],role='independent')
            with patch('bottle_scan.recovery_skill.RecoverySkill',return_value=skill):
                result=build_skill_certificate(report,model,{},certification=spec)
                self.assertEqual(len(result['observed_cells']),1)
                self.assertEqual(result['observed_cells'][0]['cell_id'],'reference_contact')
                self.assertEqual(result['status'],'validated_preregistered_observation_cells')
                changed=deepcopy(spec)
                changed['observation_cells'][0]['centroid_bounds_px'][0][0]=[-100,100]
                rejected=build_skill_certificate(report,model,{},certification=changed)
                self.assertEqual(rejected['observed_cells'],[])
                self.assertIn('differ',rejected['reasons'][0])

    def test_overlapping_cells_rejected(self):
        from bottle_scan.skill_offline import _certificate_cells
        from copy import deepcopy
        spec=self.specification()
        second=deepcopy(spec['observation_cells'][0]); second['cell_id']='overlap'
        spec['observation_cells'].append(second)
        with self.assertRaisesRegex(ValueError,'overlap'):
            _certificate_cells(spec,{'task':'bottle'})


class CollectionContractIntegrationTests(unittest.TestCase):
    make_session=ObservedSkillEvaluationTests.make_session

    def terminal_row(self,completed):
        return {'attempt':1,'accepted':completed,'reason':'completed' if completed else 'motion_bounds',
                'timestamp_ns':200,'terminal':True,
                'skill_decision':{'exit':{'completed':completed,'hand_back':True},'recovery':{'action':None}}}

    def test_normal_terminal_without_action_needs_no_receipt(self):
        from bottle_scan.skill_offline import evaluate_skill_sessions,write_json
        with tempfile.TemporaryDirectory() as tmp:
            root=self.make_session(tmp,label='already_normal')
            path=root/'trials'/'0001'/'meta.json';row=json.loads(path.read_text())
            row['correction_history']=[self.terminal_row(True)]
            row['probe_regrasps']=[]
            row['skill_observations'][1]['skill_outputs']['invocation'].update(needs_recovery=False,invoke=False)
            write_json(path,row)
            r=evaluate_skill_sessions([root],role='development')
            self.assertEqual(r['recovery']['action_attempts'],0)
            self.assertEqual(r['recovery']['missing_receipts'],0)
            self.assertEqual(r['coverage']['terminal_history_entries'],1)
            self.assertEqual(r['exit']['TP'],1)

    def test_recovery_action_then_terminal_counts_only_action(self):
        from bottle_scan.skill_offline import evaluate_skill_sessions,write_json
        with tempfile.TemporaryDirectory() as tmp:
            root=self.make_session(tmp)
            path=root/'trials'/'0001'/'meta.json';row=json.loads(path.read_text())
            row['correction_history'].append(self.terminal_row(True))
            write_json(path,row)
            r=evaluate_skill_sessions([root],role='development')
            self.assertEqual(r['recovery']['action_attempts'],1)
            self.assertEqual(r['recovery']['measured_actions'],1)
            self.assertEqual(r['recovery']['missing_receipts'],0)
            self.assertEqual(r['recovery']['successes'],1)

    def test_blocked_action_does_not_require_applied_receipt(self):
        from bottle_scan.skill_offline import evaluate_skill_sessions,write_json
        with tempfile.TemporaryDirectory() as tmp:
            root=self.make_session(tmp,reached=False,complete=False,receipt=False)
            path=root/'trials'/'0001'/'meta.json';row=json.loads(path.read_text())
            row['correction_history'][0].update(blocked=True,blocked_reason='bounds',applied_delta_mm=0.)
            row['correction_history'].append(self.terminal_row(False))
            write_json(path,row)
            r=evaluate_skill_sessions([root],role='development')
            self.assertEqual(r['recovery']['blocked_actions'],1)
            self.assertEqual(r['recovery']['missing_receipts'],0)
            self.assertEqual(r['recovery']['missing_fresh_post_action'],0)
            self.assertEqual(r['recovery']['failures'],1)

    def extension_session(self,tmp,role):
        from bottle_scan.skill_offline import write_json
        root=self.make_session(tmp,role='final_validation' if role=='independent' else 'development')
        meta=json.loads((root/'session_meta.json').read_text())
        model={'fixture':'extension model'}; settings={'runtime':{'action_axis':'l'},'context':{},
                                                      'mode':'development','certificate':None}
        meta.update(schema='ambiguity_session_v1',cond='B',evaluation_role=role,fit=model,
                    skill_extension_B={'fit':model,'settings':settings,'evaluation_role':role})
        meta['config']['recovery_skill']=settings
        write_json(root/'session_meta.json',meta)
        path=root/'trials'/'0001'/'meta.json';row=json.loads(path.read_text())
        row.update(kind='skill_extension_probe',evaluation_role=role,
                   perturbation={'a_mm':0.,'l_mm':4.,'tilt_deg':0.},probe_force_target=1.2)
        row['correction_history'].append(self.terminal_row(True))
        write_json(path,row)
        return root

    def test_extension_B_is_evaluable_but_cannot_enter_a_training(self):
        from bottle_scan.skill_offline import evaluate_skill_sessions
        from bottle_scan.agentic_learning import _load
        for role in ('development','independent'):
            with tempfile.TemporaryDirectory() as tmp:
                root=self.extension_session(tmp,role)
                r=evaluate_skill_sessions([root],role=role)
                self.assertEqual(r['coverage']['trials'],1)
                self.assertEqual(r['recovery']['missing_receipts'],0)
                self.assertEqual(r['cells'][0]['conditions']['contact_height_l_mm'],4.)
                self.assertEqual(r['cells'][0]['conditions']['probe_force'],1.2)
                sessions,rows,issues=_load([root],{})
                self.assertEqual(rows,[])
                self.assertIn('unsupported session schema',issues[0])

    def test_extension_trial_role_mismatch_rejected(self):
        from bottle_scan.skill_offline import evaluate_skill_sessions,write_json
        with tempfile.TemporaryDirectory() as tmp:
            root=self.extension_session(tmp,'independent')
            path=root/'trials'/'0001'/'meta.json';row=json.loads(path.read_text())
            row['evaluation_role']='development';write_json(path,row)
            with self.assertRaisesRegex(ValueError,'evaluation role differs'):
                evaluate_skill_sessions([root],role='independent')


class CertificateTimingAndTerminalTests(unittest.TestCase):
    certificate_fixture=EfficiencyAndCertificateTests.certificate_fixture

    def test_too_close_or_stale_windows_cannot_certify(self):
        from bottle_scan.skill_offline import evaluate_skill_sessions,build_skill_certificate,write_json
        from unittest.mock import patch
        for fault in ('too_close','stale'):
            with tempfile.TemporaryDirectory() as tmp:
                root,model,skill=self.certificate_fixture(tmp,30)
                for path in (root/'trials').glob('*/meta.json'):
                    row=json.loads(path.read_text())
                    last=row['skill_observations'][-1]['observation']
                    if fault=='too_close':
                        last.update(timestamp_ns=300000101,now_ns=300000101)
                    else:
                        last['now_ns']+=2100000000
                    write_json(path,row)
                report=evaluate_skill_sessions([root],role='independent')
                with patch('bottle_scan.recovery_skill.RecoverySkill',return_value=skill):
                    cert=build_skill_certificate(report,model,{})
                self.assertEqual(cert['observed_cells'],[])

    def test_normal_no_action_terminal_can_certify_with_real_history_shape(self):
        from bottle_scan.skill_offline import evaluate_skill_sessions,build_skill_certificate,write_json
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as tmp:
            root,model,skill=self.certificate_fixture(tmp,30)
            for path in (root/'trials').glob('*/meta.json'):
                row=json.loads(path.read_text())
                row['skill_ground_truth']['initial_class']='already_normal'
                row['skill_observations'][0]['skill_outputs']['invocation'].update(needs_recovery=False,invoke=False)
                row['correction_history']=[CollectionContractIntegrationTests().terminal_row(True)]
                row['probe_regrasps']=[]
                write_json(path,row)
            report=evaluate_skill_sessions([root],role='independent')
            self.assertEqual(report['recovery']['action_attempts'],0)
            with patch('bottle_scan.recovery_skill.RecoverySkill',return_value=skill):
                cert=build_skill_certificate(report,model,{})
            self.assertEqual(len(cert['observed_cells']),1)

if __name__ == "__main__":
    unittest.main()
