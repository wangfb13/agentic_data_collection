"""Offline artifact integration; no acquisition, device connection or network."""
import copy
import json
from unittest.mock import Mock, patch

import yaml

from bottle_scan import rsi
from bottle_scan.contact_policy import _seal
from bottle_scan.e1e2_core import defaults
from bottle_scan.session import atomic_json, fingerprint
from tests.test_agentic_learning import EvidenceFixture
from tests.test_rsi import settings as rsi_settings


class RSIConditionArtifactTests(EvidenceFixture):
    def setUp(self):
        super().setUp()
        self.execution_config = defaults()
        self.execution_config['axes_tcp'] = {
            'a': [1., 0., 0.], 'l': [0., 0., 1.], 'c': [0., 1., 0.]}
        self.execution_config['servo']['tol_mm'] = 1.25
        self.execution_config['force']['probe_stable_s'] = .35
        self.execution_config['tactile']['probe_frames'] = 7
        self.seed = self.make(patch_cfg=self.execution_config)
        self.settings = rsi_settings(self.root)
        self.settings['contact_policy'].update(epochs=120, ensemble_size=3)
        self.settings['seed_sessions'] = [str(self.seed)]
        self.base_path = self.root / 'execution.yaml'
        self.base_path.write_text(yaml.safe_dump(self.execution_config))
        self.settings['base_config'] = str(self.base_path)

    def campaign(self):
        no_execution = Mock(side_effect=AssertionError('artifact tests cannot execute a batch'))
        runtime_config = copy.deepcopy(self.execution_config)
        runtime_config['gripper']['min_normalized'] = .336
        with patch('bottle_scan.agentic.resolve_base', return_value=runtime_config):
            campaign = rsi.RSICampaign(self.root / 'campaign', settings=self.settings,
                                      executor=no_execution, ask=Mock(side_effect=AssertionError('no prompt')),
                                      say=lambda _: None)
        self.addCleanup(no_execution.assert_not_called)
        return campaign

    def assert_thresholds(self, model):
        conditions = model['skill_conditions']
        self.assertEqual(conditions['version'], 'contact_policy_conditions_v1')
        self.assertEqual(conditions['thresholds']['tol_mm'], 1.25)
        self.assertEqual(conditions['thresholds']['probe_stable_s'], .35)
        self.assertEqual(conditions['thresholds']['probe_frames'], 7)
        self.assertTrue(conditions['start']['description_zh'])
        self.assertTrue(conditions['exit']['description_zh'])

    def test_standalone_train_saves_conditions_without_resolving_gripper_or_prompting(self):
        # An unresolved gripper floor is irrelevant to this offline export.
        self.assertIsNone(self.execution_config['gripper']['min_normalized'])
        config_path = self.root / 'rsi.yaml'
        config_path.write_text(yaml.safe_dump(self.settings))
        destination = self.root / 'standalone.json'
        messages = []
        with patch('bottle_scan.agentic.resolve_base', side_effect=AssertionError('no runtime setup')):
            result = rsi.main(['train', '--simulate', '--config', str(config_path),
                               '--session', str(self.seed), '--out', str(destination)],
                              input_fn=Mock(side_effect=AssertionError('offline training must not prompt')),
                              output=messages.append,
                              executor=Mock(side_effect=AssertionError('no robot executor')))
        self.assertEqual(result, 0, messages)
        model = json.loads(destination.read_text())
        self.assert_thresholds(model)
        self.assertEqual(json.loads(messages[-1])['skill_conditions'], model['skill_conditions'])

    def test_campaign_training_freeze_and_report_use_actual_execution_thresholds(self):
        campaign = self.campaign()
        identifier = campaign.train([self.seed])
        model = campaign.model(identifier)
        self.assert_thresholds(model)
        campaign.state['current_policy_id'] = identifier
        models_before = copy.deepcopy(campaign.state['models'])
        campaign.prepare_final()
        self.assertEqual(campaign.state['models'], models_before)
        self.assertEqual(campaign.state['frozen']['model_id'], identifier)
        self.assertEqual(campaign.state['frozen']['skill_conditions'], model['skill_conditions'])
        self.assertEqual(campaign.state['frozen']['policy_sha256'], fingerprint(model))
        self.assertEqual(campaign.state['batches'][-1]['model_id'], identifier)
        self.assertEqual(model['source_sessions'], [str(self.seed)])
        campaign.report()
        report = (campaign.root / 'REPORT.md').read_text()
        for phrase in (model['family'], campaign.state['models'][identifier],
                       '何时调用：', '何时成功退出：', '1.25 mm', '0.35 s', '7 帧'):
            self.assertIn(phrase, report)

    def test_campaign_train_reuses_identical_artifact_and_refuses_overwriting_a_collision(self):
        campaign = self.campaign()
        identifier = campaign.train([self.seed])
        destination = campaign.root / campaign.state['models'][identifier]
        before = destination.read_bytes()
        with patch('bottle_scan.rsi.atomic_json', wraps=atomic_json) as write:
            self.assertEqual(campaign.train([self.seed]), identifier)
        self.assertFalse(any(call.args[0] == destination for call in write.call_args_list))
        self.assertEqual(destination.read_bytes(), before)
        changed = json.loads(before)
        changed['untrusted_note'] = 'different artifact under an existing id'
        destination.write_text(json.dumps(changed))
        damaged_bytes = destination.read_bytes()
        with self.assertRaisesRegex(ValueError, '内容寻址文件已存在但内容不同'):
            campaign.train([self.seed])
        self.assertEqual(destination.read_bytes(), damaged_bytes)

    def test_legacy_final_freeze_creates_new_artifact_without_retraining_or_rewriting(self):
        campaign = self.campaign()
        identifier = campaign.train([self.seed])
        trained = campaign.model(identifier)
        legacy = copy.deepcopy(trained)
        del legacy['skill_conditions']
        legacy = _seal(legacy)
        legacy_path = f"models/{legacy['model_id']}.json"
        atomic_json(campaign.root / legacy_path, legacy)
        campaign.state['models'][legacy['model_id']] = legacy_path
        campaign.state['current_policy_id'] = legacy['model_id']
        before_bytes = (campaign.root / legacy_path).read_bytes()
        before_models = copy.deepcopy(campaign.state['models'])

        campaign.report()
        self.assertIn('历史模型文件未写回', (campaign.root / 'REPORT.md').read_text())
        self.assertEqual(campaign.state['models'], before_models)
        self.assertEqual((campaign.root / legacy_path).read_bytes(), before_bytes)

        with patch('bottle_scan.contact_policy.train_contact_policy', side_effect=AssertionError('freeze must not retrain')):
            campaign.prepare_final()
        frozen = campaign.model(campaign.state['frozen']['model_id'])
        self.assertNotEqual(frozen['model_id'], legacy['model_id'])
        self.assert_thresholds(frozen)
        for key in ('networks', 'scaling', 'fallback_fit', 'source_trials', 'source_sessions', 'training_snapshot_sha256'):
            self.assertEqual(frozen[key], legacy[key])
        self.assertEqual((campaign.root / legacy_path).read_bytes(), before_bytes)
        self.assertEqual(campaign.state['frozen']['skill_conditions'], frozen['skill_conditions'])
        self.assertEqual(campaign.state['frozen']['policy_sha256'], fingerprint(frozen))
        self.assertEqual(campaign.state['batches'][-1]['model_id'], frozen['model_id'])
