"""No hardware: check launch contracts before any scheduler process exists."""
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from bottle_scan.a_axis_task import build_launch, main, parser


class TaskEntryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root/'run_contact_261010.sh').write_text('# fake scheduler\n')

    def args(self, command='check', *extra):
        return parser().parse_args([command, '--scheduler-root', str(self.root), *extra])

    def test_check_defaults_to_natural_main_task(self):
        command, env = build_launch(self.args(), {'RUN_ROBOT': '1'})
        self.assertEqual(command[-1], 'task-check')
        self.assertNotIn('CONTACT_EXPERIMENT_B_OFFSET_MM', env)
        self.assertEqual(env['CONTACT_SKILL_AFTER'], 'handoff')

    def test_never_launches_run_without_real_and_field_confirmation(self):
        for args, env in [(self.args('run'), {'CONTACT_TABLETOP_CONFIRMED':'1'}),
                          (self.args('run','--real'), {})]:
            with self.assertRaises(ValueError):
                build_launch(args, env)
        with patch('bottle_scan.a_axis_task.subprocess.run') as execute:
            self.assertEqual(main(['run','--scheduler-root',str(self.root)]),2)
            execute.assert_not_called()

    def test_natural_mode_rejects_hidden_prelude_or_stop(self):
        for env in ({'CONTACT_EXPERIMENT_B_OFFSET_MM':'0'}, {'CONTACT_SKILL_AFTER':'stop'},
                    {'CONTACT_SKILL_RECOVERY':'observe_only'}):
            with self.assertRaises(ValueError): build_launch(self.args(), env)
        with self.assertRaises(ValueError): build_launch(self.args('check','--offset-mm','0'), {})

    def test_pilot_is_explicit_and_keeps_stop_default(self):
        command, env = build_launch(self.args('check','--route','pilot','--offset-mm','-5'), {})
        self.assertEqual(command[-1], 'skill-check')
        self.assertEqual(env['CONTACT_EXPERIMENT_B_OFFSET_MM'], '-5.0')
        self.assertEqual(env['CONTACT_SKILL_AFTER'], 'stop')
        for offset in ['nan','inf','2','11']:
            with self.assertRaises(ValueError):
                build_launch(self.args('check','--route','pilot','--offset-mm',offset), {})

    def test_checkpoint_must_exist_and_is_passed_explicitly(self):
        checkpoint = self.root/'checkpoint'
        checkpoint.mkdir()
        with self.assertRaises(ValueError):
            build_launch(self.args('check','--checkpoint',str(checkpoint)), {})
        (checkpoint/'config.json').write_text('{}')
        _, env = build_launch(self.args('check','--checkpoint',str(checkpoint)), {})
        self.assertEqual(env['CONTACT_MAIN_POLICY_PATH'], str(checkpoint))

    def test_explicit_skill_export_is_validated_before_launch(self):
        config=self.root/'scheduler.json';config.write_text('{}')
        with patch('bottle_scan.primary_skill_export.validate_export_config',return_value={'scheduler_config':str(config)}) as validate:
            command,env=build_launch(self.args('check','--skill-config',str(config)),{})
            self.assertEqual(env['CONTACT_SCHEDULER_CONFIG'],str(config))
            self.assertEqual(command[-1],'task-check')
            validate.assert_called_once_with(config)
        with patch('bottle_scan.a_axis_task.subprocess.run') as execute:
            self.assertEqual(main(['check','--scheduler-root',str(self.root),'--skill-config',str(config)]),2)
            execute.assert_not_called()

    def test_real_launch_delegates_to_existing_scheduler(self):
        command, env = build_launch(self.args('run','--real','--trial-id','trial_1'),
                                    {'CONTACT_TABLETOP_CONFIRMED':'1'})
        self.assertEqual(command[-1], 'task-run')
        self.assertEqual(env['CONTACT_SKILL_TRIAL_ID'],'trial_1')


if __name__ == '__main__':
    unittest.main()
