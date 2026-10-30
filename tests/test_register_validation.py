"""Offline report locator changes preserve measurements and all acceptance gates."""
import builtins
import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import yaml

import register_validation
from bottle_scan.commissioning import hardware_binding
from bottle_scan.force_control import force_report_binding
from bottle_scan.session import Session, atomic_json
from bottle_scan.stop_validation import stop_report_binding
from tests.test_force_control import force_config, passing_report


def passing_stop(cfg):
    return {'schema': 'bottle_stop_validation_v2', 'simulated': False,
        'operator_confirmed': True, 'passed': True, 'binding': stop_report_binding(cfg),
        'trials': [{'mode': mode, 'repeat': i, 'simulated': False, 'valid': True, 'passed': True,
                    'last_published_target_verified': True, 'max_extra_displacement_mm': .1}
                   for mode in ('software', 'sigkill') for i in range(cfg['stop_test']['repeats'])]}


class ReportRegistrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.cfg = force_config()
        self.cfg['demo'] = False
        self.cfg['stop_test']['report_path'] = str(self.root/'old_stop.json')
        self.cfg['force']['validation']['report_path'] = str(self.root/'old_force.json')
        with patch('bottle_scan.session.code_version', return_value={'source_sha256':'fake-unit-fixture'}):
            self.session = Session.create(self.root/'session', self.cfg, np.asarray(self.cfg['T0']), simulated=False)
        self.session.progress['commissioning_passed'] = True
        self.session.meta['commissioning_acceptance'] = {'passed':True, 'operator_confirmation':'PASS', 'binding':hardware_binding(self.cfg)}
        self.session.save_progress(); self.session.save_meta()
        atomic_json(self.session.path/'plans'/'contact.json', {'immutable':'plan'})
        self.stop = self.root/'new_stop.json'; self.force = self.root/'new_force.json'
        atomic_json(self.stop, passing_stop(self.cfg)); atomic_json(self.force, passing_report(self.cfg))
        self.original_config = (self.session.path/'config_snapshot.yaml').read_bytes()
        self.original_meta = (self.session.path/'session_meta.json').read_bytes()
        self.original_progress = self.session.progress_path.read_bytes()
        self.original_plan = (self.session.path/'plans'/'contact.json').read_bytes()

    def assert_unchanged(self):
        self.assertEqual((self.session.path/'config_snapshot.yaml').read_bytes(), self.original_config)
        self.assertEqual((self.session.path/'session_meta.json').read_bytes(), self.original_meta)
        self.assertEqual(self.session.progress_path.read_bytes(), self.original_progress)
        self.assertFalse((self.session.path/'validation_registration_backups').exists())

    def test_both_valid_reports_change_only_locators_and_save_exact_backup(self):
        bindings = (hardware_binding(self.cfg), force_report_binding(self.cfg), stop_report_binding(self.cfg))
        entry = register_validation.register_reports(self.session.path, stop=self.stop, force=self.force)
        updated = Session(self.session.path)
        expected = copy.deepcopy(self.cfg)
        expected['stop_test']['report_path'] = str(self.stop.resolve())
        expected['force']['validation']['report_path'] = str(self.force.resolve())
        self.assertEqual(updated.cfg, expected)
        self.assertEqual(bindings, (hardware_binding(updated.cfg), force_report_binding(updated.cfg), stop_report_binding(updated.cfg)))
        self.assertEqual(self.session.progress_path.read_bytes(), self.original_progress)
        self.assertEqual((self.session.path/'plans'/'contact.json').read_bytes(), self.original_plan)
        self.assertEqual(updated.meta['commissioning_acceptance'], self.session.meta['commissioning_acceptance'])
        self.assertEqual(entry['state'], 'committed')
        self.assertEqual(updated.meta['validation_report_registrations'], [entry])
        self.assertEqual((self.session.path/entry['snapshot_backup']).read_bytes(), self.original_config)
        self.assertEqual(entry['snapshot_before_sha256'], hashlib.sha256(self.original_config).hexdigest())
        self.assertEqual(entry['changes'][0]['new_report_sha256'], hashlib.sha256(self.stop.read_bytes()).hexdigest())

    def test_only_force_preserves_stop_path_and_appends_prior_audit(self):
        register_validation.register_reports(self.session.path, stop=self.stop)
        first = Session(self.session.path).meta['validation_report_registrations'][0]
        register_validation.register_reports(self.session.path, force=self.force)
        updated = Session(self.session.path)
        self.assertEqual(updated.cfg['stop_test']['report_path'], str(self.stop.resolve()))
        self.assertEqual(updated.meta['validation_report_registrations'][0], first)
        self.assertEqual(len(updated.meta['validation_report_registrations']), 2)

    def test_simulated_stop_report_rejected_without_writing_session(self):
        report=passing_stop(self.cfg);report['simulated']=True;atomic_json(self.stop,report)
        with self.assertRaises(ValueError): register_validation.register_reports(self.session.path, stop=self.stop)
        self.assert_unchanged()

    def test_force_without_pass_rejected_atomically_even_when_stop_passes(self):
        report=passing_report(self.cfg);report['operator_confirmation']=None;atomic_json(self.force,report)
        with self.assertRaises(ValueError): register_validation.register_reports(self.session.path, stop=self.stop, force=self.force)
        self.assert_unchanged()

    def test_current_binding_mismatch_rejected(self):
        report=passing_report(self.cfg);report['binding']['source_sha256']='old-source';atomic_json(self.force,report)
        with self.assertRaisesRegex(ValueError,'changed'): register_validation.register_reports(self.session.path, force=self.force)
        self.assert_unchanged()

    def test_stop_incomplete_evidence_rejected_even_if_marked_pass(self):
        report=passing_stop(self.cfg);report['trials'].pop();atomic_json(self.stop,report)
        with self.assertRaisesRegex(ValueError,'repeats'): register_validation.register_reports(self.session.path, stop=self.stop)
        self.assert_unchanged()

    def test_force_failed_measurements_rejected_even_if_marked_pass(self):
        report=passing_report(self.cfg);report['trials'][0]['statistics']['left']['contact_peak']=100.;atomic_json(self.force,report)
        with self.assertRaises(ValueError): register_validation.register_reports(self.session.path, force=self.force)
        self.assert_unchanged()

    def test_no_report_argument_rejected(self):
        with self.assertRaises(ValueError): register_validation.register_reports(self.session.path)
        self.assert_unchanged()

    def test_active_trial_and_simulated_session_rejected(self):
        self.session.progress['active']='trial_0001';self.session.save_progress()
        with self.assertRaisesRegex(ValueError,'active'): register_validation.register_reports(self.session.path, stop=self.stop)
        self.session.progress['active']=None;self.session.save_progress()
        self.session.meta['simulated']=True;self.session.save_meta()
        with self.assertRaisesRegex(ValueError,'real prepared'): register_validation.register_reports(self.session.path, stop=self.stop)
        self.assertEqual((self.session.path/'config_snapshot.yaml').read_bytes(),self.original_config)

    def test_changed_report_during_validation_refused(self):
        original=register_validation.gate_stop_report
        def changed(cfg,path=None):
            report=original(cfg,path=path)
            path.write_text(path.read_text()+'\n')
            return report
        with patch.object(register_validation,'gate_stop_report',side_effect=changed), self.assertRaisesRegex(ValueError,'changed during'):
            register_validation.register_reports(self.session.path, stop=self.stop)
        self.assert_unchanged()

    def test_atomic_snapshot_failure_retains_original_and_failed_audit(self):
        with patch.object(register_validation,'_atomic_bytes',side_effect=OSError('fake disk failure')), self.assertRaises(OSError):
            register_validation.register_reports(self.session.path, force=self.force)
        self.assertEqual((self.session.path/'config_snapshot.yaml').read_bytes(),self.original_config)
        entry=Session(self.session.path).meta['validation_report_registrations'][-1]
        self.assertEqual(entry['state'],'failed_before_snapshot_commit')
        self.assertEqual((self.session.path/entry['snapshot_backup']).read_bytes(),self.original_config)

    def test_cli_success_imports_no_robot_backend_or_ros(self):
        original=builtins.__import__
        forbidden=('rospy','bridge_ros_public','bottle_scan.backend','bottle_scan.runtime','bottle_scan.hardware_arm','bottle_scan.hardware_gripper','bottle_scan.hardware_sensors')
        def guarded(name,*args,**kwargs):
            if name in forbidden:raise AssertionError('Device import forbidden: '+name)
            return original(name,*args,**kwargs)
        with patch('builtins.__import__',side_effect=guarded),contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(register_validation.main(['--session',str(self.session.path),'--stop',str(self.stop),'--force',str(self.force)]),0)


if __name__ == '__main__': unittest.main()
