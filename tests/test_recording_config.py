"""Synthetic load settings are explicit but cannot replace physical evidence."""
import copy
from pathlib import Path
import tempfile
import unittest

from bottle_scan.config import demo_config,load_config,validate_config
from bottle_scan.recording import record_stress_defaults,validate_record_stress_settings
from bottle_scan.commissioning import hardware_binding
from bottle_scan.force_control import force_report_binding
from bottle_scan.stop_validation import stop_report_binding


class RecordingConfigTests(unittest.TestCase):
    def test_both_templates_and_demo_expose_same_synthetic_load_defaults(self):
        root=Path(__file__).resolve().parents[1]
        for cfg in (demo_config(),load_config(root/"config.yaml"),load_config(root/"config.ros.yaml")):
            self.assertEqual(cfg["recording"]["stress"],record_stress_defaults())
            self.assertEqual(validate_record_stress_settings(cfg),[])

    def test_missing_invalid_and_nonfinite_load_settings_are_rejected(self):
        defaults=record_stress_defaults()
        for name in defaults:
            cfg=demo_config();del cfg["recording"]["stress"][name]
            with self.subTest(missing=name):
                self.assertTrue(any("recording.stress."+name in error for error in validate_config(cfg)))
        for name in defaults.keys()-{"image_shape"}:
            for value in (0,-1,True,float("inf"),float("nan"),"TODO"):
                cfg=demo_config();cfg["recording"]["stress"][name]=value
                with self.subTest(name=name,value=value):
                    self.assertTrue(validate_record_stress_settings(cfg))
        for value in ([1080,1920,4],[1080.5,1920,3],[0,1920,3],[True,1920,3],"1080x1920"):
            cfg=demo_config();cfg["recording"]["stress"]["image_shape"]=value
            with self.subTest(shape=value):self.assertTrue(validate_record_stress_settings(cfg))
        cfg=demo_config();cfg["recording"]["stress"]["duration_s"]=.001
        self.assertTrue(any("twenty" in error for error in validate_record_stress_settings(cfg)))

    def test_offline_stress_changes_do_not_invalidate_any_physical_report(self):
        cfg=demo_config();changed=copy.deepcopy(cfg)
        changed["recording"]["stress"].update(duration_s=7.,producer_hz=10.,image_shape=[480,640,3])
        for binding in (hardware_binding,force_report_binding,stop_report_binding):
            with self.subTest(binding=binding.__name__):self.assertEqual(binding(cfg),binding(changed))
        self.assertEqual(cfg["recording"]["stress"],record_stress_defaults())

    def test_live_source_freshness_changes_invalidate_all_physical_reports(self):
        cfg=demo_config();changed=copy.deepcopy(cfg)
        changed["recording"]["source_max_age_s"]*=2
        for binding in (hardware_binding,force_report_binding,stop_report_binding):
            with self.subTest(binding=binding.__name__):self.assertNotEqual(binding(cfg),binding(changed))

    def test_writer_capacity_remains_bound_to_recorded_force_and_commissioning(self):
        cfg=demo_config();changed=copy.deepcopy(cfg)
        changed["recording"]["writer"]["queue_bytes"]*=2
        self.assertNotEqual(hardware_binding(cfg),hardware_binding(changed))
        self.assertNotEqual(force_report_binding(cfg),force_report_binding(changed))
        self.assertEqual(stop_report_binding(cfg),stop_report_binding(changed))

    def test_stress_runner_rejects_missing_config_before_creating_output(self):
        from record_stress import run_stress
        cfg=demo_config();del cfg["recording"]["stress"]
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)/"run"
            with self.assertRaisesRegex(ValueError,"recording.stress"):run_stress(cfg,output)
            self.assertFalse(output.exists())


if __name__=="__main__":unittest.main()
