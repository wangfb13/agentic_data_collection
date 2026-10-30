"""A6 offset guards using only synthetic source frames; never ROS sockets."""
import copy
import unittest

import numpy as np

from bottle_scan.zero_guard import ZeroGuardError, guarded_zero_report, validate_zero_guard_settings


def configuration():
    return {"zero_guard":{"sample_count":100,"timeout_s":.5,"force_std_max_N":.1,
                          "dd_contact_threshold":.05,"dd_contact_max_fraction":.001,"offset_delta_warn_N":.2},
            "gripper":{"open_mm":60.,"open_tolerance_mm":.5},"tactile":{"depth_channel":0},
            "hardware":{"public_ros":{"sensors":{"force_conversion":{
                "force_scale_to_N":2.,"normal_axis":2,"normal_signs":{"left":1.,"right":-1.}},
                "startup_zero":{"unloaded_confirmed":True}}}}}


class Source:
    def __init__(self):
        self.now=1_000_000_000
        self.clock=lambda:self.now
        self.max_age_ns=250_000_000
        self.serials={"left":"L","right":"R"}
        self.dd=[np.zeros((4,5,3)),np.zeros((4,5,3))]
        self.collects=0
        self.normals=np.full(100,3.)
        self.after_collect=None
        self.duplicate=False

    def observation_snapshot(self):
        return {"dd":self.dd,"source_sequences":{"dd_left":10,"dd_right":20}}

    def collect_raw_zero_samples(self,count,timeout,*,monitor):
        self.collects+=1
        monitor()
        rows=[]
        for i,value in enumerate(self.normals):
            self.now+=1_000_000
            rows.append((self.now,np.array([0.,0.,value,0.,0.,0.]),i if not self.duplicate else 1))
        if self.after_collect:self.after_collect()
        return {"left":rows,"right":rows}


class ZeroGuardTests(unittest.TestCase):
    def setUp(self):
        self.cfg=configuration()
        self.source=Source()
        self.width=60.
        self.reference={side+"_dd":np.zeros((4,5,3)) for side in ("left","right")}

    def width_reader(self):
        return {"width_mm":self.width,"fresh":True,"valid":True,"monotonic_ns":self.source.now}

    def run_guard(self,**kwargs):
        return guarded_zero_report(self.cfg,self.source,self.width_reader,**kwargs)

    def test_first_connection_explicit_empty_and_all_100_source_frames_saved(self):
        prompts=[]
        report=self.run_guard(confirm=lambda prompt:(prompts.append(prompt) or "EMPTY"))
        self.assertEqual(len(prompts),1)
        self.assertTrue(report["first_session_empty_confirmed"])
        self.assertEqual(report["samples_per_finger"],100)
        self.assertEqual(report["fingers"]["left"]["source_fids"],list(range(100)))
        self.assertEqual(len(report["fingers"]["right"]["samples_raw"]),100)
        self.assertEqual(report["fingers"]["left"]["offset_raw"][2],3.)

    def test_old_unloaded_declaration_does_not_bypass_missing_or_declined_confirmation(self):
        for confirmation in (None,lambda _:"",lambda _:"yes",lambda _:None):
            with self.subTest(confirmation=confirmation):
                with self.assertRaisesRegex(ZeroGuardError,"EMPTY"):
                    self.run_guard(confirm=confirmation)
        self.assertEqual(self.source.collects,0)

    def test_width_out_of_tolerance_refuses_before_sampling_or_confirmation(self):
        self.width=59.
        calls=[]
        with self.assertRaisesRegex(ZeroGuardError,"w_open"):
            self.run_guard(confirm=lambda p:calls.append(p))
        self.assertEqual(calls,[])
        self.assertEqual(self.source.collects,0)

    def test_reference_contact_refuses_even_with_constant_force(self):
        self.source.dd[1][1,2,0]=.1
        with self.assertRaisesRegex(ZeroGuardError,"right DD has contact"):
            self.run_guard(reference=self.reference)
        self.assertEqual(self.source.collects,0)

    def test_reference_check_does_not_ask_for_empty_but_rejects_changed_geometry(self):
        report=self.run_guard(reference=self.reference,confirm=lambda _:self.fail("Unexpected confirmation"))
        self.assertTrue(report["unloaded_reference_present"])
        self.assertFalse(report["first_session_empty_confirmed"])
        self.reference["left_dd"]=np.zeros((2,2,3))
        with self.assertRaisesRegex(ZeroGuardError,"geometry"):
            self.run_guard(reference=self.reference)

    def test_force_variation_rejects_before_any_offset_is_applied(self):
        self.source.normals=np.tile([2.,4.],50)
        with self.assertRaisesRegex(ZeroGuardError,"fluctuation"):
            self.run_guard(reference=self.reference)

    def test_width_is_checked_again_after_sampling(self):
        self.source.after_collect=lambda:setattr(self,"width",45.)
        with self.assertRaisesRegex(ZeroGuardError,"w_open"):
            self.run_guard(reference=self.reference)

    def test_dd_is_checked_again_after_sampling(self):
        self.source.after_collect=lambda:self.source.dd[0].__setitem__((0,0,0),.2)
        with self.assertRaisesRegex(ZeroGuardError,"left DD has contact"):
            self.run_guard(reference=self.reference)

    def test_duplicate_source_fids_cannot_count_as_100_independent_frames(self):
        self.source.duplicate=True
        with self.assertRaisesRegex(ZeroGuardError,"independent"):
            self.run_guard(reference=self.reference)

    def test_force_zero_drift_warning_saves_raw_and_normal_differences(self):
        first=self.run_guard(reference=self.reference)
        self.source.normals=np.full(100,3.2)
        second=self.run_guard(reference=self.reference,previous=first)
        self.assertEqual(len(second["warnings"]),2)
        self.assertAlmostEqual(second["previous_zero_difference"]["left"]["normal_delta_N"],.4)
        self.assertAlmostEqual(second["previous_zero_difference"]["right"]["normal_delta_N"],-.4)
        self.assertEqual(first["fingers"]["left"]["offset_raw"][2],3.)

    def test_source_change_never_presents_incomparable_zero_as_drift(self):
        first=self.run_guard(reference=self.reference)
        self.source.serials["left"]="OTHER"
        second=self.run_guard(reference=self.reference,previous=first)
        self.assertEqual(second["previous_zero_difference"],{})
        self.assertIn("different source",second["warnings"][0])

    def test_unmeasured_settings_and_less_than_100_frames_refuse(self):
        for key,value in (("force_std_max_N","TODO"),("sample_count",99),("timeout_s",61),
                          ("dd_contact_max_fraction",-1),("offset_delta_warn_N",float("nan"))):
            cfg=copy.deepcopy(self.cfg)
            cfg["zero_guard"][key]=value
            self.assertTrue(validate_zero_guard_settings(cfg))
            with self.assertRaises(ZeroGuardError):
                guarded_zero_report(cfg,self.source,self.width_reader,reference=self.reference)


if __name__=="__main__":
    unittest.main()
