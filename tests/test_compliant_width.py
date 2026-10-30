"""Offline fake-I/O checks for the explicitly modelled compression interval."""
import copy,json,tempfile,types,unittest
from pathlib import Path
from unittest.mock import Mock,patch
from contextlib import nullcontext
import numpy as np
from bottle_scan.config import demo_config,validate_config
from bottle_scan.hardware_gripper import WidthCalibration,RosGripperController
from bottle_scan.wizard_width import compliant_width_mapping
from bottle_scan.wizard_dimensions import bottle_dimensions
from bottle_scan.runtime import Runner
from tests.test_hardware_gripper import CommandRos,FakeLegacy

class CompliantWidthTests(unittest.TestCase):
    def setUp(self):
        self.cfg=demo_config();self.cfg['hardware']['public_ros']['raw_normalized_min']=.3
        self.model=compliant_width_mapping({'normalized':1.,'width_mm':100.},{'normalized':.4,'width_mm':30.},self.cfg)
        self.cal=WidthCalibration(self.model['points'])
    def test_merge_regions_keeps_body_and_encloses_entire_upper(self):
        d=bottle_dimensions({'body':{'diameter_mm':30,'height_mm':60},'neck':{'diameter_mm':39,'height_mm':3},'cap':{'diameter_mm':41,'height_mm':18}})
        self.assertEqual(d,{'body':{'diameter_mm':30.,'height_mm':60.},'upper':{'diameter_mm':41.,'height_mm':21.}})
        self.assertEqual(bottle_dimensions(d),d)
    def test_bounded_model_is_not_an_extra_measured_point(self):
        self.assertEqual(len(self.model['measured_points']),2)
        self.assertFalse(self.model['linearity_check_possible'])
        self.assertEqual(self.cal.width_kind(.39),'estimated_equivalent_compression')
        self.assertEqual(self.cal.width_kind(.4),'measured_anchor_interpolation')
        self.assertLess(self.cal.to_mm(.39),30.)
        self.assertEqual(self.model['precontact_min_mm'],30.)
        self.assertGreaterEqual(self.model['estimated_normalized_range'][0],.3)
        self.assertLessEqual(30-self.model['estimated_width_range_mm'][0],2*self.cfg['geometry']['gripper']['pad_depth_mm'])
        with self.assertRaises(ValueError):self.cal.to_mm(self.model['estimated_normalized_range'][0]-.001)
        self.assertEqual(WidthCalibration(self.cal.points).compression_model,self.cal.compression_model)
    def test_model_cannot_relax_original_raw_floor_or_gel_thickness(self):
        points=copy.deepcopy(self.model['points']);points[0]['max_compression_mm']=.1
        with self.assertRaises(ValueError):WidthCalibration(points)
        c=copy.deepcopy(self.cfg);c['hardware']['public_ros']['raw_normalized_min']=.4
        with self.assertRaisesRegex(ValueError,'没有允许'):compliant_width_mapping({'normalized':1.,'width_mm':100.},{'normalized':.4,'width_mm':30.},c)
    def test_config_accepts_two_regions_and_rejects_model_that_exceeds_gel(self):
        c=copy.deepcopy(self.cfg);c['geometry']['bottle']['segments']=[{'name':'body','radius_mm':15.,'axial_min_mm':-30.,'axial_max_mm':30.},{'name':'upper','radius_mm':20.5,'axial_min_mm':30.,'axial_max_mm':51.}]
        c['hardware']['gripper_width_calibration']=self.model['points'];c['gripper']['precontact_min_mm']=30.
        self.assertEqual(validate_config(c),[])
        c['geometry']['gripper']['pad_depth_mm']=.1
        self.assertTrue(any('compression' in e for e in validate_config(c)))
    def test_complete_body_cache_continues_without_neck_cap_or_new_actuation(self):
        from bottle_scan.wizard_real import capture_widths
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);folder=root/'width_body';folder.mkdir()
            result={'normalized':.4,'width_mm':30.,'normal_signs':[1.,1.]}
            (folder/'calibration.json').write_text(json.dumps({'result':result,'release_complete':True,'recording_complete':True,
                'commands':[{'measured_normalized':1.},{'measured_normalized':.999},{'measured_normalized':.4}]}))
            (folder/'recording_status.json').write_text(json.dumps({'complete':True}))
            state={'dimensions':{'body':{'diameter_mm':30.,'height_mm':60.},'upper':{'diameter_mm':41.,'height_mm':21.}}}
            reader=Mock();reader.gripper.return_value={'normalized':1.};ui=Mock()
            with patch('bottle_scan.wizard_real.WidthBootstrap') as bootstrap:
                model=capture_widths(self.cfg,state,reader,ui,root)
            bootstrap.assert_not_called()
            self.assertLess(self.cfg['gripper']['min_mm'],30.)
            self.assertEqual(self.cfg['gripper']['precontact_min_mm'],30.)
            self.assertTrue(state['width_model_confirmation']['operator_confirmed'])
            self.assertEqual(len(model['measured_points']),2)

    def make_controller(self,norm=.4,force=(.3,.3)):
        self.now=10_000_000_000;self.ros=CommandRos();self.legacy=FakeLegacy();self.legacy.proposal=390.
        self.force={'normal_N':list(force),'monotonic_ns':self.now,'source_sequence':[1,1]}
        ctrl=RosGripperController('right',self.cal,lambda:self.force,min_width_mm=self.model['estimated_width_range_mm'][0],max_width_mm=100.,max_force_N=4.,force_deadband_N=.1,force_step_counts=1,force_settle_s=.1,max_age_s=.25,rospy_module=self.ros,message_type=types.SimpleNamespace,clock_ns=lambda:self.now,start_worker=False,legacy_functions=self.legacy,compression_contact_threshold=.2).connect()
        self.addCleanup(ctrl.close);self.ros.callback(types.SimpleNamespace(data=norm));return ctrl
    def test_position_closure_cannot_enter_compression_region(self):
        ctrl=self.make_controller()
        with self.assertRaisesRegex(ValueError,'Position closure'):ctrl.command_close(29.,2.)
        self.assertEqual(self.ros.published,[])
    def test_force_mode_requires_both_contacts(self):
        ctrl=self.make_controller(force=(.3,0.))
        with self.assertRaisesRegex(RuntimeError,'Bilateral'):ctrl.set_force_target(2.)
        self.assertEqual(self.ros.published,[])
    def test_force_closure_is_bounded_recorded_as_estimate_and_lost_contact_holds(self):
        ctrl=self.make_controller();ctrl.set_force_target(2.);ctrl._tick()
        self.assertAlmostEqual(self.ros.published[-1],.39)
        self.now+=200_000_000;self.force.update(monotonic_ns=self.now,source_sequence=[2,2],normal_N=[.1,.3]);self.ros.callback(types.SimpleNamespace(data=.39))
        snap=ctrl.snapshot();self.assertTrue(snap['width_is_estimate']);self.assertFalse(snap['width_calibrated'])
        ctrl._tick();self.assertEqual(ctrl.snapshot()['controller_state'],'hold');self.assertAlmostEqual(self.ros.published[-1],.39)
        self.assertTrue(any(x.metadata.get('width_is_estimate') for x in ctrl.drain_samples() if x.stream=='gripper_command'))
    def runner(self,contact):
        r=Mock();r.cfg=copy.deepcopy(self.cfg);r.cfg['gripper'].update(precontact_min_mm=30.,min_mm=24.)
        r.phase.return_value=nullcontext();r._healthy_snapshot.return_value={'width_mm':100.}
        r._closure_floor.return_value=(24.,[]);r.threshold=np.array([.2,.2])
        r.meta={'auto':{'contact_times_ns':[None,None],'contact_widths_mm':[None,None]}}
        r.tick.return_value={'width_mm':30.,'normal_N':[.3,.3] if contact else [0.,0.],'timestamp_ns':1000}
        return r
    def test_runtime_unlocks_compression_only_after_bilateral_contact(self):
        r=self.runner(True);self.assertTrue(Runner.close_contact(r))
        self.assertEqual([x.args[0] for x in r.backend.set_gripper_min_width.call_args_list],[30.,24.])
        r.backend.command_close.assert_called_once_with(30.,r.cfg['gripper']['close_speed_mm_s'])
        self.assertTrue(r.meta['auto']['compression_authorized_after_bilateral_contact'])
    def test_runtime_missing_contact_never_unlocks_compression(self):
        r=self.runner(False);self.assertFalse(Runner.close_contact(r))
        r.backend.set_gripper_min_width.assert_called_once_with(30.)
        self.assertEqual(r.meta['auto']['closure_stop_reason'],'measured_contact_floor_without_bilateral_contact')

if __name__=='__main__':unittest.main()
