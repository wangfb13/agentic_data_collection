import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import opening_repeat
from bottle_scan.config import demo_config, validate_config, unresolved
from bottle_scan.force_control import force_report_binding
from bottle_scan.session import Session


class OpeningRepeatTests(unittest.TestCase):
    def test_dry_run_never_imports_hardware_bridge(self):
        import builtins
        original = builtins.__import__
        def guarded(name, *args, **kwargs):
            if name in ('bridge_ros_public', 'rospy'):
                raise AssertionError('Dry-run must not import ROS')
            return original(name, *args, **kwargs)
        with patch('builtins.__import__', side_effect=guarded), contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(opening_repeat.main(['--dry-run', '--demo']), 0)
        self.assertIn('"repetitions": 10', output.getvalue())
        self.assertIn('DRY-RUN', output.getvalue())

    def test_simulation_reuses_close_probe_release_without_arm_motion(self):
        from bottle_scan.backend import SimBackend
        cfg = demo_config()
        cfg['opening_repeat']['repetitions'] = 2
        cfg['gripper']['close_speed_mm_s'] = 20.
        cfg['close']['max_overtravel_mm'] = 3.  # Synthetic fixture only.
        with tempfile.TemporaryDirectory() as tmp, patch.object(opening_repeat,'load_config',return_value=cfg), \
                patch.object(SimBackend,'move_linear',side_effect=AssertionError('No arm motion')), \
                contextlib.redirect_stdout(io.StringIO()):
            path = Path(tmp)/'report.json'
            self.assertEqual(opening_repeat.main(['--simulate','--output',str(path)]),0)
            report = json.loads(path.read_text())
            self.assertTrue(report['completed'])
            self.assertTrue(report['simulated'])
            self.assertEqual(report['operator_confirmation'], 'SIMULATION_ONLY')
            self.assertEqual(report['contact_normalized']['left']['count'],2)
            self.assertEqual(len(report['trials']),2)
            self.assertFalse(report['opening_readout_resolution']['hardware_encoder_resolution_verified'])
            self.assertIn('synthetic',report['opening_readout_resolution']['sources'][0])
            self.assertTrue(path.with_suffix('.csv').is_file())
            meta = json.loads((path.parent/'report_data/cycle_001/meta.json').read_text())
            self.assertEqual([s['name'] for s in meta['stages']],['RELEASE','CLOSE','PROBE','RELEASE'])

    def test_failed_contact_holds_without_automatic_release(self):
        from bottle_scan.runtime import Runner, SafetyFault
        from bottle_scan.backend import SimBackend
        cfg=demo_config(); cfg['gripper']['close_speed_mm_s']=20.
        with tempfile.TemporaryDirectory() as tmp:
            session=Session.create(Path(tmp)/'session',cfg,cfg['T0'],simulated=True)
            backend=SimBackend(cfg,session.T0)
            runner=Runner(session,backend,single_step=False,unattended=True)
            folder=Path(tmp)/'cycle';folder.mkdir()
            with patch.object(runner,'close_contact',return_value=False), self.assertRaises(SafetyFault):
                opening_repeat.run_cycle(runner,folder,1)
            self.assertEqual(sum(call[0]=='open' for call in backend.calls),1)
            self.assertEqual(sum(call[0]=='move' for call in backend.calls),0)
            self.assertEqual(json.loads((folder/'meta.json').read_text())['status'],'aborted')

    def test_first_real_step_requires_completed_operator_pass_and_same_binding(self):
        cfg=demo_config()
        with tempfile.TemporaryDirectory() as tmp:
            session=Session.create(Path(tmp)/'session',cfg,cfg['T0'],simulated=False)
            with self.assertRaisesRegex(ValueError,'single-step'):
                opening_repeat.require_first_single_step(session,False)
            opening_repeat.require_first_single_step(session,True)
            session.meta['opening_repeat_first_single_step']={'operator_confirmation':'SIMULATION_ONLY','simulated':True,'binding':force_report_binding(cfg)}
            with self.assertRaises(ValueError):opening_repeat.require_first_single_step(session,False)
            session.meta['opening_repeat_first_single_step'].update(operator_confirmation='PASS',simulated=False)
            report_path=Path(tmp)/'verified_report.json'
            report={'schema':'bottle_opening_repeat_v2','completed':True,'cleanup_complete':True,
                    'simulated':False,'operator_confirmation':'PASS','binding':force_report_binding(cfg),
                    'repetitions':1,'trials':[{'status':'completed'}]}
            session.meta['opening_repeat_first_single_step']['report_path']=str(report_path)
            for key,value in [('completed',False),('cleanup_complete',False),('simulated',True),
                              ('operator_confirmation','SIMULATION_ONLY'),('binding',{})]:
                damaged=copy.deepcopy(report);damaged[key]=value
                report_path.write_text(json.dumps(damaged))
                with self.subTest(key=key),self.assertRaises(ValueError):
                    opening_repeat.require_first_single_step(session,False)
            report_path.write_text(json.dumps(report))
            opening_repeat.require_first_single_step(session,False)
            session.cfg['gripper']['close_speed_mm_s']+=1.
            with self.assertRaises(ValueError):opening_repeat.require_first_single_step(session,False)

    def test_first_pass_is_committed_only_after_csv_report_and_cleanup_succeed(self):
        # The entire hardware boundary and cycle are fake. This tests commit
        # ordering only; no ROS node, device or command publisher is created.
        for fault in ('csv', 'cleanup', 'report', None):
            with self.subTest(fault=fault), tempfile.TemporaryDirectory() as tmp:
                cfg=demo_config();cfg['demo']=False;cfg['opening_repeat']['repetitions']=1
                session=Session.create(Path(tmp)/'session',cfg,cfg['T0'],simulated=False)
                destination=Path(tmp)/'report.json'
                events=[]
                class FakeBackend:
                    def prepare_zero(self,*args,**kwargs):pass
                    def park_observation(self):return {}
                    def monotonic_ns(self):return 1
                    def provenance(self):return {'fake_unit_test':True}
                    def close(self):
                        events.append('close')
                        if fault=='cleanup':raise OSError('injected cleanup failure')
                fake_runner=types.SimpleNamespace(_healthy_snapshot=lambda:{'tcp':cfg['T0']})
                contacts=[{'finger':side,'normalized':.5,'width_mm':50.,'source':'fake_unit_test',
                           'position_minus_contact_s':0.} for side in ('left','right')]
                trial={'trial_id':'fake','status':'completed','probe_not_settled':False,'contacts':contacts}
                original_open=Path.open
                original_atomic=opening_repeat.atomic_json
                def guarded_open(path,*args,**kwargs):
                    if path==destination.with_suffix('.csv'):
                        events.append('csv')
                        if fault=='csv':raise OSError('injected CSV failure')
                    return original_open(path,*args,**kwargs)
                def guarded_atomic(path,value):
                    if Path(path)==destination and value.get('cleanup_complete'):
                        events.append('report')
                        if fault=='report':raise OSError('injected report failure')
                    return original_atomic(path,value)
                with contextlib.ExitStack() as stack:
                    stack.enter_context(patch.object(opening_repeat,'require_real_ready',return_value=None))
                    stack.enter_context(patch('bottle_scan.backend.load_hardware',return_value=FakeBackend()))
                    stack.enter_context(patch('bottle_scan.runtime.Runner',return_value=fake_runner))
                    stack.enter_context(patch('bottle_scan.park_guard.ParkGuard.check',return_value={}))
                    stack.enter_context(patch.object(opening_repeat,'run_cycle',return_value=(trial,[])))
                    stack.enter_context(patch.object(opening_repeat,'summarize',return_value={}))
                    stack.enter_context(patch('builtins.input',side_effect=['MOVE','PASS']))
                    stack.enter_context(patch.object(Path,'open',guarded_open))
                    stack.enter_context(patch.object(opening_repeat,'atomic_json',side_effect=guarded_atomic))
                    stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                    command=['--real','--single-step','--session',str(session.path),'--output',str(destination)]
                    if fault is None:
                        self.assertEqual(opening_repeat.main(command),0)
                    else:
                        with self.assertRaises(OSError):opening_repeat.main(command)
                reloaded=Session(session.path)
                if fault is None:
                    self.assertIn('opening_repeat_first_single_step',reloaded.meta)
                    opening_repeat.require_first_single_step(reloaded,False)
                    self.assertEqual(events,['csv','close','report'])
                else:
                    self.assertNotIn('opening_repeat_first_single_step',reloaded.meta)
                    with self.assertRaises(ValueError):opening_repeat.require_first_single_step(reloaded,False)

    def test_caliper_fit_preserves_knots_and_refuses_reversed_mapping(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'pairs.csv'
            path.write_text('normalized,width_mm\n0.3,20\n0.6,50\n1.0,90\n')
            fit=opening_repeat.fit_manual_pairs(path)
            self.assertAlmostEqual(fit['slope_mm_per_normalized'],100.)
            self.assertAlmostEqual(fit['intercept_mm'],-10.)
            self.assertLess(fit['rmse_mm'],1e-10)
            self.assertFalse(fit['automatic_config_update'])
            path.write_text('normalized,width_mm\n0.3,50\n0.6,20\n')
            with self.assertRaises(ValueError):opening_repeat.fit_manual_pairs(path)

    def test_contact_event_alignment_records_skew_and_rejects_stale(self):
        cfg=demo_config()
        meta={'trial_id':'case','status':'completed','auto':{'contact_times_ns':[1000000000,1020000000],'probe_not_settled':False}}
        records=[{'timestamp_ns':1010000000,'normalized':.5,'width_mm':50.,'source':'native_ROS_normalized_feedback'}]
        report=opening_repeat.cycle_measurements(meta,records,cfg)
        self.assertAlmostEqual(report['contacts'][0]['position_minus_contact_s'],.01)
        self.assertAlmostEqual(report['contacts'][1]['position_minus_contact_s'],-.01)
        records[0]['timestamp_ns']=2000000000
        with self.assertRaisesRegex(ValueError,'fresh'):opening_repeat.cycle_measurements(meta,records,cfg)

    def test_added_quality_and_writer_settings_fail_closed(self):
        for section,key,value in [('probe','settle_s',0),('probe','timeout_s',.1),
                                  ('probe','window_frames',True),('hold','window_s',3.),
                                  ('lower','unload_ratio',1.),('lower','vertical_mapping_confirmed','yes'),
                                  ('preparation','height_crosscheck_tolerance_mm',0)]:
            cfg=demo_config();cfg[section][key]=value
            self.assertTrue(any(section+'.'+key in e for e in validate_config(cfg)),(section,key))
        cfg=demo_config();cfg['recording']['writer']['queue_items']=True
        self.assertTrue(any('queue_items' in e for e in validate_config(cfg)))

    def test_explicit_optional_calibration_sentinel_does_not_waive_todos(self):
        cfg=demo_config()
        cfg['tactile'].update(pixel_pitch_mm='unavailable',centroid_drift_mm='unavailable',cop_plane_z_mm='unavailable')
        self.assertEqual(validate_config(cfg),[])
        self.assertEqual(unresolved(cfg),[])
        cfg['tactile']['centroid_drift_px']='TODO_measured_limit'
        self.assertIn('tactile.centroid_drift_px',unresolved(cfg))
        self.assertTrue(any('centroid_drift_px' in x for x in validate_config(cfg)))
        cfg['tactile']['centroid_drift_px']=5.
        cfg['tactile']['pixel_pitch_mm']='TODO'
        self.assertTrue(any('pixel_pitch_mm' in x for x in validate_config(cfg)))


if __name__=='__main__':unittest.main()
