"""Offline batch/resume and explicit synthetic plant regression tests."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from bottle_scan.agentic_executor import execute_batch, recover_context
from bottle_scan.agentic_sim import default_T0
from bottle_scan.e1e2_core import defaults
from bottle_scan.session import atomic_json


def cfg():
    result = defaults()
    result['gripper']['min_normalized'] = .336
    result['force']['probe_stable_s'] = .001
    result['force']['squeeze_stable_s'] = .001
    result['hold_s'] = .001
    return result


def fit():
    return {'schema': 'e1e2_fit_v1', 'usable_for_B': True, 'fingers': [
        {'coordinate': 'u', 'slope_px_per_mm': -3.3, 'intercept_px': 64., 'direction_sign': -1},
        {'coordinate': 'u', 'slope_px_per_mm': -3.1, 'intercept_px': 62., 'direction_sign': -1}]}


def plan(cond='A', offsets=(0., 4.), bottle='light'):
    return [dict(id=f'{i+1:05d}', index=i+1, cond=cond, bottle=bottle,
                 delta_mm=float(d), kind='scan', repeat=0) for i, d in enumerate(offsets)]


def ask_default(message):
    return 'MOVE' if 'MOVE' in message else ''


class FakeHardware:
    """Finite fake IO drives inherited real Runner stages, never ROS."""
    def __init__(self, settings, pose=None, fail_lift=False):
        self.cfg = settings
        self.p = default_T0() if pose is None else np.asarray(pose, float).copy()
        self.width = 1.
        self.normal = np.zeros(2)
        self.mode = 'hold'
        self.clock = 1_000_000_000
        self.writer = None
        self.events = []
        self.zero = {'contact_thresholds_sdk': [.05, .05]}
        self.dd_reference = [np.zeros((20, 30, 3), np.float32) for _ in range(2)]
        self.legacy_evidence = {}
        self.fail_lift = fail_lift
        self.stage = None
    def connect(self): self.events.append('connect')
    def close(self): self.events.append('close')
    def check(self): pass
    def ownership(self): self.events.append('ownership')
    def pose(self): return self.p.copy()
    def now(self): self.clock += 1_000_000; return self.clock
    def set_context(self, stage, trial_id=None): self.stage = stage; self.events.append(stage)
    def snapshot(self):
        self.clock += 20_000_000
        return dict(normal=self.normal.copy(), width=self.width, width_stamp=self.clock,
                    width_sequence=self.clock, force_sequences=[self.clock]*2,
                    force_stamps=[self.clock]*2, wrench_zeroed=np.zeros((2, 6)),
                    timestamp_ns=self.clock, pose=self.p.copy(), gripper_motion={'mode': self.mode})
    def wait(self, seconds): self.clock += max(1, int(seconds*1e9))
    def begin_record(self, folder): self.writer = Path(folder); self.writer.mkdir(parents=True, exist_ok=True)
    def end_record(self, error=None):
        if self.writer is not None:
            atomic_json(self.writer / 'recording_status.json', {'complete': not bool(error)})
            self.writer = None
    def zero_unloaded(self): self.events.append('zero'); return copy.deepcopy(self.zero)
    def keyframes(self, reason, gel=False): return {'reason': reason, 'fake': True}
    def open(self): self.events.append('open'); self.width = 1.; self.normal[:] = 0; self.mode = 'open'
    def hold(self): self.events.append('hold'); self.mode = 'hold'
    def close_to_floor(self): self.events.append('close_grip'); self.width = .36; self.normal[:] = .1; self.mode = 'close'
    def pause_probe(self): self.mode = 'hold'
    def force(self, target, tolerance): self.normal[:] = target; self.mode = 'force'
    def probe_force(self, target, tolerance): self.force(target, tolerance)
    def probe_window(self, since):
        dd = np.zeros((20, 30, 3), np.float32)
        dd[6:12, 10:18, 0] = 1.
        return [dict(median_dd=dd, frame_ns=[since+i for i in range(5)], frame_fids=list(range(5))) for _ in range(2)]
    def move(self, target, speed):
        self.events.append(('move', np.asarray(target).tolist()))
        if self.fail_lift and self.stage == 'LIFT': raise RuntimeError('injected lift failure')
        self.p = np.asarray(target).copy()
    def motion_done(self): return True
    def stop_arm(self): self.events.append('stop_arm')
    def recovery_open(self): self.events.append('recovery_open'); self.open()


class ExecutorTests(unittest.TestCase):
    def test_completed_batch_never_connects_hardware(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            execute_batch(root, plan(), cfg(), None, {}, simulate=True, say=lambda _: None)
            def forbidden(_): self.fail('hardware factory must not run')
            result = execute_batch(root, plan(), cfg(), None, {}, simulate=True, say=lambda _: None, hardware_factory=forbidden)
            self.assertTrue(result['complete'])
            self.assertEqual(result['completed_count'], 0)

    def test_first_batch_captures_T0_and_inherited_stages(self):
        with tempfile.TemporaryDirectory() as td:
            hw = FakeHardware(cfg())
            result = execute_batch(Path(td), plan(offsets=(0.,)), cfg(), None, {},
                                   hardware_factory=lambda _: hw, ask=ask_default, say=lambda _: None)
            self.assertTrue(result['complete'])
            self.assertTrue(np.allclose(result['context']['T0'], default_T0()))
            self.assertIn('LIFT', hw.events)
            row = json.loads((Path(td)/'trials/00001/meta.json').read_text())
            self.assertIn('actual_delta_mm', row['probe_initial'])
            self.assertTrue(row['recording_complete'])
            self.assertTrue(row['hold_complete'])

    def test_shared_T0_returns_from_prior_open_retreat_and_no_retouch_baseline(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            hw1 = FakeHardware(cfg())
            first = execute_batch(root/'one', plan(offsets=(4.,)), cfg(), None, {},
                                  hardware_factory=lambda _: hw1, ask=ask_default, say=lambda _: None)
            context = first['context']; context['continuous_campaign'] = True
            hw2 = FakeHardware(cfg(), pose=hw1.p)
            questions = []
            def ask(message): questions.append(message); return ask_default(message)
            second = execute_batch(root/'two', plan(offsets=(6.,)), cfg(), None, context,
                                   hardware_factory=lambda _: hw2, authorize=False, ask=ask, say=lambda _: None)
            self.assertTrue(second['complete'])
            self.assertEqual(first['context']['T0'], second['context']['T0'])
            self.assertIn('BATCH_RETURN_T0', hw2.events)
            self.assertEqual(hw2.events.count('close_grip'), 1)
            self.assertFalse(any('MOVE' in q for q in questions))
            moves = [e for e in hw2.events if isinstance(e, tuple) and e[0] == 'move']
            self.assertTrue(np.allclose(moves[0][1], default_T0()))

    def test_resume_requires_MOVE_even_with_persisted_continuous_context(self):
        with tempfile.TemporaryDirectory() as td:
            context = {'T0': default_T0().tolist(), 'T0_accepted_unix_ns': 1, 'continuous_campaign': True}
            hw = FakeHardware(cfg())
            questions = []
            def ask(message): questions.append(message); return ask_default(message)
            execute_batch(Path(td), plan(offsets=(0.,)), cfg(), None, context,
                          hardware_factory=lambda _: hw, ask=ask, say=lambda _: None, authorize=True)
            self.assertEqual(sum('MOVE' in q for q in questions), 1)

    def test_unsafe_airborne_crash_persists_and_blocks_reconnect(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            hw = FakeHardware(cfg(), fail_lift=True)
            with self.assertRaisesRegex(RuntimeError, 'injected'):
                execute_batch(root, plan(offsets=(0.,)), cfg(), None, {},
                              hardware_factory=lambda _: hw, ask=ask_default, say=lambda _: None)
            context = recover_context(root)
            self.assertTrue(context['unsafe_recovery'])
            self.assertTrue(context['airborne'])
            self.assertIn('hold', hw.events)
            self.assertNotIn('recovery_open', hw.events)
            def forbidden(_): self.fail('unsafe resume connected')
            with self.assertRaisesRegex(RuntimeError, '空中'):
                execute_batch(root, plan(offsets=(0.,)), cfg(), None, context,
                              hardware_factory=forbidden, ask=ask_default, say=lambda _: None)

    def test_unknown_start_position_is_not_automatically_moved(self):
        with tempfile.TemporaryDirectory() as td:
            context = {'T0': default_T0().tolist(), 'T0_accepted_unix_ns': 1}
            bad = default_T0(); bad[0, 3] += 30.
            hw = FakeHardware(cfg(), bad)
            with self.assertRaisesRegex(RuntimeError, '不在原 T0'):
                execute_batch(Path(td), plan(), cfg(), None, context,
                              hardware_factory=lambda _: hw, ask=ask_default, say=lambda _: None)
            self.assertFalse(any(isinstance(e, tuple) and e[0] == 'move' for e in hw.events))

    def test_realistic_retreat_feedback_residual_does_not_need_exact_collinearity(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            hw1 = FakeHardware(cfg())
            first = execute_batch(root/'one', plan(offsets=(4.,)), cfg(), None, {},
                                  hardware_factory=lambda _: hw1, ask=ask_default, say=lambda _: None)
            context = first['context']; context['continuous_campaign'] = True
            # Actual measured pose has harmless transverse servo residual.
            actual = hw1.p.copy(); actual[2, 3] += .8
            context['last_safe_pose'] = actual.tolist()
            hw2 = FakeHardware(cfg(), actual)
            result = execute_batch(root/'two', plan(offsets=(6.,)), cfg(), None, context,
                                   hardware_factory=lambda _: hw2, authorize=False, ask=ask_default, say=lambda _: None)
            self.assertTrue(result['complete'])
            self.assertIn('BATCH_RETURN_T0', hw2.events)

    def test_invalid_context_rejected_before_factory(self):
        with tempfile.TemporaryDirectory() as td:
            bad = default_T0(); bad[0, 0] = 7.
            with self.assertRaisesRegex(ValueError, '刚体'):
                execute_batch(Path(td), plan(), cfg(), None, {'T0': bad.tolist()},
                              hardware_factory=lambda _: self.fail('invalid T0 connected'), say=lambda _: None)

    def test_expired_budget_does_not_create_hardware(self):
        with tempfile.TemporaryDirectory() as td:
            result = execute_batch(Path(td), plan(), cfg(), None, {'deadline_unix_ns': 1},
                                   hardware_factory=lambda _: self.fail('expired budget connected'), say=lambda _: None)
            self.assertTrue(result['deadline_reached'])
            self.assertTrue(result['quit'])
            self.assertFalse(result['complete'])
            self.assertEqual(result['completed_count'], 0)

    def test_budget_expiry_finishes_current_trial_then_stops(self):
        with tempfile.TemporaryDirectory() as td:
            wall = [1]
            hw = FakeHardware(cfg())
            def ask(message):
                if 'q本次后退出' in message:
                    wall[0] = 2000
                return ask_default(message)
            with patch('bottle_scan.agentic_executor.time.time_ns', side_effect=lambda: wall[0]):
                result = execute_batch(Path(td), plan(), cfg(), None, {'deadline_unix_ns': 1000},
                                       hardware_factory=lambda _: hw, ask=ask, say=lambda _: None)
            self.assertTrue(result['deadline_reached'])
            self.assertFalse(result['complete'])
            self.assertEqual(result['completed_count'], 1)
            self.assertEqual(hw.events.count('LIFT'), 1)
            row = json.loads((Path(td)/'trials/00001/meta.json').read_text())
            self.assertTrue(row['complete'])
            self.assertTrue(row['hold_complete'])
            self.assertFalse((Path(td)/'trials/00002').exists())

    def test_simulation_budget_expiry_also_stops_between_trials(self):
        with tempfile.TemporaryDirectory() as td:
            wall = [1]; c = cfg(); c['agentic'] = {'simulation_interactive': True}
            def ask(message): wall[0] = 2000; return '1'
            with patch('bottle_scan.agentic_executor.time.time_ns', side_effect=lambda: wall[0]):
                result = execute_batch(Path(td), plan(), c, None, {'deadline_unix_ns': 1000},
                                       simulate=True, ask=ask, say=lambda _: None)
            self.assertTrue(result['deadline_reached'])
            self.assertEqual(result['completed_count'], 1)
            self.assertFalse(result['complete'])

    def test_resume_runtime_deadline_and_authorization_override_checkpoint(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            atomic_json(root/'session_meta.json', {'agentic_context': {'deadline_unix_ns': 1, 'continuous_campaign': True}})
            recovered = recover_context(root, {'deadline_unix_ns': 2000, 'continuous_campaign': False})
            self.assertEqual(recovered['deadline_unix_ns'], 2000)
            self.assertFalse(recovered['continuous_campaign'])

    def test_pause_request_on_real_label_does_not_execute_next_trial(self):
        with tempfile.TemporaryDirectory() as td:
            hw = FakeHardware(cfg())
            labels = iter(['q', '1'])
            def ask(message):
                return next(labels) if 'q本次后退出' in message else ask_default(message)
            result = execute_batch(Path(td), plan(), cfg(), None, {},
                                   hardware_factory=lambda _: hw, ask=ask, say=lambda _: None)
            self.assertTrue(result['quit'])
            self.assertFalse(result['complete'])
            self.assertEqual(result['completed_count'], 1)
            self.assertEqual(hw.events.count('LIFT'), 1)

    def test_mixed_condition_plan_swaps_correction_without_delta_leak(self):
        with tempfile.TemporaryDirectory() as td:
            items = plan(offsets=(0., 0.)); items[1]['cond'] = 'B'
            calls = []
            class FakeCorrection:
                def decide(self, status, descriptors, bilateral):
                    calls.append((status, descriptors, bilateral))
                    return dict(accepted=True, e_mm=0., delta_mm=0., edges=[], reason='test')
            with patch('bottle_scan.agentic_executor._correction', return_value=FakeCorrection()):
                result = execute_batch(Path(td), items, cfg(), fit(), {}, hardware_factory=FakeHardware,
                                       ask=ask_default, say=lambda _: None)
            self.assertTrue(result['complete'])
            self.assertEqual(len(calls), 1)
            self.assertEqual(len(calls[0]), 3)

    def test_q_pauses_after_label_and_resume_keeps_completed_trial(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); c = cfg(); c['agentic'] = {'simulation_interactive': True}
            replies = iter(['q', '1'])
            result = execute_batch(root, plan(), c, None, {}, simulate=True,
                                   ask=lambda _: next(replies), say=lambda _: None)
            self.assertTrue(result['quit']); self.assertFalse(result['complete'])
            first_bytes = (root/'trials/00001/meta.json').read_bytes()
            resumed = execute_batch(root, plan(), c, None, result['context'], simulate=True,
                                    ask=lambda _: '1', say=lambda _: None)
            self.assertTrue(resumed['complete']); self.assertEqual(resumed['completed_count'], 1)
            self.assertEqual(first_bytes, (root/'trials/00001/meta.json').read_bytes())

    def test_zero_reference_crosses_batch_boundary(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); context = {}
            for i in range(4):
                result = execute_batch(root/str(i), plan(offsets=(0.,)), cfg(), None, context,
                                       simulate=True, say=lambda _: None)
                context = result['context']
            row = json.loads((root/'3/trials/00001/meta.json').read_text())
            self.assertTrue(row['zero_drift']['checked'])
            self.assertEqual(len(context['zero_samples']), 4)
            self.assertEqual(np.asarray(context['zero_baseline_px']).shape, (2, 2))

    def test_simulation_records_are_explicit_and_B_can_correct(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            result = execute_batch(root, plan('B', offsets=(-10., 10.)), cfg(), fit(), {},
                                   simulate=True, say=lambda _: None)
            self.assertTrue(result['complete'])
            meta = json.loads((root/'session_meta.json').read_text())
            self.assertIs(meta['simulated'], True)
            for folder in (root/'trials').iterdir():
                row = json.loads((folder/'meta.json').read_text())
                self.assertTrue(row['simulated'])
                self.assertTrue(row['lifted'])
                self.assertGreater(row['regrasps'], 0)
                self.assertTrue((folder/'synthetic_streams.npz').exists())

    def test_plan_and_mode_mutation_blocked_before_hardware(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            execute_batch(root, plan(), cfg(), None, {}, simulate=True, say=lambda _: None)
            with self.assertRaisesRegex(ValueError, '不能更换'):
                execute_batch(root, plan(offsets=(0., 8.)), cfg(), None, {}, simulate=True, say=lambda _: None)
            with self.assertRaisesRegex(ValueError, '不能更换'):
                execute_batch(root, plan(), cfg(), None, {}, simulate=False, say=lambda _: None)


if __name__ == '__main__':
    unittest.main()
