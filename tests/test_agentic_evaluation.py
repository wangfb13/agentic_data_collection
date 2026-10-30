"""Offline stopping evidence; real metadata summaries, no hardware imports."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from bottle_scan.agentic_evaluation import prospective_B_evidence


class ProspectiveBEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.settings = {'validation': {'adaptive_scope': False, 'offsets_mm': [-3.25, 0., 7.125],
                                        'success_target': .8},
                         'learning': {'min_B_per_cell': 1, 'min_B_batches': 1},
                         'policy': {'gains': [.5, .75, 1.]}}
        self.report = {'best_model': {'model_id': 'current', 'usable_for_B': True}}
        self.batches = []

    def row(self, offset, bottle='light', **change):
        return dict({'cond': 'B', 'kind': 'scan', 'delta_mm': offset, 'bottle': bottle,
                     'label': 'normal', 'complete': True, 'recording_complete': True,
                     'lifted': True, 'hold_complete': True, 'dropped': False, 'regrasps': 0}, **change)

    def full(self, **change):
        return [self.row(offset, bottle, **change) for offset in self.settings['validation']['offsets_mm']
                for bottle in ('light', 'heavy')]

    def batch(self, rows, gain=1., model='current', **change):
        index = len(self.batches) + 1
        batch = {'id': str(index), 'path': f'batches/{index}', 'phase': 'explore',
                 'status': 'complete', 'model_id': model,
                 'candidate': {'cond': 'B', 'policy': {'gain': gain}}}
        batch.update(change)
        for i, row in enumerate(rows):
            path = self.root / batch['path'] / 'trials' / str(i) / 'meta.json'
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(dict(row, id=str(i))))
        self.batches.append(batch)
        return batch

    def evidence(self):
        return prospective_B_evidence(self.report, self.batches, self.root, self.settings, None)

    def test_zero_only_cannot_approve_freezing(self):
        self.batch([self.row(0, bottle) for bottle in ('light', 'heavy')])
        result = self.evidence()
        self.assertFalse(result['eligible'])
        self.assertIsNone(result['chosen_gain'])
        self.assertIsNone(result['success_rate'])
        self.assertEqual(len(result['gains'][0]['missing_cells']), 4)

    def test_old_models_and_validation_batches_never_supply_coverage(self):
        self.batch(self.full(), model='old')
        self.batch(self.full(), phase='validation')
        self.batch(self.full(), status='paused')
        self.batch(self.full(), candidate={'cond': 'A', 'policy': {'gain': 1.}})
        self.assertEqual(self.evidence()['gains'], [])

    def test_different_gains_cannot_fill_each_others_missing_cells(self):
        rows = self.full()
        self.batch(rows[:3], gain=.75)
        self.batch(rows[3:], gain=1.)
        result = self.evidence()
        self.assertFalse(result['eligible'])
        self.assertIsNone(result['chosen_gain'])
        self.assertEqual(len(result['gains']), 2)

    def test_offsets_use_numeric_tolerance_not_robot_arrival_tolerance(self):
        rows = self.full()
        rows[0]['delta_mm'] += .1
        self.batch(rows)
        self.assertFalse(self.evidence()['eligible'])
        self.batches.clear()
        rows = self.full()
        rows[0]['delta_mm'] += 1e-9
        self.batch(rows)
        self.assertTrue(self.evidence()['eligible'])

    def test_invalid_incomplete_unrecorded_and_monitors_do_not_count(self):
        for change in ({'label': 'invalid'}, {'label': None}, {'complete': False},
                       {'recording_complete': False}, {'kind': 'zero_monitor'}, {'cond': 'A'}):
            with self.subTest(change=change):
                self.batches.clear()
                rows = self.full()
                rows[0].update(change)
                self.batch(rows)
                self.assertFalse(self.evidence()['eligible'])

    def test_both_bottles_required(self):
        self.batch([row for row in self.full() if row['bottle'] == 'light'])
        self.assertFalse(self.evidence()['eligible'])

    def test_unknown_and_unauthorized_gains_do_not_count(self):
        for gain in (None, True, .6, float('nan'), '1'):
            self.batch(self.full(), gain=gain)
        result = self.evidence()
        self.assertFalse(result['eligible'])
        self.assertEqual(result['gains'], [])
        self.assertEqual(result['excluded']['unknown_gain_batches'], 5)

    def test_bottle_weights_not_trial_counts_determine_each_offset_rate(self):
        rows = self.full()
        rows[1].update(dropped=True)
        rows.extend([self.row(-3.25, 'light') for _ in range(30)])
        self.batch(rows)
        result = self.evidence()
        self.assertFalse(result['eligible'])
        self.assertEqual(result['gains'][0]['per_offset'][0]['success_rate'], .5)
        self.assertAlmostEqual(result['success_rate'], 2.5 / 3)

    def test_each_offset_must_pass_even_if_aggregate_passes(self):
        rows = self.full()
        rows[1].update(dropped=True)
        self.batch(rows)
        result = self.evidence()
        self.assertGreater(result['success_rate'], .8)
        self.assertFalse(result['eligible'])

    def test_complete_gain_wins_over_zero_only_gain(self):
        self.batch(self.full(regrasps=1), gain=.75)
        self.batch([self.row(0., bottle) for bottle in ('light', 'heavy')], gain=1.)
        result = self.evidence()
        self.assertTrue(result['eligible'])
        self.assertEqual(result['chosen_gain'], .75)

    def test_fewer_regrasps_break_equal_success_tie(self):
        self.batch(self.full(regrasps=2), gain=1.)
        self.batch(self.full(regrasps=1), gain=.75)
        self.assertEqual(self.evidence()['chosen_gain'], .75)

    def test_passing_every_offset_takes_priority_over_higher_aggregate(self):
        rows = self.full()
        rows[1].update(dropped=True)
        self.batch(rows, gain=1.)  # Mean 0.833, but one offset is only 0.5.
        rows = []
        for row in self.full():
            rows.extend([dict(row, dropped=(repeat == 0)) for repeat in range(5)])
        self.batch(rows, gain=.75)  # Every offset passes at exactly 0.8.
        result = self.evidence()
        self.assertTrue(result['eligible'])
        self.assertEqual(result['chosen_gain'], .75)

    def test_per_cell_minimum_and_batch_minimum_apply_to_same_gain(self):
        self.settings['learning'].update(min_B_per_cell=2, min_B_batches=2)
        self.batch(self.full(), gain=.75)
        self.batch(self.full(), gain=1.)
        self.assertFalse(self.evidence()['eligible'])
        self.batch(self.full(), gain=.75)
        result = self.evidence()
        self.assertTrue(result['eligible'])
        self.assertEqual(result['chosen_gain'], .75)
        self.assertEqual(result['B_batches_on_current_model'], 3)

    def test_repeated_batch_reference_cannot_inflate_evidence(self):
        self.settings['learning']['min_B_batches'] = 2
        batch = self.batch(self.full())
        self.batches.append(copy.deepcopy(batch))
        result = self.evidence()
        self.assertFalse(result['eligible'])
        self.assertEqual(result['B_batches_on_current_model'], 1)

    def test_required_evaluation_offsets_are_not_inferred_from_latest_batch(self):
        self.batch(self.full())
        self.batch([self.row(0., bottle) for bottle in ('light', 'heavy')])
        self.assertEqual(self.evidence()['required_offsets_mm'], [-3.25, 0., 7.125])

    def test_complete_but_unsuccessful_gain_remains_explicitly_ineligible(self):
        self.batch(self.full(dropped=True), gain=.5)
        result = self.evidence()
        self.assertFalse(result['eligible'])
        self.assertEqual(result['chosen_gain'], .5)
        self.assertEqual(result['success_rate'], 0.)


if __name__ == '__main__':
    unittest.main()
