"""Synthetic metadata review tests; no real annotations or robot commands."""
from copy import deepcopy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import h5py
import numpy as np
import yaml

from bottle_scan import skill_label_review as review
from bottle_scan.agentic_learning import _load, _observations
from bottle_scan.contact_policy import train_contact_policy, _split
from bottle_scan.learned_skill import _collect, _group, diagnose_training_data, train_learned_skill
from tests.test_learned_skill import LearnedSkillFixture, learned_settings, TARGET_ID


class LabelReviewTests(LearnedSkillFixture):
    def source(self):
        root=self.make_learned('A')
        meta=json.loads((root/'session_meta.json').read_text())
        self.target={'id':TARGET_ID,'description':'Synthetic independently defined region',
                     'normal':'inside target','needs_recovery':'outside target','completion':'inside stable target'}
        config=self.root/'target.yaml';config.write_text(yaml.safe_dump({'target_definition':self.target}))
        # Test raw TCP data remain byte-identical and are reused read-only.
        first=next((root/'trials').glob('*/meta.json')).parent
        with h5py.File(first/'streams.h5','w') as handle:
            g=handle.create_group('streams/tcp_measured')
            g.create_dataset('monotonic_ns',data=[1000]);g.create_dataset('data',data=[np.eye(4)])
        return root,config

    def prepared(self):
        root,config=self.source();directory=self.root/'review'
        result=review.prepare_review(root,config,directory,reference_trial=next((root/'trials').iterdir()).name)
        return root,directory,result

    def annotate(self,directory,*,unknown=False):
        answers=review._read(directory/'annotations.json')
        manifest=review._read(directory/'manifest.json')
        answers['reviewer']='synthetic-test-human'
        answers['attestation']={k:True for k in answers['attestation']}
        # Explicit simulated annotations; production never derives these from an offset.
        for item,trial in zip(answers['trials'],manifest['trials']):
            item['reviewed']=True
            if unknown:continue
            original=review._read(Path(trial['source_trial'])/'meta.json')
            item['initial_class']=original['skill_ground_truth']['initial_class']
            by_time={x['timestamp_ns']:x['target_reached'] for x in original['skill_window_labels']}
            for window in item['windows']:window['target_reached']=by_time[window['timestamp_ns']]
            # Deliberately leave final_target_reached unknown: heads still train.
        path=directory/'human.json';path.write_text(json.dumps(answers));return path

    def test_prepare_unknown_only_no_offset_truth_or_model_claims(self):
        root,directory,result=self.prepared()
        answers=review._read(directory/'annotations.json')
        self.assertTrue(result['all_labels_unknown']);self.assertEqual(result['physical_trials_added'],0)
        self.assertTrue(all(x['initial_class'] is None and x['final_target_reached'] is None and not x['reviewed'] for x in answers['trials']))
        self.assertTrue(all(w['target_reached'] is None for x in answers['trials'] for w in x['windows']))
        page=(directory/'index.html').read_text()
        self.assertNotIn('delta_mm',page);self.assertNotIn('hold_complete',page);self.assertIn('不是原始 DD',page)
        self.assertIn('NOT raw DD',next(directory.glob('*.svg')).read_text())
        self.assertIn('class=initial',page);self.assertIn('class=window-label',page)
        self.assertIn('class=reviewed',page);self.assertIn('annotations.json',page)
        self.assertNotIn('fetch(',page);self.assertNotIn('XMLHttpRequest',page)
        self.assertNotIn('checked>',page)
        self.assertFalse(any('rospy' in x for x in __import__('sys').modules))

    def test_browser_template_keeps_nanosecond_ids_exact_and_unknown_by_default(self):
        stamp=9007199254740993
        source={'reviewer':'','attestation':{},'trials':[{'id':'001','reviewed':False,
            'initial_class':None,'final_contact_timestamp_ns':stamp,'final_target_reached':None,
            'windows':[{'timestamp_ns':stamp,'target_reached':None}]}]}
        script=review._review_form_script(source)[-1]
        payload=json.loads(script.split('const template = ',1)[1].split(';\nfunction answer',1)[0])
        self.assertEqual(payload['trials'][0]['windows'][0]['timestamp_ns'],str(stamp))
        self.assertEqual(payload['trials'][0]['final_contact_timestamp_ns'],str(stamp))
        self.assertEqual(source['trials'][0]['windows'][0]['timestamp_ns'],stamp)
        self.assertIsNone(payload['trials'][0]['initial_class'])
        self.assertIsNone(payload['trials'][0]['windows'][0]['target_reached'])
        self.assertFalse(payload['trials'][0]['reviewed'])
        self.assertIn('createObjectURL',script)
        self.assertIn('timestamp_ns|final_contact_timestamp_ns',script)

    def test_original_images_are_primary_and_do_not_prefill_contact_labels(self):
        root,config=self.source();directory=self.root/'raw_review'
        def exported(trial,out):
            return [{'timestamp_ns':w['timestamp_ns'],'pair_skew_ms':2.,
                     **{side:{'status':'available','filename':'images/'+trial['id']+'_'+side+'.png',
                              'frame_timestamp_ns':w['timestamp_ns']+30_000_000,
                              'observation_delta_ms':30.} for side in ('left','right')}}
                    for w in trial['windows']]
        with patch('bottle_scan.skill_review_images.export_contact_images',side_effect=exported):
            result=review.prepare_review(root,config,directory)
        page=(directory/'index.html').read_text()
        self.assertGreater(result['original_images_available'],0)
        self.assertTrue((directory/'image_evidence.json').is_file())
        self.assertLess(page.index('<div class=tactile-pair>'),page.index('<details class=descriptor-geometry>'))
        self.assertIn('原始触觉图',page);self.assertIn('+30.00 ms',page)
        answers=review._read(directory/'annotations.json')
        self.assertTrue(all(t['initial_class'] is None and t['final_target_reached'] is None
                            and not t['reviewed'] for t in answers['trials']))

    def test_missing_original_is_explicit_and_never_replaced_by_diagram(self):
        page=review._contact_images_html({'left':{'status':'unavailable','reason':'missing exact frame'},
                                         'right':{'status':'unavailable','reason':'<bad>'}})
        self.assertNotIn('<img',page)
        self.assertIn('没有可核验的对应原图',page)
        self.assertIn('&lt;bad&gt;',page)

    def test_apply_preserves_original_and_reuses_raw_with_unknown_final(self):
        root,directory,_=self.prepared();before={str(p):p.read_bytes() for p in root.rglob('*') if p.is_file()}
        annotations=self.annotate(directory);out=self.root/'view'
        with patch('bottle_scan.agentic_executor.execute_batch',side_effect=AssertionError('hardware')):
            result=review.apply_review(directory,annotations,out)
        self.assertEqual(result['physical_trials_added'],0)
        self.assertEqual(before,{str(p):p.read_bytes() for p in root.rglob('*') if p.is_file()})
        raw=next(out.rglob('streams.h5'));self.assertTrue(raw.is_symlink())
        self.assertEqual(raw.resolve(),next(root.rglob('streams.h5')))
        sessions,rows,issues=_load([out],learned_settings());self.assertEqual(issues,[])
        self.assertTrue(all(r['canonical_session']==str(root) for r in rows))
        self.assertTrue(all(r['skill_ground_truth']['final_target_reached'] is None for r in rows))
        learned=train_learned_skill([out],learned_settings())
        self.assertTrue(learned['usable_for_B'],learned['blocking_reasons'])
        self.assertGreater(learned['invocation_head']['class_counts']['positive'],0)
        with self.assertRaises(FileExistsError):review.apply_review(directory,annotations,out)

    def test_original_plus_view_and_two_views_block_duplicate_physical_trials(self):
        root,directory,_=self.prepared();answers=self.annotate(directory)
        first=self.root/'view1';second=self.root/'view2'
        review.apply_review(directory,answers,first);review.apply_review(directory,answers,second)
        for roots in ([root,first],[first,second]):
            with self.subTest(roots=roots):
                _,rows,issues=_load(roots,learned_settings())
                self.assertTrue(any('duplicate physical trial' in x for x in issues))
                self.assertEqual(len(rows),60)
                model=train_learned_skill(roots,learned_settings())
                self.assertFalse(model['usable_for_B'])
                self.assertFalse(model['recovery_model']['usable_for_B'])

    def test_view_does_not_change_physical_split_groups(self):
        root,directory,_=self.prepared();answers=self.annotate(directory);out=self.root/'view'
        review.apply_review(directory,answers,out)
        _,original_rows,issues=_load([root],learned_settings());self.assertFalse(issues)
        _,view_rows,issues=_load([out],learned_settings());self.assertFalse(issues)
        self.assertEqual([_group(r) for r in original_rows],[_group(r) for r in view_rows])
        original,_=_observations(original_rows,original_rows[0]['session_meta'],learned_settings())
        reviewed,_=_observations(view_rows,view_rows[0]['session_meta'],learned_settings())
        self.assertEqual([o['group'] for o in original],[o['group'] for o in reviewed])
        # Changing only view names cannot change held-out-session membership.
        data=[{'session':'viewA','canonical_session':'physicalA','group':'a'},
              {'session':'B','group':'b'},{'session':'C','group':'c'}]*12
        from bottle_scan.contact_policy import _settings
        cfg=_settings(learned_settings())[1]
        a=_split(data,cfg)
        changed=[{**o,'session':'renamedView'} if o['session']=='viewA' else o for o in data]
        b=_split(changed,cfg)
        self.assertTrue(np.array_equal(a[0],b[0]));self.assertTrue(np.array_equal(a[1],b[1]))

    def test_unknowns_are_not_converted_to_negative_or_final_success(self):
        root,directory,_=self.prepared();answers=self.annotate(directory,unknown=True);out=self.root/'view'
        review.apply_review(directory,answers,out)
        model=train_learned_skill([out],learned_settings())
        self.assertFalse(model['usable_for_B'])
        self.assertEqual(model['invocation_head']['class_counts']['physical_trials'],0)
        self.assertEqual(model['exit_head']['class_counts']['physical_trials'],0)

    def test_attestation_exact_timestamps_and_conflicting_labels_required(self):
        root,directory,_=self.prepared();answers=self.annotate(directory);good=review._read(answers)
        mutations=[lambda a:a['attestation'].update(independent_of_experimental_offset=False),
            lambda a:a['trials'][0]['windows'][0].update(timestamp_ns=999999),
            lambda a:a['trials'][0]['windows'].append(deepcopy(a['trials'][0]['windows'][0])),
            lambda a:a['trials'][0].update(initial_class='already_normal',windows=[{**w,'target_reached':False} for w in a['trials'][0]['windows']]),
            lambda a:a['trials'][0].update(final_target_reached=False)]
        for change in mutations:
            with self.subTest(change=change):
                bad=deepcopy(good);change(bad);answers.write_text(json.dumps(bad))
                with self.assertRaises(ValueError):review.apply_review(directory,answers,self.root/'badview')
                self.assertFalse((self.root/'badview').exists())

    def test_view_cannot_edit_observations_or_source_roles(self):
        root,directory,_=self.prepared();answers=self.annotate(directory);out=self.root/'view'
        review.apply_review(directory,answers,out)
        p=next((out/'trials').glob('*/meta.json'));row=review._read(p)
        row['probe_initial']['descriptors'][0]['centroid_px'][0]+=3
        p.write_text(json.dumps(row))
        _,_,issues=_load([out],learned_settings())
        self.assertTrue(any('differs from original evidence' in x for x in issues))
        meta=review._read(out/'session_meta.json');meta['agentic']['role']='final_validation'
        (out/'session_meta.json').write_text(json.dumps(meta))
        _,_,issues=_load([out],learned_settings())
        self.assertTrue(any('cannot change source roles' in x for x in issues))

    def test_resealed_window_manifest_cannot_invent_observed_contact(self):
        root,directory,_=self.prepared();answers=self.annotate(directory);out=self.root/'view'
        review.apply_review(directory,answers,out)
        manifest=review._read(out/'review_manifest.json')
        manifest['trials'][0]['windows'][0]['observation']['stable_duration_s']=100.
        annotations=review._read(out/'review_annotations.json');annotations['manifest_sha256']=review._hash(manifest)
        meta=review._read(out/'session_meta.json')
        meta['skill_label_review'].update(manifest_sha256=review._hash(manifest),annotations_sha256=review._hash(annotations))
        for name,value in [('review_manifest.json',manifest),('review_annotations.json',annotations),('session_meta.json',meta)]:
            (out/name).write_text(json.dumps(value))
        _,_,issues=_load([out],learned_settings())
        self.assertTrue(any('window manifest differs from original' in x for x in issues))

    def test_conflicting_recorded_observations_at_one_timestamp_are_rejected(self):
        root,config=self.source();path=next((root/'trials').glob('*/meta.json'));row=review._read(path)
        extra=deepcopy(row['skill_observations'][0]);extra['observation']['bilateral']=False
        row['skill_observations'].append(extra);path.write_text(json.dumps(row))
        with self.assertRaisesRegex(ValueError,'conflicting recorded observations'):
            review.prepare_review(root,config,self.root/'conflict_review')

    def test_source_or_view_mutations_are_blocked_and_final_roles_cannot_review(self):
        root,directory,_=self.prepared();answers=self.annotate(directory);out=self.root/'view'
        review.apply_review(directory,answers,out)
        with self.assertRaisesRegex(ValueError,'review-of-review'):
            review.prepare_review(out,self.root/'target.yaml',self.root/'nested')
        p=next((root/'trials').glob('*/meta.json'));original=review._read(p);changed=deepcopy(original);changed['probe_initial']['descriptors'][0]['centroid_px'][0]+=1
        p.write_text(json.dumps(changed))
        with self.assertRaisesRegex(ValueError,'source evidence changed'):review.apply_review(directory,answers,self.root/'other')
        _,_,issues=_load([out],learned_settings());self.assertTrue(any('metadata changed' in x for x in issues))
        p.write_text(json.dumps(original))
        # Original byte hash intentionally differs after reserialization, still blocked.
        meta=review._read(root/'session_meta.json');meta['agentic']['role']='final_validation'
        (root/'session_meta.json').write_text(json.dumps(meta))
        with self.assertRaisesRegex(ValueError,'original exploration/development'):
            review.prepare_review(root,self.root/'target.yaml',self.root/'final_review')
