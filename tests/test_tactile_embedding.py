"""Frozen encoder wiring, training preprocessing, cache and optionality tests."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from bottle_scan.state_graph.modalities.tactile_embedding import FrozenEncoder,compose,sha256

SOURCE=Path('outputs/tactile_encoder_v3_20261112/models/model.py')
HAS_TORCH=importlib.util.find_spec('torch') is not None


class OptionalModalityTests(unittest.TestCase):
    def test_scheduler_extracts_matching_encoded_frames_for_optional_modality(self):
        import h5py
        from skills.contact_position_a.scheduler import trials_from_events
        from bottle_scan.state_graph.skill_interface import load_spec
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);path=root/'streams.h5';baseline=np.zeros((8,8,3),np.float32)
            decoded=np.repeat(baseline[None],5,axis=0);decoded[:,2:6,2:6,0]=.1
            encoded=(decoded*1000+np.array([0,30000,30000])).astype(np.uint16);times=np.arange(5,dtype=np.int64)+10
            with h5py.File(path,'w') as h:
                for side in ('left','right'):
                    for name,data in [('dd_'+side,decoded),('dd_encoded_'+side,encoded),('wrench_'+side,np.ones((5,6)))]:
                        g=h.create_group('streams/'+name);g.create_dataset('data',data=data);g.create_dataset('monotonic_ns',data=times);g.attrs['committed_count']=5
            np.savez(root/'reference.npz',left=baseline,right=baseline)
            event={'event':'state_graph_window','group_id':'g','session':'s','trial_id':'t','node_id':'n','window_index':0,'phase':'B_pre','t_ns':14,
                   'probe':{'stable_start_ns':10,'timestamp_ns':14,'window':[{'frame_ns':times.tolist(),'frame_fids':list(range(5))}]*2},
                   'raw_evidence':{'streams':str(path),'dd_reference':str(root/'reference.npz'),'wrist':{'available':False}},
                   'wrench':[1.]*12,'visual_features':[0.]*384,'attrs':{},'preconditions':{'bilateral_contact':True},'plan_coords':{}}
            trial=next(trials_from_events([event],load_spec('skills/contact_position_a/spec.json')))
            for side,ref in trial['windows'][0]['tactile_frames'].items():
                self.assertEqual(ref['dataset'],'streams/dd_encoded_'+side+'/data')
                self.assertEqual(ref['indices'],list(range(5)))
                self.assertEqual(ref['times_ns'],times.tolist())

    def test_base_spec_does_not_enable_embedding(self):
        from bottle_scan.state_graph.skill_interface import load_spec
        spec=load_spec('skills/contact_position_a/spec.json')
        self.assertNotIn('tac_embed',spec['observation']['modalities'])
        for name in ('mae','force'):
            candidate=load_spec('skills/contact_position_a/spec.tac_'+name+'.json')
            self.assertEqual(set(candidate['observation']['modalities']),set(spec['observation']['modalities'])|{'tac_embed'})
            self.assertEqual(candidate['dynamics']['inputs'],spec['dynamics']['inputs'])

    def test_composition_has_fixed_left_right_order(self):
        left=np.array([2.,3.]);right=np.array([5.,7.])
        np.testing.assert_equal(compose(left,right,'concat'),[2,3,5,7])
        np.testing.assert_equal(compose(left,right,'concat_difference'),[2,3,5,7,-3,-4])
        np.testing.assert_equal(compose(left,right,'single_left'),left)
        np.testing.assert_equal(compose(left,right,'single_right'),right)

    def test_generic_extractor_accepts_feature_evidence_and_real_version(self):
        from bottle_scan.state_graph.extract_nodes import extract_nodes
        from bottle_scan.state_graph.skill_interface import load_spec
        import bottle_scan.state_graph.extract_nodes as module
        spec=load_spec('tests/fixtures/skill_toy/spec.json');original=module.load_callable
        first=next(iter(spec['observation']['modalities']))
        name=spec['observation']['modalities'][first]['extractor']
        def load(path):
            fn=original(path)
            return (lambda window,spec:{'values':fn(window,spec),'version':'actual_model_version','evidence':{'input':'sensor_only'}}) if path==name else fn
        with patch.object(module,'load_callable',side_effect=load):rows,_=extract_nodes([],spec)
        self.assertEqual(rows[0]['extractor_versions'][first],'actual_model_version')
        self.assertEqual(rows[0]['feature_evidence'][first]['input'],'sensor_only')


@unittest.skipUnless(HAS_TORCH and SOURCE.exists(),'Torch and copied trained model source required')
class FrozenEncodingTests(unittest.TestCase):
    def setUp(self):
        import torch
        torch.set_num_threads(2)
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        definition=importlib.util.spec_from_file_location('test_tactile_source',SOURCE)
        module=importlib.util.module_from_spec(definition);definition.loader.exec_module(module)
        torch.manual_seed(4)
        model=module.TactileMAE(image_size=(32,48),patch_size=8,encoder_dim=16,encoder_layers=1,encoder_heads=2,decoder_dim=16,decoder_layers=1,decoder_heads=2)
        norm={'offset':[0,30000,30000],'mean':[1.,2.,3.],'std':[2.,4.,8.]}
        checkpoint=self.root/'model.pt';torch.save({'model_config':model.config,'model':model.state_dict(),'normalization':norm},checkpoint)
        artifact={'schema':'tactile_encoder_artifact_v1','checkpoint':str(checkpoint),'model_source':str(SOURCE.resolve()),'checkpoint_sha256':sha256(checkpoint),
                  'source_sha256':sha256(SOURCE),'normalization':norm,'model_config':model.config,'source_encoded_shape':[16,24,3]}
        path=self.root/'artifact.json';path.write_text(json.dumps(artifact))
        self.settings={'artifact':str(path),'cache_dir':str(self.root/'cache'),'representation':'concat','device':'cpu','frames_per_finger':5}
        self.encoder=FrozenEncoder(self.settings)
        rng=np.random.default_rng(7)
        self.frames=(np.array([20,30000,30000])+rng.integers(-10,11,(5,16,24,3))).astype(np.uint16)
        self.window={'tactile_arrays':{'left':self.frames,'right':self.frames[::-1].copy()+np.uint16(2)}}

    def test_training_preprocessing_exact_and_no_flip_clip_or_baseline(self):
        import torch
        import torch.nn.functional as F
        n=self.encoder.normalization
        expected=(self.frames.astype(np.float32)-np.array(n['offset'],np.float32)-np.array(n['mean'],np.float32))/np.array(n['std'],np.float32)
        expected=F.interpolate(torch.from_numpy(expected.transpose(0,3,1,2).copy()),size=(32,48),mode='bilinear',align_corners=False)
        torch.testing.assert_close(self.encoder.prepare(self.frames),expected,rtol=0,atol=0)
        self.assertLess(float(expected[:,1:].min()),0)
        with self.assertRaisesRegex(ValueError,'uint16'):self.encoder.prepare(self.frames.astype(np.float32))

    def test_frozen_full_view_pre_projection_and_no_rng_consumption(self):
        import torch
        self.encoder._load();model=self.encoder.model
        model.force_projector.forward=lambda x:(_ for _ in ()).throw(AssertionError('projector used'))
        model.force_head.forward=lambda x:(_ for _ in ()).throw(AssertionError('force head used'))
        state=torch.random.get_rng_state().clone();actual=self.encoder.encode_arrays(self.frames)
        torch.testing.assert_close(state,torch.random.get_rng_state(),rtol=0,atol=0)
        self.assertFalse(model.training);self.assertTrue(all(not p.requires_grad for p in model.parameters()))
        with torch.inference_mode():expected=model.encode(self.encoder.prepare(self.frames)).numpy()
        np.testing.assert_allclose(actual,expected,atol=1e-6)

    def test_pair_window_pooling_and_cache_only_without_model_load(self):
        result=self.encoder.window(self.window)
        z=self.encoder.encode_arrays(np.concatenate(list(self.window['tactile_arrays'].values())))
        np.testing.assert_allclose(result['values'],np.r_[np.median(z[:5],0),np.median(z[5:],0)],atol=1e-6)
        cached=FrozenEncoder({**self.settings,'cache_only':True})
        cached._load=lambda:(_ for _ in ()).throw(AssertionError('model loaded on cache hit'))
        self.assertEqual(cached.window(self.window),result)
        self.assertEqual(cached.cache_hits,1);self.assertIsNone(cached.model)
        changed=copy.deepcopy(self.window);changed['tactile_arrays']['left'][0,0,0,0]+=1
        with self.assertRaisesRegex(ValueError,'cache missing'):cached.window(changed)

    def test_representation_versions_and_metadata_are_independent(self):
        first=self.encoder.window(self.window)
        changed={**self.window,'attrs':{'mass':'changed'},'plan_coords':{'x':999},'post':{'success':False}}
        self.assertEqual(self.encoder.window(changed),first)
        other=FrozenEncoder({**self.settings,'representation':'concat_difference'})
        value=other.window(self.window)
        self.assertNotEqual(first['version'],value['version'])
        self.assertEqual(len(value['values']),48)
        np.testing.assert_allclose(value['values'][32:],np.array(first['values'][:16])-first['values'][16:],atol=1e-6)


if __name__=='__main__':unittest.main()
