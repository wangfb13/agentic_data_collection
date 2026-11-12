import copy
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from skills.contact_position_a.legacy_v2.features import normalize_legacy, channel_statistics, read_dd_frames, adapt_vit_state_keys
from skills.contact_position_a.legacy_v2.noise import observation_vectors, fit_noise, noise_distance, modality_distance


def rows():
    rng=np.random.default_rng(2); result=[]
    for i in range(18):
        p={k:float(rng.normal()) for k in ('u_mm','v_mm','area_mm2','edge_dist_root_mm','edge_dist_tip_mm')}
        p.update({k:float(rng.normal()) for k in ('u_px','v_px','area_px','edge_dist_root_px','edge_dist_tip_px')})
        result.append(dict(node_id=str(i),session='s',object_id='white_light',planned_delta_mm=i//3,
            z_tac_left=rng.normal(size=768).tolist(),z_tac_right=rng.normal(size=768).tolist(),
            sensor_contact_left=p,sensor_contact_right={k:v+.1 for k,v in p.items()},
            wrench_left=rng.normal(size=6).tolist(),wrench_right=rng.normal(size=6).tolist(),
            gripper_opening=float(rng.normal(.4,.01)),z_vis=rng.normal(size=384).tolist(),legacy_features=rng.normal(size=20).tolist()))
    return result


class FeatureTests(unittest.TestCase):
    def test_transformer_key_mapping_is_bijective_and_never_drops_weights(self):
        old={'encoder_blocks.0.attention.q_proj.weight':np.ones((2,2)),
             'encoder_blocks.0.mlp.fc1.bias':np.ones(2),'unexpected':np.ones(1)}
        keys=['encoder_blocks.0.attention.attention.query.weight','encoder_blocks.0.intermediate.dense.bias']
        mapped=adapt_vit_state_keys(old,keys)
        self.assertEqual(set(mapped),set(keys)|{'unexpected'})
        self.assertIs(mapped[keys[0]],old['encoder_blocks.0.attention.q_proj.weight'])

    def test_signed_normalization_and_exact_legacy_clipping(self):
        a=np.array([[[1500,24000,30000],[5000,35000,40000]]],dtype=np.uint16)
        n=normalize_legacy(a)
        np.testing.assert_allclose(n,[[[.5,0,.5],[1,1,1]]])
        self.assertEqual(a[0,0,1],24000)

    def test_clipping_report_counts_full_pixels(self):
        a=np.array([[[1500,24000,30000],[5000,35000,40000]]],dtype=np.uint16)
        s=channel_statistics([a]);self.assertEqual(s['pixels'],2)
        np.testing.assert_allclose(s['clipped_fraction'],[.5,.5,.5])

    def test_frame_references_are_not_synthesized_or_repeated(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'streams.h5'
            with h5py.File(path,'w') as f:f.create_dataset('streams/dd_encoded_left/data',data=np.zeros((6,2,3,3),dtype=np.uint16))
            ref=dict(h5_path=str(path),dataset='streams/dd_encoded_left/data',indices=[0,1,2,3,4])
            self.assertEqual(len(read_dd_frames(ref)),5)
            ref['indices']=[0,0,1,2,3]
            with self.assertRaises(ValueError):read_dd_frames(ref)

    def test_label_fields_cannot_change_observation_distance(self):
        r=rows();m=fit_noise(r);a=copy.deepcopy(r[0]);b=copy.deepcopy(a)
        b.update(tcp_offset_a=1e8,tcp_offset_l=-1e8,planned_delta_mm=55,downstream_success=False,R=99,cluster_id='bad')
        for k,v in observation_vectors(a).items():np.testing.assert_equal(v,observation_vectors(b)[k])
        self.assertEqual(noise_distance(a,b,m),0.)

    def test_pixel_position_does_not_require_display_mm(self):
        r=rows()
        for row in r:
            for side in ('left','right'):
                for key in list(row['sensor_contact_'+side]):
                    if key.endswith(('mm','mm2')):row['sensor_contact_'+side][key]=None
        m=fit_noise(r)
        self.assertTrue(np.isfinite(noise_distance(r[0],r[1],m)))
        self.assertTrue(np.isfinite(noise_distance(r[0],r[1],m,['pos_px','force','open'])))
        self.assertEqual(m['modalities']['pos']['status'],'ready')

    def test_centroid_diagnostic_is_explicitly_distinct_from_full_position(self):
        r=rows()
        for row in r:
            for side in ('left','right'):
                c=row['sensor_contact_'+side]
                for k in ('edge_dist_root_px','edge_dist_tip_px'):
                    c[k]=None
                for k in list(c):
                    if k.endswith(('mm','mm2')):c[k]=None
        m=fit_noise(r)
        self.assertEqual(m['modalities']['pos_px_centroid']['dimension'],6)
        self.assertEqual(m['modalities']['pos_px_centroid']['status'],'ready')
        self.assertNotEqual(m['modalities']['pos_px']['status'],'ready')
        self.assertEqual(m['modalities']['pos']['status'],'ready')

    def test_held_out_rows_cannot_fit_pca_or_noise(self):
        r=rows();base=fit_noise(r)
        outside=copy.deepcopy(r[0]);outside.update(node_id='test',split='test')
        outside['z_tac_left']=[1e9]*768
        after=fit_noise(r+[outside]);self.assertEqual(base,after)

    def test_actual_extraction_evaluation_flags_override_explicit_fit_ids(self):
        r=rows();base=fit_noise(r)
        held=[]
        for i, fields in enumerate(({'evaluation_only':True},
                                   {'source_role':'formal_validation'},
                                   {'source_role':'validation','split':'train'})):
            row=copy.deepcopy(r[0]);row.update(node_id='held'+str(i),**fields)
            row['z_tac_left']=[1e9]*768;held.append(row)
        ids=[x['node_id'] for x in r+held]
        self.assertEqual(base,fit_noise(r+held))
        self.assertEqual(base,fit_noise(r+held,{'fit_node_ids':ids}))

    def test_explicit_fit_subset_controls_pca(self):
        r=rows();ids=[x['node_id'] for x in r[:12]];m=fit_noise(r,{'fit_node_ids':ids})
        self.assertEqual(m['fit_node_ids'],ids)
        self.assertEqual(m['modalities']['tac']['dimension'],11)
        self.assertEqual(m['modalities']['tac']['n'],12)

    def test_no_repeat_is_not_invented_noise(self):
        r=rows()
        for i,row in enumerate(r):row['planned_delta_mm']=i
        m=fit_noise(r)
        self.assertEqual(m['modalities']['open']['status'],'insufficient_repeat_groups')
        self.assertTrue(np.isinf(noise_distance(r[0],r[1],m,['open'])))

    def test_post_correction_windows_are_not_initial_repeats(self):
        r=rows()
        for i,row in enumerate(r):row.update(phase='A',window_index=0,trial_id=str(i))
        post=copy.deepcopy(r[0]);post.update(node_id='post',phase='B_post',window_index=1,gripper_opening=100.)
        m=fit_noise(r+[post],{'modalities':['open']})
        self.assertEqual(m['modalities']['open']['n'],19)
        self.assertEqual(m['modalities']['open']['n_residuals'],18)
        used=[n for g in m['modalities']['open']['repeat_sources'] for n in g['node_ids']]
        self.assertNotIn('post',used)

    def test_metric_matches_document_formula(self):
        r=rows();m=fit_noise(r);a,b=r[0],r[1]
        item=m['modalities']['force'];x=observation_vectors(a)['force']-observation_vectors(b)['force']
        want=np.sqrt(x@np.asarray(item['precision'])@x)/item['tau']
        self.assertAlmostEqual(want,modality_distance(a,b,m,'force'))
        values=[modality_distance(a,b,m,k)*(.5 if k=='vis' else 1) for k in ('pos','force','open')]
        self.assertAlmostEqual(max(values),noise_distance(a,b,m))


if __name__=='__main__':unittest.main()
