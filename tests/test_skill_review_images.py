"""Exact recording-to-review image binding; synthetic pixels, no hardware."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import struct
import tempfile
import unittest
import zlib

import h5py
import numpy as np

from bottle_scan.recording import H5Writer
from bottle_scan.skill_review_images import export_contact_images

T = 1_000_000_000


def decode_png(path):
    data=Path(path).read_bytes();assert data[:8]==b'\x89PNG\r\n\x1a\n'
    at=8;payload=b''
    while at<len(data):
        size=struct.unpack('>I',data[at:at+4])[0];kind=data[at+4:at+8];body=data[at+8:at+8+size]
        if kind==b'IHDR':width,height,bits,color,_,_,_=struct.unpack('>IIBBBBB',body)
        if kind==b'IDAT':payload+=body
        at+=size+12
    channels={0:1,2:3,6:4}[color];raw=zlib.decompress(payload)
    rows=np.frombuffer(raw,dtype=np.uint8).reshape(height,1+width*channels)
    assert bits==8 and np.all(rows[:,0]==0)
    shape=(height,width) if channels==1 else (height,width,channels)
    return rows[:,1:].reshape(shape)


class ReviewImageTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.folder=self.root/'trial';self.folder.mkdir()
        self.out=self.root/'review'
        self.images={'left':np.arange(24,dtype=np.uint8).reshape(4,6),
                     'right':(np.arange(24,dtype=np.uint8).reshape(4,6)+123)}
        self.probe={'timestamp_ns':T,'keyframe':{'reason':'PROBE_last','requested_ns':T+1_000_000,
                    'gel':{'left':T+20_000_000,'right':T+23_000_000}}}
        self.row={'id':'00001','probe_initial':deepcopy(self.probe)}
        self.trial={'id':'00001','source_trial':str(self.folder),
                    'windows':[{'timestamp_ns':T,'observation':{'timestamp_ns':T}}]}
        self.write_meta();self.write_h5()

    def write_meta(self):
        path=self.folder/'meta.json';path.write_text(json.dumps(self.row))
        self.trial['source_metadata_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()

    def metadata(self,side,array,**changes):
        return {'keyframe_reason':'PROBE_last','e1e2_stage':'PROBE','stage':'PROBE',
                'trial_id':'00001','serial':'sensor_'+side,'fid':42,
                'raw':{'present':True,'dtype':str(array.dtype),'shape':list(array.shape),'bytes':array.nbytes},
                'representation':'original_39010_no_resize_no_lossy_encoding',**changes}

    def write_h5(self, *, reason=None, stage=None, skip=None, extra=False, metadata_changes=None):
        path=self.folder/'streams.h5'
        if path.exists():path.unlink()
        writer=H5Writer(path)
        for side,array in self.images.items():
            if side==skip:continue
            changes=dict(metadata_changes or {})
            if reason:changes['keyframe_reason']=reason
            if stage:changes.update(e1e2_stage=stage,stage=stage)
            meta=self.metadata(side,array,**changes)
            # A tempting nearby HOLD image must never be substituted.
            writer.append('gel_keyframe_'+side,np.full_like(array,7),T+19_000_000,
                          attrs=self.metadata(side,array,keyframe_reason='HOLD_mid',e1e2_stage='HOLD',stage='HOLD'))
            writer.append('gel_keyframe_'+side,array,self.probe['keyframe']['gel'][side],attrs=meta)
            if extra:writer.append('gel_keyframe_'+side,array,self.probe['keyframe']['gel'][side],attrs=meta)
        writer.close()

    def export(self):
        return export_contact_images(self.trial,self.out)

    def test_exact_keyframe_preserves_original_pixels_and_source(self):
        original=(self.folder/'streams.h5').read_bytes();metadata=(self.folder/'meta.json').read_bytes()
        manifest=deepcopy(self.trial);result=self.export()[0]
        self.assertEqual(result['status'],'available');self.assertEqual(result['pair_skew_ms'],3.)
        for side,array in self.images.items():
            evidence=result[side]
            np.testing.assert_array_equal(decode_png(self.out/evidence['filename']),array)
            self.assertEqual(evidence['frame_index'],1)
            self.assertEqual(evidence['source_frame_sha256'],hashlib.sha256(array.tobytes()).hexdigest())
            self.assertEqual(evidence['png_sha256'],hashlib.sha256((self.out/evidence['filename']).read_bytes()).hexdigest())
            self.assertEqual(evidence['pixel_transform'],'none');self.assertEqual(evidence['source_shape'],[4,6])
            self.assertEqual(evidence['metadata']['keyframe_reason'],'PROBE_last')
            self.assertNotIn('source_h5_sha256',evidence)
        self.assertEqual(self.trial,manifest)
        self.assertEqual((self.folder/'streams.h5').read_bytes(),original)
        self.assertEqual((self.folder/'meta.json').read_bytes(),metadata)

    def test_exact_observation_binding_includes_regrasp_and_confirmation(self):
        self.row.pop('probe_initial')
        self.row['probe_regrasps']=[{'probe':deepcopy(self.probe)}];self.write_meta()
        result=self.export()[0];self.assertEqual(result['probe_paths'],['probe_regrasps[0].probe'])
        np.testing.assert_array_equal(decode_png(self.out/result['left']['filename']),self.images['left'])
        self.out=self.root/'confirmation_review';self.row.pop('probe_regrasps')
        self.row['probe_confirmations']=[{'probe':deepcopy(self.probe)}];self.write_meta()
        self.assertEqual(self.export()[0]['probe_paths'],['probe_confirmations[0].probe'])

    def test_no_nearest_or_HOLD_substitution_without_exact_probe(self):
        self.row['probe_initial']['timestamp_ns']=T+1;self.write_meta()
        result=self.export()[0]
        self.assertEqual(result['left']['reason'],'no_exact_probe_for_observation')
        self.assertFalse(self.out.exists())

    def test_HOLD_frame_referenced_as_probe_is_rejected(self):
        self.write_h5(reason='HOLD_mid',stage='HOLD')
        result=self.export()[0]
        self.assertEqual(result['left']['reason'],'frame_is_not_PROBE_last')
        self.assertFalse(self.out.exists())

    def test_uncommitted_and_truncated_payloads_are_not_read(self):
        with h5py.File(self.folder/'streams.h5','r+') as h:
            h['streams/gel_keyframe_left'].attrs['committed_count']=1
            h['streams/gel_keyframe_right'].attrs['committed_count']=3
        result=self.export()[0]
        self.assertEqual(result['left']['reason'],'exact_gel_keyframe_not_committed')
        self.assertEqual(result['right']['reason'],'truncated_committed_stream')
        self.assertFalse(self.out.exists())

    def test_missing_commit_count_and_duplicate_timestamp_are_rejected(self):
        self.write_h5(extra=True)
        with h5py.File(self.folder/'streams.h5','r+') as h:
            del h['streams/gel_keyframe_right'].attrs['committed_count']
        result=self.export()[0]
        self.assertEqual(result['left']['reason'],'ambiguous_exact_gel_keyframe')
        self.assertEqual(result['right']['reason'],'missing_or_invalid_committed_count')

    def test_missing_one_finger_is_explicit_partial_not_synthesized(self):
        self.write_h5(skip='right');result=self.export()[0]
        self.assertEqual(result['status'],'partial');self.assertEqual(result['left']['status'],'available')
        self.assertEqual(result['right']['reason'],'gel_keyframe_stream_missing')
        self.assertIsNone(result['pair_skew_ms']);self.assertIsNone(result['right']['filename'])

    def test_request_time_missing_old_and_frame_delay_out_of_range_rejected(self):
        for requested,frame,reason in ((None,T+20_000_000,'missing_keyframe_request_timestamp'),
                  (T-1,T+20_000_000,'keyframe_request_not_current_for_observation'),
                  (T+1_000_000,T+600_000_000,'gel_keyframe_outside_requested_capture_window'),
                  (T+30_000_000,T+20_000_000,'gel_keyframe_outside_requested_capture_window')):
            self.row['probe_initial']=deepcopy(self.probe)
            self.row['probe_initial']['keyframe'].update(requested_ns=requested,gel={'left':frame,'right':frame})
            self.write_meta()
            with self.subTest(requested=requested,frame=frame):
                self.assertEqual(self.export()[0]['left']['reason'],reason)

    def test_corrupt_raw_spec_metadata_and_image_dtype_are_unavailable(self):
        self.write_h5(metadata_changes={'raw':{'present':True,'dtype':'uint8','shape':[2,2],'bytes':4}})
        self.assertEqual(self.export()[0]['left']['reason'],'frame_payload_metadata_mismatch')
        self.write_h5()
        with h5py.File(self.folder/'streams.h5','r+') as h:
            h['streams/gel_keyframe_left/metadata_json'][1]='not json'
        self.assertIn('invalid_source_frame',self.export()[0]['left']['reason'])

    def test_color_requires_explicit_encoding_and_RGB_preserves_values(self):
        self.images={side:np.stack([array,array+1,array+2],axis=-1) for side,array in self.images.items()}
        self.write_h5()
        result=self.export()[0];self.assertEqual(result['left']['reason'],'color_encoding_missing_or_unsupported')
        self.write_h5(metadata_changes={'encoding':'rgb8'})
        result=self.export()[0]
        np.testing.assert_array_equal(decode_png(self.out/result['left']['filename']),self.images['left'])
        self.assertEqual(result['left']['png_mode'],'RGB')

    def test_recorded_trial_and_explicit_serial_mismatches_rejected(self):
        self.write_h5(metadata_changes={'trial_id':'another'})
        self.assertEqual(self.export()[0]['left']['reason'],'frame_trial_identity_mismatch')
        self.write_h5();self.trial['sensor_serials']={'left':'wrong','right':'sensor_right'}
        result=self.export()[0]
        self.assertEqual(result['left']['reason'],'frame_sensor_serial_mismatch')
        self.assertTrue(result['right']['serial_mapping_verified'])

    def test_serial_crosscheck_uses_exact_probe_DD_sources(self):
        self.row['probe_initial']['window']=[{'frame_ns':[T-10]},{'frame_ns':[T-20]}];self.write_meta()
        with h5py.File(self.folder/'streams.h5','r+') as h:
            for side,stamp in (('left',T-10),('right',T-20)):
                g=h.create_group('streams/dd_'+side);g.attrs['committed_count']=1
                g.create_dataset('monotonic_ns',data=[stamp])
                g.create_dataset('metadata_json',data=[json.dumps({'serial':'sensor_'+side})],dtype=h5py.string_dtype())
        result=self.export()[0]
        self.assertTrue(result['left']['serial_mapping_verified'])
        self.assertEqual(result['left']['serial_mapping_provenance'],['exact_probe_dd_frames'])

    def test_source_change_and_missing_recording_do_not_create_images(self):
        self.trial['source_metadata_sha256']='changed'
        self.assertEqual(self.export()[0]['left']['reason'],'source_metadata_changed')
        self.write_meta();(self.folder/'streams.h5').unlink()
        self.assertEqual(self.export()[0]['left']['reason'],'source_unavailable:FileNotFoundError')
        self.assertFalse(self.out.exists())

    def test_ambiguous_probe_binding_and_same_sensor_pair_are_rejected(self):
        other=deepcopy(self.probe);other['keyframe']['requested_ns']+=1
        self.row['probe_confirmations']=[{'probe':other}];self.write_meta()
        self.assertEqual(self.export()[0]['left']['reason'],'ambiguous_probe_for_observation')
        self.row.pop('probe_confirmations');self.write_meta()
        self.write_h5(metadata_changes={'serial':'same'})
        result=self.export()[0]
        self.assertEqual(result['left']['reason'],'same_serial_on_both_fingers')
        self.assertFalse(self.out.exists())

    def test_original_source_directory_cannot_be_output(self):
        with self.assertRaisesRegex(ValueError,'outside the original trial'):
            export_contact_images(self.trial,self.folder/'images')


if __name__=='__main__':unittest.main()
