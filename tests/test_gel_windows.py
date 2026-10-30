import copy
import json
from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np

from bottle_scan.recording import H5Writer,finalize_gel_windows,dd_processing_metadata


class GelWindowTests(unittest.TestCase):
    def setUp(self):
        self.directory=tempfile.TemporaryDirectory()
        self.root=Path(self.directory.name)
        with H5Writer(self.root/"streams.h5") as writer:
            for index in range(10):
                stamp=index*100
                for side in ("left","right"):
                    writer.append("gel_raw_"+side,np.full((4,5,3),index,np.uint8),stamp,
                                  device_timestamp=100.+index,attrs={"fid":index,"unit":"uint8"})
                writer.append("dd_left",np.full((2,3,3),index,np.float32),stamp)
        window={"frames":[{"monotonic_ns":200},{"monotonic_ns":400}],"start_ns":0,"end_ns":450,
                "probe_not_settled":False,"fallback_reason":None}
        hold={"frames":[{"monotonic_ns":700}],"start_ns":650,"end_ns":850}
        self.meta={"descriptor_windows":{"PROBE":{"left":window,"right":copy.deepcopy(window)},
                                         "HOLD":{"left":hold,"right":copy.deepcopy(hold)}}}
    def tearDown(self):self.directory.cleanup()

    def test_windows_preserve_payload_dtype_source_indices_and_nongel_stream(self):
        report=finalize_gel_windows(self.root,self.meta,{"record":{"gel_raw":"windows"}})
        self.assertEqual(report["streams"]["gel_raw_left"]["retained_frames"],5)
        with h5py.File(self.root/"streams.h5","r") as handle:
            group=handle["streams/gel_raw_left"]
            self.assertEqual(group["data"].dtype,np.uint8)
            np.testing.assert_array_equal(group["sequence"][:],[2,3,4,7,8])
            np.testing.assert_array_equal(group["monotonic_ns"][:],[200,300,400,700,800])
            np.testing.assert_array_equal(group["data"][:,0,0,0],[2,3,4,7,8])
            self.assertEqual(group.attrs["committed_count"],5)
            self.assertEqual(json.loads(group["metadata_json"][0])["fid"],2)
            self.assertEqual(handle["streams/dd_left/data"].shape[0],10)

    def test_missing_windows_do_not_fabricate_or_keep_unrelated_phase_frames(self):
        meta={}
        report=finalize_gel_windows(self.root,meta,{"record":{"gel_raw":"windows"}})
        self.assertEqual(report["streams"]["gel_raw_left"]["reason"],"no_selected_descriptor_window")
        with h5py.File(self.root/"streams.h5","r") as handle:
            self.assertEqual(handle["streams/gel_raw_left/data"].shape[0],0)
            self.assertEqual(handle["streams/gel_raw_left"].attrs["committed_count"],0)

    def test_full_keeps_all_frames_and_off_keeps_none(self):
        report=finalize_gel_windows(self.root,{}, {"record":{"gel_raw":"full"}})
        self.assertEqual(report["streams"]["gel_raw_right"]["retained_frames"],10)
        report=finalize_gel_windows(self.root,{}, {"record":{"gel_raw":"off"}})
        self.assertEqual(report["streams"]["gel_raw_right"]["retained_frames"],0)
        with h5py.File(self.root/"streams.h5","r") as handle:
            self.assertEqual(handle["streams/gel_raw_right/data"].shape[0],0)

    def test_provenance_preserves_unknown_version_and_quantization(self):
        result=dd_processing_metadata({"sensor_provenance":{"native_dd":False,"dd_encoded_shape":[96,128,3],
            "dd_encoded_dtype":"uint16","dd_decoded_dtype":"float32","dd_decoding":"scale1000_offset30000_float16"}})
        self.assertEqual(result["service_version"],"unknown_not_transmitted")
        self.assertEqual(result["resolution"],[96,128,3])
        self.assertFalse(result["native_dd"])
        self.assertEqual(result["quantization"],"scale1000_offset30000_float16")


if __name__=="__main__":unittest.main()
