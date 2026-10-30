import json
import os
from pathlib import Path
import tempfile
import time
import unittest

import h5py
import numpy as np

from bottle_scan.recording import AsyncRecorder,RecordingFault,async_recording_defaults


class AsyncRecordingTests(unittest.TestCase):
    def setUp(self):self.directory=tempfile.TemporaryDirectory();self.path=Path(self.directory.name)/"record"
    def tearDown(self):self.directory.cleanup()

    def test_child_owns_files_and_payload_is_frozen_before_return(self):
        writer=AsyncRecorder(self.path,_test_options={"delay_s":.02})
        self.assertNotEqual(writer.stats()["writer_pid"],os.getpid())
        array=np.arange(12,dtype=np.float32).reshape(3,4);original=array.copy()
        self.assertEqual(writer.h5.append("dd",array,10,attrs={"unit":"sdk"}),0)
        array[:]=99
        self.assertEqual(writer.h5.append("force",np.array([1.,2.]),11),0)
        self.assertEqual(writer.h5.append("dd",original,12),1)
        image=np.zeros((16,17,3),np.uint8)
        self.assertEqual(writer.jpg.append("left_wrist",image,20),"left_wrist/00000000.jpg")
        self.assertEqual(writer.jpg.append("left_wrist",image,21),"left_wrist/00000001.jpg")
        metadata={"stage":"HOLD","values":[1,2]};writer.json("meta.json",metadata);metadata["values"][0]=999
        writer.close()
        with h5py.File(self.path/"streams.h5","r") as saved:
            np.testing.assert_array_equal(saved["streams/dd/data"][0],original)
            self.assertEqual(saved["streams/dd"].attrs["committed_count"],2)
            self.assertEqual(saved["streams/dd/data"].compression,"lzf")
            self.assertIn("lossless_lzf",saved.attrs["payload_policy"])
        self.assertEqual(json.loads((self.path/"meta.json").read_text())["values"],[1,2])
        self.assertTrue((self.path/"images/left_wrist/00000001.jpg").exists())
        self.assertTrue(json.loads((self.path/"recording_status.json").read_text())["complete"])

    def test_byte_queue_full_is_immediate_and_marks_incomplete(self):
        settings=async_recording_defaults();settings.update(queue_bytes=10000)
        writer=AsyncRecorder(self.path,settings=settings,_test_options={"delay_s":.25})
        data=np.ones(1500,np.float32)
        writer.h5.append("dd",data,1)
        started=time.monotonic()
        with self.assertRaisesRegex(RecordingFault,"byte queue full"):writer.h5.append("dd",data,2)
        self.assertLess(time.monotonic()-started,.1) # a 250 ms disk delay cannot block submit
        with self.assertRaises(RecordingFault):writer.close()
        saved=json.loads((self.path/"recording_status.json").read_text())
        self.assertFalse(saved["complete"]);self.assertEqual(saved["committed_messages"],1)

    def test_writer_failure_latches_without_waiting_in_control_loop(self):
        writer=AsyncRecorder(self.path,_test_options={"fail_after":0})
        writer.h5.append("dd",np.ones(3),1)
        deadline=time.monotonic()+2
        while writer.error is None and time.monotonic()<deadline:time.sleep(.005)
        self.assertIsInstance(writer.error,RecordingFault)
        with self.assertRaises(RecordingFault):writer.close()
        self.assertFalse(json.loads((self.path/"recording_status.json").read_text())["complete"])

    def test_worker_sigkill_is_detected_and_not_a_successful_recording(self):
        writer=AsyncRecorder(self.path)
        writer._process.kill();writer._process.join(timeout=2)
        with self.assertRaisesRegex(RecordingFault,"exited"):writer.check()
        with self.assertRaises(RecordingFault):writer.close()
        self.assertFalse(json.loads((self.path/"recording_status.json").read_text())["complete"])

    def test_metadata_cannot_escape_recording_directory(self):
        writer=AsyncRecorder(self.path)
        with self.assertRaises(ValueError):writer.json("../other.json",{})
        writer.close()

    def test_source_failure_marks_accepted_prefix_incomplete(self):
        writer=AsyncRecorder(self.path,_test_options={"delay_s":.05})
        writer.h5.append("dd",np.ones(3),1)
        started=time.monotonic()
        writer.fail("source queue overflow")
        self.assertLess(time.monotonic()-started,.02)
        with self.assertRaisesRegex(RecordingFault,"source queue overflow"):writer.check()
        with self.assertRaisesRegex(RecordingFault,"source queue overflow"):writer.close()
        saved=json.loads((self.path/"recording_status.json").read_text())
        self.assertFalse(saved["complete"])
        self.assertEqual(saved["committed_messages"],1)

    def test_context_source_failure_preserves_original_exception_and_incomplete_manifest(self):
        with self.assertRaisesRegex(ValueError,"camera source failed"):
            with AsyncRecorder(self.path) as writer:
                writer.h5.append("dd",np.ones(3),1)
                raise ValueError("camera source failed")
        saved=json.loads((self.path/"recording_status.json").read_text())
        self.assertFalse(saved["complete"])
        self.assertIn("camera source failed",saved["error"])

    def test_wall_clock_pressure_does_not_inherit_slow_disk_waits(self):
        from record_stress import run_stress
        from bottle_scan.config import demo_config
        from bottle_scan.recording import record_stress_defaults
        cfg=demo_config();cfg["recording"]["stress"]=record_stress_defaults()
        cfg["recording"]["stress"].update(duration_s=.3,write_delay_s=.04,image_shape=[480,640,3])
        report=run_stress(cfg,Path(self.directory.name)/"pressure")
        self.assertTrue(report["passed"],report["comparison"])
        self.assertTrue(report["simulated_data"])
        self.assertGreater(report["pressure"]["writer"]["submitted_messages"],10)
        self.assertLess(report["pressure"]["submission_s"]["p95_s"],.04)
        self.assertTrue(all(row["count"]>=20 for row in report["pressure"]["loops"].values()))


if __name__=="__main__":unittest.main()
