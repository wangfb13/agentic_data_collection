import json
import threading
import time
import unittest
import numpy as np

from bottle_scan.hardware_sensors import (
    SensorFault, SharedSensorHub, WrenchCalibration, decode_native_tactile,
    encode_native_tactile, inspect_wrench_pub, inspect_dd_pub, ActualX5SensorSource, decode_existing_dd,
)


def packet(serial="L", fid=7, sequence=0, reference="fixed", dtype="float32"):
    return encode_native_tactile(serial, np.arange(12,dtype=dtype).reshape(3,4),
        np.ones((3,4,2),dtype=dtype), np.arange(6,dtype=np.float64),
        {"source_sequence":sequence,"fid":fid,"depth_fid":fid,"deformation_fid":fid,
         "force_fid":fid,"force_fid_verified":True,"session_id":"source1",
         "reference_sha256":reference,"timestamp_kind":"sdk_read_wall_time",
         "depth_unit":"sdk_raw","flow_unit":"sdk_raw","wrench_unit":"sdk_raw",
         "timestamp":123456.75})


class HubTests(unittest.TestCase):
    def setUp(self):
        self.now = 1_000_000_000

    def hub(self, **kwargs):
        cal = WrenchCalibration(np.eye(6)*2, np.ones(6), 2, 1, "Nmm", "fake fixture calibration")
        return SharedSensorHub(serials={"left":"L","right":"R"},dd_shape=(3,4,3),dd_dtype="float32",
            calibrations={"left":cal,"right":cal},clock=lambda:self.now,**kwargs)

    def test_codec_preserves_native_dtype_shape_and_bytes(self):
        m,a=decode_native_tactile(packet())
        self.assertEqual(a["depth"].shape,(3,4))
        self.assertEqual(a["flow"].shape,(3,4,2))
        self.assertEqual(a["wrench"].dtype,np.dtype("float64"))
        np.testing.assert_array_equal(a["depth"],np.arange(12,dtype="float32").reshape(3,4))
        self.assertEqual(m["depth_unit"],"sdk_raw")

    def test_encoded_dd_old_wire_and_size_corruption_rejected(self):
        with self.assertRaises(SensorFault):decode_native_tactile([b"L",b'{"schema":"dm_x5_dd_v1"}',bytes(32)])
        p=packet();p[-1]=p[-1][:-1]
        with self.assertRaises(SensorFault):decode_native_tactile(p)

    def test_untagged_force_remains_explicit_and_misalignment_rejected(self):
        p=packet();m=json.loads(p[1]);m["force_fid_verified"]=False;m["force_fid"]=None
        p[1]=json.dumps(m).encode()
        self.assertFalse(decode_native_tactile(p)[0]["force_fid_verified"])
        m["depth_fid"]+=1;p[1]=json.dumps(m).encode()
        with self.assertRaises(SensorFault):decode_native_tactile(p)

    def test_hub_raw_and_calibrated_streams_are_separate(self):
        hub=self.hub();hub.ingest_tactile(packet());hub.ingest_tactile(packet("R"))
        snap=hub.snapshot();samples={s.stream:s for s in hub.drain_samples()}
        self.assertEqual(samples["sdk_depth_left"].data.shape,(3,4))
        self.assertEqual(samples["sdk_flow_left"].data.shape,(3,4,2))
        self.assertEqual(samples["wrench_raw_left"].data.dtype,np.dtype("float64"))
        np.testing.assert_array_equal(snap["normal_N"],[2.,2.])
        np.testing.assert_array_equal(samples["wrench_raw_left"].data,np.arange(6))
        np.testing.assert_array_equal(samples["wrench_left"].data,(np.arange(6)-1)*2)
        self.assertEqual(samples["dd_left"].metadata["representation"],"lossless_stack_depth_flow")
        snap["dd"][0][:]=99
        self.assertNotEqual(hub.snapshot()["dd"][0][0,0,0],99)
        self.assertEqual(hub.drain_samples(),[])

    def test_source_reset_sdk_gap_and_duplicate_latch_fault(self):
        for second in (packet(fid=8,sequence=1,reference="changed"),packet(fid=9,sequence=1),packet()):
            with self.subTest(second=second[1]):
                hub=self.hub();hub.ingest_tactile(packet())
                with self.assertRaises(SensorFault):hub.ingest_tactile(second)
                self.assertTrue(hub.fault)
                with self.assertRaises(SensorFault):hub.snapshot()

    def test_stale_individual_source_cannot_be_masked_by_new_left(self):
        hub=self.hub();hub.ingest_tactile(packet());hub.ingest_tactile(packet("R"))
        self.now+=300_000_000;hub.ingest_tactile(packet(fid=8,sequence=1))
        with self.assertRaisesRegex(SensorFault,"Stale source dd_right"):hub.snapshot()

    def test_bounded_pair_skew(self):
        hub=self.hub();hub.ingest_tactile(packet());hub.ingest_tactile(packet("R"))
        self.now+=150_000_000;hub.ingest_tactile(packet(fid=8,sequence=1))
        with self.assertRaisesRegex(SensorFault,"skew"):hub.snapshot()

    def test_snapshot_feedback_uses_same_native_wrench_fids_after_only_left_advances(self):
        hub=self.hub();hub.ingest_tactile(packet());hub.ingest_tactile(packet("R"))
        hub.drain_samples()
        m,arrays=decode_native_tactile(packet(fid=8,sequence=1))
        arrays["wrench"][2]=8.
        self.now+=1_000_000
        hub.ingest_tactile(encode_native_tactile("L",**arrays,metadata=m))
        snap=hub.snapshot()
        np.testing.assert_array_equal(snap["normal_N"],[14.,2.])
        self.assertEqual(snap["source_sequences"]["wrench_left"],8)
        self.assertEqual(snap["source_sequences"]["wrench_right"],7)
        self.assertEqual(snap["source_timestamp_ns"]["wrench_left"],self.now)
        self.assertFalse(any(x.stream=="normal_force_N" for x in hub.drain_samples()))

    def test_overflow_and_shape_mismatch_latch(self):
        hub=self.hub(max_queue=2)
        with self.assertRaisesRegex(SensorFault,"overflow"):hub.ingest_tactile(packet())
        hub=self.hub()
        with self.assertRaisesRegex(SensorFault,"dtype"):hub.ingest_tactile(packet(dtype="float64"))

    def test_fresh_images_require_both_new_views(self):
        hub=self.hub();img=np.zeros((8,9,3),np.uint8)
        for name in ("left_wrist","right_wrist"):
            hub.ingest_camera(name,img,source_sequence=0,source_session="camera1")
        result=[]
        t=threading.Thread(target=lambda:result.append(hub.images(timeout_s=.2)))
        t.start();time.sleep(.01);self.now+=1
        hub.ingest_camera("left_wrist",img+1,source_sequence=1,source_session="camera1")
        time.sleep(.01);self.assertTrue(t.is_alive())
        hub.ingest_camera("right_wrist",img+2,source_sequence=1,source_session="camera1")
        t.join(.2);self.assertFalse(t.is_alive())
        self.assertEqual(int(result[0]["right_wrist"][0,0,0]),2)
        with self.assertRaisesRegex(SensorFault,"both fresh"):hub.images(timeout_s=.001)

    def test_calibration_not_silently_defaulted(self):
        with self.assertRaises(ValueError):WrenchCalibration(np.eye(6),np.zeros(6),2,1,"Nmm","TODO")


class DiagnosticTests(unittest.TestCase):
    def test_existing_pub_diagnostic_only_subscribes_and_cleans_up(self):
        import zmq
        now=[1_000_000_000]
        class FakeSocket:
            def __init__(self):
                self.options=[];self.closed=False;self.messages=[]
                for fid in (8,9):
                    for serial in ("L","R"):
                        self.messages.append([serial.encode(),json.dumps({"schema":"dm_x5_wrench_v1",
                            "serial":serial,"fid":fid,"wrench":[0,1,2,3,4,5],"timestamp":42.,
                            "session_id":"existing","force_fid_verified":True}).encode()])
            def setsockopt(self,*args):self.options.append(args)
            def connect(self,address):self.address=address
            def poll(self,*args):now[0]+=20_000_000;return bool(self.messages)
            def recv_multipart(self):return self.messages.pop(0)
            def close(self,*args):self.closed=True
        class FakeContext:
            def __init__(self):self.sock=FakeSocket();self.terminated=False
            def socket(self,kind):self.kind=kind;return self.sock
            def term(self):self.terminated=True
        ctx=FakeContext()
        report=inspect_wrench_pub(serials=("L","R"),duration_s=.12,_context_factory=lambda:ctx,_clock=lambda:now[0])
        self.assertEqual(ctx.kind,zmq.SUB)
        self.assertTrue(ctx.terminated and ctx.sock.closed)
        self.assertTrue(report["ready"])
        self.assertFalse(report["physical_force_unit_verified"])
        self.assertFalse(report["native_dd_available"])
        self.assertEqual(report["sensors"]["L"]["count"],2)
        with self.assertRaises(ValueError):inspect_wrench_pub(duration_s=6)


def existing_packet(serial="L",fid=7):
    data=np.full((96,128,3),30000,dtype="<u2");data[...,0]=1234
    meta={"schema":"dm_x5_dd_v1","serial":serial,"fid":fid,"depth_fid":fid,"deformation_fid":fid,
          "session_id":"existing","reference_sha256":"fixed","timestamp":12.,
          "frame":{"dtype":"uint16","byteorder":"little","shape":[96,128,3],
                   "bytes":data.nbytes,"channels":["D","Dx","Dy"]},
          "encoding":{"scale":[1000.]*3,"offset":[0.,30000.,30000.],"quantization":"truncate_uint16_then_inter_area"}}
    return [serial.encode(),json.dumps(meta).encode(),data.tobytes()]


def wrench_packet(serial="L",fid=7,normal=2.):
    return [serial.encode(),json.dumps({"schema":"dm_x5_wrench_v1","serial":serial,"fid":fid,
            "wrench":[0.,0.,normal,0.,0.,0.],"timestamp":42.}).encode()]


class ExistingX5Tests(unittest.TestCase):
    def setUp(self):
        self.cfg={"hardware":{"public_ros":{"sensors":{
            "serials":{"left":"L","right":"R"},"dd_address":"tcp://right:39030",
            "wrench_address":"tcp://right:39020","wrist_endpoints":{
            "left_wrist":"tcp://left:5552","right_wrist":"tcp://right:5552"}}}}}
        self.now=1000000000
        cal=WrenchCalibration(np.eye(6)*2,np.zeros(6),2,1,"Nmm","fake measured calibration")
        self.cal={"left":cal,"right":cal}

    def test_sdk_mode_needs_no_N_scale_and_records_all_force_definitions(self):
        from bottle_scan.hardware_sensors import SdkWrenchProjection
        self.cfg["force"]={"unit":"sdk"}
        self.cfg["hardware"]["public_ros"]["sensors"]["force_conversion"]={
            "normal_axis":2,"normal_signs":{"left":1,"right":1}}
        cal={s:SdkWrenchProjection(np.zeros(6),2,1) for s in ("left","right")}
        source=ActualX5SensorSource(self.cfg,calibrations=cal,clock=lambda:self.now)
        for serial in ("L","R"):
            source._ingest("dd",existing_packet(serial),self.now)
            source._ingest("wrench",wrench_packet(serial),self.now)
        samples={item.stream:item for item in source.drain_samples()}
        self.assertIn("normal_force_sdk",samples)
        self.assertIn("normal_mean_sdk",samples)
        self.assertIn("normal_sum_sdk",samples)
        self.assertNotIn("normal_force_N",samples)
        self.assertEqual(samples["normal_mean_sdk"].data,2.)
        self.assertEqual(samples["normal_sum_sdk"].data,4.)
        self.assertEqual(samples["wrench_left"].metadata["unit"],["sdk"]*6)
        self.assertEqual(source.snapshot()["force_unit"],"sdk")

    def test_sdk_compression_negative_raw_Fz_produces_positive_normal(self):
        from bottle_scan.hardware_sensors import SdkWrenchProjection
        self.cfg['force']={'unit':'sdk'}
        self.cfg['hardware']['public_ros']['sensors']['force_conversion']={
            'normal_axis':2,'normal_signs':{'left':-1,'right':-1}}
        cal={side:SdkWrenchProjection(np.zeros(6),2,-1) for side in ('left','right')}
        source=ActualX5SensorSource(self.cfg,calibrations=cal,clock=lambda:self.now)
        for serial in ('L','R'):
            source._ingest('dd',existing_packet(serial),self.now)
            source._ingest('wrench',wrench_packet(serial,normal=-2.),self.now)
        samples={x.stream:x for x in source.drain_samples()}
        np.testing.assert_allclose(samples['normal_force_sdk'].data,[2.,2.])
        self.assertEqual(samples['normal_mean_sdk'].data,2.)
        self.assertEqual(samples['normal_sum_sdk'].data,4.)
        self.assertEqual(samples['wrench_raw_left'].data[2],-2.)
        self.assertEqual(samples['wrench_left'].data[2],-2.)
        np.testing.assert_allclose(source.snapshot()['normal_N'],[2.,2.])

    def test_N_mode_without_measured_scale_is_rejected_before_connect(self):
        self.cfg["force"]={"unit":"N"}
        self.cfg["hardware"]["public_ros"]["sensors"]["force_conversion"]={
            "normal_axis":2,"normal_signs":{"left":1,"right":1}}
        with self.assertRaisesRegex(ValueError,"force conversion"):
            ActualX5SensorSource(self.cfg)

    def test_verified_N_force_preserves_unknown_moment_as_sdk(self):
        from bottle_scan.hardware_sensors import ForceOnlyWrenchCalibration
        projection=ForceOnlyWrenchCalibration(np.zeros(6),2,1,"sdk","measured N coefficient",2.)
        np.testing.assert_array_equal(projection.apply(np.ones(6)),[2,2,2,1,1,1])

    def test_gel_raw_windows_preserve_original_bytes_and_drop_only_outside_windows(self):
        self.cfg["record"]={"gel_raw":"windows"}
        self.cfg["hardware"]["public_ros"]["sensors"]["gel_raw_address"]="tcp://right:39010"
        source=ActualX5SensorSource(self.cfg,clock=lambda:self.now)
        data=np.arange(24,dtype=np.uint8).reshape(4,6)
        def packet(fid):
            meta={"serial":"L","fid":fid,"session_id":"source", "reference_sha256":"ref",
                  "raw":{"present":True,"dtype":"uint8","shape":[4,6],"bytes":24}}
            return [b"L",json.dumps(meta).encode(),data.tobytes()]
        source.set_context(stage="CLOSE")
        source._ingest("gel_raw",packet(1),self.now)
        self.assertEqual(source.drain_samples(),[])
        source.set_context(stage="PROBE")
        source._ingest("gel_raw",packet(2),self.now)
        recorded=source.drain_samples()
        self.assertEqual(len(recorded),1)
        self.assertEqual(recorded[0].stream,"gel_raw_left")
        np.testing.assert_array_equal(recorded[0].data,data)
        self.assertEqual(recorded[0].data.dtype,np.uint8)
        self.assertEqual(recorded[0].metadata["source_gap_count"],0)

    def test_jpeg_decode_never_holds_force_feedback_condition(self):
        import cv2
        source=ActualX5SensorSource(self.cfg,clock=lambda:self.now)
        ok,jpeg=cv2.imencode(".jpg",np.zeros((8,9,3),np.uint8))
        original=cv2.imdecode
        def decoder(*args,**kwargs):
            self.assertFalse(source._condition._is_owned())
            return original(*args,**kwargs)
        from unittest.mock import patch
        with patch.object(cv2,"imdecode",side_effect=decoder):
            source._ingest("left_wrist",[jpeg.tobytes()],self.now)

    def test_actual_snapshot_archives_encoded_data_with_honest_provenance(self):
        source=ActualX5SensorSource(self.cfg,calibrations=self.cal,clock=lambda:self.now)
        for serial in ("L","R"):
            source._ingest("dd",existing_packet(serial),self.now)
            source._ingest("wrench",wrench_packet(serial),self.now)
        snap=source.snapshot();samples={x.stream:x for x in source.drain_samples()}
        self.assertEqual(snap["timestamp_ns"],self.now)
        self.assertIsNone(snap["fault"])
        self.assertEqual(snap["dd"][0].shape,(96,128,3))
        np.testing.assert_array_equal(snap["normal_N"],[4.,4.])
        self.assertFalse(samples["dd_left"].metadata["native_dd"])
        self.assertEqual(samples["dd_encoded_left"].data.dtype,np.dtype("<u2"))
        self.assertEqual(float(snap["dd"][0][0,0,0]),float(np.float16(1.234)))

    def test_snapshot_feedback_matches_individual_fids_when_one_finger_advances(self):
        source=ActualX5SensorSource(self.cfg,calibrations=self.cal,clock=lambda:self.now)
        for serial in ("L","R"):
            source._ingest("dd",existing_packet(serial),self.now)
            source._ingest("wrench",wrench_packet(serial),self.now)
        source.drain_samples()
        old_right_stamp=self.now
        self.now+=1_000_000
        source._ingest("wrench",wrench_packet("L",8,normal=9.),self.now)
        snap=source.snapshot()
        np.testing.assert_array_equal(snap["normal_N"],[18.,4.])
        np.testing.assert_array_equal(snap["normal_N"],snap["wrench"][:,2])
        self.assertEqual(snap["source_sequences"]["wrench_left"],8)
        self.assertEqual(snap["source_sequences"]["wrench_right"],7)
        self.assertEqual(snap["source_timestamp_ns"]["wrench_left"],self.now)
        self.assertEqual(snap["source_timestamp_ns"]["wrench_right"],old_right_stamp)
        self.assertFalse(any(x.stream=="normal_force_N" for x in source.drain_samples()))

    def test_dd_wire_shape_and_dtype_change_rejected_before_recording(self):
        source=ActualX5SensorSource(self.cfg)
        source._ingest("dd",existing_packet(),self.now)
        source.drain_samples()
        for key,value in (("shape",[48,256,3]),("dtype","float32")):
            parts=existing_packet(fid=8);m=json.loads(parts[1]);m["frame"][key]=value;parts[1]=json.dumps(m).encode()
            with self.assertRaises(SensorFault):source._ingest("dd",parts,self.now)
            self.assertEqual(source.drain_samples(),[])
        self.assertEqual(source._latest["dd_left"].data.shape,(96,128,3))
        self.assertEqual(source._latest["dd_left"].data.dtype,np.dtype("float32"))

    def test_raw_uint8_image_cannot_be_mislabelled_dd(self):
        parts=[b"L",json.dumps({"serial":"L","raw":{"dtype":"uint8","shape":[480,640]}}).encode(),bytes(480*640)]
        with self.assertRaises(SensorFault):decode_existing_dd(parts)

    def test_legacy_encoded_dd_marks_unverified_pairing(self):
        parts=existing_packet();m=json.loads(parts[1]);m.pop("schema");m.pop("encoding")
        m["frame"]["channels"]=["depth","deformation_x","deformation_y"]
        m["depth"]={"channel":0,"scale":1000.,"offset":0.}
        m["deformation"]={"channels":[1,2],"scale":1000.,"offset":30000.}
        parts[1]=json.dumps(m).encode()
        meta,encoded,dd=decode_existing_dd(parts)
        self.assertFalse(meta["depth_flow_fid_alignment_verified"])
        self.assertEqual(meta["dd_source"],"legacy_depth_deformation_u16")

    def test_missing_physical_conversion_is_not_identity_conversion(self):
        source=ActualX5SensorSource(self.cfg)
        with self.assertRaisesRegex(SensorFault,"N conversion"):source.snapshot()

    def test_existing_camera_jpeg_protocol_preserves_original_bytes(self):
        import cv2
        source=ActualX5SensorSource(self.cfg,clock=lambda:self.now)
        original=np.full((8,9,3),51,np.uint8)
        ok,jpeg=cv2.imencode('.jpg',original);self.assertTrue(ok)
        payload=jpeg.tobytes();source._ingest("left_wrist",[payload],self.now)
        sample=source.drain_samples()[0]
        self.assertEqual(sample.data,payload)
        import hashlib
        self.assertEqual(sample.metadata["receiver_validated_jpeg_sha256"],hashlib.sha256(payload).hexdigest())
        self.assertIsNone(sample.metadata["source_sequence"])
        self.assertEqual(source._latest["camera/left_wrist"].data.shape,(8,9,3))

    def test_selected_camera_samples_keep_actual_distinct_receive_timestamps(self):
        import cv2
        source=ActualX5SensorSource(self.cfg,clock=lambda:self.now)
        _,jpeg=cv2.imencode('.jpg',np.full((8,9,3),51,np.uint8))
        result=[]
        thread=threading.Thread(target=lambda:result.append(source.image_samples(timeout_s=.2)))
        thread.start();time.sleep(.01)
        self.now+=1_000_000;left_time=self.now
        source._ingest("left_wrist",[jpeg.tobytes()],self.now)
        time.sleep(.01);self.assertTrue(thread.is_alive())
        self.now+=1_000_000;right_time=self.now
        source._ingest("right_wrist",[jpeg.tobytes()],self.now)
        thread.join(.2);self.assertFalse(thread.is_alive())
        samples=result[0]
        self.assertEqual(samples["left_wrist"].monotonic_ns,left_time)
        self.assertEqual(samples["right_wrist"].monotonic_ns,right_time)
        self.assertEqual(samples["left_wrist"].metadata["timestamp_kind"],"host_receive_monotonic_only")
        self.assertIsNone(samples["left_wrist"].device_timestamp)
        samples["left_wrist"].data[:]=0
        self.assertTrue(source._latest["camera/left_wrist"].data.any())

    def test_jpeg_queue_keeps_each_received_frame_and_rejects_size_change(self):
        import cv2
        source=ActualX5SensorSource(self.cfg,clock=lambda:self.now)
        packets=[]
        for value in (11,55,99):
            _,encoded=cv2.imencode('.jpg',np.full((8,9,3),value,np.uint8));payload=encoded.tobytes()
            packets.append(payload);self.now+=1;source._ingest("left_wrist",[payload],self.now)
        samples=source.drain_samples()
        self.assertEqual([x.data for x in samples],packets)
        self.assertEqual([x.metadata["receiver_sequence"] for x in samples],[0,1,2])
        self.assertEqual([x.monotonic_ns for x in samples],sorted(x.monotonic_ns for x in samples))
        _,different=cv2.imencode('.jpg',np.zeros((9,8,3),np.uint8))
        with self.assertRaisesRegex(SensorFault,"dimensions changed"):
            source._ingest("left_wrist",[different.tobytes()],self.now)
        self.assertEqual(source.drain_samples(),[])

    def test_unloaded_startup_zero_uses_new_messages_and_never_sdk_reset(self):
        sensors=self.cfg["hardware"]["public_ros"]["sensors"]
        sensors["force_conversion"]={"force_scale_to_N":2.,"moment_scale_to_unit":1.,"moment_unit":"Nmm",
            "normal_axis":2,"normal_signs":{"left":1,"right":1},"evidence":"fake verified measurement"}
        # Old YAML unloaded confirmation does not authorize a new offset.
        sensors["startup_zero"]={"samples":2,"timeout_s":.5,"unloaded_confirmed":True}
        self.cfg.update(zero_guard={"sample_count":100,"timeout_s":2.,"force_std_max_N":.1,
                        "dd_contact_threshold":.05,"dd_contact_max_fraction":.001,"offset_delta_warn_N":.2},
                        gripper={"open_mm":60.,"open_tolerance_mm":.5},tactile={"depth_channel":0})
        source=ActualX5SensorSource(self.cfg,clock=lambda:self.now)
        for serial in ("L","R"):
            source._ingest("dd",existing_packet(serial),self.now)
        # Already-cached frames must not count toward the requested 100.
        for fid in range(1,101):
            self.now+=1
            for serial in ("L","R"):source._ingest("wrench",wrench_packet(serial,fid,normal=99.),self.now)
        source.drain_samples()
        outcome=[]
        started=threading.Event()
        def width_reader():
            started.set()
            return {"fresh":True,"valid":True,"width_mm":60.,"monotonic_ns":self.now}
        thread=threading.Thread(target=lambda:outcome.append(source.zero_unloaded(
            width_reader=width_reader,confirm=lambda _:"EMPTY")))
        thread.start();self.assertTrue(started.wait(.5));time.sleep(.01)
        for fid in range(101,201):
            self.now+=1_000_000
            for serial in ("L","R"):source._ingest("wrench",wrench_packet(serial,fid,3.),self.now)
        deadline=time.monotonic()+1
        while source.calibrations is None and time.monotonic()<deadline:time.sleep(.001)
        self.assertIsNotNone(source.calibrations)
        self.now+=1_000_000
        for serial in ("L","R"):source._ingest("wrench",wrench_packet(serial,201,7.),self.now)
        thread.join(1.);self.assertFalse(thread.is_alive())
        self.assertEqual(outcome[0]["fingers"]["left"]["offset_raw"][2],3.)
        self.assertEqual(outcome[0]["fingers"]["left"]["source_fids"],list(range(101,201)))
        np.testing.assert_array_equal(source._latest["normal_force_N"].data,[8.,8.])
        with self.assertRaisesRegex(SensorFault,"direct zero is forbidden"):source.zero_unloaded()

    def test_raw_observation_available_without_zero_and_never_calls_it_newtons(self):
        source=ActualX5SensorSource(self.cfg,clock=lambda:self.now)
        for serial in ("L","R"):
            source._ingest("dd",existing_packet(serial),self.now)
            source._ingest("wrench",wrench_packet(serial),self.now)
        value=source.observation_snapshot()
        self.assertNotIn("normal_N",value)
        self.assertEqual(value["wrench_unit"],"unknown_sdk_raw")
        np.testing.assert_array_equal(value["wrench_raw"][:,2],[2.,2.])
        self.assertIsNone(source.calibrations)
        with self.assertRaisesRegex(SensorFault,"direct zero is forbidden"):source.zero_unloaded()

    def test_transport_start_has_no_implicit_zero_even_with_old_yaml_confirmation(self):
        import cv2
        from unittest.mock import Mock
        self.cfg["hardware"]["public_ros"]["sensors"].update(
            force_conversion={"force_scale_to_N":2.,"moment_scale_to_unit":1.,"moment_unit":"Nmm",
                "normal_axis":2,"normal_signs":{"left":1,"right":1},"evidence":"fake verified measurement"},
            startup_zero={"samples":100,"timeout_s":8.,"unloaded_confirmed":True})
        class Context:
            def term(self):pass
        source=ActualX5SensorSource(self.cfg,clock=lambda:self.now,_context_factory=Context)
        for serial in ("L","R"):
            source._ingest("dd",existing_packet(serial),self.now)
            source._ingest("wrench",wrench_packet(serial),self.now)
        _,jpeg=cv2.imencode('.jpg',np.zeros((8,9,3),np.uint8))
        for name in ("left_wrist","right_wrist"):source._ingest(name,[jpeg.tobytes()],self.now)
        source._receive=lambda kind:source._stop.wait(.5)
        source.zero_unloaded=Mock(side_effect=AssertionError("start must not zero"))
        try:
            source.start(timeout_s=.1)
            source.zero_unloaded.assert_not_called()
            self.assertIsNone(source.calibrations)
            self.assertIsNone(source.startup_zero_report)
        finally:source.close()

    def test_failed_guard_cannot_install_offset_or_consume_old_unloaded_declaration(self):
        from bottle_scan.zero_guard import ZeroGuardError
        self.cfg["hardware"]["public_ros"]["sensors"].update(
            force_conversion={"force_scale_to_N":2.,"moment_scale_to_unit":1.,"moment_unit":"Nmm",
                "normal_axis":2,"normal_signs":{"left":1,"right":1},"evidence":"fake verified measurement"},
            startup_zero={"unloaded_confirmed":True})
        self.cfg.update(zero_guard={"sample_count":100,"timeout_s":.05,"force_std_max_N":.1,
                        "dd_contact_threshold":.05,"dd_contact_max_fraction":.001,"offset_delta_warn_N":.2},
                        gripper={"open_mm":60.,"open_tolerance_mm":.5},tactile={"depth_channel":0})
        source=ActualX5SensorSource(self.cfg,clock=lambda:self.now)
        for serial in ("L","R"):
            source._ingest("dd",existing_packet(serial),self.now)
            source._ingest("wrench",wrench_packet(serial),self.now)
        for width,answer,reason in ((40.,"EMPTY","w_open"),(60.,"yes","EMPTY")):
            with self.assertRaisesRegex(ZeroGuardError,reason):
                source.zero_unloaded(width_reader=lambda:{"width_mm":width,"fresh":True,"valid":True,
                                                         "monotonic_ns":self.now},confirm=lambda _:answer)
            self.assertIsNone(source.calibrations)
            self.assertIsNone(source.startup_zero_report)
            self.assertEqual(source.last_zero_attempt["status"],"failed")

    def test_missing_dd_startup_closes_every_data_socket(self):
        import zmq
        class FakeSocket:
            def __init__(self,kind):self.kind=kind;self.closed=False
            def setsockopt(self,*args):pass
            def connect(self,address):pass
            def poll(self,*args):time.sleep(.001);return False
            def close(self,*args):self.closed=True
        class FakeContext:
            def __init__(self):self.sockets=[];self.terminated=False
            def socket(self,kind):
                sock=FakeSocket(kind);self.sockets.append(sock);return sock
            def term(self):self.terminated=True
        ctx=FakeContext()
        source=ActualX5SensorSource(self.cfg,_context_factory=lambda:ctx)
        with self.assertRaisesRegex(SensorFault,"missing/stale streams"):source.start(timeout_s=.02)
        self.assertCountEqual([x.kind for x in ctx.sockets],[zmq.SUB,zmq.SUB,zmq.PULL,zmq.PULL])
        self.assertTrue(ctx.terminated)
        self.assertTrue(all(x.closed for x in ctx.sockets))

    def test_dd_diagnostic_uses_only_sub_and_rejects_raw_image_as_dd(self):
        import zmq
        now=[1_000_000_000]
        class FakeSocket:
            def __init__(self):
                self.messages=[[b"L",json.dumps({"serial":"L","fid":5,"raw":{
                    "dtype":"uint8","shape":[480,640]}}).encode(),b"raw"]]
            def setsockopt(self,*args):pass
            def connect(self,*args):pass
            def poll(self,*args):now[0]+=20_000_000;return bool(self.messages)
            def recv_multipart(self):return self.messages.pop(0)
            def close(self,*args):pass
        class FakeContext:
            def socket(self,kind):self.kind=kind;return FakeSocket()
            def term(self):pass
        ctx=FakeContext()
        result=inspect_dd_pub(serials=("L","R"),duration_s=.05,
            _context_factory=lambda:ctx,_clock=lambda:now[0])
        self.assertEqual(ctx.kind,zmq.SUB)
        self.assertFalse(result["ready"])
        self.assertEqual(result["sensors"]["L"]["dd_count"],0)
        self.assertEqual(result["sensors"]["L"]["raw_image_metadata"]["dtype"],"uint8")


if __name__=="__main__":unittest.main()
