import tempfile
import unittest
from pathlib import Path
import numpy as np
from skills.contact_position_a.legacy_v2.contact_geometry import contact_geometry,window_contact_geometry,object_contact_position
from skills.contact_position_a.legacy_v2.extract_nodes import probe_windows,downstream_outcome,measured_offsets,_zero_reference,_window_indices,_roles,write_nodes,_receipt
class GeometryTests(unittest.TestCase):
    def test_threshold_weighted_centroid_and_units(self):
        ref=np.ones((6,8,3))*7;dd=ref.copy()
        dd[2,2,0]+=1;dd[2,4,0]+=3;dd[0,0,0]+=.01
        d=contact_geometry(dd,ref,threshold=.1,pixel_pitch_mm=[.2,.3],root_edge="left")
        self.assertEqual(d["area_px"],2);self.assertAlmostEqual(d["u_px"],3.5)
        self.assertAlmostEqual(d["u_mm"],.7);self.assertAlmostEqual(d["area_mm2"],.12)
        self.assertAlmostEqual(d["edge_dist_root_mm"],.4)
    def test_missing_calibration_keeps_mm_null(self):
        dd=np.ones((4,6,3));d=contact_geometry(dd,np.zeros_like(dd),threshold=.1)
        self.assertIsNone(d["u_mm"]);self.assertIsNone(d["edge_class"])
        self.assertIsNone(object_contact_position(d,np.eye(4),{})["h_mm"])
    def test_object_transform_requires_explicit_geometry(self):
        d={"u_mm":2,"v_mm":3}
        result=object_contact_position(d,np.eye(4),{"T_tcp_sensor":np.eye(4).tolist(),"T_base_object":np.eye(4).tolist(),"plane_z_mm":5})
        self.assertEqual(result["h_mm"],5);self.assertAlmostEqual(result["theta_rad"],np.arctan2(3,2))
    def test_window_features_do_not_depend_on_analysis_labels(self):
        dd=np.zeros((5,6,8,3));dd[:,2:4,2:5,0]=1
        d=window_contact_geometry(dd,np.zeros((6,8,3)),threshold=.1)
        self.assertNotIn("tcp_offset_a",d);self.assertNotIn("planned_delta_mm",d)

class ExtractTests(unittest.TestCase):
    def test_window_order_dedupe(self):
        a={"valid":True,"timestamp_ns":10};b={"valid":True,"timestamp_ns":20}
        trial={"probe_initial":a,"probe_final":b,"probe_regrasps":[{"probe":b}]}
        self.assertEqual(len(probe_windows(trial)),2)
    def test_success_belongs_only_to_terminal_window(self):
        trial={"stages":[{"name":"SQUEEZE"}],"status":"hold_complete","target_reached":True,"lifted":True,"hold_complete":True,"dropped":False,"label":"normal"}
        self.assertIsNone(downstream_outcome(trial,False)[0]);self.assertIs(downstream_outcome(trial,True)[0],True)
        trial["target_reached"]=False
        self.assertIs(downstream_outcome(trial,True)[0],False)
    def test_no_downstream_test_is_unknown(self):
        self.assertIsNone(downstream_outcome({"status":"skill_incomplete","label":"normal"},True)[0])
    def test_actual_position_never_planned_command(self):
        t=np.eye(4);t[:3,3]=[4,2,6]
        d=measured_offsets(t,np.eye(4),{"a":[0,0,1],"l":[1,0,0]})
        self.assertEqual(d,{"a":6.,"l":4.})
    def test_reference_must_match_trial_zero(self):
        session={"zero_history":[{"created_unix_ns":1,"dd_reference_file":"first.npz"},{"created_unix_ns":2,"dd_reference_file":"second.npz"}]}
        self.assertEqual(_zero_reference(session,{"zero":{"created_unix_ns":1}})[0],Path("first.npz"))
        with self.assertRaises(ValueError):_zero_reference(session,{"zero":{"created_unix_ns":3}})
    def test_frame_indices_are_independent_and_exact(self):
        s={"times":np.arange(10)}
        self.assertEqual(_window_indices(s,[1,2,3,4,5],0,6),[1,2,3,4,5])
        with self.assertRaises(ValueError):_window_indices(s,[1,1,1,1,1],0,6)
        with self.assertRaises(ValueError):_window_indices(s,[1,2,3,4,99],0,6)
    def test_formal_b_and_validation_not_training(self):
        self.assertTrue(_roles(Path("e1e2_B_formal01"),{"cond":"B"},{})[1])
        self.assertTrue(_roles(Path("rsi"),{"agentic":{"role":"validation"}},{})[1])
        self.assertFalse(_roles(Path("dev"),{"agentic":{"role":"development"}},{})[1])
    def test_observed_receipt_cannot_bridge_missing_probe_and_multiple_actions(self):
        before={"t_ns":10,"tcp_offset_a":0.,"tcp_receipt":{"index":1}}
        after={"t_ns":100,"tcp_offset_a":-12.,"tcp_receipt":{"index":5}}
        trial={"correction_history":[{"timestamp_ns":20,"applied_delta_mm":-6.},{"timestamp_ns":60,"applied_delta_mm":-6.}],"stages":[{"name":"REGRASP_CORRECT","start_ns":30,"end_ns":40},{"name":"REGRASP_CORRECT","start_ns":70,"end_ns":80}]}
        self.assertIsNone(_receipt(trial,1,before,after,Path("meta.json")))
        trial["correction_history"]=trial["correction_history"][:1];trial["stages"]=trial["stages"][:1]
        self.assertEqual(_receipt(trial,1,before,after,Path("meta.json"))["applied_delta_mm"],-6.)
    def test_parquet_roundtrip_nested_json(self):
        import pyarrow.parquet as pq
        with tempfile.TemporaryDirectory() as tmp:
            write_nodes([{"node_id":"one","frames":{"left":{"indices":[1,2,3,4,5]}},"legacy_features":[1.,2.],"downstream_success":None}],tmp)
            table=pq.read_table(Path(tmp)/"nodes.parquet")
            self.assertEqual(table.num_rows,1)
            self.assertIn(b"json_encoded_columns",table.schema.metadata)
if __name__=="__main__":unittest.main()
