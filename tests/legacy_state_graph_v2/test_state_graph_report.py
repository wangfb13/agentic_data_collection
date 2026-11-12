"""Schema and separation regressions for the historical report/replay."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from skills.contact_position_a.legacy_v2 import report, cli

def row(ident, **kw):
    d={"node_id":ident,"session":"white/session","trial_id":ident,"window_index":0,
       "timestamp_ns":10,"t_ns":10,"phase":"A","bilateral_contact":True,
       "object_id":"white_light","protocol_id":"p","closure_floor":.336,
       "planned_delta_mm":0.,"tcp_offset_a":0.,"downstream_success":None,
       "original_skill_ground_truth":{"initial_class":"already_normal","final_target_reached":True},
       "original_skill_window_labels":None,"downstream_evidence":{"evaluated":False},
       "protocol":{"probe_force_sdk":1.5,"target_force_sdk":5.,"closure_floor":.336},
       "rule_id":"old","trial_started_unix_ns":100,"session_created_unix_ns":90}
    d.update(kw);return d

def noise_stub(rows,cfg=None):
    return {"modalities":{m:{"status":"insufficient_data"} for m in (cfg or {}).get("modalities",[])}}

class ReportSchemaTests(unittest.TestCase):
    def _validate(self,rows):
        with tempfile.TemporaryDirectory() as d,patch.object(report,"score_ablation",return_value={"status":"unavailable","V1":None,"V2":None,"V3":None}),patch.object(report,"fit_noise",side_effect=noise_stub),patch.object(report,"distance_matrix",side_effect=lambda rs,*a,**kw:np.full((len(rs),len(rs)),np.inf)),patch.object(report,"validate_e1_e2",return_value={"status":"unavailable"}):
            return report.run_validation(rows,{"R":{r["node_id"]:None for r in rows}}, {}, [{"node_id":r["node_id"],"invocation_label":None,"exit_label":0} for r in rows],[],[],d)
    def test_actual_white_b_nullable_window_labels(self):
        result=self._validate([row("B",phase="B_pre")])
        c=result["V7"]["comparisons"][0]
        self.assertEqual(c["I_window_label"],0)
        self.assertIsNone(c["beta_window_label"])
    def test_initial_label_not_assigned_to_post_window(self):
        result=self._validate([row("B1",phase="B_post",window_index=1,
            original_skill_ground_truth={"initial_class":"needs_recovery","final_target_reached":True})])
        self.assertIsNone(result["V7"]["comparisons"][0]["I_window_label"])
    def test_only_exact_timestamp_beta_is_used(self):
        r=row("A",original_skill_window_labels=[
            {"timestamp_ns":9,"target_reached":False},
            {"timestamp_ns":10,"target_reached":True}])
        self.assertIs(self._validate([r])["V7"]["comparisons"][0]["beta_window_label"],True)
    def test_formal_B_is_only_external_transition_validation(self):
        a=row("A",session="/outputs/e1e2_A_formal01")
        b=row("B",session="/outputs/e1e2_B_formal01",phase="B_pre",evaluation_only=True,downstream_success=True)
        calls=[]
        def build(rs,nm,cfg):
            calls.append((rs,cfg))
            return {"counts":{"kinematic_confirmed":0,"kinematic_refuted":0,"kinematic_unverified":0},"training_labels_allowed":False}
        with tempfile.TemporaryDirectory() as d,patch.object(report,"fit_noise",side_effect=noise_stub),patch("skills.contact_position_a.legacy_v2.graph.build_graph",side_effect=build):
            result=report.validate_e1_e2([b,a],d,{})
        self.assertFalse(result["B_used_for_training"])
        for rs,cfg in calls:
            self.assertEqual([x["node_id"] for x in rs],["A"])
            self.assertEqual([x["node_id"] for x in cfg["validation_rows"]],["B"])

class ReplayTests(unittest.TestCase):
    def test_order_is_wall_time_and_trials_are_atomic(self):
        later=row("later",trial_started_unix_ns=200,t_ns=1)
        pre=row("pre",trial_id="early",phase="B_pre",trial_started_unix_ns=100,t_ns=900,window_index=0)
        post=row("post",trial_id="early",phase="B_post",trial_started_unix_ns=100,t_ns=901,window_index=1)
        sealed=row("sealed",evaluation_only=True,trial_started_unix_ns=50)
        prefixes=[]
        def build(rs,*args):
            prefixes.append([r["node_id"] for r in rs])
            return {},{"graph_version":"g"},[],[],[]
        cfg={"replay_every_trials":1,"rule":{"rule_id":"rule","active_axes":["a"],"range_mm":{"a":[-5,5]},"objects":["white_light","white_heavy"]},"spec":{}}
        with tempfile.TemporaryDirectory() as d,patch.object(cli,"build_artifacts",side_effect=build),patch("bottle_scan.skill_director.symbolic_context",return_value={"applicable_operators":[]}),patch("bottle_scan.skill_director.advance_symbolic",return_value={"status":"external_agent_disabled"}),patch("skills.contact_position_a.legacy_v2.director_bridge.update_hypotheses",return_value=[]):
            result=cli.history_replay([later,post,sealed,pre],d,cfg)
        self.assertEqual(prefixes,[["pre","post"],["pre","post","later"]])
        self.assertEqual(result["status"],"protocol_replay_only")
        self.assertFalse(result["external_llm_called"])

if __name__=="__main__":unittest.main()
