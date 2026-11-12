import copy
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from bottle_scan.skill_height_learning import train_height_policy
from bottle_scan.skill_extension import make_probe_design, ExtensionRecoveryRunner, RecoveryProbeBounds
from bottle_scan.skill_exploration import default_exploration_config
from bottle_scan.e1e2_core import defaults
from bottle_scan.session import atomic_json
from bottle_scan.contact_policy import ContactPolicy
from bottle_scan.recovery_skill import RecoverySkill
from tests.test_ambiguity_collect import CapturingHardware, answer


def create_data(root, *, ambiguous=False):
    cfg=defaults(); cfg["gripper"]["min_normalized"]=.336
    ex=default_exploration_config()
    ex["variables"]["a_mm"].update(range=[-4.,4.],levels=[-4.,0.,4.])
    ex["variables"]["probe_force"].update(reference=1.5,levels=[1.,1.5,2.])
    design=make_probe_design(ex,cfg,"height_controlled",repeats=8)
    T0=np.eye(4);T0[2,3]=200
    meta={"schema":"ambiguity_session_v1","simulated":True,"config":cfg,
          "T0":T0.tolist(),"ambiguity_design":design,"evaluation_role":"development"}
    atomic_json(root/"session_meta.json",meta)
    for row in design["plan"]:
        x=row["perturbation"]["l_mm"]
        T=T0.copy();T[:3,3]+=np.asarray(cfg["axes_tcp"]["l"])*x
        center=40. if ambiguous else 40.+x*3.
        desc={"valid":True,"centroid_px":[center,45.],"bbox_xyxy_px":[center-4,41,center+4,49],
              "area_px":64.,"principal_angle_deg":0.,"touches_edge":False,"edge_sides":{}}
        probe={"valid":True,"stable_probe":True,"measurement_timestamp_ns":1000000000,
               "measured_tcp_mm":T.tolist(),"wrench_zeroed":[[0,0,-1.5,0,0,0]]*2,"descriptors":[copy.deepcopy(desc),copy.deepcopy(desc)]}
        trial={**row,"complete":True,"recording_complete":True,"label":"normal","probe_initial":probe}
        atomic_json(root/"trials"/row["id"]/"meta.json",trial)
        atomic_json(root/"trials"/row["id"]/"recording_status.json",{"complete":True})
    return cfg,design


def test_measured_l_model_train_and_axis_guard(tmp_path):
    cfg,design=create_data(tmp_path)
    model=train_height_policy([tmp_path],{"contact_policy":{"epochs":30}})
    assert model["usable_for_B"],model["blocking_reasons"]
    assert model["recovery_action_axis"]=="l"
    policy=ContactPolicy(model,cfg["servo"])
    skill=RecoverySkill(model,cfg["servo"],{"action_axis":"l"},mode="development")
    assert set(model["split"]["training_groups"]).isdisjoint(model["split"]["validation_groups"])
    with pytest.raises(ValueError):RecoverySkill(model,cfg["servo"],{"action_axis":"a"},mode="development")
    # Editing hidden planned values cannot affect model inputs or measured labels.
    for path in (tmp_path/"trials").glob("*/meta.json"):
        value=json.loads(path.read_text());value["perturbation"]["l_mm"]=123456.
        value["delta_mm"]=-999.;atomic_json(path,value)
    other=train_height_policy([tmp_path],{"contact_policy":{"epochs":30}})
    assert other["networks"]==model["networks"]
    assert other["validation"]==model["validation"]


def test_ambiguous_height_contact_is_not_usable(tmp_path):
    create_data(tmp_path,ambiguous=True)
    model=train_height_policy([tmp_path],{"contact_policy":{"epochs":10}})
    assert not model["usable_for_B"]
    assert any("separable" in x or "invertible" in x for x in model["blocking_reasons"])


def test_independent_or_force_session_cannot_train(tmp_path):
    create_data(tmp_path)
    path=tmp_path/"session_meta.json";meta=json.loads(path.read_text())
    meta["evaluation_role"]="independent";atomic_json(path,meta)
    model=train_height_policy([tmp_path])
    assert not model["usable_for_B"]


def test_B_fake_trial_only_moves_l_or_retreat_and_records_handoff(tmp_path):
    source=tmp_path/"source";cfg,design=create_data(source)
    model=train_height_policy([source],{"contact_policy":{"epochs":30}})
    out=tmp_path/"B";out.mkdir()
    hw=CapturingHardware(cfg)
    args=SimpleNamespace(out=out,step=False,cond="A",limit=None)
    runner=ExtensionRecoveryRunner(args,cfg,design,hw,fit=model,mode="development",
                                  evaluation_role="development",ask=answer,say=lambda _:None)
    runner.T0=hw.pose();runner.bounds=RecoveryProbeBounds(cfg,runner.T0,design)
    runner.last_commanded_pose=runner.T0.copy();runner.physical_completed=2
    runner.meta={"schema":"ambiguity_session_v1","T0":runner.T0.tolist(),
                 "config":cfg,"ambiguity_design":design,"plan":design["plan"]}
    runner.trial(design["plan"][0])
    record=json.loads((out/"trials"/design["plan"][0]["id"]/"meta.json").read_text())
    assert record["cond"]=="B"
    assert "skill_handoff" in record
    assert not record["lifted"]
    assert record["skill_handoff"]["completed"] in (True,False)
    assert runner.cost_ledger.summary()["categories"]["B_development"]["trial_attempts"]==1


def test_l_B_action_projects_receipt_on_l_and_reobserves(tmp_path,monkeypatch):
    source=tmp_path/"source";cfg,design=create_data(source)
    model=train_height_policy([source],{"contact_policy":{"epochs":20}})
    out=tmp_path/"B";out.mkdir()
    hw=CapturingHardware(cfg)
    args=SimpleNamespace(out=out,step=False,cond="A",limit=None)
    runner=ExtensionRecoveryRunner(args,cfg,design,hw,fit=model,mode="development",
                                  evaluation_role="development",ask=answer,say=lambda _:None)
    runner.T0=hw.pose();runner.bounds=RecoveryProbeBounds(cfg,runner.T0,design)
    runner.last_commanded_pose=runner.T0.copy();runner.physical_completed=2
    runner.meta={"schema":"ambiguity_session_v1","T0":runner.T0.tolist(),
                 "config":cfg,"ambiguity_design":design,"plan":design["plan"]}
    receipts=[]
    class ScriptedSkill:
        def __init__(self,*args,**kwargs): self.events=[]
        def start(self,obs):
            return {"exit":{"hand_back":False,"completed":False,"reason":"test_requires_l"},
                    "recovery":{"action":{"axis":"l","delta_mm":-4.}}}
        def advance(self,obs,applied_action=None):
            receipts.append(applied_action)
            return {"exit":{"hand_back":True,"completed":True,"reason":"test_observed_target"},
                    "recovery":{"action":None}}
        def abort(self,reason): raise AssertionError(reason)
    monkeypatch.setattr("bottle_scan.skill_collection.RecoverySkill",ScriptedSkill)
    item=next(x for x in design["plan"] if x["perturbation"]["l_mm"]==4.)
    runner.trial(item)
    record=json.loads((out/"trials"/item["id"]/"meta.json").read_text())
    assert receipts[0]["action_mm"]==pytest.approx(-4.)
    assert record["regrasps"]==1
    assert record["skill_handoff"]["completed"]
    initial=np.asarray(record["probe_initial"]["measured_tcp_mm"])
    final=np.asarray(record["probe_final"]["measured_tcp_mm"])
    assert float((final[:3,3]-initial[:3,3]) @ runner.bounds.l)==pytest.approx(-4.)
    assert float((final[:3,3]-initial[:3,3]) @ runner.bounds.a)==pytest.approx(0.)
    assert not record["lifted"]


def test_B_reference_preflight_precedes_zero_or_authorization(tmp_path):
    source=tmp_path/"source";cfg,design=create_data(source)
    model=train_height_policy([source],{"contact_policy":{"epochs":20}})
    out=tmp_path/"B";out.mkdir()
    hw=CapturingHardware(cfg)
    hw.zero_unloaded=lambda:pytest.fail("zero must not occur on mismatched T0")
    args=SimpleNamespace(out=out,step=False,cond="A",limit=None)
    runner=ExtensionRecoveryRunner(args,cfg,design,hw,fit=model,mode="development",
                                  evaluation_role="development",
                                  ask=lambda _:pytest.fail("authorization must not precede T0 check"),say=lambda _:None)
    with pytest.raises(ValueError,match="reference T0"):
        runner.initial()
    changed=copy.deepcopy(cfg);changed["force"]["probe_fraction"]=.4
    with pytest.raises(ValueError,match="protocol"):
        ExtensionRecoveryRunner(args,changed,design,hw,fit=model,mode="development",
                                evaluation_role="development",ask=answer,say=lambda _:None)
