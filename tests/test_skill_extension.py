import copy
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from bottle_scan.skill_extension import make_probe_design, validate_probe_design, main, ExtensionProbeRunner
from bottle_scan.skill_exploration import default_exploration_config
from bottle_scan.ambiguity_plan import validate_design, ProbeBounds
from bottle_scan.e1e2_core import defaults
from bottle_scan.session import atomic_json
from tests.test_ambiguity_collect import CapturingHardware, answer


def settings():
    cfg = defaults(); cfg["gripper"]["min_normalized"] = .336
    explore = default_exploration_config()
    explore["variables"]["a_mm"].update(range=[-4.,4.], levels=[-4.,0.,4.])
    explore["variables"]["probe_force"].update(reference=1.5,levels=[1.,1.5,2.])
    return explore, cfg


def test_plan_hash_tamper_and_bounds():
    ex, cfg = settings()
    design = make_probe_design(ex,cfg,"height_controlled")
    assert validate_design(design) == design
    assert len(design["plan"]) == 9
    assert all(p["delta_mm"] == 0 for p in design["plan"])
    bad = copy.deepcopy(design); bad["plan"][1]["perturbation"]["l_mm"] = 100
    with pytest.raises(ValueError): validate_probe_design(bad)
    hw = CapturingHardware(cfg)
    bounds = ProbeBounds(cfg,hw.pose(),design)
    assert all(bounds.check(bounds.target(x)) is None for x in design["plan"])


def test_force_only_does_not_change_geometry_or_hard_limits():
    ex,cfg = settings()
    d = make_probe_design(ex,cfg,"force_controlled")
    assert {x["probe_force_target"] for x in d["plan"]} == {1.,1.5,2.}
    assert all(x["perturbation"] == {"a_mm":0.,"l_mm":0.,"tilt_deg":0.} for x in d["plan"])
    ex["variables"]["probe_force"]["levels"] = [100.]
    with pytest.raises(ValueError): make_probe_design(ex,cfg,"force_controlled")


def test_existing_translation_envelope_preserved():
    ex,cfg = settings(); ex["variables"]["a_mm"]["levels"] = [-10.,0.,10.]
    with pytest.raises(ValueError): make_probe_design(ex,cfg,"small_combinations")


def test_collect_requires_explicit_real_and_does_not_connect(tmp_path):
    with pytest.raises(SystemExit):
        main(["collect","--out",str(tmp_path),"--design-file","missing"],make_hardware=lambda _:pytest.fail("hardware"))


def test_fake_trial_reobserves_measured_height_and_records_cost(tmp_path):
    ex,cfg = settings(); design = make_probe_design(ex,cfg,"height_controlled")
    hw = CapturingHardware(cfg)
    args = SimpleNamespace(out=tmp_path,step=False,cond="A",limit=None)
    runner = ExtensionProbeRunner(args,cfg,design,hw,ask=answer,say=lambda _:None)
    runner.T0 = hw.pose(); runner.bounds = ProbeBounds(cfg,runner.T0,design)
    runner.last_commanded_pose = runner.T0.copy(); runner.physical_completed = 2
    runner.meta = {"schema":"ambiguity_session_v1","T0":runner.T0.tolist(),
                   "config":cfg,"ambiguity_design":design,"plan":design["plan"]}
    item = next(p for p in design["plan"] if p["perturbation"]["l_mm"] != 0)
    runner.trial(item)
    record = json.loads((tmp_path/"trials"/item["id"]/"meta.json").read_text())
    assert record["skill_extension"]["runtime_context"]["measured_contact_height_l_mm"] == pytest.approx(item["perturbation"]["l_mm"])
    assert record["skill_extension"]["completion_label"] == "unknown"
    assert cfg["force"]["probe_fraction"] == pytest.approx(.3)
    assert runner.cost_ledger.summary()["new_physical_attempts"] == 1
    assert not record["lifted"]
