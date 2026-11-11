"""Staged independent-contact-factor exploration using the existing PROBE runner.

The extension is data collection, not a claim of validated height recovery.
No imports of ROS or hardware connections occur until collect --real.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import time
import numpy as np
import yaml
from .skill_exploration import (default_exploration_config, validate_exploration_config,
                               extension_design, CostLedger)
from .session import fingerprint, atomic_json
from .e1e2_core import load_settings, validate
from .ambiguity_collect import ProbeRunner, main as probe_main
from .ambiguity_plan import plan_report, ProbeBounds

def load_extension_config(path):
    path = Path(path).resolve()
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict) or data.get("schema") != "bottle_skill_extension_config_v1":
        raise ValueError("invalid skill extension config")
    base = Path(data["base_config"])
    if not base.is_absolute():
        base = path.parent / base
    cfg = load_settings(base)
    validate(cfg, require_resolved=False)
    exploration = default_exploration_config()
    exploration.update(deepcopy(data.get("exploration", {})))
    exploration["variables"] = deepcopy(default_exploration_config()["variables"])
    for key, overrides in data.get("variables", {}).items():
        if key not in exploration["variables"]:
            raise ValueError("unknown physical variable " + key)
        exploration["variables"][key].update(overrides)
    force = exploration["variables"]["probe_force"]
    if force["reference"] is None:
        force["reference"] = cfg["force"]["target"] * cfg["force"]["probe_fraction"]
    validate_exploration_config(exploration)
    return data, exploration, cfg, base

def make_probe_design(exploration, cfg, stage, *, repeats=3, seed=20261105, bottle="bottle"):
    if type(repeats) is not int or not 3 <= repeats <= 10:
        raise ValueError("3..10 independent contact repetitions required")
    if not isinstance(bottle, str) or not bottle or "/" in bottle or "\\" in bottle:
        raise ValueError("invalid bottle name")
    cfg = deepcopy(cfg)
    validate(cfg, require_resolved=False)
    exploration = validate_exploration_config(exploration)
    a, l = (np.asarray(cfg["axes_tcp"][axis], float) for axis in ("a", "l"))
    if abs(float(a @ l)) > 1e-6:
        raise ValueError("extension contact factor l must be physically independent of a")
    design = extension_design(exploration, stage, seed=seed)
    plan = []
    for repeat in range(repeats):
        block = extension_design(exploration, stage, seed=seed + repeat)
        for i, row in enumerate(block["rows"]):
            condition = row["condition"]
            a_mm, l_mm, force = condition["a_mm"], condition["contact_height_l_mm"], condition["probe_force"]
            if math.hypot(a_mm, l_mm) > 8. or abs(a_mm) > 5. or abs(l_mm) > 5.:
                raise ValueError("extension must remain within original 8mm translation, 5mm per-axis envelope")
            local = deepcopy(cfg)
            local["force"]["probe_fraction"] = force / local["force"]["target"]
            validate(local, require_resolved=False)
            plan.append({"id": f"{len(plan)+1:05d}", "index": len(plan)+1, "cond": "A",
                         "bottle": bottle, "repeat": repeat, "kind": "skill_extension_probe",
                         "state_id": fingerprint(condition)[:12], "delta_mm": a_mm,
                         "perturbation": {"a_mm": a_mm, "l_mm": l_mm, "tilt_deg": 0.},
                         "probe_force_target": force, "purpose": row["purpose"],
                         "experiment_condition": condition,
                         "evaluation_role": "development"})
    result = {"schema": "skill_extension_probe_design_v1", "stage": stage, "repeats": repeats,
              "seed": seed, "bottle": bottle, "states": len(design["rows"]), "step_mm": 0., "tilt_deg": 0.,
              "tilt_axis": "c", "rotation_pivot": "tcp", "coordinates": "original_T0_local_axes",
              "max_translation_mm": 8., "max_tilt_deg": 0., "probe_only": True,
              "exploration_config": exploration, "execution_config": cfg, "plan": plan,
              "fixed_controls": design["fixed_controls"], "status": "planned_unexecuted",
              "label_semantics": "measured contact observations; geometric assignments are not optimal action labels",
              "deployment_status": "unsupported until real recovery and exit validation"}
    result["content_sha256"] = fingerprint(result)
    return result

def validate_probe_design(design):
    expected = make_probe_design(design["exploration_config"], design["execution_config"],
                                 design["stage"], repeats=design["repeats"], seed=design["seed"],
                                 bottle=design["bottle"])
    if expected != design:
        raise ValueError("skill extension design content/hash mismatch")
    return design

class ExtensionProbeRunner(ProbeRunner):
    def __init__(self, args, cfg, design, hw, **kwargs):
        validate_probe_design(design)
        expected = design["execution_config"]
        for section in ("axes_tcp", "force", "motion", "tactile"):
            # Probe fraction changes per row. Other force limits/settings cannot.
            actual_values, expected_values = deepcopy(cfg[section]), deepcopy(expected[section])
            if section == "force":
                actual_values.pop("probe_fraction"); expected_values.pop("probe_fraction")
            if actual_values != expected_values:
                raise ValueError("extension execution config differs: " + section)
        super().__init__(args, cfg, design, hw, **kwargs)
        self.cost_ledger = CostLedger(args.out)
        self.reference_fraction = expected["force"]["probe_fraction"]

    def label(self):
        super().label()
        if self.current.get("cond") == "A":
            initial=self.read("首次轻触：1需要恢复，2已经正常，3暂不适用，回车未知：")
            self.current["skill_ground_truth"]={
                "source":"operator_review_of_observed_contact",
                "initial_class":{"1":"needs_recovery","2":"already_normal","3":"not_applicable"}.get(initial),
                "final_target_reached":None,
                "independent_of_experimental_offset":True}
            self.save()

    def trial(self, item):
        started = time.monotonic()
        force = item["probe_force_target"]
        self.cfg["force"]["probe_fraction"] = force / self.cfg["force"]["target"]
        validate(self.cfg)
        # Hardware normally holds the same config object; enforce agreement for
        # adapters that copied it, without altering force targets or hard limits.
        self.hw.cfg["force"]["probe_fraction"] = self.cfg["force"]["probe_fraction"]
        try:
            return super().trial(item)
        finally:
            if self.current and self.current.get("id") == item["id"]:
                probe = self.current.get("probe_initial") or {}
                actual_tcp = probe.get("measured_tcp_mm")
                measured_l = None
                if actual_tcp is not None and self.T0 is not None:
                    measured_l = float((np.asarray(actual_tcp)[:3, 3] - self.T0[:3, 3]) @ self.bounds.l)
                self.current["skill_extension"] = {
                    "stage": self.design["stage"], "evaluation_role": "development",
                    "runtime_context": {"measured_contact_height_l_mm": measured_l,
                                        "probe_force_target": force,
                                        "measured_wrench_zeroed": probe.get("wrench_zeroed")},
                    "analysis_only": {"experiment_condition": item["experiment_condition"]},
                    "initial_class": self.current.get("skill_ground_truth", {}).get("initial_class"),
                    "applicability_label": "unknown", "need_recovery_label": "unknown",
                    "completion_label": "unknown",
                    "note": "Probe-only collection does not observe a recovery trajectory or terminal success."
                }
                self.save()
                attempted = self.current.get("status") != "skipped_bound"
                failed = attempted and (not self.current.get("complete") or self.current.get("label") != "normal" or not probe.get("valid"))
                attempt_id = str(self.folder) + ":" + str(self.current.get("started_unix_ns"))
                category = ("A_exploration" if self.current.get("cond") == "A" else
                            "B_independent" if self.current.get("evaluation_role") == "independent" else "B_development")
                if self.current.get("cond") == "B":
                    failed = attempted and not self.current.get("skill_handoff", {}).get("completed", False)
                self.cost_ledger.record(attempt_id, category, trial_attempts=int(attempted),
                    failed_attempts=int(failed), wall_seconds=time.monotonic()-started,
                    robot_seconds=None, reset_seconds=None, evidence_refs=[str(self.folder/"meta.json")],
                    attempt_ids=[attempt_id] if attempted else [], metadata={"stage": self.design["stage"],
                        "reset_note": "operator time not separately instrumented", "skipped_bound": not attempted})
            self.cfg["force"]["probe_fraction"] = self.reference_fraction
            self.hw.cfg["force"]["probe_fraction"] = self.reference_fraction
            self.save()

class RecoveryProbeBounds(ProbeBounds):
    def pre(self, target, distance=None):
        # Retain the original full open-hand pre offset; do not introduce a new
        # lateral correction path at a shortened near-contact distance.
        return super().pre(target)


class ExtensionRecoveryRunner(ExtensionProbeRunner):
    allowed_recovery_axes = ("l",)

    def __init__(self, args, cfg, design, hw, *, fit, mode, evaluation_role, certificate=None, **kwargs):
        if design["stage"] != "height_controlled":
            raise ValueError("l B initially supports fixed-a, fixed-force height conditions only")
        from .recovery_skill import RecoverySkill
        if fit.get("axes_tcp") != cfg["axes_tcp"]:
            raise ValueError("l recovery calibration axes differ")
        from .agentic_learning import _hash, _signature, _pose
        if fit.get("signature") != _hash(_signature({"config":cfg,"simulated":bool(fit.get("simulated"))})):
            raise ValueError("l recovery model force/tactile/hardware protocol differs")
        if _pose(fit.get("reference_T0")) is None:
            raise ValueError("l recovery requires a valid measured model reference T0")
        runtime = {"action_axis":"l", "max_total_action_mm":8.,
                   "required_context":{"task":"bottle","contact_protocol":"fixed_a_height_l",
                                       "probe_target_sdk":cfg["force"]["target"]*cfg["force"]["probe_fraction"]}}
        settings = {"runtime":runtime, "context":{"task":"bottle","contact_protocol":"fixed_a_height_l"},
                    "mode":mode, "certificate":certificate}
        RecoverySkill(fit, cfg["servo"], runtime, certificate, mode=mode)
        super().__init__(args,cfg,design,hw,**kwargs)
        contract = {"fit":fit,"settings":settings,"evaluation_role":evaluation_role}
        if self.meta and self.meta.get("skill_extension_B") != contract:
            raise ValueError("cannot resume with changed model/mode/role/certificate")
        self.fit, self.evaluation_role, self.contract = fit, evaluation_role, contract
        self.cfg["recovery_skill"] = settings
        self.context = {"batch_role":"final_validation" if evaluation_role=="independent" else "development"}

    def initial(self):
        from .geometry import pose_error
        # Hardware is connected, but no zeroing, authorization or motion has
        # happened here. New B sessions retain the trained measured T0 instead
        # of silently accepting the operator's current pose as a new origin.
        reference=np.asarray(self.fit["reference_T0"],float)
        mm,deg=pose_error(reference,self.hw.pose())
        if mm>self.cfg["motion"]["pose_tolerance_mm"] or deg>self.cfg["motion"]["pose_tolerance_deg"]:
            raise ValueError("restore measured model reference T0 before B initialization")
        if self.meta is None:
            from .e1e2_core import config_identity
            self.meta={"schema":"ambiguity_session_v1","cond":"B","simulated":False,
                "protocol":"probe_only_height_recovery","config":self.cfg,
                "config_identity":config_identity(self.cfg,"A",None),
                "ambiguity_design":self.design,"plan":self.design["plan"],
                "T0":reference.tolist(),"created_unix_ns":time.time_ns(),"zero_history":[],
                "training_eligible":False,"recovery_action_ground_truth":False,
                "fit":self.fit,"evaluation_role":self.evaluation_role,"skill_extension_B":self.contract}
            self.save()
        super().initial()
        from .geometry import pose_error
        if self.fit.get("reference_T0") is None:
            raise ValueError("l recovery model lacks measured calibration T0")
        mm,deg=pose_error(np.asarray(self.fit["reference_T0"]),self.T0)
        if mm>self.cfg["motion"]["pose_tolerance_mm"] or deg>self.cfg["motion"]["pose_tolerance_deg"]:
            raise ValueError("B must restore original measured l calibration T0 before recovery")
        self.bounds = RecoveryProbeBounds(self.cfg,self.T0,self.design)
        self.meta.update(cond="B", fit=self.fit, evaluation_role=self.evaluation_role,
                         skill_extension_B=self.contract,
                         plan=[{**r,"cond":"B","evaluation_role":self.evaluation_role} for r in self.design["plan"]])
        self.save()

    def observation(self, status, probe):
        from .skill_collection import SkillBatchRunner
        return SkillBatchRunner.observation(self,status,probe)

    def record_observation(self, phase, observation, result=None):
        from .skill_collection import SkillBatchRunner
        return SkillBatchRunner.record_observation(self,phase,observation,result)

    def touch(self, prefix=""):
        status,probe=super().touch(prefix)
        self.record_observation("first_touch" if not self.current.get("skill_observations") else "recovery_observation",
                                self.observation(status,probe))
        return status,probe

    def after_initial_probe(self,target,status,probe):
        from .skill_collection import SkillBatchRunner
        self.current["regrasps"]=0
        return SkillBatchRunner.apply_B(self,target,status,probe)

    def trial(self,item):
        return super().trial({**item,"cond":"B","evaluation_role":self.evaluation_role})

    def label(self):
        super().label()
        initial=self.read("回看首次轻触：1需要恢复，2已经正常，3暂不适用，回车未知：")
        final=self.read("交回控制权前是否达到目标稳定双侧内部接触：1是，2否，回车未知：")
        self.current["skill_ground_truth"]={
            "source":"operator_review_of_observed_contact",
            "initial_class":{"1":"needs_recovery","2":"already_normal","3":"not_applicable"}.get(initial),
            "final_target_reached":{"1":True,"2":False}.get(final),
            "independent_of_experimental_offset":True}
        self.save()


def main(argv=None, *, make_hardware=None, input_fn=input, output=print):
    parser = argparse.ArgumentParser(description="分阶段药瓶接触因素扩展（离线计划或显式实采）")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1]/"config.skill.extension.yaml")
    p.add_argument("--stage", choices=("height_controlled", "force_controlled", "small_combinations"), required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--seed", type=int, default=20261105)
    p.add_argument("--bottle", default="bottle")
    p.add_argument("--pose-file", type=Path)
    p = sub.add_parser("collect")
    p.add_argument("--design-file", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--real", action="store_true")
    p.add_argument("--step", action="store_true")
    p.add_argument("--limit", type=int)
    p.add_argument("--min-normalized", type=float)
    p.add_argument("--fit", type=Path, help="Frozen measured-l ContactPolicy; absent means A probe collection")
    p.add_argument("--mode", choices=("deployment", "development"), default="deployment")
    p.add_argument("--role", choices=("development", "independent"), default="development")
    p.add_argument("--certificate", type=Path)
    p = sub.add_parser("train")
    p.add_argument("--session", type=Path, action="append", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=400)
    args = parser.parse_args(argv)
    try:
        if args.command == "train":
            from .skill_height_learning import train_height_policy
            if args.out.exists():
                raise ValueError("model output exists; use a new version path")
            model = train_height_policy(args.session, {"contact_policy":{"epochs":args.epochs}}, out=args.out)
            output(json.dumps({"model_id":model["model_id"], "usable_for_B":model["usable_for_B"],
                               "blocking_reasons":model["blocking_reasons"]}, ensure_ascii=False))
            return 0 if model["usable_for_B"] else 2
        if args.command == "plan":
            _, exploration, cfg, base = load_extension_config(args.config)
            design = make_probe_design(exploration, cfg, args.stage, repeats=args.repeats, seed=args.seed, bottle=args.bottle)
            if args.out.exists():
                raise ValueError("plan output already exists; use a new filename")
            atomic_json(args.out, design)
            T0 = None
            if args.pose_file:
                value = json.loads(args.pose_file.read_text())
                T0 = value.get("T0", value.get("T_base_tcp_mm")) if isinstance(value, dict) else value
            atomic_json(args.out.with_suffix(".preview.json"), plan_report(design, cfg, T0))
            output(f"planned_unexecuted: {len(design['plan'])} PROBE trials; {args.out.resolve()}")
            output("No hardware connected. l is calibrated local contact height, not automatically world z.")
            return 0
        if not args.real:
            parser.error("collect requires --real; use plan while robot is off")
        design = validate_probe_design(json.loads(args.design_file.read_text()))
        # Preserve the immutable base execution config alongside this execution,
        # never update a shared operator config file.
        args.out.mkdir(parents=True, exist_ok=True)
        cfg_path = args.out.parent / (args.out.name + ".execution_config.yaml")
        if cfg_path.exists() and yaml.safe_load(cfg_path.read_text()) != design["execution_config"]:
            raise ValueError("execution config differs from immutable design")
        cfg_path.write_text(yaml.safe_dump(design["execution_config"], allow_unicode=True))
        command = ["collect", "--real", "--config", str(cfg_path), "--design-file", str(args.design_file),
                   "--out", str(args.out)]
        if args.step: command.append("--step")
        if args.limit is not None: command += ["--limit", str(args.limit)]
        if args.min_normalized is not None: command += ["--min-normalized", str(args.min_normalized)]
        factory = ExtensionProbeRunner
        if args.fit:
            fit = json.loads(args.fit.read_text())
            if fit.get("simulated"):
                raise ValueError("simulated l model cannot be used by collect --real")
            certificate = json.loads(args.certificate.read_text()) if args.certificate else None
            def factory(*pos, **kw):
                return ExtensionRecoveryRunner(*pos, **kw, fit=fit, mode=args.mode,
                                               evaluation_role=args.role, certificate=certificate)
        elif args.role != "development":
            raise ValueError("independent B evaluation requires a frozen --fit")
        return probe_main(command, make_hardware=make_hardware, input_fn=input_fn,
                          output=output, runner_factory=factory)
    except Exception as exc:
        output("skill extension stopped: " + str(exc))
        return 2

if __name__ == "__main__":
    raise SystemExit(main())
