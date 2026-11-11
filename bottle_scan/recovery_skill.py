"""Complete, observation-only recovery skill; importing never opens hardware.

Primary policy must reach a safe pre-grasp pose first. The adapter performs the
first light touch, then this skill owns bounded regrasp/reobserve operations.
Completion means stable bilateral interior contact at the calibrated contact
target, NOT a successful lift. The primary policy owns squeeze/lift/transport.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
from typing import Protocol

from .agentic_learning import AgenticCorrection
from .contact_policy import ContactPolicy

SCHEMA = "bottle_recovery_skill_v1"
DEFAULT_CONFIG = {
    "target": {
        "meaning": "stable bilateral interior contact at calibrated reference",
        "min_stable_duration_s": 0.2,
        "confirmation_windows": 2,
    },
    "max_regrasps": 3,
    "max_total_action_mm": 25.0,
    "max_observation_age_s": 2.0,
    "tracking_tolerance_mm": 2.0,
    "required_context": {},
    "action_axis": "a",
    "action_unit": "mm",
    "execution_interface": "open_retreat_translate_reapproach_probe",
}


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                    separators=(",", ":")).encode()).hexdigest()


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


FORBIDDEN_CONTEXT_KEYS = {
    "delta_mm", "planned_offset_mm", "offset_mm", "ground_truth", "actual_error",
    "target_action_mm", "x_mm", "measured_probe_displacement_mm", "label",
}


def _valid_context(context):
    return isinstance(context, dict) and not (set(context) & FORBIDDEN_CONTEXT_KEYS)


def _settings(config, servo):
    cfg = deepcopy(DEFAULT_CONFIG)
    for key, value in (config or {}).items():
        if key not in cfg:
            raise ValueError("unknown recovery skill setting: " + key)
        if key == "target":
            cfg[key].update(value)
        else:
            cfg[key] = deepcopy(value)
    for key in ("max_total_action_mm", "max_observation_age_s", "tracking_tolerance_mm"):
        if not _number(cfg[key]) or cfg[key] <= 0:
            raise ValueError("invalid " + key)
    if type(cfg["max_regrasps"]) is not int or not 0 <= cfg["max_regrasps"] <= min(3, servo["max_regrasps"]):
        raise ValueError("skill must not increase existing regrasp limit")
    if cfg["max_total_action_mm"] > 25:
        raise ValueError("skill cumulative action budget exceeds existing envelope")
    target = cfg["target"]
    if (type(target["confirmation_windows"]) is not int or target["confirmation_windows"] < 2 or
            not _number(target["min_stable_duration_s"]) or target["min_stable_duration_s"] < .2):
        raise ValueError("target needs two independent stable observation windows")
    if not _valid_context(cfg["required_context"]):
        raise ValueError("required_context must map observable names; experimental truth is prohibited")
    if cfg["action_axis"] not in ("a", "l") or cfg["action_unit"] != "mm":
        raise ValueError("this learner controls only explicitly calibrated a/l translation")
    if cfg["execution_interface"] != DEFAULT_CONFIG["execution_interface"]:
        raise ValueError("unsupported action execution interface")
    return cfg


class RecoveryAdapter(Protocol):
    """Adapter must retain original force, width, motion and command-owner guards.

    execute() returns {'applied': bool, 'action_mm': measured applied motion,
    'completed_ns': monotonic completion time}. observe() returns a new stable
    window timestamped on the SAME monotonic clock, after the completed action.
    handoff() must treat incomplete as failure/replan, never as permission to lift.
    """
    def initial_probe(self) -> dict: ...
    def execute(self, action: dict) -> dict: ...
    def observe(self) -> dict: ...
    def handoff(self, result: dict) -> None: ...


class RecoverySkill:
    """Frozen model + invocation + explicit terminal and ownership semantics.

    'development' is an explicit experimental mode. It can exercise a candidate
    inside its observed domain, but verified remains false until a B certificate
    exists. It must never be substituted for deployment without caller choice.
    """
    def __init__(self, model, servo, config=None, certificate=None, *, mode="deployment"):
        if mode not in ("deployment", "development"):
            raise ValueError("mode must be deployment or development")
        self.model = deepcopy(model)
        self.servo = deepcopy(servo)
        self.config = _settings(config, servo)
        axis = self.config["action_axis"]
        if model.get("recovery_action_axis", "a") != axis:
            raise ValueError("model physical action axis differs from configured executor")
        if axis == "l" and (model.get("recovery_label_source") != "measured_tcp_projection_l" or
                            "a" not in model.get("recovery_controlled_axes", [])):
            raise ValueError("l recovery requires measured l supervision with a controlled")
        self.config_sha256 = _hash({"config": self.config, "servo": self.servo})
        self.model_id = model.get("model_id") or _hash(model)
        self.model_sha256 = _hash(model)
        self.mode = mode
        self.certificate = deepcopy(certificate or {})
        self.learned_state = None
        if model.get("schema") in ("learned_a_skill_v1", "learned_a_skill_set_v1"):
            if axis != "a":
                raise ValueError("joint learned skill currently supports a only")
            from .learned_skill import LearnedSkillModel
            self.learned_state = LearnedSkillModel(model)
            self.policy = ContactPolicy(model["recovery_model"], servo)
        elif model.get("schema") == "contact_policy_v1":
            self.policy = ContactPolicy(model, servo)
        elif model.get("schema") == "agentic_fit_v1":
            self.policy = AgenticCorrection(model, servo)
        else:
            raise ValueError("unsupported skill recovery model")
        self.reset()

    def reset(self):
        self.active = False
        self.pending_action = None
        self.last_timestamp_ns = None
        self.confirmations = 0
        self.last_target_timestamp_ns = None
        self.regrasps = 0
        self.total_action_mm = 0.
        self.events = []
        self.final_result = None

    def artifact(self):
        return {
            "schema": SCHEMA, "model_id": self.model_id,
            "model_sha256": self.model_sha256, "config_sha256": self.config_sha256,
            "config": deepcopy(self.config), "mode": self.mode,
            "invocation_output": {
                "needs_recovery": "true/false/null from actual contact and tactile policy",
                "applicable": "observable domain AND matching B-validated certificate",
                "invoke": "needs_recovery and permission within this runtime mode",
            },
            "recovery_output": {
                "model_schema": self.model["schema"], "action_axis": self.config["action_axis"], "unit": "mm",
                "execution": self.config["execution_interface"],
                "must_reobserve": True,
            },
            "exit_output": {
                "success": "fresh stable bilateral non-edge target contact across confirmation windows",
                "incomplete": "invalid/unvalidated context, stale sensing, safety, rejected action or exhausted budget",
                "handoff": "success -> primary squeeze/lift; incomplete -> primary replan/operator",
            },
            "ownership": [
                "primary positions safely before first light touch",
                "adapter.initial_probe performs first light touch",
                "skill evaluates invocation from this observation",
                "skill owns bounded recovery and fresh observations",
                "handoff includes completed flag; only success permits task continuation",
            ],
            "certificate": deepcopy(self.certificate),
        }

    def save_bundle(self, directory):
        """Export an immutable self-contained bundle without overwriting a version."""
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=True)
        model_path, manifest_path = root / "model.json", root / "skill.json"
        if model_path.exists() or manifest_path.exists():
            raise FileExistsError("skill bundle already exists; choose a new version directory")
        manifest = {**self.artifact(), "model_file": "model.json",
                    "servo": deepcopy(self.servo)}
        # Manifest is written last; partial exports cannot be loaded as a bundle.
        with model_path.open("x") as stream:
            json.dump(self.model, stream, indent=2, allow_nan=False)
            stream.write("\n")
        with manifest_path.open("x") as stream:
            json.dump(manifest, stream, indent=2, allow_nan=False)
            stream.write("\n")
        return str(manifest_path)

    def _certificate_match(self, observation):
        c = self.certificate
        if (c.get("model_id") != self.model_id or c.get("model_sha256") != self.model_sha256 or
                c.get("config_sha256") != self.config_sha256 or c.get("schema") != "skill_validation_certificate_v1"):
            return False, [], "missing_or_mismatched_validation_certificate"
        context = observation.get("context") or {}
        for cell in c.get("observed_cells", []):
            if cell.get("status") != observation.get("status"):
                continue
            if not all(cell.get(key) is True for key in
                       ("invocation_validated", "recovery_validated", "exit_validated")):
                continue
            refs = cell.get("evidence_refs")
            if not isinstance(refs, list) or not refs or not all(isinstance(r, str) and r for r in refs):
                continue
            if not _valid_context(cell.get("context")) or any(context.get(k) != v for k, v in cell["context"].items()):
                continue
            # No wildcard context/interval inferred from nominal perturbations.
            if any(cell["context"].get(k) != v for k, v in self.config["required_context"].items()):
                continue
            desc = observation.get("descriptors") or []
            bounds = cell.get("centroid_bounds_px")
            if observation.get("status") == "contact":
                try:
                    if len(bounds) != 2 or len(desc) != 2:
                        continue
                    inside = all(
                        len(bounds[i]) == 2 and all(
                            len(bounds[i][j]) == 2 and
                            all(_number(x) for x in bounds[i][j]) and
                            bounds[i][j][0] <= float(desc[i]["centroid_px"][j]) <= bounds[i][j][1]
                            for j in range(2)) for i in range(2))
                except (KeyError, IndexError, TypeError, ValueError):
                    inside = False
                if not inside:
                    continue
            return True, refs, "validated_observation_cell"
        return False, [], "outside_validated_observation_cells"

    def evaluate(self, observation):
        """Stateless three-output assessment. Never executes or declares completion."""
        o = observation
        context = o.get("context") or {}
        reasons = []
        if o.get("first_touch_complete") is not True:
            reasons.append("first_light_touch_required")
        if o.get("safety_ok") is not True:
            reasons.append("safety_state_unavailable_or_unsafe")
        if type(o.get("timestamp_ns")) is not int or o["timestamp_ns"] < 0:
            reasons.append("missing_monotonic_observation_timestamp")
        now = o.get("now_ns")
        if now is not None and (type(now) is not int or type(o.get("timestamp_ns")) is not int or
                not 0 <= now - o["timestamp_ns"] <= self.config["max_observation_age_s"] * 1e9):
            reasons.append("stale_observation")
        for key, value in self.config["required_context"].items():
            if context.get(key) != value:
                reasons.append("context_mismatch:" + key)
        descriptors = o.get("descriptors") or []
        if self.learned_state is not None:
            return self._evaluate_learned(o, reasons)
        try:
            decision = self.policy.decide(o.get("status"), descriptors, o.get("bilateral") is True)
        except (TypeError, ValueError, KeyError, IndexError):
            decision = {"accepted": False, "rejected": True, "delta_mm": 0.,
                        "reason": "invalid_observation"}
        if self.config["action_axis"] == "l" and (
                o.get("status") in ("no_contact", "stall_no_contact") or
                any(isinstance(d, dict) and d.get("touches_edge") for d in descriptors)):
            decision = {"accepted": False, "rejected": True, "delta_mm": 0.,
                        "reason": "l_directional_fallback_not_validated"}
        valid_contact = (o.get("status") == "contact" and o.get("bilateral") is True and
                         len(descriptors) == 2 and all(
                             isinstance(d, dict) and d.get("valid") is True and
                             not d.get("touches_edge", False) for d in descriptors))
        stable = (o.get("force_stable") is True and _number(o.get("stable_duration_s")) and
                  o["stable_duration_s"] >= self.config["target"]["min_stable_duration_s"])
        target = valid_contact and stable and decision.get("accepted") is True
        if decision.get("accepted") is True and not valid_contact:
            decision = {"accepted": False, "rejected": True, "delta_mm": 0.,
                        "reason": "invalid_target_contact"}
        if decision.get("rejected") or decision.get("reason") == "unobservable_or_outside_model_domain":
            needs = None
        elif o.get("status") in ("no_contact", "stall_no_contact") or (
                _number(decision.get("e_mm")) and not decision.get("accepted")) or decision.get("edges"):
            needs = True
        elif decision.get("accepted") is True and valid_contact:
            needs = False
        else:
            needs = None
        supported = not reasons and not decision.get("rejected") and needs is not None
        verified, refs, validation_reason = self._certificate_match(o)
        applicable = supported and verified
        permitted = supported and (verified or self.mode == "development")
        invoke = needs is True and permitted
        delta = decision.get("delta_mm", 0.)
        if not _number(delta) or abs(delta) > self.servo["max_delta_mm"]:
            permitted = invoke = False
            reasons.append("invalid_or_oversized_action")
        action = None
        if invoke and abs(delta) > 0:
            action = {"type": "regrasp", "axis": self.config["action_axis"], "delta_mm": float(delta),
                      "unit": "mm", "interface": self.config["execution_interface"],
                      "reobserve_required": True}
        if needs is True and permitted and action is None:
            permitted = invoke = False
            reasons.append("no_effective_recovery_action")
        return {
            "schema": SCHEMA, "model_id": self.model_id, "mode": self.mode,
            "invocation": {
                "needs_recovery": needs, "applicable": applicable, "verified": verified,
                "candidate_supported": supported, "permitted": permitted, "invoke": invoke,
                "validation_reason": validation_reason, "evidence_refs": refs,
                "reasons": reasons,
            },
            "recovery": {"action": action, "decision": decision, "reobserve_required": action is not None},
            "exit": {"status": "target_candidate" if target else "not_completed",
                     "target_observed": target, "completed": False, "hand_back": False,
                     "target": deepcopy(self.config["target"]),
                     "reason": "confirmation_required" if target else "target_not_established"},
        }

    def _evaluate_learned(self, observation, reasons):
        """Learned semantics inside the unchanged physical/ownership guards.

        A no-action normal state does not require an action-regression estimate.
        An action always requires supported recovery prediction and real feedback.
        """
        o = observation
        reasons = list(reasons)
        if (o.get("context") or {}).get("target_definition_id") != self.model.get("target_definition_id"):
            reasons.append("context_mismatch:target_definition_id")
        descriptors = o.get("descriptors") or []
        valid_contact = (o.get("status") == "contact" and o.get("bilateral") is True and
                         len(descriptors) == 2 and all(isinstance(d, dict) and
                         d.get("valid") is True and not d.get("touches_edge", False)
                         for d in descriptors))
        stable = (o.get("force_stable") is True and _number(o.get("stable_duration_s")) and
                  o["stable_duration_s"] >= self.config["target"]["min_stable_duration_s"])
        try:
            call = self.learned_state.invocation(o)
            exit_prediction = self.learned_state.exit(o)
        except (ValueError, TypeError, KeyError, IndexError):
            call = exit_prediction = {"valid": False, "positive": None,
                                      "reason": "invalid_observation", "probability": None}
        needs = call.get("positive") if call.get("valid") is True else None
        completed_state = exit_prediction.get("positive") if exit_prediction.get("valid") is True else None
        decision = {"accepted": False, "rejected": False, "delta_mm": 0.,
                    "reason": "learned_contact_state", "model_id": self.model_id}
        if not valid_contact:
            decision.update(rejected=True, reason="learned_policy_requires_bilateral_interior_contact")
        elif type(needs) is not bool:
            decision.update(rejected=True, reason=str(call.get("reason") or "invocation_unknown"))
        elif type(completed_state) is not bool:
            decision.update(rejected=True, reason=str(exit_prediction.get("reason") or "exit_unknown"))
        elif needs == completed_state:
            # Independent heads disagree about the meaning of the same state.
            # Neither a movement nor a successful handoff is justified.
            decision.update(rejected=True, reason="learned_state_disagreement")
        elif needs:
            try:
                prediction = self.learned_state.recovery_prediction(self.policy.predict(descriptors))
            except (ValueError, TypeError, KeyError, IndexError):
                prediction = {"valid": False, "reason": "invalid_recovery_observation"}
            if prediction.get("valid") is not True:
                decision.update(prediction, rejected=True, accepted=False, delta_mm=0.)
            else:
                amount = prediction.get("action_mm")
                limit = min(self.servo["max_delta_mm"],
                            self.model["recovery_model"]["settings"]["max_action_mm"])
                if not _number(amount):
                    decision.update(rejected=True, reason="invalid_or_oversized_action")
                else:
                    delta = max(-limit, min(limit, self.policy.gain * amount))
                    decision.update(prediction, delta_mm=float(delta), e_mm=-amount,
                                    reason="learned_contact_recovery", accepted=False)
        else:
            decision.update(accepted=True, reason="learned_target_contact")
        supported = not reasons and not decision.get("rejected") and type(needs) is bool
        verified, refs, validation_reason = self._certificate_match(o)
        permitted = supported and (verified or self.mode == "development")
        invoke = needs is True and permitted
        delta = decision.get("delta_mm", 0.)
        if not _number(delta) or abs(delta) > self.servo["max_delta_mm"]:
            permitted = invoke = False
            reasons.append("invalid_or_oversized_action")
        action = None
        if invoke and abs(delta) > 0:
            action = {"type": "regrasp", "axis": "a", "delta_mm": float(delta),
                      "unit": "mm", "interface": self.config["execution_interface"],
                      "reobserve_required": True}
        if needs is True and permitted and action is None:
            permitted = invoke = False
            reasons.append("no_effective_recovery_action")
        target = (not decision.get("rejected") and valid_contact and stable and
                  needs is False and completed_state is True)
        return {"schema": SCHEMA, "model_id": self.model_id, "mode": self.mode,
                "learned_conditions": {"invocation": call, "exit": exit_prediction},
                "invocation": {"needs_recovery": needs, "applicable": supported and verified,
                    "verified": verified, "candidate_supported": supported, "permitted": permitted,
                    "invoke": invoke, "validation_reason": validation_reason,
                    "evidence_refs": refs, "reasons": reasons},
                "recovery": {"action": action, "decision": decision,
                             "reobserve_required": action is not None},
                "exit": {"status": "target_candidate" if target else "not_completed",
                         "target_observed": target, "completed": False, "hand_back": False,
                         "target": deepcopy(self.config["target"]),
                         "reason": "confirmation_required" if target else "target_not_established",
                         "achieved_state": ({"target_definition_id":self.model.get("target_definition_id"),
                            "timestamp_ns":o.get("timestamp_ns"),
                            "contact_features":[v for d in descriptors for v in d.get("centroid_px",[])],
                            **deepcopy(exit_prediction.get("acceptable_set_match") or {})} if target else None)}}

    def _terminal(self, result, status, reason):
        self.active = False
        self.pending_action = None
        result["recovery"]["action"] = None
        result["recovery"]["reobserve_required"] = False
        result["exit"].update(status=status, completed=status == "success",
                              hand_back=True, reason=reason,
                              next_owner="primary_policy",
                              continuation="squeeze_lift_allowed" if status == "success" else "replan_or_operator")
        if status != "success":
            result["exit"].pop("achieved_state", None)
        self.final_result = deepcopy(result)
        return result

    def _process(self, observation):
        result = self.evaluate(observation)
        ts = observation.get("timestamp_ns")
        if self.last_timestamp_ns is not None and (type(ts) is not int or ts <= self.last_timestamp_ns):
            result = self._terminal(result, "incomplete", "new_observation_required")
        elif not result["invocation"]["permitted"]:
            decision = result["recovery"]["decision"]
            result = self._terminal(result, "incomplete",
                                    ";".join(result["invocation"]["reasons"]) or
                                    (decision.get("reason") if decision.get("rejected") else None) or
                                    result["invocation"]["validation_reason"])
        else:
            self.last_timestamp_ns = ts
            if result["exit"]["target_observed"]:
                if (self.last_target_timestamp_ns is None or
                        ts - self.last_target_timestamp_ns >= self.config["target"]["min_stable_duration_s"] * 1e9):
                    self.confirmations += 1
                    self.last_target_timestamp_ns = ts
                if self.confirmations >= self.config["target"]["confirmation_windows"]:
                    result = self._terminal(result, "success", "stable_target_contact_confirmed")
                else:
                    result["exit"].update(status="observing", reason="confirm_target_with_fresh_window")
            else:
                self.confirmations = 0
                self.last_target_timestamp_ns = None
                action = result["recovery"]["action"]
                if action:
                    if (self.regrasps >= self.config["max_regrasps"] or
                            self.total_action_mm + abs(action["delta_mm"]) > self.config["max_total_action_mm"]):
                        result = self._terminal(result, "incomplete", "recovery_budget_exhausted")
                    else:
                        self.pending_action = deepcopy(action)
                        result["exit"].update(status="recovering", reason="execute_then_reobserve")
                else:
                    # Geometric target without stable force: bounded observation,
                    # never falsely complete or apply an arbitrary movement.
                    result["exit"].update(status="observing", reason="await_stable_contact")
        self.events.append({"event": "observation", "timestamp_ns": ts,
                            "result": deepcopy(result)})
        return result

    def start(self, observation):
        if self.active:
            raise RuntimeError("skill already owns control")
        self.reset()
        self.active = True
        return self._process(observation)

    def advance(self, observation, *, applied_action=None):
        if not self.active:
            raise RuntimeError("skill does not own control; call start")
        if self.pending_action is not None:
            requested = self.pending_action
            receipt = applied_action or {}
            ts = observation.get("timestamp_ns")
            completed = receipt.get("completed_ns")
            applied = receipt.get("action_mm")
            valid = (receipt.get("applied") is True and _number(applied) and
                     abs(applied - requested["delta_mm"]) <= self.config["tracking_tolerance_mm"] and
                     type(completed) is int and type(ts) is int and
                     self.last_timestamp_ns <= completed < ts)
            self.events.append({"event": "execution", "requested": requested,
                                "receipt": deepcopy(receipt), "verified": valid})
            if not valid:
                result = self._terminal(self.evaluate(observation), "incomplete", "unverified_execution_or_reobservation")
                self.events.append({"event": "termination", "result": deepcopy(result)})
                return result
            self.regrasps += 1
            self.total_action_mm += abs(applied)
            self.pending_action = None
        elif applied_action is not None:
            raise ValueError("unsolicited action receipt")
        return self._process(observation)

    def abort(self, reason):
        """Explicit incomplete handoff for external deadline, stop or adapter failure."""
        if not isinstance(reason, str) or not reason:
            raise ValueError("abort reason required")
        result = deepcopy(self.events[-1].get("result")) if self.events and self.events[-1].get("result") else self.evaluate({})
        result = self._terminal(result, "incomplete", reason)
        self.events.append({"event": "termination", "result": deepcopy(result)})
        return result

    def run(self, adapter: RecoveryAdapter, *, max_observations=12):
        """Explicit adapter invocation; no default hardware connection or motion."""
        if type(max_observations) is not int or max_observations < 2:
            raise ValueError("at least two observation windows required")
        result = self.evaluate({})
        try:
            result = self.start(adapter.initial_probe())
            for _ in range(max_observations - 1):
                if result["exit"]["hand_back"]:
                    break
                action = result["recovery"]["action"]
                receipt = adapter.execute(deepcopy(action)) if action else None
                result = self.advance(adapter.observe(), applied_action=receipt)
            if not result["exit"]["hand_back"]:
                result = self._terminal(result, "incomplete", "observation_budget_exhausted")
        except BaseException as exc:
            result = self._terminal(result, "incomplete", "adapter_failure:" + type(exc).__name__)
            self.events.append({"event": "adapter_failure", "exception_type": type(exc).__name__})
            adapter.handoff(result)
            raise
        adapter.handoff(result)
        return result



def load_skill_bundle(path, *, mode="deployment"):
    """Load verified model/config references; deployment is always the default.

    This does not grant validation: absent or mismatched certificate still
    rejects invocation. Explicit development mode is required to run experiments.
    """
    manifest_path = Path(path)
    if manifest_path.is_dir():
        manifest_path = manifest_path / "skill.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != SCHEMA or manifest.get("model_file") != "model.json":
        raise ValueError("not a self-contained recovery skill bundle")
    model = json.loads((manifest_path.parent / "model.json").read_text())
    if _hash(model) != manifest.get("model_sha256"):
        raise ValueError("bundle model hash mismatch")
    skill = RecoverySkill(model, manifest["servo"], manifest["config"],
                          manifest.get("certificate"), mode=mode)
    if (skill.config_sha256 != manifest.get("config_sha256") or
            skill.model_id != manifest.get("model_id")):
        raise ValueError("bundle config/model identity mismatch")
    return skill
