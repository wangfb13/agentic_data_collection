"""Evidence-driven skill experiments and actual-cost provenance.

Pure file/planning code: importing this module cannot connect to a robot.
Independent evaluation is write-only evidence for reports, never a planning input.
Historical experience can propose a hypothesis; MethodStore remains the only
authority for method comparison and promotion.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import random
import time
import fcntl

from .session import atomic_json, fingerprint

SCHEMA = "bottle_skill_exploration_v1"
MODES = ("fixed", "simple_feedback", "agent_evidence")
STAGES = ("a_closed_loop", "height_controlled", "force_controlled", "small_combinations")
CATEGORIES = ("A_exploration", "B_development", "B_independent", "failed_attempt",
              "method_applicability", "robot", "reset", "source_experience", "offline_compute")
INDEPENDENT_ROLES = {"independent", "B_independent", "final_validation", "frozen_validation", "validation"}


def _finite(value):
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value)


def _nonnegative(value, name, integer=False, nullable=False):
    if nullable and value is None:
        return value
    if not _finite(value) or value < 0 or (integer and type(value) is not int):
        raise ValueError(name + " must be finite and nonnegative")
    return value


def _refs(refs):
    if not isinstance(refs, list) or not refs or any(not isinstance(r, str) or not r for r in refs):
        raise ValueError("nonempty evidence_refs are required")
    return sorted(set(refs))


def default_exploration_config():
    """Amounts are explicit design defaults, NOT validated operating envelopes."""
    return {
        "schema": SCHEMA,
        "task_id": "bottle_contact_recovery",
        "variables": {
            "a_mm": {"meaning": "Translation along original T0 local finger axis a",
                     "unit": "mm", "executor": "trial.delta_mm", "range": [-10., 10.],
                     "levels": [-10., -5., 0., 5., 10.], "reference": 0.,
                     "runtime_source": "tactile recovery estimate; planned value is analysis-only"},
            "contact_height_l_mm": {"meaning": "Contact displacement along original T0 local l, perpendicular to a; not necessarily world z",
                     "unit": "mm", "executor": "trial.perturbation.l_mm", "range": [-4., 4.],
                     "levels": [-4., 0., 4.], "reference": 0.,
                     "runtime_source": "measured TCP projected on calibrated local l"},
            "probe_force": {"meaning": "Per-finger probe target in existing force interface units",
                     "unit": "sdk", "executor": "config.force.probe_fraction = probe_force / config.force.target", "range": [0., 100.],
                     "levels": [], "reference": None,
                     "runtime_source": "measured per-finger normal force and commanded probe target"}
        },
        "a_reference_mm": 0.,
        "reps_per_condition": 2,
        "max_round_trials": 12,
        "max_combinations": 4,
        "history_enabled": True,
        "history_limit": 5,
        "minimum_development_trials": 4,
        "reference_every": 4,
        "learner": "existing_contact_policy",
        "action_space": {"a_mm": [-3., 3.], "contact_height_l_mm": [-4., 4.],
                         "regrasp": "open_retract_translate_recontact"},
        "extension_status": "awaits_real_data_and_applicability_checks"
    }


def validate_exploration_config(config):
    value = deepcopy(config)
    if value.get("schema", SCHEMA) != SCHEMA or not isinstance(value.get("task_id"), str):
        raise ValueError("invalid exploration schema/task")
    variables = value.get("variables")
    if not isinstance(variables, dict) or not {"a_mm", "contact_height_l_mm", "probe_force"} <= set(variables):
        raise ValueError("a, independent local-l displacement, and probe force must be configured")
    for name, variable in variables.items():
        for key in ("meaning", "unit", "executor", "runtime_source"):
            if not isinstance(variable.get(key), str) or not variable[key]:
                raise ValueError(name + " requires physical meaning/unit/executor/runtime_source")
        bounds = variable.get("range")
        if not isinstance(bounds, list) or len(bounds) != 2 or not all(_finite(x) for x in bounds) or bounds[0] >= bounds[1]:
            raise ValueError(name + " requires a finite increasing range")
        levels = variable.get("levels", [])
        if not isinstance(levels, list) or len(set(levels)) != len(levels) or any(not _finite(x) or not bounds[0] <= x <= bounds[1] for x in levels):
            raise ValueError(name + " has invalid levels")
        reference = variable.get("reference")
        if reference is not None and (not _finite(reference) or not bounds[0] <= reference <= bounds[1]):
            raise ValueError(name + " reference is outside configured range")
    for name in ("reps_per_condition", "max_round_trials", "max_combinations", "minimum_development_trials", "reference_every", "history_limit"):
        if type(value.get(name)) is not int or value[name] < 1:
            raise ValueError(name + " must be a positive integer")
    if value["reps_per_condition"] % 2:
        raise ValueError("reps_per_condition must balance light/heavy bottle blocks")
    if type(value.get("history_enabled")) is not bool:
        raise ValueError("history_enabled must be explicit")
    return value


def _development(report):
    if not isinstance(report, dict):
        raise ValueError("feedback must be a report")
    # A combined report is accepted ONLY by explicitly extracting development.
    report = report.get("development", report)
    if report.get("role") in INDEPENDENT_ROLES:
        raise ValueError("independent evaluation must never guide collection or skill updates")
    if report.get("role") not in ("development", "B_development", "offline_development"):
        raise ValueError("feedback needs explicit development role")
    return report


def diagnose_feedback(report):
    """Identify missing evidence separately for invocation, action and exit.

    Proxy geometry is useful for designing experiments, never proof that a
    physical contact target or a new recovery action was actually achieved.
    """
    report = _development(report)
    refs = _refs(report.get("evidence_refs", []))
    findings = []
    def add(component, problem, experiment, reason):
        item = {"component": component, "problem": problem, "experiment": experiment,
                "reason": reason, "evidence_refs": refs}
        if item not in findings:
            findings.append(item)
    call = report.get("invocation", report.get("call", {}))
    recover = report.get("recovery", {})
    exit_report = report.get("exit", {})
    coverage = report.get("coverage", {})
    details = report.get("details", [])
    missing_initial = coverage.get("initial_unlabelled", 0)
    missing_final = coverage.get("final_unlabelled", 0)
    if details:
        missing_initial = max(missing_initial, sum(
            d.get("initial_class") not in ("already_normal", "needs_recovery", "not_applicable")
            or d.get("label_source_verified") is not True for d in details))
        missing_final = max(missing_final, sum(
            type(d.get("final_target_reached")) is not bool
            or d.get("label_source_verified") is not True for d in details))
    # `unknown` conflates missing truth and an abstaining prediction in older
    # reports. It must not be presented as proof that an operator omitted labels.
    if missing_initial or not call.get("n") or call.get("proxy_only") or "proxy" in call.get("label_definition", ""):
        add("invocation", "Missing independent labels for need/applicability",
            "control_variable", "Collect normal, recoverable and temporarily inapplicable contacts with independent contact-state labels")
    if call.get("unknown", 0):
        add("invocation", "Invocation prediction or evidence is unknown",
            "fill_gap", "Separate model abstention from missing truth; inspect recorded policy reasons before selecting new contacts")
    rejection = report.get("rejection_diagnostics", {}).get("terminal_reasons", {})
    if rejection.get("policy_uncertain", 0):
        add("recovery", "Recovery prediction uncertainty exceeds its frozen gate", "fill_gap",
            "Compare model/features offline and repeat uncertain contact conditions; do not relax the uncertainty gate to obtain motion")
    if rejection.get("policy_out_of_distribution", 0):
        add("applicability", "Observed tactile features are outside training support", "control_variable",
            "Check reference and feature reliability, then collect supported a-axis contacts at fixed height and probe force")
    if call.get("FP", call.get("false_positive", 0)):
        add("invocation", "Unnecessary calls on normal contact", "repeat_reference",
            "Repeat verified normal conditions to separate reference drift from a call threshold error")
    if call.get("FN", call.get("false_negative", 0)):
        add("invocation", "Missed recovery calls", "fill_gap",
            "Add boundary and missed-call conditions with matched probe settings")
    if recover.get("rejected", 0) or recover.get("unknown", 0) or not recover.get("n"):
        add("recovery", "Missing supported action evidence", "fill_gap",
            "Collect pre/action/post trajectories in rejected or unsupported conditions")
    if recover.get("failures", 0) or recover.get("p90_mm", 0) > report.get("target_p90_mm", 2.):
        add("recovery", "Recovery action remains inaccurate", "control_variable",
            "Hold height/force fixed, repeat a-axis conditions; distinguish action error from changing contact conditions")
    if exit_report.get("false_success", 0):
        add("exit", "Premature successful exit", "repeat_reference",
            "Repeat target and near-target post-action contacts with independent completion labels")
    if exit_report.get("missed_completion", 0):
        add("exit", "Completion not recognized", "fill_gap",
            "Collect normal terminal contacts and immediate post-recovery windows")
    if missing_final or exit_report.get("proxy_only") or not exit_report.get("n"):
        add("exit", "No independently observed target-state completion", "control_variable",
            "Record bilateral stable target contact after action; missing labels remain unknown")
    if exit_report.get("unknown", 0) or exit_report.get("unknown_no_terminal_handoff", 0):
        add("exit", "Completion prediction or terminal handoff is unknown", "fill_gap",
            "Inspect completion confidence and recorded handoff separately from independently observed target-state labels")
    if report.get("reference_drift"):
        add("applicability", "Reference condition shifted", "repeat_reference",
            "Re-run a zero condition before interpreting between-condition changes")
    for gap in report.get("gaps", []):
        text = gap if isinstance(gap, str) else json.dumps(gap, ensure_ascii=False, sort_keys=True)
        add("evidence", text, "fill_gap", "Resolve documented missing evidence before expanding a claimed operating range")
    if not findings:
        add("applicability", "Observed development metrics alone do not certify a larger range", "small_combination",
            "Check a small number of controlled combinations; keep independent evaluation sealed")
    return findings


def _condition(config, **changes):
    values = {name: variable.get("reference") for name, variable in config["variables"].items()}
    values.update(changes)
    return values


def extension_design(config, stage, *, seed=0):
    """Unexecuted stage-wise plans: fixed controls, references, few combinations.

    l and a orthogonality and reachable endpoints must also be verified using
    actual axes/T0 by the hardware adapter. These plans do not certify motion.
    """
    cfg = validate_exploration_config(config)
    if stage not in STAGES:
        raise ValueError("unknown extension stage")
    force = cfg["variables"]["probe_force"]
    if force["reference"] is None:
        raise ValueError("configure probe_force reference from calibrated existing config before execution")
    a_levels = cfg["variables"]["a_mm"]["levels"]
    h_levels = cfg["variables"]["contact_height_l_mm"]["levels"]
    f_levels = force["levels"]
    if stage == "a_closed_loop":
        conditions = [_condition(cfg, a_mm=a) for a in a_levels]
    elif stage == "height_controlled":
        conditions = [_condition(cfg, contact_height_l_mm=h) for h in h_levels]
    elif stage == "force_controlled":
        if len(f_levels) < 2:
            raise ValueError("force study needs at least two calibrated force levels")
        conditions = [_condition(cfg, probe_force=f) for f in f_levels]
    else:
        nonzero_a = [a for a in a_levels if a != cfg["variables"]["a_mm"]["reference"]]
        nonzero_h = [h for h in h_levels if h != cfg["variables"]["contact_height_l_mm"]["reference"]]
        if not nonzero_a or not nonzero_h:
            raise ValueError("combination study needs non-reference a and l levels")
        pairs = [(a, h) for a in (min(nonzero_a), max(nonzero_a)) for h in (min(nonzero_h), max(nonzero_h))]
        conditions = [_condition(cfg, a_mm=a, contact_height_l_mm=h) for a, h in pairs[:cfg["max_combinations"]]]
    reference = _condition(cfg)
    conditions = [x for x in conditions if x != reference]
    random.Random(seed).shuffle(conditions)
    rows = []
    # Reference conditions are counted within budgets, never free insertions.
    for i, condition in enumerate([reference] + conditions):
        if i > 0 and i % cfg["reference_every"] == 0:
            rows.append({"condition": deepcopy(reference), "purpose": "reference_repeat"})
        rows.append({"condition": condition, "purpose": "reference_repeat" if condition == reference else stage})
    return {"schema": "skill_extension_design_v1", "stage": stage, "seed": seed,
            "status": "planned_unexecuted", "rows": rows,
            "fixed_controls": {k: v for k, v in reference.items()
                               if k not in {"a_closed_loop": ["a_mm"], "height_controlled": ["contact_height_l_mm"],
                                            "force_controlled": ["probe_force"], "small_combinations": ["a_mm", "contact_height_l_mm"]}[stage]},
            "next_stage_requires": "actual development evidence and an independently validated applicable scope",
            "geometry_requirements": ["a and l are orthogonal in calibrated axes_tcp", "actual T0 endpoint bounds pass",
                                       "open and retract before any pose correction", "force limits remain unchanged"]}


def choose_experiments(report, config, *, mode="agent_evidence", seed=0, remaining_trials=None, stage="a_closed_loop"):
    """Deterministic evidence heuristic baseline; not an LLM-generated proposal.

    The Agent may use these suggestions or submit another validated plan.
    Fixed and simple_feedback modes share identical variable ranges and budgets.
    """
    cfg = validate_exploration_config(config)
    dev = _development(report)
    if mode not in MODES:
        raise ValueError("unknown collection strategy")
    budget = cfg["max_round_trials"] if remaining_trials is None else min(cfg["max_round_trials"], remaining_trials)
    _nonnegative(budget, "remaining_trials", integer=True)
    budget -= budget % cfg["reps_per_condition"]
    issues = diagnose_feedback(dev)
    design = extension_design(cfg, stage, seed=seed)
    candidates = deepcopy(design["rows"])
    cell_rows = dev.get("cells", [])
    def failure(condition):
        matching = [c for c in cell_rows if c.get("conditions", c.get("condition")) == condition]
        return sum(c.get("failures", 0) for c in matching) / max(1, sum(c.get("n", 0) for c in matching))
    if mode == "simple_feedback":
        candidates.sort(key=lambda x: -failure(x["condition"]))
    elif mode == "agent_evidence":
        needs_reference = any(i["experiment"] == "repeat_reference" for i in issues)
        def priority(row):
            reference = row["purpose"] == "reference_repeat"
            coverage = sum(c.get("n", 0) for c in cell_rows if c.get("conditions", c.get("condition")) == row["condition"])
            return (0 if reference and needs_reference else 1, -failure(row["condition"]), coverage)
        candidates.sort(key=priority)
    selected = []
    for row in candidates[:budget // cfg["reps_per_condition"]]:
        applicable = [x for x in issues if row["purpose"] != "reference_repeat" or x["experiment"] == "repeat_reference"] or issues
        selected.append({**row, "reps": cfg["reps_per_condition"], "bottle_blocks": ["light", "heavy"],
                         "condition_kind": "A" if mode == "fixed" else "A_or_B_requires_explicit_execution_choice",
                         "reason": applicable[0]["reason"], "problem_refs": [fingerprint(x) for x in applicable]})
    return {"schema": "skill_experiment_selection_v1", "strategy": mode,
            "selector": "deterministic_evidence_heuristic", "stage": stage,
            "findings": issues, "selected": selected, "planned_trial_attempts": sum(r["reps"] for r in selected),
            "budget_includes_references_and_failures": True, "role": "development",
            "evidence_refs": dev["evidence_refs"], "status": "planned_unexecuted"}


def file_evidence(paths):
    """Hash original evidence without copying raw records or changing their source."""
    result = []
    for item in paths:
        path = Path(item).resolve()
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        result.append({"path": str(path), "sha256": digest.hexdigest(), "bytes": path.stat().st_size})
    return result


class CostLedger:
    """Immutable actual execution events; unknown is distinct from measured zero.

    One physical attempt belongs to one event. Annotate that event's failures,
    robot/reset times instead of adding a second charged attempt. Historical
    source experience is reported separately from this experiment's new budget.
    Aggregated historical counts may omit attempt_ids but must cite evidence.
    """
    def __init__(self, root):
        self.root = Path(root)
        self.events = self.root / "cost_events"
        self.events.mkdir(parents=True, exist_ok=True)

    def record(self, event_id, category, *, trial_attempts=0, failed_attempts=0,
               wall_seconds=None, robot_seconds=None, reset_seconds=None,
               evidence_refs, actual=True, attempt_ids=None, metadata=None):
        if not isinstance(event_id, str) or not event_id or category not in CATEGORIES or type(actual) is not bool:
            raise ValueError("invalid event identity/category/actual flag")
        for name, value in (("trial_attempts", trial_attempts), ("failed_attempts", failed_attempts)):
            _nonnegative(value, name, integer=True)
        if failed_attempts > trial_attempts:
            raise ValueError("failures exceed attempts")
        for name, value in (("wall_seconds", wall_seconds), ("robot_seconds", robot_seconds), ("reset_seconds", reset_seconds)):
            _nonnegative(value, name, nullable=True)
        ids = list(attempt_ids or [])
        if len(set(ids)) != len(ids) or any(not isinstance(x, str) or not x for x in ids):
            raise ValueError("attempt identities must be unique strings")
        if ids and len(ids) != trial_attempts:
            raise ValueError("attempt_ids must enumerate all charged attempts")
        if not actual and trial_attempts:
            raise ValueError("planned or replay-only data cannot count as newly executed physical attempts")
        record = {"schema": "skill_actual_cost_v1", "event_id": event_id, "category": category,
                  "actual": actual, "trial_attempts": trial_attempts, "failed_attempts": failed_attempts,
                  "wall_seconds": wall_seconds, "robot_seconds": robot_seconds, "reset_seconds": reset_seconds,
                  "attempt_ids": ids, "evidence_refs": _refs(evidence_refs), "metadata": deepcopy(metadata or {})}
        identity = hashlib.sha256(event_id.encode()).hexdigest()
        with (self.root / ".cost.lock").open("a") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            path = self.events / (identity + ".json")
            if path.exists():
                if json.loads(path.read_text()) != record:
                    raise ValueError("cost event is immutable; duplicate event_id has changed content")
                return record
            seen = {x for p in self.events.glob("*.json") for x in json.loads(p.read_text()).get("attempt_ids", [])}
            if seen.intersection(ids):
                raise ValueError("physical attempt is already charged in another event")
            atomic_json(path, record)
        return record

    def summary(self):
        records = [json.loads(p.read_text()) for p in sorted(self.events.glob("*.json"))]
        categories = {}
        for category in CATEGORIES:
            rows = [r for r in records if r["category"] == category]
            costs = {}
            for key in ("wall_seconds", "robot_seconds", "reset_seconds"):
                costs[key] = {"known_total": sum(r[key] for r in rows if r[key] is not None),
                              "unknown_events": sum(r[key] is None for r in rows)}
            categories[category] = {"events": len(rows), "trial_attempts": sum(r["trial_attempts"] for r in rows),
                                    "failed_attempts": sum(r["failed_attempts"] for r in rows), **costs}
        new = sum(r["trial_attempts"] for r in records if r["actual"] and r["category"] != "source_experience")
        return {"schema": "skill_cost_summary_v1", "categories": categories, "new_physical_attempts": new,
                "source_experience_attempts": categories["source_experience"]["trial_attempts"],
                "interpretation": "Known totals exclude missing measurements. Unknown events are not zero. Offline replay selections are not new robot interactions."}


def record_round(root, round_id, *, feedback, selection, result, skill_update, evidence_paths=()):
    """Append an immutable problem -> experiment -> result -> skill update chain."""
    dev = _development(feedback)
    if result.get("role") in INDEPENDENT_ROLES or skill_update.get("uses_independent_evaluation"):
        raise ValueError("independent evaluation cannot update a skill in this experiment")
    if not isinstance(round_id, str) or not round_id or round_id in (".", "..") or "/" in round_id or "\\" in round_id:
        raise ValueError("invalid round_id")
    if not isinstance(skill_update, dict) or "status" not in skill_update:
        raise ValueError("skill update needs explicit status (including unchanged/pending)")
    record = {"schema": "skill_exploration_round_v1", "round_id": round_id,
              "problems": diagnose_feedback(dev), "experiment_selection": deepcopy(selection),
              "result": deepcopy(result), "skill_update": deepcopy(skill_update),
              "original_evidence": file_evidence(evidence_paths),
              "evidence_refs": dev["evidence_refs"]}
    path = Path(root) / "rounds" / (round_id + ".json")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and json.loads(path.read_text()) != record:
        raise ValueError("round evidence is immutable")
    atomic_json(path, record)
    return record


def memory_context(store, context, *, history_enabled=True, limit=5):
    """Memory-off leaves current-task evidence intact, suppresses prior retrieval."""
    if type(history_enabled) is not bool:
        raise ValueError("history_enabled must be explicit boolean")
    return {"history_enabled": history_enabled,
            "methods": store.retrieve(context, limit=limit) if history_enabled else [],
            "note": "Retrieved methods require current applicability evidence; retrieval never promotes a method."}


def persist_method_experience(store, method, *, task_id, context, round_record,
                              simulated, comparison=None, applicability_check=None):
    """Reuse immutable RSI methods, evidence verification and promotion rules.

    Offline replay evidence has no physical comparison and therefore stays
    pending. A supplied comparison is always recomputed by MethodStore.
    """
    method_id = store.register(method)
    if type(simulated) is not bool or context.get("simulated", simulated) != simulated:
        raise ValueError("explicit source simulation identity required")
    record = {"method_id": method_id, "task_id": task_id, "simulated": simulated,
              "context": deepcopy(context), "created_unix_ns": time.time_ns(),
              "skill_round": deepcopy(round_record),
              "applicability_check": deepcopy(applicability_check or {"status": "pending"}),
              "notes": "Skill exploration evidence; method status is derived only by existing independent comparison evaluator."}
    if comparison is not None:
        if round_record.get("result", {}).get("offline_replay"):
            raise ValueError("offline replay cannot be promoted as a real method comparison")
        record["comparison"] = comparison
    return {"method_id": method_id, "experience_id": store.record_experience(record)}


def comparison_protocol(config, *, interaction_budget, source_evidence_refs):
    cfg = validate_exploration_config(config)
    _nonnegative(interaction_budget, "interaction_budget", integer=True)
    contract = {"schema": "skill_fair_comparison_v1", "task_id": cfg["task_id"], "modes": list(MODES),
                "learner": cfg["learner"], "action_space": deepcopy(cfg["action_space"]),
                "variables": deepcopy(cfg["variables"]), "interaction_budget_per_arm": interaction_budget,
                "source_evidence_refs": _refs(source_evidence_refs),
                "counted": ["A", "B_development", "failed_attempts", "reference_repeats", "applicability_checks_with_robot"],
                "independent_evaluation": "same frozen conditions and reserved budget per arm; never used for updates",
                "history_ablation": "same source data and current-task evidence; only historical method retrieval changes",
                "runtime_inputs": "observations and measured context only; experiment assignments are analysis labels"}
    return {**contract, "protocol_sha256": fingerprint(contract)}


def attach_offline_provenance(root, *, memory_root=None, prior_run_roots=()):
    """Attach existing offline results to RSI memory and actual-cost records.

    This is a replay audit, not a new physical experiment or a promotion. Original
    per-round and source files are retained and hashed; reruns are idempotent.
    """
    from .rsi_methods import MethodStore, baseline_method, validate_method
    from .agentic_learning import _signature, _hash
    root = Path(root).resolve()
    memory = MethodStore(memory_root or root / "method_memory")
    ledger = CostLedger(root)
    source_file = root / "source_costs.jsonl"
    sources = [json.loads(line) for line in source_file.read_text().splitlines() if line.strip()]
    if not sources:
        raise ValueError("offline provenance requires recorded source attempts")
    for row in sources:
        path = Path(row["trial_path"]) / "meta.json"
        if file_evidence([path])[0]["sha256"] != row["metadata_sha256"]:
            raise ValueError("source metadata changed since offline run: " + str(path))
        ledger.record("source:" + row["trial_path"], "source_experience",
                      trial_attempts=row["attempts"], failed_attempts=int(row.get("failed", False)),
                      wall_seconds=row.get("wall_seconds"), robot_seconds=row.get("robot_motion_seconds"),
                      reset_seconds=row.get("reset_seconds"), evidence_refs=[str(path), str(source_file)],
                      attempt_ids=[row["trial_path"]] if row["attempts"] == 1 else [],
                      metadata={"historical_role": row.get("historical_role"),
                                "recorded_stage_seconds": row.get("recorded_stage_seconds"),
                                "recorded_stage_is_not_robot_motion_only": True,
                                "source_metadata_sha256": row["metadata_sha256"]})
    reference_path = Path(sources[0]["session"]) / "session_meta.json"
    reference = json.loads(reference_path.read_text())
    context = {"simulated": bool(reference.get("simulated")),
               "hardware_signature": "protocol_only:" + _hash(_signature(reference)),
               "hardware_identity_limit": "Protocol identity only; cannot retrieve/presuppose previous physical hardware identity",
               "source_reference": file_evidence([reference_path])[0],
               "task": "bottle contact recovery", "evidence_type": "offline_replay"}
    # Preserve the stronger original RSI hardware/protocol identity if the
    # source campaign recorded one and its protocol agrees with this dataset.
    for source in sources:
        folder = Path(source["session"])
        candidate = folder.parent.parent / "rsi.json" if folder.parent.name == "batches" else folder / "rsi.json"
        if candidate.exists():
            state = json.loads(candidate.read_text())
            cfg = state.get("base_config")
            if cfg and state.get("hardware_signature") and _signature({"config": cfg, "simulated": context["simulated"]}) == _signature(reference):
                context["hardware_signature"] = state["hardware_signature"]
                context["hardware_signature_evidence"] = file_evidence([candidate])[0]
                context.pop("hardware_identity_limit", None)
                break
    base = baseline_method()
    baseline_id = memory.register(base)
    linked = []
    for round_path in sorted(root.glob("*/round_*/round.json")):
        row = json.loads(round_path.read_text())
        geometry, b_dev = row["development_geometry"], row["B_development"]
        refs = [str(round_path), str(root / "split_manifest.json")]
        invocation = {**geometry.get("invocation", {}), "n": geometry.get("n", 0),
                      "unknown": b_dev.get("invocation", {}).get("unknown", b_dev.get("n", 0)),
                      "label_definition": "measured geometry proxy only"}
        exit_status = b_dev.get("exit", {})
        feedback = {"role": "offline_development", "evidence_refs": refs, "invocation": invocation,
                    "recovery": {k: geometry.get(k, 0) for k in ("n", "mae_mm", "p90_mm", "rejected")},
                    "exit": {"n": 0, "unknown": exit_status.get("unknown_missing_prospective_confirmation", b_dev.get("n", 0)),
                             "proxy_only": True},
                    "gaps": b_dev.get("limitations", [])}
        identity = row["mode"] + "_round_" + str(row["round"]).zfill(2)
        selection = {"strategy": row["mode"], "experiment_choice": row["experiment_choice"],
                     "reason": row["reason"], "replayed_A_acquisition_trials": row["budget_selected_A_trials"],
                     "status": "executed_offline_replay", "new_physical_attempts": 0}
        result = {"role": "offline_development", "offline_replay": True,
                  "geometry": {k: v for k, v in geometry.items() if k != "details"},
                  "legacy_B": {k: v for k, v in b_dev.items() if k != "details"},
                  "source_round": str(round_path)}
        update = {"status": "recovery_model_retrained_deployment_unverified", **row["skill_update"]}
        round_record = record_round(root, identity, feedback=feedback, selection=selection,
                                    result=result, skill_update=update, evidence_paths=[round_path])
        method = deepcopy(base); method.pop("method_id")
        method.update(name="Bottle offline " + row["mode"], parent_id=baseline_id,
                      hypothesis="Replay hypothesis: " + row["mode"] + " can prioritize current-task acquisition.",
                      applicability="This bottle protocol only. Offline geometric proxies do not validate physical recovery, call scope or exit.",
                      evidence_refs=["offline_round:" + identity],
                      expected_outcome="Testable collection choice; requires independent physical comparison before promotion.")
        if row["mode"] == "fixed":
            method["score"] = {"op":"constant","value":0.}
        elif row["mode"] == "agent_evidence":
            method["score"] = {"op":"add","args":[{"op":"feature","name":"uncertainty"},{"op":"feature","name":"coverage_gap"}]}
        method = validate_method(method)
        marker = root / "rounds" / (identity + ".memory.json")
        if marker.exists():
            link = json.loads(marker.read_text())
            if link["round_sha256"] != fingerprint(round_record):
                raise ValueError("existing memory link no longer matches immutable round")
        else:
            link = persist_method_experience(memory, method, task_id="bottle_skill_offline_20261106",
                        context=context, round_record=round_record, simulated=context["simulated"],
                        applicability_check={"status":"unverified_offline_only",
                          "missing":["prospective recovery", "independent target-state labels", "post-action confirmation windows"]})
            link["round_sha256"] = fingerprint(round_record)
            atomic_json(marker, link)
        linked.append({"round_id":identity, **link})
        ledger.record("training:" + identity, "offline_compute", wall_seconds=row["cost"]["offline_training_wall_seconds"],
                      robot_seconds=0., reset_seconds=0., evidence_refs=refs,
                      metadata={"mode":row["mode"], "replay_selected_A":row["budget_selected_A_trials"],
                                "training_wall_only":True, "new_robot_attempts":0})
        ledger.record("applicability:" + identity, "method_applicability",
                      robot_seconds=0., reset_seconds=0., evidence_refs=refs,
                      metadata={"status":"not_verified", "basis":"offline protocol/geometry checks",
                                "time_not_separately_instrumented":True})
    for prior in prior_run_roots:
        prior = Path(prior).resolve()
        for path in sorted(prior.glob("*/round_*/round.json")):
            row = json.loads(path.read_text())
            ledger.record("development_attempt:" + str(path), "offline_compute",
                wall_seconds=row["cost"]["offline_training_wall_seconds"], robot_seconds=0., reset_seconds=0.,
                evidence_refs=[str(path)], metadata={"execution_status":"superseded_development_attempt",
                                                   "excluded_from_final_metrics":True})
    summary = ledger.summary()
    atomic_json(root / "cost_summary.json", summary)
    atomic_json(root / "method_memory_links.json", {"context":context,"rounds":linked,
                "lifecycle":memory.list_methods(), "promotion":"none; no prospective independent comparison"})
    return {"rounds":len(linked), "cost_summary":summary, "memory_root":str(memory.root)}
