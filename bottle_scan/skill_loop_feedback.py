"""A-axis round diagnosis, bounded collection proposals and explicit stop gates.

Pure analysis: no hardware imports. A pilot gate is not a validation certificate.
Independent evidence can assess a frozen version, but never proposes its update.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import math

from .skill_exploration import (choose_experiments,
                               validate_exploration_config, INDEPENDENT_ROLES)


DEFAULT_CRITERIA = {
    "expected_role": "development", "min_trials": 12,
    "min_normal": 3, "min_needs": 3, "min_target": 3,
    "min_not_target": 3, "min_verified_recoveries": 3,
    "min_invocation_recall": .8, "min_normal_specificity": .9,
    "min_recovery_success_rate": .8, "min_exit_recall": .8,
    "max_rejection_rate": .2, "max_false_success": 0,
    "require_task_success": False, "min_task_success_rate": .8,
}


def acceptance_criteria(value=None):
    cfg = deepcopy(DEFAULT_CRITERIA)
    if value:
        unknown = set(value) - set(cfg)
        if unknown:
            raise ValueError("unknown acceptance criteria: " + ", ".join(sorted(unknown)))
        cfg.update(value)
    if cfg["expected_role"] not in ("development", "independent"):
        raise ValueError("acceptance expected_role must be development or independent")
    for name in ("min_trials", "min_normal", "min_needs", "min_target",
                 "min_not_target", "min_verified_recoveries"):
        if type(cfg[name]) is not int or cfg[name] < 1:
            raise ValueError(name + " must be a positive integer")
    if type(cfg["max_false_success"]) is not int or cfg["max_false_success"] != 0:
        raise ValueError("premature successful exits must remain disallowed")
    if type(cfg["require_task_success"]) is not bool:
        raise ValueError("require_task_success must be explicit boolean")
    for name in ("min_invocation_recall", "min_normal_specificity", "min_recovery_success_rate",
                 "min_exit_recall", "max_rejection_rate", "min_task_success_rate"):
        if type(cfg[name]) not in (float, int) or not math.isfinite(cfg[name]) or not 0 <= cfg[name] <= 1:
            raise ValueError(name + " must be in [0,1]")
    return cfg


def _ratio(n, d):
    return n / d if d else None


def _verified_action(action):
    values = (action.get("command_mm"), action.get("measured_action_mm"))
    return (action.get("blocked") is False and action.get("axis") == "a" and
            all(type(v) in (float, int) and math.isfinite(v) and abs(v) > 1e-6 for v in values) and
            action.get("receipt_verified") is True and
            action.get("fresh_post_action_observation") is True)


def _outcomes(report):
    rows = report.get("task_outcomes") or []
    if isinstance(rows, dict):
        rows = list(rows.values())
    return rows


def _summarize_training_feedback(report, model):
    """Describe observed training labels without inventing B results."""
    model = model or {}
    counts = {name: deepcopy((model.get(name + "_head") or {}).get("class_counts", {}))
              for name in ("invocation", "exit")}
    reasons = list(model.get("blocking_reasons") or [])
    refs = list(report.get("evidence_refs") or [])
    findings = []
    contradictions = [x for x in reasons if str(x).startswith("contradictory_contact_labels:")]
    if contradictions:
        findings.append({"code": "contradictory_contact_labels", "component": "labels",
            "reason": "; ".join(contradictions),
            "collection_goal": "Review the original observed windows; never infer or silently overwrite labels.",
            "evidence_refs": refs})
    if reasons:
        findings.append({"code": "training_blocked", "component": "training",
            "reason": "; ".join(map(str, reasons)),
            "collection_goal": "Resolve verified label conflicts and the reported training data gaps before B.",
            "evidence_refs": refs})
    return {"schema": "a_skill_round_diagnosis_v1", "role": "training",
            "evidence_stage": "training_before_current_model_B", "B_evaluation": "not_evaluated_for_current_model",
            "learned_class_counts": counts,
            "class_counts_note": "Observed labelled samples; conflicting labels remain included and need review.",
            "data_gaps": deepcopy(model.get("data_gaps") or []),
            "blocking_reasons": reasons, "findings": findings,
            "summary_method": "deterministic_training_diagnosis_not_llm", "evidence_refs": refs}


def summarize_round(report, model=None):
    """Explain evidence gaps without turning model abstention into missing truth."""
    if report.get("feedback_source") == "training_data":
        return _summarize_training_feedback(report, model)
    details = report.get("details") or []
    classes = Counter(d.get("initial_class") or "unknown" for d in details)
    targets = Counter(str(d.get("final_target_reached")) for d in details)
    decisions = [d.get("initial_model_decision") or {} for d in details]
    reasons = Counter(report.get("rejection_diagnostics", {}).get("terminal_reasons", {}))
    if not reasons:
        reasons.update(d.get("diagnostic_terminal_reason") or
                       (d.get("terminal_exit") or {}).get("reason") or "unknown"
                       for d in details if (d.get("terminal_exit") or {}).get("completed") is False)
    outcomes = _outcomes(report)
    outcome_status = Counter(x.get("status", "unknown") for x in outcomes)
    missing_initial = sum(d.get("label_source_verified") is not True or
                          d.get("initial_class") not in ("already_normal", "needs_recovery", "not_applicable")
                          for d in details)
    missing_final = sum(d.get("label_source_verified") is not True or
                        type(d.get("final_target_reached")) is not bool for d in details)
    findings = []
    def add(code, component, reason, collection):
        findings.append(dict(code=code, component=component, reason=reason,
                             collection_goal=collection, evidence_refs=list(report.get("evidence_refs") or [])))
    if missing_initial or missing_final or not details:
        add("missing_contact_labels", "labels", "Some contact-state truth is absent or unverified",
            "Review recorded contact windows against the frozen target definition; keep unknowns unknown")
    if not classes["needs_recovery"]:
        add("no_needs_recovery_examples", "invocation", "No observed need-to-recover class",
            "Collect genuinely non-target but safely recoverable a-axis contacts; offsets alone are not labels")
    if not classes["already_normal"]:
        add("no_normal_examples", "invocation", "No observed normal controls",
            "Repeat confirmed target contacts to measure unnecessary calls")
    if not classes["not_applicable"]:
        add("unsupported_scope_unmeasured", "applicability", "No labelled unsupported-contact controls",
            "Use safely observed unsupported or unobservable contacts; do not manufacture dangerous trials")
    windows = report.get("completion_validation") or {}
    if not windows.get("target_count", targets["True"]):
        add("no_target_examples", "exit", "No independently labelled target-contact windows",
            "Record confirmed stable target contacts, including the unchanged zero reference")
    if not windows.get("not_target_count", targets["False"]):
        add("no_not_target_examples", "exit", "No independently labelled non-target completion windows",
            "Review pre-action and intermediate non-target windows; successful terminal contacts alone cannot train stopping")
    if reasons.get("policy_uncertain") or any("uncertain" in k for k in reasons):
        add("model_uncertainty", "recovery", "Recorded model abstention, not missing operator truth",
            "Compare compact models and features offline; add repeated uncertain contact states within frozen limits")
    if reasons.get("policy_out_of_distribution") or any("out_of_distribution" in k for k in reasons):
        add("model_ood", "applicability", "Observed features outside frozen model support",
            "Check reference/feature reliability and collect relevant a-axis states at unchanged probe force")
    actions = report.get("actions") or []
    good = [a for a in actions if _verified_action(a)]
    if not good:
        add("no_verified_recovery_actions", "recovery", "No measured action followed by fresh tactile observation",
            "Collect approved pre-action/post-action trajectories, including unsuccessful outcomes")
    if report.get("exit", {}).get("false_success", 0):
        add("premature_exit", "exit", "Successful exit was declared before observed target contact",
            "Collect completion boundaries and still-non-target windows after each action")
    if report.get("exit", {}).get("missed_completion", 0):
        add("missed_exit", "exit", "Observed target contact was not recognized",
            "Repeat confirmed target windows and test learned completion predictions")
    if outcome_status["force_not_reached"]:
        add("downstream_force_not_reached", "downstream", "Squeeze failed after the contact stage",
            "Inspect force/opening protocol separately; do not relabel contact merely to explain squeeze failure")
    if model and model.get("blocking_reasons"):
        add("training_blocked", "training", "; ".join(map(str, model["blocking_reasons"])),
            "Resolve learner data_gaps before real learned-policy B execution")
    return {"schema": "a_skill_round_diagnosis_v1", "role": report.get("role"),
            "model_ids": list(report.get("model_ids") or []), "classes": dict(classes),
            "terminal_target_labels": dict(targets), "missing_initial_labels": missing_initial,
            "missing_final_labels": missing_final, "model_unknown_predictions": report.get("invocation", {}).get("unknown", 0),
            "rejection_reasons": dict(reasons), "task_outcomes": dict(outcome_status),
            "uncertainty_mm": [d["uncertainty_mm"] for d in decisions if "uncertainty_mm" in d],
            "ood_distances": [d["ood_distance"] for d in decisions if "ood_distance" in d],
            "verified_actions": len(good), "findings": findings,
            "data_gaps": deepcopy((model or {}).get("data_gaps", [])),
            "learned_class_counts": {name: deepcopy((model or {}).get(name + "_head", {}).get("class_counts", {}))
                                     for name in ("invocation", "exit")},
            "summary_method": "deterministic_evidence_diagnosis_not_llm",
            "evidence_refs": list(report.get("evidence_refs") or [])}


def _assess_target_set(details, model):
    """Verify observed exits and coverage of disconnected acceptable regions.

    A component is an overlapping union of empirical neighborhoods, not a claim
    that every point in that region was tested. One observed exit is only pilot
    coverage; the ordinary invocation/recovery/exit rate checks still apply.
    """
    import numpy as np
    from .acceptable_contact_set import validate, match_features
    target = model.get('acceptable_set') or {}
    try:
        validate(target, model.get('target_definition_id'))
    except (ValueError, TypeError, KeyError):
        return {'valid': False, 'reasons': ['acceptable contact set artifact is invalid'],
                'components': [], 'unobserved_components': []}
    anchors = target['anchors']
    centers = (np.asarray([a['features'] for a in anchors])-target['scaling']['mean'])/target['scaling']['scale']
    unseen = set(range(len(anchors))); components = []
    while unseen:
        pending = [min(unseen)]; group = set()
        while pending:
            i = pending.pop()
            if i not in unseen:
                continue
            unseen.remove(i); group.add(i)
            for j in sorted(unseen):
                distance = float(np.sqrt(np.mean((centers[i]-centers[j])**2)))
                if distance <= anchors[i]['radius']+anchors[j]['radius']:
                    pending.append(j)
        components.append(sorted(anchors[i]['id'] for i in group))
    visited = set(); invalid = []
    for detail in details:
        terminal = detail.get('terminal_exit') or {}
        if terminal.get('completed') is not True:
            continue
        achieved = terminal.get('achieved_state') or {}
        try:
            features = np.asarray(achieved.get('contact_features'), float)
            if features.shape != (4,) or not np.isfinite(features).all():
                raise ValueError('missing features')
            match = match_features(target, features)
            checked = (detail.get('set_terminal_observation_verified') is True and
                       achieved.get('set_id') == target['set_id'] and
                       achieved.get('target_definition_id') == model['target_definition_id'] and
                       achieved.get('supported') is True and match['supported'] and
                       type(achieved.get('timestamp_ns')) is int and achieved['timestamp_ns'] > 0 and
                       achieved.get('anchor_ids') == match['anchor_ids'])
        except (ValueError, TypeError, KeyError):
            checked = False
        if not checked:
            invalid.append(detail.get('trial_path'))
        elif detail.get('final_target_reached') is True and detail.get('label_source_verified') is True:
            visited.update(match['anchor_ids'])
    missing = [i for i, component in enumerate(components) if not visited.intersection(component)]
    reasons = []
    if invalid:
        reasons.append('completed exits lack verified observed acceptable-set membership')
    if missing:
        reasons.append('some disconnected acceptable contact regions have no verified B target observations')
    return {'valid': not reasons, 'set_id': target['set_id'], 'reasons': reasons,
            'components': components, 'observed_anchor_ids': sorted(visited),
            'unobserved_components': missing, 'invalid_completed_trials': invalid,
            'interpretation': 'Observed pilot coverage of each supported region; not exhaustive coverage or reliability proof'}


def assess_acceptance(report, criteria=None, *, expected_model_id, model=None):
    """Conservative, configurable pilot stop gate for ONE frozen learned version.

    Missing evidence fails. Unknown predictions stay in rate denominators.
    Passing development only requests a new, frozen independent evaluation.
    A pass never issues a deployment certificate or expands motion limits.
    """
    cfg = acceptance_criteria(criteria)
    reasons = []
    def require(condition, reason):
        if not condition:
            reasons.append(reason)
    require(isinstance(expected_model_id, str) and bool(expected_model_id), "expected frozen model id missing")
    require(report.get("role") == cfg["expected_role"], "evaluation role differs from frozen acceptance role")
    require(report.get("model_ids") == [expected_model_id], "evaluation mixes or mismatches frozen model versions")
    require(report.get("session_complete") is True, "B session is incomplete or completion is unverified")
    require(report.get("simulated") is False, "simulated/unknown evidence cannot establish physical acceptance")
    require(bool(report.get("evidence_refs")), "source evidence references missing")
    model = model or report.get("learned_model") or {}
    require(model.get("schema") in ("learned_a_skill_v1", "learned_a_skill_set_v1") and model.get("model_id") == expected_model_id,
            "three-part learned skill artifact missing or mismatched")
    require(model.get("usable_for_B") is True, "learned skill is not ready for B")
    for name in ("invocation_head", "exit_head"):
        require((model.get(name) or {}).get("usable") is True, name + " has not passed its training gates")
    require((model.get("recovery_model") or {}).get("usable_for_B") is True,
            "recovery model has not passed its training gates")
    require(bool(model.get("target_definition_id")) and report.get("target_definition_id") == model.get("target_definition_id"),
            "contact target definition missing or mismatched")
    details = report.get("details") or []
    coverage = report.get("coverage") or {}
    n = len(details)
    require(coverage.get("trials") == n and n >= cfg["min_trials"], "insufficient or inconsistent B trial count")
    require(coverage.get("recording_verified") == n and n > 0, "some B trials lack complete recordings")
    require(len({d.get("trial_path") for d in details}) == n, "duplicate physical trials in acceptance evidence")
    require(all(d.get("model_id") == expected_model_id for d in details), "trial model identity mismatch")
    require(all(d.get("label_source_verified") is True for d in details), "unverified independent contact-state labels")
    normal = [d for d in details if d.get("initial_class") == "already_normal"]
    needs = [d for d in details if d.get("initial_class") == "needs_recovery"]
    labels_complete = all(d.get("initial_class") in ("already_normal", "needs_recovery", "not_applicable")
                          and type(d.get("final_target_reached")) is bool for d in details)
    require(labels_complete, "contact-state labels incomplete")
    require(all((d.get("terminal_exit") or {}).get("hand_back") is True and
                type((d.get("terminal_exit") or {}).get("completed")) is bool for d in details),
            "terminal skill handoff missing")
    require(not any(d.get("final_target_reached") is False and
                    (d.get("terminal_exit") or {}).get("completed") is True for d in details),
            "premature terminal completion observed")
    require(len(normal) >= cfg["min_normal"], "insufficient normal-contact controls")
    require(len(needs) >= cfg["min_needs"], "insufficient needs-recovery trials")
    require(not any(d.get("initial_class") == "not_applicable" and
                    (d.get("invocation") or {}).get("invoke") is True for d in details),
            "skill invoked on an independently labelled inapplicable contact")
    called = sum((d.get("invocation") or {}).get("invoke") is True for d in needs)
    normal_correct = sum((d.get("invocation") or {}).get("needs_recovery") is False and
                         (d.get("invocation") or {}).get("invoke") is False for d in normal)
    rejected = sum((d.get("invocation") or {}).get("needs_recovery") is None or
                   (d.get("invocation") or {}).get("permitted") is not True for d in details)
    actions = report.get("actions") or []
    valid_paths = {d.get("trial_path") for d in details}
    require(all(a.get("trial_path") in valid_paths for a in actions), "actions reference unknown trials")
    good_paths = {a.get("trial_path") for a in actions if _verified_action(a)}
    require(all(a.get("blocked") is True or _verified_action(a) for a in actions),
            "verified a-axis action receipt or fresh post-action observation missing")
    recovered = sum(d.get("trial_path") in good_paths and d.get("final_target_reached") is True and
                    (d.get("invocation") or {}).get("invoke") is True and
                    (d.get("terminal_exit") or {}).get("completed") is True for d in needs)
    require(recovered >= cfg["min_verified_recoveries"], "insufficient verified successful recovery trials")
    # Completion windows, when available, include pre-terminal negative states;
    # requiring failed terminal trials would penalize successful recovery itself.
    window = report.get("completion_validation")
    if window is not None:
        tp, tn, fp, fn = (window.get(k, 0) for k in ("TP", "TN", "FP", "FN"))
        unknown = window.get("unknown", 0)
        target_n, not_target_n = window.get("target_count", 0), window.get("not_target_count", 0)
        counts = (tp, tn, fp, fn, unknown, target_n, not_target_n)
        valid_counts = all(type(x) is int and x >= 0 for x in counts)
        require(valid_counts and tp+tn+fp+fn+unknown == target_n+not_target_n and
                tp+fn <= target_n and tn+fp <= not_target_n,
                "inconsistent completion-window evidence")
        if not valid_counts:
            tp = tn = fp = fn = unknown = target_n = not_target_n = 0
        require(window.get("label_source_verified") is True, "completion-window labels unverified")
    else:
        target_n = sum(d.get("final_target_reached") is True for d in details)
        not_target_n = sum(d.get("final_target_reached") is False for d in details)
        tp = sum(d.get("final_target_reached") is True and (d.get("terminal_exit") or {}).get("completed") is True for d in details)
        fp = sum(d.get("final_target_reached") is False and (d.get("terminal_exit") or {}).get("completed") is True for d in details)
        unknown = sum((d.get("terminal_exit") or {}).get("hand_back") is not True for d in details)
    require(target_n >= cfg["min_target"], "insufficient observed target-contact windows")
    require(not_target_n >= cfg["min_not_target"], "insufficient observed non-target-contact windows")
    require(fp <= cfg["max_false_success"], "premature successful exit observed")
    require(not unknown, "completion predictions or terminal handoffs are unknown")
    rates = {"invocation_recall": _ratio(called, len(needs)),
             "normal_specificity": _ratio(normal_correct, len(normal)),
             "recovery_success_rate": _ratio(recovered, len(needs)),
             "exit_recall": _ratio(tp, target_n), "rejection_rate": _ratio(rejected, n)}
    terminal_targets = [d for d in details if d.get("final_target_reached") is True]
    rates["terminal_exit_recall"] = _ratio(sum((d.get("terminal_exit") or {}).get("completed") is True
                                              for d in terminal_targets), len(terminal_targets))
    require(rates["terminal_exit_recall"] is not None and rates["terminal_exit_recall"] >= cfg["min_exit_recall"],
            "terminal exit recall below target or unavailable")
    for key in ("invocation_recall", "normal_specificity", "recovery_success_rate", "exit_recall"):
        require(rates[key] is not None and rates[key] >= cfg["min_"+key], key + " below target or unavailable")
    require(rates["rejection_rate"] is not None and rates["rejection_rate"] <= cfg["max_rejection_rate"],
            "rejection rate above target or unavailable")
    outcomes = _outcomes(report)
    task_successes = sum(x.get("success") is True for x in outcomes)
    task_rate = _ratio(task_successes, n)
    if cfg["require_task_success"]:
        require(len(outcomes) == n and len({x.get("trial_path") for x in outcomes}) == n and
                {x.get("trial_path") for x in outcomes} == valid_paths,
                "downstream task outcomes incomplete or mismatched")
        require(task_rate is not None and task_rate >= cfg["min_task_success_rate"],
                "downstream task success below target or unavailable")
    metrics = {"trials": n, "normal": len(normal), "needs_recovery": len(needs),
               "verified_recoveries": recovered, "target_windows": target_n,
               "not_target_windows": not_target_n, "false_success": fp,
               "unknown_completion": unknown, "task_success_rate": task_rate, **rates}
    if model.get('schema') == 'learned_a_skill_set_v1' or model.get('target_mode') == 'acceptable_set':
        set_evidence = _assess_target_set(details, model)
        reasons.extend(set_evidence['reasons'])
        metrics['acceptable_set'] = set_evidence
    return {"schema": "a_skill_acceptance_v1", "passed": not reasons,
            "reasons": reasons, "metrics": metrics, "criteria": cfg,
            "model_id": expected_model_id, "certificate_issued": False,
            "interpretation": "Pilot evidence gate; no statistical reliability or deployment certificate is implied",
            "next_stage": ("frozen_independent_B" if cfg["expected_role"] == "development" else "complete")
                          if not reasons else "needs_more_evidence_or_model_revision"}


def propose_collection(report, exploration_config, *, model=None, mode="agent_evidence",
                       seed=0, remaining_trials=None):
    """Return an unexecuted, bounded a-only plan with an explicit zero control."""
    if report.get("role") in INDEPENDENT_ROLES:
        raise ValueError("independent evaluation must never guide model updates")
    cfg = validate_exploration_config(exploration_config)
    if cfg["variables"]["a_mm"]["reference"] != 0 or cfg["variables"]["contact_height_l_mm"]["reference"] != 0:
        raise ValueError("a-axis loop requires zero a/l reference")
    if 0 not in cfg["variables"]["a_mm"]["levels"]:
        raise ValueError("a-axis loop requires an explicit zero reference level")
    if not all(-25 <= x <= 25 for x in cfg["variables"]["a_mm"]["range"]):
        raise ValueError("a-axis exploration exceeds existing 25 mm envelope")
    # Existing cell failures describe final operator labels. For selection only,
    # also count recorded abstention and actual-call/exit errors in their cells.
    # Keep source reports unchanged and publish the selection scoring provenance.
    scored = deepcopy(report)
    by_path = {d.get("trial_path"): d for d in report.get("details", [])}
    for cell in scored.get("cells", []):
        problems = 0
        for ref in cell.get("evidence_refs", []):
            d = by_path.get(ref.rsplit("/", 1)[0], {})
            inv = d.get("invocation") or {}
            expected = d.get("initial_class")
            bad = (bool(d) and (inv.get("needs_recovery") is None or
                   (expected == "needs_recovery" and inv.get("invoke") is not True) or
                   (expected == "already_normal" and inv.get("invoke") is True) or
                   ((d.get("terminal_exit") or {}).get("completed") != d.get("final_target_reached"))))
            problems += int(bad)
        cell["failures"] = max(cell.get("failures", 0), problems)
    proposal = choose_experiments(scored, cfg, mode=mode, seed=seed,
                                  remaining_trials=remaining_trials, stage="a_closed_loop")
    diagnosis = summarize_round(report, model)
    reference = {name: v["reference"] for name, v in cfg["variables"].items()}
    selected = proposal["selected"]
    if selected and not any(x["condition"] == reference for x in selected):
        selected[-1] = {"condition": reference, "purpose": "reference_repeat",
                        "reps": cfg["reps_per_condition"], "bottle_blocks": ["light", "heavy"],
                        "condition_kind": "A", "reason": "Budgeted unchanged zero reference for drift/control comparison",
                        "problem_refs": []}
    selected.sort(key=lambda x: x["condition"] != reference)
    if cfg["reps_per_condition"] > cfg["reference_every"]:
        raise ValueError("repetitions exceed physical-trial reference interval")
    since_reference = 0
    for index, item in enumerate(selected):
        if item["condition"] == reference:
            since_reference = 0
        elif since_reference + item["reps"] > cfg["reference_every"]:
            selected[index] = {"condition": deepcopy(reference), "purpose": "reference_repeat",
                               "reps": cfg["reps_per_condition"], "bottle_blocks": ["light", "heavy"],
                               "condition_kind": "A", "reason": "Budgeted periodic zero reference",
                               "problem_refs": []}
            since_reference = 0
        else:
            since_reference += item["reps"]
    goals = [f["collection_goal"] for f in diagnosis["findings"]]
    for item in selected:
        item["condition_kind"] = "A"
        item["required_observations"] = list(goals)
        item["label_warning"] = "Planned offset is never a contact-state label; label the observed state against the frozen target definition"
    proposal.update(diagnosis=diagnosis, target_definition_id=(model or {}).get("target_definition_id"),
                    fixed_controls={"contact_height_l_mm": 0, "probe_force": reference["probe_force"]},
                    selection_score="maximum of observed cell failures and recorded model/call/exit problems; source report unchanged",
                    stop_claim=False, execution_authorized=False,
                    selector="deterministic_evidence_heuristic_not_llm")
    return proposal
