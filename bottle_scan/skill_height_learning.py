"""Measured local-l geometric imitation using the unchanged ContactPolicy learner.

No planned perturbation is used as a label or runtime feature. Only fixed-a,
fixed-force height sessions enter; force/combined/independent sessions cannot
silently supply training samples. A trained candidate still needs prospective B.
"""
from copy import deepcopy
import json
from pathlib import Path
import numpy as np
from .agentic_learning import _hash, _signature, _pose, _fit
from .contact_policy import (_settings, _network, _predict, _seal, _write,
                             feature_names, feature_vector)
from .skill_exploration import file_evidence


def train_height_policy(roots, settings=None, *, out=None):
    settings, cfg = _settings(settings)
    roots = sorted({Path(x).resolve() for x in roots})
    model = {"schema":"contact_policy_v1", "family":"numpy_tanh_mlp_ensemble",
             "usable_for_B":False, "settings":cfg, "source_sessions":[], "source_trials":[],
             "fingers":[], "blocking_reasons":[], "recovery_action_axis":"l",
             "recovery_label_source":"measured_tcp_projection_l", "recovery_controlled_axes":["a"],
             "feature_definition":{"version":"contact_descriptor_v1", "names":feature_names(cfg["dd_pool"]),
                                   "dd_pool":cfg["dd_pool"], "runtime_inputs":["two_current_tactile_descriptors"],
                                   "forbidden_runtime_inputs":["planned_l_mm","TCP","delta_mm","trial_id"]},
             "label_definition":"negative measured PROBE TCP displacement on original local l; geometric return to T0, not optimal action",
             "deployment_certificate":"not_issued_requires_independent_prospective_B",
             "caveats":["Local l is not automatically world z.", "Bottle motion invalidates geometric labels.",
                        "No a-axis directional fallback is authorized for l.",
                        "Geometric validation does not prove physical recovery or contact completion."]}
    observations, features, skipped = [], [], []
    reference = None
    for root in roots:
        meta = json.loads((root/"session_meta.json").read_text())
        design = meta.get("ambiguity_design", {})
        from .skill_extension import validate_probe_design
        validate_probe_design(design)
        if design.get("stage") != "height_controlled" or meta.get("evaluation_role", "development") != "development":
            model["blocking_reasons"].append(str(root)+": requires controlled-height development session")
            continue
        if reference is None: reference = meta
        if _signature(meta) != _signature(reference) or not np.allclose(meta["T0"], reference["T0"], atol=1e-6):
            model["blocking_reasons"].append(str(root)+": protocol/T0 mismatch")
            continue
        T0 = _pose(meta["T0"])
        if T0 is None:
            model["blocking_reasons"].append(str(root)+": invalid measured T0"); continue
        axis = T0[:3,:3] @ np.asarray(meta["config"]["axes_tcp"]["l"])
        a = T0[:3,:3] @ np.asarray(meta["config"]["axes_tcp"]["a"])
        force = meta["config"]["force"]["target"] * meta["config"]["force"]["probe_fraction"]
        for path in sorted((root/"trials").glob("*/meta.json")):
            row = json.loads(path.read_text())
            p = row.get("probe_initial") or {}
            measured = _pose(p.get("measured_tcp_mm"))
            recording = path.parent/"recording_status.json"
            eligible = (row.get("complete") and row.get("recording_complete") and recording.exists()
                        and json.loads(recording.read_text()).get("complete")
                        and row.get("label") == "normal" and not row.get("bottle_moved")
                        and row.get("cond") == "A" and p.get("valid") and p.get("stable_probe")
                        and measured is not None and p.get("measurement_timestamp_ns")
                        and row.get("evaluation_role", "development") == "development")
            if not eligible:
                skipped.append(str(path)); continue
            delta = measured[:3,3] - T0[:3,3]
            x = float(delta @ axis)
            orthogonal = float(np.linalg.norm(delta-axis*x))
            rotation = float(np.degrees(np.arccos(np.clip((np.trace(T0[:3,:3].T @ measured[:3,:3])-1)/2,-1,1))))
            actual_force = row.get("probe_force_target")
            wrench = np.asarray(p.get("wrench_zeroed", []), float)
            measured_force_ok = (wrench.shape == (2, 6) and np.isfinite(wrench).all())
            if measured_force_ok:
                force_cfg = meta["config"]["force"]
                normals = wrench[:, force_cfg["normal_axis"]] * np.asarray(force_cfg["normal_signs"])
                measured_force_ok = (np.all(normals > force_cfg["min_contact"]) and
                    abs(float(normals.mean())-force) <= force_cfg["probe_tolerance"])
            if abs(x)>8. or not measured_force_ok or orthogonal > 1. or rotation > 1. or abs(float(actual_force or -100)-force) > 1e-6:
                skipped.append(str(path)); continue
            try:
                descriptors = p["descriptors"]
                if any(d.get("touches_edge") for d in descriptors): raise ValueError("edge")
                vector = feature_vector(descriptors,dd_pool=cfg["dd_pool"])
            except (ValueError, TypeError, KeyError):
                skipped.append(str(path)); continue
            source = file_evidence([path])[0]
            observations.append({"id":row["id"],"session":str(root),"trial_path":str(path.parent),
                "metadata_sha256":source["sha256"],"group":str(root)+":repeat:"+str(row["repeat"]),
                "descriptors":descriptors,"x_mm":x,"target_action_mm":-x,
                "planned_delta_mm":round(x,1),  # Existing fitter grouping only; measured value.
                "label_source":"A_measured_l_return_to_T0_not_optimal",
                "measurement":{"source":"measured_tcp_projection_l","x_mm":x,"off_axis_mm":orthogonal,
                               "timestamp_ns":p["measurement_timestamp_ns"]}})
            features.append(vector)
    model["excluded_trial_files"] = skipped
    model["source_trials"] = [{k:v for k,v in x.items() if k not in ("descriptors","planned_delta_mm")} for x in observations]
    model["source_sessions"] = [str(x) for x in roots]
    if reference:
        model.update(reference_T0=reference["T0"],axes_tcp=reference["config"]["axes_tcp"],
                     normal_axis=reference["config"]["force"]["normal_axis"],
                     normal_signs=reference["config"]["force"]["normal_signs"],
                     signature=_hash(_signature(reference)),simulated=bool(reference.get("simulated",False)))
    if len(observations) < cfg["min_samples"]:
        model["blocking_reasons"].append(f"need {cfg['min_samples']} independent recorded height samples; got {len(observations)}")
    groups = sorted({x["group"] for x in observations})
    if len(groups) < 3:
        model["blocking_reasons"].append("need >=3 repeated-contact blocks")
    if model["blocking_reasons"]: return _write(_seal(model),out)
    rng = np.random.default_rng(cfg["seed"])
    held = set(np.asarray(groups)[rng.permutation(len(groups))[:max(1,int(np.ceil(len(groups)*cfg["validation_fraction"])))]])
    train = np.asarray([i for i,o in enumerate(observations) if o["group"] not in held],int)
    valid = np.asarray([i for i,o in enumerate(observations) if o["group"] in held],int)
    if len(train)<cfg["min_train_samples"] or len(valid)<cfg["min_validation_samples"]:
        model["blocking_reasons"].append("whole-repeat holdout leaves insufficient training/validation samples")
        return _write(_seal(model),out)
    X = np.asarray(features); y = np.asarray([o["target_action_mm"] for o in observations])
    mean,scale = X[train].mean(0),X[train].std(0); scale=np.where(scale>1e-6,scale,1.)
    Z=(X-mean)/scale; y_scale=max(float(y[train].std()),1.)
    distinct = len(set(round(float(v),0) for v in y[train]))
    conflicts=0; neighbors=[]
    for i in train:
        other=[j for j in train if observations[j]["group"]!=observations[i]["group"]]
        dist=np.sqrt(np.mean((Z[other]-Z[i])**2,axis=1))
        neighbors.append(float(dist.min()))
        conflicts += int(any(d<.15 and abs(y[j]-y[i])>2. for d,j in zip(dist,other)))
    model["identifiability"]={"independent_measured_levels":distinct,
        "training_ambiguous_fraction":conflicts/len(train),"max_allowed_ambiguous_fraction":.2,
        "definition":"different-repeat tactile RMS <0.15 with measured label difference >2 mm"}
    if distinct<3 or np.ptp(y[train])<2. or conflicts/len(train)>.2:
        model["blocking_reasons"].append("height contact observations do not establish separable physical conditions")
    fallback=_fit([observations[i] for i in train],"linear",False,settings.get("learning",{}))
    if fallback is None: model["blocking_reasons"].append("no invertible measured-l tactile mapping")
    if model["blocking_reasons"]: return _write(_seal(model),out)
    networks=[_network(Z[train],y[train]/y_scale,cfg,rng) for _ in range(cfg["ensemble_size"])]
    predictions=np.asarray([_predict(net,Z[valid])*y_scale for net in networks]).mean(0)
    errors=np.abs(predictions-y[valid]); mae=float(errors.mean()); p90=float(np.quantile(errors,.9))
    radius=max(cfg["ood_min_radius"],float(np.quantile(neighbors,.95))*cfg["ood_radius_multiplier"])
    model.update(networks=networks,fallback_fit=fallback,fingers=fallback["fingers"],
        common_range_mm=fallback["common_range_mm"],common_linear_range_mm=fallback["common_linear_range_mm"],
        scaling={"mean":mean.tolist(),"scale":scale.tolist(),"y_scale":y_scale},
        ood={"training_support":Z[train].tolist(),"rms_radius":radius,"calibration":"training-only cross-repeat neighbor distance"},
        uncertainty={"validation_error_p90_mm":p90,"max_mm":cfg["max_uncertainty_mm"]},
        validation={"n":len(valid),"mae_mm":mae,"p90_mm":p90,"method":"held_out_whole_contact_repeat",
                    "predictions_mm":predictions.tolist(),"targets_mm":y[valid].tolist(),"independent_final_validation":False},
        split={"training_groups":sorted(set(groups)-held),"validation_groups":sorted(held),
               "selection":"seeded repeat groups; no feature/label access"},
        training_count=len(train),validation_count=len(valid),
        training_snapshot_sha256=_hash({"features":X.tolist(),"labels":y.tolist(),"sources":model["source_trials"]}))
    if mae>cfg["max_validation_mae_mm"] or p90>min(cfg["max_validation_p90_mm"],cfg["max_uncertainty_mm"]):
        model["blocking_reasons"].append("held-out measured-l geometric error exceeds gate")
    model["usable_for_B"]=not model["blocking_reasons"]
    return _write(_seal(model),out)
