"""Training-only repeatability models for observation-space graph distances."""
from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict

import numpy as np

MODALITIES = ("pos", "force", "open")
DEPLOYMENT_FIELDS = ("z_tac_left", "z_tac_right", "sensor_contact_left",
                     "sensor_contact_right", "wrench_left", "wrench_right",
                     "gripper_opening", "z_vis", "legacy_features")


def _finite(value):
    if value is None:
        return None
    try:
        a = np.asarray(value, dtype=float).reshape(-1)
    except (ValueError, TypeError):
        return None
    return a if a.size and np.isfinite(a).all() else None


def _position(value, units="mm"):
    if not isinstance(value, dict):
        return None
    keys = (("u_mm", "v_mm", "area_mm2", "edge_dist_root_mm", "edge_dist_tip_mm") if units=="mm" else
            ("u_px", "v_px", "area_px", "edge_dist_root_px", "edge_dist_tip_px"))
    return _finite([value.get(k) for k in keys])


def observation_vectors(row):
    """Explicit allow-list: no result, TCP, planned action or graph labels."""
    l, r = _finite(row.get("z_tac_left")), _finite(row.get("z_tac_right"))
    pl, pr = _position(row.get("sensor_contact_left")), _position(row.get("sensor_contact_right"))
    ppl, ppr = _position(row.get("sensor_contact_left"), "px"), _position(row.get("sensor_contact_right"), "px")
    pc = []
    for side in ("left", "right"):
        value = row.get("sensor_contact_"+side) or {}
        pc.append(_finite([value.get(k) for k in ("u_px", "v_px", "area_px")]))
    fl, fr = _finite(row.get("wrench_left")), _finite(row.get("wrench_right"))
    return {
        "tac": np.concatenate([l, r]) if l is not None and r is not None else None,
        # Mean+difference contains exactly the two fingers' geometry, without
        # additionally counting duplicate left/right coordinates in the metric.
        "pos": np.concatenate([(pc[0]+pc[1])/2,pc[0]-pc[1]]) if all(x is not None for x in pc) else None,
        "pos_mm": np.concatenate([(pl + pr) / 2, pl - pr]) if pl is not None and pr is not None else None,
        "pos_px": np.concatenate([(ppl + ppr) / 2, ppl - ppr]) if ppl is not None and ppr is not None else None,
        "pos_px_centroid": np.concatenate([(pc[0]+pc[1])/2,pc[0]-pc[1]]) if all(x is not None for x in pc) else None,
        "force": np.concatenate([fl, fr]) if fl is not None and fr is not None else None,
        "open": _finite(row.get("gripper_opening")),
        "vis": _finite(row.get("z_vis")),
        "legacy": _finite(row.get("legacy_features")),
    }


def modality_vector(row, name):
    return observation_vectors(row).get(name)


def repeat_group(row):
    # Experimental offset is used ONLY to identify repeats, never as a vector.
    return (str(row.get("session")), str(row.get("object_id")), row.get("planned_delta_mm"), row.get("protocol_id"), row.get("time_period"))


def fit_noise(rows, config=None):
    """Fit on the supplied training rows, excluding explicitly held-out roles.

    config.fit_node_ids optionally specifies the exact training subset. Refit
    this function for every fold; never reuse a full-data PCA during evaluation.
    v2: independent differences use 2*Sigma and family-wise alpha/M thresholds.
    """
    from scipy.stats import chi2
    from sklearn.covariance import LedoitWolf
    from sklearn.decomposition import PCA

    cfg = dict(config or {})
    ids = set(cfg["fit_node_ids"]) if "fit_node_ids" in cfg else None
    held = {"independent", "test", "final", "validation", "formal_validation", "calibration"}
    def eligible(row):
        from .provenance import frozen_independent
        if row.get("evaluation_only", False) or frozen_independent(row):
            return False
        return not any(str(row.get(key, "")).lower() in held
                       for key in ("split", "data_role", "role", "source_role"))
    train = [r for r in rows if (ids is None or r.get("node_id") in ids) and eligible(r)]
    model = {"schema": "state_graph_noise_v2", "fit_node_ids": [r["node_id"] for r in train],
             "source": "within_repeat_group_residual_covariance", "quantile": 1-float(cfg.get("alpha",.05))/len(cfg.get("distance_modalities",cfg.get("modalities",MODALITIES))),
             "alpha":float(cfg.get("alpha",.05)), "covariance_multiplier":2.,
             "distance_modalities":list(cfg.get("distance_modalities",cfg.get("modalities",MODALITIES))),
             "position_units":"pixels; physical mm only for display",
             "pca_dimension_requested": int(cfg.get("pca_dim", 32)),
             "w_vis": float(cfg.get("w_vis", .5)), "modalities": {},
             "feature_allowlist": list(DEPLOYMENT_FIELDS),
             "notes": ["Independent observation differences have covariance 2*Sigma; Bonferroni uses the active modality count, including optional modalities only when enabled.",
                       "Missing modalities remain unavailable; thresholds are not fitted to acceptance tests."]}
    for name in cfg.get("modalities", ("pos", "force", "open", "tac", "vis", "legacy", "pos_px", "pos_px_centroid", "pos_mm")):
        kept = [(r, modality_vector(r, name)) for r in train]
        kept = [(r, x) for r, x in kept if x is not None]
        item = {"status": "insufficient_data", "n": len(kept), "n_train": len(train)}
        model["modalities"][name] = item
        if len(kept) < 3:
            continue
        dimensions = {len(x) for _, x in kept}
        if len(dimensions) != 1:
            item.update(status="inconsistent_dimension")
            continue
        X = np.stack([x for _, x in kept]); dim = X.shape[1]
        mean = X.mean(0)
        if dim > 32:
            k = min(int(cfg.get("pca_dim", 32)), len(X)-1, dim)
            pca = PCA(n_components=k, svd_solver="full").fit(X)
            components = pca.components_
            Y = pca.transform(X)
            mean = pca.mean_
        else:
            components = None
            Y = X-mean
        groups = defaultdict(list)
        seen_trials = set()
        for i, (row, _) in enumerate(kept):
            trial_key = (row.get("session"), row.get("trial_id", row.get("node_id")))
            initial = row.get("phase", "A") in ("A", "B_pre") and int(row.get("window_index", 0)) == 0
            if initial and trial_key not in seen_trials and row.get("planned_delta_mm") is not None:
                groups[repeat_group(row)].append(i)
                seen_trials.add(trial_key)
        repeats = {g: indexes for g, indexes in groups.items() if len(indexes) >= 2}
        residuals = [Y[indexes] - Y[indexes].mean(0) for indexes in repeats.values()]
        item.update(input_dimension=dim, dimension=Y.shape[1], df=Y.shape[1],
                    pca_mean=mean.tolist(), pca_components=components.tolist() if components is not None else None,
                    repeat_sources=[{"session": g[0], "object_id": g[1], "planned_delta_mm": g[2],
                                     "node_ids": [kept[i][0]["node_id"] for i in inds]}
                                    for g, inds in sorted(repeats.items(), key=lambda x:str(x[0]))])
        if not residuals or sum(len(r) for r in residuals) < 3:
            item["status"] = "insufficient_repeat_groups"
            continue
        residuals = np.concatenate(residuals)
        fit = LedoitWolf(assume_centered=True).fit(residuals)
        covariance = fit.covariance_
        # Numerical floor only; not a measurement-noise/acceptance tuning knob.
        floor = max(float(np.trace(covariance)/len(covariance))*1e-9, 1e-12)
        covariance = covariance + np.eye(len(covariance))*floor
        item.update(status="ready", n_residuals=len(residuals), shrinkage=float(fit.shrinkage_),
                    single_observation_covariance=covariance.tolist(), covariance=(2*covariance).tolist(), precision=np.linalg.inv(2*covariance).tolist(),
                    numerical_floor=floor, tau=float(np.sqrt(chi2.ppf(model["quantile"], Y.shape[1]))))
    model["model_id"] = hashlib.sha256(json.dumps(model, sort_keys=True).encode()).hexdigest()
    return model


def transformed(row, name, model):
    x = modality_vector(row, name)
    item = model.get("modalities", {}).get(name, {})
    if x is None or item.get("status") != "ready" or len(x) != item["input_dimension"]:
        return None
    x = x-np.asarray(item["pca_mean"])
    if item["pca_components"] is not None:
        x = x@np.asarray(item["pca_components"]).T
    return x


def modality_distance(a, b, model, name, modalities=None):
    x, y = transformed(a, name, model), transformed(b, name, model)
    if x is None or y is None:
        return math.inf
    delta = x-y; item = model["modalities"][name]
    return float(np.sqrt(max(0., delta @ np.asarray(item["precision"]) @ delta)) / threshold(model,name,modalities))


def noise_distance(a, b, model, modalities=None):
    names = list(MODALITIES if modalities is None else modalities)
    if not names:
        raise ValueError("Distance requires at least one observation modality")
    values = [modality_distance(a, b, model, name,names)*(model.get("w_vis", .5) if name=="vis" else 1.)
              for name in names]
    return max(values)


def distance(a, b, model, config=None):
    return noise_distance(a, b, model, (config or {}).get("modalities"))


def threshold(model,name,modalities=None):
    item=model['modalities'][name]
    if model.get('schema')!='state_graph_noise_v2':return item['tau']
    from scipy.stats import chi2
    names=modalities if modalities is not None else model['distance_modalities']
    return float(np.sqrt(chi2.ppf(1-model.get('alpha',.05)/len(names),item['df'])))
