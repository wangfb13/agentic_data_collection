"""Offline Batch 6 modality comparison. Run from the experiment repo in the 5090 container."""
import json
from collections import Counter
from copy import deepcopy
from pathlib import Path

import joblib
import numpy as np

from bottle_scan.state_graph.common import group_id
from bottle_scan.state_graph.dynamics import fit_dynamics
from bottle_scan.state_graph.graph import build_graph
from bottle_scan.state_graph.labels import generate_labels
from bottle_scan.state_graph.noise import fit_noise
from bottle_scan.state_graph.skill import SymbolicRecoverySkill, train_models


campaign = Path("outputs/state_graph/campaign_white_a_v4_fixed5_history")
rev = campaign / "revisions/0007"


def read(path):
    return json.loads(path.read_text())


def project(row, modalities):
    return {**row, **{k: {m: row[k][m] for m in modalities}
                      for k in ("obs", "extractor_versions")}}


def train(train_rows, original_spec, modalities):
    spec = deepcopy(original_spec)
    original = spec["observation"]["modalities"]
    spec["observation"]["modalities"] = {m: original[m] for m in modalities}
    rows = [project(r, modalities) for r in train_rows]
    noise = fit_noise(rows, spec)
    assert set(noise["active_modalities"]) == set(modalities)
    dynamics = fit_dynamics(rows, noise, spec)
    graph = build_graph(rows, noise, spec, dynamics)
    labels = generate_labels(graph, rows, spec)
    bundle, status = train_models(rows, graph, labels, noise, dynamics, spec)
    assert status["active_model_trained"]
    return SymbolicRecoverySkill(bundle)


def prediction(p):
    return dict(state=p["state"], p_hat=p["value"]["p_hat"],
                invoke=p["value"]["invocation_probability"],
                exit=p["value"]["exit_probability"],
                action_mm=(p.get("action") or {}).get("u", [None])[0],
                reason=p.get("reason"))


def same_as_recorded(actual, recorded):
    return (actual["state"] == recorded["state"]
            and all(np.isclose(actual[k], recorded[k], rtol=0, atol=1e-7)
                    for k in ("p_hat", "invoke", "exit"))
            and (actual["action_mm"] == recorded["action_mm"] or
                 (None not in (actual["action_mm"], recorded["action_mm"])
                  and np.isclose(actual["action_mm"], recorded["action_mm"], rtol=0, atol=1e-7))))


def main():
    frozen = joblib.load(rev / "models/bundle.joblib")
    pool = read(campaign / "seed_nodes.json")
    for batch in range(1, 6):
        for path in sorted((campaign / "trials").glob(f"batch_{batch}_*/result.json")):
            pool.extend(read(path)["nodes"])
    by_id = {r["node_id"]: r for r in pool}
    assert len(by_id) == len(pool)
    rows = [by_id[s["node_id"]] for s in frozen["support"]]
    assert all(all(r[k] == v for k, v in s.items()) for r, s in zip(rows, frozen["support"]))
    train_ids = {r["node_id"] for r in rows}
    groups = {group_id(r) for r in rows}
    modes = {"with_wrist": ("tac", "force", "vis"), "without_wrist": ("tac", "force")}
    skills = {k: train(rows, frozen["spec"], m) for k, m in modes.items()}
    cid = frozen["spec"]["active_post_condition"]
    results = []
    for path in sorted((campaign / "trials").glob("batch_6_*/result.json")):
        record = read(path)
        request = read(path.parent / "request.json")
        all_files = list((path.parent / "episodes").glob("episode_*/events.jsonl"))
        assert len(all_files) == 1
        events = [json.loads(s) for s in all_files[0].read_text().splitlines() if s]
        windows = [e for e in events if e["event"] == "state_graph_window"]
        decisions = [e for e in events if e["event"] == "recovery_skill_decision"]
        nodes = {r["node_id"]: r for r in record["nodes"]}
        assert len(windows) == len(nodes) == len(decisions)
        trial = dict(trial=path.parent.name, mass=request["block"]["mass_class"],
                     planned_a=request["settings"]["a_offset_mm"], windows=[],
                     fault=[e.get("reason") for e in events if e["event"] == "fault"])
        for w, d in zip(windows, decisions):
            node = nodes[w["node_id"]]
            assert node["node_id"] not in train_ids and group_id(node) not in groups
            assert d["skill_outputs"]["graph_model_id"] == frozen["model_id"]
            x = {**node, **{k: w["live_features"][k] for k in ("obs", "extractor_versions")}}
            logged = prediction(d["skill_outputs"]["recovery"]["decision"]["prediction"])
            pred = {m: prediction(skills[m].infer(project(x, names))) for m, names in modes.items()}
            posts = [e for e in events if e["event"] == "state_graph_consumer_result"
                     and e["node_id"] == node["node_id"]]
            outcome = None
            if posts:
                assert len(posts) == 1 and posts[0]["is_terminal"]
                consumer = posts[0]["consumer_outcome"]
                assert consumer["controller"] == "e1e2_fixed_squeeze" and consumer["params"]["closure_floor"] == .33
                evidence = consumer["evidence"][0]
                outcome = bool(evidence["target_reached"] and evidence["lifted"]
                               and evidence["hold_complete"] and not evidence["dropped"])
            assert node["post"][cid]["success"] == outcome
            trial["windows"].append(dict(node_id=node["node_id"], phase=node["phase"],
                                         logged=logged, measured_action_mm=(w.get("u_receipt") or {}).get("action_mm"),
                                         replay=pred, same_as_field=bool(same_as_recorded(pred["with_wrist"], logged)),
                                         fixed5=outcome))
        results.append(trial)
    summary = {}
    for m in modes:
        first = [t["windows"][0] for t in results if t["windows"]]
        observed = [w for t in results for w in t["windows"] if w["fixed5"] is not None]
        def counts(items):
            return {k: dict(Counter(w["replay"][m]["state"] for w in items if w["fixed5"] is truth))
                    for k,truth in [("fixed5_success",True),("fixed5_failure",False)]}
        summary[m] = dict(first_states=dict(Counter(w["replay"][m]["state"] for w in first)),
                          first_with_outcome=counts(first), all_endpoint_windows=counts(observed))
    batch5 = read(campaign / "reviews/batch_5_f4ad8ab2.json")
    arms = [{k:a[k] for k in ("trial_id","source_matched","distance_to_source","requested_actions",
                              "measured_actions","consumer_success")} for a in batch5["arms"]]
    print(json.dumps(dict(batch6=results, summary=summary, batch5=arms, post_condition=cid,
                          model_id=frozen["model_id"], train_windows=len(rows), train_groups=len(groups),
                          train_node_ids=[r["node_id"] for r in rows], held_out_groups_disjoint=True,
                          train_graph_status={m:dict(Counter(s["status"] for s in skill.bundle["graph"]["reachability"][cid].values()))
                                              for m,skill in skills.items()}),
                     allow_nan=False, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
