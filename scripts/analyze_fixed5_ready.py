"""Read-only replay of fixed-squeeze outcomes and pre-squeeze Ready features.

Run from the repository root using .venv/bin/python scripts/analyze_fixed5_ready.py
--out <report-directory>. The active campaign and robot are never opened for writing.
"""
import argparse
import csv
import json
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / "outputs/state_graph/campaign_white_a_v4_fixed5_history"
HISTORY = ROOT / "outputs/white_a_director_live_0330_20261110/batches"
ARCHIVES = {
    "historic_0330_1": HISTORY / "001/A",
    "historic_0330_2": HISTORY / "002/A",
    "earlier_0336_1": ROOT / "outputs/white_a_current_x5_20261109",
    "earlier_0336_2": ROOT / "outputs/white_a_current_x5_20261110",
    "earlier_0336_3": ROOT / "outputs/white_a_current_x5_20261110_fast",
}
GEOM = [f"{side}_{field}" for side in ("left", "right") for field in ("u", "v", "log_area")]
PHYSICAL = GEOM + ["opening", "normal_left", "normal_right"]
MODELS = {"contact": PHYSICAL, "geometry_only": GEOM, "opening_only": ["opening"],
          "contact_and_tcp": PHYSICAL + ["a_mm"]}
TOLS = [0.5, 0.6, 0.75, 1.0]


def read(path):
    return json.loads(path.read_text())


def stream(h, name):
    g = h["streams"][name]
    n = int(g.attrs.get("committed_count", len(g["data"])))
    t = g["monotonic_ns"][:n]
    assert len(t) and np.all(np.diff(t) >= 0), name
    return t, g["data"][:n]


def at(t, v, timestamp):
    idx = np.searchsorted(t, timestamp, side="right") - 1
    if idx < 0 or timestamp - t[idx] > 250_000_000:
        raise ValueError("Unavailable fresh pre-PROBE-end observation")
    return v[idx]


def pre_features(probe, pose, opening, t0):
    # Both generations use the completed PROBE's two describe() outputs.
    out = {"opening": float(opening), "normal_left": probe["normal_sdk"][0],
           "normal_right": probe["normal_sdk"][1],
           "a_mm": float((pose[:3, 3] - t0[:3, 3]) @ t0[:3, 2])}
    assert probe["valid"] and len(probe["descriptors"]) == 2
    for side, d in zip(("left", "right"), probe["descriptors"]):
        assert d["valid"] and d["area_px"] > 0 and list(d["shape_hw"]) == [96, 128]
        u, v = d["centroid_px"]
        out.update({f"{side}_u": float(u), f"{side}_v": float(v),
                    f"{side}_area": d["area_px"], f"{side}_log_area": float(np.log(d["area_px"]))})
    assert np.isfinite(list(out.values())).all()
    return out


def longest_run(t, ok):
    start = previous = None
    longest = 0.
    for now, valid in zip(t, ok):
        if not valid:
            start = None
        elif start is None or now - previous > 100_000_000:
            start = now
        else:
            longest = max(longest, (now - start) * 1e-9)
        previous = now
    return longest


def describe_trial(path, meta, t0, thresholds, probe, first, row):
    with h5py.File(path, "r") as h:
        tw, width = stream(h, "gripper_normalized")
        tp, pose = stream(h, "tcp_measured")
        for prefix, p in [("", probe), ("initial_", first)]:
            stamp = p["timestamp_ns"]
            pre = pre_features(p, at(tp, pose, stamp), at(tw, width, stamp), np.array(t0))
            row.update({prefix + k: v for k, v in pre.items()})
        stage, = [s for s in meta["stages"] if s["name"] == "SQUEEZE"]
        start, end = stage["start_ns"], stage["end_ns"]
        assert first["timestamp_ns"] <= probe["timestamp_ns"] < start <= end
        tf, normals = stream(h, "normal_force_sdk")
    i = (tf >= start) & (tf <= end)
    t, normals = tf[i], normals[i]
    mean = normals.mean(1)
    wi = (tw >= start) & (tw <= end)
    closed = width[wi]
    row.update(h5=str(path), probe_ns=probe["timestamp_ns"], squeeze_s=(end-start)*1e-9,
               end_opening=float(np.median(width[wi & (tw >= end-1_000_000_000)])),
               min_opening=float(min(closed)), peak_mean=float(max(mean)),
               end_force=float(np.median(mean[t >= end-1_000_000_000])))
    # A trace sensitivity diagnostic. Original trials retain their actual controller receipts.
    for tol in TOLS:
        ok = (abs(mean-5.) <= tol) & (normals >= np.asarray(thresholds)).all(axis=1)
        row[f"band_{tol}"] = longest_run(t, ok)
    curve = {"id": row["id"], "elapsed_s": ((t-start)*1e-9).tolist(),
             "left": normals[:, 0].tolist(), "right": normals[:, 1].tolist(),
             "width_s": ((tw[wi]-start)*1e-9).tolist(), "width": closed.tolist()}
    return row, curve


def check_protocol(cfg):
    assert cfg["force"]["target"] == 5. and cfg["force"]["unit"] == "sdk"
    assert cfg["force"]["squeeze_tolerance"] == .5 and cfg["force"]["squeeze_stable_s"] == .5
    assert cfg["motion"]["lift_mm"] == 40 and cfg["hold_s"] == 2
    assert cfg["axes_tcp"]["a"] == [0, 0, 1]


def collect():
    records, curves, excluded = [], [], []
    for batch, folder in ARCHIVES.items():
        session = read(folder/"session_meta.json")
        assert session["simulated"] is False
        check_protocol(session["config"])
        floor = session["config"]["gripper"]["min_normalized"]
        for p in sorted((folder/"trials").glob("*/meta.json")):
            m = read(p)
            base = dict(id=f"{batch}_{p.parent.name}", batch=batch, mass=m["bottle"],
                        setting=m["delta_mm"], floor=floor, corrected=False)
            passed = m.get("status") == "hold_complete" and m.get("lifted") and m.get("target_reached") and m.get("hold_complete") and m.get("dropped") is False
            failed = m.get("status") == "force_not_reached" and m.get("target_reached") is False and not m.get("lifted")
            if not (m.get("complete") and m.get("recording_complete") and (passed or failed)):
                excluded.append({**base, "why":m.get("status")}); continue
            first, probe = m["probe_initial"], m["probe_final"]
            assert not m["regrasps"] and first["timestamp_ns"] == probe["timestamp_ns"]
            row, curve = describe_trial(p.with_name("streams.h5"), m, session["T0"], m["zero"]["contact_thresholds_sdk"],
                                        probe, first, {**base, "success": bool(passed)})
            records.append(row); curves.append(curve)
    state = read(CAMPAIGN/"state.json")
    for trial in state["trials"]:
        batch = "batch_" + trial["trial_id"].split("_")[1]
        if batch not in ("batch_3","batch_4","batch_5","batch_6") or trial["status"] != "complete":
            continue
        folder = Path(trial["directory"])
        req, res = trial["request"], read(folder/"result.json")
        base = dict(id=trial["trial_id"], batch=batch, mass=req["block"]["mass_class"],
                    setting=req["settings"]["a_offset_mm"],floor=req["settings"]["closure_floor"])
        terminal = [n for n in res["nodes"] if n["post"][req["post_condition"]]["evaluated"]]
        if not terminal:
            excluded.append({**base, "why":res.get("attempt_reason") or res.get("unassessed_reason")}); continue
        node, = terminal
        events_path, = folder.glob("episodes/*/events.jsonl")
        events = [json.loads(l) for l in events_path.read_text().splitlines() if l.strip()]
        windows = {e["node_id"]:e for e in events if e["event"]=="state_graph_window"}
        first, = [e for e in windows.values() if e["window_index"]==0]
        final = windows[node["node_id"]]
        decisions = [e["skill_outputs"] for e in events if e["event"]=="recovery_skill_decision"]
        p = Path(final["raw_evidence"]["streams"])
        session = read(p.with_name("session_meta.json"))
        check_protocol(session["config"])
        assert req["post_condition"] == "fixed_squeeze_5sdk_lift_floor0330" and base["floor"] == .330
        prep, = events_path.parent.glob("preparation_*.json")
        row,curve = describe_trial(p, read(p.with_name("meta.json")),session["T0"],
                                   read(prep)["zero"]["contact_thresholds_sdk"], final["probe"], first["probe"],
                                   {**base,"success":node["post"][req["post_condition"]]["success"],
                                    "corrected":final["window_index"]>0,
                                    "old_ready":decisions[-1]["invocation"]["status"]=="ready" if req["mode"]=="learned" else None})
        records.append(row); curves.append(curve)
    assert len({r["probe_ns"] for r in records}) == len(records)
    return records,curves,excluded


def matrix(rows, cols):
    return np.array([[r[c] for c in cols] for r in rows])


def score(y, pred):
    y=np.asarray(y,bool); pred=np.asarray(pred,bool)
    return {"correct_ready":int(sum(y & pred)),"false_ready":int(sum(~y & pred)),
            "missed_ready":int(sum(y & ~pred)),"correct_rejection":int(sum(~y & ~pred))}


def fit_head(train,cols):
    y=[r["success"] for r in train]
    if len(set(y))<2:
        return None
    pipe=make_pipeline(StandardScaler(),LogisticRegression(C=1.,class_weight="balanced",max_iter=1000))
    pipe.fit(matrix(train,cols),y)
    return pipe


def evaluate(rows):
    main=[r for r in rows if r["floor"]==.330]
    early=[r for r in main if r["batch"]!="batch_6" and not r["corrected"]]
    hold=[r for r in main if r["batch"]=="batch_6" and not r["corrected"]]
    assert all(r["old_ready"] for r in hold)
    results={}
    scores=[]
    def run(name, train, test, cols):
        fitted=fit_head(train,cols)
        if fitted is None:
            results[name]={"unavailable":"single training class"};return
        p=fitted.predict_proba(matrix(test,cols))[:,1]
        y=[r["success"] for r in test]
        results[name]={**score(y,p>=.5),"n_train":len(train),"n_test":len(y),
                       "auc":float(roc_auc_score(y,p)) if len(set(y))>1 else None}
        scores.extend({"split":name,"id":r["id"],"success":r["success"],"p":float(prob)}
                      for r,prob in zip(test,p))
    for name,cols in MODELS.items():
        run("direct_"+name,early,hold,cols)
    results["deployed_direct"]=score([r["success"] for r in hold],[r["old_ready"] for r in hold])
    expanded=[r for r in main if r["batch"]!="batch_6"]
    run("terminal_training_contact",expanded,hold,PHYSICAL)
    run("corrected_endpoint_separate",expanded,[r for r in main if r["batch"]=="batch_6" and r["corrected"]],PHYSICAL)
    # Train on known stricter protocol only as a sensitivity comparison.
    run("transfer_0336", [r for r in rows if r["floor"]==.336],hold,PHYSICAL)
    for b in sorted({r["batch"] for r in early}):
        run("leave_out_"+b,[r for r in early if r["batch"]!=b],[r for r in early if r["batch"]==b],PHYSICAL)
    recent=[r for r in early if r["batch"].startswith("batch_")]
    run("leave_out_both_archives",recent,[r for r in early if r not in recent],PHYSICAL)
    for b in sorted({r["batch"] for r in early}):
        run("sensitivity_remove_"+b,[r for r in early if r["batch"]!=b],hold,PHYSICAL)
    fitted=fit_head(early,PHYSICAL)
    scaler,clf=fitted.steps[0][1],fitted.steps[1][1]
    candidate={"intended_use":"offline fixed5 endpoint score; low score means unconfirmed Ready",
               "training_ids":[r["id"] for r in early],"features":PHYSICAL,
               "mean":scaler.mean_.tolist(),"scale":scaler.scale_.tolist(),
               "coef":clf.coef_[0].tolist(),"intercept":float(clf.intercept_[0])}
    probs=fitted.predict_proba(matrix(hold,PHYSICAL))[:,1]
    z=(matrix(hold,PHYSICAL)-scaler.mean_)/scaler.scale_
    assert np.allclose(probs,1/(1+np.exp(-(z@clf.coef_[0]+clf.intercept_[0]))))
    threshold=read(CAMPAIGN/"spec.json")["post_conditions"][0]["ready_p"]
    candidate["reference_threshold"]=threshold
    results["reference_gate_heldout"]=score([r["success"] for r in hold],probs>=threshold)
    return results,scores,candidate


def band_summary(rows):
    output={}
    for floor in sorted({r["floor"] for r in rows}):
        part=[r for r in rows if r["floor"]==floor]
        for batchset,subset in [("all",part),("live",[r for r in part if r["batch"].startswith("batch_")])]:
            if not subset:continue
            output[str(floor)+"_"+batchset]={str(tol):{
                str(truth):[int(sum(r[f"band_{tol}"]>=.5 for r in subset if r["success"]==truth)),
                            sum(r["success"]==truth for r in subset)] for truth in (True,False)} for tol in TOLS}
    return output


def compare_pairs(records):
    """Compare opposite outcomes from equal offsets using the actual pre-B graph metric."""
    import sys
    if str(ROOT) not in sys.path:
        sys.path.insert(0,str(ROOT))
    from bottle_scan.state_graph.noise import measurement_distance
    rev=CAMPAIGN/"revisions/0007"
    noise=read(rev/"noise_model.json")
    held=[r for r in records if r["batch"]=="batch_6" and not r["corrected"]]
    nodes={}
    for r in held:
        n,=read(CAMPAIGN/"trials"/r["id"]/"result.json")["nodes"]
        assert n["node_id"] not in noise["fit_node_ids"]
        nodes[r["id"]]=n
    pairs=[]
    for yes in held:
        if not yes["success"]:continue
        for no in held:
            if no["success"] or yes["setting"]!=no["setting"]:continue
            pairs.append({"success_id":yes["id"],"failure_id":no["id"],
                          "graph_distance":measurement_distance(nodes[yes["id"]],nodes[no["id"]],noise),
                          "opening_delta":yes["opening"]-no["opening"],
                          "left_centroid_distance_px":float(np.hypot(yes["left_u"]-no["left_u"],yes["left_v"]-no["left_v"])),
                          "right_centroid_distance_px":float(np.hypot(yes["right_u"]-no["right_u"],yes["right_v"]-no["right_v"]))})
    return sorted(pairs,key=lambda p:p["graph_distance"])


def write_csv(path, rows):
    cols=list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=cols);w.writeheader();w.writerows(rows)


def plot(rows,curves,out):
    by_id={c["id"]:c for c in curves}
    fig,ax=plt.subplots(2,2,figsize=(11,7.8),constrained_layout=True)
    selected=[r for r in rows if r["batch"]=="batch_6" and not r["corrected"]]
    colors={"05":"#aa2f2b","07":"#126080","08":"#d77f21",
            "09":"#3194ad","11":"#198474","12":"#963c65"}
    for row in selected:
        color=colors.get(row["id"][-2:], "#166f81" if row["success"] else "#bd442c")
        label=row["id"][-2:]+" "+row["mass"]+" "+("pass" if row["success"] else "fail")
        marker="o" if row["mass"]=="light" else "^"
        for axis,side in zip(ax[0],["left","right"]):
            axis.scatter(row[side+"_u"],row[side+"_v"],s=row[side+"_area"]/16,
                         color=color,alpha=.56,marker=marker,edgecolors="black",linewidths=.6)
            axis.annotate(row["id"][-2:],(row[side+"_u"],row[side+"_v"]),xytext=(3,5),textcoords="offset points",fontsize=8)
        c=by_id[row["id"]];t=np.asarray(c["elapsed_s"])
        y=(np.array(c["left"])+np.array(c["right"]))/2
        ax[1,0].plot(t,y,color=color,alpha=.65,lw=.85,label=label)
        ax[1,1].plot(c["width_s"],c["width"],color=color,alpha=.65,lw=.9)
    for axis,side in zip(ax[0],["Left","Right"]):
        axis.set_title(side+" pad at first contact");axis.set_xlabel("u (pixels)");axis.set_ylabel("v (pixels)")
        side_key=side.lower()
        x_center=np.mean([r[side_key+"_u"] for r in selected])
        y_center=np.mean([r[side_key+"_v"] for r in selected])
        axis.set_xlim(x_center-20,x_center+20)
        axis.set_ylim(y_center+10,y_center-10)
        axis.set_aspect("equal",adjustable="box")
    ax[1,0].axhspan(4.5,5.5,alpha=.14,color="green");ax[1,0].set_ylabel("Mean normal force (SDK)")
    ax[1,0].legend(fontsize=7,ncol=2);ax[1,0].set_title("Force after SQUEEZE starts")
    ax[1,1].axhline(.33,color="grey",ls="--");ax[1,1].set_ylabel("Normalized opening");ax[1,1].set_title("Measured opening")
    for axis in ax[1]:axis.set_xlabel("Time since SQUEEZE start (s)")
    fig.suptitle("Held-out Batch 6 | direct contacts only | marker area scales with contact area")
    fig.savefig(out/"heldout.png",dpi=160);plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--out",type=Path,required=True)
    out=parser.parse_args().out;out.mkdir(parents=True,exist_ok=True)
    rows,curves,excluded=collect()
    results,pred,candidate=evaluate(rows)
    (out/"ready_candidate.json").write_text(json.dumps(candidate,indent=2))
    write_csv(out/"trials.csv",rows);write_csv(out/"predictions.csv",pred)
    (out/"curves.json").write_text(json.dumps(curves,allow_nan=False))
    info={"scores":results,"bands":band_summary(rows),"excluded":excluded,"opposite_pairs":compare_pairs(rows)}
    (out/"summary.json").write_text(json.dumps(info,indent=2,allow_nan=False))
    plot(rows,curves,out)
    print(json.dumps(results,indent=2))

if __name__=="__main__":
    main()
