"""Read-only extraction of stable contact windows from historical E1/E2 data.

JSONL is the canonical nested representation. Parquet preserves numeric vector
columns; heterogeneous provenance dictionaries are JSON-encoded columns.
No SDK/robot code is imported, no planned target is used as measured position.
"""
from __future__ import annotations
import csv
import glob
import hashlib
import json
from collections import Counter
from pathlib import Path
import h5py
import numpy as np
from bottle_scan.analysis import _stream, _read_reference
from bottle_scan.contact_policy import feature_vector
from .contact_geometry import window_contact_geometry, contact_mean_diff, object_contact_position

DEFAULT_SOURCES = [
    "outputs/white_a_current_x5_20261110_fast",
    "outputs/white_a_director_live_0330_20261110/batches/001/A",
    "outputs/white_b_current_x5_dev_20261109",
    "outputs/e1e2_A_formal01",
    "outputs/e1e2_B_formal01",
    "outputs/rsi_contact01/batches/*",
]
def _clean(v):
    if isinstance(v,np.ndarray):return _clean(v.tolist())
    if isinstance(v,np.generic):return _clean(v.item())
    if isinstance(v,float) and not np.isfinite(v):return None
    if isinstance(v,dict):return {str(k):_clean(x) for k,x in v.items()}
    if isinstance(v,(tuple,list)):return [_clean(x) for x in v]
    return v

def _hash(value):
    return hashlib.sha256(json.dumps(_clean(value),sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()).hexdigest()

def _file_hash(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for part in iter(lambda:f.read(1024*1024),b""):h.update(part)
    return h.hexdigest()

def probe_windows(trial):
    """Order actual metadata observations, deduplicating final aliases by time."""
    candidates=[("probe_initial",trial.get("probe_initial"))]
    for i,r in enumerate(trial.get("probe_regrasps") or []):
        candidates.append((f"probe_regrasps/{i}/probe",r.get("probe")))
    candidates.append(("probe_final",trial.get("probe_final")))
    unique={}
    for ref,p in candidates:
        if isinstance(p,dict) and p.get("valid") and p.get("timestamp_ns"):
            t=int(p["timestamp_ns"])
            if t not in unique:unique[t]=(ref,p)
    return [unique[t] for t in sorted(unique)]

def downstream_outcome(trial,terminal):
    """Terminal downstream outcome only; absence of a test is not failure."""
    fields={k:trial.get(k) for k in ("status","target_reached","lifted","hold_complete","dropped","label")}
    stages={s.get("name") for s in trial.get("stages",[]) if isinstance(s,dict)}
    if not terminal:
        return None,"not_terminal_window",{"evaluated":False,"reason":"later_contact_window_exists"}
    attempted=bool("SQUEEZE" in stages or trial.get("target_reached") or trial.get("lifted"))
    if not attempted:
        return None,"downstream_not_tested",{"evaluated":False,**fields}
    passed=all(trial.get(k) is True for k in ("target_reached","lifted","hold_complete")) and trial.get("dropped") is False and trial.get("label")=="normal"
    if trial.get("label") is None and all(trial.get(k) is True for k in ("target_reached","lifted","hold_complete")) and not trial.get("dropped"):
        return None,"missing_normal_label",{"evaluated":True,**fields}
    return bool(passed),trial.get("status","unknown"),{"evaluated":True,**fields}

def measured_offsets(tcp,T0,axes):
    tcp=np.asarray(tcp,float);T0=np.asarray(T0,float)
    if tcp.shape!=(4,4) or T0.shape!=(4,4) or not np.isfinite(tcp).all() or not np.isfinite(T0).all():
        return {"a":None,"l":None}
    delta=tcp[:3,3]-T0[:3,3];result={}
    for name in ("a","l"):
        vec=axes.get(name)
        if vec is None:result[name]=None;continue
        unit=T0[:3,:3]@np.asarray(vec,float)
        result[name]=float(delta@unit/np.linalg.norm(unit))
    return result

def _sources(sources):
    found={}
    for source in sources:
        pattern=str(source.get("path")) if isinstance(source,dict) else str(source)
        for match in sorted(glob.glob(pattern)):
            p=Path(match).resolve()
            if p.is_file() and p.name=="session_meta.json":p=p.parent
            if (p/"session_meta.json").exists():found[str(p)]=source if isinstance(source,dict) else {}
            else:
                for m in sorted(p.glob("**/session_meta.json")):found[str(m.parent.resolve())]=source if isinstance(source,dict) else {}
    return [(Path(k),v) for k,v in sorted(found.items())]

def _reference_array(path,side):
    if path.suffix==".npz":
        with np.load(path,allow_pickle=False) as z:
            if side in z:return np.asarray(z[side])
    return _read_reference(path,side)

def _zero_reference(session,trial):
    z=trial.get("zero") or {};hist=session.get("zero_history") or []
    if z.get("dd_reference_file"):return Path(z["dd_reference_file"]),z
    matched=[h for h in hist if z.get("created_unix_ns") is not None and h.get("created_unix_ns")==z["created_unix_ns"]]
    if not matched and z.get("offset_raw") is not None:
        matched=[h for h in hist if h.get("offset_raw")==z["offset_raw"]]
    if len(matched)==1 and matched[0].get("dd_reference_file"):return Path(matched[0]["dd_reference_file"]),matched[0]
    raise ValueError("unmatched_unloaded_dd_reference")

def _window_indices(stream,desired,start,end,n=5):
    if stream is None:raise ValueError("missing_stream")
    times=stream["times"]
    if desired:
        indices=[]
        for stamp in desired:
            loc=int(np.searchsorted(times,int(stamp)))
            if loc>=len(times) or int(times[loc])!=int(stamp):
                raise ValueError("recorded_probe_frame_not_in_stream")
            indices.append(loc)
        indices=list(dict.fromkeys(indices))
    else:
        indices=np.flatnonzero((times>=start)&(times<=end)).tolist()
    if len(indices)<n:raise ValueError("fewer_than_five_independent_dd_frames")
    selected=np.linspace(0,len(indices)-1,n,dtype=int)
    return [indices[i] for i in selected]

def _median_stream(stream,start,end,expected=None):
    if stream is None:return None,None
    idx=np.flatnonzero((stream["times"]>=start)&(stream["times"]<=end))
    if not len(idx):return None,None
    data=np.asarray(stream["data"][idx],float)
    if expected and data.shape[1:]!=expected:raise ValueError("invalid_stream_shape")
    if not np.isfinite(data).all():raise ValueError("nonfinite_stream_window")
    return np.median(data,axis=0),{"indices":[int(idx[0]),int(idx[-1])],"count":len(idx),"time_ns":[int(stream["times"][idx[0]]),int(stream["times"][idx[-1]])]}

def _measured_tcp(stream,probe,start,end):
    # Closest actually recorded pose to the midpoint, never command stream.
    if stream is not None:
        idx=np.flatnonzero((stream["times"]>=start)&(stream["times"]<=end))
        if len(idx):
            k=int(idx[np.argmin(abs(stream["times"][idx]-(start+end)//2))])
            value=np.asarray(stream["data"][k],float)
            if value.shape==(4,4) and np.isfinite(value).all():
                return value,{"source":"streams/tcp_measured/data","index":k,"timestamp_ns":int(stream["times"][k])}
    value=np.asarray(probe.get("measured_tcp_mm"),float)
    if value.shape==(4,4) and np.isfinite(value).all() and probe.get("tcp_sample_timestamp_ns") is not None:
        return value,{"source":"probe.measured_tcp_mm","timestamp_ns":probe["tcp_sample_timestamp_ns"]}
    return None,{"source":None,"reason":"no_measured_tcp_in_window"}

def _wrist_frame(trial_dir,midpoint,camera="right_wrist"):
    index=trial_dir/"images"/"timestamps.csv"
    if not index.exists():return None
    with index.open() as f:
        rows=[r for r in csv.DictReader(f) if r.get("camera")==camera and r.get("monotonic_ns")]
    if not rows:return None
    r=min(rows,key=lambda r:abs(int(r["monotonic_ns"])-midpoint))
    path=trial_dir/"images"/r["path"]
    if not path.exists():return None
    return {"path":str(path.resolve()),"encoding":"jpeg","camera":camera,
        "timestamp_ns":int(r["monotonic_ns"]),"age_from_midpoint_ns":abs(int(r["monotonic_ns"])-midpoint),
        "association":"nearest_recorded_frame","index_path":str(index),"sha256":_file_hash(path)}

def _roles(path,session,source):
    role=source.get("role") or (session.get("agentic") or {}).get("role") or (session.get("agentic_context") or {}).get("batch_role")
    cond=session.get("cond")
    if role is None:role="formal_validation" if path.name=="e1e2_B_formal01" else "historical_exploration" if cond=="A" else "unspecified"
    evaluation=bool(source.get("evaluation_only")) or role in ("validation","frozen_validation","final_validation","independent","formal_validation") or "independent" in role
    return role,evaluation

def _object_id(path,trial,source):
    if source.get("object_id"):return source["object_id"]
    color=source.get("object_family") or ("white" if "white" in str(path) else "green")
    bottle=trial.get("bottle")
    return f"{color}_{bottle}" if bottle in ("light","heavy") else None

def _protocol(cfg):
    force=cfg.get("force") or {};grip=cfg.get("gripper") or {};tac=cfg.get("tactile") or {}
    return {"downstream_controller":cfg.get("downstream_controller","e1e2_squeeze"),"downstream_model_id":cfg.get("downstream_model_id"),"closure_floor":grip.get("min_normalized"),"probe_force_sdk":float(force.get("target",5))*float(force.get("probe_fraction",.3)),
        "probe_tolerance_sdk":force.get("probe_tolerance"),"target_force_sdk":force.get("target"),
        "probe_stable_s":force.get("probe_stable_s"),"depth_threshold":tac.get("dd_threshold"),
        "normal_axis":force.get("normal_axis"),"normal_signs":force.get("normal_signs")}

def _root_edges(session,config):
    edges={};evidence={}
    for finger in (session.get("fit") or {}).get("fingers",[]):
        side=finger.get("finger")
        if finger.get("root_edge") in ("left","right","top","bottom"):
            edges[side]=finger["root_edge"];evidence[side]="session_meta.fit.fingers.root_edge"
    for side,value in (config.get("root_edges") or {}).items():
        edges[side]=value;evidence[side]="extraction_config.root_edges"
    return edges,evidence

def _receipt(trial,window_index,before,after,meta_path):
    applied=[(i,x) for i,x in enumerate(trial.get("correction_history") or []) if isinstance(x,dict) and x.get("applied_delta_mm") is not None and not x.get("blocked")]
    if window_index<=0:return None
    between=[(i,x) for i,x in applied if x.get("timestamp_ns") is not None and before["t_ns"]<=int(x["timestamp_ns"])<after["t_ns"]]
    if len(between)!=1:return None
    i,action=between[0]
    stages=[s for s in trial.get("stages",[]) if s.get("name") in ("REGRASP_RETREAT","REGRASP_CORRECT","REGRASP_APPROACH") and
            s.get("start_ns",0)>=before["t_ns"] and s.get("end_ns",float("inf"))<=after["t_ns"]]
    if sum(s.get("name")=="REGRASP_CORRECT" for s in stages)!=1 or before.get("tcp_offset_a") is None or after.get("tcp_offset_a") is None:return None
    stamp=action.get("timestamp_ns")
    if stamp is not None and not before["t_ns"]<=stamp<after["t_ns"]:return None
    return {"executed":True,"source":str(meta_path),"metadata_pointer":f"/correction_history/{i}",
        "action_record":action,"stage_receipts":stages,
        "before_tcp_receipt":before["tcp_receipt"],"after_tcp_receipt":after["tcp_receipt"],
        "measured_delta_a_mm":after["tcp_offset_a"]-before["tcp_offset_a"],
        "applied_delta_mm":float(action["applied_delta_mm"]),"fresh_post_probe":True}

def write_nodes(rows,out_dir):
    out=Path(out_dir);out.mkdir(parents=True,exist_ok=True)
    with (out/"nodes.jsonl").open("w") as f:
        for row in rows:f.write(json.dumps(_clean(row),ensure_ascii=False,allow_nan=False)+"\n")
    import pyarrow as pa
    import pyarrow.parquet as pq
    nested={k for row in rows for k,v in row.items() if isinstance(v,dict) or isinstance(v,list) and any(isinstance(x,dict) for x in v)}
    packed=[{k:json.dumps(_clean(v),ensure_ascii=False,sort_keys=True) if k in nested and v is not None else _clean(v) for k,v in r.items()} for r in rows]
    table=pa.Table.from_pylist(packed)
    metadata=dict(table.schema.metadata or {})
    metadata[b"json_encoded_columns"]=json.dumps(sorted(nested)).encode()
    metadata[b"schema_version"]=b"symbolic_contact_nodes_v1"
    pq.write_table(table.replace_schema_metadata(metadata),out/"nodes.parquet")
    return {"json_columns":sorted(nested),"rows":len(rows)}

def extract_sessions(sources=None,out_dir="outputs/state_graph/20261112_symbolic_v1",config=None):
    config=config or {};rows=[];excluded=[];source_report=[];missing=Counter()
    discovered=_sources(DEFAULT_SOURCES if sources is None else sources)
    # A later fit can calibrate edge orientation for its explicitly named source
    # sessions. This is a pixel-edge convention, never a physical mm pitch.
    root_registry={}
    for fit_dir,_ in discovered:
        fit_meta=fit_dir/"session_meta.json"
        fit_session=json.loads(fit_meta.read_text())
        fit=fit_session.get("fit") or {}
        if not isinstance(fit,dict):continue
        fitted={f.get("finger"):f.get("root_edge") for f in fit.get("fingers",[]) if f.get("root_edge") in ("left","right","top","bottom")}
        if not fitted:continue
        for named in fit.get("source_sessions",[]):
            named_path=Path(named)
            if not named_path.is_absolute():named_path=Path.cwd()/named_path
            key=str(named_path.resolve())
            root_registry.setdefault(key,[]).append({"edges":fitted,"source":str(fit_meta),"sha256":_file_hash(fit_meta)})
    out=Path(out_dir);out.mkdir(parents=True,exist_ok=True)
    for session_dir,source in discovered:
        meta_path=session_dir/"session_meta.json";session=json.loads(meta_path.read_text())
        cfg=session.get("config") or {};protocol=_protocol(cfg);protocol_id=_hash(protocol)[:20]
        role,evaluation=_roles(session_dir,session,source);cond=session.get("cond")
        session_hash=_file_hash(meta_path);root_edges,root_evidence=_root_edges(session,config)
        if not root_edges and str(session_dir) in root_registry:
            candidates=root_registry[str(session_dir)]
            signatures={_hash(c["edges"]) for c in candidates}
            if len(signatures)==1:
                root_edges=dict(candidates[0]["edges"])
                root_evidence={side:{"type":"fit_explicit_source_session","calibrations":candidates} for side in root_edges}
        source_stat={"session":str(session_dir),"session_meta_sha256":session_hash,"source_role":role,"evaluation_only":evaluation,
                     "trials":0,"nodes":0,"protocol":protocol,"protocol_id":protocol_id}
        for trial_path in sorted((session_dir/"trials").glob("*/meta.json")):
            source_stat["trials"]+=1;trial=json.loads(trial_path.read_text());trial_id=str(trial.get("id",trial_path.parent.name))
            windows=probe_windows(trial);trial_hash=_file_hash(trial_path)
            if not windows:
                excluded.append({"session":str(session_dir),"trial_id":trial_id,"reason":"no_valid_probe_window","status":trial.get("status")});continue
            h5path=trial_path.parent/"streams.h5"
            try:
                reference_path,zero=_zero_reference(session,trial)
                reference={s:_reference_array(reference_path,s) for s in ("left","right")}
                reference_hash=_file_hash(reference_path)
                handle=h5py.File(h5path,"r")
            except (OSError,ValueError,KeyError) as error:
                excluded.append({"session":str(session_dir),"trial_id":trial_id,"reason":str(error)});continue
            with handle:
                cache={}
                def stream(name):
                    if name not in cache:cache[name]=_stream(handle,name)
                    return cache[name]
                previous=None
                for wi,(probe_ref,probe) in enumerate(windows):
                    base={"session":str(session_dir),"trial_id":trial_id,"window_index":wi}
                    try:
                        end=int(probe["timestamp_ns"]);start=int(probe.get("stable_start_ns") or end-int(float((cfg.get("force") or {}).get("probe_stable_s",.2))*1e9))
                        if start>end:raise ValueError("invalid_probe_time_interval")
                        force=cfg.get("force") or {};tactile=cfg.get("tactile") or {};frames={};geometry={}
                        for si,side in enumerate(("left","right")):
                            window=(probe.get("window") or [{},{}])[si]
                            desired=window.get("frame_ns") or []
                            decoded=stream("dd_"+side);encoded=stream("dd_encoded_"+side)
                            di=_window_indices(decoded,desired,start,end);ei=_window_indices(encoded,desired,start,end)
                            raw=np.asarray(decoded["data"][di])
                            frames[side]={"h5_path":str(h5path.resolve()),"dataset":f"streams/dd_encoded_{side}/data",
                                "indices":ei,"times_ns":[int(encoded["times"][i]) for i in ei],
                                "decoded_dataset":f"streams/dd_{side}/data","decoded_indices":di,
                                "decoded_times_ns":[int(decoded["times"][i]) for i in di],
                                "decoded_frame_sha256":hashlib.sha256(raw.tobytes()).hexdigest()}
                            pitch=(config.get("pixel_pitch_mm") or {}).get(side) if isinstance(config.get("pixel_pitch_mm"),dict) else config.get("pixel_pitch_mm")
                            if pitch is None:pitch=tactile.get("pixel_pitch_mm")
                            geometry[side]=window_contact_geometry(raw,reference[side],
                                threshold=float(tactile.get("dd_threshold",.022)),depth_channel=int(tactile.get("depth_channel",0)),
                                pixel_pitch_mm=pitch,root_edge=root_edges.get(side),edge_width_px=int(tactile.get("edge_pixels",3)))
                        wrench=[];wrench_receipts={}
                        for side in ("left","right"):
                            v,receipt=_median_stream(stream("wrench_"+side),start,end,(6,))
                            if v is None:raise ValueError("no_zeroed_wrench_in_probe_window")
                            wrench.append(v.tolist());wrench_receipts[side]=receipt
                        axis=int(force.get("normal_axis",2));signs=force.get("normal_signs",[-1,-1])
                        normal=[float(wrench[i][axis]*signs[i]) for i in range(2)]
                        thresholds=zero.get("contact_thresholds_sdk") or [float(force.get("min_contact",.05))]*2
                        bilateral=all(normal[i]>=thresholds[i] for i in range(2))
                        if not bilateral:raise ValueError("not_bilateral_contact_at_recorded_threshold")
                        opening,opening_receipt=_median_stream(stream("gripper_normalized"),start,end)
                        if opening is None:raise ValueError("no_gripper_measurement_in_probe_window")
                        opening=float(np.asarray(opening).ravel()[0]);floor=protocol["closure_floor"]
                        tcp,tcp_receipt=_measured_tcp(stream("tcp_measured"),probe,start,end)
                        offsets=measured_offsets(tcp,session.get("T0"),cfg.get("axes_tcp") or {}) if tcp is not None else {"a":None,"l":None}
                        object_id=_object_id(session_dir,trial,source)
                        if object_id is None:raise ValueError("unknown_object_id")
                        gid=f"{session_dir}:{trial_id}"
                        node_id=_hash({"session":str(session_dir),"trial_id":trial_id,"probe_ns":end})[:24]
                        outcome,status,evidence=downstream_outcome(trial,wi==len(windows)-1)
                        mean,diff=contact_mean_diff(geometry["left"],geometry["right"])
                        target=protocol["probe_force_sdk"];tolerance=float(protocol["probe_tolerance_sdk"] or .25)
                        bins=["below" if f<target-tolerance else "above" if f>target+tolerance else "probe_band" for f in normal]
                        try:legacy=feature_vector(probe.get("descriptors")).tolist()
                        except (ValueError,TypeError):legacy=None
                        objpos={side:object_contact_position(geometry[side],tcp,(config.get("object_contact_calibration") or {}).get(side)) for side in ("left","right")}
                        row={**base,"node_id":node_id,"t_ns":end,"timestamp_ns":end,"window_start_ns":start,
                            "phase":"A" if cond=="A" else "B_pre" if wi==0 else "B_post","cond":cond,
                            "session_created_unix_ns":session.get("created_unix_ns"),"trial_started_unix_ns":trial.get("started_unix_ns"),
                            "time_basis":"t_ns is recorded host monotonic; chronology uses trial_started_unix_ns then window_index",
                            "source_role":role,"evaluation_only":evaluation,"group_id":gid,
                            "object_id":object_id,"protocol_id":protocol_id,"protocol":protocol,"closure_floor":floor,
                            "bilateral_contact":bilateral,"contact_thresholds_sdk":thresholds,
                            "gripper_state":"open" if opening>=float((cfg.get("gripper") or {}).get("open_normalized",1))-.002 else "near_floor" if floor is not None and opening-float(floor)<=.002 else "closing",
                            "edge_class":{s:geometry[s]["edge_class"] for s in ("left","right")},"root_edge_evidence":root_evidence,
                            "force_bin":bins,"normal_sdk":normal,"encoder_version":None,
                            "sensor_contact_left":geometry["left"],"sensor_contact_right":geometry["right"],
                            "sensor_contact_mean":mean,"sensor_contact_diff":diff,
                            "wrench_left":wrench[0],"wrench_right":wrench[1],"wrench_units":"SDK; independent physical calibration not established",
                            "wrench_receipts":wrench_receipts,"gripper_opening":opening,"gripper_receipt":opening_receipt,
                            "tcp_offset_a":offsets["a"],"tcp_offset_l":offsets["l"],"measured_tcp_mm":None if tcp is None else tcp.tolist(),"tcp_receipt":tcp_receipt,
                            "object_contact_h":{s:objpos[s]["h_mm"] for s in objpos},"object_contact_theta":{s:objpos[s]["theta_rad"] for s in objpos},
                            "object_contact_evidence":objpos,"planned_delta_mm":trial.get("delta_mm"),
                            "zero_monitor":trial.get("kind") in ("zero","monitor","zero_monitor") or trial.get("delta_mm")==0,
                            "trial_kind":trial.get("kind"),"repeat":trial.get("repeat"),
                            "rule_id":trial.get("rule_id") or (session.get("agentic") or {}).get("rule_id") or "historical:"+protocol_id+":"+_hash(str(session_dir))[:8],
                            "downstream_success":outcome,"downstream_status":status,"downstream_evidence":evidence,
                            "original_label":trial.get("label"),"original_skill_ground_truth":trial.get("skill_ground_truth"),
                            "original_skill_window_labels":trial.get("skill_window_labels"),"legacy_features":legacy,
                            "frames":frames,"wrist_frame":_wrist_frame(trial_path.parent,(start+end)//2),
                            "unloaded_reference":{"path":str(reference_path),"sha256":reference_hash,"created_unix_ns":zero.get("created_unix_ns")},
                            "provenance":{"meta_path":str(trial_path),"metadata_sha256":trial_hash,"session_meta_path":str(meta_path),"session_meta_sha256":session_hash,
                                "probe_pointer":probe_ref,"h5_path":str(h5path),"h5_bytes":h5path.stat().st_size,"h5_mtime_ns":h5path.stat().st_mtime_ns,
                                "hash_scope":"metadata/reference/full selected decoded frames; not full H5"},
                            "previous_node_id":None,"executed_delta_a_mm":None,"execution_receipt":None}
                        if previous is not None:
                            receipt=_receipt(trial,wi,previous,row,trial_path)
                            if receipt:
                                row.update(previous_node_id=previous["node_id"],executed_delta_a_mm=receipt["applied_delta_mm"],execution_receipt=receipt)
                        for name,value in (("physical_pixel_pitch",geometry["left"]["u_mm"]),("root_edge",geometry["left"]["edge_class"]),("measured_tcp",offsets["a"]),("wrist_frame",row["wrist_frame"]),("legacy_features",legacy)):
                            if value is None:missing[name]+=1
                        missing["object_contact_calibration"]+=int(any(x["h_mm"] is None for x in objpos.values()))
                        rows.append(_clean(row));source_stat["nodes"]+=1;previous=row
                    except (OSError,ValueError,KeyError,IndexError,TypeError) as error:
                        excluded.append({**base,"reason":str(error),"probe_pointer":probe_ref,"status":trial.get("status")})
                        previous=None
        source_report.append(source_stat)
        # Checkpoint extraction progress for independently running feature worker.
        with (out/"nodes_partial.jsonl").open("w") as f:
            for row in rows:f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+"\n")
    packed=write_nodes(rows,out)
    report={"schema":"symbolic_node_extraction_v1","nodes":len(rows),"trials":sum(s["trials"] for s in source_report),
        "sources":source_report,"excluded":excluded,"exclusion_counts":dict(Counter(x["reason"] for x in excluded)),
        "missing_calibration_or_modality":dict(missing),"parquet":packed,
        "limitations":["Unknown physical pixel pitch remains null; px fields are not mm.",
            "T0 alone does not calibrate bottle axis or TCP-to-sensor plane; object h/theta remain null without explicit transforms.",
            "Recorded SDK six-axis force is software-zeroed but not an independent force label.",
            "Formal/independent/validation data retain evaluation_only and must not train graph labels.",
            "Downstream outcomes belong only to terminal contact window, not preceding B windows.",
            "No entire H5 file hashes computed; metadata and selected window arrays are hashed.",
            "Frozen protocol hash includes closure floor, probe/target forces, tolerance, stability and tactile threshold."]}
    (out/"extraction_report.json").write_text(json.dumps(_clean(report),ensure_ascii=False,indent=2,allow_nan=False))
    return rows

if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--out",default="outputs/state_graph/20261112_symbolic_v1");p.add_argument("--source",action="append")
    args=p.parse_args();result=extract_sessions(args.source,args.out);print(json.dumps({"nodes":len(result),"out":args.out}))
