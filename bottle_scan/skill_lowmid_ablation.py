"""Frozen low/mid-budget replay ablations. File-only, never imports robot hardware."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from .agentic_learning import _hash, _success
from .contact_policy import train_contact_policy, FINAL_ROLES
from .skill_offline import (discover, choose_batch, stage_training, geometry_evaluation,
    raw_prediction, write_json, provenance)
from .skill_repeated_benchmark import load_data, validate_split

MODES = ("fixed","simple_feedback","agent_evidence","offset_session_balanced",
         "no_reference","no_B_hard","no_coverage")
BUDGETS = (24,32,40,48)
COMPONENTS = {"agent_evidence":("reference","B_hard","coverage"),
    "no_reference":("B_hard","coverage"),"no_B_hard":("reference","coverage"),
    "no_coverage":("reference","B_hard")}

def canonical_metadata(pool):
    # Selection receives no tactile feature or measured supervision from candidates.
    return [{"trial_path":o["trial_path"],"planned_delta_mm":float(o["planned_delta_mm"]),
             "session":o["session"]} for o in pool]

def b_difficulty(feedback,model):
    result=[]
    for r in feedback:
        if r.get("kind")!="scan": continue
        weight=1.+(not _success(r))*2.
        if model and model.get("networks") and r.get("probe_initial",{}).get("valid"):
            try: weight += min(abs(raw_prediction(model,r["probe_initial"]["descriptors"])),15.)/5.
            except (ValueError,KeyError,TypeError): pass
        result.append((float(r["delta_mm"]),float(weight)))
    return result

def select_additions(candidates, selected_paths, count, mode, difficulty, tie_seed, *,
                     start_added=0, fixed_order=None):
    """Pure selection from available planned conditions/session IDs; no truth/features."""
    if mode not in MODES: raise ValueError("unknown strategy")
    allowed={"trial_path","planned_delta_mm","session"}
    if any(set(c)!=allowed for c in candidates):raise ValueError("candidate metadata allowlist violated")
    selected_set=set(selected_paths)
    metadata={c["trial_path"]:c for c in candidates}
    have=[metadata[p] for p in selected_paths]
    remaining=[c for c in candidates if c["trial_path"] not in selected_set]
    if count<0 or len(remaining)<count:raise ValueError("insufficient candidates; never clip budget")
    rng=np.random.default_rng(tie_seed);rng.shuffle(remaining)
    n_offsets=len({c["planned_delta_mm"] for c in candidates})
    n_sessions=len({c["session"] for c in candidates})
    def hard_score(c):
        return max((w/(1.+abs(c["planned_delta_mm"]-x)) for x,w in difficulty),default=0.)
    chosen=[];log=[]
    fixed_ranks={p:i for i,p in enumerate(fixed_order or [])}
    for i in range(count):
        available=[c for c in remaining if c["trial_path"] not in {x["trial_path"] for x in chosen}]
        prior=have+chosen
        components=COMPONENTS.get(mode)
        component=components[(start_added+i)%len(components)] if components else mode
        score=None
        if component=="fixed":
            if set(fixed_ranks)!=set(c["trial_path"] for c in remaining)|selected_set.difference(set(selected_paths[:24])):
                # Remaining plus previously consumed fixed tail must be contained in original tail.
                if not all(c["trial_path"] in fixed_ranks for c in available):
                    raise ValueError("fixed ordering missing candidate")
            item=min(available,key=lambda c:fixed_ranks[c["trial_path"]])
            score=fixed_ranks[item["trial_path"]]
        elif component in ("simple_feedback","B_hard"):
            item=max(available,key=hard_score);score=hard_score(item)
        elif component=="reference":
            item=min(available,key=lambda c:abs(c["planned_delta_mm"]));score=abs(item["planned_delta_mm"])
        elif component=="coverage":
            def distance(c):return min(abs(c["planned_delta_mm"]-p["planned_delta_mm"]) for p in prior)
            item=max(available,key=distance);score=distance(item)
        elif component=="offset_session_balanced":
            offsets=Counter(c["planned_delta_mm"] for c in prior)
            sessions=Counter(c["session"] for c in prior)
            # Equal-weight joint normalized loading, no outcome/feature access.
            def imbalance(c):
                n=len(prior)+1
                return .5*((offsets[c["planned_delta_mm"]]+1)*n_offsets/n+
                           (sessions[c["session"]]+1)*n_sessions/n)
            item=min(available,key=imbalance);score=imbalance(item)
        else:raise AssertionError(component)
        chosen.append(item)
        log.append({"trial_path":item["trial_path"],"component":component,
            "global_addition_index":start_added+i,"planned_delta_mm":item["planned_delta_mm"],
            "session":item["session"],"score":float(score),
            "score_information":"planned condition/source session and allowed historical B only; no candidate tactile or measured label"})
    return [c["trial_path"] for c in chosen],log

def initial_and_fixed_order(pool,acq_seed):
    initial,_=choose_batch(pool,[],24,"fixed",[],None,acq_seed+1,2.)
    initial_paths=[o["trial_path"] for o in initial]
    remaining=[o for o in pool if o["trial_path"] not in set(initial_paths)]
    rng=np.random.default_rng(acq_seed+2);rng.shuffle(remaining)
    return initial_paths,[o["trial_path"] for o in remaining]

def descriptive_metrics(evaluation, obsmeta, blocks):
    details=evaluation.get("details",[])
    areas=defaultdict(list);session_errors=defaultdict(list);group_errors=defaultdict(list)
    runtime_reasons=Counter()
    for x in details:
        error=abs(x["prediction_mm"]-x["target_action_mm"])
        a=-x["target_action_mm"]
        region="negative_a_lt_minus2" if a<-2 else "positive_a_gt2" if a>2 else "reference_abs_a_le2"
        areas[region].append(error)
        source=obsmeta[x["trial_path"]]
        session_errors[source["session"]].append(error)
        group_errors[blocks[x["trial_path"]]].append(error)
        pred=x["runtime_prediction"]
        if not pred.get("valid"):runtime_reasons[pred.get("reason","unknown")]+=1
    return {"per_region":{k:{"n":len(v),"mae_mm":float(np.mean(v)),"p90_mm":float(np.quantile(v,.9))} for k,v in areas.items()},
        "session_balanced_mae_mm":float(np.mean([np.mean(v) for v in session_errors.values()])) if session_errors else None,
        "repeat_group_balanced_mae_mm":float(np.mean([np.mean(v) for v in group_errors.values()])) if group_errors else None,
        "rejection_rate":evaluation.get("rejected",0)/evaluation["n"] if evaluation["n"] else None,
        "runtime_rejection_reasons":dict(runtime_reasons)}

def run_job(args):
    roots,out,split,acq_seed,model_seed,epochs=args
    _,rows,obs,rowmap,reference,_,_=load_data(roots)
    roles=split["roles"];obsmap={o["trial_path"]:o for o in obs}
    pool=[o for o in obs if roles[o["trial_path"]]=="pool"]
    dev=[o for o in obs if roles[o["trial_path"]]=="development"]
    test=[o for o in obs if roles[o["trial_path"]]=="independent"]
    bdev=[r for r in rows if r.get("cond")=="B" and r["session_role"] not in FINAL_ROLES]
    initial,fixed_order=initial_and_fixed_order(pool,acq_seed)
    candidates=canonical_metadata(pool)
    tolerance=float(reference["config"]["servo"]["tol_mm"])
    records=[]
    for mode in MODES:
        selected=initial[:];model=None
        for index,budget in enumerate(BUDGETS):
            if index==0:
                additions=selected[:]
                selection_log=[{"trial_path":p,"component":"shared_initial24","global_addition_index":None} for p in additions]
            else:
                difficulty=b_difficulty(bdev,model) if mode in ("simple_feedback","agent_evidence","no_reference","no_coverage") else []
                additions,selection_log=select_additions(candidates,selected,8,mode,difficulty,
                    acq_seed+index+1,start_added=len(selected)-24,fixed_order=fixed_order)
                selected+=additions
            if len(selected)!=budget or len(set(selected))!=budget:raise AssertionError("budget violation")
            if budget==48 and mode=="fixed" and selected!=initial+fixed_order[:24]:raise AssertionError("old fixed48 mismatch")
            dest=Path(out)/"runs"/split["split_id"]/f"a{acq_seed}_m{model_seed}"/mode/f"n{budget:03d}"
            selected_obs=[obsmap[p] for p in selected]
            training_roots,held=stage_training(dest/"training_view",selected_obs,dev,rowmap)
            settings={"learning":{"min_levels":3},"contact_policy":{"seed":model_seed,"epochs":epochs,
                "frozen_validation_groups":held}}
            before=time.monotonic()
            model=train_contact_policy(training_roots,settings,out=dest/"model.json")
            elapsed=time.monotonic()-before
            if model.get("training_count")!=budget:raise AssertionError("actual learner budget mismatch")
            train_metrics=geometry_evaluation(model,selected_obs,tolerance)
            dev_metrics=geometry_evaluation(model,dev,tolerance)
            test_metrics=geometry_evaluation(model,test,tolerance)
            descriptive=descriptive_metrics(test_metrics,obsmap,split["blocks"])
            record={"suite":split["suite"],"split_id":split["split_id"],"acquisition_seed":acq_seed,
                "model_seed":model_seed,"mode":mode,"budget":budget,"actual_training_count":model["training_count"],
                "selected_trials":selected[:],"added_trials":additions,"selection_log":selection_log,
                "model_id":model["model_id"],"usable_for_B":model["usable_for_B"],
                "blocking_reasons":model["blocking_reasons"],
                "training":train_metrics,"development":dev_metrics,"test":test_metrics,
                "descriptive":descriptive,"shared_A_development":len(dev),"shared_B_development":len(bdev),
                "training_wall_seconds":elapsed,"new_robot_interactions":0}
            write_json(dest/"result.json",record)
            compact={k:v for k,v in record.items() if k not in ("selected_trials","added_trials","selection_log","training","development","test")}
            compact.update(result_path=str(dest/"result.json"),
                training={k:v for k,v in train_metrics.items() if k!="details"},
                development={k:v for k,v in dev_metrics.items() if k!="details"},
                test={k:v for k,v in test_metrics.items() if k!="details"})
            records.append(compact)
    return records

def aggregate(records,protocol):
    grouped=defaultdict(list)
    for r in records:grouped[(r["suite"],r["mode"],r["budget"])].append(r)
    summary=[]
    for (suite,mode,budget),rs in sorted(grouped.items()):
        row={"suite":suite,"mode":mode,"budget":budget,"runs":len(rs),"usable_models":sum(r["usable_for_B"] for r in rs)}
        for field in ("mae_mm","p90_mm"):
            vals=[r["test"][field] for r in rs if r["test"][field] is not None]
            row["mean_"+field]=float(np.mean(vals)) if vals else None
            row["valid_"+field+"_runs"]=len(vals)
        row["mean_rejection_rate"]=float(np.mean([r["descriptive"]["rejection_rate"] for r in rs]))
        row["mean_session_balanced_mae_mm"]=float(np.mean([r["descriptive"]["session_balanced_mae_mm"] for r in rs]))
        row["per_region"]={}
        for name in ("negative_a_lt_minus2","reference_abs_a_le2","positive_a_gt2"):
            vals=[r["descriptive"]["per_region"][name] for r in rs if name in r["descriptive"]["per_region"]]
            row["per_region"][name]={"runs":len(vals),
                "mean_mae_mm":float(np.mean([v["mae_mm"] for v in vals])) if vals else None,
                "mean_p90_mm":float(np.mean([v["p90_mm"] for v in vals])) if vals else None}
        summary.append(row)
    bysplit=defaultdict(list)
    for r in records:bysplit[(r["suite"],r["split_id"],r["mode"],r["budget"])].append(r)
    paired=[]
    for suite in sorted({s["suite"] for s in protocol["splits"]}):
        for budget in BUDGETS:
            for baseline in ("fixed","simple_feedback","offset_session_balanced","no_reference","no_B_hard","no_coverage"):
                differences=[];p90diff=[];rejectiondiff=[];positive=[];split_details=[]
                for split in protocol["splits"]:
                    a=bysplit.get((suite,split["split_id"],"agent_evidence",budget))
                    b=bysplit.get((suite,split["split_id"],baseline,budget))
                    if not a or not b:continue
                    if any(r["test"]["mae_mm"] is None for r in a+b):
                        split_details.append({"split_id":split["split_id"],"status":"missing_predictions"});continue
                    diff=float(np.mean([r["test"]["mae_mm"] for r in a])-np.mean([r["test"]["mae_mm"] for r in b]))
                    dp=float(np.mean([r["test"]["p90_mm"] for r in a])-np.mean([r["test"]["p90_mm"] for r in b]))
                    dr=float(np.mean([r["descriptive"]["rejection_rate"] for r in a])-np.mean([r["descriptive"]["rejection_rate"] for r in b]))
                    pa=[r["descriptive"]["per_region"].get("positive_a_gt2",{}).get("mae_mm") for r in a]
                    pb=[r["descriptive"]["per_region"].get("positive_a_gt2",{}).get("mae_mm") for r in b]
                    positive_diff=float(np.mean(pa)-np.mean(pb)) if None not in pa+pb else None
                    differences.append(diff);p90diff.append(dp);rejectiondiff.append(dr)
                    if positive_diff is not None:positive.append(positive_diff)
                    split_details.append({"split_id":split["split_id"],"full_minus_baseline_mae_mm":diff,
                        "full_minus_baseline_p90_mm":dp,"full_minus_baseline_rejection_rate":dr,
                        "positive_a_full_minus_baseline_mae_mm":positive_diff})
                paired.append({"suite":suite,"budget":budget,"contrast":"agent_evidence_minus_"+baseline,
                    "n_outer_splits":len(differences),"mean_difference_mm":float(np.mean(differences)) if differences else None,
                    "mean_p90_difference_mm":float(np.mean(p90diff)) if p90diff else None,
                    "mean_rejection_rate_difference":float(np.mean(rejectiondiff)) if rejectiondiff else None,
                    "mean_positive_a_difference_mm":float(np.mean(positive)) if positive else None,
                    "full_wins":sum(d < -1e-10 for d in differences),"ties":sum(abs(d)<=1e-10 for d in differences),
                    "split_details":split_details,"interpretation":"overlapping outer splits; seeds averaged within split; descriptive, not independent replicate inference"})
    return {"schema":"lowmid_ablation_summary_v1","status":"complete","fits":len(records),"aggregate":summary,
        "paired_split_contrasts":paired,"new_robot_interactions":0,
        "scope":"retrospective geometric imitation; no new physical recovery evidence",
        "batching_caveat":protocol["batching_caveat"]}

def run(source,out,workers=8):
    source=Path(source).resolve();out=Path(out).resolve()
    if out.exists() and any(out.iterdir()):raise ValueError("fresh immutable output required")
    source_protocol=json.loads((source/"protocol.json").read_text())
    source_hash=json.loads((source/"protocol_hash.json").read_text())["sha256"]
    if _hash(source_protocol)!=source_hash:raise ValueError("source protocol hash mismatch")
    roots=source_protocol["roots"]
    sessions,rows,obs,rowmap,reference,excluded,compat=load_data(roots)
    oldobs={o["trial_path"]:o["metadata_sha256"] for o in source_protocol["observations"]}
    current={o["trial_path"]:rowmap[o["trial_path"]]["metadata_sha256"] for o in obs}
    if current!=oldobs:raise ValueError("eligible data or source metadata changed")
    splits=deepcopy(source_protocol["splits"])
    for split in splits:
        split["budgets"]=list(BUDGETS)
        validate_split(obs,split["roles"],split["blocks"],BUDGETS)
    current_b=[provenance(r) for r in rows if r.get("cond")=="B" and r["session_role"] not in FINAL_ROLES]
    if current_b!=source_protocol["feedback_B"]:raise ValueError("historical B feedback source changed")
    protocol={"schema":"lowmid_ablation_protocol_v1","frozen_unix_ns":time.time_ns(),
        "source_protocol_path":str(source/"protocol.json"),"source_protocol_sha256":source_hash,
        "eligible_A":len(obs),"observations":source_protocol["observations"],"roots":roots,"splits":splits,
        "methods":list(MODES),"budgets":list(BUDGETS),"acquisition_seeds":source_protocol["acquisition_seeds"],
        "model_seeds":source_protocol["model_seeds"],"epochs":500,"learner":"unchanged ContactPolicy",
        "source_sessions":source_protocol["source_sessions"],"feedback_B":current_b,
        "candidate_selection_allowlist":["trial_path","planned_delta_mm","session"],
        "component_cycles":{k:list(v) for k,v in COMPONENTS.items()},
        "full_cycle":"continuous reference->B_hard->coverage across three8-sample batches; total8 each",
        "ablations":"remaining two components alternate continuously;4 each per8-sample batch,12 each by48",
        "balanced_formula":"minimize0.5*((n_offset+1)*K_offsets/(N+1)+(n_session+1)*K_sessions/(N+1)); equal uniform-category loads using candidate nominal offset/source-session only;seed tie break",
        "fixed_order":"same acq+2 pre-shuffled remainder as prior24->48 fixed; consume8 at each checkpoint",
        "tie_seed":"acquisition_seed + checkpoint_index + 1; initial24 uses old acquisition_seed+1",
        "batching_caveat":"New common24/32/40/48 retraining checkpoints. Full component cycle remains8/8/8 additions by48, but B scores update every8 instead of every24; adaptive paths are not identical to old24->48. Initial24 exactly matches old; fixed48 matches old ordering.",
        "seed_semantics":"model seeds affect adaptive B scores and selection; this is a coupled pipeline, not isolated optimizer variance",
        "expected_fits":len(splits)*2*3*len(MODES)*len(BUDGETS),
        "software_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "cost_semantics":"equal selected A budget plus same per-split development A and33historical B;source310attempts separate",
        "new_robot_interactions":0,"source_attempts":len(rows)}
    out.mkdir(parents=True,exist_ok=True)
    write_json(out/"protocol.json",protocol);write_json(out/"protocol_hash.json",{"sha256":_hash(protocol)})
    jobs=[(roots,str(out),s,a,m,500) for s in splits for a in protocol["acquisition_seeds"] for m in protocol["model_seeds"]]
    start=time.monotonic();records=[];failed=[];done=0
    with ProcessPoolExecutor(max_workers=workers) as executor:
        pending={executor.submit(run_job,j):j for j in jobs}
        for future in as_completed(pending):
            j=pending[future];done+=1
            try:records.extend(future.result())
            except Exception as exc:failed.append({"split":j[2]["split_id"],"acq_seed":j[3],"model_seed":j[4],"error":repr(exc)})
            progress={"done_jobs":done,"total_jobs":len(jobs),"completed_fits":len(records),"expected_fits":protocol["expected_fits"],
                "failed_jobs":failed,"elapsed_s":time.monotonic()-start,"hardware_connected":False}
            write_json(out/"progress.json",progress)
            print(json.dumps(progress),flush=True)
    if failed:
        write_json(out/"failed_jobs.json",failed);raise RuntimeError("failed jobs retained; no silent exclusion")
    if len(records)!=protocol["expected_fits"]:raise AssertionError("missing fits")
    if _hash(json.loads((out/"protocol.json").read_text()))!=_hash(protocol):raise AssertionError("protocol modified")
    write_json(out/"results_index.json",records)
    summary=aggregate(records,protocol);write_json(out/"summary.json",summary)
    write_json(out/"compute_cost.json",{"elapsed_wall_seconds":time.monotonic()-start,
        "summed_fit_seconds":sum(r["training_wall_seconds"] for r in records),"workers":workers,
        "fits":len(records),"new_robot_interactions":0,"source_attempts_separate":len(rows)})
    lines=["# Low/mid budget component ablation", "", protocol["batching_caveat"], "",
        "|suite|method|budget|mean MAE mm|mean P90 mm|usable models|mean rejection rate|",
        "|---|---|---:|---:|---:|---:|---:|"]
    for r in summary["aggregate"]:
        lines.append(f"|{r['suite']}|{r['mode']}|{r['budget']}|{r['mean_mae_mm']:.4f}|{r['mean_p90_mm']:.4f}|{r['usable_models']}/{r['runs']}|{r['mean_rejection_rate']:.4f}|")
    lines+=["","All source and model files remain on5090. No robot started. Raw geometry includes unusable models; deployment gates are separately reported.",
        "Outer splits reuse physical samples. Pairing averages seeds within each split; no naive significance from4368fits.",
        "Session72-named splits hold out RSI sessions; repeat72-named splits are within-session repeat groups. Names identify inherited splitting schemes, not current48budget.",
        "The balanced source-session baseline is an offline data-availability control, not an executable choice of a future physical session."]
    (out/"REPORT.md").write_text("\n".join(lines)+"\n")
    return summary

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source",default="outputs/skill_repeated_20261106")
    p.add_argument("--out",required=True);p.add_argument("--workers",type=int,default=8)
    a=p.parse_args(argv);r=run(a.source,a.out,a.workers)
    print(json.dumps({"status":r["status"],"fits":r["fits"]}))
    return 0
