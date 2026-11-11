"""Preregistered repeated grouped replay; all IO is file-only, no robot imports."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from itertools import combinations
import json
import os
from pathlib import Path
import time
import numpy as np
from .agentic_learning import _load, _compatibility, _hash
from .contact_policy import train_contact_policy, FINAL_ROLES
from .skill_offline import (discover, eligible_A, choose_batch, stage_training,
    geometry_evaluation, write_json, provenance)

MODES = ("fixed", "simple_feedback", "agent_evidence")
GLOBAL_SEED = 20261106

def load_data(roots):
    settings={"learning":{"min_levels":3}}
    sessions,rows,issues=_load(roots,settings)
    incompatible,evidence=_compatibility(sessions,settings)
    if issues or incompatible:
        raise ValueError(str(issues+incompatible))
    reference=sessions[0]["meta"]
    observations,excluded=eligible_A(rows,reference,settings)
    rowmap={r["trial_path"]:r for r in rows}
    return sessions,rows,observations,rowmap,reference,excluded,evidence

def monitor_blocks(observations, rowmap):
    """Keep each monitor with the nearest preceding scan's repeat in its bottle."""
    result={}
    for o in observations:
        row=rowmap[o["trial_path"]]
        if row.get("repeat") is not None:
            result[o["trial_path"]]=o["group"]
            continue
        peers=[p for p in observations if p["session"]==o["session"] and
               rowmap[p["trial_path"]].get("bottle")==row.get("bottle") and
               rowmap[p["trial_path"]].get("repeat") is not None]
        if not peers:
            raise ValueError("monitor has no physical repetition block")
        index=int(row.get("index") or row["id"])
        previous=[p for p in peers if int(rowmap[p["trial_path"]].get("index") or rowmap[p["trial_path"]]["id"]) < index]
        if previous:
            nearest=max(previous,key=lambda p:int(rowmap[p["trial_path"]].get("index") or rowmap[p["trial_path"]]["id"]))
        else:
            nearest=min(peers,key=lambda p:abs(int(rowmap[p["trial_path"]].get("index") or rowmap[p["trial_path"]]["id"])-index))
        result[o["trial_path"]]=nearest["group"]
    return result

def validate_split(observations, roles, blocks, budgets):
    if set(roles)!=set(o["trial_path"] for o in observations):
        raise ValueError("split must assign every eligible trial once")
    counts=Counter(roles.values())
    if set(counts)!={"pool","development","independent"} or counts["pool"]<max(budgets):
        raise ValueError("insufficient pool or missing role; budgets cannot be clipped")
    for key in ("group","block"):
        grouped=defaultdict(set)
        for o in observations:
            grouped[o["group"] if key=="group" else blocks[o["trial_path"]]].add(roles[o["trial_path"]])
        if any(len(r)>1 for r in grouped.values()):
            raise ValueError("physical repeat/monitor block leaks across split")

def _make_split(observations, blocks, held, development, suite, number, seed, budgets, search_attempts):
    roles={o["trial_path"]:("independent" if blocks[o["trial_path"]] in held else
          "development" if blocks[o["trial_path"]] in development else "pool") for o in observations}
    validate_split(observations,roles,blocks,budgets)
    return {"split_id":f"{suite}_{number:02d}","suite":suite,"split_seed":seed,
            "budgets":budgets,"roles":roles,"blocks":blocks,"counts":dict(Counter(roles.values())),
            "groups_by_role":{r:sorted({blocks[o["trial_path"]] for o in observations if roles[o["trial_path"]]==r})
                              for r in ("pool","development","independent")},
            "search_attempts":search_attempts,
            "selection_information":"identities, session/bottle/repeat metadata and counts only; no features, labels or losses"}

def make_splits(observations,rowmap):
    sessions=Counter(o["session"] for o in observations)
    largest=max(sessions,key=sessions.get)
    small=sorted(k for k in sessions if k!=largest)
    if len(sessions)!=13 or len(observations)!=122:
        raise ValueError("this preregistered dataset changed; create/review a new protocol")
    blocks={o["trial_path"]:o["session"] for o in observations}
    splits=[]
    # Enumerate count-feasible whole-session partitions, no outcome access.
    candidates=[]
    for held in combinations(small,6):
        n=sum(sessions[k] for k in held)
        if not 32<=n<=36: continue
        rest=[k for k in small if k not in held]
        for dev in combinations(rest,2):
            d=sum(sessions[k] for k in dev)
            if 10<=d<=14 and len(observations)-n-d>=72:
                candidates.append((set(held),set(dev)))
    rng=np.random.default_rng(GLOBAL_SEED)
    order=rng.permutation(len(candidates))
    used=set()
    for attempt,i in enumerate(order,1):
        held,dev=candidates[i]
        key=tuple(sorted(held))
        if key in used: continue
        used.add(key)
        splits.append(_make_split(observations,blocks,held,dev,"session72",len(used),
                                 GLOBAL_SEED,[24,48,72],attempt))
        if len(used)==10:break
    if len(used)!=10:raise ValueError("not enough distinct whole-session partitions")
    # Within-session generalization, stronger than frame/random-trial splitting.
    repeat_blocks=monitor_blocks(observations,rowmap)
    counts=Counter(repeat_blocks.values())
    keys=sorted(counts)
    block_source={repeat_blocks[o["trial_path"]]:o["session"] for o in observations}
    used=set()
    for number in range(1,11):
        seed=GLOBAL_SEED+100+number
        rng=np.random.default_rng(seed)
        for attempt in range(1,20001):
            order=[keys[i] for i in rng.permutation(len(keys))]
            held=set(); n=0
            for key in order:
                if n<32 and n+counts[key]<=36:
                    held.add(key);n+=counts[key]
            f=sum(counts[k] for k in held if block_source[k]==largest)
            if not 32<=n<=36 or not 12<=f<=22 or n-f<12 or tuple(sorted(held)) in used:continue
            dev=set();d=0
            for key in order:
                if key not in held and d<10 and d+counts[key]<=14:
                    dev.add(key);d+=counts[key]
            if not 10<=d<=14 or 122-n-d<72:continue
            used.add(tuple(sorted(held)))
            splits.append(_make_split(observations,repeat_blocks,held,dev,"repeat72",number,
                                      seed,[24,48,72],attempt))
            break
        else:raise ValueError("cannot make count-feasible grouped-repeat split")
    # Strong reverse-domain test: all old formal A held out, limited common budget.
    devsets=[set(c) for size in (2,3) for c in combinations(small,size)
             if 10<=sum(sessions[k] for k in c)<=16 and 65-sum(sessions[k] for k in c)>=48]
    rng=np.random.default_rng(GLOBAL_SEED+500)
    for number,i in enumerate(rng.permutation(len(devsets))[:6],1):
        splits.append(_make_split(observations,blocks,{largest},devsets[i],"leave_formal48",
                                  number,GLOBAL_SEED+500,[24,48],number))
    return splits,largest

def _job(job):
    """One paired split/acquisition/model-seed block: all methods share evidence."""
    started=time.monotonic()
    roots,out,split,acq_seed,model_seed,epochs=job
    sessions,rows,obs,rowmap,reference,_,_=load_data(roots)
    roles=split["roles"]
    pool=[o for o in obs if roles[o["trial_path"]]=="pool"]
    dev=[o for o in obs if roles[o["trial_path"]]=="development"]
    test=[o for o in obs if roles[o["trial_path"]]=="independent"]
    Bdev=[r for r in rows if r.get("cond")=="B" and r["session_role"] not in FINAL_ROLES]
    tolerance=float(reference["config"]["servo"]["tol_mm"])
    job_id=f"{split['split_id']}/a{acq_seed}_m{model_seed}"
    results=[]
    common_initial=None
    for mode in MODES:
        selected=[];model=None
        for index,budget in enumerate(split["budgets"],1):
            batch,rationale=choose_batch(pool,selected,budget-len(selected),mode,Bdev,model,acq_seed+index,tolerance)
            selected+=batch
            paths=[o["trial_path"] for o in selected]
            if index==1:
                if common_initial is None:common_initial=paths[:]
                if paths!=common_initial:raise AssertionError("initial acquisitions differ across methods")
            if len(paths)!=budget or len(set(paths))!=budget:raise AssertionError("unequal/duplicate budget")
            dest=Path(out)/"runs"/job_id/mode/f"n{budget:03d}"
            train_roots,held=stage_training(dest/"training_view",selected,dev,rowmap)
            settings={"learning":{"min_levels":3},"contact_policy":{"seed":model_seed,"epochs":epochs,
                       "frozen_validation_groups":held}}
            before=time.monotonic()
            model=train_contact_policy(train_roots,settings,out=dest/"model.json")
            elapsed=time.monotonic()-before
            # Frozen checkpoints are never selected by outer-test results.
            training=geometry_evaluation(model,selected,tolerance)
            development=geometry_evaluation(model,dev,tolerance)
            outer=geometry_evaluation(model,test,tolerance)
            if model.get("training_count") != budget:
                raise AssertionError("actual learner sample count differs from budget")
            result={"suite":split["suite"],"split_id":split["split_id"],"acquisition_seed":acq_seed,
                "model_seed":model_seed,"mode":mode,"budget":budget,"actual_training_count":model.get("training_count"),
                "selected_trials":paths,"added_trials":[o["trial_path"] for o in batch],
                "selection_reason":rationale,"model_id":model["model_id"],"model_path":str(dest/"model.json"),
                "usable_for_B":model["usable_for_B"],"blocking_reasons":model["blocking_reasons"],
                "training":training,"development":development,"test":outer,
                "shared_A_development":len(dev),"shared_B_development":len(Bdev),
                "new_robot_interactions":0,"training_wall_seconds":elapsed}
            write_json(dest/"result.json",result)
            results.append({k:v for k,v in result.items() if k not in ("training","development","test","selected_trials","added_trials")}|{
                "result_path":str(dest/"result.json"),"training":{k:v for k,v in training.items() if k!="details"},
                "development":{k:v for k,v in development.items() if k!="details"},
                "test":{k:v for k,v in outer.items() if k!="details"}})
    return {"job_id":job_id,"results":results,"wall_seconds":time.monotonic()-started}

def summarize(results,splits,out):
    buckets=defaultdict(list)
    for r in results:buckets[(r["suite"],r["mode"],r["budget"])].append(r)
    aggregate=[]
    for (suite,mode,budget),rs in sorted(buckets.items()):
        vals=[r["test"]["mae_mm"] for r in rs if r["test"].get("mae_mm") is not None]
        p90=[r["test"]["p90_mm"] for r in rs if r["test"].get("p90_mm") is not None]
        aggregate.append({"suite":suite,"mode":mode,"budget":budget,"runs":len(rs),
            "trained_predictions":len(vals),"usable_models":sum(r["usable_for_B"] for r in rs),
            "mae_mean_mm":float(np.mean(vals)) if vals else None,"mae_median_mm":float(np.median(vals)) if vals else None,
            "mae_q10_mm":float(np.quantile(vals,.1)) if vals else None,"mae_q90_mm":float(np.quantile(vals,.9)) if vals else None,
            "p90_mean_mm":float(np.mean(p90)) if p90 else None,
            "test_n_min":min(r["test"]["n"] for r in rs),"test_n_max":max(r["test"]["n"] for r in rs),
            "blocking_reasons":dict(Counter(x for r in rs for x in r["blocking_reasons"]))})
    split_means={}
    for r in results:
        key=(r["suite"],r["split_id"],r["mode"],r["budget"])
        split_means.setdefault(key,[]).append(r["test"].get("mae_mm"))
    split_means={k:float(np.mean([v for v in vals if v is not None])) for k,vals in split_means.items() if any(v is not None for v in vals)}
    paired=[]
    for suite in sorted({s["suite"] for s in splits}):
        for budget in sorted({r["budget"] for r in results if r["suite"]==suite}):
            for comparator in ("fixed","simple_feedback"):
                diffs=[]
                for s in splits:
                    k=(suite,s["split_id"],"agent_evidence",budget);j=(suite,s["split_id"],comparator,budget)
                    if k in split_means and j in split_means:diffs.append(split_means[k]-split_means[j])
                paired.append({"suite":suite,"budget":budget,"contrast":"agent_evidence_minus_"+comparator,
                    "n_outer_splits":len(diffs),"differences_by_split_mm":diffs,
                    "mean_difference_mm":float(np.mean(diffs)) if diffs else None,
                    "outer_split_wins":sum(d<0 for d in diffs),"ties":sum(d==0 for d in diffs),
                    "inference":"descriptive overlapping resamples, not independent experimental replicates"})
    growth=[]
    for suite in sorted({s["suite"] for s in splits}):
        high=max(r["budget"] for r in results if r["suite"]==suite)
        for mode in MODES:
            diffs=[]
            for s in splits:
                k=(suite,s["split_id"],mode,high);j=(suite,s["split_id"],mode,24)
                if k in split_means and j in split_means:diffs.append(split_means[k]-split_means[j])
            growth.append({"suite":suite,"mode":mode,"contrast":f"n{high}_minus_n24",
                "mean_difference_mm":float(np.mean(diffs)),"outer_split_improvements":sum(d<0 for d in diffs),
                "n_outer_splits":len(diffs),"differences_by_split_mm":diffs})
    report={"schema":"repeated_grouped_replay_report_v1","status":"complete","fits":len(results),
        "new_robot_interactions":0,"aggregate":aggregate,"paired_outer_split_summary":paired,"budget_growth":growth,
        "test_reuse":{suite:{"unique_test_trials":len({p for s in splits if s["suite"]==suite for p,r in s["roles"].items() if r=="independent"}),
            "n_splits":sum(s["suite"]==suite for s in splits),
            "test_appearances":dict(Counter(p for s in splits if s["suite"]==suite for p,r in s["roles"].items() if r=="independent"))}
            for suite in sorted({s["suite"] for s in splits})},
        "interpretation":["Geometry imitation, not prospective physical recovery success.",
            "Outer splits reuse finite observations; no p-values or naive fit-level confidence intervals.",
            "session72 tests small RSI sessions; formal A always remains in training.",
            "repeat72 tests within previously observed sessions using disjoint complete repeat/monitor blocks.",
            "leave_formal48 is reverse-domain stress and cannot support 72 samples.",
            "Model seeds also affect adaptive future acquisition through B predictions; optimization alone is isolated by a separate fixed-selection diagnostic.",
            "Historical sources may have been exposed to earlier collection methods. This is retrospective replay."]}
    write_json(Path(out)/"results_index.json",results)
    write_json(Path(out)/"summary.json",report)
    return report

def run(roots,out,workers=8,epochs=500):
    out=Path(out).resolve()
    if out.exists() and any(out.iterdir()):raise ValueError("fresh output required")
    out.mkdir(parents=True,exist_ok=True)
    sessions,rows,obs,rowmap,reference,excluded,compat=load_data(roots)
    splits,largest=make_splits(obs,rowmap)
    acq_seeds=[1101,2202];model_seeds=[3101,3102,3103]
    protocol={"schema":"repeated_grouped_replay_protocol_v1","frozen_unix_ns":time.time_ns(),
        "roots":roots,"eligible_A":len(obs),"source_attempts":len(rows),
        "eligible_by_session":dict(Counter(o["session"] for o in obs)),
        "excluded_A":excluded,"source_sessions":[{k:s[k] for k in ("path","metadata_sha256","role")} for s in sessions],
        "observations":[{**provenance(rowmap[o["trial_path"]]),"group":o["group"],
            "target_action_mm":-o["x_mm"],"analysis_only_planned_a_mm":o["planned_delta_mm"],
            "bottle":rowmap[o["trial_path"]].get("bottle")} for o in obs],
        "largest_formal_session":largest,"splits":splits,
        "acquisition_seeds":acq_seeds,"model_seeds":model_seeds,"epochs":epochs,
        "modes":MODES,"learner":"unchanged ContactPolicy numpy_tanh_mlp_ensemble",
        "feedback_B":[provenance(r) for r in rows if r.get("cond")=="B" and r["session_role"] not in FINAL_ROLES],
        "historical_final_B_never_used_for_selection":True,
        "test_n_goal":"32–36 in 72-sample suites; 57 in reverse-domain suite",
        "expected_fits":sum(len(s["budgets"])*len(MODES)*len(acq_seeds)*len(model_seeds) for s in splits),
        "selection_freeze":"Protocol/splits recorded before fitting. Only group identities/counts used for splits.",
        "seed_interaction":"model initialization changes adaptive B scores after24; fixed-selection diagnosis isolates optimizer effect",
        "cost_semantics":"equal selected A budget plus shared per-split development A and33B; replay never counts as new robot interaction",
        "new_robot_interactions":0}
    write_json(out/"protocol.json",protocol)
    write_json(out/"protocol_hash.json",{"sha256":_hash(protocol)})
    jobs=[(roots,str(out),s,a,m,epochs) for s in splits for a in acq_seeds for m in model_seeds]
    completed=[];failures=[];started=time.monotonic()
    with ProcessPoolExecutor(max_workers=workers) as executor:
        pending={executor.submit(_job,j):j for j in jobs}
        for future in as_completed(pending):
            j=pending[future]
            try:
                value=future.result()
                completed.extend(value["results"])
                print(json.dumps({"done_jobs":len(completed)//9,"fits":len(completed),"job":value["job_id"],
                                  "elapsed_s":round(time.monotonic()-started,1)}),flush=True)
            except Exception as exc:
                failures.append({"split_id":j[2]["split_id"],"acquisition_seed":j[3],"model_seed":j[4],"error":repr(exc)})
            write_json(out/"progress.json",{"completed_fits":len(completed),"expected_fits":protocol["expected_fits"],
                "failed_jobs":failures,"elapsed_s":time.monotonic()-started,"hardware_connected":False})
    if failures:
        write_json(out/"failed_jobs.json",failures)
        raise RuntimeError("jobs failed; no silent exclusion: "+str(failures))
    if len(completed)!=protocol["expected_fits"]:raise AssertionError("missing budget runs")
    if _hash(json.loads((out/"protocol.json").read_text()))!=json.loads((out/"protocol_hash.json").read_text())["sha256"]:
        raise AssertionError("protocol changed during run")
    report=summarize(completed,splits,out)
    write_json(out/"compute_cost.json",{"elapsed_wall_seconds":time.monotonic()-started,
        "summed_fit_seconds":sum(r["training_wall_seconds"] for r in completed),"workers":workers,
        "new_robot_interactions":0,"source_attempts_separate":len(rows)})
    return report

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",required=True);p.add_argument("--data-root",default="outputs")
    p.add_argument("--workers",type=int,default=8);p.add_argument("--epochs",type=int,default=500)
    a=p.parse_args(argv)
    r=run(discover(a.data_root),a.out,a.workers,a.epochs)
    print(json.dumps({"status":r["status"],"fits":r["fits"]}))
    return 0
