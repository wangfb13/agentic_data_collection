"""Post-hoc mechanism diagnostics on frozen original bottle replay selections.

No hardware imports, no policy promotion, no independent test-based filtering.
Uses the exact existing ContactPolicy _network and verifies numerical parity.
"""
from __future__ import annotations
import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from .agentic_learning import _load
from .contact_policy import _network, _predict, feature_vector
from .skill_offline import discover, eligible_A, write_json

MODES = ('fixed','simple_feedback','agent_evidence')

def fit_predictions(X, y, cfg, seed, epochs, eval_X, *, scaling=None):
    """Same learner; optional train-subset scaling is diagnostic only."""
    if scaling is None:
        mean=X.mean(0); scale=X.std(0); scale=np.where(scale>1e-6,scale,1.)
        ys=max(float(y.std()),1.)
    else:
        mean,scale,ys=scaling
    Z=(X-mean)/scale
    settings={**cfg,'seed':int(seed),'epochs':int(epochs)}
    rng=np.random.default_rng(seed)
    nets=[_network(Z,y/ys,settings,rng) for _ in range(settings['ensemble_size'])]
    pred={k:np.asarray([_predict(n,(v-mean)/scale)*ys for n in nets]).mean(0) for k,v in eval_X.items()}
    return pred,(mean,scale,ys)

def stats(pred,y):
    error=np.abs(pred-y)
    return {'n':len(y),'mae_mm':float(error.mean()),'p90_mm':float(np.quantile(error,.9)),
            'max_mm':float(error.max()),'rmse_mm':float(np.sqrt(np.mean((pred-y)**2))),
            'bias_mm':float(np.mean(pred-y))}

def source_ids(model, split):
    return [str(Path(model['source_trials'][i]['trial_path']).resolve()) for i in model['split'][split+'_indices']]

def summary_distribution(items):
    x=np.asarray([o['x_mm'] for o in items])
    return {'n':len(items),'session_counts':dict(Counter(o['session'] for o in items)),
            'planned_offset_counts':dict(Counter(str(o['planned_delta_mm']) for o in items)),
            'measured_x_min_mm':float(x.min()),'measured_x_max_mm':float(x.max()),
            'label_std_mm':float(x.std()),'mean_off_axis_mm':float(np.mean([o['measurement'].get('off_axis_mm',0) for o in items])),
            'mean_probe_spread_mm':float(np.mean([o['measurement'].get('spread_mm',0) for o in items]))}

def run(original,out):
    started=time.monotonic(); original=Path(original);out=Path(out)
    if out.exists() and any(out.iterdir()): raise ValueError('diagnostic output must be new')
    out.mkdir(parents=True,exist_ok=True)
    seeds=[20261105+i for i in range(10)]
    protocol={'schema':'skill_budget_diagnosis_protocol_v1','original':str(original.resolve()),
        'classification':'post_hoc_mechanism_diagnosis_not_confirmatory_ablation',
        'frozen_data':'original three modes and original 24/48/72 selections, unchanged original split',
        'training_seeds':seeds,'base_epochs':500,'optimization_diagnostic':{'epochs':1500,'seeds':seeds[:3],'budgets':[24,72]},
        'scaling_diagnostic':{'72_scaling':'mean/std/y_std fixed to same method 24 training subset','seeds':seeds[:3]},
        'linear_reference':'ridge regression same standardized inputs and l2=.001, no bootstrapping',
        'evaluation_use':'all settings declared before runs; never selected for deployment based on test',
        'new_robot_interactions':0,'created_unix_ns':time.time_ns()}
    write_json(out/'protocol.json',protocol)
    sessions,rows,issues=_load(discover('outputs'),{})
    if issues: raise ValueError(str(issues))
    observations,excluded=eligible_A(rows,sessions[0]['meta'],{})
    bypath={str(Path(o['trial_path']).resolve()):o for o in observations}
    roles=json.loads((original/'split_manifest.json').read_text())['A_roles']
    final=[o for o in observations if roles[o['trial_path']]=='independent']
    dev=[o for o in observations if roles[o['trial_path']]=='development']
    Xfinal=np.asarray([feature_vector(o['descriptors']) for o in final]); yfinal=-np.asarray([o['x_mm'] for o in final])
    Xdev=np.asarray([feature_vector(o['descriptors']) for o in dev]); ydev=-np.asarray([o['x_mm'] for o in dev])
    models={}; data={}; distributions={}; original_details=[]; original_support=[]
    for mode in MODES:
        for i,budget in enumerate((24,48,72),1):
            m=json.loads((original/mode/f'round_{i:02d}'/'contact_policy.json').read_text())
            ids=source_ids(m,'training'); dids=source_ids(m,'validation')
            assert set(ids).isdisjoint({o['trial_path'] for o in final})
            assert set(ids).isdisjoint(dids)
            train=[bypath[p] for p in ids]
            X=np.asarray([feature_vector(o['descriptors']) for o in train]); y=-np.asarray([o['x_mm'] for o in train])
            np.testing.assert_allclose(y,[m['source_trials'][j]['target_action_mm'] for j in m['split']['training_indices']],rtol=0,atol=1e-10)
            models[(mode,budget)]=m;data[(mode,budget)]=(X,y,train)
            distributions[f'{mode}_{budget}']=summary_distribution(train)
            z=(Xfinal-np.asarray(m['scaling']['mean']))/m['scaling']['scale']
            pp=np.asarray([_predict(n,z)*m['scaling']['y_scale'] for n in m['networks']]).mean(0)
            support=np.asarray(m['ood']['training_support'])
            for o,pred,point in zip(final,pp,z):
                dist=float(np.sqrt(np.mean((support-point)**2,axis=1)).min())
                original_details.append({'mode':mode,'budget':budget,'trial_path':o['trial_path'],
                    'session':o['session'],'planned_offset_mm':o['planned_delta_mm'],'target_action_mm':-o['x_mm'],
                    'prediction_mm':float(pred),'signed_error_mm':float(pred+o['x_mm']),'absolute_error_mm':float(abs(pred+o['x_mm']))})
                original_support.append({'mode':mode,'budget':budget,'trial_path':o['trial_path'],
                    'standardized_nn_rms':dist,'ood_radius':m['ood']['rms_radius'],'outside_support':dist>m['ood']['rms_radius']})
    for mode in MODES:
        initial={x['trial_path'] for x in data[(mode,24)][2]}
        added=[x for x in data[(mode,72)][2] if x['trial_path'] not in initial]
        distributions[f'{mode}_added_25_72']=summary_distribution(added)
    distributions['original_test']=summary_distribution(final);distributions['original_development']=summary_distribution(dev)
    write_json(out/'data_distribution.json',distributions)
    write_json(out/'original_test_predictions.json',original_details);write_json(out/'original_test_support.json',original_support)
    records=[];details=[];parity=[]
    for mode in MODES:
        for budget in (24,48,72):
            X,y,train=data[(mode,budget)];m=models[(mode,budget)]
            evaluation={'train':X,'development':Xdev,'test':Xfinal}
            for seed in seeds:
                begin=time.monotonic()
                pp,scaling=fit_predictions(X,y,m['settings'],seed,500,evaluation)
                if seed==20261105:
                    z=(Xfinal-np.asarray(m['scaling']['mean']))/m['scaling']['scale']
                    expected=np.asarray([_predict(n,z)*m['scaling']['y_scale'] for n in m['networks']]).mean(0)
                    error=float(np.max(np.abs(pp['test']-expected)))
                    parity.append({'mode':mode,'budget':budget,'max_abs_prediction_difference':error})
                    np.testing.assert_allclose(pp['test'],expected,atol=1e-9,rtol=0)
                rec={'mode':mode,'budget':budget,'seed':seed,'experiment':'base_500','epochs':500,
                     'seconds':time.monotonic()-begin,
                     **{k:stats(pp[k],v) for k,v in [('train',y),('development',ydev),('test',yfinal)]}}
                records.append(rec)
                for o,target,pred in zip(final,yfinal,pp['test']):
                    details.append({'mode':mode,'budget':budget,'seed':seed,'experiment':'base_500','trial_path':o['trial_path'],
                                    'target_action_mm':float(target),'prediction_mm':float(pred),'absolute_error_mm':float(abs(pred-target))})
            for experiment in ('longer_1500','fixed24_scaling'):
                if experiment=='longer_1500' and budget not in (24,72):continue
                if experiment=='fixed24_scaling' and budget!=72:continue
                X24,y24,_=data[(mode,24)]
                sc=(X24.mean(0),np.where(X24.std(0)>1e-6,X24.std(0),1.),max(float(y24.std()),1.))
                for seed in seeds[:3]:
                    begin=time.monotonic();epochs=1500 if experiment=='longer_1500' else 500
                    pp,_=fit_predictions(X,y,m['settings'],seed,epochs,evaluation,
                                         scaling=sc if experiment=='fixed24_scaling' else None)
                    records.append({'mode':mode,'budget':budget,'seed':seed,'experiment':experiment,'epochs':epochs,
                        'seconds':time.monotonic()-begin,
                        **{k:stats(pp[k],v) for k,v in [('train',y),('development',ydev),('test',yfinal)]}})
            mean=X.mean(0);scale=np.where(X.std(0)>1e-6,X.std(0),1.)
            Z=np.column_stack([(X-mean)/scale,np.ones(len(X))]);ridge=np.eye(Z.shape[1])*.001;ridge[-1,-1]=0
            w=np.linalg.pinv(Z.T@Z+ridge)@Z.T@y
            pp={k:np.column_stack([(v-mean)/scale,np.ones(len(v))])@w for k,v in evaluation.items()}
            records.append({'mode':mode,'budget':budget,'seed':None,'experiment':'linear_ridge_reference','epochs':0,
                **{k:stats(pp[k],v) for k,v in [('train',y),('development',ydev),('test',yfinal)]}})
            print(mode,budget,'done',flush=True)
            write_json(out/'runs.json',records)
    write_json(out/'parity.json',parity);write_json(out/'per_trial_seed_predictions.json',details)
    groups={}
    for r in records:
        key=(r['experiment'],r['mode'],r['budget'])
        groups.setdefault(key,[]).append(r)
    aggregate=[]
    for (experiment,mode,budget),rr in groups.items():
        entry={'experiment':experiment,'mode':mode,'budget':budget,'runs':len(rr)}
        for part in ('train','development','test'):
            val=np.asarray([r[part]['mae_mm'] for r in rr])
            entry[part]={'mean_mae_mm':float(val.mean()),'sd_across_training_seeds_mm':float(val.std(ddof=1)) if len(val)>1 else None,
                         'min_mae_mm':float(val.min()),'max_mae_mm':float(val.max())}
        aggregate.append(entry)
    contrasts=[]
    for mode in MODES:
        for part in ('test','development','train'):
            r24={r['seed']:r[part]['mae_mm'] for r in records if r['mode']==mode and r['budget']==24 and r['experiment']=='base_500'}
            r72={r['seed']:r[part]['mae_mm'] for r in records if r['mode']==mode and r['budget']==72 and r['experiment']=='base_500'}
            delta=np.asarray([r72[s]-r24[s] for s in seeds])
            contrasts.append({'mode':mode,'evaluation':part,'seeds':len(seeds),
                'mean_72_minus_24_mae_mm':float(delta.mean()),'median_difference_mm':float(np.median(delta)),
                '72_better_seed_count':int((delta<0).sum()),'72_worse_seed_count':int((delta>0).sum()),
                'min_difference_mm':float(delta.min()),'max_difference_mm':float(delta.max())})
    report={'schema':'skill_budget_diagnosis_report_v1','protocol':protocol,'aggregate':aggregate,
        'paired_budget_contrasts':contrasts,'parity_passed':True,'model_fits':sum(r['experiment']!='linear_ridge_reference' for r in records),
        'linear_fits':9,'wall_seconds':time.monotonic()-started,'original_test_n':len(final),'development_n':len(dev),
        'source_observations':len(observations),'new_robot_interactions':0,
        'limitations':['Original 5 test observations are reused across seeds; seeds are not new physical trials.',
            'Original development12 used model gating and is not fresh independent evaluation.',
            'All mechanisms diagnostics are post-hoc and must not be substituted for preregistered main comparison.',
            'Neither label noise nor reference drift is established from disagreement alone.']}
    write_json(out/'report.json',report)
    return report

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--original',default='outputs/skill_offline_20261106_v2')
    p.add_argument('--out',default='outputs/skill_budget_diagnosis_20261106')
    a=p.parse_args(argv);r=run(a.original,a.out);print(json.dumps({'fits':r['model_fits'],'seconds':r['wall_seconds']}))
if __name__=='__main__':main()
