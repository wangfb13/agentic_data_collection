"""Fit and accept each physical component independently on held-out trials."""
import numpy as np
from .common import group_folds


def fit_axis_metric(rows, spec, result):
    from .v5_metric import pairs, fit_head
    cfg=spec['observation']['metric']
    all_items=pairs(rows,spec)
    folds=[(pairs([rows[i] for i in tr],spec),pairs([rows[i] for i in te],spec)) for tr,te in group_folds(rows,cfg['folds'])]
    heads=[];accepted=[];reports={}
    for j,name in enumerate(('r','h')):
        def subset(items):
            return [(x,y[j:j+1],kind,ids) for x,y,kind,ids in items]
        report=reports[name]=dict(status='insufficient_measured_transitions_or_repeats',heldout_error_mm=None)
        if len(result['fit_groups'])<cfg['min_groups']:continue
        head=fit_head(subset(all_items),cfg,rank=1)
        if head is None:continue
        scores=[]
        for train,test in folds:
            W=fit_head(subset(train),cfg,rank=1)
            if W is None:break
            for x,y,kind,ids in subset(test):
                scores.append(dict(axis=name,evidence=ids,kind=kind,error_mm=abs(float((x@W-y)[0])),target_mm=float(y[0])))
        else:
            errs=[s['error_mm'] for s in scores if s['kind']=='action']
            repeated=[s['error_mm'] for s in scores if s['kind']=='repeat']
            moved=[abs(s['target_mm']) for s in scores if s['kind']=='action' and abs(s['target_mm'])>=cfg['min_axis_motion_mm']]
            if not errs or not repeated or not moved:continue
            mae=float(np.mean(errs));repeat=float(np.mean(repeated))
            baseline=float(np.mean([abs(s['target_mm']) for s in scores if s['kind']=='action']))
            valid=(mae < cfg['max_error_by_axis_mm'][name] and mae < baseline*(1-cfg['min_relative_gain'])
                   and repeat < min(moved)*spec['graph']['similar_radius'])
            report.update(status='ready' if valid else 'heldout_validation_failed',heldout_error_mm=mae,
                          repeat_distance_mm=repeat,baseline_mae_mm=baseline,action_resolution_mm=min(moved))
            result['heldout_samples'].extend(scores)
            if valid:
                heads.append(head[:,0]);accepted.append(name)
    result.update(axes=accepted,axis_reports=reports)
    if accepted:
        result.update(status='ready',coefficients=np.column_stack(heads).tolist(),
                      action_error_mm=max(reports[k]['heldout_error_mm'] for k in accepted),
                      repeat_distance_mm=max(reports[k]['repeat_distance_mm'] for k in accepted))
    return result
