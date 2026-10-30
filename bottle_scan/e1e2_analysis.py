"""Offline E1/E2 fits, trial-level rates and Gaussian-weighted robustness."""
from __future__ import annotations
import argparse
import csv
import json
import math
from pathlib import Path
import numpy as np
import yaml
from .session import atomic_json,plain,fingerprint


def load_trials(roots):
    rows=[]
    for root in roots:
        root=Path(root);meta=json.loads((root/'session_meta.json').read_text())
        if meta.get('schema')!='e1e2_session_v1' or meta.get('simulated') is not False:raise ValueError('只分析独立 E1/E2 实机格式：'+str(root))
        paths=list(sorted((root/'trials').glob('*/meta.json')))+list(sorted((root/'interrupted').glob('*/meta.json')))
        for path in paths:
            m=json.loads(path.read_text())
            m.update(session=str(root.resolve()),trial_path=str(path.parent.resolve()),session_meta=meta,
                     archived_attempt=path.parent.parent.name=='interrupted')
            status=path.parent/'recording_status.json'
            m['analysis_recording_complete']=bool(m.get('recording_complete') and status.exists() and json.loads(status.read_text()).get('complete'))
            rows.append(m)
    return rows


def _ols(x,y):
    x=np.asarray(x,float);y=np.asarray(y,float);slope,intercept=np.polyfit(x,y,1)
    residual=y-(slope*x+intercept);ss=float(np.sum((y-y.mean())**2))
    return {'slope_px_per_mm':float(slope),'intercept_px':float(intercept),
            'r2':float(1-np.sum(residual**2)/ss) if ss>1e-12 else 0.,
            'rmse_px':float(np.sqrt(np.mean(residual**2))),'n':len(x)}


def fit_e1(rows,*,min_r2=.9,min_offsets=3,min_slope=.05):
    """Only INITIAL, valid, non-edge A scans; B results never teach B itself."""
    selected=[r for r in rows if r.get('complete',True) and not r.get('archived_attempt') and r.get('cond')=='A' and r.get('kind')=='scan' and
              r.get('analysis_recording_complete',r.get('recording_complete')) and r.get('label')!='invalid' and
              r.get('probe_initial',{}).get('valid')]
    fingers=[]
    for finger in range(2):
        observations=[]
        for row in selected:
            ds=row['probe_initial'].get('descriptors',[])
            if len(ds)!=2:continue
            d=ds[finger]
            if d.get('valid') and not d.get('touches_edge'):
                observations.append((float(row['delta_mm']),d['centroid_px'],row['bottle']))
        levels=sorted(set(x[0] for x in observations));candidates=[]
        for left in range(len(levels)):
            for right in range(left+min_offsets-1,len(levels)):
                lo,hi=levels[left],levels[right]
                if lo>0 or hi<0:continue
                subset=[x for x in observations if lo<=x[0]<=hi]
                # Each level must have actual repetitions, not one lucky point.
                if any(sum(x[0]==d for x in subset)<2 for d in levels[left:right+1]):continue
                for coordinate,idx in [('u',0),('v',1)]:
                    x=np.array([r[0] for r in subset]);y=np.array([r[1][idx] for r in subset])
                    fit=_ols(x,y)
                    if fit['r2']<min_r2 or abs(fit['slope_px_per_mm'])<min_slope:continue
                    within=[];by=[]
                    for delta in levels[left:right+1]:
                        vals=y[x==delta];std=float(vals.std(ddof=1));within.extend((vals-vals.mean()).tolist())
                        by.append({'delta_mm':delta,'n':len(vals),'std_px':std,'equivalent_std_mm':std/abs(fit['slope_px_per_mm'])})
                    dof=sum(v['n']-1 for v in by)
                    sigma=math.sqrt(sum((v['n']-1)*v['std_px']**2 for v in by)/dof) if dof else None
                    # Direction sign maps increasing observed delta to increasing
                    # sensor coordinate. High signed coordinate is the root side.
                    fit.update(finger=('left','right')[finger],coordinate=coordinate,linear_range_mm=[lo,hi],
                        direction_sign=int(np.sign(fit['slope_px_per_mm'])),within_level_std_px=sigma,
                        resolution_1sigma_mm=sigma/abs(fit['slope_px_per_mm']) if sigma is not None else None,
                        resolution_95_halfwidth_mm=1.96*sigma/abs(fit['slope_px_per_mm']) if sigma is not None else None,
                        repeat_stats=by,root_edge=('right' if coordinate=='u' else 'bottom') if fit['slope_px_per_mm']>0 else ('left' if coordinate=='u' else 'top'))
                    candidates.append(fit)
        best=max(candidates,key=lambda f:(f['linear_range_mm'][1]-f['linear_range_mm'][0],f['r2'],abs(f['slope_px_per_mm']))) if candidates else None
        fingers.append(best or {'finger':('left','right')[finger],'usable':False,'reason':'没有包含零点、至少3档且每档2次的可靠非贴边线性区间'})
    usable=all('slope_px_per_mm' in f for f in fingers)
    common=[max(f['linear_range_mm'][0] for f in fingers),min(f['linear_range_mm'][1] for f in fingers)] if usable else None
    if common and common[1]-common[0]<1e-6:usable=False
    identities={fingerprint(r['session_meta']['config'].get('axes_tcp')) for r in selected if 'session_meta' in r}
    if len(identities)>1:raise ValueError('E1 混入不同 a/c/l 定义，不能合并拟合')
    return {'schema':'e1e2_fit_v1','usable_for_B':usable,'fingers':fingers,'common_linear_range_mm':common,
            'fit_policy':{'min_r2':min_r2,'min_distinct_offsets':min_offsets,'min_repetitions_per_level':2,
                          'largest_contiguous_observed_interval_including_zero':True,'initial_A_only':True,
                          'edge_contacts_excluded':True,'min_abs_slope_px_per_mm':min_slope},
            'resolution_definition':'pooled within-offset sample SD / abs(slope); 95% single-observation halfwidth=1.96 SD; not a hardware pixel pitch',
            'source_sessions':sorted(set(r.get('session','') for r in selected)),
            'axes_tcp':selected[0].get('session_meta',{}).get('config',{}).get('axes_tcp') if selected else None,
            'normal_axis':selected[0].get('session_meta',{}).get('config',{}).get('force',{}).get('normal_axis') if selected else None,
            'normal_signs':selected[0].get('session_meta',{}).get('config',{}).get('force',{}).get('normal_signs') if selected else None}


def wilson(k,n,z=1.959963984540054):
    if not n:return None,None
    # Exact binomial endpoints avoid a 1e-16 negative plotting error at 0/100%.
    if k==0:return 0.,z*z/(n+z*z)
    if k==n:return n/(n+z*z),1.
    p=k/n;den=1+z*z/n;middle=(p+z*z/(2*n))/den
    half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return max(0.,middle-half),min(1.,middle+half)


def proportion_errorbars(values,lower,upper):
    values,lower,upper=np.broadcast_arrays(np.asarray(values,float),np.asarray(lower,float),np.asarray(upper,float))
    errors=np.stack([values-lower,upper-values])
    if not np.all(np.isfinite(errors)) or np.any(errors < -1e-12):
        raise ValueError('比例置信区间不包含估计值，无法绘制误差棒')
    # Only numerical roundoff is clipped; genuinely inconsistent bounds fail.
    return np.maximum(errors,0.)


def eligible(r):
    return r.get('complete',True) and not r.get('archived_attempt') and r.get('kind')=='scan' and r.get('status')!='skipped_bound' and r.get('label')!='invalid' and r.get('analysis_recording_complete',r.get('recording_complete'))


def success(r):
    return bool(r.get('lifted') and r.get('hold_complete') and not r.get('dropped') and r.get('label')=='normal')


def rates(rows):
    result=[]
    for cond in ('A','B'):
        for bottle in ['all']+sorted(set(r['bottle'] for r in rows)):
            subset=[r for r in rows if eligible(r) and r['cond']==cond and (bottle=='all' or r['bottle']==bottle)]
            for delta in sorted(set(float(r['delta_mm']) for r in subset)):
                group=[r for r in subset if r['delta_mm']==delta];n=len(group);k=sum(success(r) for r in group)
                valid=sum(bool(r.get('probe_final',{}).get('valid') and r.get('bilateral')) for r in group)
                initial=sum(bool(r.get('probe_initial',{}).get('valid')) for r in group)
                lo,hi=wilson(k,n);clo,chi=wilson(valid,n)
                result.append(dict(cond=cond,bottle=bottle,delta_mm=delta,n=n,successes=k,success_rate=k/n,ci_low=lo,ci_high=hi,
                    contact_valid=valid,contact_rate=valid/n,contact_ci_low=clo,contact_ci_high=chi,
                    initial_contact_rate=initial/n,mean_regrasps=float(np.mean([r.get('regrasps',0) for r in group]))))
    return result


def tolerance(rate_rows,cond):
    """First measured <80% on each side; no interpolation or invented crossing."""
    r={float(x['delta_mm']):x for x in rate_rows if x['cond']==cond and x['bottle']=='all'}
    if 0. not in r:return {'status':'no_zero_measurement','negative_mm':None,'positive_mm':None}
    out={'definition':'first measured offset outward from zero with empirical success_rate < 0.8; no interpolation',
         'zero_success_rate':r[0.]['success_rate']}
    for name,side in [('negative',-1),('positive',1)]:
        levels=sorted([d for d in r if d*side>0],key=abs)
        crossing=0. if r[0.]['success_rate']<.8 else next((d for d in levels if r[d]['success_rate']<.8),None)
        out[name+'_mm']=crossing;out[name+'_status']='measured_crossing' if crossing is not None else 'not_reached_in_observed_range'
        out[name+'_tested_max_abs_mm']=max([abs(d) for d in levels],default=None)
    measured=[abs(out[x+'_mm']) for x in ('negative','positive') if out[x+'_mm'] is not None]
    out['conservative_first_drop_abs_mm']=min(measured) if len(measured)==2 else None
    return out


def gaussian_weighted(rate_rows):
    """Normal(0,sigma) masses over midpoint bins, conditioned on observed range.

    Also report unconditional bounds treating unobserved tails as failures or
    successes. Tail mass is never silently credited to the outermost sample.
    """
    from scipy.special import ndtr
    result=[]
    for cond in ('A','B'):
        r=sorted([x for x in rate_rows if x['cond']==cond and x['bottle']=='all'],key=lambda x:x['delta_mm'])
        if len(r)<2:continue
        x=np.array([v['delta_mm'] for v in r]);edges=np.r_[x[0],(x[:-1]+x[1:])/2,x[-1]]
        for sigma in [2.,4.,6.,8.,10.,12.,16.]:
            masses=np.diff(ndtr(edges/sigma));coverage=float(masses.sum())
            total=float(masses@np.array([v['success_rate'] for v in r]));weights=masses/coverage
            # Independent-bin uncertainty propagated by a conservative weighted
            # sum of Wilson endpoints (not mislabelled as a new exact CI).
            result.append({'cond':cond,'sigma_mm':sigma,'weighted_success_conditional':total/coverage,
                'observed_gaussian_mass':coverage,'unobserved_tail_mass':1-coverage,
                'unconditional_lower_bound':total,'unconditional_upper_bound':total+1-coverage,
                'weighted_wilson_endpoint_low':float(weights@np.array([v['ci_low'] for v in r])),
                'weighted_wilson_endpoint_high':float(weights@np.array([v['ci_high'] for v in r])),
                'range_mm':[float(x[0]),float(x[-1])],
                'definition':'midpoint-bin normal masses conditional on tested span; outer tails explicitly unknown'})
    return result


def save_csv(path,rows,fields=None):
    fields=fields or (list(rows[0]) if rows else [])
    with Path(path).open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for row in rows:w.writerow({k:json.dumps(plain(v),ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in row.items() if k in fields})


def descriptor_rows(rows):
    result=[]
    for r in rows:
        for i,d in enumerate(r.get('probe_initial',{}).get('descriptors',[])):
            centroid=d.get('centroid_px') or [None,None];bbox=d.get('bbox_xyxy_px') or [None]*4
            result.append(dict(session=r['session'],id=r['id'],cond=r['cond'],bottle=r['bottle'],kind=r['kind'],delta_mm=r['delta_mm'],
                complete=r.get('complete',True),archived_attempt=r.get('archived_attempt',False),label=r.get('label'),finger=('left','right')[i],centroid_u_px=centroid[0],centroid_v_px=centroid[1],
                bbox_left_px=bbox[0],bbox_top_px=bbox[1],bbox_right_px=bbox[2],bbox_bottom_px=bbox[3],
                area_px=d.get('area_px'),principal_angle_deg=d.get('principal_angle_deg'),touches_edge=d.get('touches_edge'),
                edge_sides=d.get('edge_sides'),valid=d.get('valid'),recording_complete=r.get('analysis_recording_complete')))
    return result


def check_compatibility(rows):
    # Goal offsets/condition/repetitions may differ, sensing and force protocols
    # must be comparable. Do not group a changed finger/force unit silently.
    bindings={}
    for r in rows:
        cfg=r.get('session_meta',{}).get('config',{})
        selected={k:cfg.get(k) for k in ['arm','axes_tcp','force','tactile','hold_s']}
        bindings[r.get('session','')]=fingerprint(selected)
    if len(set(bindings.values()))>1:raise ValueError('不同 session 的臂／轴／力／DD／HOLD 设置不兼容，请分开分析')


def figures(out,rows,ds,fit,rr,weighted):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for finger in ('left','right'):
        selected=[d for d in ds if d['finger']==finger and d['cond']=='A' and d['kind']=='scan' and d['recording_complete'] and d['complete'] and not d['archived_attempt'] and d['label']!='invalid']
        fig,axes=plt.subplots(4,2,figsize=(11,13),constrained_layout=True)
        keys=['centroid_u_px','centroid_v_px','bbox_left_px','bbox_right_px','bbox_top_px','bbox_bottom_px','area_px','principal_angle_deg']
        for ax,key in zip(axes.flat,keys):
            for edge,color in [(False,'tab:blue'),(True,'tab:orange')]:
                s=[d for d in selected if d[key] is not None and bool(d['touches_edge'])==edge]
                ax.scatter([d['delta_mm'] for d in s],[d[key] for d in s],s=18,alpha=.6,c=color,label='edge' if edge else 'interior')
            ax.set(xlabel='a offset (mm)',ylabel=key);ax.grid(alpha=.2)
        f=next((v for v in fit['fingers'] if v['finger']==finger),{})
        if 'slope_px_per_mm' in f:
            ax=axes.flat[0 if f['coordinate']=='u' else 1];x=np.array(f['linear_range_mm']);ax.plot(x,f['slope_px_per_mm']*x+f['intercept_px'],'k-',label=f"fit R2={f['r2']:.3f}");ax.legend()
        fig.suptitle('E1 initial PROBE descriptors: '+finger);fig.savefig(out/f'e1_descriptors_{finger}.png',dpi=160);plt.close(fig)
        fig,ax=plt.subplots(figsize=(8,4),constrained_layout=True)
        edges=['left','right','top','bottom']
        for j,edge in enumerate(edges):
            group=[d for d in selected if d.get('edge_sides')]
            ax.scatter([d['delta_mm'] for d in group],[j+(.25 if d['edge_sides'][edge] else 0) for d in group],s=18,alpha=.5,label=edge)
        ax.set(xlabel='a offset (mm)',ylabel='edge (offset +0.25 = touching)',yticks=range(4),yticklabels=edges)
        ax.legend();fig.savefig(out/f'e1_edges_{finger}.png',dpi=160);plt.close(fig)
    for key,title,file in [('success_rate','E2 success (Wilson 95%)','e2_success.png'),('contact_rate','E2 valid contact (Wilson 95%)','e2_contact.png'),('mean_regrasps','Mean regrasp count','e2_regrasps.png')]:
        fig,ax=plt.subplots(figsize=(8,5),constrained_layout=True)
        for cond,color in [('A','tab:blue'),('B','tab:orange')]:
            group=sorted([r for r in rr if r['bottle']=='all' and r['cond']==cond],key=lambda x:x['delta_mm'])
            if not group:continue
            x=[r['delta_mm'] for r in group];y=np.array([r[key] for r in group])
            if key=='mean_regrasps':ax.plot(x,y,'o-',label=cond,color=color)
            else:
                lo='ci_low' if key=='success_rate' else 'contact_ci_low';hi='ci_high' if key=='success_rate' else 'contact_ci_high'
                ax.errorbar(x,y,yerr=proportion_errorbars(y,[r[lo] for r in group],[r[hi] for r in group]),fmt='o-',capsize=3,label=cond,color=color)
        if key!='mean_regrasps':ax.axhline(.8,c='gray',ls='--');ax.set_ylim(-.02,1.02)
        ax.set(xlabel='a offset (mm)',ylabel=key,title=title);ax.grid(alpha=.2)
        if ax.get_legend_handles_labels()[0]:ax.legend()
        fig.savefig(out/file,dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,5),constrained_layout=True)
    for cond in ('A','B'):
        group=[r for r in weighted if r['cond']==cond]
        if group:ax.plot([r['sigma_mm'] for r in group],[r['weighted_success_conditional'] for r in group],'o-',label=cond)
    ax.set(xlabel='Gaussian sigma (mm)',ylabel='Success conditional on observed span',ylim=(-.02,1.02));ax.grid(alpha=.2)
    if ax.get_legend_handles_labels()[0]:ax.legend()
    fig.savefig(out/'e2_weighted_success.png',dpi=160);plt.close(fig)
    # Back-up drift plot keeps chronological zero points and fixed first-three
    # reference; no requirement to invent an mm mapping in a weak E1 dataset.
    fig,axes=plt.subplots(2,1,figsize=(9,6),constrained_layout=True)
    zeros=[r for r in rows if r.get('zero_centroids_px') and r.get('analysis_recording_complete')]
    for i,ax in enumerate(axes):
        for j,axis in enumerate(('u','v')):
            ax.plot(range(1,len(zeros)+1),[r['zero_centroids_px'][i][j] for r in zeros],'o-',label=axis)
        ax.set(xlabel='zero trial in chronological input-session order',ylabel=('left','right')[i]+' centroid (px)');ax.legend();ax.grid(alpha=.2)
    fig.savefig(out/'zero_offset_centroid_drift.png',dpi=160);plt.close(fig)


def analyze(roots,out,*,min_r2=.9):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);rows=load_trials(roots);check_compatibility(rows)
    fit=fit_e1(rows,min_r2=min_r2);ds=descriptor_rows(rows);rr=rates(rows);weighted=gaussian_weighted(rr)
    (out/'e1_fit.yaml').write_text(yaml.safe_dump(plain(fit),allow_unicode=True,sort_keys=False),encoding='utf-8')
    save_csv(out/'probe_descriptors.csv',ds)
    failures=[{'session':r['session'],'id':r['id'],'cond':r['cond'],'delta_mm':r['delta_mm'],'bottle':r['bottle'],
               'initial_status':r.get('initial_status'),'final_status':r.get('status')} for r in rows if r.get('initial_status') in ('no_contact','stall_no_contact') or r.get('status') in ('no_contact','stall_no_contact')]
    save_csv(out/'no_contact_offsets.csv',failures,fields=['session','id','cond','delta_mm','bottle','initial_status','final_status'])
    save_csv(out/'e2_rates.csv',rr);save_csv(out/'e2_weighted_success.csv',weighted)
    interrupted=[{k:r.get(k) for k in ['session','id','cond','bottle','delta_mm','status','error','archived_attempt','trial_path']} for r in rows if not r.get('complete') or r.get('archived_attempt')]
    save_csv(out/'interrupted_attempts.csv',interrupted,fields=['session','id','cond','bottle','delta_mm','status','error','archived_attempt','trial_path'])
    report={'schema':'e1e2_analysis_v1','sessions':[str(Path(x).resolve()) for x in roots],'completed_rows':sum(bool(r.get('complete')) and not r.get('archived_attempt') for r in rows),'attempt_rows_including_interrupted':len(rows),
        'eligible_E2_scan_trials':sum(eligible(r) for r in rows),'excluded':{
            'interrupted_or_archived_attempts':sum(not r.get('complete') or bool(r.get('archived_attempt')) for r in rows),
            'invalid_labels':sum(r.get('label')=='invalid' for r in rows),'incomplete_recordings':sum(not r.get('analysis_recording_complete') for r in rows),
            'bound_skips':sum(r.get('status')=='skipped_bound' for r in rows),'zero_monitors_not_in_scan_denominator':sum(r.get('kind')=='zero_monitor' for r in rows)},
        'success_definition':'LIFT completed AND HOLD completed without any below-contact drop AND human normal AND recording complete',
        'contact_definition':'final bilateral PROBE with valid DD descriptors; initial-contact rate also reported',
        'e1_fit_usable_for_B':fit['usable_for_B'],'task_tolerance_A':tolerance(rr,'A'),'tactile_limit_B':tolerance(rr,'B'),
        'weighted_policy':'Normal(0,sigma) midpoint bins; normalized only within observed range, plus unobserved-tail bounds',
        'warnings':[]}
    if interrupted:report['warnings'].append('存在中断尝试，已在 interrupted_attempts.csv 单列；主成功率只代表完整、有效记录，不能忽略中断偏差')
    if not fit['usable_for_B']:report['warnings'].append('E1 拟合不足／不可靠，B 将拒绝使用此 fit')
    if not any(r['cond']=='B' for r in rows):report['warnings'].append('尚无 B 数据，触觉极限和 A/B 比较尚不可得')
    figures(out,rows,ds,fit,rr,weighted);atomic_json(out/'analysis_report.json',report)
    return report


def main(argv=None):
    p=argparse.ArgumentParser(description='E1/E2 离线分析；多个目录联合比较 A/B')
    p.add_argument('sessions',nargs='+',type=Path);p.add_argument('--out','--output',dest='out',type=Path)
    p.add_argument('--min-r2',type=float,default=.9)
    args=p.parse_args(argv)
    if not 0<args.min_r2<=1:p.error('--min-r2 需在 (0,1]')
    out=args.out or args.sessions[0]/'analysis_e1e2'
    try:
        report=analyze(args.sessions,out,min_r2=args.min_r2)
        print('完成记录：',report['completed_rows'],'；E2有效扫描试验：',report['eligible_E2_scan_trials'])
        print('E1拟合可用于B：',report['e1_fit_usable_for_B'])
        print('任务容差 A：',json.dumps(report['task_tolerance_A'],ensure_ascii=False))
        print('触觉极限 B：',json.dumps(report['tactile_limit_B'],ensure_ascii=False))
        print('结果：',out.resolve())
        for warning in report['warnings']:print('提示：',warning)
        return 0
    except Exception as exc:print('E1/E2分析失败：'+str(exc));return 2
