"""Historical, group-held-out V1--V8 diagnostics; never controls hardware."""
from collections import Counter,defaultdict
from itertools import combinations
import json
import math
from pathlib import Path
import numpy as np
from .noise import fit_noise, transformed, repeat_group

GATES=('object_id','protocol_id','gripper_state','edge_class','force_bin','encoder_version')
ABLATIONS={'old_20':['legacy'],'position_force_open':['pos','force','open'],
           'plus_tactile':['pos','force','open','tac'],'plus_vision':['pos','force','open','tac','vis']}

def clean(x):
    if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)):return [clean(v) for v in x]
    if isinstance(x,np.ndarray):return clean(x.tolist())
    if isinstance(x,np.generic):return clean(x.item())
    if isinstance(x,float) and not math.isfinite(x):return None
    return x

def write(path,x):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(clean(x),ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def distance_matrix(rows,noise,modalities,ignore_protocol=False,diagnostic=False,ignore_time_period=False):
    from .graph import pairwise_distances
    return pairwise_distances(rows,noise,{'modalities':modalities,
        'ignore_protocol':ignore_protocol,'ignore_time_period':ignore_time_period,'optional_edge_orientation':True})


def group_id(r):
    return str(r.get('group_id') or (str(r.get('session'))+'|'+str(r.get('trial_id'))))

def score_ablation(rows,modalities,noise,*,folds=5,diagnostic=False):
    from sklearn.model_selection import GroupKFold
    noise=fit_noise(rows,{'modalities':modalities})
    missing={n:noise.get('modalities',{}).get(n,{}).get('status','missing') for n in modalities
             if noise.get('modalities',{}).get(n,{}).get('status')!='ready'}
    if missing:return {'status':'unavailable','missing_modalities':missing,'V1':None,'V2':None,'V3':None}
    d=distance_matrix(rows,noise,modalities,diagnostic=diagnostic)
    zeros=defaultdict(list)
    for i,r in enumerate(rows):
        if r.get('planned_delta_mm')==0 and r.get('phase') in ('A','B_pre'):
            zeros[repeat_group(r)].append(i)
    pairs=[(i,j) for ids in zeros.values() for i,j in combinations(ids,2) if rows[i]['trial_id']!=rows[j]['trial_id']]
    comparable=[(i,j) for i,j in pairs if np.isfinite(d[i,j])]
    v1={'comparable_pairs':len(comparable),'within_noise_pairs':sum(int(d[i,j]<=1) for i,j in pairs),
        'fraction_within_noise_among_comparable':float(np.mean([d[i,j]<=1 for i,j in comparable])) if comparable else None,
        'gate_or_feature_incomparable_pairs':sum(not np.isfinite(d[i,j]) for i,j in pairs),'active_modalities':len(modalities),'bonferroni_quantile':noise['quantile'],'covariance_multiplier':2,'pairs':len(pairs),'fraction_within_noise':float(np.mean([d[i,j]<=1 for i,j in pairs])) if pairs else None,
        'evaluation':'repeatability on fitted repeat noise; not independent task validation'}
    groups=np.array([group_id(r) for r in rows]);n_groups=len(set(groups))
    predictions=[];inconsistent=[];consistent=0;neighbor_trials=0
    if n_groups>=2:
        for train,test in GroupKFold(n_splits=min(folds,n_groups)).split(rows,groups=groups):
            nm=fit_noise([rows[i] for i in train],{'modalities':modalities})
            dm=distance_matrix(rows,nm,modalities,diagnostic=diagnostic)
            for i in test:
                neighbors=[j for j in train[np.argsort(dm[i,train])] if math.isfinite(dm[i,j])][:5]
                actual=rows[i].get('tcp_offset_a')
                measured=[j for j in neighbors if isinstance(rows[j].get('tcp_offset_a'),(int,float))]
                if isinstance(actual,(int,float)) and measured:
                    prediction=float(np.mean([rows[j]['tcp_offset_a'] for j in measured]))
                    predictions.append({'node_id':rows[i]['node_id'],'actual_mm':actual,'prediction_mm':prediction,'error_mm':abs(prediction-actual),'neighbor_ids':[rows[j]['node_id'] for j in measured]})
                truth=rows[i].get('downstream_success')
                if type(truth) is not bool:continue
                for j in neighbors:
                    other=rows[j].get('downstream_success')
                    if type(other) is not bool:continue
                    neighbor_trials+=1
                    if truth==other:consistent+=1
                    else:inconsistent.append({'node_id':rows[i]['node_id'],'neighbor_id':rows[j]['node_id'],'d_meas':float(dm[i,j]),'within_noise':bool(dm[i,j]<=1)})
    return {'status':'computed','modalities':modalities,'diagnostic_only':diagnostic,'V1':v1,
      'V2':{'mae_mm':float(np.mean([r['error_mm'] for r in predictions])) if predictions else None,
            'evaluated':len(predictions),'total_nodes':len(rows),'group_count':n_groups,'folds':min(folds,n_groups),
            'split':'whole trial/repeat group; PCA and noise refit inside train fold','predictions':predictions},
      'V3':{'agreement':consistent/neighbor_trials if neighbor_trials else None,'neighbor_comparisons':neighbor_trials,'inconsistent':inconsistent,
            'status':'needs_review' if inconsistent else ('consistent' if neighbor_trials else 'unavailable')}}

def _R(graph):
    if isinstance(graph.get('R'),dict):return graph['R']
    ns=graph.get('nodes',[])
    if isinstance(ns,dict):return {k:v.get('R') for k,v in ns.items()}
    return {r['node_id']:r.get('R') for r in ns}

def run_validation(rows,graph,noise,labels,clusters,facts,out_dir,config=None):
    cfg=config or {};out=Path(out_dir);out.mkdir(parents=True,exist_ok=True)
    eligible=[r for r in rows if not r.get('evaluation_only') and r.get('bilateral_contact') is True]
    scores={}
    for name,mods in ABLATIONS.items():
        print('state-graph ablation',name,flush=True)
        scores[name]=score_ablation(eligible,mods,noise,folds=cfg.get('folds',5),diagnostic=name.endswith('diagnostic'))
        write(out/('ablation_'+name+'.json'),scores[name])
    boundary_ids={n for f in facts if f.get('pred') in ('boundary_gap','conflict') for n in f.get('evidence',[])}
    for score in scores.values():
        v3=score.get('V3')
        if v3:
            for row in v3['inconsistent']:row['has_boundary_or_conflict_evidence']=bool({row['node_id'],row['neighbor_id']}&boundary_ids)
            v3['inconsistent_near_recorded_boundary_fraction']=sum(x['has_boundary_or_conflict_evidence'] for x in v3['inconsistent'])/len(v3['inconsistent']) if v3['inconsistent'] else None
    full=scores['position_force_open'];v1=full.get('V1');v2=full.get('V2');base=scores['old_20'].get('V2')
    main1={'status':'unavailable' if not v1 or not v1['pairs'] else ('pass' if v1['fraction_within_noise']>=.9 else 'fail'),'result':v1,'criterion':'repeat fraction >= 0.90'}
    main2={'status':'unavailable','result':v2,'baseline':base,'criterion':'MAE < 1.5 mm and lower than legacy baseline'}
    if v2 and base:
        aa={r['node_id']:r for r in v2['predictions']};bb={r['node_id']:r for r in base['predictions']};common=sorted(set(aa)&set(bb))
        main2['paired_comparison']={'common_node_ids':common,'n_common':len(common),'total_nodes':len(eligible),
            'model_mae_mm':float(np.mean([aa[i]['error_mm'] for i in common])) if common else None,
            'baseline_mae_mm':float(np.mean([bb[i]['error_mm'] for i in common])) if common else None}
        if common:
            paired=main2['paired_comparison'];main2['status']='pass' if paired['model_mae_mm']<1.5 and paired['model_mae_mm']<paired['baseline_mae_mm'] else 'fail'
            main2['scope']='Only common evaluable trial-held-out observations; unavailable gates/features are excluded and coverage is reported.'
    white=[r for r in eligible if str(r.get('object_id','')).startswith('white') and r.get('phase')=='A']
    diagnostic={};diagnostic_facts={}
    from .graph import build_graph
    from .facts import generate_facts
    for name,mods,ignore_time in [('primary',['pos','force','open'],False),('cross_period_diagnostic',['pos','force','open'],True),('optional_tactile',['pos','force','open','tac'],False)]:
        nm=fit_noise(white,{'modalities':mods});dm=distance_matrix(white,nm,mods,ignore_protocol=True,ignore_time_period=ignore_time);pairs=[];cross_floor_pairs=0;comparable_cross_floor_pairs=0
        for i,j in combinations(range(len(white)),2):
            a,b=white[i],white[j]
            if a.get('closure_floor')==b.get('closure_floor') or type(a.get('downstream_success')) is not bool or type(b.get('downstream_success')) is not bool:continue
            cross_floor_pairs+=1
            comparable_cross_floor_pairs+=int(np.isfinite(dm[i,j]))
            if a['downstream_success']!=b['downstream_success'] and dm[i,j]<=1:
                pairs.append({'node_ids':[a['node_id'],b['node_id']],'closure_floor':[a.get('closure_floor'),b.get('closure_floor')],'d_meas':float(dm[i,j])})
        dg=build_graph(white,nm,{'modalities':mods,'ignore_protocol':True,'ignore_time_period':ignore_time})
        df=generate_facts(dg,white,[],{'clusters':[]})
        diagnostic_facts[name]=[f for f in df if f['pred']=='conflict' and (f.get('metrics') or {}).get('attribute')=='closure_floor']
        write(out/('V4_'+name+'_graph.json'),dg);write(out/('V4_'+name+'_facts.json'),df)
        ready=all(nm['modalities'][m]['status']=='ready' for m in mods)
        diagnostic[name]={'status':'computed' if ready else 'unavailable','conflicts':pairs,'protocol_gate_removed_only_for_this_diagnostic':True,'time_period_gate_removed':ignore_time,'cross_floor_pairs':cross_floor_pairs,'comparable_cross_floor_pairs':comparable_cross_floor_pairs}
    from .operators import applicable_operators
    for name,item in diagnostic.items():
        fs=diagnostic_facts[name]
        item['closure_floor_conflict_fact_ids']=[f['id'] for f in fs]
        ops=applicable_operators(fs,cfg.get('spec',{'recordable_attributes':['closure_floor']}),cfg.get('rule',{}))
        item['applicable_operators']=ops
        item['hidden_variable_operator_available']=any(x['op']=='probe_hidden_variable' and x['bindings'].get('var')=='closure_floor' for x in ops)
    V4={'status':'pass' if diagnostic['primary']['conflicts'] and diagnostic['primary']['hidden_variable_operator_available'] else ('fail' if diagnostic['primary']['status']=='computed' and diagnostic['primary']['comparable_cross_floor_pairs'] else 'unavailable'),
        'result':diagnostic,'criterion':'cross-floor conflict plus declared closure_floor operator','diagnostic_only':True,
        'reason':'No cross-floor within-period comparable observations' if not diagnostic['primary']['comparable_cross_floor_pairs'] else 'See measured conflicts and operator evidence'}
    from .boundary import success_boundaries
    V5=success_boundaries(eligible,cfg)
    V6=validate_e1_e2(rows,out,cfg)
    labelmap={r['node_id']:r for r in labels};comparisons=[]
    for r in eligible:
        old=r.get('original_skill_ground_truth') or r.get('legacy_ground_truth') or r.get('skill_ground_truth') or {}
        if old:
            inv={'already_normal':0,'needs_recovery':1,'needs_adjustment':1,'need_recovery':1}.get(old.get('initial_class')) if r.get('window_index')==0 else None
            targets=[w for w in (r.get('original_skill_window_labels') or []) if w.get('timestamp_ns')==r.get('timestamp_ns')]
            beta=targets[0].get('target_reached') if targets else (old.get('final_target_reached') if (r.get('downstream_evidence') or {}).get('evaluated') else None)
            comparisons.append({'node_id':r['node_id'],'existing':old,'I_window_label':inv,'beta_window_label':beta,'graph':labelmap.get(r['node_id'])})
    matrices={'I':[[0,0],[0,0]],'beta':[[0,0],[0,0]]};disagreements=[]
    for row in comparisons:
        new=row['graph'] or {}
        for head,gt,pred in [('I',row['I_window_label'],new.get('invocation_label')),('beta',row['beta_window_label'],new.get('exit_label'))]:
            if gt in (0,1) and pred in (0,1):
                matrices[head][int(gt)][int(pred)]+=1
                if int(gt)!=int(pred):disagreements.append({'node_id':row['node_id'],'head':head,'old':int(gt),'graph':int(pred)})
    V7={'status':'computed' if comparisons else 'unavailable','comparisons':comparisons,'confusion_matrices':matrices,
        'matrix_axes':'rows=old label, columns=graph label; each ordered 0,1','disagreements':disagreements,'original_labels_modified':False}
    result={'dataset_summary':{'nodes':len(rows),'trials':len({(r['session'],r['trial_id']) for r in rows}),'sessions':len({r['session'] for r in rows}),'development_nodes':len(eligible),'evaluation_only_nodes':sum(bool(r.get('evaluation_only')) for r in rows),'missing_mm_position':sum((r.get('sensor_contact_left') or {}).get('u_mm') is None for r in rows),'missing_edge_direction':sum(any(v is None for v in (r.get('edge_class') or {}).values()) for r in rows)},'schema':'state_graph_validation_v2','V1':main1,'V2':main2,
        'V3':{'status':(full.get('V3') or {}).get('status','unavailable'),'result':full.get('V3')},
        'V4':V4,'V5':V5,'V6':V6,'V7':V7,'V8':{'status':'computed','ablations':scores},
        'V9':{'status':'pending','reason':'Symbolic history replay follows graph validation.'},
        'robot_started':False,'old_records_modified':False,
        'limitations':['Historical B is development data under this revision; only explicitly frozen new-campaign independent B stays sealed.',
          '2*Sigma and Bonferroni are formula corrections, not a guarantee of 95% empirical acceptance with finite, non-Gaussian data.',
          'Hardware serial/cover identity is incomplete. Time periods encode provenance cohorts, not verified identical hardware.',
          'Historical squeeze outcomes are not ForcePolicy outcomes. Retrospective replay cannot measure collection efficiency.']}
    result['time_strata']=stratified_summary(eligible,full,noise)
    write(out/'time_strata.json',result['time_strata'])
    result['offline_acceptance_pass']=all(result[k]['status']=='pass' for k in ('V1','V2','V4'))
    write(out/'validation.json',result)
    return result

def render_report(result,path):
    def num(x):return '不可估计' if x is None else f'{x:.3f}'
    ds=result['dataset_summary'];g=result.get('graph_counts',{})
    lines=['# 符号状态图 v2 离线复算','',
      '本版执行用户修订：像素几何、2Σ、Bonferroni、历史 B 纳入开发、同日同 T0 跨 session 连边、区分 unsupported/unreachable、下游控制器与时间时期分层。原数据和 v1 产物保持只读，未启动真机。','',
      f"节点 {ds['nodes']}，试验 {ds['trials']}，session {ds['sessions']}；开发节点 {ds['development_nodes']}，本轮冻结独立节点 {ds['evaluation_only_nodes']}。",'',
      '主距离是两指接触质心（px）/接触像素面积（pixel count）的均值与差值，加六维力和开度。mm、物体 h/θ 为展示信息，缺失不阻塞计算。根尖方向未知时不作根尖硬门控；已知且不同仍区分。触觉编码器、视觉均为可选消融，未重新训练编码器。', '',
      '每种距离按启用模态数 M 使用 χ²(1−0.05/M)，精度矩阵为 (2Σ)⁻¹。协方差和 PCA 仅从训练折拟合；完整试验的修正前后窗口不跨折。','',
      '| 项目 | 结果 |','|---|---|']
    for k in [f'V{i}' for i in range(1,10)]:lines.append(f'| {k} | {result[k]["status"]} |')
    lines+=['','## 图结构','',f'统计：`{g}`',
      '跨 session 运动学边要求当日、同对象/协议/时期，T0 平移差≤0.1 mm、旋转差≤0.1°且 a 轴一致。边仍为待验证假设。',
      'R=∞ 本身不再表示不可恢复：无实测失败转移支持时标 unsupported；已有真实纠正、纠正后失败且图中无成功路径时才标 unreachable。add_axis 需要至少三个独立试验组的此类证据。','',
      '## V1–V3 / V8','', '| 距离 | M | 零点重复一致率 | 留出位置 MAE mm | 预测覆盖 | 结果邻居一致率 |','|---|---:|---:|---:|---:|---:|']
    for name,v in result['V8']['ablations'].items():
        a=v.get('V1') or {};b=v.get('V2') or {};c=v.get('V3') or {}
        lines.append(f'| {name} | {len(v.get("modalities",[]))} | {num(a.get("fraction_within_noise"))} | {num(b.get("mae_mm"))} | {b.get("evaluated",0)}/{b.get("total_nodes",0)} | {num(c.get("agreement"))} |')
    lines+=['','V1 总体比例把门控不可比对数一并列出，不等同于连续距离的统计拒绝率；主距离在可比对中一致率为 '+num((result['V1'].get('result') or {}).get('fraction_within_noise_among_comparable'))+'。V1 是既有零点重复上的拟合后复测，不能替代未来独立重复。V2 新旧距离在共同可评估节点上比较，详见 validation.json。门控导致的不可比对数也单独记录。','',
      '## V4 闭合下限冲突','', '| 范围 | 可比跨下限对 | 冲突对 | closure_floor 操作可用 |','|---|---:|---:|---|']
    for k,v in result['V4']['result'].items():lines.append(f'| {k} | {v["comparable_cross_floor_pairs"]} | {len(v["conflicts"])} | {v["hidden_variable_operator_available"]} |')
    lines+=['','跨时期诊断会明确去掉时期门控，仅用于发现潜在混杂，不能作为训练图或证明闭合下限的单独因果效果。','',
      '## V5 成功概率边界','',
      '按对象、下游控制器、协议、时期和 T0 分组拟合逻辑回归。使用真实 TCP 投影到相同 a 方向；+a=预抓位到 T0 的接近/前移，−a=退让/后移。不给全成功样本拟造边界或外包中心。', '',
      '| 对象 | 下限 | 下游 | 状态 | p=0.5 边界 mm | 95% bootstrap 区间 |','|---|---:|---|---|---:|---|']
    for f in result['V5']['fits']:lines.append(f'| {f["object_id"]} | {f["closure_floor"]} | {f["downstream_controller"]} | {f["status"]} | {num(f.get("boundary_mm"))} | {f.get("boundary_ci95_mm")} |')
    for c in result['V5']['comparisons']:lines.append(f'- 重瓶减轻瓶的边界差：{num(c["heavy_minus_light_boundary_mm"])} mm；95% 区间 {c["difference_ci95_mm"]}。')
    lines+=['','这些数据的下游是 E1/E2 SQUEEZE，不是 ForcePolicy；因此不用于确认或否定部署链路“后移约 2 mm”的观察。样本少且假设单调边界，区间按试验 bootstrap 得到，不能当真机独立验收。','',
      '## V6 按完整试验留出的实测转移','',
      '每折用开发 A 和其余 B 建图，只用该折留出的 B 转移检查边；留出结果不改变训练图。报告的是匹配证据，不把一次 refuted 扩大成运动学假设普遍失效。',
      f"汇总：`{result['V6'].get('summary')}`",'',
      '## 时间分层','', '| 日期 | 节点 | B 节点 | V1 | V2 MAE mm |','|---|---:|---:|---:|---:|']
    for v in result.get('time_strata',[]):lines.append(f'| {v["date"]} | {v["nodes"]} | {v["B_nodes"]} | {num(v["V1"])} | {num(v["V2_mae_mm"])} |')
    lines+=['','## V7 / V9 与后续','',f"旧标签对照矩阵：`{result['V7']['confusion_matrices']}`；旧标签未覆盖。",
      'V9 只检查历史事实、适用操作和时间顺序。预算为 0（离线无新增试验），不再假设 30 次；没有把既有后续采样算作 agent 的采集效率。未请求外部 LLM 时不会生成假决定。',
      'ForcePolicy 数据另采：使用独立 downstream_controller=forcepolicy 协议，记录模型标识、0.330 下限、接管前/纠正后触觉窗口及真实调力/提起/保持结果。采集模板见 forcepolicy_collection_plan.json，本次未执行。', '',
      '完整明细与新旧对照：validation.json、time_strata.json、V6_group_cv.json、V4_*_graph.json、coverage_partition_evidence.json。']
    Path(path).write_text('\n'.join(lines)+'\n')


def stratified_summary(rows,score,noise):
    mods=['pos','force','open'];dm=distance_matrix(rows,noise,mods)
    pred={r['node_id']:r['error_mm'] for r in (score.get('V2') or {}).get('predictions',[])}
    groups=defaultdict(list)
    for i,r in enumerate(rows):groups[r.get('acquisition_day','unknown')].append(i)
    out=[]
    for day,ids in sorted(groups.items()):
        pairs=[(i,j) for i,j in combinations(ids,2) if rows[i].get('planned_delta_mm')==0 and rows[j].get('planned_delta_mm')==0
               and rows[i].get('window_index',0)==rows[j].get('window_index',0)==0
               and repeat_group(rows[i])==repeat_group(rows[j]) and group_id(rows[i])!=group_id(rows[j])]
        errors=[pred[rows[i]['node_id']] for i in ids if rows[i]['node_id'] in pred]
        out.append({'date':day,'time_periods':sorted({rows[i].get('time_period','unknown') for i in ids}),
            'nodes':len(ids),'B_nodes':sum(rows[i].get('phase')!='A' for i in ids),'zero_pairs':len(pairs),
            'V1':float(np.mean([dm[i,j]<=1 for i,j in pairs])) if pairs else None,
            'V2_mae_mm':float(np.mean(errors)) if errors else None,'V2_evaluated':len(errors)})
    return out


def validate_e1_e2(rows,out,cfg):
    from sklearn.model_selection import GroupKFold
    from .graph import build_graph
    development=[r for r in rows if not r.get('evaluation_only') and str(r.get('object_id','')).startswith('green')]
    groups=np.array([group_id(r) for r in development]);n=len(set(groups));folds=[]
    if n<2:return {'status':'unavailable','reason':'Insufficient development groups','summary':{},'B_used_for_training':False}
    for k,(tr,te) in enumerate(GroupKFold(n_splits=min(cfg.get('folds',5),n)).split(development,groups=groups)):
        train=[development[i] for i in tr];held=[development[i] for i in te if development[i].get('phase') in ('B_pre','B_post')]
        held_ids={r['node_id'] for r in held};mods=cfg.get('graph',{}).get('modalities',['pos','force','open'])
        nm=fit_noise(train,{'modalities':mods});gc={**cfg.get('graph',{}),'modalities':mods}
        before=build_graph(train,nm,gc)
        evaluated=build_graph(train,nm,{**gc,'validation_rows':held})
        checked=[]
        for edge in evaluated['edges']+evaluated['refuted_edges']:
            evidence=[v for v in edge.get('validation',[]) if v['source'] in held_ids]
            if not evidence:continue
            checked.append({'edge_id':edge['edge_id'],'status':'confirmed' if all(e['matches'] for e in evidence) else 'refuted','evidence':evidence})
        all_edges=sum(e['kind']=='kinematic' for e in before['edges']+before['refuted_edges'])
        folds.append({'fold':k,'train_groups':sorted(set(groups[tr])),'heldout_groups':sorted(set(groups[te])),
            'train_nodes':[r['node_id'] for r in train],'heldout_B_nodes':sorted(held_ids),
            'train_observed_edges':before['counts']['observed'],'heldout_observed_edges':evaluated['counts']['external_observed_validation'],
            'kinematic_candidates':all_edges,'confirmed':sum(e['status']=='confirmed' for e in checked),
            'refuted':sum(e['status']=='refuted' for e in checked),'unverified_by_heldout':all_edges-len(checked),'checks':checked,
            'training_graph_version':before['graph_version'],'evaluation_graph_version':evaluated['graph_version'],'heldout_outcomes_in_training':False})
    totals={key:sum(f[key] for f in folds) for key in ('confirmed','refuted','unverified_by_heldout','heldout_observed_edges')}
    checked=totals['confirmed']+totals['refuted'];totals['tested_fraction']=checked/(checked+totals['unverified_by_heldout']) if checked+totals['unverified_by_heldout'] else None
    totals['refuted_fraction_of_tested']=totals['refuted']/checked if checked else None
    result={'status':'computed','folds':folds,'summary':totals,'B_used_for_training':True,
        'meaning':'Edge-fold counts; heldout whole-trial B tests only, not independent campaign effectiveness.'}
    write(Path(out)/'V6_group_cv.json',result)
    return result
