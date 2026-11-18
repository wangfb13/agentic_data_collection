"""Grouped offline diagnostics. Retrospective evidence is not independent deployment success."""
from copy import deepcopy
from collections import defaultdict
import numpy as np
from sklearn.linear_model import Ridge,LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from .common import group_folds,group_id,independent,write
from .noise import fit_noise,transform,pairwise_distances,drift_report
from .dynamics import fit_dynamics,evaluate_dynamics,transitions
from .sink import heldout_probability
from .graph import build_graph
from .labels import generate_labels
from .skill import train_models
from .value import predict_value
from .policy import predict_policy
from .facts import shared_ready_regions,coordinate_evidence


def auc(y,p):return float(roc_auc_score(y,p)) if len(y)>1 and len(set(y))>1 else None

def mean(values):return float(np.mean(values)) if values else None

def reliability(samples):
    bins=[]
    for lo,hi in zip(np.linspace(0,1,6)[:-1],np.linspace(0,1,6)[1:]):
        part=[s for s in samples if lo<=s['p_hat']<hi or hi==1 and s['p_hat']==1]
        bins.append({'interval':[float(lo),float(hi)],'n':len(part),'predicted':mean([s['p_hat'] for s in part]),'observed':mean([s['success'] for s in part])})
    return {'status':'computed' if samples else 'unavailable','n':len(samples),'brier':mean([(s['p_hat']-s['success'])**2 for s in samples]),'bins':bins,'samples':samples}


def grouped_diagnostics(rows,spec,out):
    out.mkdir(parents=True,exist_ok=True);folds=group_folds(rows,spec.get('validation',{}).get('folds',5))
    modalities=list(spec['observation']['modalities']);coords=sorted(k for k in spec.get('perturbation_variables',{}) if any(coordinate_evidence(r,k) is not None for r in rows))
    probes={n:{'coordinates':{v:[] for v in coords},'post_conditions':{p['id']:[] for p in spec['post_conditions']}} for n in modalities}
    held_repeats=[];baseline_scores={p['id']:[] for p in spec['post_conditions']}
    calibration={p['id']:[] for p in spec['post_conditions']};dynamics=[];model_checks={p['id']:[] for p in spec['post_conditions']};evidence=[]
    for fold,(tr,te) in enumerate(folds):
        print('grouped validation fold',fold+1,len(folds),flush=True)
        train=[rows[i] for i in tr];held=[rows[i] for i in te];nm=fit_noise(train,spec)
        evidence.append({'fold':fold,'train_groups':sorted({group_id(r) for r in train}),'heldout_groups':sorted({group_id(r) for r in held}),'noise_fit_node_ids':nm['fit_node_ids']})
        for name in modalities:
            if nm['modalities'][name]['status']!='ready':continue
            X=np.array([transform(r,name,nm) for r in train]);Z=np.array([transform(r,name,nm) for r in held]);scale=StandardScaler().fit(X);X=scale.transform(X);Z=scale.transform(Z)
            for var in coords:
                a=[i for i,r in enumerate(train) if coordinate_evidence(r,var) is not None]
                b=[i for i,r in enumerate(held) if coordinate_evidence(r,var) is not None]
                if len(a)<3 or not b:continue
                model=Ridge(alpha=1.).fit(X[a],[coordinate_evidence(train[i],var) for i in a]);pred=model.predict(Z[b])
                probes[name]['coordinates'][var].extend({'node_id':held[i]['node_id'],'actual':coordinate_evidence(held[i],var),'prediction':float(v),'error':abs(float(v)-coordinate_evidence(held[i],var))} for i,v in zip(b,pred))
            for p in spec['post_conditions']:
                cid=p['id'];a=[i for i,r in enumerate(train) if r['is_terminal'] and r['post'][cid]['evaluated']];b=[i for i,r in enumerate(held) if r['is_terminal'] and r['post'][cid]['evaluated']]
                y=[int(train[i]['post'][cid]['success']) for i in a]
                if len(set(y))<2 or not b:continue
                classifier=Ridge(alpha=1.).fit(X[a],y);pred=classifier.predict(Z[b])
                probes[name]['post_conditions'][cid].extend({'node_id':held[i]['node_id'],'success':int(held[i]['post'][cid]['success']),'prediction':float(v)} for i,v in zip(b,pred))
        H=pairwise_distances(held,nm)
        for i,a in enumerate(held):
            for j,b in enumerate(held[:i]):
                if a.get('window_index',0)==b.get('window_index',0)==0 and a.get('repeat_key') is not None and a.get('repeat_key')==b.get('repeat_key') and group_id(a)!=group_id(b):
                    held_repeats.append({'fold':fold,'nodes':[a['node_id'],b['node_id']],'distance':float(H[i,j]),'within_noise':bool(H[i,j]<=1)})
        for p in spec['post_conditions']:calibration[p['id']]+=heldout_probability(held,train,nm,spec,p['id'])
        if not nm['active_modalities'] or not all(nm['modalities'][n]['status']=='ready' for n in nm['active_modalities']):continue
        f=fit_dynamics(train,nm,spec,seed=fold);fd=evaluate_dynamics(f,held,spec);dynamics+=fd['samples']
        # Held-out topology/labels are evaluation targets only. They are never
        # used in fitting V, policy, dynamics, PCA or repeated-noise estimates.
        g=build_graph(train,nm,spec,f);labels=generate_labels(g,train,spec)
        bundle,_=train_models(train,g,labels,nm,f,spec)
        held_graph=build_graph(held,nm,spec,f);held_labels=generate_labels(held_graph,held,spec)
        lookup={(l['post_condition'],l['node_id']):l for l in held_labels['nodes']}
        for p in spec['post_conditions']:
            cid=p['id'];ms=bundle['models'][cid]
            terminal_train=[r for r in train if r.get('is_terminal') and r['post'][cid]['evaluated']]
            if terminal_train:
                from .noise import feature_vector
                features=np.array([feature_vector(r,nm) for r in terminal_train]);truth=np.array([r['post'][cid]['success'] for r in terminal_train],float)
                scale=StandardScaler().fit(features);direct=Ridge(alpha=1).fit(scale.transform(features),truth)
                for r in held:
                    if not r.get('is_terminal') or not r['post'][cid]['evaluated']:continue
                    pv=predict_value(ms['V'],r,nm);x=feature_vector(r,nm)[None]
                    baseline_scores[cid].append({'node_id':r['node_id'],'truth':int(r['post'][cid]['success']),'constant':float(truth.mean()),'direct':float(np.clip(direct.predict(scale.transform(x))[0],0,1)),
                                                'graph_value_probability':pv.get('p_hat')})
            train_values=[x['value_target'] for x in labels['nodes'] if x['post_condition']==cid and x['value_target'] is not None]
            for row in held:
                label=lookup[cid,row['node_id']];prediction=predict_value(ms['V'],row,nm)
                if label['value_target'] is not None and prediction['status']=='predicted':
                    model_checks[cid].append({'kind':'value','node_id':row['node_id'],'actual':label['value_target'],'prediction':prediction['V'],'constant_error':abs(np.mean(train_values)-label['value_target']) if train_values else None,'error':abs(prediction['V']-label['value_target'])})
            for t in transitions(held,spec):
                a,b=held_graph['R'][cid][t['source']['node_id']],held_graph['R'][cid][t['target']['node_id']]
                prediction=predict_policy(ms['pi'],t['source'],nm)
                if a is None or b is None or prediction['status']!='predicted':continue
                difference=np.asarray(prediction['u'])-t['u'];tolerance=np.asarray(spec['task']['action_space'].get('evaluation_match_tolerance',np.asarray(spec['safety_envelope']['max_action'])*.1))
                model_checks[cid].append({'kind':'policy','source':t['source']['node_id'],'target':t['target']['node_id'],'predicted_u':prediction['u'],'logged_u':t['u'],
                                         'action_error':float(np.linalg.norm(difference)),'logged_R_decreased':b<a,'action_supported_by_logged_transition':bool(np.all(abs(difference)<=tolerance)),
                                         'scope':'Approximate action agreement on held-out logged transitions, not a counterfactual execution.'})
        write(out/f'fold_{fold}.json',{'split':evidence[-1],'dynamics':fd,'train_graph':g['graph_version'],'heldout_graph':held_graph['graph_version'],'fit_groups':f.get('fit_groups',[])})
    V2={}
    for name,p in probes.items():
        V2[name]={'coordinates':{var:{'mae':mean([s['error'] for s in values]),'n':len(values),'predictions':values} for var,values in p['coordinates'].items()},
                  'post_conditions':{cid:{'auc':auc([s['success'] for s in values],[s['prediction'] for s in values]),'n':len(values),'predictions':values} for cid,values in p['post_conditions'].items()}}
    V6={'status':'computed' if dynamics else 'unavailable','n':len(dynamics),'endpoint_within_noise':mean([s['correct_within_noise'] for s in dynamics]),
        'variance_augmented_within_noise':mean([s['variance_augmented_distance']<=1 for s in dynamics]),
        'sigma_2x_coverage':mean([float(np.mean(s['within_2sigma'])) for s in dynamics]),
        'p_off_auc':auc([s['off_label'] for s in dynamics],[s['p_off'] for s in dynamics]),'samples':dynamics,'raw_component_mae':np.mean([s['absolute_error'] for s in dynamics],axis=0).tolist() if dynamics else None,
        'baselines':{name:{'raw_component_mae':np.mean([s['baseline_absolute_error'][name] for s in dynamics],axis=0).tolist() if dynamics else None} for name in ('identity','mean_delta','linear')},
        'interpretation':'Sigma is ensemble spread; reported coverage is measured, not assumed calibrated. Off-event labels are model residuals, not causal event labels.'}
    V8={}
    for cid,checks in model_checks.items():
        val=[s for s in checks if s['kind']=='value'];pol=[s for s in checks if s['kind']=='policy'];supported=[s for s in pol if s['action_supported_by_logged_transition']]
        V8[cid]={'constant_value_mae':mean([s['constant_error'] for s in val if s['constant_error'] is not None]),'value_mae':mean([s['error'] for s in val]),'value_n':len(val),'policy_action_mae':mean([s['action_error'] for s in pol]),'policy_n':len(pol),
                 'logged_R_descent_rate':mean([s['logged_R_decreased'] for s in pol]),'policy_supported_n':len(supported),
                 'policy_supported_R_descent_rate':mean([s['logged_R_decreased'] for s in supported]),'checks':checks}
    baseline_report={cid:{name:{'brier':mean([(s[name]-s['truth'])**2 for s in values if s[name] is not None]),'n':sum(s[name] is not None for s in values)} for name in ('constant','direct','graph_value_probability')} for cid,values in baseline_scores.items()}
    return {'heldout_repeatability':{'pairs':len(held_repeats),'fraction_within_noise':mean([s['within_noise'] for s in held_repeats]),'samples':held_repeats},'outcome_baselines':baseline_report,'V2':{'status':'computed','probes':V2,'split':'Whole trial groups; train-only scaling/PCA; outcome scores come from an auxiliary Ridge probe; AUC uses its unbounded linear score.'},
            'V3':{c:reliability(values) for c,values in calibration.items()},'V6':V6,'V8':V8,'fold_evidence':evidence}


def run_validation(rows,spec,noise,graph,labels,clusters,facts,out):
    from pathlib import Path
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    D=pairwise_distances(rows,noise);zero=defaultdict(list)
    for i,r in enumerate(rows):
        if r.get('is_reference') and r.get('repeat_key') is not None and r.get('window_index',0)==0:zero[r['repeat_key']].append(i)
    pairs=[(a,b) for group in zero.values() for j,a in enumerate(group) for b in group[:j] if group_id(rows[a])!=group_id(rows[b])]
    V1={'status':'computed' if pairs else 'unavailable','pairs':len(pairs),'fraction_within_noise':mean([D[a,b]<=1 for a,b in pairs]),'incomparable':sum(not np.isfinite(D[a,b]) for a,b in pairs),
        'scope':'training_fit_check_only','notes':'Use heldout_repeatability for evaluation; this fitted repeat check is not independent validation.'}
    grouped=grouped_diagnostics(rows,spec,out/'group_cv')
    posts=[p['id'] for p in spec['post_conditions']];joint=[]
    for i,c in enumerate(posts):
        for d in posts[:i]:
            common=[r['node_id'] for r in rows if graph['probabilities'][c][r['node_id']]['n'] and graph['probabilities'][d][r['node_id']]['n']]
            joint.append({'conditions':[c,d],'supported_nodes':len(common),'mean_probability_difference':mean([graph['probabilities'][c][n]['p_hat']-graph['probabilities'][d][n]['p_hat'] for n in common])})
    V4={'status':'computed','comparisons':joint,'hidden_protocol_variables':[f for f in facts if f['pred']=='hidden_variable_candidate' and spec['perturbation_variables'].get(f['args'][1],{}).get('kind')=='protocol'],
        'note':'A parameter that uniquely determines post-condition identity cannot be tested within that condition by Fisher; cross-condition estimates remain descriptive.'}
    V5={'status':'computed','stochastic_regions':[f for f in facts if f['pred']=='stochastic_outcome'],
        'hidden_properties':[f for f in facts if f['pred']=='hidden_variable_candidate' and spec['perturbation_variables'].get(f['args'][1],{}).get('kind')=='object_property'],
        'shared_ready':{c:{v:shared_ready_regions(rows,clusters,spec,v,c) for v in spec.get('task',{}).get('hidden_object_properties',[])} for c in posts}}
    V7={'status':'computed','original_labels_modified':False,'comparisons':[]}
    for row in rows:
        old=row.get('legacy_labels')
        if old:V7['comparisons'].append({'node_id':row['node_id'],'legacy':old,'graph_labels':[x for x in labels['nodes'] if x['node_id']==row['node_id']]})
    without=build_graph(rows,noise,spec,None)
    for cid,v in grouped['V8'].items():
        v['unsupported_with_inference']=mean([s['status']=='unsupported' for s in graph['reachability'][cid].values()]);v['unsupported_without_inference']=mean([s['status']=='unsupported' for s in without['reachability'][cid].values()])
    result={'schema':'state_graph_validation_v3','task_id':graph['task_id'],'nodes':len(rows),'trials':len({group_id(r) for r in rows}),
            'active_post_condition':spec['active_post_condition'],'post_evidence':{c:sum(r['is_terminal'] and r['post'][c]['evaluated'] for r in rows) for c in posts},
            'graph_counts':graph['counts'],'V1':V1,**grouped,'V4':V4,'V5':V5,'V7':V7,
            'V9':{'status':'pending'},'V10':cross_condition_generalization(rows,spec,out),
            'drift':drift_report(rows,noise),'robot_started':False,'independent_campaign_effectiveness_proven':False,
            'limitations':['Historical grouped cross-validation is development evidence only.','Low-dimensional dynamics distance uses only declared predicted components; other modalities are not claimed to be predicted.',
                           'No post-condition observations means no fitted readiness model for that consumer.','Task-specific aliases are retained only for compatibility and never read by models.']}
    write(out/'validation.json',result);return result


def cross_condition_generalization(rows,spec,out):
    variable=spec.get('generalization',{}).get('variable')
    if not variable:return {'status':'unavailable','reason':'No generalization variable specified'}
    values=sorted({str(r['attrs'].get(variable)) for r in rows if r['attrs'].get(variable) not in (None,'unrecorded','unknown')})
    if len(values)<2:return {'status':'unavailable','variable':variable,'recorded_levels':values,'reason':'At least two recorded conditions are required; historical missing values are not inferred.'}
    reports=[]
    for level in values:
        held=[r for r in rows if str(r['attrs'].get(variable))==level];train=[r for r in rows if str(r['attrs'].get(variable)) in values and str(r['attrs'].get(variable))!=level]
        held_groups={group_id(r) for r in held};train=[r for r in train if group_id(r) not in held_groups]
        noise=fit_noise(train,spec)
        if not train or not all(m['status']=='ready' for m in noise['modalities'].values()):
            reports.append({'heldout_level':level,'status':'insufficient_train_noise'});continue
        f=fit_dynamics(train,noise,spec);g=build_graph(train,noise,spec,f);labels=generate_labels(g,train,spec);bundle,_=train_models(train,g,labels,noise,f,spec)
        hg=build_graph(held,noise,spec,f);errors=[];cid=spec['active_post_condition']
        for row in held:
            pred=predict_value(bundle['models'][cid]['V'],row,noise);truth=hg['R'][cid][row['node_id']]
            if truth is not None and pred['status']=='predicted':errors.append(abs(pred['V']-truth))
        reports.append({'heldout_level':level,'train_groups':sorted({group_id(r) for r in train}),'heldout_groups':sorted(held_groups),
                        'dynamics':evaluate_dynamics(f,held,spec),'value_mae':mean(errors),'value_n':len(errors)})
    return {'status':'computed','variable':variable,'directions':reports}


def render_report(reports,fixture,path):
    lines=['# 通用符号状态图 V3 离线报告','',
           '本次只做代码、历史数据分析与合成通用性验证。未启动机器人；没有把历史后置条件结果改写为未实测消费者的结果。','',
           '主距离由规格声明的观测组成；任务是唯一门控，协议、时期、隐藏物体属性仅记录。只有有实测支持的 Ready 节点以零代价连到汇点；相似边不参与恢复距离。Beta 平滑概率与直接交接风险另报。运动学边已移除，推断边来自观测动力学。','',
           'V、π、f 的全部设计矩阵仅来自 obs 和 f 所需的动作。坐标只用于探针目标和实验规划，记录属性只用于分析。','',
           '重复噪声额外修正了估计组均值消耗的自由度：每组残差乘 sqrt(n/(n−1))，再做 Ledoit–Wolf 收缩及 2Σ。否则只有两次重复时，Σ 会被低估一半。本轮所有真实/合成任务及折内估计统一使用此公式；旧版产物不变。','']
    for report in reports:
        lines += ['## '+report['task_id'],'',f"节点 {report['nodes']}，试验组 {report['trials']}；边数 {report['graph_counts']}。",f"后置条件实测终止窗口：{report['post_evidence']}。",'',
                  f"V1 零点重复：{report['V1']}。",'', '| 模态 | 坐标 | 留出 MAE | n |','|---|---|---:|---:|']
        for modality,p in report['V2']['probes'].items():
            for coordinate,v in p['coordinates'].items():lines.append(f"| {modality} | {coordinate} | {v['mae']} | {v['n']} |")
        lines+=['','V3 是训练邻域对留出终止样本的概率校准，可靠性图另存；没有邻域证据的样本单列缺失。',f"V4 多后置条件：{report['V4']['comparisons']}。",f"V5 随机区域 {len(report['V5']['stochastic_regions'])}，显著隐藏属性证据 {len(report['V5']['hidden_properties'])}。",'',
                f"V6：{ {k:v for k,v in report['V6'].items() if k!='samples'} }",'',
                '| 后置条件 | V 留出误差 | 策略支持的实测转移数 | 支持转移 R 下降率 | unsupported（有/无推断） |','|---|---:|---:|---:|---|']
        for cid,v in report['V8'].items():lines.append(f"| {cid} | {v['value_mae']} | {v['policy_supported_n']} | {v['policy_supported_R_descent_rate']} | {v['unsupported_with_inference']} / {v['unsupported_without_inference']} |")
        lines+=['','策略支持指预测动作与已记录动作在声明容差内；这里没有实际执行预测动作，不能当作策略的真机成功率。',
                f"V7 旧标签对照 {len(report['V7']['comparisons'])} 个窗口，旧标签未修改。",f"V9：{report['V9']['status']}。历史回放不证明采集效率。",f"V10：{report['V10']['status']}，{report['V10'].get('reason','见详细分条件结果')}。",'']
    lines+=['## 图表','', '![重复一致率与坐标探针](figures/repeatability_and_probes.png)','',
            '![留出可靠性图](figures/reliability.png)','', '![动力学与不确定度校准](figures/dynamics.png)','',
            '## V11 框架通用性','',str(fixture),'',
            '二维合成实例仅证明规格接口、动作维度、多后置条件与模型流程可运行；不代替真实新任务实验。', '',
            '## 下一步真机数据','',
            '按实例规格记录消费者版本、每次修正前后的观测、实际动作回执和 L1/L2/L3 结果；隐藏属性按块平衡；至少两个已命名接近姿态；同一盖板下成对比较参数。冻结模型及评估协议后再采独立 B。',
            '对低维动力学未预测的视觉等分量，推断边匹配只使用声明的预测分量；可选规划暂存其他观测不变，是近似而非真实视觉动力学。']
    Path=__import__('pathlib').Path;Path(path).write_text('\n'.join(lines)+'\n')
