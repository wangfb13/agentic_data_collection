"""Rebuild both graph revisions from immutable historical windows, holding out entire trials."""
import argparse
from collections import Counter
from pathlib import Path
import json
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from bottle_scan.state_graph import cli
from bottle_scan.state_graph.common import group_folds,group_id,independent,digest,write
from bottle_scan.state_graph.noise import fit_noise
from bottle_scan.state_graph.sink import heldout_probability,threshold
from bottle_scan.state_graph.v5_profile import configure,protect_output


def counts(entries):
    return dict(sorted(Counter(entries).items()))


def summarize(rows,spec,graph,clusters,cid,folds,labels):
    reach=graph['reachability'][cid]
    def outcomes(test,train):
        predicted=heldout_probability(test,train,fit_noise(train,spec),spec,cid)
        lookup={r['node_id']:r for r in predicted}
        cond=next(c for c in spec['post_conditions'] if c['id']==cid)
        return [dict(node=r['node_id'],group=group_id(r),accepted=bool(p and
                    p.get('ready',p['n_train_neighbors']>=cond.get('min_ready_trials',3) and p['p_hat']>=cond['ready_p'])),
                    success=r['post'][cid]['success'],**({k:p[k] for k in ('h_ready','threshold')} if p and 'h_ready' in p else {})) for r in test if r['is_terminal'] and r['post'][cid]['evaluated']
                for p in [lookup.get(r['node_id'])]]
    tests=[];split=[]
    for tr,te in folds:
        train=[rows[i] for i in tr];test=[rows[i] for i in te]
        a={group_id(r) for r in train};b={group_id(r) for r in test}
        assert a.isdisjoint(b)
        predictions=outcomes(test,train);tests.extend(predictions)
        split.append({'test_groups':sorted(b),'n_train_groups':len(a)})
    edges={e['edge_id']:e for e in graph['edges']}
    sources=[]
    for label in labels['nodes']:
        if label['post_condition']!=cid or label['status']!='recoverable':continue
        source=reach[label['node_id']].get('action_source')
        if not source:
            path=[edges[e] for e in label.get('path_edge_ids',[]) if edges[e]['kind']!='sink']
            source='uses_inferred' if any(e['kind']=='inferred' for e in path) else 'observed' if path else 'unresolved'
        sources.append(source)
    return dict(states=counts(v['status'] for v in reach.values()),
                symbols=counts(c['symbols'][cid].split('(')[0] for c in clusters['clusters']),
                recover_sources=counts(sources),
                error_release=sum(x['accepted'] and not x['success'] for x in tests),
                correct_release=sum(x['accepted'] and x['success'] for x in tests),total_tested=len(tests),
                actual_successes=sum(x['success'] for x in tests),heldout=tests,folds=split,
                radii=graph.get('radii'),resolution=graph.get('resolution'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/state_graph_v5_history_comparison.json')
    parser.add_argument('--out');parser.add_argument('--report',default='docs/experiments/state_graph_v5_comparison.md')
    args=parser.parse_args();cfg=json.loads(Path(args.config).read_text());src=Path(cfg['source'])
    out=Path(args.out or (cfg['output']+'_H'));protect_output(out,kind='historical_comparison')
    allrows=[json.loads(x) for x in (src/'nodes.jsonl').read_text().splitlines() if x.strip()]
    cid=cfg['primary_post_condition']; results={};lines=['# 状态图 v4、v5 与 v5-H 的历史对照','',
        f"数据来自 `{src}/nodes.jsonl`。按试次组做 {cfg['folds']} 折交叉验证，同一试次的所有窗口只进入一折。每一折用训练组重新估计噪声、尺度和 Ready 邻域。",
        '',f'主表下游为 `{cid}`。正确放行指留出终端实际成功且被 Ready 门放行，错误放行指留出终端实际失败却被放行。表中分母含无近邻和证据不足的终端。',
        '','| 物体 | 版本 | Ready | Recoverable | Unreachable | Unsupported | 正确放行 | 错误放行 | 留出成功 / 可评估 |','|---|---|---:|---:|---:|---:|---:|---:|---:|']
    detail=[]
    for obj in cfg['objects']:
        raw=json.loads((src/obj/'spec.json').read_text())
        rows=[r for r in allrows if not independent(r) and r['task_id'].split('|')[0]==obj]
        folds=group_folds(rows,cfg['folds']);results[obj]={}
        for label,version,h in [('v4',4,False),('v5',5,False),('v5-H',5,True)]:
            spec=configure(raw,cfg['overrides'] if version==5 else {},version=version,revision_h=h)
            spec['active_post_condition']=cid
            dest=out/obj/label
            print(obj,version,'rebuild',flush=True)
            built=cli.build(rows,spec,dest,validate=False);g=built['graph']
            s=summarize(rows,spec,g,built['clusters'],cid,folds,built['labels'])
            s['ready_calibration']=g.get('ready_calibration',{}).get(cid)
            s['metric_axes']=json.loads((dest/'noise_model.json').read_text()).get('tac_metric',{}).get('axis_reports')
            results[obj][label]=s;write(dest/'heldout_comparison.json',s)
            ns=s['states']; lines.append(f"| {obj} | {label} | "+' | '.join(str(ns.get(k,0)) for k in
                ('ready','recoverable','unreachable','unsupported'))+f" | {s['correct_release']} | {s['error_release']} | {s['actual_successes']} / {s['total_tested']} |")
            detail+=['',f'### {obj} · {label}','',f"共 {len(rows)} 个窗口、{len(set(map(group_id,rows)))} 个试次组。簇符号分布为 `{s['symbols']}`。可恢复标签来源为 `{s['recover_sources']}`。",
                '',f"有效半径为 `{s['radii'] or 'v4，统一 D ≤ 1'}`。该次实际解析的全部开关和数值保存在 `{dest}/spec.json`，节点、标签及逐样本留出预测位于同目录。"]
            if h:
                m=s['ready_calibration'];ns=s['states']
                detail+=['',f"核带宽 {m['h_ready']}，后验下界门槛 {m['threshold']}，校准试次组 {m['training_groups']}。",
                         '',f"逐轴尺度与可观测判定 `{json.dumps({k:{f:v.get(f) for f in ('observable','scale','used','reason','action_median','repeat_median','n_actions','n_repeat_pairs')} for k,v in s['resolution']['by_axis'].items()},ensure_ascii=False)}`。",
                         '',f"度量头逐轴结果 `{s['metric_axes']}`。历史对照采用已有特征。"]
                if obj=='green_bottle':
                    v4=results[obj]['v4'];tol=spec['report']['green_correct_release_relative_tolerance']
                    s['acceptance']=dict(correct_release_min=v4['correct_release']*(1-tol),error_release_max=v4['error_release'],passed=s['correct_release']>=v4['correct_release']*(1-tol) and s['error_release']<=v4['error_release'])
                    detail+=['',f"按预设容差 {tol:.0%}，留出放行验收 `{s['acceptance']}`。"]
                if obj=='white_bottle':detail+=['',f"白瓶 Ready 节点 {ns.get('ready',0)} 个。带宽和下界均由训练组终端证据选取，未取得可用门槛时所有节点拒绝放行。"]
        assert results[obj]['v4']['folds']==results[obj]['v5']['folds']==results[obj]['v5-H']['folds']
    lines+=detail+['','## 证据范围','',
        '三版输入均沿用历史窗口实际保存的触觉几何、力与腕图特征。历史表没有指面毫米标定、逐图块读出、同一试次配对的双下游结果或新导演的事前预测，因此这些功能只做合成验证，历史 McNemar 和预测准确率均不可估计。v4 的来源从标签的实际路径回查。uses_inferred 表示路径包含模型推断，observed 表示整条路径仅含实测转移。','','## 复现','',
        '从仓库根目录运行 `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/compare_state_graph_versions.py`。',
        '',f"历史输入内容摘要为 `{digest(allrows)}`。报告数值只适用于上述下游与已有对象。"]
    lines+=['','## Ready 收缩发生在哪里','']
    for obj in cfg['objects']:
        path=out/obj/'v5'
        fitted=json.loads((path/'graph.json').read_text())
        conf=json.loads((path/'spec.json').read_text())
        cutoff=next(s['ready_p'] for s in conf['post_conditions'] if s['id']==cid)
        pr=list(fitted['probabilities'][cid].values())
        if pr:
            lines.append(f"- {obj} 的全量图中，单个 Ready 邻域最多含 {max(p['n'] for p in pr)} 个独立终端组。最高后验下界为 {max(p['p_lower'] for p in pr):.3f}，放行门槛为 {cutoff:g}。")
    profile_dir=Path('bottle_scan/state_graph/profiles')
    lines+=['','## 参数','',f"历史特征适配覆盖值为 `{json.dumps(cfg['overrides'],ensure_ascii=False)}`。"]
    for profile in sorted(profile_dir.glob('*.json')):
        lines+=['',f'**{profile.stem}**','','```json',json.dumps(json.loads(profile.read_text()),ensure_ascii=False,indent=2),'```']
    lines+=['','## 留出与全量图的区别','',
        '留出预测使用折内估计的噪声和动作尺度，图节点计数来自全量重建。两者的邻域可以不同。错误放行减少时，应同时查看正确放行数和拒绝覆盖，不能据此单独判定部署性能提升。']
    path=Path(args.report);path.parent.mkdir(exist_ok=True,parents=True);path.write_text('\n'.join(lines)+'\n')
    write(out/'comparison.json',results)


if __name__=='__main__':main()

