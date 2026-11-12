"""Opt-in offline state-graph pipeline; no robot connection or execution."""
import argparse
from copy import deepcopy
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import subprocess
import time

from .report import clean, write, run_validation, render_report

ROOT=Path(__file__).resolve().parents[3]
DEFAULT_OUT='outputs/state_graph/20261112_symbolic_v2'
DEFAULT_CONFIG='config.state_graph.symbolic_v2_20261112.json'

def read_rows(path):
    p=Path(path)
    if p.suffix=='.jsonl':return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
    if p.suffix=='.parquet':
        import pyarrow.parquet as pq
        rows=pq.read_table(p).to_pylist()
        for row in rows:
            for k,v in list(row.items()):
                if isinstance(v,str) and v[:1] in ('{','['):
                    try:row[k]=json.loads(v)
                    except json.JSONDecodeError:pass
        return rows
    return json.loads(p.read_text())

def save_labels(labels,path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    # JSON nested evidence keeps heterogeneous graph receipt fields lossless.
    flat=[{k:json.dumps(clean(v),ensure_ascii=False) if isinstance(v,(list,dict)) else v for k,v in r.items()} for r in labels]
    pq.write_table(pa.Table.from_pylist(flat),path)

def build_artifacts(rows,out,cfg):
    from .noise import fit_noise
    from .graph import build_graph
    from .labels import generate_labels
    from .clusters import cluster_nodes
    from .facts import generate_facts
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    dev=[r for r in rows if not r.get('evaluation_only') and r.get('bilateral_contact') is True]
    nm=fit_noise(dev,cfg.get('noise',{}))
    g=build_graph(dev,nm,cfg.get('graph',{}))
    labels=generate_labels(g,dev)
    clusters=cluster_nodes(g,dev,nm,cfg.get('graph',{}))
    facts=generate_facts(g,dev,labels,clusters,dict(cfg.get('facts',{})))
    from .skill import coverage_facts
    coverage=coverage_facts(dev,cfg.get('training',{}))
    facts=[f for f in facts if f['pred']!='coverage_ok']+coverage['facts']
    write(out/'coverage_partition_evidence.json',coverage)
    for name,value in [('noise_model',nm),('graph',g),('labels',labels),('clusters',clusters),('facts',facts)]:write(out/(name+'.json'),value)
    save_labels(labels,out/'labels.parquet')
    write(out/'config_snapshot.json',cfg)
    return nm,g,labels,clusters,facts

def history_replay(rows,out,cfg,allow_agent=False):
    """Prefix-only chronological replay at collection-batch checkpoints.

    Session ordering uses saved baseline UTC; within a session timestamps are
    monotonic sensor clocks. The replay does not pretend its proposals caused
    the already recorded subsequent experiments.
    """
    from bottle_scan.skill_director import symbolic_context,advance_symbolic
    from .director_bridge import update_hypotheses
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    replay_run=out/f'run_{time.time_ns()}'
    replay_run.mkdir()
    trials=defaultdict(list)
    for row in rows:
        if not row.get('evaluation_only') and row.get('bilateral_contact') is True:trials[(row['session'],row['trial_id'])].append(row)
    def utc(key):
        return min((r.get('trial_started_unix_ns') or r.get('session_created_unix_ns') or 2**63-1) for r in trials[key]),key
    ordered=sorted(trials,key=utc);prefixes=[];events=[];hypotheses=[];history=[]
    settings=cfg.get('provider_settings') or {}
    if allow_agent and cfg.get('provider_config'):
        import yaml
        settings=yaml.safe_load(Path(cfg['provider_config']).read_text())['planner']['provider']
    settings=dict(settings)
    if cfg.get('provider_container'):settings['container']=cfg['provider_container']
    if os.environ.get('SYMBOLIC_AGENT_PROXY'):settings['proxy']=os.environ['SYMBOLIC_AGENT_PROXY']
    llm_available=allow_agent
    stride=int(cfg.get('replay_every_trials',20))
    for end in range(stride,len(ordered)+stride,stride):
        keys=ordered[max(0,end-stride):min(end,len(ordered))]
        if not keys:continue
        members=[r for key in keys for r in sorted(trials[key],key=lambda r:r.get('window_index',0))]
        prefixes+=members;session=keys[-1][0]
        dest=replay_run/f'{len(events)+1:03d}'
        nm,g,labels,clusters,facts=build_artifacts(prefixes,dest,cfg)
        rule=dict(cfg['rule']);rule['rule_id']=str(members[-1].get('rule_id') or session)
        current_members=[r for r in prefixes if r['session']==session]
        positions=sorted(set(r['planned_delta_mm'] for r in current_members if isinstance(r.get('planned_delta_mm'),(int,float))))
        if positions:rule['range_mm']={'a':[min(positions),max(positions)]}
        rule['objects']=sorted(set(r['object_id'] for r in current_members))
        spec=deepcopy(cfg['spec']);spec['fixed_protocol']=deepcopy(members[-1]['protocol'])
        spec['measurement_availability']={k:v.get('status') for k,v in nm.get('modalities',{}).items()}
        spec.setdefault('collection_config',{}).setdefault('force',{})['probe_fraction']=members[-1]['protocol']['probe_force_sdk']/members[-1]['protocol']['target_force_sdk']
        scope_ids={r['node_id'] for r in prefixes if r['object_id'] in rule['objects'] and r.get('time_period')==members[-1].get('time_period')}
        scoped_facts=[f for f in facts if f.get('evidence') and set(f['evidence'])<=scope_ids]
        rule['reason']='Recorded acquisition rule for historical replay; proposals are not executed.'
        write(dest/'rules/rule_0000.json',rule)
        hypotheses=update_hypotheses(hypotheses,facts,g['graph_version'])
        ctx=symbolic_context(spec,rule,scoped_facts,hypotheses=hypotheses,history=history,
            graph_version=g['graph_version'],observed_positions={'a':positions})
        result=advance_symbolic(ctx,dest/'decisions',allow_agent=llm_available,provider_settings=settings)
        if result.get('next_rule'):write(dest/'rules/rule_0001.json',result['next_rule'])
        hypotheses=result.get('hypotheses',hypotheses)
        event={'last_session':session,'prefix_trials':min(end,len(ordered)),'prefix_nodes':len(prefixes),'new_nodes':len(members),
            'last_trial_started_unix_ns':utc(keys[-1])[0], 'graph_version':g['graph_version'],
            'facts':dict(Counter(f['pred'] for f in scoped_facts if f['value'])),
            'objects':rule['objects'],'fixed_protocol':spec['fixed_protocol'],
            'applicable_operators':ctx['applicable_operators'],
            'decision_status':result['status'],'operator':(result.get('answer') or {}).get('operator'),
            'replay_artifacts':str(dest),'external_llm_called':llm_available}
        if any(a.get('transport_error') for a in result.get('attempts',[])):llm_available=False
        events.append(event);history.append(event)
        print('state-graph replay',event['prefix_trials'],len(ordered),flush=True)
    report={'status':('computed' if events and all(e['decision_status']=='accepted' for e in events) else 'partial_or_failed') if allow_agent else 'protocol_replay_only','events':events,
        'ordering':'trial_started_unix_ns then window_index, whole-trial atomic prefixes',
        'snapshot_frequency':f'every {stride} completed historical trials and final remainder',
        'hypotheses':hypotheses,'external_llm_called':allow_agent,'robot_started':False,
        'collection_efficiency_established':False,'assumed_new_trial_budget':0,
        'limitation':'Retrospective protocol replay; future historical actions were not caused by these proposals. No collection-efficiency conclusion.'}
    write(out/'replay.json',report)
    return report

def analyze(rows,out,cfg,allow_agent=False,replay_result=None):
    out=Path(out)
    from .provenance import prepare_nodes
    from .extract_nodes import write_nodes
    rows=prepare_nodes(rows,cfg.get('provenance',{}))
    write_nodes(rows,out)
    write(out/'features/nodes_with_features.json',rows)
    nm,g,labels,clusters,facts=build_artifacts(rows,out,cfg)
    result=run_validation(rows,g,nm,labels,clusters,facts,out/'reports',cfg)
    result['V9']=replay_result if replay_result is not None else history_replay(rows,out/'replay',cfg,allow_agent)
    from .skill import train_graph_skill
    training_out=out/'skill'/f'run_{time.time_ns()}'
    training=train_graph_skill([r for r in rows if not r.get('evaluation_only') and r.get('bilateral_contact') is True],g,nm,clusters,result,config=cfg.get('training',{}),out=training_out)
    write(out/'skill/latest.json',{'training_dir':str(training_out),'status':training['training_status']['status']})
    from .skill import model_execution_report
    model_report=model_execution_report(training)
    write(out/'skill/model_execution_report.json',model_report)
    if model_report.get('usable_for_B'):
        from .facts import generate_facts
        development=[r for r in rows if not r.get('evaluation_only') and r.get('bilateral_contact') is True]
        refreshed=generate_facts(g,development,labels,clusters,{**cfg.get('facts',{}),'model_report':model_report})
        facts=[f for f in refreshed if f['pred']!='coverage_ok']+[f for f in facts if f['pred']=='coverage_ok']
        write(out/'facts.json',facts)
    result['training']=training
    result['graph_counts']=g['counts']
    result['fact_counts']=dict(Counter(f['pred'] for f in facts if f['value']))
    write(out/'reports/validation.json',result)
    render_report(result,out/'reports/REPORT_zh.md')
    return {'out':str(out),'nodes':len(rows),'graph_counts':g['counts'],
        'validation':{k:result[k]['status'] for k in [f'V{i}' for i in range(1,10)]},
        'training_status':training.get('training_status',{}).get('status'),'robot_started':False}

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['extract','features','analyze','replay','run'])
    parser.add_argument('--config',default=DEFAULT_CONFIG)
    parser.add_argument('--out',default=DEFAULT_OUT)
    parser.add_argument('--nodes')
    parser.add_argument('--allow-agent',action='store_true',help='Allow historical fact summaries to the configured external LLM; never robot execution.')
    args=parser.parse_args(argv);cfg=json.loads(Path(args.config).read_text());out=Path(args.out).resolve()
    if args.command in ('extract','run'):
        from .extract_nodes import extract_sessions
        extract_sessions(cfg.get('sources'),out_dir=out,config=cfg.get('extraction'))
    if args.command in ('features','run'):
        assets=Path(cfg.get('feature_assets_from',DEFAULT_OUT)).resolve()/'features'
        deps=assets/'deps312'
        for name in ('legacy_mae','dinov2'):
            destination=out/'features/models'/name;source=assets/'models'/name
            if not destination.exists() and source.exists():
                destination.parent.mkdir(parents=True,exist_ok=True);destination.symlink_to(source,target_is_directory=True)
        subprocess.run(['docker','exec','-w',str(ROOT),'-e','PYTHONPATH='+str(deps),'-e','OMP_NUM_THREADS=4',
            'daimon_ros1_das','/root/miniconda3/envs/starvla/bin/python','-m','bottle_scan.state_graph.features',
            '--nodes',args.nodes or str(out/'nodes.jsonl'),'--out',str(out),'--device','cuda'],check=True)
    if args.command in ('analyze','replay','run'):
        source=args.nodes or (out/'features/nodes_with_features.json' if (out/'features/nodes_with_features.json').exists() else cfg.get('source_features',out/'features/nodes_with_features.json'))
        rows=read_rows(source)
        from .provenance import prepare_nodes
        rows=prepare_nodes(rows,cfg.get('provenance',{}))
        if args.command=='replay':result=history_replay(rows,out/'replay',cfg,args.allow_agent)
        else:result=analyze(rows,out,cfg,args.allow_agent)
        print(json.dumps(clean(result),ensure_ascii=False))
    return 0
