"""Offline specification-driven pipeline. Instance adapters own all task parsing."""
import argparse
from copy import deepcopy
from collections import defaultdict,Counter
from pathlib import Path
import json
import time
import numpy as np
from .common import clean,write,group_id,independent,digest
from .skill_interface import load_spec
from .extract_nodes import extract_nodes
from .noise import fit_noise
from .dynamics import fit_dynamics
from .graph import build_graph
from .labels import generate_labels
from .clusters import cluster_nodes
from .skill import train_models
from .facts import generate_facts,coordinate_evidence,observed_positions
from .director_bridge import context,decide
from .report import run_validation,render_report


def build(rows,spec,out,validate=True,assessments=(),previous_clusters=None):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    expected_schema=spec.get('graph_schema','state_graph_v4')
    if (out/'graph.json').exists() and json.loads((out/'graph.json').read_text()).get('schema')!=expected_schema:
        raise ValueError('Use different output directories for different graph versions')
    if expected_schema=='state_graph_v5':
        rows=[r for r in rows if not independent(r)]
    noise=fit_noise(rows,spec);f=fit_dynamics(rows,noise,spec)
    graph=build_graph(rows,noise,spec,f);labels=generate_labels(graph,rows,spec);clusters=cluster_nodes(rows,graph,noise,labels,spec)
    from .clusters import reconcile_regions
    clusters=reconcile_regions(clusters,previous_clusters)
    bundle,status=train_models(rows,graph,labels,noise,f,spec,out/'models')
    facts=generate_facts(graph,rows,labels,clusters,noise,spec,bundle)
    acceptance={}
    if spec.get('acceptance',{}).get('development_B'):
        from .acceptance import evaluate_B
        for role in ('development','independent'):
            if role+'_B' not in spec['acceptance']:continue
            acceptance[role]=evaluate_B(assessments,spec,role)
            facts.extend(acceptance[role]['facts'])
        write(out/'B_acceptance.json',acceptance)
    for name,value in [('noise_model',noise),('graph',graph),('labels',labels),('clusters',clusters),('facts',facts),('spec',spec)]:write(out/(name+'.json'),value)
    report=run_validation(rows,spec,noise,graph,labels,clusters,facts,out/'reports') if validate else None
    if report is not None:report['B_acceptance']=acceptance
    return {'noise':noise,'graph':graph,'labels':labels,'clusters':clusters,'facts':facts,'bundle':bundle,'training':status,'report':report}


def initial_rule(rows,spec):
    variables={}
    for name,v in spec['perturbation_variables'].items():
        setting={'active':v.get('active',True)}
        if v['kind']=='continuous':
            values=[x for r in rows if (x:=coordinate_evidence(r,name,spec)) is not None]
            setting['range']=[min(values),max(values)] if values else list(v.get('initial_range',v['envelope']))
        else:setting['levels']=v.get('levels',[])
        variables[name]=setting
    return {'rule_id':(rows[-1].get('rule_id') or rows[-1]['session']) if rows else 'rule_seed','variables':variables,'post_condition':spec['active_post_condition']}


def history_replay(rows,spec,out,allow_agent=False,settings=None):
    groups=defaultdict(list)
    for r in rows:groups[group_id(r)].append(r)
    ordered=sorted(groups,key=lambda k:(min(r.get('provenance',{}).get('trial_started_unix_ns',r['t_ns']) or 0 for r in groups[k]),k))
    events=[];prefix=[];hypotheses=[];endpoints=sorted(set([max(1,int(len(ordered)*fraction)) for fraction in (.25,.5,.75,1.)]))
    previous=0
    for end in endpoints:
        prefix.extend(r for k in ordered[previous:end] for r in groups[k]);previous=end
        state=build(prefix,spec,Path(out)/str(end),validate=False);rule=initial_rule(prefix,spec)
        positions=observed_positions(prefix,spec)
        from .director_bridge import update_hypotheses
        hypotheses=update_hypotheses(hypotheses,state['facts'],state['graph']['graph_version'])
        ctx=context(spec,rule,state['facts'],hypotheses=hypotheses,history=events,graph_version=state['graph']['graph_version'],observed_positions=positions,budget_left=0 if not allow_agent else spec['budget']['max_new_trials'])
        result=decide(ctx,Path(out)/str(end)/'director',allow_agent=allow_agent,provider_settings=settings)
        hypotheses=result.get('hypotheses',hypotheses)
        events.append({'prefix_trials':end,'prefix_nodes':len(prefix),'status':result['status'],'graph_version':state['graph']['graph_version'],
                       'facts':dict(Counter(f['pred'] for f in state['facts'])),'applicable_operators':ctx['applicable_operators'],'answer':result.get('answer')})
        print('replay prefix',end,len(ordered),flush=True)
    report={'status':'agent_replay' if allow_agent else 'protocol_replay_only','events':events,'external_llm_called':allow_agent,'new_trials_executed':0,
            'collection_efficiency_established':False,'ordering':'whole trials by recorded acquisition time; models/noise fitted only on each prefix'}
    write(Path(out)/'replay.json',report);return report


def pipeline(spec,sources,out,allow_agent=False,settings=None,replay=True,assessments=(),director_state=None):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);rows,extraction=extract_nodes(sources,spec,out)
    grouped=defaultdict(list)
    for r in rows:
        if not independent(r):grouped[r['task_id']].append(r)
    reports=[];states={}
    for task,members in sorted(grouped.items()):
        task_spec=deepcopy(spec);task_spec['task']['object_category']=task.split('|')[0];dest=out/'tasks'/digest(task)[:12]
        print('task graph',task,len(members),flush=True)
        task_assessments=[r for r in assessments if r.get('task_id')==task]
        state=build(members,task_spec,dest,assessments=task_assessments)
        if replay:state['report']['V9']=history_replay(members,task_spec,dest/'replay',allow_agent,settings)
        else:state['report']['V9']={'status':'not_requested'}
        write(dest/'reports/validation.json',state['report']);states[task]=state;reports.append(state['report'])
        # Current decisions consume actual B acceptance, separately from the
        # retrospective prefix replay (which must not see future B outcomes).
        rule=initial_rule(members,task_spec)
        ctx=context(task_spec,rule,state['facts'],state=director_state,graph_version=state['graph']['graph_version'])
        write(dest/'director_current_context.json',ctx)
    write(out/'task_index.json',{t: {'directory':str(out/'tasks'/digest(t)[:12]),'graph_version':s['graph']['graph_version'],'training':s['training']} for t,s in states.items()})
    return reports,states,rows


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=['extract','analyze','run','replay'])
    parser.add_argument('--graph-version',type=int,choices=[4,5]);parser.add_argument('--config',default='config.state_graph.v5.json');parser.add_argument('--spec');parser.add_argument('--out');parser.add_argument('--allow-agent',action='store_true');parser.add_argument('--skip-fixture',action='store_true')
    args=parser.parse_args(argv);cfg=json.loads(Path(args.config).read_text());spec=load_spec(args.spec or cfg['spec']);out=Path(args.out or cfg['out'])
    version=args.graph_version or cfg.get('graph_version',4)
    from .v5_profile import configure
    spec=configure(spec,cfg.get('graph_overrides'),version=version)
    if version==5:
        from .v5_profile import protect_output
        protect_output(out, kind='analysis')
    elif (out/'v5_output.json').exists():
        raise ValueError('Choose a separate legacy replay directory')
    if args.command=='extract':extract_nodes(cfg.get('sources',spec.get('data',{}).get('sources',[])),spec,out);return 0
    assessments=[]
    for source in cfg.get('B_assessments',[]):
        path=Path(source)
        records=[json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.suffix=='.jsonl' else json.loads(path.read_text())
        assessments.extend(records)
    reports,states,rows=pipeline(spec,cfg.get('sources',spec.get('data',{}).get('sources',[])),out,args.allow_agent,cfg.get('provider_settings'),assessments=assessments,director_state=cfg.get('director_state'))
    fixture={'status':'not_requested'}
    if cfg.get('validation_spec') and not args.skip_fixture:
        other=load_spec(cfg['validation_spec']);fr,fs,fn=pipeline(other,[],out/'genericity',replay=False)
        forbidden=[]
        for p in Path(__file__).parent.rglob('*.py'):
            for token in cfg.get('forbidden_instance_literals',[]):
                if token in p.read_text():forbidden.append({'file':str(p),'token':token})
        fixture={'status':'pass' if fs and not forbidden and all(s['training']['active_model_trained'] for s in fs.values()) else 'fail',
                 'action_dimension':other['task']['action_space']['dim'],'post_conditions':[c['id'] for c in other['post_conditions']],
                 'nodes':len(fn),'forbidden_literals':forbidden,'task_reports':{k:s['training'] for k,s in fs.items()}}
        write(out/'V11_genericity.json',fixture)
    from .plots import render_plots
    render_plots(reports,out/'figures')
    render_report(reports,fixture,out/'REPORT_zh.md')
    write(out/'summary.json',{'reports':reports,'V11':fixture,'robot_started':False,'new_trials_executed':0})
    print(json.dumps(clean({'out':str(out),'tasks':list(states),'V11':fixture['status'],'robot_started':False}),ensure_ascii=False));return 0
