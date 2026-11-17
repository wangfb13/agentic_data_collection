"""Generic instance-driven extraction, with no task-specific parsing."""
from .common import digest,write,clean
from .skill_interface import load_callable,task_id,model_observation,repeat_key


def extract_nodes(sources,spec,out=None):
    iterate=load_callable(spec['adapters']['trials'])
    observe=load_callable(spec['observation']['window']['extractor'])
    action=load_callable(spec['adapters']['action'])
    nodes=[];excluded=[]
    for trial in iterate(sources,spec):
        for window in observe(trial,spec):
            required=spec['observation']['window'].get('requires',[])
            if not all(window.get('preconditions',{}).get(k) is True for k in required):
                excluded.append({'trial':trial['trial_id'],'window':window.get('window_index'),'reason':'window_precondition'});continue
            row={k:window.get(k,trial.get(k)) for k in ('session','trial_id','window_index','phase','t_ns','group_id')}
            row.update(node_id=window.get('node_id') or digest(row)[:24],skill_id=spec['skill_id'],
                       task_id=window.get('task_id',task_id(spec)),window_valid=True,
                       obs={},extractor_versions={},attrs={},plan_coords=window.get('plan_coords',{}),
                       repeat_settings=window.get('repeat_settings',{}),is_reference=bool(window.get('is_reference')),
                       is_terminal=bool(window.get('is_terminal')),post={},
                       provenance=window.get('provenance',{}),previous_node_id=window.get('previous_node_id'))
            for k in ('evaluation_only','frozen_independent','evaluation_scope','rule_id','legacy_labels','action_executed','data_role','round_index','source','target_consumer','controller_role','expert_source'):
                if k in window:row[k]=window[k]
            names=set(spec['observation'].get('record_only',[]))|set(spec.get('perturbation_variables',{}))
            row['attrs']={k:window.get('attrs',{}).get(k) for k in names}
            for name,modality in spec['observation']['modalities'].items():
                recorded=window.get('recorded_features')
                if recorded:
                    row['obs'][name]=recorded['obs'][name];row['extractor_versions'][name]=recorded['extractor_versions'][name]
                    row.setdefault('feature_evidence',{})[name]=recorded.get('feature_evidence',{}).get(name,{})
                    continue
                fn=load_callable(modality['extractor'])
                source=window
                if (spec.get('graph_schema')=='state_graph_v5' and modality.get('encoder_features') and
                        modality['extractor']=='bottle_scan.state_graph.modalities.tactile_embedding:embedding' and
                        not any(k in window for k in ('tactile_arrays','tactile_frames')) and window.get('provenance',{}).get('source_file')):
                    from .representation_p3 import recorded_frames
                    arrays,_=recorded_frames(window)
                    source={**window,'tactile_arrays':arrays}
                extracted=fn(source,spec)
                if isinstance(extracted,dict) and 'values' in extracted:
                    row['obs'][name]=extracted['values']
                    row['extractor_versions'][name]=extracted['version']
                    row.setdefault('feature_evidence',{})[name]=extracted.get('evidence',{})
                else:
                    row['obs'][name]=extracted
                    row['extractor_versions'][name]=window.get('extractor_versions',{}).get(name,modality.get('version',modality['extractor']))
            try:model_observation(row,spec['observation']['modalities'])
            except (ValueError,TypeError) as exc:
                excluded.append({'trial':trial['trial_id'],'window':window.get('window_index'),'reason':str(exc)});continue
            executed=action(trial,window,spec)
            row.update(u_executed=executed.get('u'),u_receipt=executed.get('receipt'),risk=window.get('risk',{}))
            for condition in spec['post_conditions']:
                result=load_callable(condition['evaluator'])(trial,window,condition)
                if not row['is_terminal']:result={'evaluated':False,'success':None,'level':None,'evidence':[]}
                if result.get('evaluated') and type(result.get('success')) is not bool:
                    raise ValueError('An evaluated post condition requires a Boolean result')
                row['post'][condition['id']]=result
            row['repeat_key']=repeat_key(row,spec)
            for key,value in window.get('deprecated_aliases',{}).items():
                if key not in row:row[key]=value
            row['deprecated_fields']=list(window.get('deprecated_aliases',{}))
            nodes.append(clean(row))
    if len({n['node_id'] for n in nodes})!=len(nodes):raise ValueError('Duplicate extracted node')
    report={'nodes':len(nodes),'represented_trials':len({n['group_id'] for n in nodes}),'excluded':excluded,'robot_started':False}
    if out:
        from pathlib import Path
        p=Path(out);p.mkdir(parents=True,exist_ok=True)
        (p/'nodes.jsonl').write_text(''.join(__import__('json').dumps(n,ensure_ascii=False,allow_nan=False)+'\n' for n in nodes))
        write(p/'extraction_report.json',report)
        try:
            import pyarrow as pa,pyarrow.parquet as pq,json
            flat=[{k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in n.items()} for n in nodes]
            pq.write_table(pa.Table.from_pylist(flat),p/'nodes.parquet')
        except ImportError:pass
    return nodes,report
