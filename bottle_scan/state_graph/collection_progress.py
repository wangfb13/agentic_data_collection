"""Declare an auditable acquisition target and measure it before any development B."""
from pathlib import Path
from .common import digest, group_id
from .axis_resolution import single_axis
from .amplitude_coverage import terminal_samples, bin_index
from .repeat_evidence import indices
from .facts import fact
from .v5_facts import coordinate


def target(answer, spec, widths, facts):
    op,a=answer['operator'],answer['args']
    cells=(answer.get('plan') or {}).get('cells',[])
    if not cells:return None
    axes=spec['task']['action_space']['continuous_displacement_variables']
    cid=a.get('post_condition',spec['active_post_condition'])
    t=dict(kind='evaluated_outcomes',consumer=cid)
    if op=='collect_repeat_block':
        t=dict(kind='qualified_repeats',modalities=a['modalities'],condition=a['condition'],point={k:0 for k in axes},resolution_mm=dict(widths))
    elif op in ('probe_recovery_frontier','request_expert_recovery','set_allocation'):
        role=a.get('role')
        t.update(kind='initial_episode_outcomes',role={'direct_baseline':'direct','policy_frontier':'policy',
                 'expert_recovery':'expert'}.get(role),expert_witness=op=='request_expert_recovery',
                 points=[{k:c['settings'][k] for k in axes if k in c['settings']} for c in cells],
                 tolerances=dict(widths))
    elif op=='recalibrate_reference':
        t=dict(kind='placement_witnesses')
    elif op=='probe_amplitude' or ('target_mm' in a and 'interval' in a):
        f=next(f for f in facts if f['pred']=='amplitude_gap' and f['args']==[a['var'],a['side'],a['interval']])
        m=f['metrics']; width=m['bin_width_mm']; origin=m['reference_mm']
        t.update(kind='amplitude_bin',axis=a['var'],side=a['side'],origin=origin,width=width,
                 index=bin_index(abs(a['target_mm']-origin),width))
    elif op in ('collect_transitions','verify_edges'):
        signed=sorted({(axis,1 if u>0 else -1) for c in cells for act in c['actions']
                       for axis,u in zip(axes,act.get('u',[])) if u})
        t=dict(kind='qualified_transitions',signed_axes=[list(s) for s in signed])
        if a.get('intervals'):t['start_region']=a['intervals']
        if a.get('target_point'):t['start_region']={k:[v-widths[k],v+widths[k]] for k,v in a['target_point'].items()}
        if op=='verify_edges':t.update(source_node_id=a['source_node_id'],edge_id=a['edge_id'])
    else:
        points=[{k:c.get('settings',{})[k] for k in axes if k in c.get('settings',{})} for c in cells]
        if all(points):t.update(points=points,tolerances={k:widths[k] for p in points for k in p})
    if a.get('condition'):t['condition']=a['condition']
    n=sum(c['reps']*len(c.get('blocks',[{}])) for c in cells)
    identity={k:v for k,v in t.items() if k not in ('resolution_mm','tolerances')}
    return dict(counter=t,target_id=digest([op,identity])[:20],expected_increment=n)


def members(t,rows,graph,spec):
    kind=t['kind'];cond=t.get('condition',{})
    def matches(attrs):
        return all(attrs.get(k)==v for k,v in cond.items())
    if kind=='qualified_repeats':
        if 'node_ids' in graph:
            fitted=set(graph['node_ids'])
            rows=[r for r in rows if r['node_id'] in fitted]
        widths=t['resolution_mm']
        return {rows[i]['node_id'] for ids in indices(rows,spec,widths=widths) for i in ids
                if matches(rows[i]['attrs']) and all(abs(coordinate(rows[i],k)[0]-x)<=widths[k]
                    and (rows[i].get('repeat_settings',{}).get(k) or 0)==x for k,x in t['point'].items())}
    if kind=='placement_witnesses':
        return {r['provenance']['placement_reference_id'] for r in rows if r.get('is_reference')
                and r['provenance'].get('placement_reference_valid') and r['provenance'].get('placement_reference_id')}
    if kind=='qualified_transitions':
        byid={r['node_id']:r for r in rows}
        axes=spec['task']['action_space']['continuous_displacement_variables']
        hits=set()
        for e in graph['edges']:
            if e['kind']!='observed':continue
            i=single_axis(e,spec);r=byid.get(e['source'])
            if i is None or not r or not matches(r['attrs']):continue
            if [axes[i],1 if e['u'][i]>0 else -1] not in t['signed_axes']:continue
            if any(coordinate(r,k)[0] is None or not lo<=coordinate(r,k)[0]<=hi for k,(lo,hi) in t.get('start_region',{}).items()):continue
            if t.get('source_node_id'):
                if r.get('provenance',{}).get('bound_source_node_id')!=t['source_node_id'] and r['node_id']!=t['source_node_id']:continue
            hits.add(e['target'])
        return hits
    first={group_id(r):r for r in rows if r.get('window_index')==0 and not r.get('previous_node_id') and r.get('window_valid')}
    expert_moves={group_id(r) for r in rows if r.get('expert_success') and (r.get('u_receipt') or {}).get('applied')}
    found=set()
    for s in terminal_samples(rows,spec,t['consumer']):
        if not matches(s['condition']):continue
        coords=s['coords']
        if kind=='initial_episode_outcomes':
            r=first.get(s['group'])
            if not r or (t['role'] and r.get('controller_role')!=t['role']):continue
            if t['expert_witness'] and (not s['success'] or s['group'] not in expert_moves):continue
            coords={k:coordinate(r,k)[0] for k in t['tolerances'] if coordinate(r,k)[0] is not None}
        if kind=='amplitude_bin':
            x=s['coords'].get(t['axis'])
            if x is None:continue
            d=(x-t['origin'])*(1 if t['side']=='positive' else -1)
            if d<0 or bin_index(d,t['width'])!=t['index']:continue
        if t.get('points') and not any(all(k in coords and abs(coords[k]-v)<=t['tolerances'][k] for k,v in p.items()) for p in t['points']):continue
        found.add(s['group'])
    return found


def declaration(answer, ctx):
    declared=target(answer,ctx['spec'],ctx['point_evidence']['widths'],ctx['facts'])
    if not declared:return
    for rec in ctx.get('active_cooldowns',[]):
        if rec['operator']==answer['operator'] and rec['target_id']==declared['target_id']:
            raise ValueError('operator_no_progress cooldown: '+declared['target_id'])
    answer['evidence_target']=declared


def check(out,state,spec,model,rows,planned,*,token=None):
    from .campaign import read
    idx=state['round_index'];d=planned['decision'];decl=d['evidence_target']
    graph=read(model['graph_path']) if 'graph_path' in model else model
    after=members(decl['counter'],rows,graph,spec)
    before=set(planned['evidence_before'])
    trials=[t for t in state['trials'] if t['request'].get('round_index')==idx and
            t['request'].get('operation')==d['operator'] and (token is None or t['request'].get('stage_token')==token) and t['request']['role'] not in ('development','independent')]
    trace=[];batch=set()
    for trial in trials:
        path=Path(trial['directory'])/'result.json'
        res=read(path) if trial['status']=='complete' and path.exists() else {}
        nodes=res.get('nodes',[])
        for r in nodes:
            batch.update([r['node_id'],group_id(r)])
            if r.get('is_reference'):batch.add(r['provenance'].get('placement_reference_id'))
        trace.append(dict(trial_id=trial['trial_id'],status=trial['status'],node_count=len(nodes),
            blocking_facts=res.get('blocking_facts',[]),
            reason=res.get('failure_cause') or res.get('attempt_reason') or trial.get('reason') or trial.get('error'),
            consumers={k:v.get('evaluated') for k,v in res.get('terminal_outcomes',{}).items()},
            skipped_pair=(res.get('pairing') or {}).get('reason')))
    # Attribute new evidence to this batch; do not count an older singleton
    # promoted into a repeat group by its first new companion as a new probe.
    added=sorted((after-before)&batch)
    reason=None
    if not added:
        if not trace:reason='not_executed'
        elif any(x['status']=='budget_skipped' or 'skip' in str(x['skipped_pair']) for x in trace):reason='skipped'
        elif all(x['status']!='complete' for x in trace):reason='not_executed'
        elif any(x['reason'] in ('repeat_group_mismatch','repeat_placement_changed','repeat_tcp_changed') for x in trace):reason='group_mismatch'
        elif all(not x['node_count'] or x['reason'] for x in trace):reason='invalid_result'
        elif decl['counter']['kind'] in ('qualified_repeats','amplitude_bin','qualified_transitions') or any(any(x['consumers'].values()) for x in trace):reason='group_mismatch'
        else:reason='invalid_result'
    return dict(**decl,round_index=idx,operator=d['operator'],before=len(before),after=len(after),
                actual_increment=len(added),added=added,reason=reason,execution=trace)


def restrictions(history, next_round, duration, rows, graph, spec, facts=()):
    active=[];repeated={}
    for rec in history:
        op=rec['operator']
        if rec['actual_increment']:
            repeated.pop(op,None)
            continue
        repeated.setdefault(op,[]).append(rec)
        if next_round<=rec['round_index']+duration and rec['round_index']<next_round:
            # New qualifying evidence for the identical target demonstrates
            # that the recorded obstacle no longer prevents evidence production.
            recorded={f for x in rec['execution'] for f in x.get('blocking_facts',[])}
            # Explicit resolution facts refer to the original blocker evidence ID.
            cleared={f['args'][0] for f in facts if f.get('value') and f['pred']=='blocking_fact_resolved' and f['args']}
            if recorded and recorded<=cleared:continue
            if len(members(rec['counter'],rows,graph,spec))<=rec['after']:
                active.append(rec)
    return active,{op:items for op,items in repeated.items() if len(items)>=2}


def no_progress_fact(record):
    return fact('operator_no_progress',[record['operator'],record['target_id'],record['reason']],
                [r['trial_id'] for r in record['execution']],record)
