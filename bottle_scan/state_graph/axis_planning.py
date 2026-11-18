"""Enumerate physically distinct proposals before applying the director score."""
from copy import deepcopy
from .common import digest


def targeted_offers(offers, facts, spec, rule):
    cid = spec['active_post_condition']
    axes = [v for v in spec['task']['action_space'].get('continuous_displacement_variables', ())
            if rule['variables'].get(v, {}).get('active', spec['perturbation_variables'][v].get('active', True))]
    gaps = [f for f in facts if f['pred']=='amplitude_gap' and f['args'][0] in axes and f['metrics']['consumer']==cid]
    scored = spec.get('director', {}).get('score_based_axis_choice')
    missing = {f['args'][1]:f for f in facts if scored and f['pred']=='development_class_missing'
               and len(f['args'])==2 and f['args'][0]==cid}
    result = []
    for offer in offers:
        bindings = offer['bindings']
        if scored and offer['op']=='collect_transitions' and not bindings.get('action_design'):
            for var in sorted(axes):
                for side in ('negative', 'positive'):
                    result.append({**offer,'bindings':dict(bindings,var=var,side=side,action_design='directional')})
        else:
            result.append(offer)
    for f in gaps:
        var,side,interval=f['args']
        bindings=dict(var=var,side=side,interval=interval,target_mm=f['metrics']['target_mm'],post_condition=cid)
        premises=[f['id']]
        if 'NeedAction' in missing:
            bindings['missing_class']='NeedAction'
            premises.append(missing['NeedAction']['id'])
            # Independent direct trial at the candidate start; do not infer
            # this outcome from the post-action probe.
            result.append(dict(op='collect_outcomes',bindings=dict(bindings,
                target_point={var:bindings['target_mm']}),
                premises_satisfied_by=list(premises)))
        result.append(dict(op='probe_amplitude',bindings=bindings,premises_satisfied_by=premises))
    for f in facts:
        if f['pred']=='best_covered_ready_region' and f['args']==[cid] and 'Ready' in missing:
            m=f['metrics']
            result.append(dict(op='collect_outcomes',bindings=dict(post_condition=cid,
                target_point=m['target_point'],condition=m['condition'],missing_class='Ready'),
                premises_satisfied_by=[missing['Ready']['id'],f['id']]))
    return result


def axis_targets(proposal, spec):
    a = proposal['args']
    if a.get('var') and a.get('side'):
        return [(a['var'],a['side'])]
    axes = spec['task']['action_space'].get('continuous_displacement_variables', [])
    result=set()
    for cell in (proposal.get('plan') or {}).get('cells', []):
        for action in cell.get('actions', []):
            for var,u in zip(axes, action.get('u', [])):
                if u:
                    result.add((var,'positive' if u>0 else 'negative'))
        for var in axes:
            delta=cell.get('settings', {}).get(var, spec['perturbation_variables'][var].get('reference',0))-spec['perturbation_variables'][var].get('reference',0)
            if delta:
                result.add((var,'positive' if delta>0 else 'negative'))
    return sorted(result)


def select(ctx):
    spec=ctx['spec']
    history=ctx.get('history', [])
    last={var for var,_ in axis_targets(history[-1]['decision'],spec)} if history and 'decision' in history[-1] else set(ctx.get('imported_last_axes',[]))
    gaps={(f['args'][0],f['args'][1]):f for f in ctx['facts'] if f['pred']=='amplitude_gap'}
    prior=ctx.get('effective_priors', {}).get('placement_sampling', {}) if spec['director'].get('use_priors') else {}
    entries=[]
    for item in ctx['operator_scores']:
        plan=item.get('proposal')
        summary=dict(operator=item['operator'],bindings=item['bindings'],score=item.get('score'),
                     feasible=plan is not None)
        if plan is None:
            entries.append(dict(summary,unavailable_reason=item.get('unavailable_reason')))
            continue
        targets=axis_targets(plan,spec)
        widths=[]
        axes=spec['task']['action_space'].get('continuous_displacement_variables', [])
        for axis,side in targets:
            gap=gaps.get((axis,side))
            if not gap or axis not in axes:
                continue
            lo,hi=gap['args'][2]
            origin=spec['perturbation_variables'][axis].get('reference',0)
            idx=axes.index(axis)
            for cell in (plan.get('plan') or {}).get('cells',[]):
                end=cell.get('settings',{}).get(axis,origin)
                end+=sum(a['u'][idx] for a in cell.get('actions',[]) if a.get('u'))
                inside=(lo < end <= hi) if side=='positive' else (lo <= end < hi)
                if inside:
                    widths.append(hi-lo)
                    break
        entries.append(dict(summary,proposal_id=digest(plan),axes=targets,plan=plan.get('plan'),
            gap_width_mm=max(widths,default=0.),unused_axis=bool(targets) and all(k not in last for k,_ in targets),
            preferred=any(prior.get(v.split('_')[0],{}).get(s)=='preferred' for v,s in targets)))
    pool=[e for e in entries if e['feasible']]
    if not pool:
        raise ValueError('No feasible validated proposal')
    # Completion / pause are lifecycle outcomes. Among collection offers the
    # first rank is always D5, including proposals on an underrepresented axis.
    if any(e['operator']=='complete' for e in pool):
        pool=[e for e in pool if e['operator']=='complete']
    elif any(e['operator']!='pause' for e in pool):
        pool=[e for e in pool if e['operator']!='pause']
    decisions=[]
    for key in ('score','gap_width_mm','unused_axis','preferred','proposal_id'):
        if len(pool)==1:break
        if key=='preferred' and not prior:continue
        def rank(e):
            value=e[key]
            return value if value is not None else -float('inf')
        best=(min if key=='proposal_id' else max)(rank(e) for e in pool)
        before=len(pool)
        pool=[e for e in pool if rank(e)==best]
        if before!=len(pool):
            decisions.append(dict(criterion=key,before=before,remaining=len(pool)))
    winner=pool[0]
    answer=deepcopy(next(i['proposal'] for i in ctx['operator_scores'] if i.get('proposal') and digest(i['proposal'])==winner['proposal_id']))
    answer['selection_audit']=dict(candidates=entries,last_round_axes=sorted(last),
        selected=winner['proposal_id'],tie_breaks=decisions,prior_decisive=any(d['criterion']=='preferred' for d in decisions))
    return answer
