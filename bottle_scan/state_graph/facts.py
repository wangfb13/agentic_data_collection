"""Generic evidence predicates grounded in observations and evaluated outcomes."""
from collections import defaultdict
from itertools import combinations
import numpy as np
from scipy.stats import fisher_exact
from .common import digest,group_id
from .noise import drift_report


def fact(pred,args,evidence,metrics=None):
    evidence=sorted(set(evidence));return {'id':'F_'+digest([pred,args,evidence])[:18],'pred':pred,'args':args,'value':True,'evidence':evidence,'metrics':metrics or {}}


def _terminal(rows,cid):return [r for r in rows if r.get('is_terminal') and r['post'][cid]['evaluated']]


def shared_ready_regions(rows,clusters,spec,var,cid):
    byid={r['node_id']:r for r in rows};definition=spec['perturbation_variables'][var]
    condition=next(p for p in spec['post_conditions'] if p['id']==cid);a,b=condition.get('p_prior',[1,1]);ready=[]
    for cluster in clusters['clusters']:
        members=_terminal([byid[i] for i in cluster['node_ids']],cid);good=True
        for level in definition.get('levels',[]):
            subset=[r for r in members if r['attrs'].get(var)==level]
            p=(sum(r['post'][cid]['success'] for r in subset)+a)/(len(subset)+a+b)
            good=good and bool(subset) and p>=condition.get('ready_p',.8)
        if good and definition.get('levels'):ready.append(cluster['cluster_id'])
    return ready


def coordinate_evidence(row, variable):
    provenance=row.get('provenance',{})
    if provenance.get('coordinate_schema')=='fixed_contact_a_v1':
        from bottle_scan.max_expert_demonstration_finder.evidence import measured
        if variable!='a_offset_mm':return None
        label=provenance.get('supervision') or {}
        if not label.get('record') or not row.get('t_ns'):return None
        from bottle_scan.max_expert_demonstration_finder.proof import check_measurement
        meta={'reference_digest':provenance.get('reference_digest'),'target_definition_id':provenance.get('target_definition_id'),'axis_tcp':provenance.get('a_tcp')}
        if not check_measurement(label,meta,'first_contact',row['t_ns']):return None
        return measured(provenance.get('supervision'),provenance.get('reference_digest'),
                        provenance.get('target_definition_id'),stamp=row.get('t_ns'),
                        axis=provenance.get('a_tcp'),invalid=provenance.get('invalid_basis_ids',()))
    else:
        value=row.get('plan_coords',{}).get(variable)
    return value if type(value) in (int,float) and np.isfinite(value) else None


def observed_positions(rows,spec):
    return {v:sorted({x for r in rows if (x:=coordinate_evidence(r,v)) is not None})
            for v,d in spec['perturbation_variables'].items() if d['kind']=='continuous'}


def generate_facts(graph,rows,labels,clusters,noise,spec,models=None):
    facts=[];active=spec['active_post_condition']
    if any(m['status']!='ready' for m in noise['modalities'].values()):
        facts.append(fact('repeat_calibration_missing',[active],[r['node_id'] for r in rows],{'requires':'independent repeated windows at matched settings'}))
    outcomes=[r for r in rows if r.get('is_terminal') and r['post'][active]['evaluated']]
    if len(outcomes)<spec.get('models',{}).get('min_value_nodes',8):
        facts.append(fact('consumer_evidence_missing',[active],[r['node_id'] for r in outcomes],{'n':len(outcomes)}))
    byid={r['node_id']:r for r in rows};variables=spec.get('perturbation_variables',{})
    trials=[]
    for c in spec['post_conditions']:
        cid=c['id'];terminal=_terminal(rows,cid);rules=defaultdict(list)
        for r in terminal:rules[r.get('rule_id') or r['session']].append(r)
        for rule,members in rules.items():
            if all(r['post'][cid]['success'] for r in members):facts.append(fact('no_failure',[rule,cid],[r['node_id'] for r in members],{'terminal_count':len(members)}))
        for var,definition in variables.items():
            actual=[(r,coordinate_evidence(r,var)) for r in terminal]
            if definition['kind']=='continuous':
                actual=[(r,x) for r,x in actual if isinstance(x,(int,float)) and np.isfinite(x)];ref=definition.get('reference',0)
                for side,condition in [('negative',lambda x:x<ref),('positive',lambda x:x>ref),('reference',lambda x:x==ref)]:
                    failures=[r['node_id'] for r,x in actual if condition(x) and not r['post'][cid]['success']]
                    if failures:facts.append(fact('failure_present',[var,side,cid],failures))
                ordered=sorted(actual,key=lambda item:item[1]);step=definition.get('nominal_step',1.)
                # Adjacent actual observations across sessions; no unobserved
                # reference-coordinate value is used as a model feature.
                for (a,x),(b,y) in zip(ordered,ordered[1:]):
                    if y-x>2*step and a['post'][cid]['success']!=b['post'][cid]['success']:
                        facts.append(fact('boundary_gap',[var,[x,y],cid],[a['node_id'],b['node_id']],{'gap':y-x,'nominal_step':step,**({('gap_'+definition['unit']):y-x} if definition.get('unit') else {})}))
            counts=defaultdict(list)
            for r in rows:
                value=coordinate_evidence(r,var) if definition['kind']=='continuous' else r['attrs'].get(var)
                if value is not None:counts[str(value)].append(r['node_id'])
            facts.append(fact('coverage',[var,cid],[i for ids in counts.values() for i in ids],{'values':list(counts),'observed_counts':{v:len(ids) for v,ids in counts.items()},'terminal_counts':{v:sum(byid[i]['is_terminal'] and byid[i]['post'][cid]['evaluated'] for i in ids) for v,ids in counts.items()}}))
        for cluster in clusters['clusters']:
            st=cluster['statistics'][cid]
            if .2<=st['p_hat_median']<=.8 and st['terminal_count']>=4:
                evidence=st['terminal_ids'];facts.append(fact('stochastic_outcome',[cluster['cluster_id'],cid],evidence,st))
                members=[byid[i] for i in evidence]
                for var in set(variables)|set(spec['observation'].get('record_only',[])):
                    # Exact tests are for recorded categorical/discrete levels.
                    levels=sorted({str(r['attrs'].get(var)) for r in members if r['attrs'].get(var) is not None})
                    if not 2<=len(levels)<=8:continue
                    for x,y in combinations(levels,2):
                        subsets=[[r for r in members if str(r['attrs'].get(var))==z] for z in (x,y)]
                        table=[[sum(r['post'][cid]['success'] for r in part),sum(not r['post'][cid]['success'] for r in part)] for part in subsets]
                        _,p=fisher_exact(table)
                        trials.append({'cluster':cluster['cluster_id'],'var':var,'condition':cid,'levels':[x,y],'table':table,'p':float(p),'evidence':evidence})
        for ident,status in graph['reachability'][cid].items():
            if status['status'] in ('unsupported','unreachable'):
                facts.append(fact(status['status'],[ident,cid],[ident],{'trial_groups':status['trial_groups'],'observed_edges':status['evidence']}))
        path_ids={i for l in labels['nodes'] if l['post_condition']==cid for i in l['path_edge_ids']}
        movement=[e for e in graph['edges'] if e['edge_id'] in path_ids and e['kind'] in ('observed','inferred')]
        inferred=[e for e in movement if e['kind']=='inferred']
        if inferred:
            top=max(inferred,key=lambda e:e['uncertainty'])
            facts.append(fact('inferred_edge_unverified',[len(inferred),cid],[e['source'] for e in inferred],
                              {'path_fraction':len(inferred)/len(movement),'source_node_id':top['source'],'edge_id':top['edge_id'],'uncertainty':top['uncertainty'],'u':top['u']}))
    M=len(trials)
    for t in trials:
        corrected=min(1.,t['p']*M)
        if corrected<.05:
            shared=shared_ready_regions(rows,clusters,spec,t['var'],t['condition']) if t['var'] in variables else []
            facts.append(fact('hidden_variable_candidate',[t['cluster'],t['var'],t['condition']],t['evidence'],
                              {'p_raw':t['p'],'p_adjusted':corrected,'tests':M,'levels':t['levels'],'table':t['table'],'shared_ready_regions':shared,'scope':'observed_regions_only'}))
    for d in drift_report(rows,noise):
        if d['shift_detected']:facts.append(fact('drift',[d['session']],d['node_ids'],d))
    if models:
        from .value import predict_value
        from .dynamics import predict
        from .skill_interface import action_grid
        active=spec['active_post_condition'];ms=models['models'][active]
        for cluster in clusters['clusters']:
            representative=cluster['prototype']
            if ms['V'].get('status')=='trained':
                v=predict_value(ms['V'],representative,noise)
                if v['std']>ms['V']['uncertainty_q90']:
                    facts.append(fact('high_disagreement',[cluster['cluster_id']],cluster['node_ids'],{'std':v['std'],'threshold':ms['V']['uncertainty_q90']}))
            dm=models['dynamics']
            if dm.get('status')=='trained':
                uncertainty=max((predict(dm,representative,u)['uncertainty'] for u in action_grid(spec,graph['edges'])), default=None)
                if uncertainty is not None and uncertainty>dm['uncertainty_q90']:
                    facts.append(fact('dynamics_uncertain',[cluster['cluster_id']],cluster['node_ids'],{'uncertainty':uncertainty,'threshold':dm['uncertainty_q90']}))
        if ms['pi'].get('status')!='trained' or models['dynamics'].get('status')!='trained':
            facts.append(fact('transition_evidence_missing',[active],[r['node_id'] for r in rows],{'policy_samples':ms['pi'].get('n',0),'transition_samples':models['dynamics'].get('n',0),'requires':'before, typed executed action receipt, after, actual consumer outcome'}))
        for name,model in [('V',ms['V']),('pi',ms['pi']),('f',models['dynamics'])]:
            if model.get('status')=='trained':facts.append(fact('coverage_ok',[name],model.get('fit_node_ids',[]),{'n':model.get('n'),'post_condition':active}))
        if all(ms[k].get('status')=='trained' for k in ('V','pi')):
            facts.append(fact('model_executable',[active],ms['V']['fit_node_ids'],{'independent_effectiveness_proven':False}))
            if spec.get('acceptance', {}).get('requires_ready_and_need_action_coverage'):
                regions = clusters['clusters']
                present = {c['symbols'][active].split('(', 1)[0] for c in regions}
                missing = {'Ready', 'NeedAction'} - present
                if missing:
                    facts.append(fact('development_class_missing', [active],
                                      [i for c in regions for i in c['node_ids']],
                                      {'observed_classes': sorted(present), 'missing_classes': sorted(missing)}))
    return facts
