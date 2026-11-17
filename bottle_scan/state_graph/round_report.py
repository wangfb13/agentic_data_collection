"""Evidence-only round and paired-consumer summaries."""
from collections import Counter, defaultdict
from itertools import combinations
from math import floor, sqrt
from statistics import NormalDist
from scipy.stats import binomtest


def success_interval(n, successes, confidence):
    if not n:
        return {'n':0,'successes':0,'rate':None,'wilson':None}
    z=NormalDist().inv_cdf((1+confidence)/2)
    p=successes/n
    middle=(p+z*z/(2*n))/(1+z*z/n)
    half=z*sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return {'n':n,'successes':successes,'rate':p,'wilson':[max(0,middle-half),min(1,middle+half)]}


def paired_test(pairs, confidence):
    n=len(pairs)
    if not n:
        return {'n':0,'rate_difference':None,'interval':None,'mcnemar_exact_p':None}
    left_only=sum(a and not b for a,b in pairs)
    right_only=sum(b and not a for a,b in pairs)
    # Simultaneous exact marginal intervals for the two discordant probabilities.
    level=1-(1-confidence)/2
    a=binomtest(left_only,n).proportion_ci(level)
    b=binomtest(right_only,n).proportion_ci(level)
    return {'n':n,'left_only':left_only,'right_only':right_only,
            'rate_difference':(left_only-right_only)/n,'interval':[a.low-b.high,a.high-b.low],
            'mcnemar_exact_p':binomtest(left_only,left_only+right_only).pvalue if left_only+right_only else 1.}


def paired_report(episodes, graph, spec):
    config=spec.get('report',{})
    if not config.get('paired_consumers'):
        return {}
    width=config['coordinate_bin_mm']
    if width<=0:raise ValueError('Coordinate bin width must be positive')
    comparisons={}
    for left,right in combinations([c['id'] for c in spec['post_conditions']],2):
        seen=set();pairs=[]; bins=defaultdict(list)
        for e in episodes:
            outcomes=e['terminal_outcomes']
            if not e['paired'] or not all(c in outcomes for c in (left,right)):continue
            if e['group_id'] in seen:raise ValueError('Paired analysis needs independent episode groups')
            seen.add(e['group_id'])
            values=(outcomes[left]['success'],outcomes[right]['success'])
            pairs.append(values)
            coords=e.get('initial_coords') or {}
            key=', '.join(f'{k}=[{floor(x/width)*width:g},{(floor(x/width)+1)*width:g})' for k,x in sorted(coords.items()))
            bins[key or 'coordinate_unavailable'].append(values)
        reach=graph['reachability']
        def ready(k):return {i for i,v in reach.get(k,{}).items() if v.get('status')=='ready'}
        l,r=ready(left),ready(right)
        comparisons[f'{left} minus {right}']={'all':paired_test(pairs,config['confidence']),
            'by_initial_coordinate':{k:paired_test(v,config['confidence']) for k,v in sorted(bins.items())},
            'ready_only_left':sorted(l-r),'ready_only_right':sorted(r-l)}
    return comparisons


def report(episodes, graph, facts, predictions, spec, index, previous=None):
    target=spec['active_post_condition'];counts={};dev={};failures=Counter()
    current=[e for e in episodes if e['round_index']==index]
    for c in spec['post_conditions']:
        cid=c['id'];counts[cid]=dict(Counter(v['status'] for v in graph['reachability'].get(cid,{}).values()))
        bins=defaultdict(list)
        for e in current:
            o=e['terminal_outcomes'].get(cid)
            if e['data_role']=='dev_validation':
                bins[str(e['mass'] or 'unspecified')].append(o)
            if o and not o['success']:failures[f"{cid}/{o.get('reason') or e['failure_cause'] or 'unspecified'}"]+=1
        dev[cid]={}
        for k,v in bins.items():
            tested=[x for x in v if x]
            dev[cid][k]={**success_interval(len(tested),sum(x['success'] for x in tested),spec['loop']['confidence']),
                          'attempts':len(v),'unevaluated':len(v)-len(tested)}
    for e in current:
        if e['status']=='failed' and not e['terminal_outcomes']:
            failures[f"before_consumer/{e['failure_cause'] or 'unspecified'}"]+=1
    status=Counter(p['status'] for p in predictions)
    denom=status['confirmed']+status['refuted']
    frontier=[f for f in facts if f['pred'] in ('policy_frontier','expert_frontier','direct_boundary','frontier_gap')]
    data={'round_index':index,'graph_version':graph['graph_version'],'target_consumer':target,
            'regions':counts,'frontiers':frontier,'development':dev,'failure_causes':dict(failures),
            'predictions':predictions,'prediction_accuracy':status['confirmed']/denom if denom else None,
            'prediction_denominator':denom,'paired_consumers':paired_report(current,graph,spec)}
    if previous:
        changes={}
        for field in ('regions','frontiers','development','failure_causes','prediction_accuracy','paired_consumers'):
            if data[field]!=previous.get(field):
                changes[field]={'previous':previous.get(field),'current':data[field]}
        data['changes']=changes
    else:data['changes']={}
    return data


def summary_markdown(claims, data):
    if not claims:raise ValueError('Round summary needs at least one grounded claim')
    lines=['# 本轮结果','']
    for claim in claims:
        paths=claim['paths']
        if not paths or not claim.get('text','').strip():raise ValueError('Missing summary evidence or claim')
        for path in paths:
            if not isinstance(path,str) or not path.startswith('/') or path=='/':
                raise ValueError(f'Invalid round report pointer {path}')
            node=data
            try:
                for part in path[1:].split('/'):
                    part=part.replace('~1','/').replace('~0','~')
                    if isinstance(node,list) and (not part.isdigit() or int(part)>=len(node)):raise IndexError(part)
                    node=node[int(part)] if isinstance(node,list) else node[part]
            except (KeyError,IndexError,ValueError,TypeError) as exc:
                raise ValueError(f'Unknown round report field {path}') from exc
        lines.append('- '+claim['text']+' '+' '.join('(`round_report.json#'+p+'`)' for p in paths))
    return '\n'.join(lines)+'\n'

