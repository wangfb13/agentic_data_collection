"""Outer-fold operating points and enrichment among shared failed trials."""
from collections import defaultdict
from math import floor
from scipy.stats import fisher_exact
from .common import group_id
from .facts import coordinate_evidence, fact


def operating_curve(predictions,thresholds):
    result=[]
    for threshold in thresholds:
        selected=[r for r in predictions if r['eligible'] and r['score'] is not None and r['score']>=threshold]
        result.append(dict(threshold=threshold,correct=sum(r['success'] for r in selected),
                           false=sum(not r['success'] for r in selected)))
    return result


def error_analysis(versions,rows,spec,settings,*,coordinate_reader=None):
    by_node={r['node_id']:r for r in rows}
    base=list(versions.values())[0]['heldout']
    failed={r['group'] for r in base if not r['success']}
    common=failed.copy()
    for result in versions.values():
        common &= {x['group'] for x in result['heldout'] if not x['success'] and x['accepted']}
    records=defaultdict(list)
    for x in base:
        if x['success']:continue
        row=by_node[x['node']]
        attrs=row.get('attrs',{});prov=row.get('provenance',{})
        location={k:coordinate_evidence(row,k,spec) for k,v in spec['perturbation_variables'].items() if v['kind']=='continuous'}
        source=prov.get('coordinate_source')
        if coordinate_reader:
            location,source=coordinate_reader(row,location,source)
        records[x['group']].append(dict(node=x['node'],trial_id=row['trial_id'],mass_class=attrs.get('mass_class'),
            session=row.get('session'),placement=prov.get('placement_group') or attrs.get('placement_group') or
            row.get('placement_group') or prov.get('placement_id') or prov.get('placement_reference_id'),time_period=attrs.get('time_period'),
            coordinates_mm=location,coordinate_source=source,source_file=prov.get('meta_path')))
    levels=defaultdict(set)
    for group,windows in records.items():
        for key in settings['error_attributes']:
            values={v.get(key) for v in windows}
            if len(values)==1 and None not in values:levels[key,values.pop()].add(group)
        for key in windows[0]['coordinates_mm']:
            bins={floor(v['coordinates_mm'][key]/settings['coordinate_bin_mm']) for v in windows if v['coordinates_mm'][key] is not None}
            if len(bins)==1 and all(v['coordinates_mm'][key] is not None for v in windows):
                levels[key,bins.pop()].add(group)
    tests=[]
    for (attr,level),groups in levels.items():
        known=set().union(*(ids for (k,_),ids in levels.items() if k==attr))
        if len(known)==len(groups) or not (known&common):continue
        other=known-groups
        p=float(fisher_exact([[len(groups&common),len(groups-common)],[len(other&common),len(other-common)]],alternative='greater').pvalue)
        tests.append(dict(attr=attr,level=level,p=p,n=len(groups),known_failures=len(known),shared=len(groups&common),groups=sorted(groups&common)))
    facts=[]
    for item in tests:
        item['p_adjusted']=min(1.,item['p']*len(tests))
        if item['p_adjusted']<settings['enrichment_alpha']:
            facts.append(fact('hidden_variable_in_ready',[item['attr']],item['groups'],item))
    return dict(failure_groups=len(failed),common_failure_groups=sorted(common),
                records={g:records[g] for g in sorted(common)},enrichment_tests=tests,facts=facts)


def comparison_tables(versions):
    lines=['| 版本 | 门槛 | 正确放行 | 错误放行 |','|---|---:|---:|---:|']
    counts=set()
    for version,result in versions.items():
        for p in result['operating_curve']:
            lines.append(f"| {version} | {p['threshold']:g} | {p['correct']} | {p['false']} |")
            counts.add(p['false'])
    lines+=['','相同错误放行数时的最高正确放行数。空格（—）表示该版本的候选门槛未取得这个错误数。外层曲线用于比较，不用于选取部署门槛。','',
            '| 错误放行 | '+' | '.join(versions)+' |','|---:'+'|---:'*len(versions)+'|']
    for n in sorted(counts):
        cells=[max((p['correct'] for p in v['operating_curve'] if p['false']==n),default='—') for v in versions.values()]
        lines.append('| '+ ' | '.join(map(str,[n,*cells]))+' |')
    return lines
