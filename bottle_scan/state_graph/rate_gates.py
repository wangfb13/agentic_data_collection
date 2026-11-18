"""Predeclare denominators; evaluate on the independently observed classes."""
import math
from statistics import NormalDist
from .round_report import success_interval

DENOMINATOR={'invocation_recall':'NeedAction','recovery_success_rate':'NeedAction',
             'normal_specificity':'Ready','exit_recall':'target','task_success_rate':'all'}


def minimum_counts(acceptance, role):
    z=NormalDist().inv_cdf((1+acceptance['rate_confidence'])/2)
    required={}
    for name in acceptance['primary_rates']:
        if name not in DENOMINATOR or 'min_'+name not in acceptance[role+'_B']:
            raise ValueError('Missing primary rate definition or threshold: '+name)
        target=acceptance[role+'_B']['min_'+name]
        if not 0<target<1:raise ValueError('Wilson primary threshold must be between zero and one')
        required[name]=math.ceil(z*z*target/(1-target))
    return required


def check_plan(plan,total,acceptance):
    required=minimum_counts(acceptance,'independent')
    counts=plan.get('planned_denominators',{})
    if any(type(x) is not int or x<0 or x>total for x in counts.values()) or counts.get('Ready',0)+counts.get('NeedAction',0)>total:
        raise ValueError('Invalid independent B denominator allocation')
    if any('expected_initial_class' in c for c in plan.get('cells',[])):
        from collections import Counter
        allocated=Counter()
        for c in plan['cells']:
            label=c.get('expected_initial_class')
            if label not in ('Ready','NeedAction'):raise ValueError('Unassigned independent B start class')
            allocated[label]+=c['reps']*len(c.get('blocks',[{}]))
        if any(counts.get(k,0)!=allocated[k] for k in ('Ready','NeedAction')):raise ValueError('Independent B denominator allocation disagrees with its cells')
    for rate,n in required.items():
        cls=DENOMINATOR[rate];available=total if cls=='all' else counts.get(cls,0)
        if available<n:raise ValueError(f'Independent B insufficient planned denominators: {counts}, total={total}; minimum per primary rate={required}')


def bounds(records,acceptance):
    result={}
    for name in acceptance['primary_rates']:
        field=DENOMINATOR[name]
        group=[r for r in records if field=='all' or r.get('initial_truth')==field or
               (field=='target' and r.get('target_truth') is True)]
        truth={'invocation_recall':('invoked',True),'normal_specificity':('invoked',False),
               'recovery_success_rate':('success',True),'exit_recall':('exit_declared',True),
               'task_success_rate':('task_success',True)}[name]
        s=sum(r.get(truth[0]) is truth[1] for r in group)
        result[name]=success_interval(len(group),s,acceptance['rate_confidence'])
    return result
