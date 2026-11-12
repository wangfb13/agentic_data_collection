"""Specification-declared B acceptance from explicitly assessed trial outcomes."""
from .facts import fact
from .common import group_id


def evaluate_B(records,spec,role,condition=None):
    condition=condition or spec['active_post_condition'];config=spec['acceptance'][role+'_B']
    chosen=[r for r in records if r.get('role')==role and r.get('post_condition')==condition and r.get('evaluated') is True]
    if role=='independent':chosen=[r for r in chosen if r.get('frozen_model_id') and r.get('post_freeze') is True]
    if len({r['group_id'] for r in chosen})!=len(chosen):raise ValueError('Duplicate assessed trial group')
    normal=[r for r in chosen if r.get('initial_truth')=='Ready'];needs=[r for r in chosen if r.get('initial_truth')=='NeedAction']
    target=[r for r in chosen if r.get('target_truth') is True];not_target=[r for r in chosen if r.get('target_truth') is False]
    def rate(part,key):return sum(r.get(key) is True for r in part)/len(part) if part else None
    metrics={'trials':len(chosen),'normal':len(normal),'needs':len(needs),'target':len(target),'not_target':len(not_target),
             'verified_recoveries':sum(r.get('invoked') is True and r.get('success') is True for r in needs),
             'invocation_recall':rate(needs,'invoked'),'normal_specificity':sum(r.get('invoked') is False for r in normal)/len(normal) if normal else None,
             'recovery_success_rate':rate(needs,'success'),'exit_recall':rate(target,'exit_declared'),
             'rejection_rate':rate(chosen,'rejected'),'false_success':sum(r.get('exit_declared') is True for r in not_target),'task_success_rate':rate(chosen,'task_success')}
    checks={}
    for key,value in config.items():
        if key.startswith('min_'):
            name=key[4:]
            if name=='task_success_rate' and not config.get('require_task_success'):continue
            checks[key]=metrics.get(name) is not None and metrics[name]>=value
        elif key.startswith('max_'):
            name=key[4:];checks[key]=metrics.get(name) is not None and metrics[name]<=value
    if spec['acceptance'].get('requires_ready_and_need_action_coverage'):checks['both_initial_states']=bool(normal and needs)
    passed=bool(checks) and all(checks.values());predicate='indep_B_pass' if role=='independent' else 'dev_B_pass'
    return {'passed':passed,'metrics':metrics,'checks':checks,'facts':[fact(predicate,[condition],[r['group_id'] for r in chosen],metrics)] if passed else [],'role':role}
