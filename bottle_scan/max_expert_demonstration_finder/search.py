"""Finite, separately sided search. Only fresh pre-registered confirmation certifies cells."""
from collections import Counter
import math
from scipy.stats import beta
from .evidence import require,num,ROLES,sha
from .episodes import summarize
from .ledger import costs

def validate_manifest(m):
    for k in ('reference_digest','target_definition_id','model_id','controller','condition','protocol','post_condition','artifacts','limits','statistics','station','reject_rules','tool_version','sensor_version'):
        require(bool(m.get(k)), 'missing '+k)
    require(m['action_source'] in ('graph_policy','offset_servo') and m['environment'] in ('physical','simulation'),'source/environment')
    require(m['post_condition']['consumer']['version'] and m['post_condition']['evaluator'] in
            ('skills.contact_position_a.ready:fixed_ready','skills.contact_position_a.ready:handoff_ready'),'unknown consumer contract')
    l=m['limits'];s=m['statistics'];cap=m['execution_caps']
    for k in ('resolution','expansion','prepare_mm','step_mm','total_mm','seconds','physical_attempts','max_steps','max_software_attempts'):
        require(num(l[k]) and l[k]>0,'limit '+k)
    for k in ('project_physical_attempts','physical_attempts','max_steps','max_software_attempts'):require(type(l[k]) is int,'integer '+k)
    for k in ('prepare_mm','expansion','step_mm','total_mm','seconds','max_steps'):
        require(l[k]<=cap[k],'exceeds execution envelope: '+k)
    require(l['project_physical_attempts']>=l['physical_attempts'],'invalid project budget')
    low,high=l['range'];require(cap['range'][0]<=low<0<high<=cap['range'][1] and max(-low,high)<=l['prepare_mm'],'initial envelope')
    require(all(math.isclose(v/l['resolution'],round(v/l['resolution'])) for v in (low,high,l['expansion'])) and l['resolution']<=l['expansion'],'resolution')
    require(0<=l['cell_tolerance']<l['resolution']/2 and 0<s['target']<1 and 0<s['confidence']<1,'statistics')
    require(type(l.get('max_exploration_replacements',0)) is int and l.get('max_exploration_replacements',0)>=0,'invalid retry limit')
    for k in ('screen_n','confirmation_n'):require(type(s[k]) is int and s[k]>0,k)
    require(m['condition'].get('mass_class') in ('light','heavy') and m['condition'].get('placement_group') and m['condition'].get('approach_group'),'condition')
    if m.get('expert'):
        ex=m['expert']
        require(all(ex.get(k) for k in ('id','version','source','observations','allowed_actions')) and type(ex['stronger']) is bool
                and ex['allowed_actions']==['tcp_translation_a'],'expert scope')
    require(m['axis_tcp'] and abs(sum(v*v for v in m['axis_tcp'])-1)<1e-6 and len(m['axis_tcp'])==3,'tool axis')

def grid(m):
    l=m['limits'];d=l['resolution']
    return [round(j*d,8) for j in range(round(l['range'][0]/d),round(l['range'][1]/d)+1)]

def cell(x,m,error=0.):
    if x is None:return None
    g=grid(m);v=min(g,key=lambda y:abs(y-x))
    return v if abs(v-x)+error<=m['limits']['cell_tolerance'] else None

def measured_cell(e,m,prep=False):
    key='preparation' if prep else 'first_contact'
    uncert=(e.get(key+'_measurement') or {}).get('record',{}).get('uncertainty_mm',0)
    return cell(e.get(key+'_offset_mm'),m,uncert)

def interval(k,n,alpha):
    if n==0:return [0.,1.]
    return [float(beta.ppf(alpha/2,k,n-k+1)) if k else 0.,float(beta.ppf(1-alpha/2,k+1,n-k)) if k<n else 1.]

def report(rr,attempts):
    m=rr['manifest'];g=grid(m);st=m['statistics'];plan=rr['plan'];co=costs(attempts)
    all_ep=[summarize(a,m) for a in attempts];ep=[e for e in all_ep if e['ended_ns'] is not None]
    explore=[e for e in ep if e['phase']=='explore']; facts=[]
    def fact(name,ids,**metrics):
        facts.append(dict(id='F_'+sha([rr['digest'],rr['id'],name,sorted(ids),metrics])[:18],pred=name,value=True,evidence=sorted(ids),metrics=metrics))
    def role_data(role):
        rows=[e for e in ep if e['controller_role']==role]
        rolekey=next(k for k,v in ROLES.items() if v==role)
        cells=[]; prep=[]; rates=lambda xs:{'n':len(xs),'successes':sum(e['success'] for e in xs),'rate':sum(e['success'] for e in xs)/len(xs) if xs else None}
        for x in g:
            xs=[e for e in rows if e['phase']=='explore' and measured_cell(e,m)==x]
            ps=[e for e in rows if e['phase']=='explore' and measured_cell(e,m,True)==x]
            prep.append({'x':x,**rates(ps)})
            cc=[e for e in rows if e['phase']=='confirm' and e['requested_offset_mm']==x]
            matched=[e for e in cc if measured_cell(e,m)==x]
            pre=plan and x in plan['points'] and rolekey in plan['roles']
            independent_units={e['independent_trial_unit'] for e in cc if e['independent_trial_unit']}
            complete=bool(pre and len(cc)==plan['n'] and len(matched)==len(cc) and len(independent_units)==len(cc))
            ci=interval(sum(e['success'] for e in cc),len(cc),(1-st['confidence'])/plan['family_size']) if complete else None
            status='supported' if ci and ci[0]>=st['target'] else 'below_target' if ci and ci[1]<st['target'] else 'inconclusive'
            ids=[e['attempt_id'] for e in xs+cc]
            if ids and role=='policy':fact('recovery_cell_'+status,ids,x=x)
            initial=rates(xs);sampled=bool(len(xs)>=st['screen_n']);p=initial['rate']
            cells.append(dict(x=x,screen=initial,descriptive_interval=interval(initial['successes'],len(xs),1-st['confidence']),
                trend=('pass' if p>=st['target'] else 'fail') if sampled else None,
                confirmation={'allocated_completed':len(cc),'matched':len(matched),'independent_units':len(independent_units),'n_required':plan['n'] if pre else None,
                'successes':sum(e['success'] for e in matched),'interval':ci,'status':status},evidence=ids))
        by={v['x']:v for v in cells};side={}
        for sign,key in ((-1,'negative'),(1,'positive')):
            ordered=sorted((v for v in g if v*sign>=0),key=abs)
            tested=[x for x in ordered if by[x]['trend']];nonmon=[];br=[]
            for a,b in zip(tested,tested[1:]):
                if by[a]['trend']!=by[b]['trend']:
                    br.append([a,b])
                    if by[b]['trend']=='pass':nonmon.append([a,b])
            success=[e['first_contact_offset_mm'] for e in rows if e['success'] and e['first_contact_offset_mm'] is not None and e['first_contact_offset_mm']*sign>=0]
            chain=[]
            for x in ordered:
                if by[x]['confirmation']['status']!='supported':break
                chain.append(x)
            unknown=[];segment=[]
            for x in ordered:
                if by[x]['confirmation']['status']=='inconclusive':segment.append(x)
                elif segment:unknown.append([segment[0],segment[-1]]);segment=[]
            if segment:unknown.append([segment[0],segment[-1]])
            side[key]=dict(farthest_observed_success=max(success,key=abs) if success else None,
                verified_connected_range=[0,chain[-1]] if chain else None,verified_grid=chain,resolution_mm=m['limits']['resolution'],
                boundary_brackets=br,nonmonotonic_regions=nonmon,unknown_intervals=unknown,
                envelope_limited=by[ordered[-1]]['trend']=='pass',
                budget_limited=rr['project_remaining']<=0 or co['physical_attempts_charged']>=m['limits']['physical_attempts'] or len(attempts)>=m['limits']['max_software_attempts'])
            if role=='policy':
                for pred,vs in [('recovery_boundary_bracket',br),('recovery_nonmonotonic',nonmon)]:
                    for a,b in vs:fact(pred,by[a]['evidence']+by[b]['evidence'],interval=[a,b])
                if side[key]['envelope_limited']:fact('frontier_envelope_limited',by[ordered[-1]]['evidence'],side=key)
        valid=[e for e in rows if e['first_contact_state']=='valid_bilateral']
        return dict(overall=rates(rows),physically_dispatched=rates([e for e in rows if e['charged_dispatches']]),conditional_on_initial_need_action=rates([e for e in rows if e['initial_model_state']=='NeedAction']),valid_first_contact=len(valid),contact_fraction=len(valid)/len(rows) if rows else None,
            conditional_on_valid_contact=rates(valid),first_coordinate_coverage=sum(e['first_contact_offset_mm'] is not None for e in rows),
            trusted_mm_fraction=sum(e['first_contact_offset_mm'] is not None for e in rows)/len(rows) if rows else None,
            preparation_cells=prep,cells=cells,sides=side,pending=sum(e['ended_ns'] is None and e['controller_role']==role for e in all_ep))
    direct=role_data('direct');pol=role_data('policy')
    seeds=[e['attempt_id'] for e in explore if e['autonomous_recovery_witness']]
    witnesses=[dict(attempt_id=e['attempt_id'],x=e['first_contact_offset_mm'],expert=e['expert'],demonstration_id=e['demonstration_id'])
               for e in explore if e['expert_witness']]
    if not seeds:fact('recovery_seed_missing',[],model=m['model_id'])
    if witnesses:fact('expert_witness_available',[w['attempt_id'] for w in witnesses])
    missing=[e['attempt_id'] for e in ep if e['first_contact_offset_mm'] is None]
    if missing:fact('initial_coordinate_missing',missing)
    if rr['stage']=='explore':fact('recovery_frontier_open',[],round=rr['id'])
    candidate_failures=[e for e in explore if e['controller_role']=='policy' and not e['success']]
    expert_requests=[dict(round_id=rr['id'],model_id=m['model_id'],role='expert_recovery',
        requested_offset_mm=x,condition=m['condition'],provider=m.get('expert'),status='available' if m.get('expert') else 'awaiting_named_provider',
        supporting_attempts=[e['attempt_id'] for e in candidate_failures if e['requested_offset_mm']==x])
        for x in sorted({e['requested_offset_mm'] for e in candidate_failures} or ({0} if not seeds else set()))]
    return dict(frozen_configuration=m,pending_attempts=[e['attempt_id'] for e in all_ep if e['ended_ns'] is None],expert_requests=expert_requests,project_physical_budget_left=rr['project_remaining'],round_id=rr['id'],round_digest=rr['digest'],model_id=m['model_id'],environment=m['environment'],
        phase=rr['stage'],condition=m['condition'],post_condition=m['post_condition'],direct_success=direct,policy_recovery_frontier=pol,
        expert_witnessed_recoverable_set=witnesses,autonomous_action_witnesses=seeds,confirmation_plan=plan,facts=facts,cost=co,
        valid_evaluations=sum(e['consumer_tested'] for e in ep),phenomena=dict(Counter(p for e in ep for p in e['phenomena'])),
        episodes=all_ep,interpretation='discrete initial-state grid; exploratory brackets are not reliability certificates')

def offers(rr,attempts):
    m=rr['manifest'];l=m['limits'];g=grid(m);s=m['statistics'];counts=Counter((a['role'],a['x']) for a in attempts if a['phase']==rr['stage'])
    if rr['project_remaining']<=0 or rr['stage']=='closed' or any(a['end'] is None for a in attempts) or len(attempts)>=l['max_software_attempts'] or costs(attempts)['physical_attempts_charged']>=l['physical_attempts']:return []
    revision=sha([rr['stage'],rr['plan'],rr['project_remaining'],[(a['id'],a['end']) for a in attempts]])
    def op(name,role,x):
        return dict(op=name,role=role,x=x,round_id=rr['id'],model_id=m['model_id'],manifest_digest=rr['digest'],revision=revision)
    if rr['stage']=='confirm':
        p=rr['plan'];return sorted([op('validate_recovery_frontier',r,x) for x in p['points'] for r in p['roles'] if counts[r,x]<p['n']],
            key=lambda v:(counts[v['role'],v['x']],abs(v['x']),v['x'],v['role']))
    data=report(rr,attempts);ps=data['policy_recovery_frontier'];d=m['limits']['resolution'];seen={x for (role,x),n in counts.items() if role==('policy_frontier' if m['policy_executable'] else 'direct_baseline') and n}
    brackets=[z for side in ps['sides'].values() for z in side['boundary_brackets']]
    inversions=[z for side in ps['sides'].values() for z in side['nonmonotonic_regions']]
    ans=[];ex=m.get('expert');fail={e['requested_offset_mm'] for e in data['episodes'] if e['controller_role']=='policy' and not e['success']}
    def add(priority,n,r,x):ans.append(((priority,counts[r,x],abs(x),x,r),op(n,r,x)))
    for x in g:
        if ex and (x in fail or (not data['autonomous_action_witnesses'] and abs(x)<=l['expansion'])) and counts['expert_recovery',x]<s['screen_n'] and not any(w['x'] is not None and cell(w['x'],m)==x for w in data['expert_witnessed_recoverable_set']):
            add(0 if (x in fail or not data['expert_witnessed_recoverable_set']) else 4,'request_expert_recovery','expert_recovery',x)
        side_seen=[abs(z) for z in seen if z*x>=0]
        if abs(x)>max([0]+side_seen)+l['expansion']+1e-9:continue
        inversion=any(x in z for z in inversions)
        refine=any(min(a,b)<x<max(a,b) and (inversions or abs(x-(a+b)/2)<=d/2) for a,b in brackets)
        for r in ('policy_frontier','direct_baseline'):
            n=counts[r,x]
            tech=sum(a['role']==r and a['x']==x and (a['reason'] or '').startswith(('interrupted','technical')) for a in attempts)
            if n>=s['screen_n']*(2 if inversion else 1)+min(tech,l.get('max_exploration_replacements',0)):continue
            if r=='policy_frontier' and not m['policy_executable']:continue
            if x!=0 and not seen and r!='expert_recovery' and m['policy_executable']:continue
            coarse=math.isclose(abs(x)/l['expansion'],round(abs(x)/l['expansion']))
            add(1 if inversion or refine else 2 if coarse else 3,'repeat_recovery_cell' if n else
                'refine_recovery_bracket' if refine else 'probe_recovery_frontier',r,x)
    return [v for _,v in sorted(ans,key=lambda v:v[0])]

def validate_choice(request,rr,attempts):
    require(request in offers(rr,attempts),'stale, unbound or inapplicable frontier plan')
    return request

