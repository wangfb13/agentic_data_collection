from copy import deepcopy
import pytest
from bottle_scan.max_expert_demonstration_finder.evidence import label_record,measured,SIGN
from bottle_scan.max_expert_demonstration_finder.demo import Fixture,fixture
from bottle_scan.max_expert_demonstration_finder.search import report,offers,validate_choice
from bottle_scan.max_expert_demonstration_finder.ledger import Ledger,costs
from bottle_scan.max_expert_demonstration_finder.engine import execute,reconcile
from bottle_scan.max_expert_demonstration_finder.episodes import summarize
from bottle_scan.max_expert_demonstration_finder.train import training_examples
from bottle_scan.state_graph.facts import coordinate_evidence

def setup(tmp,mode='monotone',**params):
    m=fixture(mode=mode);m.update(params);db=Ledger(tmp/'store.db');rid=db.new(m);return m,db,rid,Fixture(m)

def run(t,x=0,r='policy_frontier'):
    m,d,i,s=t;c=next(c for c in offers(d.round(i),d.attempts(i)) if c['x']==x and c['role']==r)
    return d.get(execute(d,i,c,s,s))

def test_different_starts_same_end_and_direct_controls(tmp_path):
    t=setup(tmp_path);run(t);a=run(t,2.5);b=run(t,-2.5);m,db,r,_=t
    for v in (a,b):
        e=summarize(v,m);assert e['trajectory'][-1]['after']['sensor']==0 and e['autonomous_recovery_witness']
    p=report(db.round(r),db.attempts(r));assert not p['direct_success']['overall']['n']
    assert {x['x'] for x in p['policy_recovery_frontier']['cells'] if x['screen']['n']}=={-2.5,0,2.5}
    assert not summarize(run(t,2.5,'direct_baseline'),m)['success']

@pytest.mark.parametrize('contact',['no_contact','unilateral','unstable_bilateral'])
def test_no_node_denominator_and_no_consumer(tmp_path,contact):
    t=setup(tmp_path,mode='no_contact');m,d,r,s=t;s.extra={'contact_state':contact,'measurement':None}
    e=summarize(run(t),m)
    assert e['first_contact_state']==contact and e['first_valid_node'] is None and not e['recovery_completed'] and e['consumer_outcome'] is None
    s=report(d.round(r),d.attempts(r))['policy_recovery_frontier']
    assert s['overall']==dict(n=1,successes=0,rate=0.) and s['first_coordinate_coverage']==0 and s['preparation_cells'][2]['n']==1

def test_post_contact_not_promoted_and_expert_witness(tmp_path):
    t=setup(tmp_path);m,d,r,s=t;m2=deepcopy(m);m2['expert']['permitted_start_contacts']=['unilateral']
    r=d.new(m2);s=Fixture(m2);t=(m2,d,r,s)
    observe=s.observe
    def obs(first):
        o=observe(first)
        if first:o.update(contact_state='unilateral',node_id=None,measurement=None)
        return o
    s.observe=obs
    e=summarize(run(t,2.5,'expert_recovery'),m2)
    assert e['first_contact_offset_mm'] is None and e['first_valid_node'] and e['first_contact_state']=='unilateral'
    assert not e['autonomous_recovery_witness']

def test_ready_without_actual_post(tmp_path):
    t=setup(tmp_path);m,d,r,s=t;f=s.perform
    def incomplete(req):
        v=f(req)
        if req['operation']=='post':v.update(consumer_outcome={'L1':True},status='complete')
        return v
    s.perform=incomplete;e=summarize(run(t),m)
    assert e['recovery_completed'] and e['consumer_outcome'] is None and not e['success']

def test_moved_missing_label_and_cost(tmp_path):
    t=setup(tmp_path);m,d,r,s=t;s.extra={'object_moved':True};e=summarize(run(t),m)
    assert not e['success'] and e['first_contact_offset_mm'] is None and e['preparation_offset_mm'] is None and e['charged_dispatches']==2

def test_freeze_role_isolation_and_unknown_no_planned_action(tmp_path):
    t=setup(tmp_path,mode='reject');m,d,r,s=t;run(t);e=summarize(run(t,2.5),m)
    assert e['termination_reason']=='model_ood' and not e['steps_attempted']
    changed=deepcopy(m);changed['reject_rules']['other']=True
    assert d.round(r)['manifest']==m
    co=offers(d.round(r),d.attempts(r))[0];co['model_id']='else'
    with pytest.raises(ValueError):validate_choice(co,d.round(r),d.attempts(r))
    r2=d.new(changed);assert report(d.round(r2),[])['policy_recovery_frontier']['overall']['n']==0

def test_hole_and_envelope(tmp_path):
    t=setup(tmp_path,mode='island')
    for x in (0,2.5,5):run(t,x)
    _,d,r,_=t;s=report(d.round(r),d.attempts(r))['policy_recovery_frontier']['sides']['positive']
    assert s['nonmonotonic_regions']==[[2.5,5.0]] and s['envelope_limited'] and s['farthest_observed_success']==5 and s['verified_connected_range'] is None

def test_exact_confirm_no_peeking_and_holdout_excluded(tmp_path):
    m=fixture();m['statistics'].update(screen_n=3,confirmation_n=5,target=.9);d=Ledger(tmp_path/'t');r=d.new(m);s=Fixture(m);t=(m,d,r,s)
    for _ in range(3):run(t)
    d.confirm(r,[0]); 
    for _ in range(3):run(t)
    raw=d.attempts(r)
    p=report(d.round(r),raw)['policy_recovery_frontier']['cells'][2]
    assert p['screen']['successes']==3 and p['confirmation']['interval'] is None
    for _ in range(2):run(t)
    cc=report(d.round(r),d.attempts(r))['policy_recovery_frontier']['cells'][2]['confirmation']
    assert cc['status']=='inconclusive' and cc['interval'] and cc['matched']==5
    assert not training_examples(d.attempts(r),m)[1] and all(c['role']=='direct_baseline' for c in offers(d.round(r),d.attempts(r)))
    with pytest.raises(ValueError):d.confirm(r,[0])

def test_cost_retry_restart_idempotency(tmp_path):
    t=setup(tmp_path,mode='technical');m,d,r,s=t;a=run(t);pending=a['id']
    d2=Ledger(tmp_path/'store.db');assert d2.pending()==pending
    assert not offers(d2.round(r),d2.attempts(r))
    with pytest.raises(ValueError):d2.dispatch(pending,'touch',{})
    reconcile(d2,pending,[],'checked stopped');reconcile(d2,pending,[],'checked stopped')
    c=costs(d2.attempts(r));assert c['physical_attempts_charged']==1 and c['unresolved_dispatches']==1 and c['touches']==1

def test_human_takeover_not_autonomous_and_stronger_expert(tmp_path):
    t=setup(tmp_path);m,d,r,s=t;s.extra={'human_intervention':True}
    assert not summarize(run(t),m)['success']
    a=run(t,2.5,'expert_recovery');e=summarize(a,m)
    assert e['expert_witness'] and not e['autonomous_recovery_witness'] and training_examples([a],m)[0]==[a['id']]
    stronger=deepcopy(m);stronger['expert']['stronger']=True
    assert not training_examples([a],stronger)[0]
    a['phase']='confirm';assert not training_examples([a],m)[0]

def test_exhausted_step_stays_in_denominator(tmp_path):
    t=setup(tmp_path);m,d,r,s=t;run(t)
    s.decide=lambda obs,step:s.act(obs,-.25,'graph_policy')
    e=summarize(run(t,2.5),m);assert e['termination_reason']=='steps_exhausted' and not e['success'] and e['steps_measured']==2

def test_no_truth_or_spoofed_identity(tmp_path):
    t=setup(tmp_path);m,d,r,s=t;a=run(t);first=a['events']['reply/touch']['observation'];lab=first['measurement']
    row={'plan_coords':{'a_offset_mm':0},'t_ns':first['timestamp_ns'],'provenance':{'coordinate_schema':'fixed_contact_a_v1','reference_digest':'fixture_ref','target_definition_id':'white_fixture'}}
    assert coordinate_evidence(row,'a_offset_mm') is None
    # Adding a plausible value or a digest alone still does not point to a real measurement.
    row['provenance']['supervision']=lab;assert coordinate_evidence(row,'a_offset_mm') is None
    import json,hashlib
    raw=tmp_path/'actual.log';raw.write_text('test acquisition')
    sample=dict(lab['record'],raw_source=str(tmp_path/'raw.json'))
    (tmp_path/'raw.json').write_text(json.dumps({'measurement':{k:v for k,v in sample.items() if k!='raw_source'},
        'acquisition_files':[{'path':str(raw),'sha256':hashlib.sha256(raw.read_bytes()).hexdigest()}]}))
    lab=label_record(sample);row['provenance']['supervision']=lab;assert coordinate_evidence(row,'a_offset_mm')==0
    for k,v in [('reading_mm',123),('reference_digest','other'),('unit','m'),('axis_sign','object'),('valid',False),('phase','post')]:
        bad=deepcopy(lab);bad['record'][k]=v
        if k!='reading_mm':bad=label_record(bad['record'])
        row['provenance']['supervision']=bad;assert coordinate_evidence(row,'a_offset_mm') is None


def test_standard_witness_signed_projection_and_invalidation():
    import numpy as np
    from bottle_scan.max_expert_demonstration_finder.evidence import sha
    T=np.eye(4);T[:3,:3]=[[0,0,1],[0,1,0],[-1,0,0]]
    w=dict(accepted_standard=True,raw_source='fixture:standard',timestamp_ns=1,reference_digest='r',placement_id='one',tcp_mm=T.tolist(),a_tcp=[0,0,1])
    T[0,3]=5
    rec=dict(source='measured_local_standard_tcp',reference_digest='r',target_definition_id='t',unit='mm',axis_sign=SIGN,valid=True,
             object_moved=False,phase='first_contact',timestamp_ns=3,raw_source='fixture:tcp',uncertainty_mm=.1,standard_witness=w,
             basis_id=sha(w),placement_id='one',tcp_mm=T.tolist())
    l=label_record(rec);assert measured(l,'r','t',axis=[0,0,1],stamp=3)==5
    assert measured(l,'r','t',axis=[0,0,1],invalid=[sha(w)]) is None
    assert measured(label_record(dict(rec,placement_id='two')),'r','t',axis=[0,0,1]) is None

def test_reject_receipt_and_unverified_sensor_return(tmp_path):
    t=setup(tmp_path);m,d,r,s=t;run(t);old=s.perform
    def broken(req):
        v=old(req)
        if req['operation']=='correct_retouch':v['receipt']['actual_delta_a_mm']*=2
        return v
    s.perform=broken;e=summarize(run(t,2.5),m)
    assert not e['success'] and not e['steps_measured'] and 'invalid_receipt' in e['phenomena']

def test_full_budget_and_no_optimistic_midpoint(tmp_path):
    m=fixture(False);m['limits']['physical_attempts']=1
    db=Ledger(tmp_path/'b');r=db.new(m);s=Fixture(m)
    run((m,db,r,s));rr=db.round(r);assert not offers(rr,db.attempts(r))
    assert report(rr,db.attempts(r))['policy_recovery_frontier']['sides']['positive']['budget_limited']

def test_missing_provider_never_manufactures_demonstration(tmp_path):
    m=fixture(False);db=Ledger(tmp_path/'b');r=db.new(m)
    rep=report(db.round(r),[]);assert not rep['expert_witnessed_recoverable_set']
    assert rep['expert_requests'][0]['status']=='awaiting_named_provider'
    assert not any(v['role']=='expert_recovery' for v in offers(db.round(r),[]))

def test_real_record_content_and_uncertainty(tmp_path):
    import json,hashlib
    from bottle_scan.max_expert_demonstration_finder.proof import check_measurement
    from bottle_scan.max_expert_demonstration_finder.search import measured_cell
    t=setup(tmp_path);m,_,_,_=t;l=run(t)['events']['reply/touch']['observation']['measurement']
    record=l['record'];raw=tmp_path/'raw';raw.write_bytes(b'instrument bytes for test')
    record['raw_source']=str(tmp_path/'sidecar.json')
    l=label_record(record);assert not check_measurement(l,m,'first_contact',record['timestamp_ns'])
    (tmp_path/'sidecar.json').write_text(json.dumps({'measurement':{k:v for k,v in record.items() if k!='raw_source'},'acquisition_files':[{'path':str(raw),'sha256':hashlib.sha256(raw.read_bytes()).hexdigest()}]}))
    assert check_measurement(l,m,'first_contact',record['timestamp_ns'])
    raw.write_bytes(b'changed');assert not check_measurement(l,m,'first_contact',record['timestamp_ns'])
    record['uncertainty_mm']=6;assert measured_cell({'first_contact_offset_mm':0,'first_contact_measurement':label_record(record)},m) is None

def test_old_choice_cannot_dispatch_next_repeat(tmp_path):
    from bottle_scan.max_expert_demonstration_finder.evidence import sha
    m=fixture();m['statistics']['screen_n']=3
    d=Ledger(tmp_path/'b');r=d.new(m);s=Fixture(m)
    c=next(v for v in offers(d.round(r),[]) if v['role']=='policy_frontier')
    execute(d,r,c,s,s)
    with pytest.raises(ValueError):execute(d,r,c,s,s)
    assert len(d.attempts(r))==1

def test_replacement_charges_and_confirmation_allocation(tmp_path):
    m=fixture(False,mode='technical');m['limits']['max_exploration_replacements']=1
    d=Ledger(tmp_path/'b');r=d.new(m);s=Fixture(m);t=(m,d,r,s)
    a=run(t);reconcile(d,a['id'],[],'checked')
    a=run(t);reconcile(d,a['id'],[],'checked')
    assert costs(d.attempts(r))['physical_attempts_charged']==2
    assert not any(x['role']=='policy_frontier' and x['x']==0 for x in offers(d.round(r),d.attempts(r)))

def test_early_termination_never_confirms_entire_family(tmp_path):
    t=setup(tmp_path);m,d,r,s=t;run(t);d.confirm(r,[0]);run(t);d.close(r)
    c=report(d.round(r),d.attempts(r))['policy_recovery_frontier']['cells'][2]
    assert c['confirmation']['interval'] is None

def test_forcepolicy_actual_consumer_extra_null_variant_and_wrong_version():
    from bottle_scan.max_expert_demonstration_finder.episodes import post_result
    c={'id':'handoff','evaluator':'skills.contact_position_a.ready:handoff_ready','consumer':{'version':'frozen','controller':'existing_forcepolicy','params':{'closure_floor':.33}}}
    o={'post_condition':'handoff','consumer_outcome':{'version':'frozen','controller':'existing_forcepolicy','params':{'closure_floor':.33,'variant':None},'L1':True,'evidence':[{'event':'main_resumed'}]}}
    assert post_result(o,c) is True
    o['consumer_outcome']['params']['variant']='other';assert post_result(o,c) is None
