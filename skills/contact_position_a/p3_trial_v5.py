"""Evidence-producing multistep rollout and separate randomized terminal measurements."""
from pathlib import Path
import random
import time
import joblib
import numpy as np

from bottle_scan.geometry import pose_error
from bottle_scan.state_graph import campaign
from bottle_scan.state_graph.noise import measurement_distance
from bottle_scan.state_graph.skill import SymbolicRecoverySkill
from .ah_geometry import coord_names,action_kind,position_limits,validate_cell,position_allowed
from .p3_trial import observe_rows,learned_action,direct_evidence


def expert_action(raw,spec):
    provenance=raw['provenance']
    if provenance.get('coordinate_source')!='measured' or not provenance.get('placement_reference_valid'):
        raise ValueError('scripted_expert_requires_measured_placement')
    x=np.array([raw['plan_coords'][k] for k in coord_names(spec)])
    if not position_allowed(x,spec):raise ValueError('start_outside_envelope')
    limit=np.array(spec['task']['action_space']['max_step'])
    k=int(np.argmax(abs(x)))
    if abs(x[k])<=spec['geometry']['coordinate_residual_mm']:return None
    u=np.zeros(2)
    u[k]=np.clip(-x[k],-limit[k],limit[k])
    return {'kind':action_kind(spec),'u':u.tolist()}


def consumer_plan(target,spec,req):
    options=spec.get('consumer',{})
    conditions=[c['id'] for c in spec['post_conditions']]
    if target not in conditions:raise ValueError('Unknown recovery target')
    if not options.get('dual_paired'):
        return [target]
    if len(conditions)!=2:raise ValueError('Dual protocol requires exactly two declared consumers')
    rng=random.Random(req['consumer_seed'])
    rng.shuffle(conditions)
    return conditions


def pair_status(first,second,bundle,spec):
    if second['provenance'].get('placement_continuity') is False:
        return {'paired':False,'reason':'placement_changed_or_unconfirmed'}
    if not bundle:return {'paired':False,'reason':'no_prior_pair_metric'}
    cfg=spec['consumer']
    pos,ang=pose_error(first['provenance']['measured_pose'],second['provenance']['measured_pose'])
    distance=measurement_distance(first,second,bundle['noise'])
    radius=bundle['graph'].get('radii',{}).get('similar_radius',1.)
    passed=bool(distance<=radius and pos<=cfg['pair_position_tolerance_mm'] and ang<=cfg['pair_orientation_tolerance_deg'])
    return dict(paired=passed,distance=distance if np.isfinite(distance) else None,similar_radius=radius,
                position_error_mm=pos,angle_error_deg=ang,model_id=bundle['model_id'])


def frozen_recipe(bundle,request,base_spec):
    """Verify the campaign recipe and independently frozen candidate before selecting its target."""
    if bundle['model_id']!=request['model_id']:raise ValueError('Model identity mismatch')
    if role_frozen:=request.get('frozen_model_id'):
        if role_frozen!=bundle['model_id']:raise ValueError('Frozen model mismatch')
    fitted=bundle['spec']
    if 'effective_spec_digest' in request or 'base_spec_digest' in request:
        persisted=campaign.read(request['spec'])
        if campaign.digest(persisted)!=request.get('base_spec_digest'):
            raise ValueError('Persisted campaign specification differs from request')
        active=base_spec['active_post_condition']
        if active not in (persisted['active_post_condition'],request['post_condition']) or {**base_spec,'active_post_condition':persisted['active_post_condition']}!=persisted:
            raise ValueError('Live experiment contract changed outside the selected target')
        expected=request.get('effective_spec_digest')
        if campaign.digest(fitted)!=expected or campaign.digest(campaign.read(request['effective_spec']))!=expected:
            raise ValueError('Frozen feature recipe or persisted training specification differs from request')
    elif fitted!=base_spec:
        raise ValueError('Model uses another recipe without its training receipt')
    return fitted


def model_row(raw,trial,directory,spec):
    # Extractors can use exact encoded frames captured at the original contact.
    # Save the compact source locator or feature vector, never all raw arrays in JSON.
    return observe_rows([raw],trial,directory,spec)[0]


def run_trial(request,directory,spec,cfg,io,*,sources=None):
    directory=Path(directory)
    start=time.monotonic()
    trial,role=request['trial_id'],request['role']
    target=request['target_consumer'] if spec.get('consumer',{}).get('target_from_rule') else spec['active_post_condition']
    if request['post_condition']!=target:raise ValueError('Recovery target differs from experiment request')
    declared={'direct_baseline':'direct','expert_recovery':'expert','policy_frontier':'policy',
              'development':'policy','independent':'policy','exploration':'planned' if request['actions'] else 'direct'}
    ctl=declared[role]
    if (ctl=='policy')!=(request['mode']=='learned') or (ctl!='planned' and request['actions']):
        raise ValueError('Unexpected action source for declared role')
    validate_cell({'actions':request['actions'],'settings':request['settings'],'blocks':[request['block']]},spec,'',{})
    transitions_only=cfg.get('execution',{}).get('scope')=='transitions_only'
    if transitions_only and role!='exploration':raise ValueError('Transition acquisition cannot evaluate consumer or recovery roles')
    order=[] if transitions_only else consumer_plan(target,spec,request)
    bundle=skill=None
    fitted_spec=spec
    if request.get('model_bundle'):
        bundle=joblib.load(request['model_bundle'])
        fitted_spec=frozen_recipe(bundle,request,spec)
        if role=='independent' and request.get('frozen_model_id')!=request['model_id']:
            raise ValueError('Independent B requires the frozen version')
        bundle={**bundle,'spec':{**fitted_spec,'active_post_condition':target}}
    if ctl=='policy':
        if bundle is None:raise ValueError('No frozen recovery skill')
        skill=SymbolicRecoverySkill(bundle)
        if sources is None:
            base=Path(request['campaign'])
            sources=campaign.all_rows(base,campaign.read(base/'state.json'))
    reason=None
    truth={'initial_truth':None}
    wins,receipts,actions,predictions,terms=[],[],[],[],[]
    outcomes={}
    pair={'paired':False,'reason':'single_terminal_window'}
    expert=request.get('expert_source')
    tags=dict(data_role=request.get('data_role'),round_index=request.get('round_index'),controller_role=ctl,
              target_consumer=target,expert_source=expert if ctl=='expert' else None)
    io.confirm(request)
    if hasattr(io,'prepare'):io.prepare(request)
    maximum=spec['runtime']['max_steps']
    declared_exit=False
    for index in range(maximum+1):
        with io.contact(index) as raw:
            if raw is None:
                reason=getattr(io,'contact_failure',None) or 'contact_lost'
                break
            raw.update(tags)
            wins.append(raw)
            if index:
                io.link(wins[-2],raw,receipts[-1],actions[-1]['u'])
                if not raw.get('u_receipt'):
                    reason='unmatched_motion_receipt'
                    break
            row=model_row(raw,trial,directory,fitted_spec) if skill is not None else None
            action=None
            if ctl=='policy':
                if index==0:truth=direct_evidence(row,bundle,sources,request['block']['mass_class'])
                pred=skill.infer(row)
                predictions.append(pred)
                campaign.atomic(directory/'decisions.json',dict(model_id=bundle['model_id'],target_consumer=target,predictions=predictions,**truth))
                if pred['state']=='ready':
                    declared_exit=True
                elif pred['state']!='recoverable':
                    reason='model_rejected'
                    break
                elif index>=maximum:
                    reason='steps_exhausted'
                    break
                elif spec['graph'].get('success_value'):
                    action=pred.get('action')
                else:
                    action=learned_action(pred,bundle,spec)
                if not declared_exit and not action:
                    reason='no_executable_action'
                    break
            elif ctl=='expert':
                if expert=='scripted_coordinate':
                    try:action=expert_action(raw,spec)
                    except ValueError as e:
                        reason=str(e)
                        break
                elif expert=='teleop' and hasattr(io,'expert_action'):
                    action=io.expert_action(raw)
                else:
                    reason='expert_provider_unavailable'
                    break
            elif index<len(request['actions']):
                action=request['actions'][index]
            if action and index==maximum:
                reason='steps_exhausted'
                break
            if not action:
                raw['is_terminal']=True
                raw['provenance']['direct_A']=ctl=='direct'
                if transitions_only:break
                outcome=io.consume_for(order[0]) if hasattr(io,'consume_for') else io.consume()
                raw['consumer_outcome']={**outcome,'consumer_id':order[0]}
                outcomes[order[0]]=raw['consumer_outcome']
                terms.append(raw)
                break
            # Validate against independently measured current position or cumulative planned path.
            measured=raw['provenance'].get('coordinate_source')=='measured'
            cell={'settings':raw['plan_coords'] if measured else request['settings'],
                  'actions':[action] if measured else [*actions,action]}
            try:validate_cell(cell,spec,'',{})
            except ValueError:
                reason='action_outside_envelope'
                break
        campaign.atomic(directory/f'action_{index}.json',action)
        receipts.append(io.move(action,wins[-1]))
        actions.append(action)
    if len(order)==2 and terms:
        primary=outcomes[order[0]]
        disrupted=any(primary.get(k) for k in ('contact_lost','object_moved','dropped')) or primary.get('return_contact') is False
        if disrupted:
            pair={'paired':False,'reason':'second_skipped_object_or_contact_change'}
        else:
            with io.contact(len(wins)) as raw:
                if raw is None:pair={'paired':False,'reason':'second_contact_invalid'}
                elif raw['provenance'].get('placement_continuity') is False:
                    raw.update(tags,is_terminal=False)
                    wins.append(raw)
                    pair={'paired':False,'reason':'second_skipped_placement_changed_or_unconfirmed'}
                else:
                    raw.update(tags,is_terminal=True)
                    raw['provenance']['direct_A']=ctl=='direct'
                    wins.append(raw)
                    if bundle:
                        a,b=(model_row(w,trial,directory,fitted_spec) for w in wins[-2:])
                        pair=pair_status(a,b,bundle,spec)
                    else:pair={'paired':False,'reason':'no_prior_pair_metric'}
                    outcome=io.consume_for(order[1])
                    raw['consumer_outcome']={**outcome,'consumer_id':order[1]}
                    outcomes[order[1]]=raw['consumer_outcome']
                    terms.append(raw)
    reference_windows=getattr(io,'reference_windows',[])
    for raw in reference_windows:
        raw.update(tags)
    # Only actually successful expert demonstrations supply training transitions.
    if ctl=='expert' and actions:
        o=outcomes.get(target,{})
        for raw in wins:
            if raw.get('u_receipt'):
                raw['source']='expert' if o.get('L1') is True and o.get('L2') is True else 'expert_attempt'
                raw['u_receipt']['source']=raw['source']
    rows=observe_rows([*wins,*reference_windows],trial,directory,spec)
    if fitted_spec['observation']['modalities']!=spec['observation']['modalities'] or fitted_spec['observation'].get('tac_version')!=spec['observation'].get('tac_version'):
        by_id={r['node_id']:r for r in rows}
        methods=fitted_spec['observation']['representations']['current'].split('+')
        from .p3_trial import observe_rows as extract
        effective_rows=extract([*wins,*reference_windows],trial,directory,fitted_spec)
        for eff in effective_rows:
            original=by_id[eff['node_id']]
            for method in methods:
                key=('tac' if method in ('legacy_geometry','calibrated_rh') else 'force' if method=='force' else fitted_spec['observation']['metric']['input_modality'])
                if key not in eff['obs']:continue
                name=fitted_spec['observation']['tac_embed_readout'] if method=='tac_metric' else method
                original.setdefault('representations',{})[name]=eff['obs'][key]
                original.setdefault('representation_versions',{})[name]=eff['extractor_versions'][key]
    posts={raw['consumer_outcome']['consumer_id']:next(r for r in rows if r['node_id']==raw['node_id'])['post'][raw['consumer_outcome']['consumer_id']] for raw in terms}
    post=posts.get(target,{})
    success=post.get('success')
    if not reason and success is False:reason=outcomes[target].get('failure_cause') or 'consumer_failed'
    first=next((r for r in rows if r['window_index']==0),{})
    assess=None if ctl!='policy' else dict(**truth,evaluated=bool(post.get('evaluated')),success=success,
           target_truth=success,exit_declared=declared_exit,invoked=bool(receipts),rejected=not declared_exit or success is None,
           evidence=post.get('evidence',[]),task_success=None)
    result=dict(trial_id=trial,status='failed' if reason else 'complete',nodes=rows,assessment=assess,
                attempt_reason=reason,failure_cause=reason,initial_coords=first.get('plan_coords',{}),
                initial_coordinate_evidence=first.get('provenance',{}),consumer_order=order,
                consumer_seed=request.get('consumer_seed'),paired=pair['paired'],pairing=pair,
                terminal_outcomes=posts,consumer_outcomes=outcomes,consumer_outcome=outcomes.get(target),
                preparation=getattr(io,'preparation_report',None),
                action_commands=actions,move_receipts=list(map(str,receipts)),cost={'actions':len(receipts),'probe_refreshes':getattr(io,'probe_refresh_count',0),'wall_seconds':time.monotonic()-start},**tags)
    campaign.atomic(directory/'completed_capture.json',result)
    return result
