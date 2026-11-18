"""One identified light contact per repeat request, sharing a placement and TCP."""
import time
from pathlib import Path
from bottle_scan.geometry import pose_error
from bottle_scan.state_graph import campaign
from .p3_trial import observe_rows


def collect(io, req, directory, spec, start):
    if (not spec['operators'].get('repeat_block') or req['role']!='exploration'
            or req.get('actions') or req.get('mode')!='planned'):
        raise ValueError('Invalid repeat-only request')
    token=req['stage_token']
    pin_path=Path(req['campaign'])/'repeat_blocks'/(campaign.digest(token)+'.json')
    pose_tol=spec['geometry']['verification_pose_tolerance_mm']
    angle_tol=spec['geometry']['orientation_tolerance_deg']
    def matches(a,b):
        p,r=pose_error(a,b)
        return p<=pose_tol and r<=angle_tol
    pin=campaign.read(pin_path) if pin_path.exists() else None
    reason=None
    if pin is None:
        if req['repeat']!=0:
            reason='first_repeat_reference_unavailable'
        else:
            io.prepare(req)
            ref=io.reference or {}
            if not ref.get('valid'):
                reason='unwitnessed_placement'
            else:
                pin=dict(stage_token=token,placement_reference_id=ref['placement_reference_id'],
                         mass=req['block']['mass_class'],pose=ref['pose'],valid=True)
                campaign.atomic(pin_path,pin)
    else:
        io.reference_path=Path(req['campaign'])/'placement_references'/io.placement/'active.json'
        io.reference=campaign.read(io.reference_path) if io.reference_path.exists() else {}
        if not (pin['valid'] and io.reference.get('valid') and
                io.reference.get('placement_reference_id')==pin['placement_reference_id'] and
                pin['mass']==req['block']['mass_class']):
            reason='repeat_placement_changed'
        elif not io.repeat_continuity():
            reason='repeat_placement_changed'
            io.invalidate_reference()
    wins=[]
    if not reason and not matches(pin['pose'],io.repeat_pose()):
        reason='repeat_tcp_changed'
    contact_attempted=not reason
    if contact_attempted:
        with io.contact(0) as raw:
            if raw is None:
                reason=getattr(io,'contact_failure',None) or 'contact_lost'
            else:
                p=raw['provenance']
                if not (p.get('placement_reference_valid') and p.get('placement_reference_id')==pin['placement_reference_id']
                        and matches(pin['pose'],p['measured_pose'])):
                    reason='repeat_group_mismatch'
                else:
                    raw.update(is_terminal=False,controller_role='repeat',repeat_settings=req['settings'])
                    p.update(repeat_block=token,repeat_ordinal=req['repeat'],repeat_placement_id=pin['placement_reference_id'])
                    wins.append(raw)
    if pin and reason in ('repeat_tcp_changed','repeat_placement_changed','repeat_group_mismatch'):
        campaign.atomic(pin_path,dict(pin,valid=False,reason=reason))
    rows=observe_rows([*wins,*getattr(io,'reference_windows',[])],req['trial_id'],directory,spec)
    result=dict(trial_id=req['trial_id'],status='failed' if reason else 'complete',nodes=rows,
                controller_role='repeat',assessment=None,terminal_outcomes={},consumer_outcomes={},consumer_order=[],
                reference_check=getattr(io,'reference_check',None),attempt_reason=reason,failure_cause=reason,
                repeat_block=token,placement_reference_id=(pin or {}).get('placement_reference_id'),
                cost=dict(actions=0,block_contact_attempts=int(contact_attempted),
                          setup_contact_attempts=getattr(io,'repeat_contacts',{}).get('setup'),wall_seconds=time.monotonic()-start))
    campaign.atomic(directory/'completed_capture.json',result)
    return result
