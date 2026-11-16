"""Offline build of the fixed-reference offset estimator and tactile-only Ready/graph models."""
import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path

from bottle_scan.contact_policy import train_contact_policy
from bottle_scan.state_graph.campaign import all_rows, read, atomic
from bottle_scan.state_graph.cli import build
from bottle_scan.state_graph.common import independent
from bottle_scan.state_graph.skill_interface import repeat_key, model_observation
from .contact_reference import validate_reference


def rebuild(campaign, ref, out):
    spec=read(campaign/'spec.json')
    spec['observation']['modalities']={k:v for k,v in spec['observation']['modalities'].items() if k in ('tac','force')}
    if set(spec['observation']['modalities'])!={'tac','force'}:raise ValueError('Need historical tactile and wrench features')
    spec.pop('tactile_encoder',None)
    spec['local_contact_reference_digest']=ref['reference_digest']
    spec['task']['action_space']['frame']='tcp_at_action_start'
    source=Path(read(campaign/'config.json').get('seed_campaign',campaign)).resolve()
    rows=[]
    for raw in all_rows(campaign,read(campaign/'state.json')):
        if independent(raw) or raw.get('evaluation_only'):continue
        r=deepcopy(raw)
        for field in ('obs','extractor_versions','feature_evidence'):
            r[field]={k:v for k,v in r.get(field,{}).items() if k in ('tac','force')}
        for k,v in spec['observation']['modalities'].items():
            if r['extractor_versions'].get(k)!=v['version']:raise ValueError('Feature version mismatch: '+k)
        model_observation(r,spec['observation']['modalities'])
        # Preserve group separation. Legacy comparison axes belonged to a fixed-orientation
        # experiment; these actions are demonstrations only, never geometric offset labels.
        r['is_reference']=False
        r['provenance']['historical_repeat_context']=r['provenance'].get('repeat_context')
        # Only this campaign's recorded fixed-mark batches declare matching setups.
        placement=r['session']
        known_fixed_campaign=source==Path('outputs/state_graph/campaign_white_a_v4').resolve()
        if known_fixed_campaign and Path(r['session']).is_relative_to(source/'trials') and r['trial_id']:
            trial_dir=Path(r['session']).relative_to(source/'trials').parts[0]
            placement=str(source)+'/'+trial_dir.rsplit('_',1)[0]
        if r['provenance'].get('coordinate_schema')=='fixed_contact_a_v1':
            if r['provenance'].get('reference_digest')!=ref['reference_digest']:
                raise ValueError('Recorded window belongs to a different fixed contact reference')
        else:
            r['provenance']['repeat_context']=[ref['reference_digest'],placement,r['attrs'].get('sensor_version'),
                                               r['attrs'].get('time_period'),r['attrs'].get('approach_pose')]
        r['repeat_key']=repeat_key(r,spec)
        rows.append(r)
    atomic(out/'spec.json',spec)
    built=build(rows,spec,out/'graph',validate=False)
    cid=spec['active_post_condition']
    states=Counter(v['status'] for v in built['graph']['reachability'][cid].values())
    outcomes=Counter(str(r['post'][cid]['success']) for r in rows if r['is_terminal'] and r['post'][cid]['evaluated'])
    return {**built['training'],'support_state_counts':dict(states),'observed_terminal_outcomes':dict(outcomes)}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference',default='skills/contact_position_a/white_reference.json')
    parser.add_argument('--manifest',default='skills/contact_position_a/white_A_sources.json')
    parser.add_argument('--campaign',default='outputs/state_graph/campaign_white_a_v4_forcepolicy')
    parser.add_argument('--out',required=True,type=Path)
    a=parser.parse_args(argv);out=a.out;out.mkdir(parents=True,exist_ok=True)
    ref=validate_reference(read(a.reference));manifest=read(a.manifest)
    model=train_contact_policy([v['path'] for v in manifest['measured_A_sessions'].values()],
        {'contact_policy':{'include_successful_B':False}},contact_reference=ref,label_manifest=manifest)
    atomic(out/'offset.json',model)
    report={'reference':ref['target_definition_id'],'offset_usable':model['usable_for_B'],
            'offset_validation':model.get('validation'),'offset_blockers':model['blocking_reasons'],
            'training_count':model.get('training_count'),'placement_generalization_verified':False,
            'graph':rebuild(Path(a.campaign),ref,out),'robot_started':False}
    atomic(out/'offline_report.json',report)
    print(json.dumps(report,ensure_ascii=False))


if __name__=='__main__':main()
