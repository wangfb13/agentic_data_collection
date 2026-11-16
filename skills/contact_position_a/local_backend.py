"""Opt-in adapter for the installed V4 scheduler. Imports here open no robot connections."""
import hashlib
import json
import os
from pathlib import Path
import time

import joblib
import numpy as np
import yaml

from bottle_scan.contact_policy import ContactPolicy, _skill_conditions
from bottle_scan.e1e2_core import validate
from .contact_reference import probe_protocol, validate_reference
from .local_motion import support_error, validate_station


def read(path):
    return json.loads(Path(path).read_text())


def load_local(config):
    from .ready import HANDOFF
    cfg=config['execution_config']
    validate(cfg)
    settings=config['contact_reference_execution']
    ref=validate_reference(read(settings['reference']))
    station=validate_station(read(settings['station']))
    if (ref['protocol']!=probe_protocol(cfg) or not settings.get('tool_version') or not settings.get('sensor_version')
            or any(settings[k]!=station.get(k) for k in ('tool_version','sensor_version'))
            or station.get('reference_digest')!=ref['reference_digest']
            or station.get('probe_protocol_and_finger_order_verified') is not True):
        raise ValueError('Reference and current station/tool/tactile protocol do not match')
    if (config.get('skill_after')!='handoff' or config.get('skill_recovery')!='enabled'
            or config.get('single_correction_open_handoff') or os.environ.get('CONTACT_EXPERIMENT_B_OFFSET_MM')):
        raise ValueError('Local natural contact requires handoff with no injected offset')
    if (not 0<settings['observation_max_age_s']<=1 or not 0<settings['gain']<=1
            or not 0<=settings['offset_deadband_mm']<=cfg['servo']['max_delta_mm']
            or type(settings['max_steps']) is not int or not 0<settings['max_steps']<=cfg['servo']['max_regrasps']
            or not 0<config['max_skill_s']<=180):
        raise ValueError('Invalid local execution limits')
    hwfile=Path(cfg['hardware_config'])
    if hashlib.sha256(hwfile.read_bytes()).hexdigest()!=config['hardware_config_sha256']:
        raise ValueError('Hardware configuration mismatch')
    hardware=yaml.safe_load(hwfile.read_text())['hardware']
    serials=hardware['public_ros']['sensors']['serials']
    if (station.get('finger_serials') != serials or hardware['sensor_serials'] != serials
            or station.get('tcp_definition') != hardware.get('tcp_definition')
            or station['tcp_definition'] != 'live TF arm_right_7_link -> right_gripper_base_link'):
        raise ValueError('Surveyed TCP frame or finger serial/order differs from hardware profile')
    if (cfg['arm']!='right' or hardware['public_ros']['sensors']['gripper_side']!=cfg['arm']
            or cfg['gripper']['min_normalized']<hardware['public_ros']['raw_normalized_min']):
        raise ValueError('Hardware arm/gripper mismatch')
    request=read(config['state_graph_execution']['request'])
    spec=read(request['spec'])
    if (request['role']!='deployment' or request['mode']!='learned' or request['actions'] or request['settings']
            or spec['active_post_condition']!=HANDOFF or request['post_condition']!=HANDOFF
            or spec.get('local_contact_reference_digest')!=ref['reference_digest']
            or set(spec['observation']['modalities'])!={'tac','force'}):
        raise ValueError('Incompatible local deployment spec')
    bundle=joblib.load(request['model_bundle'])
    if bundle['spec']!=spec or request['model_id']!=bundle['model_id']:raise ValueError('Graph model/spec mismatch')
    heads=bundle['models'][HANDOFF]
    source=settings['action_source']
    if heads['V']['status']!='trained':raise ValueError('Ready model lacks training evidence')
    offset=None
    if source=='offset_servo':
        m=read(settings['offset_model'])
        if (m.get('contact_reference_digest')!=ref['reference_digest'] or m['settings']['include_successful_B']
                or m.get('simulated') is not False or m.get('feature_definition',{}).get('dd_pool')
                or not m.get('trained_offset_range_mm')):
            raise ValueError('Geometric estimator was not trained for this fixed reference')
        offset=ContactPolicy(m,cfg['servo'])
    elif source!='graph_policy' or heads['pi']['status']!='trained':
        raise ValueError('Graph action model unavailable or unknown source')
    report={'target_definition_id':ref['target_definition_id'],'reference_digest':ref['reference_digest'],
            'action_source':source,'graph_model_id':request['model_id'],'offset_model_id':offset.fit['model_id'] if offset else None,
            'station':station['survey_id'],'post_condition':HANDOFF,'hardware_started':False}
    return ref,station,offset,report


class StableProbe:
    _completed_stable_probe=ContactPolicy._completed_stable_probe

    def __init__(self,cfg):
        self.skill_conditions=_skill_conditions(cfg)


def initialize_backend(backend, config):
    ref,station,offset,report=load_local(config)
    backend.cfg=config['execution_config']
    backend.contact_reference=ref
    backend.station=station
    backend.offset_policy=offset
    backend.policy=StableProbe(backend.cfg)
    backend.model={'model_id':report['offset_model_id'] or report['graph_model_id']}
    backend.local_settings=config['contact_reference_execution']
    backend.action_source=report['action_source']
    backend.limits={'max_skill_s':config['max_skill_s']}
    backend.validation=report
    backend.pending_recontact=None
    backend._operator_confirmed=lambda: os.environ.get('CONTACT_LOCAL_STATION_CONFIRMED')=='1'


def move_segment(runner,target):
    backend=runner.backend
    start=runner.action_start_pose
    speed=backend.cfg['motion']['near_speed_mm_s']
    tolerance=backend.cfg['motion']['pose_tolerance_mm']
    def check(pose):
        reason=support_error(pose,backend.station)
        if reason or not np.allclose(pose[:3,:3],start[:3,:3],atol=.01):
            raise RuntimeError(reason or 'orientation_changed_during_local_action')
    sample=backend._sample()
    check(sample['pose']);check(target)
    distance=np.linalg.norm(target[:3,3]-sample['pose'][:3,3])
    deadline=time.monotonic()+distance/speed+backend.cfg['motion']['timeout_margin_s']
    runner.engaged=True
    runner.hw.move(target,speed)
    while True:
        sample=backend._sample();check(sample['pose'])
        if runner.hw.motion_done():
            if np.linalg.norm(sample['pose'][:3,3]-target[:3,3])>tolerance:
                raise RuntimeError('local_translation_endpoint_outside_tolerance')
            return 'reached'
        if time.monotonic()>deadline:raise RuntimeError('local_translation_timeout')
        runner.hw.wait(.01)

def resume_contract(backend,result,observation):
    """ForcePolicy may acquire the gripper only after a verified current Ready decision."""
    from bottle_scan.contact_scheduler import validate_primary_resume_contract
    if result.get('success') is not True:
        return None
    skill=backend.recovery
    final=result['skill_outputs']
    prediction=final['recovery']['decision']['prediction']['post_condition_prediction']
    if (prediction['state']!='ready' or not final['exit']['completed']
            or final['exit']['post_condition']!=skill.spec['active_post_condition']):
        raise ValueError('Missing Ready evidence for the selected consumer')
    sample=backend._sample()
    backend._require_supported(sample)
    force=backend.cfg['force']
    normal=np.asarray(sample['normal'])
    if (not np.all(normal>=backend.hw.zero['contact_thresholds_sdk']) or
            abs(float(normal.mean())-force['target']*force['probe_fraction'])>force['probe_tolerance']):
        raise ValueError('Contact changed before ForcePolicy handoff')
    invoked=any(row['skill_outputs']['invocation']['invoke'] is True for row in result['observations'])
    applied=any((row.get('receipt') or {}).get('applied') is True and abs(row['receipt']['action_mm'])>1e-9
                for row in result['corrections'])
    if invoked!=applied:raise ValueError('Recovery selection and measured receipt disagree')
    contract={'schema':'a_skill_primary_resume_v1','skill_id':'contact_position_a',
        'model_id':skill.model_id,'model_sha256':skill.model_sha256,
        'target_definition_id':backend.contact_reference['target_definition_id'],
        'post_condition':final['exit']['post_condition'],'phase':'before_squeeze','contact_state':'bilateral_stable_in_target',
        'resume_allowed':True,'target_contact_verified':True,'outcome':'restored' if applied else 'already_normal',
        'invoked':invoked,'observation_timestamp_ns':observation['timestamp_ns'],
        'latest_observation':{k:observation[k] for k in ('descriptors','bilateral','force_stable','stable_duration_s')},
        'measured_state':{'tcp_pose_mm':np.asarray(sample['pose']).tolist(),'timestamp_ns':int(sample['timestamp_ns']),
                          'normal_force_sdk':normal.tolist(),'gripper_opening_normalized':float(sample['width'])},
        'next_stage':'primary_policy_fresh_inference','next_controller':'existing_forcepolicy',
        'primary_resume_requires':'existing_forcepolicy_ready'}
    return validate_primary_resume_contract(contract,model_id=skill.model_id,model_sha256=skill.model_sha256,
        target_definition_id=contract['target_definition_id'],now_ns=int(backend.hw.now()),
        max_age_s=skill.config['max_observation_age_s'],min_stable_duration_s=force['probe_stable_s'])
