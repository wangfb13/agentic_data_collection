"""Prepare the natural-contact V4 entry. By default only run the launcher's offline check."""
import argparse
import json
import os
from pathlib import Path
import subprocess

import joblib
import yaml
from bottle_scan.state_graph.campaign import atomic,read
from .contact_reference import validate_reference
from .local_motion import validate_station

ROOT=Path(__file__).resolve().parents[2]


def prepare(assets,station_path,out,source,placement=None,repeat_setup=None):
    assets=Path(assets).resolve();station_path=Path(station_path).resolve();out=Path(out).resolve()
    station=validate_station(read(station_path))
    cfg=read(ROOT/'config.state_graph.campaign_v4.json')['execution']
    config=read(cfg['scheduler_config'])
    for key in ('reference_session_meta','model_path','reference_session_meta_sha256','reference_session_original_path',
                'source_policy_id','source_campaign','support_confirmation','recovery_skill_bundle','recovery_skill_mode'):
        config.pop(key,None)
    ref_path=ROOT/'skills/contact_position_a/white_reference.json'
    validate_reference(read(ref_path))
    spec=read(assets/'spec.json')
    bundle=joblib.load(assets/'graph/models/bundle.joblib')
    request={'schema':'graph_trial_request_v4','trial_id':'natural_contact','role':'deployment','mode':'learned',
             'spec':str(assets/'spec.json'),'model_id':bundle['model_id'],'model_bundle':str(assets/'graph/models/bundle.joblib'),
             'post_condition':spec['active_post_condition'],'settings':{},'block':{},'actions':[]}
    settings={'reference':str(ref_path),'station':str(station_path),'sensor_version':station['sensor_version'],
              'tool_version':station['tool_version'],'placement_approach_group':placement,'repeat_setup_id':repeat_setup,
              'action_source':source,'max_steps':config['max_regrasps'],'gain':config['execution_config']['servo'].get('gain',1.),
              'offset_deadband_mm':config['execution_config']['servo']['tol_mm'],'observation_max_age_s':.5}
    if source=='offset_servo':settings['offset_model']=str(assets/'offset.json')
    config.update(contact_reference_execution=settings,state_graph_execution={'repository':str(ROOT),'request':str(out/'request.json')},
                  single_correction_open_handoff=False,skill_after='handoff',skill_recovery='enabled')
    config['execution_config']['hardware_config']=str((ROOT/config['execution_config']['hardware_config']).resolve())
    out.mkdir(parents=True,exist_ok=True)
    atomic(out/'request.json',request);atomic(out/'scheduler_config.json',config)
    env=dict(os.environ);env.pop('CONTACT_EXPERIMENT_B_OFFSET_MM',None);env.pop('CONTACT_TABLETOP_CONFIRMED',None)
    floor=next(c for c in spec['post_conditions'] if c['id']==spec['active_post_condition'])['consumer']['params']['closure_floor']
    env.update(cfg['environment'],RIGHT_GRIPPER_CLOSE_LIMIT=str(floor),RIGHT_FORCE_POSITION_MIN=str(floor),
               CONTACT_SCHEDULER_CONFIG=str(out/'scheduler_config.json'),
               CONTACT_SCHEDULER_LOG_DIR=str(out/'episodes'),CONTACT_SKILL_AFTER='handoff',CONTACT_SKILL_RECOVERY='enabled')
    return cfg['entry'],env


def offline_station(directory):
    """Only a test volume. The real scheduler also rejects it before hardware prepare."""
    config=read(ROOT/'config.state_graph.v4_scheduler.json')
    hw=yaml.safe_load((ROOT/config['execution_config']['hardware_config']).read_text())['hardware']
    station=read(ROOT/'skills/contact_position_a/station.template.json')
    station.update(offline_fixture=True,survey_id='OFFLINE_ONLY',sensor_version='fixture',tool_version='fixture',
        finger_serials=hw['sensor_serials'],tcp_definition=hw['tcp_definition'],
        probe_protocol_and_finger_order_verified=True,full_open_tool_clearance_verified=True,
        tcp_min_base_mm=[-50]*3,tcp_max_base_mm=[50]*3,allowed_rotations=[[[1,0,0],[0,1,0],[0,0,1]]])
    path=Path(directory)/'offline_station.json'
    atomic(path,station)
    return path


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--action-source',choices=['offset_servo','graph_policy'],required=True)
    p.add_argument('--assets',default=ROOT/'outputs/local_contact_a_v1')
    station=p.add_mutually_exclusive_group(required=True)
    station.add_argument('--station');station.add_argument('--offline-fixture',action='store_true')
    p.add_argument('--out',required=True)
    p.add_argument('--placement-group');p.add_argument('--repeat-setup');p.add_argument('--real',action='store_true')
    a=p.parse_args(argv)
    if a.offline_fixture and a.real:p.error('--offline-fixture is only for task-check')
    if a.offline_fixture:a.station=offline_station(a.out)
    entry,env=prepare(a.assets,a.station,a.out,a.action_source,a.placement_group,a.repeat_setup)
    result=subprocess.run(['bash',entry,'task-check'],env=env,check=False)
    if result.returncode or not a.real:return result.returncode
    if read(a.station).get('offline_fixture'):raise ValueError('Test fixture cannot be used for real operation')
    if input('确认本机已核验的工作区、瓶子有支撑、夹爪全开空载、遥操已停止。输入 MOVE：').strip()!='MOVE':return 1
    env['CONTACT_LOCAL_STATION_CONFIRMED']='1'
    return subprocess.run(['bash',entry,'task-run'],env=env,check=False).returncode


if __name__=='__main__':raise SystemExit(main())
