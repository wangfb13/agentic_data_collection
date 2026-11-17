"""Controlled port (distinct from natural deployment). No offset injection into the natural scheduler."""
import hashlib
import importlib.util
import json
import time
from pathlib import Path
from types import SimpleNamespace
import joblib
import yaml
from bottle_scan.max_expert_demonstration_finder.evidence import sha,require
from bottle_scan.state_graph.skill import SymbolicRecoverySkill
from bottle_scan.contact_policy import ContactPolicy
from .contact_reference import validate_reference,probe_protocol
from .local_decision import decide_local
from .local_motion import validate_station,recover_once
from .local_backend import read

def filehash(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()
def artifact(p):
    p=Path(p).resolve();return {'path':str(p),'sha256':filehash(p)}
def manifest(cfg):
    a=Path(cfg['assets']);paths={'bundle':a/'graph/models/bundle.joblib','spec':a/'spec.json',
        'reference':cfg['reference'],'scheduler':cfg['scheduler']}
    if cfg['action_source']=='offset_servo':paths['offset']=cfg.get('offset_model',a/'offset.json')
    if (a/'training_rows.json').is_file():paths['training_rows']=a/'training_rows.json'
    if cfg.get('station'):paths['station']=cfg['station']
    if cfg.get('driver'):paths['driver']=cfg['driver']
    files={k:artifact(p) for k,p in paths.items()}
    ref=validate_reference(read(paths['reference']));spec=read(paths['spec']);b=joblib.load(paths['bundle'])
    require(b['spec']==spec and spec.get('local_contact_reference_digest')==ref['reference_digest'],'bundle/spec/reference mismatch')
    c=next(v for v in spec['post_conditions'] if v['id']==spec['active_post_condition'])
    ctl=read(cfg['scheduler']);hw=ctl['execution_config'];v=spec['perturbation_variables']['a_offset_mm'];mot=hw['motion'];servo=hw['servo']
    require(probe_protocol(hw)==ref['protocol'],'protocol mismatch')
    files['hardware']=artifact(hw['hardware_config']);require(files['hardware']['sha256']==ctl['hardware_config_sha256'],'invalid hardware profile')
    hw_profile=yaml.safe_load(Path(files['hardware']['path']).read_text())['hardware']
    execution=read(Path(__file__).resolve().parents[2]/'config.state_graph.campaign_v4.json')['execution']
    runtime=Path(execution['entry']).parent/'runtime/bottle_scan'
    for name in ('contact_skill_backend','contact_recovery_backend','contact_scheduler'):
        path=runtime/(name+'.py')
        if path.is_file():files[name]=artifact(path)
    if c['consumer']['controller']=='existing_forcepolicy':
        ckpt=cfg.get('consumer_checkpoint') or execution['environment'].get('RIGHT_FORCE_POLICY_PATH')
        if ckpt and Path(ckpt).is_file():files['consumer_checkpoint']=artifact(ckpt)
    station=validate_station(read(paths['station'])) if 'station' in paths else {}
    if station:require(not station.get('offline_fixture') and station['reference_digest']==ref['reference_digest'] and station.get('probe_protocol_and_finger_order_verified') is True,'physical station not verified')
    if station:
        require(station.get('finger_serials')==hw_profile['sensor_serials']==hw_profile['public_ros']['sensors']['serials']
                and station.get('tcp_definition')==hw_profile['tcp_definition'] and hw['arm']=='right','surveyed hand/serials or TCP differ')
        require(all(k in files for k in ('contact_skill_backend','contact_recovery_backend','contact_scheduler')),'unbound V4 runtime')
        if c['consumer']['controller']=='existing_forcepolicy':
            require(files.get('consumer_checkpoint',{}).get('sha256')==c['consumer']['version'],'consumer version differs from the frozen post-condition')
    source=cfg['action_source'];heads=b['models'][c['id']];good=heads['V']['status']=='trained' and (heads['pi']['status']=='trained' if source=='graph_policy' else read(paths['offset']).get('usable_for_B') is True)
    if source=='offset_servo':
        om=read(paths['offset']);require(om.get('contact_reference_digest')==ref['reference_digest'] and not om['settings']['include_successful_B'],'offset artifact incompatible')
        if good:ContactPolicy(om,servo)
    params={'gain':float(cfg.get('gain',1.)),'offset_deadband_mm':servo['tol_mm'],'observation_max_age_s':.5,'max_steps':cfg['limits']['max_steps']}
    require(0<params['gain']<=1,'invalid gain')
    identity=sha([files['bundle']['sha256'],files.get('offset'),source,params]);codes={}
    root=Path(__file__).resolve().parents[2]
    for f in [*sorted((root/'bottle_scan/max_expert_demonstration_finder').glob('*.py')),
              *[root/p for p in ('skills/contact_position_a/local_decision.py','skills/contact_position_a/local_motion.py',
              'skills/contact_position_a/local_backend.py','skills/contact_position_a/ready.py','skills/contact_position_a/frontier.py','skills/contact_position_a/frontier_recorder.py',
              'bottle_scan/state_graph/skill.py','bottle_scan/state_graph/noise.py','bottle_scan/contact_policy.py')]]:
        codes[str(f)]=filehash(f)
    return dict(environment='physical',expert_exchange=str(Path(cfg['expert_exchange']).resolve()) if cfg.get('expert_exchange') else None,model_id=identity,graph_model_id=b['model_id'],target_definition_id=ref['target_definition_id'],
        reference_digest=ref['reference_digest'],axis_tcp=ref['protocol']['axes_tcp']['a'],protocol=ref['protocol'],artifacts=files,
        code_hashes=codes,graph_version=b['graph_version'],noise_digest=sha(b['noise']),support_digest=sha(b['support']),
        controller='fixed_reference_local_a_v1',object=spec['task']['object_category'],action_source=source,policy_executable=good,
        reject_rules={'runtime':spec['runtime'],'parameters':params,'servo':servo},station=station or 'not_surveyed',pose_tolerance_mm=mot['pose_tolerance_mm'],
        sensor_version=station.get('sensor_version','unverified'),tool_version=station.get('tool_version','unverified'),
        execution_caps={'range':v['envelope'],'prepare_mm':mot['max_grasp_offset_mm'],'expansion':v['max_expansion'],
            'step_mm':min(servo['max_delta_mm'],spec['safety_envelope']['max_action'][0]),'total_mm':mot['max_grasp_offset_mm'],
            'max_steps':min(ctl['max_regrasps'],servo['max_regrasps'],spec['runtime']['max_steps']),'seconds':ctl['max_skill_s']},
        condition=cfg['condition'],post_condition=c,limits=cfg['limits'],statistics=cfg['statistics'],expert=cfg.get('expert'))

def verify_files(m):
    for f in m['artifacts'].values():require(filehash(f['path'])==f['sha256'],'frozen artifact changed: '+f['path'])
    for p,h in m['code_hashes'].items():require(filehash(p)==h,'frozen implementation changed: '+p)

class FrozenPolicy:
    def __init__(self,m,clock_ns):
        verify_files(m);files=m['artifacts'];b=joblib.load(files['bundle']['path']);self.identity=m['model_id']
        self.graph=SymbolicRecoverySkill(b);cfg=read(files['scheduler']['path'])['execution_config']
        self.backend=SimpleNamespace(action_source=m['action_source'],cfg=cfg,local_settings=m['reject_rules']['parameters'],
            contact_reference=read(files['reference']['path']),hw=SimpleNamespace(now=clock_ns),
            offset_policy=ContactPolicy(read(files['offset']['path']),cfg['servo']) if 'offset' in files else None)
    def decide(self,obs,step):
        raw={**obs['tactile'],'observation_id':obs['id'],'timestamp_ns':obs['timestamp_ns']}
        state,reason,cmd,report=decide_local(self.backend,raw,obs['row'],self.graph,step)
        return dict(state={'ready':'Ready','recoverable':'NeedAction','Unknown':'Unknown'}[state],reason=reason,
            command=cmd,offset_a_mm=(report.get('offset_estimate') or {}).get('offset_a_mm'),details=report)

class StationPort:
    """A site driver owns its exclusive V4 runner and evidence capture. No planned policy fallback."""
    environment='physical'
    def __init__(self,driver,m):
        self.driver=driver;self.m=m;self.expert_identity=driver.expert_identity
    def verify(self,m):
        verify_files(m)
        require(isinstance(m['station'],dict) and not m['station'].get('offline_fixture'),'surveyed station required')
        require(self.driver.identity()=={'survey_id':m['station']['survey_id'],'sensor_version':m['sensor_version'],
            'tool_version':m['tool_version'],'consumer':m['post_condition']['consumer'],'reference_digest':m['reference_digest'],
            'condition':m['condition']},'station, object, protocol or consumer mismatch')
    def expert_decision(self,obs,step,expert):return self.driver.expert_decision(obs,step,expert)
    def perform(self,req):
        op=req['operation']
        if op!='correct_retouch':
            return getattr(self.driver,{'prepare':'prepare_measured','touch':'touch_once','post':'test_consumer','reset':'reset'}[op])(req)
        # Reuse the checked local current-TCP implementation and measured receipt.
        # Site bridge must set a watchdog across release/move/recontact, own the lease,
        # and durably journal per-command evidence before returning.
        backend,runner=self.driver.bind_runner(req)
        require(backend.action_source==req['command']['action_source'] and backend.station==self.m['station'] and
                backend.contact_reference['reference_digest']==self.m['reference_digest'],'bound runner scope mismatch')
        row={}
        with self.driver.deadline(req['remaining_seconds']):
            _,status,probe,row=recover_once(backend,runner,req['command'],row)
        return self.driver.record_retouch(req,status,probe,row)

def load_station(m):
    verify_files(m);require('driver' in m['artifacts'] and isinstance(m['station'],dict),'no verified station driver/survey')
    s=importlib.util.spec_from_file_location('frontier_site_driver',m['artifacts']['driver']['path'])
    mod=importlib.util.module_from_spec(s);s.loader.exec_module(mod)
    driver=mod.open_controlled_port(m);return StationPort(driver,m)


def controlled_window(backend,runner,status,probe,folder,index,spec,*,measurement=None,observation=None,previous=None,receipt=None):
    """Driver helper: keep the literal contact attempt before graph extraction.
    Observation and receipt use the hardware monotonic clock, never wall time.
    """
    import numpy as np
    from .runtime import live_window
    raw=observation if observation is not None else backend._observation(status,probe)
    sample=backend.hw.snapshot()
    stamp=int((probe or {}).get('timestamp_ns') or raw['timestamp_ns'])
    thresholds=np.asarray(backend.hw.zero['contact_thresholds_sdk'])
    normals=np.asarray(sample['normal'])
    count=int((normals>=thresholds).sum())
    contact=('valid_bilateral' if raw.get('bilateral') and raw.get('force_stable') and raw.get('safety_ok') and (probe or {}).get('valid')
             else 'unstable_bilateral' if raw.get('bilateral') or count==2 else 'unilateral' if count==1 else 'no_contact')
    item=dict(id=Path(folder).name+':'+str(index),timestamp_ns=stamp,tcp_mm=np.asarray(sample['pose']).tolist(),
              contact_state=contact,tactile={k:raw.get(k) for k in ('bilateral','force_stable','safety_ok','descriptors')},
              measurement=measurement,edge=any(d.get('touches_edge') for d in raw.get('descriptors') or []),node_id=None)
    if contact=='valid_bilateral':
        window,row=live_window(backend,probe,raw,Path(folder),index,previous,receipt,spec)
        item.update(row=row,node_id=row['node_id'],id=row['node_id'],tcp_mm=window['measured_tcp_mm'])
        backend.current_observation_id=item['id'];backend.current_observation_ns=stamp
        backend.current_observation_pose=np.asarray(item['tcp_mm']).copy()
    return item

class BoundExperiment:
    """Defer read-only preflight/driver construction until an attempt exists."""
    environment='physical'
    def __init__(self,m,role):
        self.m=m;self.role=role;self.identity=m['model_id'];self.port=None;self.policy=None
    def verify(self,m):
        if self.role=='expert_recovery' and m.get('expert_exchange'):
            from .frontier_recorder import WitnessRecorder
            self.port=WitnessRecorder(m)
        else:self.port=load_station(m)
        self.port.verify(m);self.expert_identity=self.port.expert_identity
        if self.role=='policy_frontier':self.policy=FrozenPolicy(m,self.port.driver.now_ns)
    def decide(self,obs,step):return self.policy.decide(obs,step)
    def expert_decision(self,*args):return self.port.expert_decision(*args)
    def perform(self,req):return self.port.perform(req)
    def close(self):
        if isinstance(self.port,StationPort):self.port.driver.stop_and_close()
