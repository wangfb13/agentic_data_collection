"""V4 evidence adapter: reuse current camera receiver and append actual outcomes.

This module never sends robot commands. Existing V4 motion/probe control remains
responsible for execution; recording is enabled by an explicit v3 configuration.
"""
from pathlib import Path
from collections import defaultdict
import json
import hashlib
import os
import time
import numpy as np


def consumer_identity():
    path=os.environ.get('RIGHT_FORCE_POLICY_PATH');version=None
    if path and Path(path).is_file():version=hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return {'controller':'existing_forcepolicy','version':version,'checkpoint_path':path,
            'params':{'closure_floor':float(os.environ.get('RIGHT_FORCE_POSITION_MIN','.330')),
                      'target_force_mode':'predicted_by_consumer','target_force_sdk':None,
                      'variant':os.environ.get('RIGHT_FORCE_POLICY_VARIANT')}}


def capture_window(backend,probe,sample,folder,index,previous=None,receipt=None,observation=None):
    folder=Path(folder);camera_result={'available':False};config=backend.config.get('state_graph_recording',{})
    cameras=getattr(backend.robot,'cameras',{}) or {};preferred=config.get('wrist_camera_key','right_wrist')
    key=preferred if preferred in cameras else next((k for k in cameras if 'right' in k.lower() and 'wrist' in k.lower()),None)
    if key is not None:
        camera=cameras[key]
        try:
            # The existing receiver already owns the socket. No extra PULL
            # connection and no read loop is started by this adapter.
            rgb=camera.read_latest(max_age_ms=500)
            from PIL import Image
            mode=str(getattr(getattr(camera,'config',None),'color_mode','rgb')).lower()
            if mode.endswith('bgr'):rgb=rgb[:,:,::-1]
            image=folder/f'graph_window_{index:03d}_wrist.png';Image.fromarray(np.asarray(rgb,dtype=np.uint8)).save(image)
            camera_result={'available':True,'path':str(image),'camera_key':key,'frame_id':getattr(camera,'latest_frame_id',None),
                           'capture_monotonic_s':getattr(camera,'latest_timestamp',None),'recorded_monotonic_ns':time.monotonic_ns()}
        except (RuntimeError,TimeoutError,AttributeError,ValueError) as exc:camera_result={'available':False,'reason':str(exc),'camera_key':key}
    p=probe or {};node_id=folder.name+':'+str(index);pose=np.asarray(sample['pose']);T0=np.asarray(backend.T0)
    # Existing descriptors expose `valid`, not a `contact` flag. Reuse the
    # runtime's measured bilateral/stability decision when supplied.
    if observation is not None:
        contact_valid=bool(observation.get('bilateral') and observation.get('force_stable'))
    else:
        contact_valid=bool(p.get('valid') and len(p.get('descriptors',[]))==2
                           and all(d.get('valid') for d in p['descriptors'])
                           and np.all(np.asarray(sample['normal'])>=backend.hw.zero['contact_thresholds_sdk']))
    dd_reference=folder/'graph_dd_reference.npz'
    if not dd_reference.exists():np.savez_compressed(dd_reference,left=backend.hw.dd_reference[0],right=backend.hw.dd_reference[1])
    return {'schema':'contact_graph_window_v3','event':'state_graph_window','node_id':node_id,'session':str(backend.output_dir),'trial_id':folder.name,'group_id':str(folder),
            'window_index':index,'phase':'B_pre' if index==0 else 'B_post','t_ns':int(p.get('timestamp_ns',sample['timestamp_ns'])),
            'recorded_unix_ns':time.time_ns(),'trial_started_unix_ns':int(folder.name.rsplit('_',1)[-1]) if folder.name.rsplit('_',1)[-1].isdigit() else None,
            'preconditions':{'bilateral_contact':contact_valid},
            'probe':p,'wrench':np.asarray(sample['wrench_zeroed']).reshape(-1).tolist(),'gripper_opening':float(sample['width']),
            'measured_tcp_mm':pose.tolist(),'T0_mm':T0.tolist(),'a_axis_base':np.asarray(backend.bounds.a).tolist(),
            'plan_coords':{'a_offset_mm':float((pose[:3,3]-T0[:3,3])@backend.bounds.a)},
            'u_executed':None if not receipt else [receipt['action_mm']],'u_receipt':receipt,'previous_node_id':previous,
            'raw_evidence':{'streams':str(folder/'streams.h5'),'dd_reference':str(dd_reference),'wrist':camera_result},
            'attrs':{'mass_class':os.environ.get('CONTACT_GRAPH_MASS_CLASS'),'approach_pose':os.environ.get('CONTACT_GRAPH_APPROACH_POSE'),
                     'sensor_version':os.environ.get('CONTACT_GRAPH_SENSOR_VERSION'),'time_period':os.environ.get('CONTACT_GRAPH_TIME_PERIOD'),
                     'planned_a_offset_mm':float(os.environ['CONTACT_EXPERIMENT_B_OFFSET_MM']) if os.environ.get('CONTACT_EXPERIMENT_B_OFFSET_MM') else None,'closure_floor':backend.cfg['gripper']['min_normalized'],'gripper_opening':float(sample['width'])},
            'consumer':getattr(backend,'_graph_consumer',None)}


def lifecycle_event(event,window,consumer):
    if not window:return None
    if event.get('event')=='fault' and event.get('reason','').startswith('ForcePolicy did not stabilize during handoff'):
        return {'schema':'contact_graph_outcome_v3','event':'state_graph_consumer_result','node_id':window['node_id'],
                'group_id':window['group_id'],'is_terminal':True,'consumer_outcome':{**consumer,'L1':False,'L2':None,'L3':None,
                'evidence':[{'event':'fault','reason':event['reason'],'meaning':'Consumer force readiness timed out; lift was not attempted.'}]},'recorded_unix_ns':time.time_ns()}
    if event.get('event')=='main_resumed' and event.get('force_readiness_verified'):
        return {'schema':'contact_graph_outcome_v3','event':'state_graph_consumer_result','node_id':window['node_id'],
                'group_id':window['group_id'],'is_terminal':True,'consumer_outcome':{**consumer,'L1':True,'L2':None,'L3':None,
                'evidence':[{'event':'main_resumed','meaning':'Force readiness only; lift/hold and task completion not established.'}]},'recorded_unix_ns':time.time_ns()}
    if event.get('event')=='skill_completed':
        return {'schema':'contact_graph_handoff_v3','event':'state_graph_handoff','node_id':window['node_id'],'group_id':window['group_id'],
                'is_terminal':True,'consumer':consumer,'skill_result':event.get('result',{}).get('exit_result'),'recorded_unix_ns':time.time_ns()}
    return None


def append_reviewed_outcome(events_path,*,L2,L3=None,evidence):
    path=Path(events_path);events=[json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    first=next((e for e in reversed(events) if e.get('event')=='state_graph_consumer_result'),None)
    if first is None:raise ValueError('No recorded consumer readiness result to associate')
    if type(L2) is not bool or L3 is not None and type(L3) is not bool or not evidence:raise ValueError('Explicit outcome and evidence required')
    result={**first,'consumer_outcome':{**first['consumer_outcome'],'L2':L2,'L3':L3,'evidence':first['consumer_outcome']['evidence']+[{'review_evidence':evidence}]},'recorded_unix_ns':time.time_ns()}
    with path.open('a') as f:f.write(json.dumps(result,ensure_ascii=False)+'\n')
    return result


def trials_from_events(events,spec):
    groups=defaultdict(list);outcomes={};handoffs=set()
    for e in events:
        if e.get('event')=='state_graph_window':groups[e['group_id']].append(e)
        elif e.get('event')=='state_graph_consumer_result':outcomes[e['node_id']]=e['consumer_outcome'];handoffs.add(e['node_id'])
        elif e.get('event')=='state_graph_handoff':handoffs.add(e['node_id'])
    vision=None
    for group,items in groups.items():
        windows=[]
        for e in sorted(items,key=lambda e:e['window_index']):
            descriptors=e['probe'].get('descriptors',[])
            # Match historical feature extraction: median geometry of the five
            # recorded independent source frames, using the saved unloaded image.
            contacts=None;encoded_references={}
            if e.get('raw_evidence',{}).get('streams') and e['probe'].get('window'):
                from .legacy_v2.contact_geometry import window_contact_geometry
                import h5py
                with np.load(e['raw_evidence']['dd_reference']) as z:
                    contacts=[]
                    for side,window in zip(('left','right'),e['probe']['window']):
                        reference={'h5_path':e['raw_evidence']['streams'],'stream':'dd_'+side,'frame_ns':window['frame_ns'],'frame_fids':window['frame_fids']}
                        # read_dd_frames accepts indices, so recover exact source timestamps.
                        with h5py.File(reference['h5_path'],'r') as h:
                            from .legacy_v2.extract_nodes import _stream,_window_indices,_median_stream
                            stream=_stream(h,'dd_'+side)
                            indices=_window_indices(stream,window['frame_ns'],e['probe']['stable_start_ns'],e['probe']['timestamp_ns'])
                            frames=np.asarray(stream['data'][indices])
                            encoded=_stream(h,'dd_encoded_'+side)
                            if encoded is not None:
                                encoded_indices=_window_indices(encoded,window['frame_ns'],e['probe']['stable_start_ns'],e['probe']['timestamp_ns'])
                                encoded_references[side]={'h5_path':reference['h5_path'],'dataset':'streams/dd_encoded_'+side+'/data',
                                                          'indices':encoded_indices,'times_ns':[int(encoded['times'][i]) for i in encoded_indices]}
                        contacts.append(window_contact_geometry(frames,z[side],threshold=spec.get('contact_geometry',{}).get('depth_threshold',.022)))
            wrench=e['wrench']
            if contacts is not None:
                with h5py.File(e['raw_evidence']['streams'],'r') as h:
                    vectors=[_median_stream(_stream(h,'wrench_'+side),e['probe']['stable_start_ns'],e['probe']['timestamp_ns'],(6,))[0] for side in ('left','right')]
                    if all(v is not None for v in vectors):wrench=np.concatenate(vectors).tolist()
            if contacts is None:contacts=[{'u_px':d['centroid_px'][0],'v_px':d['centroid_px'][1],'area_px':d['area_px']} for d in descriptors]
            image=e['raw_evidence']['wrist'];zvis=e.get('visual_features')
            if zvis is None and image.get('available'):
                from .legacy_v2.features import FrozenVision,read_wrist
                if vision is None:vision=FrozenVision('cpu',spec['data']['visual_model_directory'])
                zvis=vision.encode(read_wrist({'path':image['path']}))
            row={k:e.get(k) for k in ('node_id','session','trial_id','group_id','window_index','phase','t_ns','u_executed','u_receipt','previous_node_id','plan_coords','attrs','preconditions')}
            row.update(is_terminal=e['node_id'] in handoffs,consumer_outcome=outcomes.get(e['node_id']),contacts=contacts,wrench=wrench,visual=zvis,tactile_frames=encoded_references,
                repeat_settings={'a_offset_mm':e['attrs'].get('planned_a_offset_mm'),'closure_floor':e['attrs'].get('closure_floor'),'approach_pose':e['attrs'].get('approach_pose')},
                is_reference=e['attrs'].get('planned_a_offset_mm')==0,
                provenance={**e['raw_evidence'],'trial_started_unix_ns':e.get('trial_started_unix_ns'),'source_type':'scheduler_episode'},
                extractor_versions={k:v['version'] for k,v in spec['observation']['modalities'].items()})
            windows.append(row)
        yield {'session':items[0]['session'],'trial_id':items[0]['trial_id'],'windows':windows}
