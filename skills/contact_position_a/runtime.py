"""Graph model adapter for the existing V4 executor, ownership and ForcePolicy handoff.

Installation only selects this adapter for an explicit state_graph_execution config.
All arm motion and force regulation remain in the existing V4 _recover_once loop.
"""
from collections import defaultdict, deque
from copy import deepcopy
from pathlib import Path
import json
import time
import types
import numpy as np
import joblib
from bottle_scan.state_graph.common import digest
from bottle_scan.state_graph.skill import SymbolicRecoverySkill
from bottle_scan.state_graph.skill_interface import load_spec, load_callable, model_observation, normalize_action, task_id
from bottle_scan.state_graph.extract_nodes import extract_nodes
from .scheduler import capture_window, consumer_identity


def live_window(backend,probe,observation,folder,index,previous,receipt,spec):
    """Same five source frames and per-frame median geometry as offline extraction."""
    event=capture_window(backend,probe,backend.hw.snapshot(),folder,index,previous,receipt,observation)
    from .legacy_v2.contact_geometry import window_contact_geometry
    contacts=[];arrays={};wrenches=[]
    with backend.hw.lock:
        for side,ref,window in zip(('left','right'),backend.hw.dd_reference,probe['window']):
            frames={e.monotonic_ns:e.data for e in backend.hw.dd[side]}
            contacts.append(window_contact_geometry([frames[t] for t in window['frame_ns']],ref,
                            threshold=spec.get('contact_geometry',{}).get('depth_threshold',.022)))
            encoded={e.monotonic_ns:e.data for e in backend._graph_buffers['dd_encoded_'+side]}
            if 'tactile_encoder' in spec:arrays[side]=np.stack([encoded[t] for t in window['frame_ns']])
            raw=[e.data for e in backend._graph_buffers['wrench_raw_'+side]
                 if probe['stable_start_ns']<=e.monotonic_ns<=probe['timestamp_ns']]
            if not raw:raise ValueError('No independent wrench frames in the probe window')
            i=0 if side=='left' else 1
            wrenches.extend((np.median(raw,axis=0)-np.asarray(backend.hw.zero['offset_raw'][i])).tolist())
    image=event['raw_evidence']['wrist']
    if not image.get('available'):raise ValueError('Current right wrist image unavailable: '+str(image))
    from .legacy_v2.features import FrozenVision,read_wrist
    if not hasattr(backend,'_graph_vision'):backend._graph_vision=FrozenVision('cpu',spec['data']['visual_model_directory'])
    visual=backend._graph_vision.encode(read_wrist({'path':image['path']}))
    window={'contacts':contacts,'wrench':wrenches,'visual':visual,'tactile_arrays':arrays}
    values={};versions={};feature_evidence={}
    for name,definition in spec['observation']['modalities'].items():
        value=load_callable(definition['extractor'])(window,spec)
        if isinstance(value,dict):values[name]=value['values'];versions[name]=value['version'];feature_evidence[name]=value.get('evidence',{})
        else:values[name]=value;versions[name]=definition['version']
    row={k:event[k] for k in ('node_id','session','trial_id','group_id','window_index','phase','t_ns','plan_coords','attrs','u_executed','u_receipt','previous_node_id')}
    row.update(task_id=task_id(spec),skill_id=spec['skill_id'],window_valid=observation['bilateral'] and observation['force_stable'],
               obs=values,extractor_versions=versions,feature_evidence=feature_evidence)
    model_observation(row,spec['observation']['modalities'])
    event['live_features']={'obs':values,'extractor_versions':versions,'feature_evidence':feature_evidence}
    event['live_contacts']=contacts;event['wrench']=wrenches;event['visual_features']=visual
    return event,row


class GraphRecoveryProxy:
    """V4's RecoverySkill interface backed by the jointly trained graph bundle."""
    def __init__(self,backend,request,spec):
        self.backend=backend;self.request=request;self.spec=spec
        self.runtime=SymbolicRecoverySkill(joblib.load(request['model_bundle'])) if request['mode']=='learned' else None
        if self.runtime and (self.runtime.bundle['model_id']!=request['model_id'] or digest(self.runtime.spec)!=digest(spec)):
            raise ValueError('Selected graph bundle/spec differs from the requested candidate')
        self.model_id=request['model_id'] or 'planned_exploration'
        self.model_sha256=self.model_id;self.config_sha256=digest(spec);self.mode=request['mode'];self.certificate=None
        self.model={'schema':'symbolic_recovery_v4'};self.config=spec.get('runtime',{});self.reset()

    def reset(self):self.events=[];self.index=0;self.previous=None;self.window=None;self.last=None

    def _outputs(self,state,reason,action=None,prediction=None):
        ready=state=='ready';invoke=state=='recoverable'
        result={'invocation':{'invoke':invoke if state!='Unknown' else None,'status':state,'reason':reason},
                'recovery':{'action':action,'decision':{'reason':reason,'prediction':prediction}},
                'exit':{'completed':ready,'status':'success' if ready else 'running' if invoke else 'incomplete',
                        'hand_back':not invoke,'reason':reason,'target_observed':ready,
                        'next_owner':'forcepolicy' if ready else None},
                'graph_model_id':self.model_id,'mode':self.mode}
        self.last=result;self.events.append(deepcopy(result));return result

    def start(self,observation):return self.advance(observation)

    def advance(self,observation,applied_action=None):
        if not observation.get('bilateral') or not observation.get('force_stable'):return self.abort('incomplete_contact_window')
        folders=list(self.backend.output_dir.glob('invocation_*'))
        folder=max(folders,key=lambda p:p.stat().st_mtime_ns)
        self.window,row=live_window(self.backend,self.backend._graph_probe,observation,folder,self.index,self.previous,applied_action,self.spec)
        self.window['repeat_context']=digest([self.request.get('campaign'),self.backend.T0.tolist(),self.window['attrs'].get('sensor_version'),self.backend.config.get('hardware_config_sha256')])
        self.window['campaign_request']={k:self.request.get(k) for k in ('trial_id','role','model_id','frozen_model_id','post_freeze','rule_id')}
        self.previous=row['node_id']
        if self.mode=='planned':
            actions=self.request.get('actions',[])
            action=normalize_action(actions[self.index],self.spec) if self.index<len(actions) else None
            state='recoverable' if action is not None else 'ready';prediction=None
            reason='planned_action' if action else 'exploration_handoff_for_actual_consumer_test'
        else:
            prediction=self.runtime.infer(row);state=prediction['state'];action=prediction.get('action')
            reason=prediction.get('reason',state)
            if state=='recoverable' and self.index>=self.spec.get('runtime',{}).get('max_steps',3):return self.abort('max_steps')
        self.index+=1
        command=None
        if action is not None:
            if action['kind']!='tcp_translation_a':raise ValueError('This instance does not implement capability '+action['kind'])
            command={'axis':'a','interface':'open_retreat_translate_reapproach_probe','delta_mm':float(action['u'][0])}
        outputs=self._outputs(state,reason,command,prediction)
        # Exploration handoff is an intervention, never learned readiness evidence.
        if self.mode=='planned':outputs['exit']['target_observed']=False
        return outputs

    def abort(self,reason):return self._outputs('Unknown',reason)


def install_backend(backend,settings):
    request=json.loads(Path(settings['request']).read_text());spec=load_spec(request['spec'])
    backend._graph_consumer=consumer_identity()
    condition=next(c for c in spec['post_conditions'] if c['id']==spec['active_post_condition'])
    expected=condition['consumer']
    if expected.get('version') and expected['version']!=backend._graph_consumer['version']:raise ValueError('Different ForcePolicy consumer checkpoint')
    if backend._graph_consumer['params']['closure_floor']!=expected['params']['closure_floor']:raise ValueError('Different downstream closure floor')
    backend.single_correction_open_handoff=False
    backend.recovery=GraphRecoveryProxy(backend,request,spec)
    # Load frozen encoders before prepare() opens sensors or grants control.
    from .legacy_v2.features import FrozenVision
    import torch
    torch.set_num_threads(int(settings.get('cpu_threads',4)))
    backend._graph_vision=FrozenVision('cpu',spec['data']['visual_model_directory'])
    if 'tactile_encoder' in spec:
        from bottle_scan.state_graph.modalities.tactile_embedding import FrozenEncoder,_ENCODERS,digest as encoder_digest
        encoder=FrozenEncoder(spec['tactile_encoder']);encoder._load()
        _ENCODERS[encoder_digest(spec['tactile_encoder'])]=encoder
    backend._graph_buffers=defaultdict(lambda:deque(maxlen=4096))
    prepare=backend.prepare
    def prepared():
        result=prepare();append=backend.hw._append
        def capture(event):
            if event.stream.startswith(('dd_encoded_','wrench_raw_')):backend._graph_buffers[event.stream].append(event)
            return append(event)
        backend.hw._append=capture
        return result
    backend.prepare=prepared
    observe=backend._observation
    def observing(status,probe):
        backend._graph_probe=probe
        return observe(status,probe)
    backend._observation=observing
    def record(probe,folder,index,previous,receipt,observation=None):
        window=backend.recovery.window
        if window is None:return None
        backend._emit(window);return window['node_id']
    backend._state_graph_window=record
    backend._emit({'event':'graph_runtime_selected','model_id':backend.recovery.model_id,'mode':request['mode'],
                   'consumer':backend._graph_consumer,'request':request,'legacy_recovery_decisions_used':False})
