from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import json
import numpy as np
import h5py
from bottle_scan.state_graph.skill_interface import load_spec
from bottle_scan.state_graph.extract_nodes import extract_nodes
from skills.contact_position_a.scheduler import capture_window,lifecycle_event,trials_from_events,append_reviewed_outcome
from skills.contact_position_a.post import forcepolicy_l2


def test_scheduler_records_existing_camera_and_measured_transition_without_fake_l2(tmp_path):
    folder=tmp_path/'invocation_1_123';folder.mkdir()
    baseline=np.zeros((8,8,3),np.float32);frames=np.repeat(baseline[None],5,axis=0);frames[:,2:6,2:6,0]=.1
    times=np.arange(1,6,dtype=np.int64)*10
    with h5py.File(folder/'streams.h5','w') as h:
        for name,data in [('dd_left',frames),('dd_right',frames),('wrench_left',np.ones((5,6))),('wrench_right',np.ones((5,6))*2)]:
            g=h.create_group('streams/'+name);g.create_dataset('data',data=data);g.create_dataset('monotonic_ns',data=times);g.attrs['committed_count']=5
    class Camera:
        latest_timestamp=1.;latest_frame_id=3;config=SimpleNamespace(color_mode='rgb')
        def read_latest(self,max_age_ms):return np.zeros((8,8,3),np.uint8)
    backend=SimpleNamespace(config={'state_graph_recording':{}},robot=SimpleNamespace(cameras={'right_wrist':Camera()}),T0=np.eye(4),
        bounds=SimpleNamespace(a=np.array([1,0,0.])),hw=SimpleNamespace(dd_reference=[baseline,baseline],zero={'contact_thresholds_sdk':[.1,.1]}),output_dir=tmp_path,
        cfg={'gripper':{'min_normalized':.330}},_graph_consumer={'controller':'existing_forcepolicy','version':'test_checkpoint','params':{'closure_floor':.330,'target_force_sdk':5.}})
    probe={'valid':True,'stable_start_ns':10,'timestamp_ns':50,'descriptors':[{'valid':True},{'valid':True}],
           'window':[{'frame_ns':times.tolist(),'frame_fids':list(range(5))} for _ in range(2)]}
    sample={'pose':np.eye(4),'timestamp_ns':50,'wrench_zeroed':np.ones((2,6)),'normal':np.ones(2),'width':.4}
    first=capture_window(backend,probe,sample,folder,0)
    sample['pose'][0,3]=2
    second=capture_window(backend,probe,sample,folder,1,first['node_id'],{'applied':True,'action_mm':2.})
    for window in (first,second):window['visual_features']=[0.]*384
    handoff=lifecycle_event({'event':'skill_completed','result':{}},second,backend._graph_consumer)
    result=lifecycle_event({'event':'main_resumed','force_readiness_verified':True},second,backend._graph_consumer)
    assert result['consumer_outcome']['L1'] is True and result['consumer_outcome']['L2'] is None
    events=[first,second,handoff,result];path=tmp_path/'events.jsonl';path.write_text(''.join(json.dumps(e)+'\n' for e in events))
    spec=load_spec('skills/contact_position_a/spec.json')
    trials=list(trials_from_events(events,spec));window=trials[0]['windows'][1]
    assert window['u_executed']==[2.] and window['wrench'][6:]==[2.]*6
    assert window['contacts'][0]['u_px']==3.5
    assert not forcepolicy_l2(trials[0],window,spec['post_conditions'][0])['evaluated']
    append_reviewed_outcome(path,L2=True,L3=False,evidence='test synthetic review')
    nodes,report=extract_nodes([str(path)],spec)
    assert len(nodes)==2 and nodes[1]['post'][spec['active_post_condition']]['success'] is True
    assert nodes[0]['post'][spec['active_post_condition']]['evaluated'] is False


def test_instance_action_adapter_uses_existing_reprobe():
    from skills.contact_position_a.action import RecoveryAdapter
    calls=[]
    class Backend:
        def _recover_once(self,runner,target,action,row):
            calls.append(action);row['receipt']={'applied':True,'action_mm':action['delta_mm']}
            return target,'contact',{'valid':True},row
    adapter=RecoveryAdapter(Backend(),object(),np.eye(4),lambda obj:obj.probe)
    assert adapter.execute([2.])['action_mm']==2. and adapter.observe()['valid']
    assert calls[0]['interface']=='open_retreat_translate_reapproach_probe'


def test_force_readiness_failure_is_negative_but_sensor_fault_is_unassessed():
    condition=load_spec('skills/contact_position_a/spec.json')['post_conditions'][0]
    consumer={'controller':'existing_forcepolicy','version':'test','params':{'closure_floor':.330}}
    window={'node_id':'n','group_id':'g'}
    event=lifecycle_event({'event':'fault','reason':'ForcePolicy did not stabilize during handoff'},window,consumer)
    assert event['consumer_outcome']['L1'] is False and event['consumer_outcome']['L2'] is None
    result=forcepolicy_l2({},dict(is_terminal=True,consumer_outcome=event['consumer_outcome']),condition)
    assert result['evaluated'] and result['success'] is False
    assert lifecycle_event({'event':'fault','reason':'Raw finger wrench time skew'},window,consumer) is None
