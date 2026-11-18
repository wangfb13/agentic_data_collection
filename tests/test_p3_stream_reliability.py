"""No hardware: paired arrivals, immutable raw evidence, and batched persistence."""
import json
import h5py
import numpy as np
import pytest
from bottle_scan.backend import Sample
from bottle_scan.hardware_sensors import ActualX5SensorSource,SensorFault
from bottle_scan.recording import H5Writer,AsyncRecorder
from skills.contact_position_a import p3_force


def source():
    cfg={'hardware':{'public_ros':{'sensors':{
        'serials':{'left':'L','right':'R'},'dd_address':'tcp://unused:39030',
        'wrench_address':'tcp://unused:39020',
        'wrist_endpoints':{'left_wrist':'tcp://unused:5552','right_wrist':'tcp://unused:5552'}}}}}
    clock=[1_000_000_000]
    s=ActualX5SensorSource(cfg,clock=lambda:clock[0])
    for side in ('left','right'):
        s._latest['dd_'+side]=Sample('dd_'+side,np.ones((2,2,3)),clock[0],None,{'fid':1})
    return s,clock


def wrench(s,side,t,fid,value=1):
    serial=s.serials[side]
    packet=[serial.encode(),json.dumps({'schema':'dm_x5_wrench_v1','serial':serial,
        'fid':fid,'wrench':[value]*6,'timestamp':42.}).encode()]
    s._ingest('wrench',packet,t)


def test_pair_uses_recent_real_frame_without_changing_latest_safety_data():
    s,clock=source()
    wrench(s,'right',940_000_000,1,2)
    wrench(s,'left',965_000_000,1,3)
    wrench(s,'left',1_000_000_000,2,99)
    r=s.observation_snapshot(pair_skew_s=.05,pair_max_age_s=.2)
    assert r['source_sequences']['wrench_raw_left']==1
    assert r['source_timestamp_ns']['wrench_raw_left']==965_000_000
    np.testing.assert_equal(r['wrench_raw'][:,0],[3,2])
    assert s._latest['wrench_raw_left'].data[0]==99
    events=s.drain_samples()
    assert [e.data[0] for e in events]==[2,3,99]
    assert r['wrench_pairing']['skew_ns']==25_000_000
    # A fresh opposite arrival advances the pair; no frozen old pair.
    wrench(s,'right',999_000_000,2,4)
    r=s.observation_snapshot(pair_skew_s=.05,pair_max_age_s=.2)
    assert r['source_sequences']['wrench_raw_left']==2
    assert r['source_sequences']['wrench_raw_right']==2


@pytest.mark.parametrize('right_time',[800_000_000,799_000_000])
def test_no_pair_or_expired_pair_still_rejected(right_time):
    s,clock=source()
    wrench(s,'right',right_time,1)
    wrench(s,'left',1_000_000_000,1)
    with pytest.raises(SensorFault,match='no fresh admissible pair'):
        s.observation_snapshot(pair_skew_s=.05,pair_max_age_s=.2)


def test_expired_both_streams_not_masked_by_history():
    s,clock=source()
    for side in ('left','right'):wrench(s,side,700_000_000,1)
    with pytest.raises(SensorFault,match='Stale source'):
        s.observation_snapshot(pair_skew_s=.05,pair_max_age_s=.2)


def test_force_bridge_requests_same_strict_pair_contract():
    class HW:
        def force_sample(self,**kw):return kw
    assert p3_force.paired_force_sample(HW())==dict(pair_skew_s=.05,pair_max_age_s=.2)


def test_batched_writer_preserves_interleaved_payloads_and_commit_indices(tmp_path):
    w=H5Writer(tmp_path/'s.h5',compression='lzf')
    rows=[]
    for i in range(80):
        rows.extend([('force',np.arange(6,dtype=np.float64)+i,i,None,{'fid':i}),
                     ('dd',np.full((4,5,3),i,dtype=np.float32),i,None,{'fid':i})])
    assert w.append_many(rows)==[i for i in range(80) for _ in range(2)]
    w.close()
    with h5py.File(tmp_path/'s.h5') as h:
        for name in ('force','dd'):
            g=h['streams/'+name]
            assert g.attrs['committed_count']==80
            np.testing.assert_array_equal(g['sequence'][:],np.arange(80))
            for i in range(80):
                np.testing.assert_array_equal(g['data'][i],rows[2*i+(name=='dd')][1])
                assert json.loads(g['metadata_json'][i])['fid']==i


def test_batch_validation_does_not_publish_partial_commit(tmp_path):
    w=H5Writer(tmp_path/'s.h5')
    w.append('f',np.ones(6),10)
    with pytest.raises(ValueError,match='backwards'):
        w.append_many([('f',np.ones(6),11,None,None),('f',np.ones(6),9,None,None)])
    w.close()
    with h5py.File(tmp_path/'s.h5') as h:
        assert h['streams/f'].attrs['committed_count']==1
        assert len(h['streams/f/data'])==1


def test_async_batches_keep_json_close_barriers_and_all_rows(tmp_path):
    w=AsyncRecorder(tmp_path/'r')
    for i in range(400):w.h5.append('f',np.full(6,float(i)),i,attrs={'fid':i})
    w.json('middle.json',{'barrier':True})
    for i in range(400,800):w.h5.append('f',np.full(6,float(i)),i,attrs={'fid':i})
    w.close()
    manifest=json.loads((tmp_path/'r/recording_status.json').read_text())
    assert manifest['complete'] and manifest['committed_messages']==801
    with h5py.File(tmp_path/'r/streams.h5') as h:
        np.testing.assert_array_equal(h['streams/f/data'][:,0],np.arange(800))


def test_resume_archives_real_cost_and_partial_airborne_evidence(tmp_path):
    from skills.contact_position_a.p3_resume import resume_interrupted
    from bottle_scan.state_graph import campaign
    d=tmp_path/'trials/t1';d.mkdir(parents=True)
    request={'trial_id':'t1'}
    trial=dict(trial_id='t1',directory=str(d),status='started',started_ns=123,request=request,
               error='Raw finger wrench time skew',dispatch_ordinals=[1])
    state=dict(trials=[trial],pending={'trial_ids':['t1']},spent={'trials':1,'actions':0},dispatch_attempts=1)
    campaign.atomic(tmp_path/'state.json',state)
    campaign.atomic(d/'exception.json',{'reason':'Raw finger wrench time skew'})
    campaign.atomic(d/'airborne.json',{'possibly_airborne':True})
    assert resume_interrupted(tmp_path,scene_reset_confirmed=True)==['t1']
    r=campaign.read(tmp_path/'state.json')
    assert r['dispatch_attempts']==1 and r['spent']['trials']==0
    assert r['trials'][0]['status']=='queued'
    archive=tmp_path/'incomplete_attempts/t1/123'
    assert campaign.read(archive/'airborne.json')['possibly_airborne']
    assert campaign.read(archive/'operator_reset.json')['scene_reset_confirmed']
    assert resume_interrupted(tmp_path,scene_reset_confirmed=True)==[]


def test_resume_never_assumes_scene_is_reset(tmp_path):
    from skills.contact_position_a.p3_resume import resume_interrupted
    with pytest.raises(ValueError,match='确认'):
        resume_interrupted(tmp_path)
