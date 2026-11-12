"""Read-only diagnostic evidence tests; synthetic records never control hardware."""
import copy
import json
from pathlib import Path
import shutil
import numpy as np
import h5py
import pytest
from bottle_scan.ambiguity_analysis import analyze, _raw_window, _load_sources


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def descriptor(u=10., area=30., box=None):
    return {'valid':True,'centroid_px':[u,10.], 'bbox_xyxy_px':box or [4,4,16,16],
            'area_px':area,'principal_angle_deg':0.,'touches_edge':False}


def session(root, *, count=4, separated=False, schema='e1e2_session_v1', cond='A'):
    meta={'schema':schema,'created_unix_ns':1000000,'simulated':False,'cond':cond,
          'T0':np.eye(4).tolist(),'config':{'axes_tcp':{'a':[0,0,1],'l':[1,0,0],'c':[0,-1,0]},
                                           'tactile':{'depth_channel':0}}}
    dump(root/'session_meta.json',meta)
    for state,a in enumerate((-4.,4.)):
        for rep in range(count):
            i=state*count+rep+1; folder=root/'trials'/f'{i:05d}'
            pose=np.eye(4);pose[2,3]=a
            u=10+state*30 if separated else 10
            probe={'valid':True,'descriptors':[descriptor(u,box=[4+20*state,4,16+20*state,16]) for _ in range(2)],
                   'measured_tcp_mm':pose.tolist(),'timestamp_ns':1000+i,
                   'normal_sdk':[1,1], 'window':[{'frame_ns':[100,200,300]}]*2}
            row={'id':f'{i:05d}','started_unix_ns':i,'complete':True,'recording_complete':True,
                 'bottle':'same','cond':cond,'delta_mm':a,'label':'normal','probe_initial':probe,
                 'zero':{'created_unix_ns':500,'offset_raw':np.zeros((2,6)).tolist()}}
            if schema=='ambiguity_session_v1':
                row['perturbation']={'a_mm':a,'l_mm':0.,'tilt_deg':0.}
            dump(folder/'meta.json',row);dump(folder/'recording_status.json',{'complete':True})
    return root


def change(root, i, fn):
    p=root/'trials'/f'{i:05d}'/'meta.json';row=json.loads(p.read_text());fn(row);dump(p,row)


def test_conflict_candidate_is_not_claimed_physical_ambiguity(tmp_path):
    root=session(tmp_path/'s')
    report=analyze([root])
    assert report['summary']['centroid_candidate_pairs']==16
    assert report['summary']['richer_feature_separation_pairs']==16
    assert report['summary']['reliable_physical_ambiguity_established'] is False
    assert report['candidate_pairs'][0]['interpretation']=='feature_compression_candidate'


def test_separable_observations_do_not_trigger(tmp_path):
    r=analyze([session(tmp_path/'s',separated=True)])
    assert r['summary']['centroid_candidate_pairs']==0
    assert r['summary']['status']=='no_candidate_in_observed_data'


def test_insufficient_repeat_support_not_false_negative(tmp_path):
    r=analyze([session(tmp_path/'s',count=2)])
    assert r['summary']['status']=='insufficient_evidence'
    assert r['summary']['noise_calibrated_strata']==0


def test_feedback_disagreement_is_not_evidence(tmp_path):
    root=session(tmp_path/'s')
    for i in range(1,5):
        change(root,i,lambda row:row['probe_initial']['measured_tcp_mm'][2].__setitem__(3,15.))
    r=analyze([root])
    assert r['label_quality_issues']['command_feedback_disagreement']==4
    assert r['summary']['centroid_candidate_pairs']==0


def test_copied_session_deduplicates_trials(tmp_path):
    root=session(tmp_path/'s');copyroot=tmp_path/'copy';shutil.copytree(root,copyroot)
    r=analyze([root,copyroot])
    assert r['summary']['trials_loaded']==8
    assert r['summary']['duplicate_trials']==8
    assert r['summary']['centroid_candidate_pairs']==16


def test_regrasp_does_not_inherit_initial_offset(tmp_path):
    root=session(tmp_path/'s',cond='B')
    def update(row):
        p=copy.deepcopy(row['probe_initial']);del p['measured_tcp_mm']
        row['probe_regrasps']=[{'probe':p}]
    change(root,1,update)
    r=analyze([root])
    assert r['summary']['regrasp_observations']==0
    assert r['exclusions']['missing_geometric_label']==1


def test_new_npz_provides_reference_subtracted_features(tmp_path):
    root=session(tmp_path/'s',schema='ambiguity_session_v1')
    for i in range(1,9):
        change(root,i,lambda row:row['probe_initial'].update(observation_file='probe_observation.npz',stable_probe=True))
        dd=np.ones((8,8,3));ref=np.full_like(dd,.25)
        np.savez(root/'trials'/f'{i:05d}'/'probe_observation.npz',dd_left=dd,dd_right=dd,
                 reference_left=ref,reference_right=ref,wrench_zeroed=np.ones((2,6)))
    r=analyze([root],settings={'read_h5':True})
    assert all(o['features']['dd_zeroed_pool']==[.75]*32 for o in r['observations'])
    assert all(o['raw_provenance']['dd_reference_subtracted'] for o in r['observations'])
    assert all('descriptor_wrench' in o['features'] for o in r['observations'])


def test_simulated_excluded_from_real_evidence(tmp_path):
    root=session(tmp_path/'s');p=root/'session_meta.json';m=json.loads(p.read_text());m['simulated']=True;dump(p,m)
    r=analyze([root])
    assert r['summary']['observations']==0
    assert r['exclusions']['simulated_trial']==8


def test_zero_changes_are_distinct_strata(tmp_path):
    root=session(tmp_path/'s')
    for i in range(5,9):
        change(root,i,lambda row:row['zero'].__setitem__('created_unix_ns',900))
    r=analyze([root])
    assert r['summary']['strata']==2
    assert r['summary']['centroid_candidate_pairs']==0
    assert r['summary']['status']=='insufficient_evidence'


def test_moved_bottle_and_unstable_probe_not_candidates(tmp_path):
    root=session(tmp_path/'s')
    change(root,1,lambda row:row.update(bottle_moved=True))
    change(root,2,lambda row:row['probe_initial'].update(stable_probe=False))
    change(root,3,lambda row:row.update(label='pushed'))
    r=analyze([root])
    assert r['exclusions']['operator_confirmed_bottle_moved']==1
    assert r['exclusions']['invalid_or_missing_initial_probe']==1
    assert r['label_quality_issues']['operator_label_possible_object_motion']==1
    assert r['summary']['centroid_candidate_pairs']==0


def test_h5_uses_exact_saved_frames_and_raw_wrench_minus_zero(tmp_path):
    root=session(tmp_path/'s')
    folder=root/'trials'/'00001'
    with h5py.File(folder/'streams.h5','w') as h:
        h.attrs['diagnostic_subset']=True
        for side in ('left','right'):
            g=h.create_group('streams/dd_'+side)
            # One unrelated huge-valued frame between valid timestamps must not count.
            g.create_dataset('monotonic_ns',data=[100,150,200,300])
            data=np.ones((4,8,8,3));data[1]=1000
            g.create_dataset('data',data=data);g.attrs['committed_count']=4
            g=h.create_group('streams/wrench_raw_'+side)
            g.create_dataset('monotonic_ns',data=[100,200,300]);g.create_dataset('data',data=np.full((3,6),2.))
        g=h.create_group('streams/tcp_measured')
        pose=np.eye(4);pose[2,3]=-4
        g.create_dataset('monotonic_ns',data=[100,200,300]);g.create_dataset('data',data=[pose]*3)
    change(root,1,lambda row:row['zero'].update(offset_raw=np.ones((2,6)).tolist()))
    records,*_= _load_sources([root],False)
    rec=records[0];result,errors=_raw_window(rec,rec['trial']['probe_initial'],dd=True)
    assert result['raw_dd_pool_unzeroed']==[1.]*32
    np.testing.assert_allclose(result['wrench_zeroed'],np.ones((2,6)))
    assert result['raw_provenance_h5']['diagnostic_subset']
    with h5py.File(folder/'streams.h5','a') as h:
        h['streams/dd_left/monotonic_ns'][2]=201
    result,errors=_raw_window(rec,rec['trial']['probe_initial'],dd=True)
    assert 'raw_dd_pool_unzeroed' not in result
    assert any('missing or duplicate exact' in e for e in errors)


def test_missing_inputs_reported_not_silent_success(tmp_path):
    r=analyze([tmp_path/'missing'])
    assert r['issues'] and r['summary']['status']=='insufficient_evidence'


def test_report_refuses_overwrite_and_creates_artifacts(tmp_path):
    from bottle_scan.ambiguity_analysis import write_report
    r=analyze([session(tmp_path/'s')]);out=tmp_path/'out'
    write_report(r,out)
    assert all((out/name).is_file() for name in ('REPORT.md','report.json','observations.csv','candidate_pairs.csv','centroid_screen.png'))
    with pytest.raises(FileExistsError):
        write_report(r,out)


def test_legacy_reference_matches_calibration_and_stays_inside_session(tmp_path):
    from bottle_scan.ambiguity_analysis import _reference_arrays
    root=session(tmp_path/'s');p=root/'session_meta.json';meta=json.loads(p.read_text())
    meta['zero_history']=[{'created_unix_ns':500,'dd_reference_file':'/remote/project/outputs/s/baseline/abc/dd_reference.npz'}]
    dump(p,meta)
    dest=root/'baseline'/'abc'/'dd_reference.npz';dest.parent.mkdir(parents=True)
    np.savez(dest,left=np.zeros((8,8,3)),right=np.ones((8,8,3)))
    records,*_=_load_sources([root],False)
    refs,note=_reference_arrays(records[0])
    assert refs is not None and np.mean(refs[1])==1
    assert note==str(dest)
    records[0]['trial']['zero']['created_unix_ns']=501
    assert _reference_arrays(records[0])[0] is None


def test_conflicting_copied_labels_are_flagged(tmp_path):
    root=session(tmp_path/'s');other=tmp_path/'zcopy';shutil.copytree(root,other)
    change(other,1,lambda row:row.update(label='pushed'))
    r=analyze([root,other])
    assert r['label_quality_issues']['conflicting_metadata_copies']==1
    assert any(d['different_metadata'] for d in r['duplicates'])
