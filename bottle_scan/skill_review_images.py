"""Export exactly bound recorded tactile keyframes for offline human review.

Only probe references choose images. No nearest-frame/HOLD substitution, image
normalization, inferred contact texture, hardware connection or label creation.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import struct
import zlib

import numpy as np

MAX_CAPTURE_DELAY_NS = 500_000_000


class _Unavailable(ValueError):
    pass


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _stamp(value):
    return type(value) is int and value >= 0


def _unavailable(reason):
    return {'status':'unavailable','reason':str(reason),'filename':None}


def _blank(window, reason):
    return {'timestamp_ns':window.get('timestamp_ns'),'status':'unavailable',
            'left':_unavailable(reason),'right':_unavailable(reason),'pair_skew_ms':None}


def _probe(row, timestamp):
    candidates=[]
    if isinstance(row.get('probe_initial'),dict):
        candidates.append(('probe_initial',row['probe_initial']))
    for name in ('probe_regrasps','probe_confirmations'):
        for i,item in enumerate(row.get(name) or []):
            if isinstance(item,dict) and isinstance(item.get('probe'),dict):
                candidates.append((f'{name}[{i}].probe',item['probe']))
    candidates=[(path,p) for path,p in candidates if _stamp(p.get('timestamp_ns')) and p['timestamp_ns']==timestamp]
    if not candidates:raise _Unavailable('no_exact_probe_for_observation')
    first=candidates[0][1]
    if any(p!=first for _,p in candidates[1:]):
        raise _Unavailable('ambiguous_probe_for_observation')
    key=first.get('keyframe')
    if not isinstance(key,dict) or key.get('reason')!='PROBE_last':
        raise _Unavailable('probe_has_no_PROBE_last_keyframe')
    requested=key.get('requested_ns')
    if not _stamp(requested):raise _Unavailable('missing_keyframe_request_timestamp')
    if not timestamp <= requested <= timestamp+MAX_CAPTURE_DELAY_NS:
        raise _Unavailable('keyframe_request_not_current_for_observation')
    return first,[path for path,_ in candidates]


def _committed(group, fields):
    count=group.attrs.get('committed_count')
    if isinstance(count,(bool,np.bool_)) or not isinstance(count,(int,np.integer)) or count<0:
        raise _Unavailable('missing_or_invalid_committed_count')
    count=int(count)
    if any(key not in group or group[key].ndim<1 or len(group[key])<count for key in fields):
        raise _Unavailable('truncated_committed_stream')
    return count


def _metadata(value):
    if isinstance(value,bytes):value=value.decode('utf-8')
    result=json.loads(value)
    if not isinstance(result,dict):raise _Unavailable('invalid_frame_metadata')
    return result


def _expected_serial(trial, row, probe, source, side):
    """Use persisted identities, including only exact probe DD source frames."""
    values=[];provenance=[]
    for origin,item in (('review_manifest',trial),('trial_metadata',row),('probe',probe)):
        for name in ('sensor_serials','serials'):
            mapping=item.get(name)
            if isinstance(mapping,dict) and isinstance(mapping.get(side),str):
                values.append(mapping[side]);provenance.append(origin+'.'+name)
    windows=probe.get('window')
    stream='streams/dd_'+side
    if isinstance(windows,list) and len(windows)==2 and stream in source:
        bound=windows[0 if side=='left' else 1]
        stamps=bound.get('frame_ns') if isinstance(bound,dict) else None
        if isinstance(stamps,list) and stamps and all(_stamp(t) for t in stamps):
            group=source[stream]
            count=_committed(group,('monotonic_ns','metadata_json'))
            times=np.asarray(group['monotonic_ns'][:count])
            for timestamp in stamps:
                matches=np.flatnonzero(times==timestamp)
                if len(matches)>1:raise _Unavailable('ambiguous_probe_dd_source_identity')
                if len(matches)==1:
                    meta=_metadata(group['metadata_json'][int(matches[0])])
                    if meta.get('serial'):
                        values.append(meta['serial']);provenance.append('exact_probe_dd_frames')
    if any(not isinstance(v,str) or not v for v in values) or len(set(values))>1:
        raise _Unavailable('conflicting_recorded_sensor_identity')
    return (values[0] if values else None),sorted(set(provenance))


def _png(array, metadata):
    """Encode uint8 pixels losslessly, without resize/crop/contrast adjustment."""
    if array.dtype!=np.uint8 or array.size==0 or array.size>16*1024*1024:
        raise _Unavailable('unsupported_raw_image_dtype_or_size')
    original_shape=list(array.shape)
    encoding=next((value for value in (metadata.get('image_encoding'),metadata.get('encoding'),
                  (metadata.get('raw') or {}).get('encoding'),(metadata.get('raw') or {}).get('color_space'))
                  if isinstance(value,str)),None)
    encoding=encoding.lower() if encoding else None
    transform='none'
    if array.ndim==2:
        if encoding not in (None,'mono8','gray','grey','gray8','grayscale','l'):
            raise _Unavailable('grayscale_shape_encoding_conflict')
        color_type=0;mode='L';pixels=array
    elif array.ndim==3 and array.shape[2] in (3,4):
        if array.shape[2]==3 and encoding in ('rgb','rgb8'):
            color_type=2;mode='RGB';pixels=array
        elif array.shape[2]==3 and encoding in ('bgr','bgr8'):
            color_type=2;mode='RGB';pixels=array[:,:,::-1];transform='explicit_bgr_to_rgb'
        elif array.shape[2]==4 and encoding in ('rgba','rgba8'):
            color_type=6;mode='RGBA';pixels=array
        elif array.shape[2]==4 and encoding in ('bgra','bgra8'):
            color_type=6;mode='RGBA';pixels=array[:,:,[2,1,0,3]];transform='explicit_bgra_to_rgba'
        else:raise _Unavailable('color_encoding_missing_or_unsupported')
    else:raise _Unavailable('unsupported_raw_image_shape')
    height,width=pixels.shape[:2]
    if height<=0 or width<=0:raise _Unavailable('empty_image')
    def chunk(kind,payload):
        return struct.pack('>I',len(payload))+kind+payload+struct.pack('>I',zlib.crc32(kind+payload)&0xffffffff)
    scanlines=b''.join(b'\x00'+np.ascontiguousarray(line).tobytes() for line in pixels)
    result=(b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('>IIBBBBB',width,height,8,color_type,0,0,0))+
            chunk(b'IDAT',zlib.compress(scanlines))+chunk(b'IEND',b''))
    return result,{'source_shape':original_shape,'png_shape':list(pixels.shape),'dtype':'uint8',
                   'png_mode':mode,'pixel_transform':transform,'source_encoding':encoding}


def _side(source, trial, row, probe, side, timestamp, file_proof):
    key=probe['keyframe'];requested=key['requested_ns'];selected=(key.get('gel') or {}).get(side)
    if not _stamp(selected):raise _Unavailable('missing_exact_gel_keyframe_timestamp')
    if not requested <= selected <= requested+MAX_CAPTURE_DELAY_NS:
        raise _Unavailable('gel_keyframe_outside_requested_capture_window')
    stream='streams/gel_keyframe_'+side
    if stream not in source:raise _Unavailable('gel_keyframe_stream_missing')
    group=source[stream]
    count=_committed(group,('data','monotonic_ns','metadata_json'))
    stamps=np.asarray(group['monotonic_ns'][:count])
    if stamps.ndim!=1 or stamps.dtype.kind not in 'iu':raise _Unavailable('invalid_source_timestamps')
    matches=np.flatnonzero(stamps==selected)
    if len(matches)!=1:
        raise _Unavailable('exact_gel_keyframe_not_committed' if not len(matches) else 'ambiguous_exact_gel_keyframe')
    index=int(matches[0]);metadata=_metadata(group['metadata_json'][index])
    stage=metadata.get('e1e2_stage')
    if metadata.get('keyframe_reason')!='PROBE_last':raise _Unavailable('frame_is_not_PROBE_last')
    if not isinstance(stage,str) or not (stage=='PROBE' or stage.startswith('PROBE_')):
        raise _Unavailable('frame_is_not_probe_stage')
    if metadata.get('stage') not in (None,stage):raise _Unavailable('frame_stage_conflict')
    identifier=row.get('id',trial.get('id'))
    if metadata.get('trial_id')!=identifier:raise _Unavailable('frame_trial_identity_mismatch')
    serial=metadata.get('serial')
    if not isinstance(serial,str) or not serial:raise _Unavailable('frame_sensor_serial_missing')
    expected,serial_proof=_expected_serial(trial,row,probe,source,side)
    if expected is not None and serial!=expected:raise _Unavailable('frame_sensor_serial_mismatch')
    data=group['data']
    if data.dtype!=np.uint8 or np.prod(data.shape[1:],dtype=np.int64)>16*1024*1024:
        raise _Unavailable('unsupported_raw_image_dtype_or_size')
    array=np.asarray(data[index])
    raw=metadata.get('raw')
    if not isinstance(raw,dict) or (raw.get('present') is not True or raw.get('dtype')!='uint8' or
            raw.get('shape')!=list(array.shape) or raw.get('bytes')!=array.nbytes):
        raise _Unavailable('frame_payload_metadata_mismatch')
    png,pixel_info=_png(array,metadata)
    return png,{'status':'available','source':deepcopy(file_proof),'source_stream':stream,
        'frame_index':index,'frame_timestamp_ns':selected,'committed_count':count,'requested_ns':requested,
        'observation_delta_ms':(selected-timestamp)/1e6,'request_delta_ms':(selected-requested)/1e6,
        'source_frame_sha256':_sha(np.ascontiguousarray(array).tobytes()),'png_sha256':_sha(png),
        'metadata_sha256':_sha(json.dumps(metadata,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()),
        'metadata':{k:metadata.get(k) for k in ('fid','serial','keyframe_reason','e1e2_stage','stage','trial_id','representation')},
        'serial_mapping_verified':expected is not None,'serial_mapping_provenance':serial_proof,
        'binding':'exact_observation_probe_keyframe_timestamp','capture_is_after_observation':True,**pixel_info}


def export_contact_images(trial_dict_from_manifest, outdir):
    """Return one left/right status record per manifest window; never change labels.

    ``filename`` is relative to outdir. Selected source frame bytes and resulting
    PNGs are hashed; huge HDF5 recordings are identified by path/size/mtime, not
    read in full. All frame indexes are below the stream's committed_count.
    """
    trial=trial_dict_from_manifest;windows=trial.get('windows') or []
    folder=Path(trial['source_trial']).resolve();outdir=Path(outdir).resolve()
    if outdir==folder or folder in outdir.parents:
        raise ValueError('review image output must be outside the original trial')
    meta_path=folder/'meta.json';raw_path=folder/'streams.h5'
    try:
        original=meta_path.read_bytes();row=json.loads(original)
        if trial.get('source_metadata_sha256') and _sha(original)!=trial['source_metadata_sha256']:
            raise _Unavailable('source_metadata_changed')
        if row.get('id') not in (None,trial.get('id')):raise _Unavailable('source_trial_identity_changed')
        stat=raw_path.stat();proof={'path':str(raw_path),'size_bytes':stat.st_size,'mtime_ns':stat.st_mtime_ns,
                                  'source_metadata_sha256':_sha(original)}
        recorded=trial.get('raw') or {}
        if any(key in recorded and recorded[key]!=proof[key] for key in ('size_bytes','mtime_ns')):
            raise _Unavailable('source_recording_changed')
        import h5py
        source=h5py.File(raw_path,'r')
    except (OSError,ValueError,TypeError,KeyError) as error:
        reason=str(error) if isinstance(error,_Unavailable) else 'source_unavailable:'+type(error).__name__
        return [_blank(window,reason) for window in windows]
    records=[];pending=[]
    identifier=str(trial.get('id',''))
    if not re.fullmatch(r'[A-Za-z0-9_-]+',identifier):identifier=_sha(str(folder).encode())[:16]
    try:
        with source:
            for window in windows:
                timestamp=window.get('timestamp_ns')
                try:
                    if not _stamp(timestamp):raise _Unavailable('invalid_observation_timestamp')
                    probe,paths=_probe(row,timestamp)
                except (ValueError,TypeError,KeyError) as error:
                    records.append(_blank(window,str(error) if isinstance(error,_Unavailable) else 'invalid_probe_metadata'));continue
                record={'timestamp_ns':timestamp,'probe_paths':paths,'pair_skew_ms':None}
                for side in ('left','right'):
                    try:
                        png,info=_side(source,trial,row,probe,side,timestamp,proof)
                        filename=f'images/trial_{identifier}_{timestamp}_{side}.png'
                        info['filename']=filename;record[side]=info;pending.append((filename,png,record,side))
                    except (OSError,ValueError,TypeError,KeyError,IndexError,OverflowError) as error:
                        record[side]=_unavailable(str(error) if isinstance(error,_Unavailable) else 'invalid_source_frame:'+type(error).__name__)
                if all(record[side]['status']=='available' for side in ('left','right')):
                    record['pair_skew_ms']=abs(record['left']['frame_timestamp_ns']-record['right']['frame_timestamp_ns'])/1e6
                    if record['left']['metadata']['serial']==record['right']['metadata']['serial']:
                        record['left']=_unavailable('same_serial_on_both_fingers');record['right']=_unavailable('same_serial_on_both_fingers')
                        record['pair_skew_ms']=None
                record['status']='available' if all(record[s]['status']=='available' for s in ('left','right')) else 'partial' if any(record[s]['status']=='available' for s in ('left','right')) else 'unavailable'
                records.append(record)
        after=raw_path.stat()
        if (after.st_size,after.st_mtime_ns)!=(stat.st_size,stat.st_mtime_ns) or meta_path.read_bytes()!=original:
            return [_blank(window,'source_changed_during_export') for window in windows]
        for filename,png,record,side in pending:
            if record[side]['status']!='available':continue
            path=outdir/filename;path.parent.mkdir(parents=True,exist_ok=True)
            with path.open('xb') as stream:stream.write(png)
    except (OSError,ValueError,TypeError,KeyError) as error:
        # Output failures are not a reason to substitute another recorded image.
        for _,_,record,side in pending:
            if record[side]['status']=='available':record[side]=_unavailable('image_export_failed:'+type(error).__name__)
        for record in records:record['status']='unavailable';record['pair_skew_ms']=None
    return records
