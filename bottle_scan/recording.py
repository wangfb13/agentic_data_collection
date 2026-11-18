"""Independent stream recording; raw numeric payloads are never resized/quantized.

These writers have no hardware dependencies. Local monotonic timestamps are not
device acquisition times; preserve source IDs and timestamp semantics in attrs.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
import threading
import multiprocessing as mp
import os
import pickle
import queue
import tempfile
import time
from collections import deque

import cv2
import h5py
import numpy as np


def _json(value):
    def convert(item):
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, np.generic):
            return item.item()
        raise TypeError(f"Cannot serialize {type(item).__name__}")
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"), default=convert)


def _name(name):
    if not isinstance(name, str) or not name or name.startswith("/") or any(p in ("", ".", "..") for p in name.split("/")):
        raise ValueError("Stream/camera name must be a relative nonempty path")
    return name


class H5Writer:
    """Append-only raw datasets under /streams/{stream} with per-stream clocks.

    Each stream has data, monotonic_ns, device_timestamp, sequence, metadata_json.
    committed_count advances only after all sample fields flush. Readers should
    honor it after interruption. Source timestamp originals are also retained in
    metadata_json to avoid loss of precision for integer device clocks.
    """
    def __init__(self, path, *, compression="gzip"):
        if compression not in ("gzip", "lzf"):
            raise ValueError("HDF5 compression must be lossless gzip or lzf")
        self.compression = compression
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = h5py.File(self.path, "x")
        self._file.attrs["schema"] = "bottle_raw_streams_v1"
        self._file.attrs["monotonic_clock"] = "recording_host_monotonic_ns"
        self._file.attrs["payload_policy"] = f"original_shape_dtype_lossless_{compression}_no_fill_or_resample"
        self._file.attrs["data_compression"] = compression
        self._file.create_group("streams")
        self._lock = threading.Lock()

    def append(self, stream, data, monotonic_ns, device_timestamp=None, attrs=None):
        return self.append_many([(stream,data,monotonic_ns,device_timestamp,attrs)])[0]

    def append_many(self, records):
        """Commit a bounded batch without per-frame resize/flush overhead.

        Every payload, timestamp and metadata row remains intact. Counts and
        acknowledgements advance only after all fields have been flushed.
        """
        grouped={};indices=[]
        with self._lock:
            root=self._file['streams']
            for stream,data,stamp,device,attrs in records:
                name=_name(stream);array=np.asarray(data)
                if array.dtype.kind not in 'biufc' or not array.size:
                    raise ValueError('Raw HDF5 data must be a nonempty numeric array/scalar')
                if isinstance(stamp,bool) or not isinstance(stamp,(int,np.integer)) or stamp<0:
                    raise ValueError('monotonic_ns must be a nonnegative integer')
                dev=float('nan') if device is None else float(device)
                if device is not None and not math.isfinite(dev):
                    raise ValueError('Source timestamp must be finite or None')
                meta=dict(attrs or {});meta['device_timestamp_raw']=device
                text=_json(meta)
                if name not in root:
                    group=root.require_group(name)
                    if list(group.keys()):raise ValueError('Stream name collides with another stream namespace')
                    group.attrs['committed_count']=0
                    group.create_dataset('data',shape=(0,*array.shape),maxshape=(None,*array.shape),
                                         chunks=(1,*array.shape),dtype=array.dtype,compression=self.compression,shuffle=True)
                    for key,dtype in (('monotonic_ns','i8'),('device_timestamp','f8'),('sequence','i8')):
                        group.create_dataset(key,shape=(0,),maxshape=(None,),chunks=True,dtype=dtype,compression='gzip')
                    group.create_dataset('metadata_json',shape=(0,),maxshape=(None,),chunks=True,dtype=h5py.string_dtype('utf-8'))
                group=root[name]
                if 'data' not in group:raise ValueError('Stream name collides with another stream namespace')
                dataset=group['data']
                if dataset.shape[1:]!=array.shape or dataset.dtype!=array.dtype:
                    raise ValueError(f'{name}: raw shape/dtype changed; start a new explicitly identified stream')
                count=int(group.attrs['committed_count'])
                rows=grouped.setdefault(name,{'start':count,'rows':[]})['rows']
                previous=rows[-1][1] if rows else group['monotonic_ns'][count-1] if count else None
                if previous is not None and stamp<previous:
                    raise ValueError(f'{name}: host monotonic time moved backwards')
                indices.append(count+len(rows));rows.append((array,stamp,dev,text))
            try:
                for name,batch in grouped.items():
                    group=root[name];begin=batch['start'];rows=batch['rows'];end=begin+len(rows)
                    values={'data':np.stack([r[0] for r in rows]),'monotonic_ns':[r[1] for r in rows],
                            'device_timestamp':[r[2] for r in rows],'sequence':np.arange(begin,end),
                            'metadata_json':[r[3] for r in rows]}
                    for key,value in values.items():
                        group[key].resize(end,axis=0);group[key][begin:end]=value
                self._file.flush()
                for name,batch in grouped.items():
                    root[name].attrs.modify('committed_count',batch['start']+len(batch['rows']))
                self._file.flush()
            except Exception:
                for name,batch in grouped.items():
                    group=root[name]
                    for key in ('data','monotonic_ns','device_timestamp','sequence','metadata_json'):
                        group[key].resize(batch['start'],axis=0)
                    group.attrs.modify('committed_count',batch['start'])
                self._file.flush()
                raise
        return indices

    def close(self):
        with self._lock:
            if self._file:
                self._file.flush()
                self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class JpegWriter:
    """Save wrist JPEG frames plus timestamps.csv; no third camera is implied.

    Incoming JPEG bytes are stored verbatim. uint8 RGB images are encoded at the
    configured quality; JPEG is lossy and is never used for tactile raw DD.
    """
    def __init__(self, directory, quality=95):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        if not 1 <= quality <= 100:
            raise ValueError("JPEG quality must be in [1,100]")
        self.quality = int(quality)
        self._stream = (self.directory / "timestamps.csv").open("x", newline="", encoding="utf8")
        self._writer = csv.DictWriter(self._stream, fieldnames=["camera", "sequence", "path", "monotonic_ns", "device_timestamp", "metadata_json"])
        self._writer.writeheader()
        self._stream.flush()
        self._counts, self._times = {}, {}
        self._lock = threading.Lock()

    def append(self, camera, image, monotonic_ns, device_timestamp=None, attrs=None):
        camera = _name(camera)
        if isinstance(monotonic_ns, bool) or not isinstance(monotonic_ns, (int, np.integer)) or monotonic_ns < 0:
            raise ValueError("monotonic_ns must be a nonnegative integer")
        if device_timestamp is not None and not math.isfinite(float(device_timestamp)):
            raise ValueError("Source timestamp must be finite or None")
        metadata = dict(attrs or {})
        if isinstance(image, (bytes, bytearray, memoryview)):
            payload = bytes(image)
            validated = metadata.get("receiver_validated_jpeg_sha256")
            if validated is not None:
                # ActualX5SensorSource has already decoded this exact frame for
                # the template view. Bind that validation to the original bytes
                # instead of decoding both 1080p/60 Hz streams again on disk.
                if not payload.startswith(b"\xff\xd8") or hashlib.sha256(payload).hexdigest() != validated:
                    raise ValueError("Receiver-validated JPEG checksum mismatch")
                metadata["jpeg_validation"] = "receiver_decode_and_writer_sha256"
            elif not payload.startswith(b"\xff\xd8") or cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR) is None:
                raise ValueError("Camera bytes must contain a valid JPEG")
            metadata["encoding"] = "source_jpeg_verbatim"
        else:
            image = np.asarray(image)
            if image.dtype != np.uint8 or image.ndim not in (2, 3) or (image.ndim == 3 and image.shape[2] != 3):
                raise ValueError("Wrist image must be uint8 gray/RGB")
            bgr = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            ok, buffer = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
            if not ok:
                raise OSError("JPEG encoding failed")
            payload = buffer.tobytes()
            metadata.update(encoding="jpeg_from_rgb", jpeg_quality=self.quality)
        metadata_text = _json(metadata)
        with self._lock:
            if monotonic_ns < self._times.get(camera, 0):
                raise ValueError("Camera host monotonic time moved backwards")
            index = self._counts.get(camera, 0)
            relative = Path(camera) / f"{index:08d}.jpg"
            path = self.directory / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as stream:
                stream.write(payload)
            self._writer.writerow({"camera": camera, "sequence": index, "path": relative.as_posix(),
                                   "monotonic_ns": monotonic_ns, "device_timestamp": device_timestamp,
                                   "metadata_json": metadata_text})
            self._stream.flush()
            self._counts[camera], self._times[camera] = index+1, monotonic_ns
            return relative.as_posix()

    def close(self):
        with self._lock:
            self._stream.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class RecordingFault(RuntimeError):
    pass


def async_recording_defaults():
    return {"queue_bytes":128*1024*1024,"queue_items":4096,"startup_timeout_s":15.,"close_timeout_s":30.}


def record_stress_defaults():
    return {"duration_s":1.,"control_hz":200.,"producer_hz":30.,"write_delay_s":.05,
            "dd_hz":27.,"force_hz":30.,"max_added_p95_s":.01,"image_shape":[1080,1920,3]}


def validate_record_stress_settings(cfg):
    """Validate explicit synthetic-load settings, never physical calibration."""
    options=cfg.get("recording",{}).get("stress",{})
    if not isinstance(options,dict):
        return ["recording.stress must be a mapping of explicit offline test settings"]
    errors=[]
    for name in ("duration_s","control_hz","producer_hz","dd_hz","force_hz","write_delay_s","max_added_p95_s"):
        value=options.get(name)
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value<=0:
            errors.append("recording.stress."+name+" must be positive finite")
    shape=options.get("image_shape")
    if not isinstance(shape,list) or len(shape)!=3 or shape[2]!=3 or any(type(value) is not int or value<=0 for value in shape):
        errors.append("recording.stress.image_shape requires [height,width,3] positive integers")
    if not errors and options["duration_s"]*options["control_hz"]<20:
        errors.append("recording.stress requires at least twenty wall-clock cycles")
    return errors


def dd_processing_metadata(provenance):
    """Normalize observed DD provenance without inventing missing calibration."""
    source = provenance.get("sensor_provenance", provenance)
    return {"service_name":source.get("dd_service_name", "unreported"),
            "service_version":source.get("dd_service_version", "unknown_not_transmitted"),
            "resolution":source.get("dd_encoded_shape"),
            "encoded_dtype":source.get("dd_encoded_dtype"),
            "decoded_dtype":source.get("dd_decoded_dtype"),
            "quantization":source.get("dd_decoding", "unreported"),
            "observed_encoding":source.get("dd_encoding_observed"),
            "native_dd":source.get("native_dd", False),
            "source_unit":source.get("dd_source_unit", "unknown")}


def finalize_gel_windows(folder, meta, cfg):
    """Retain only selected descriptor windows after all control has ended.

    Sensors acquire PROBE/HOLD candidates because the final stable and central
    windows are not known until those stages finish. This offline, atomic H5
    rewrite retains each finger's actual selected PROBE DD time span and HOLD
    central window. It never expands a short/failed window to fabricate frames.
    Frame arrays, dtype, source sequence and timestamp metadata remain exact.
    No file replacement occurs if the rewrite fails.
    """
    mode = cfg.get("record", {}).get("gel_raw", "windows")
    if mode not in ("off", "windows", "full"):
        raise ValueError("record.gel_raw must be off, windows or full")
    root=Path(folder)
    source_path=root if root.suffix == ".h5" else root/"streams.h5"
    report={"mode":mode,"selection":"per-finger selected PROBE DD time span and central HOLD window",
            "performed_after_control":True,"streams":{}}
    meta["gel_raw_recording"]=report
    if not source_path.exists():
        report["unavailable_reason"]="raw_stream_file_missing"
        return report
    keep={}
    with h5py.File(source_path,"r") as source:
        streams=source.get("streams",{})
        for side in ("left","right"):
            name="gel_raw_"+side
            if name not in streams:
                report["streams"][name]={"candidate_frames":0,"retained_frames":0,
                    "reason":"off" if mode=="off" else "no_source_frames"}
                continue
            group=streams[name]
            count=int(group.attrs.get("committed_count",len(group["monotonic_ns"])))
            times=np.asarray(group["monotonic_ns"][:count],dtype=np.int64)
            intervals=[]
            if mode=="windows":
                for stage in ("PROBE","HOLD"):
                    window=meta.get("descriptor_windows",{}).get(stage,{}).get(side,{})
                    frames=window.get("frames",[])
                    if not frames:
                        continue
                    if stage=="PROBE":
                        stamps=[int(frame["monotonic_ns"]) for frame in frames]
                        lo,hi=min(stamps),max(stamps)
                    else:
                        lo,hi=int(window["start_ns"]),int(window["end_ns"])
                    if lo>hi:raise ValueError("Invalid gel descriptor window")
                    intervals.append({"stage":stage,"start_ns":lo,"end_ns":hi,
                        "probe_not_settled":bool(window.get("probe_not_settled",False)),
                        "fallback_reason":window.get("fallback_reason")})
            mask=np.ones(count,dtype=bool) if mode=="full" else np.zeros(count,dtype=bool)
            for interval in intervals:
                mask|=(times>=interval["start_ns"])&(times<=interval["end_ns"])
            keep[name]=np.flatnonzero(mask)
            report["streams"][name]={"candidate_frames":count,"retained_frames":int(mask.sum()),
                "windows":intervals,"reason":"no_selected_descriptor_window" if mode=="windows" and not intervals else mode,
                "source_sequence_preserved":True}
    if mode=="full" or not keep:
        return report
    handle=tempfile.NamedTemporaryFile(prefix=source_path.name+".gel-filter-",suffix=".tmp",dir=source_path.parent,delete=False)
    temporary=Path(handle.name);handle.close()
    try:
        with h5py.File(source_path,"r") as source,h5py.File(temporary,"w") as target:
            for key,value in source.attrs.items():target.attrs[key]=value
            for key in source:
                if key!="streams":source.copy(key,target)
            target_streams=target.create_group("streams")
            for key,value in source["streams"].attrs.items():target_streams.attrs[key]=value
            for name,group in source["streams"].items():
                if name not in keep:
                    source.copy(group,target_streams,name=name)
                    continue
                selected=keep[name]
                out=target_streams.create_group(name)
                for key,value in group.attrs.items():out.attrs[key]=value
                out.attrs["committed_count"]=len(selected)
                out.attrs["post_recording_selection"]="gel_descriptor_windows_v1"
                for key,dataset in group.items():
                    if not isinstance(dataset,h5py.Dataset):
                        source.copy(dataset,out,name=key)
                        continue
                    options={"dtype":dataset.dtype,"shape":(len(selected),*dataset.shape[1:]),
                             "maxshape":dataset.maxshape,"chunks":dataset.chunks,
                             "compression":dataset.compression,"compression_opts":dataset.compression_opts,
                             "shuffle":dataset.shuffle,"fletcher32":dataset.fletcher32}
                    if dataset.scaleoffset is not None:options["scaleoffset"]=dataset.scaleoffset
                    saved=out.create_dataset(key,**options)
                    for dest_index,source_index in enumerate(selected):saved[dest_index]=dataset[int(source_index)]
                    for attr,value in dataset.attrs.items():saved.attrs[attr]=value
            target.flush()
        os.replace(temporary,source_path)
    finally:
        if temporary.exists():temporary.unlink()
    return report


def _writer_process(path,requests,status,quality,test_options):
    """The only owner of compression, JPEG encoding, H5 and metadata writes."""
    from .session import atomic_json
    root=Path(path);h5=jpg=None;count=0
    manifest={"complete":False,"state":"open","committed_messages":0}
    try:
        root.mkdir(parents=True,exist_ok=True)
        atomic_json(root/"recording_status.json",manifest)
        # LZF is built into h5py, lossless and substantially faster for gel
        # images than per-frame DEFLATE. Payload dtype/shape and commit-before-
        # acknowledgement ordering stay unchanged; no sampling/queue expansion.
        h5=H5Writer(root/"streams.h5",compression="lzf");jpg=JpegWriter(root/"images",quality=quality)
        status.put(("ready",None))
        pending=None
        while True:
            message=pending if pending is not None else requests.get()
            pending=None
            if message[0]=="close":
                error=message[1]
                h5.close();jpg.close();h5=jpg=None
                manifest.update(complete=not error,state="incomplete" if error else "complete",
                                error=error,committed_messages=count)
                atomic_json(root/"recording_status.json",manifest)
                status.put(("closed",manifest));return
            _,sequence,size,payload=message
            if test_options.get("delay_s",0):time.sleep(test_options["delay_s"])
            if test_options.get("fail_after") is not None and count>=test_options["fail_after"]:
                raise OSError("Injected writer failure")
            kind,args=pickle.loads(payload)
            if kind=="h5":
                batch=[(sequence,size,args)];batch_bytes=size
                # Bounded draining, no waiting for more samples. Non-H5 writes
                # and close remain ordered barriers. Test fault injection keeps
                # its original one-message semantics.
                if not test_options:
                    while len(batch)<128 and batch_bytes<8*1024*1024:
                        try:following=requests.get_nowait()
                        except queue.Empty:break
                        if following[0]!='write':pending=following;break
                        _,seq2,size2,payload2=following
                        kind2,args2=pickle.loads(payload2)
                        if kind2!='h5':pending=following;break
                        batch.append((seq2,size2,args2));batch_bytes+=size2
                h5.append_many([row[2] for row in batch])
                count+=len(batch)
                for seq2,size2,_ in batch:status.put(('ack',(seq2,size2)))
                continue
            elif kind=="jpg":
                expected=args[-1];actual=jpg.append(*args[:-1])
                if actual!=expected:raise OSError("Deterministic JPEG filename order mismatch")
            elif kind=="json":atomic_json(root/args[0],args[1])
            else:raise ValueError("Unknown writer command "+kind)
            count+=1
            status.put(("ack",(sequence,size)))
    except BaseException as exc:
        error=type(exc).__name__+": "+str(exc)
        try:
            manifest.update(state="incomplete",complete=False,error=error,committed_messages=count)
            atomic_json(root/"recording_status.json",manifest)
        except Exception:pass
        status.put(("error",error))
    finally:
        for writer in (h5,jpg):
            if writer is not None:
                try:writer.close()
                except Exception:pass


class _AsyncH5:
    def __init__(self,owner):self.owner=owner
    def append(self,stream,data,monotonic_ns,device_timestamp=None,attrs=None):
        _name(stream)
        with self.owner._lock:
            index=self.owner._stream_counts.get(stream,0)
            self.owner._submit_locked("h5",(stream,data,monotonic_ns,device_timestamp,attrs))
            self.owner._stream_counts[stream]=index+1
            return index
    def close(self):return self.owner.close()


class _AsyncJpeg:
    def __init__(self,owner):self.owner=owner
    def append(self,camera,image,monotonic_ns,device_timestamp=None,attrs=None):
        camera=_name(camera)
        with self.owner._lock:
            index=self.owner._camera_counts.get(camera,0)
            relative=(Path(camera)/f"{index:08d}.jpg").as_posix()
            self.owner._submit_locked("jpg",(camera,image,monotonic_ns,device_timestamp,attrs,relative))
            self.owner._camera_counts[camera]=index+1
            return relative
    def close(self):return self.owner.close()


class AsyncRecorder:
    """Bounded asynchronous persistence, with exact serialized-byte accounting.

    Submission serializes an immutable message, reserves its byte size, and
    calls put_nowait. It never waits for disk/encoding/worker progress. A short
    in-process mutex keeps concurrent source and stage events ordered; the
    worker never owns this mutex. Bytes remain reserved while in flight until
    the worker acknowledges a committed write. Queue exhaustion, worker death
    and write errors latch a fault; no samples are silently dropped.

    ``close`` waits for accepted writes and must be used only after active
    motion/control has ended. The child leaves an incomplete manifest even if
    the parent disappears before graceful close.
    """
    def __init__(self,path,cfg=None,*,settings=None,_test_options=None):
        self.path=Path(path)
        opts=dict(settings or (cfg or {}).get("recording",{}).get("writer",async_recording_defaults()))
        for key in ("queue_bytes","queue_items"):
            if type(opts.get(key)) is not int or opts[key]<=0:raise ValueError("writer."+key+" must be positive integer")
        for key in ("startup_timeout_s","close_timeout_s"):
            if not isinstance(opts.get(key),(float,int)) or not math.isfinite(opts[key]) or opts[key]<=0:
                raise ValueError("writer."+key+" must be positive finite")
        self.settings=opts;self._lock=threading.RLock();self._error=None;self._closed=False;self._closing=False
        self._ready=False;self._closed_ack=False;self._bytes=0;self._sequence=0;self._camera_counts={};self._stream_counts={}
        self._inflight={};self._submit_times=deque(maxlen=4096);self._max_bytes=0;self._manifest=None
        context=mp.get_context("spawn")
        self._requests=context.Queue(maxsize=opts["queue_items"])
        self._status=context.Queue()
        quality=(cfg or {}).get("output",{}).get("jpeg_quality",95)
        self._process=context.Process(target=_writer_process,args=(str(self.path),self._requests,self._status,quality,
                                      dict(_test_options or {})),daemon=True,name="bottle_disk_writer")
        self.h5=_AsyncH5(self);self.jpg=_AsyncJpeg(self)
        self._process.start()
        deadline=time.monotonic()+opts["startup_timeout_s"]
        try:
            while not self._ready:
                self.check()
                if time.monotonic()>=deadline:raise RecordingFault("Writer startup timed out")
                time.sleep(.005)
        except BaseException:
            if self._process.is_alive():self._process.terminate();self._process.join(timeout=1)
            self._requests.cancel_join_thread();self._requests.close();self._status.close()
            raise

    def _poll(self):
        if self._closed:return
        while True:
            try:kind,value=self._status.get_nowait()
            except queue.Empty:break
            if kind=="ready":self._ready=True
            elif kind=="ack":
                sequence,size=value
                reserved=self._inflight.pop(sequence,None)
                if reserved!=size:self._error=self._error or "Writer acknowledgement accounting mismatch"
                else:self._bytes-=size
            elif kind=="error":self._error=self._error or "Disk writer failed: "+value
            elif kind=="closed":self._closed_ack=True;self._manifest=value
        if not self._process.is_alive() and not self._closed_ack and not self._closed:
            self._error=self._error or "Disk writer process exited before successful close"

    def check(self):
        with self._lock:
            self._poll()
            if self._error:raise RecordingFault(self._error)

    @property
    def error(self):
        try:self.check()
        except RecordingFault as exc:return exc
        return None

    def _submit_locked(self,kind,args):
        started=time.monotonic_ns()
        self.check()
        if self._closed or self._closing:raise RecordingFault("Recorder is already closing/closed")
        # pickle performs an immutable payload copy. JPEG compression and H5
        # conversion never run here; payload dtype/shape survive unchanged.
        payload=pickle.dumps((kind,args),protocol=5)
        size=len(payload)+128 # envelope/accounting reserve; payload length is exact
        if self._bytes+size>self.settings["queue_bytes"]:
            self._error="Recording byte queue full; acquisition incomplete, no blocking or silent drop"
            raise RecordingFault(self._error)
        sequence=self._sequence
        try:self._requests.put_nowait(("write",sequence,size,payload))
        except queue.Full as exc:
            self._error="Recording message queue full; acquisition incomplete"
            raise RecordingFault(self._error) from exc
        self._sequence+=1;self._inflight[sequence]=size;self._bytes+=size;self._max_bytes=max(self._max_bytes,self._bytes)
        self._submit_times.append((time.monotonic_ns()-started)/1e6)
        return sequence

    def _submit(self,kind,args):
        with self._lock:return self._submit_locked(kind,args)

    def json(self,relative,value):
        _name(str(relative))
        return self._submit("json",(str(relative),value))

    def fail(self, reason):
        """Latch an upstream acquisition failure without waiting for disk.

        A source/thread failure is just as incomplete as a failed write. The
        caller can invoke this before its safety response, then close only
        after motion/control has ended; close preserves all accepted samples
        while leaving the manifest explicitly incomplete.
        """
        with self._lock:
            self._error = self._error or "Acquisition incomplete: " + str(reason)

    def stats(self):
        with self._lock:
            self._poll()
            values=np.asarray(self._submit_times)
            return {"writer_pid":self._process.pid,"submitted_messages":self._sequence,
                    "outstanding_bytes":self._bytes,"peak_outstanding_bytes":self._max_bytes,
                    "byte_limit":self.settings["queue_bytes"],"fault":self._error,
                    "submit_latency_ms":{"window":"latest_at_most_4096_submissions","count":len(values),"mean":float(values.mean()) if len(values) else None,
                        "p95":float(np.percentile(values,95)) if len(values) else None,"max":float(values.max()) if len(values) else None}}

    def close(self, error=None):
        if error is not None:
            self.fail(error)
        if self._closed:
            if self._error:raise RecordingFault(self._error)
            return
        self._closing=True
        deadline=time.monotonic()+self.settings["close_timeout_s"]
        sent=False
        while time.monotonic()<deadline:
            with self._lock:self._poll()
            if self._closed_ack or not self._process.is_alive():break
            if not sent:
                try:self._requests.put_nowait(("close",self._error));sent=True
                except queue.Full:pass
            time.sleep(.005)
        self._process.join(timeout=max(0.,deadline-time.monotonic()))
        with self._lock:self._poll()
        if self._process.is_alive():
            self._error=self._error or "Writer close timeout; recording remains incomplete"
            self._process.terminate();self._process.join(timeout=1.)
        elif not self._closed_ack:self._error=self._error or "Writer closed without completion acknowledgement"
        self._closed=True
        self._requests.cancel_join_thread();self._requests.close();self._status.close()
        if self._error:raise RecordingFault(self._error)

    def __enter__(self):return self
    def __exit__(self,exception_type,exception,traceback):
        if exception is None:
            self.close()
        else:
            # Preserve the original exception while still marking source or
            # producer errors as incomplete; do not replace it with close's
            # RecordingFault acknowledgement of the same failure.
            try:self.close(error=exception)
            except RecordingFault:pass
