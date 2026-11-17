"""Verify field coordinate sidecars once on ingestion; losing a label never loses the trial."""
import hashlib
import json
from pathlib import Path
from .evidence import sha,measured

def verified_file(item):
    p=Path(item['path'])
    if not p.is_absolute() or not p.is_file():return False
    h=hashlib.sha256()
    with p.open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''):h.update(block)
    return h.hexdigest()==item['sha256']

def check_measurement(label,m,phase,stamp):
    try:
        if measured(label,m['reference_digest'],m['target_definition_id'],phase=phase,stamp=stamp,axis=m['axis_tcp']) is None:return False
        r=label['record']
        # Export from the dated instrument/TCP recorder. Cross-check data, not merely a path string.
        envelope=json.loads(Path(r['raw_source']).read_text())
        payload={k:v for k,v in r.items() if k!='raw_source'}
        return (envelope['measurement']==payload and bool(envelope['acquisition_files']) and
                all(verified_file(f) for f in envelope['acquisition_files']))
    except (KeyError,OSError,ValueError,TypeError):return False


def check_reply(reply):
    try:
        sidecar=Path(reply['raw_source'])
        if not sidecar.is_absolute():return False
        raw=json.loads(sidecar.read_text())
        return raw['reply']=={k:v for k,v in reply.items() if k!='raw_source'} and bool(raw['acquisition_files']) and all(verified_file(f) for f in raw['acquisition_files'])
    except (KeyError,OSError,ValueError,TypeError):return False
