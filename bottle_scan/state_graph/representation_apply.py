"""Instantiate the selected feature recipe identically for graph rebuild and runtime."""
from copy import deepcopy
import json
from pathlib import Path
from .common import digest
from .representation_eval import raw_candidate


def apply_representation(rows,spec,name,*,use_feature_file=True):
    result=deepcopy(spec);obs=result['observation'];rc=obs['representations']
    if name not in rc['candidates']:raise ValueError('Unknown representation: '+name)
    cfg=obs['modalities'];selected={};parts={}
    saved={}
    if use_feature_file and rc.get('feature_file'):
        with Path(rc['feature_file']).open() as handle:
            for line in handle:
                f=json.loads(line);saved[f['node_id']]=f
    methods=name.split('+')
    for method in methods:
        if method in ('legacy_geometry','calibrated_rh'):
            if method=='calibrated_rh' and not obs.get('tac_calibration'):
                raise ValueError('Calibrated geometry selection requires the finger-plane calibration')
            obs['tac_version']=method
            selected['tac']={'extractor':'bottle_scan.state_graph.modalities.contact_geometry:contact_geometry',
                             'version':'two_contact_centroid_area_px_v1' if method=='legacy_geometry' else 'calibrated_rh:'+digest(obs['tac_calibration'])}
            parts['tac']=method
        elif method=='force':
            if 'force' not in cfg:raise ValueError('Selected force modality has no declared extractor')
            selected['force']=cfg['force'];parts['force']=method
        else:
            if not result.get('tactile_encoder'):raise ValueError('Selected encoder requires a configured frozen artifact')
            readout=obs['tac_embed_readout'] if method=='tac_metric' else method
            obs['tac_embed_readout']=readout
            key=obs['metric']['input_modality']
            selected[key]={'extractor':'bottle_scan.state_graph.modalities.tactile_embedding:embedding','encoder_features':True}
            parts[key]=readout
    # Under joint combination all encoder covariance is estimated in a certified two-dimensional head.
    obs['tac_metric']=any(k in methods for k in ('tac_metric','grid','attention','soft_argmax','mean'))
    obs['modalities']=selected;rc['current']=name;obs['applied_representation']=name
    inputs=list(selected);result.setdefault('dynamics',{})['inputs']=inputs
    if result['dynamics'].get('targets'):
        targets={k:v for k,v in result['dynamics']['targets'].items() if k in selected and k!='tac'}
        if 'tac' in selected:targets['tac']=list(range(6 if obs['tac_version']=='legacy_geometry' else 14))
        result['dynamics']['targets']=targets
    converted=[]
    for original in rows:
        r=dict(original)
        if original['node_id'] in saved:r.update({k:v for k,v in saved[r['node_id']].items() if k in ('representations','representation_versions')})
        values={};versions={}
        for key,method in parts.items():
            v=raw_candidate(r,method)
            if v is None:raise ValueError(f'Missing {method} for {r["node_id"]}; compute recorded-window features before selecting')
            ver=(r.get('representation_versions') or {}).get(method)
            if not ver:ver=(r.get('extractor_versions') or {}).get(key) if method not in r.get('representations',{}) else selected[key].get('version')
            if not ver:raise ValueError(f'Missing extractor identity for {method} of {r["node_id"]}')
            values[key]=v;versions[key]=ver
        converted.append({**r,'obs':values,'extractor_versions':versions})
    return converted,result
