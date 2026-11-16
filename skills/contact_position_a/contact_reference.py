"""A fixed physical grasp reference; world TCPs only enter the legacy label importer."""
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from bottle_scan.contact_policy import feature_vector
from bottle_scan.state_graph.common import digest

SCHEMA = 'fixed_contact_reference_v1'
FEATURE = 'contact_descriptor_v1'
FRAME = 'tcp_at_action_start'
SIGN = 'hand_displacement_from_standard_along_positive_tool_a_mm'


def probe_protocol(cfg):
    f = cfg['force']
    return {'axes_tcp': cfg['axes_tcp'], 'arm': cfg['arm'], 'tactile': cfg['tactile'],
            'force': {k: f[k] for k in ('unit', 'normal_axis', 'normal_signs', 'target',
                                        'probe_fraction', 'probe_tolerance', 'probe_stable_s')}}


def validate_reference(ref):
    if (ref['schema'] != SCHEMA or ref['supported_axes'] != ['a']
            or ref['finger_order'] != ['left', 'right'] or ref['label_semantics'] != SIGN
            or ref['feature_version'] != FEATURE or not ref['target_definition_id']
            or ref['reference_digest'] != digest({k:v for k,v in ref.items() if k!='reference_digest'})):
        raise ValueError('Incompatible fixed contact reference')
    axes = np.column_stack([ref['protocol']['axes_tcp'][k] for k in ('a', 'c', 'l')])
    if axes.shape != (3,3) or not np.allclose(axes.T @ axes, np.eye(3), atol=1e-6):
        raise ValueError('Invalid tool axes')
    if not ref['standard_sample'].get('accepted_unix_ns'):
        raise ValueError('Missing operator-accepted reference contact')
    feature_vector(ref['standard_sample']['descriptors'])
    return ref


def from_accepted_sample(path, target_id, object_region):
    raw = Path(path).read_bytes()
    meta = json.loads(raw)
    p = meta.get('initial_centroids') or {}
    if meta.get('simulated') is not False or not p.get('valid') or not meta.get('T0_accepted_unix_ns'):
        raise ValueError('Need an explicitly accepted real bilateral reference PROBE')
    if any(not d.get('valid') or d.get('touches_edge') for d in p['descriptors']):
        raise ValueError('Invalid or edge reference')
    ref = dict(schema=SCHEMA, version=1, target_definition_id=target_id, object_region=object_region,
               finger_order=['left','right'], supported_axes=['a'], label_semantics=SIGN, units='mm',
               feature_version=FEATURE, protocol=probe_protocol(meta['config']),
               standard_sample={'source': str(path), 'meta_sha256': sha256(raw).hexdigest(),
                                'timestamp_ns': p['timestamp_ns'], 'accepted_unix_ns': meta['T0_accepted_unix_ns'],
                                'descriptors': deepcopy(p['descriptors'])})
    ref['reference_digest'] = digest(ref)
    return validate_reference(ref)


def local_training_sessions(sessions, ref, manifest):
    """Witness for each *local* legacy zero, never compare world poses across placements."""
    validate_reference(ref)
    if manifest['reference_digest'] != ref['reference_digest']:
        raise ValueError('Label manifest/reference mismatch')
    for s in sessions:
        witness = manifest['measured_A_sessions'].get(s['metadata_sha256'], {})
        if (not witness.get('standard_grasp_evidence') or not witness.get('placement_approach_group')
                or probe_protocol(s['meta']['config']) != ref['protocol']):
            raise ValueError('Unverified standard grasp or probe protocol for '+s['path'])

