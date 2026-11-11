"""Read-only screening for ambiguous contact observations, never model training.

A candidate means overlapping measured features with different *geometric*
labels. It is not proof that the physically optimal recovery actions differ.
Sessions, bottles, protocols, conditions and initial/regrasp observations remain
separate. Repeated frames from one trial never count as independent trials.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from .contact_policy import feature_vector
from .agentic_learning import _pose, _rotation_deg, _signature

SCHEMA = 'ambiguity_report_v1'
DEFAULTS = dict(action_gap_mm=4., action_gap_deg=4., repeat_position_mm=1.,
                repeat_angle_deg=1., noise_quantile=.5, min_repeat_dof=6,
                min_repeat_cells=2, min_repeats=3, tracking_tolerance_mm=2.,
                tracking_tolerance_deg=1., max_pairs=1000, read_h5=False)
SUPPORTED = ('e1e2_session_v1', 'ambiguity_session_v1')


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _finite(value, shape=None):
    try:
        a = np.asarray(value, float)
        return a if a.size and np.isfinite(a).all() and (shape is None or a.shape == shape) else None
    except (TypeError, ValueError):
        return None


def discover(sources, recursive=False):
    """Explicit snapshot files or session folders; recursive discovery is opt-in."""
    found = []
    for item in sources:
        p = Path(item).expanduser().resolve()
        if p.is_file():
            found.append(p)
        elif (p / 'session_meta.json').is_file():
            found.append(p)
        elif recursive and p.is_dir():
            found.extend(x.parent for x in p.rglob('session_meta.json'))
        else:
            found.append(p)  # Ingest reports missing input; never silently ignore it.
    return sorted(set(found), key=str)


def _session_id(meta, source):
    # Export copies keep this creation time and T0; path-independent dedup.
    if meta.get('created_unix_ns') is not None:
        return _hash({k: meta.get(k) for k in ('schema', 'created_unix_ns', 'T0', 'cond')})[:20]
    return _hash({'meta': meta, 'source': str(source)})[:20]


def _load_sources(sources, recursive):
    records, inventory, issues, duplicates = [], [], [], []
    seen = {}
    for source in discover(sources, recursive):
        try:
            if source.is_file():
                bundle = json.loads(source.read_text())
                if 'trials' not in bundle or not isinstance(bundle.get('session_meta'), dict):
                    raise ValueError('expected data_snapshot with session_meta and trials')
                meta, rows = bundle['session_meta'], bundle['trials']
                base = Path(bundle.get('source_root', source.parent))
                entries = [(base / 'trials' / str(r.get('id', 'unknown')), r, False) for r in rows]
                source_type = 'exported_snapshot'
            else:
                meta = json.loads((source / 'session_meta.json').read_text())
                entries = []
                for folder in ('trials', 'interrupted'):
                    for p in sorted((source / folder).glob('*/meta.json')):
                        try:
                            entries.append((p.parent, json.loads(p.read_text()), folder == 'interrupted'))
                        except (OSError, ValueError) as exc:
                            issues.append({'path': str(p), 'reason': str(exc)})
                base, source_type = source, 'session_directory'
            if meta.get('schema') not in SUPPORTED:
                raise ValueError('unsupported session schema: ' + str(meta.get('schema')))
            sid = _session_id(meta, base)
            inventory.append({'session_id': sid, 'source': str(source), 'type': source_type,
                              'trials_found': len(entries), 'simulated': meta.get('simulated'),
                              'schema': meta['schema'], 'session_role': meta.get('agentic', {}).get('role', 'unspecified')})
            for folder, trial, archived in entries:
                key = (sid, str(trial.get('id')), trial.get('started_unix_ns'), archived)
                rec = dict(trial=trial, meta=meta, folder=folder, session_root=base,
                           session_id=sid, source=str(source), source_type=source_type, archived=archived)
                if key in seen:
                    prior = seen[key]
                    conflict = _hash(prior['trial']) != _hash(trial)
                    if conflict:
                        prior['metadata_conflict'] = True
                        rec['metadata_conflict'] = True
                    duplicates.append({'session_id': sid, 'trial_id': trial.get('id'), 'duplicate_source': str(source),
                                       'different_metadata': _hash(prior['trial']) != _hash(trial)})
                    # Prefer original hierarchy with H5/recording evidence over export.
                    if source_type == 'session_directory' and prior['source_type'] != source_type:
                        records[records.index(prior)] = rec
                        seen[key] = rec
                    continue
                seen[key] = rec
                records.append(rec)
        except (OSError, ValueError, TypeError, KeyError) as exc:
            issues.append({'path': str(source), 'reason': str(exc)})
    return records, inventory, issues, duplicates


def _pooled_depth(array, channel):
    arr = np.asarray(array, float)
    if arr.ndim == 3:
        arr = arr[..., channel]
    if arr.ndim != 2 or min(arr.shape) < 4 or not np.isfinite(arr).all():
        raise ValueError('invalid DD dimensions/data')
    return [float(b.mean()) for row in np.array_split(arr, 4, axis=0)
            for b in np.array_split(row, 4, axis=1)]


def _npz_observation(rec, probe):
    result, errors = {}, []
    name = probe.get('observation_file')
    if not name:
        return result, errors
    path = (rec['folder']/name).resolve()
    if path.parent != rec['folder'].resolve():
        return result, ['observation_file must be a local trial file']
    try:
        with np.load(path, allow_pickle=False) as data:
            channel = int(rec['meta'].get('config', {}).get('tactile', {}).get('depth_channel', 0))
            pool = []
            for side in ('left', 'right'):
                arr = np.asarray(data['dd_'+side], float)
                ref = np.asarray(data['reference_'+side], float)
                if arr.shape != ref.shape:
                    raise ValueError('DD and reference shapes differ')
                pool.extend(_pooled_depth(arr-ref, channel))
            result['dd_zeroed_pool'] = pool
            wrench = _finite(data.get('wrench_zeroed'), (2,6))
            if wrench is not None:
                result['wrench_zeroed'] = wrench.tolist()
            result['raw_provenance'] = {'kind':'exact_saved_probe_median_npz', 'path':str(path),
                'dd_reference_subtracted':True, 'wrench_timing':probe.get('wrench_timing', 'unspecified')}
    except (OSError, ValueError, KeyError, IndexError) as exc:
        errors.append(str(exc))
    return result, errors


def _reference_arrays(rec):
    zero = rec['trial'].get('zero', {})
    references = []
    for candidate in rec['meta'].get('zero_history', []):
        if zero.get('created_unix_ns') is not None and candidate.get('created_unix_ns') == zero['created_unix_ns']:
            references.append(candidate.get('dd_reference_file'))
    if zero.get('dd_reference_file'):
        references.append(zero['dd_reference_file'])
    references = sorted(set(x for x in references if isinstance(x,str)))
    if len(references) != 1:
        return None, 'no unique DD reference tied to this zero calibration'
    ref = Path(references[0])
    # A remote absolute path in metadata is portable only via its recorded
    # baseline suffix. Never read arbitrary external filesystem locations.
    parts = ref.parts
    if 'baseline' not in parts:
        return None, 'DD reference path has no baseline component'
    suffix = Path(*parts[parts.index('baseline'):])
    local = (rec['session_root']/suffix).resolve()
    if rec['session_root'].resolve() not in local.parents:
        return None, 'DD reference escapes session root'
    try:
        with np.load(local, allow_pickle=False) as data:
            refs = [np.asarray(data[side],float) for side in ('left','right')]
        return refs, str(local)
    except (OSError,ValueError,KeyError) as exc:
        return None, str(exc)


def _raw_window(rec, probe, dd=False):
    """Exact saved DD frame timestamps; bounded aligned pose/wrench windows."""
    result, errors = _npz_observation(rec, probe)
    path = rec['folder'] / 'streams.h5'
    if not path.is_file():
        return result, errors + ([] if result else ['streams.h5 unavailable'])
    try:
        import h5py
        with h5py.File(path, 'r') as h:
            result['raw_provenance_h5'] = {'path':str(path),
                'diagnostic_subset':bool(h.attrs.get('diagnostic_subset',False)),
                'source_path':str(h.attrs.get('source_path',path)),
                'dd_reference_subtracted':False}
            def samples(name, stamps, nearest=False, exact=False):
                g = h['streams/' + name]
                n = min(int(g.attrs.get('committed_count', len(g['data']))), len(g['data']), len(g['monotonic_ns']))
                ts = np.asarray(g['monotonic_ns'][:n], np.int64)
                stamps = np.asarray(stamps, np.int64)
                if not len(stamps):
                    raise ValueError('no independent PROBE frame timestamps')
                if exact:
                    stamps = np.unique(stamps)
                    idx = np.flatnonzero(np.isin(ts, stamps))
                    if len(idx) != len(stamps) or set(ts[idx].tolist()) != set(stamps.tolist()):
                        raise ValueError('missing or duplicate exact PROBE DD timestamps: '+name)
                else:
                    idx = np.flatnonzero((ts >= stamps.min()) & (ts <= stamps.max()))
                if nearest and not len(idx) and len(ts):
                    j = int(np.argmin(np.abs(ts - int(np.median(stamps)))))
                    if abs(int(ts[j]) - int(np.median(stamps))) <= 50_000_000:
                        idx = np.asarray([j])
                if not len(idx):
                    raise ValueError('no committed matching PROBE frames: ' + name)
                if not exact and len(idx) > 64:
                    idx = idx[np.linspace(0, len(idx) - 1, 64).astype(int)]
                return np.asarray(g['data'][idx.tolist()], float)
            stamps = [t for w in probe.get('window', []) for t in w.get('frame_ns', [])]
            if not stamps and probe.get('timestamp_ns'):
                stamps = [probe['timestamp_ns']]
            try:
                poses = samples('tcp_measured', stamps, nearest=True)
                good = [p for p in poses if _pose(p) is not None]
                if good:
                    p = good[-1].copy()
                    p[:3, 3] = np.median(np.asarray(good)[:, :3, 3], axis=0)
                    result['measured_tcp_mm'] = p.tolist()
                    result['pose_spread_mm'] = float(np.max(np.linalg.norm(np.asarray(good)[:, :3, 3] - p[:3, 3], axis=1)))
            except (KeyError, ValueError) as exc:
                errors.append(str(exc))
            offsets = _finite(rec['trial'].get('zero',{}).get('offset_raw'), (2,6))
            if offsets is not None:
                try:
                    wrench = np.stack([np.median(samples('wrench_raw_'+side, stamps, nearest=True),axis=0)
                                       for side in ('left','right')])
                    if wrench.shape != (2,6) or not np.isfinite(wrench).all():
                        raise ValueError('invalid raw wrench')
                    result['wrench_zeroed'] = (wrench-offsets).tolist()
                    result['wrench_timing'] = 'independent_sensor_medians_within_PROBE_time_window'
                except (KeyError,ValueError) as exc:
                    errors.append(str(exc))
            if dd:
                pools, zeroed_pools, spreads = [], [], []
                references, reference_note = _reference_arrays(rec)
                result['dd_reference_note'] = reference_note
                channel = int(rec['meta'].get('config', {}).get('tactile', {}).get('depth_channel', 0))
                for finger, name in enumerate(('left', 'right')):
                    try:
                        win = probe.get('window', [])[finger]['frame_ns']
                        arr = samples('dd_' + name, win, exact=True)
                        pooled = np.asarray([_pooled_depth(frame,channel) for frame in arr])
                        # Same operation order as runtime: median DD then spatial pool.
                        median = np.median(arr,axis=0)
                        pools.extend(_pooled_depth(median,channel))
                        if references is not None:
                            if references[finger].shape != median.shape:
                                raise ValueError('DD reference shape mismatch')
                            zeroed_pools.extend(_pooled_depth(median-references[finger],channel))
                        spreads.append(float(np.median(np.std(pooled, axis=0))))
                    except (KeyError, ValueError, IndexError) as exc:
                        errors.append('dd_' + name + ': ' + str(exc))
                if len(pools) == 32:
                    result['raw_dd_pool_unzeroed'] = pools
                    result['raw_dd_temporal_sd'] = spreads
                if len(zeroed_pools) == 32:
                    result['dd_zeroed_pool'] = zeroed_pools
                    result['raw_provenance_h5']['dd_reference_subtracted'] = True
                    result['raw_provenance_h5']['dd_reference_file'] = reference_note
    except (ImportError, OSError, ValueError) as exc:
        errors.append(str(exc))
    return result, errors


def _geometry(rec, probe, role, settings, raw):
    trial, meta = rec['trial'], rec['meta']
    perturb = trial.get('perturbation')
    if isinstance(perturb, dict):
        planned = _finite([perturb.get('a_mm', 0), perturb.get('l_mm', 0), perturb.get('tilt_deg', 0)], (3,))
    else:
        planned = _finite([trial.get('delta_mm'), 0, 0], (3,))
    T0 = _pose(meta.get('T0'))
    p = _pose(probe.get('measured_tcp_mm'))
    source = 'measured_probe_inline'
    if p is None:
        p = _pose(raw.get('measured_tcp_mm'))
        source = 'measured_probe_h5'
    axes = meta.get('config', {}).get('axes_tcp', {})
    a, l = _finite(axes.get('a'), (3,)), _finite(axes.get('l'), (3,))
    measured, issues = None, []
    if p is not None and T0 is not None and a is not None and l is not None:
        if abs(np.linalg.norm(a)-1) > 1e-3 or abs(np.linalg.norm(l)-1) > 1e-3 or abs(a@l) > 1e-3:
            issues.append('invalid_axis_definition')
        else:
            delta = T0[:3, :3].T @ (p[:3, 3] - T0[:3, 3])
            R = T0[:3, :3].T @ p[:3, :3]
            tilt_axis = meta.get('ambiguity_design', {}).get('tilt_axis', 'c')
            axis = _finite(axes.get(tilt_axis), (3,))
            angle = _rotation_deg(np.eye(3), R)
            skew = np.asarray([R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0]-R[0,1]]) / 2
            if axis is not None:
                signed_angle = float(np.degrees(np.arctan2(skew@axis, (np.trace(R)-1)/2)))
                if angle-abs(signed_angle) > settings['tracking_tolerance_deg']:
                    issues.append('rotation_outside_declared_axis')
            else:
                signed_angle = 0.
                if angle > settings['tracking_tolerance_deg']:
                    issues.append('rotation_without_declared_axis')
            measured = np.asarray([delta@a, delta@l, signed_angle])
            cross = float(np.linalg.norm(delta-a*(delta@a)-l*(delta@l)))
            if cross > settings['tracking_tolerance_mm']:
                issues.append('unplanned_cross_axis_motion')
            if meta['schema'] == 'e1e2_session_v1' and (abs(measured[1]) > settings['tracking_tolerance_mm'] or angle > settings['tracking_tolerance_deg']):
                issues.append('outside_legacy_1d_protocol')
    if raw.get('pose_spread_mm', 0) > settings['repeat_position_mm']:
        issues.append('tcp_unstable_during_probe')
    # B corrections must not inherit the initial delta as the final label.
    if role != 'initial' and measured is None:
        return None, planned, 'unknown_regrasp_pose', ['regrasp_actual_pose_unavailable']
    if measured is not None and planned is not None and role == 'initial':
        if np.linalg.norm(measured[:2]-planned[:2]) > settings['tracking_tolerance_mm'] or abs(measured[2]-planned[2]) > settings['tracking_tolerance_deg']:
            issues.append('command_feedback_disagreement')
    return measured if measured is not None else planned, planned, source if measured is not None else 'nominal_only', issues


def observations(records, settings):
    obs, exclusions = [], Counter()
    for rec in records:
        row, meta = rec['trial'], rec['meta']
        if meta.get('simulated') is True:
            exclusions['simulated_trial'] += 1
            continue
        if row.get('bottle_moved') is True:
            exclusions['operator_confirmed_bottle_moved'] += 1
            continue
        if rec['archived'] or not row.get('complete') or row.get('label') == 'invalid':
            exclusions['incomplete_archived_or_invalid_trial'] += 1
            continue
        if not row.get('recording_complete'):
            exclusions['recording_incomplete'] += 1
            continue
        status_file = rec['folder'] / 'recording_status.json'
        recording_verified = False
        if status_file.is_file():
            try:
                recording_verified = bool(json.loads(status_file.read_text()).get('complete'))
                if not recording_verified:
                    exclusions['recording_status_incomplete'] += 1
                    continue
            except (OSError, ValueError):
                exclusions['recording_status_invalid'] += 1
                continue
        probes = [('initial', row.get('probe_initial') or {})]
        if row.get('cond') == 'B':
            probes += [('regrasp', p.get('probe') or {}) for p in row.get('probe_regrasps', [])]
        for index, (role, probe) in enumerate(probes):
            if not probe.get('valid') or probe.get('stable_probe') is False:
                exclusions['invalid_or_missing_' + role + '_probe'] += 1
                continue
            ds = probe.get('descriptors', [])
            try:
                descriptor = feature_vector(ds)
                centroid = np.asarray([d['centroid_px'] for d in ds], float).ravel()
            except (ValueError, KeyError, TypeError):
                exclusions['invalid_descriptors'] += 1
                continue
            raw, raw_issues = _raw_window(rec, probe, dd=True) if settings['read_h5'] else ({}, [])
            geometry, planned, source, issues = _geometry(rec, probe, role, settings, raw)
            if rec.get('metadata_conflict'):
                issues.append('conflicting_metadata_copies')
            if row.get('label') in ('pushed','tilted','tipped'):
                issues.append('operator_label_possible_object_motion')
            if geometry is None:
                exclusions['missing_geometric_label'] += 1
                continue
            features = {'centroid': centroid.tolist(), 'descriptor': descriptor.tolist()}
            try:
                features['descriptor_dd_pool'] = feature_vector(ds, dd_pool=True).tolist()
            except (ValueError, TypeError):
                pass
            normal = _finite(probe.get('normal_sdk'), (2,))
            if normal is not None:
                features['descriptor_normal'] = [*descriptor.tolist(), *normal.tolist()]
            wrench = _finite(raw.get('wrench_zeroed',probe.get('wrench_zeroed')), (2, 6))
            if wrench is not None:
                features['descriptor_wrench'] = [*descriptor.tolist(), *wrench.ravel().tolist()]
            for name in ('raw_dd_pool_unzeroed','dd_zeroed_pool'):
                if name in raw:
                    features[name] = raw[name]
            stratum = [rec['session_id'], row.get('bottle', 'unknown'), row.get('cond', 'A'), role,
                       _hash(_signature(meta))[:12], bool(meta.get('simulated')),
                       _hash(row.get('zero', {'missing_calibration':True}))[:12]]
            obs.append(dict(id=f"{rec['session_id']}:{row.get('id')}:{index}", trial_id=str(row.get('id')),
                            trial_group=f"{rec['session_id']}:{row.get('id')}", session_id=rec['session_id'],
                            source=rec['source'], source_type=rec['source_type'], path=str(rec['folder']),
                            stratum=stratum, condition=row.get('cond', 'A'), bottle=row.get('bottle', 'unknown'),
                            probe_role=role, probe_index=index, geometry=geometry.tolist(),
                            planned=planned.tolist() if planned is not None else None,
                            label_source=source, label_issues=issues, recording_verified=recording_verified,
                            timestamp_ns=probe.get('timestamp_ns'), started_unix_ns=row.get('started_unix_ns'),
                            edge=any(d.get('touches_edge', False) for d in ds), features=features,
                            raw_issues=raw_issues, raw_dd_temporal_sd=raw.get('raw_dd_temporal_sd'),
                            raw_provenance=raw.get('raw_provenance'),raw_provenance_h5=raw.get('raw_provenance_h5'),
                            dd_reference_note=raw.get('dd_reference_note'),
                            label=row.get('label'), label_history_present=bool(row.get('label_history'))))
    return obs, dict(exclusions)


def _cell(o):
    # Planned cells define repeat intent. Corrected B states require measured bins.
    point = o['planned'] if o['probe_role'] == 'initial' and o['planned'] is not None else o['geometry']
    return tuple(round(float(x), 3) for x in point)


def _repeat_groups(items, settings):
    groups = defaultdict(list)
    for i, o in enumerate(items):
        groups[_cell(o)].append(i)
    eligible = []
    for cell, indices in groups.items():
        # Multiple regrasp attempts from one trial are not independent repeats.
        by_trial = {items[i]['trial_group']: i for i in indices}
        ix = list(by_trial.values())
        if len(ix) < settings['min_repeats']:
            continue
        Y = np.asarray([items[i]['geometry'] for i in ix])
        if all(items[i]['label_source'].startswith('measured') for i in ix):
            center = np.median(Y, axis=0)
            ix = [i for i in ix if np.linalg.norm(np.asarray(items[i]['geometry'])[:2]-center[:2]) <= settings['repeat_position_mm']
                  and abs(items[i]['geometry'][2]-center[2]) <= settings['repeat_angle_deg']]
        if len(ix) >= settings['min_repeats']:
            eligible.append((cell, ix))
    return eligible


def _floors(name, dim):
    if name == 'centroid':
        return np.full(dim, .5)
    if name.startswith('descriptor'):
        base = np.tile([.5, .5, .5, .5, .5, .5, .02, .02, .02, .01], 2)
        if name == 'descriptor_dd_pool':
            base = np.r_[base[:10], np.full(16, 1e-5), base[10:], np.full(16, 1e-5)]
        return np.r_[base, np.full(max(0, dim-len(base)), .01)]
    return np.full(dim, 1e-5)


def _representation(items, name, settings):
    sub = [o for o in items if name in o['features'] and not o['label_issues']]
    if not sub:
        return {'status': 'unavailable', 'n': 0}, [], {}
    X = np.asarray([o['features'][name] for o in sub])
    groups = _repeat_groups(sub, settings)
    dof = sum(len(ix)-1 for _, ix in groups)
    noise_ok = len(groups) >= settings['min_repeat_cells'] and dof >= settings['min_repeat_dof']
    if not noise_ok:
        return {'status': 'insufficient_repeats', 'n': len(sub), 'repeat_cells': len(groups), 'repeat_dof': dof}, [], {}
    residuals = np.vstack([X[ix]-np.median(X[ix], axis=0) for _, ix in groups])
    scale = np.maximum(np.median(np.abs(residuals), axis=0)/.67448975, _floors(name, X.shape[1]))
    within = []
    for _, ix in groups:
        within.extend(float(np.sqrt(np.mean(((X[i]-X[j])/scale)**2))) for a, i in enumerate(ix) for j in ix[a+1:])
    cutoff = max(.5, float(np.quantile(within, settings['noise_quantile'])))
    candidates, distances, tested = [], {}, 0
    for i, left in enumerate(sub):
        for j in range(i+1, len(sub)):
            right = sub[j]
            if left['trial_group'] == right['trial_group']:
                continue
            gap = np.asarray(left['geometry'])-np.asarray(right['geometry'])
            mm, deg = float(np.linalg.norm(gap[:2])), abs(float(gap[2]))
            if mm < settings['action_gap_mm'] and deg < settings['action_gap_deg']:
                continue
            tested += 1
            d = float(np.sqrt(np.mean(((X[i]-X[j])/scale)**2)))
            key = tuple(sorted([left['id'], right['id']]))
            distances[key] = dict(distance=d, cutoff=cutoff, near=bool(d <= cutoff))
            if d <= cutoff:
                candidates.append({'left': left['id'], 'right': right['id'], 'distance': d,
                                   'threshold': cutoff, 'gap_mm': mm, 'gap_deg': deg,
                                   'left_geometry': left['geometry'], 'right_geometry': right['geometry'],
                                   'label_quality': 'measured' if all(o['label_source'].startswith('measured') for o in (left,right)) else 'nominal_unverified',
                                   'edge': left['edge'] or right['edge'],
                                   'recording_verified': left['recording_verified'] and right['recording_verified']})
    candidates.sort(key=lambda x: x['distance'])
    return {'status': 'screened', 'n': len(sub), 'repeat_cells': len(groups), 'repeat_dof': dof,
            'tested_pairs': tested, 'candidate_pairs': len(candidates), 'threshold': cutoff,
            'scale': scale.tolist(), 'within_repeat_distance_quantiles': np.quantile(within, [.1,.5,.9]).tolist(),
            'threshold_calibration': 'same-condition independent repeated trials; pooled MAD scales; empirical repeat-distance quantile, descriptive not a significance test'}, candidates, distances


def _drift(items):
    zeros = sorted([o for o in items if max(abs(x) for x in (_cell(o))) < 1e-6 and o['probe_role'] == 'initial'],
                   key=lambda o: o.get('started_unix_ns') or o.get('timestamp_ns') or 0)
    if len(zeros) < 6:
        return {'status': 'insufficient_zero_repeats', 'n': len(zeros)}
    X = np.asarray([o['features']['centroid'] for o in zeros])
    ref = X[:3].mean(axis=0)
    return {'status': 'descriptive_only', 'n': len(zeros), 'ids': [o['id'] for o in zeros],
            'centroids_px': X.tolist(), 'trial_ids': [o['trial_id'] for o in zeros], 'first_three_mean_px': ref.tolist(),
            'last_three_minus_first_three_px': (X[-3:].mean(axis=0)-ref).tolist(),
            'max_later_finger_drift_px': float(np.max(np.linalg.norm((X[3:]-ref).reshape(-1,2,2), axis=2))),
            'interpretation': '接触波动、瓶位、扣零或材料变化均可能导致漂移；像素不是毫米，不能据此单独确认瓶子移动。'}


def analyze(sources, *, recursive=False, settings=None):
    cfg = {**DEFAULTS, **(settings or {})}
    for k in ('action_gap_mm','action_gap_deg','repeat_position_mm','repeat_angle_deg','tracking_tolerance_mm','tracking_tolerance_deg'):
        if not isinstance(cfg[k], (int,float)) or not np.isfinite(cfg[k]) or cfg[k] <= 0:
            raise ValueError('invalid ' + k)
    if not 0 < cfg['noise_quantile'] < 1:
        raise ValueError('noise_quantile must lie strictly between zero and one')
    for k in ('min_repeat_dof','min_repeat_cells','min_repeats','max_pairs'):
        if type(cfg[k]) is not int or cfg[k] < (2 if k == 'min_repeats' else 1):
            raise ValueError('invalid ' + k)
    records, inventory, issues, duplicates = _load_sources(sources, recursive)
    obs, excluded = observations(records, cfg)
    grouped = defaultdict(list)
    for o in obs:
        grouped[tuple(o['stratum'])].append(o)
    strata, candidates = [], []
    for key, items in sorted(grouped.items(), key=lambda x:str(x[0])):
        sid = _hash(key)[:16]
        reps, pair_sets, distance_sets = {}, {}, {}
        for name in sorted({n for o in items for n in o['features']}):
            reps[name], pair_sets[name], distance_sets[name] = _representation(items, name, cfg)
        for pair in pair_sets.get('centroid', []):
            pkey = tuple(sorted([pair['left'], pair['right']]))
            comparisons = {n:d[pkey] for n,d in distance_sets.items() if pkey in d}
            separates = [n for n,d in comparisons.items() if n not in ('centroid','raw_dd_pool_unzeroed') and not d['near']]
            exploratory = [n for n,d in comparisons.items() if n == 'raw_dd_pool_unzeroed' and not d['near']]
            candidates.append({**pair, 'stratum_id': sid, 'representation_comparison': comparisons,
                               'richer_features_separate': separates,'exploratory_unzeroed_separation':exploratory,
                               'interpretation': 'feature_compression_candidate' if separates else 'unresolved_overlap_candidate'})
        strata.append({'id': sid, 'key': list(key), 'n': len(items), 'representations': reps,
                       'zero_drift': _drift(items), 'measured_labels': sum(o['label_source'].startswith('measured') for o in items),
                       'label_issues': dict(Counter(v for o in items for v in o['label_issues']))})
    sensitivity=[]
    for quantile in sorted(set((.25,cfg['noise_quantile'],.75))):
        counts=0; tested=0
        for items in grouped.values():
            statistic,pairs,_=_representation(items,'centroid',{**cfg,'noise_quantile':quantile})
            counts+=len(pairs);tested+=statistic.get('tested_pairs',0)
        sensitivity.append({'repeat_noise_quantile':quantile,'candidate_pairs':counts,'tested_pairs':tested})
    all_count = len(candidates)
    candidates.sort(key=lambda p:p['distance'])
    candidates = candidates[:cfg['max_pairs']]
    observation_by_id = {o['id']:o for o in obs}
    for pair in candidates:
        for side in ('left','right'):
            item = observation_by_id[pair[side]]
            pair[side+'_source'] = item['source']
            pair[side+'_trial_id'] = item['trial_id']
            pair[side+'_planned'] = item['planned']
    measured = sum(o['label_source'].startswith('measured') for o in obs)
    quality_issues = Counter(i for o in obs for i in o['label_issues'])
    raw_issues = Counter(i for o in obs for i in o['raw_issues'])
    adequate = any(s['representations'].get('centroid',{}).get('status') == 'screened' for s in strata)
    summary = dict(trials_loaded=len(records), observations=len(obs), initial_observations=sum(o['probe_role']=='initial' for o in obs),
                   regrasp_observations=sum(o['probe_role']=='regrasp' for o in obs), measured_labels=measured,
                   nominal_labels=sum(o['label_source']=='nominal_only' for o in obs), duplicate_trials=len(duplicates),
                   centroid_candidate_pairs=all_count, saved_candidate_pairs=len(candidates),
                   richer_feature_separation_pairs=sum(bool(p['richer_features_separate']) for p in candidates),
                   strata=len(strata), noise_calibrated_strata=sum(s['representations'].get('centroid',{}).get('status')=='screened' for s in strata),
                   status='candidate_overlap_needs_validation' if all_count else ('no_candidate_in_observed_data' if adequate else 'insufficient_evidence'),
                   raw_dd_observations=sum('raw_dd_pool_unzeroed' in o['features'] for o in obs),
                   reference_subtracted_dd_observations=sum('dd_zeroed_pool' in o['features'] for o in obs),
                   wrench_observations=sum('descriptor_wrench' in o['features'] for o in obs),
                   reliable_physical_ambiguity_established=False,
                   evidence_scope='offline diagnostic screening only; no training, planner updates or motion')
    limitations = [
        '不同几何偏移并不证明必须使用不同恢复动作；T0 回正不是经过验证的最优恢复标签。',
        'raw_dd_pool_unzeroed 是未扣 DD 参考的探索性特征；只能在同次扣零内比较，不作为排除物理歧义的充分证据。',
        '六维力按逐指原始 SDK wrench 减去该次空载均值，不旋转坐标、不改变正负号；两指时间窗中位数不宣称严格同步。',
        '候选近邻是重复噪声尺度下的描述性筛查，不是物理不可观测性、因果效应或统计显著性证明。',
        '同 session、瓶子、A/B、初接触/重抓和配置内比较；不会跨 session 零点制造冲突。',
        '重复接触波动包含姿态跟踪、复位、瓶位、扣零和传感噪声；不能仅归因为传感器噪声。',
        '更丰富特征分开某些近邻只支持特征不足的候选解释；需用独立重复和未见试验验证。',
        '未检出候选只适用于当前覆盖范围和阈值，不等于证明不存在歧义；一维数据不能验证 a/l/倾斜耦合。',
        '所有来源（包括评估数据）只用于诊断报告，不进入任何训练、方法晋升或 Agent 决策缓存。',
    ]
    if measured < len(obs):
        limitations.append('部分标签只有指令偏移，实际 PROBE 位姿未经核验；相关候选只能作为补测线索。')
    if any(not o['recording_verified'] for o in obs):
        limitations.append('部分来源缺少 recording_status.json；仅有导出元数据的 recording_complete 声明。')
    if not cfg['read_h5']:
        limitations.append('本次未读取 H5；不代表原始 DD、六维力或时间信息不存在。可用 --read-h5 对原始 session 复查。')
    return dict(schema=SCHEMA, settings=cfg, summary=summary, threshold_sensitivity=sensitivity, inventory=inventory, issues=issues,
                exclusions=excluded, label_quality_issues=dict(quality_issues), raw_read_issues=dict(raw_issues), duplicates=duplicates,
                by_stratum=strata, candidate_pairs=candidates, observations=obs, limitations=limitations)


def write_report(report, output):
    out = Path(output).expanduser().resolve()
    # Refuse output over data files; reports are fresh artifacts.
    out.mkdir(parents=True, exist_ok=True)
    if any((out/name).exists() for name in ('report.json','REPORT.md','observations.csv','candidate_pairs.csv')):
        raise FileExistsError('report output already exists; choose a new --out directory')
    (out/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    for name, rows, fields in (
        ('observations.csv', report['observations'], ['id','trial_id','session_id','source','condition','bottle','probe_role','geometry','planned','label_source','label_issues','recording_verified','edge']),
        ('candidate_pairs.csv', report['candidate_pairs'], ['stratum_id','left','right','gap_mm','gap_deg','distance','threshold','label_quality','edge','richer_features_separate','interpretation'])):
        with (out/name).open('w',newline='') as f:
            w=csv.DictWriter(f, fieldnames=fields);w.writeheader()
            for row in rows:
                w.writerow({k:json.dumps(row.get(k),ensure_ascii=False) if isinstance(row.get(k),(list,dict)) else row.get(k) for k in fields})
    s=report['summary']
    obs_by_id={o['id']:o for o in report['observations']}
    source_names=sorted({(Path(o['source']).parent.parent.name+'/batches/*') if Path(o['source']).parent.name=='batches' else _source_name(o['source']) for o in report['observations']})
    lines=['# 接触观测歧义离线诊断','',f"状态：`{s['status']}`。当前数据尚未建立可靠的物理观测歧义。",'',
           '来源：'+', '.join(source_names)+'.','',f"读取 {s['trials_loaded']} 次独立试验，得到 {s['observations']} 个有效 PROBE；实测位姿标签 {s['measured_labels']}，仅标称标签 {s['nominal_labels']}；去重 {s['duplicate_trials']} 次。",'',
           f"质心近邻候选 {s['centroid_candidate_pairs']} 对；其中已保存候选里，较丰富特征可区分 {s['richer_feature_separation_pairs']} 对。候选对不是独立样本数。",'',
           f"读取到 DD 的观测 {s['raw_dd_observations']}；有对应 DD 空载参考并已扣零的 {s['reference_subtracted_dd_observations']}；有扣零六维力的 {s['wrench_observations']}。",'',
           '| 分层（session / 瓶子 / 条件 / 接触） | 数量 | 质心筛查 | 候选对 | 实测标签 |',
           '|---|---:|---|---:|---:|']
    for g in _ordered_strata(report)[:12]:
        r=g['representations'].get('centroid',{})
        first=next(o for o in report['observations'] if o['stratum']==g['key'])
        name=_source_name(first['source'])+' / '+' / '.join(str(v) for v in g['key'][1:4])+' / zero='+str(g['key'][-1])[:6]
        lines.append(f"| {name} | {g['n']} | {r.get('status')} | {r.get('candidate_pairs',0)} | {g['measured_labels']} |")
    lines += ['', '表中优先显示有足够重复或数量较多的前 12 层；所有分层保存在 report.json。', '', '## 具体候选（不是独立证据数量）', '']
    if report['candidate_pairs']:
        for pair in report['candidate_pairs'][:20]:
            a,b=obs_by_id[pair['left']],obs_by_id[pair['right']]
            lines.append(f"- {_source_name(a['source'])} {a['bottle']}：试验 {a['trial_id']}（计划 {a['planned']}）与 {b['trial_id']}（计划 {b['planned']}）；实测平移差 {pair['gap_mm']:.3f} mm、角度差 {pair['gap_deg']:.3f}°。较丰富特征分开：{pair['richer_features_separate'] or '当前筛查未分开'}。")
        lines.append('这些配对可能共享试验；需要独立重复验证，不能把一对近邻称为稳定的物理歧义。')
    else:
        lines.append('当前阈值和覆盖范围内未检出候选；不等于证明不存在歧义。')
    lines += ['', '阈值敏感性（重复距离分位数 → 候选对数）：'+', '.join(f"{x['repeat_noise_quantile']:.2f} → {x['candidate_pairs']}" for x in report.get('threshold_sensitivity',[]))+'。这是筛查参数变化，不是新增独立证据。', '', '## 重复接触与零点漂移', '']
    for g in _ordered_strata(report):
        d=g['zero_drift']
        if d.get('status')=='descriptive_only':
            first=next(o for o in report['observations'] if o['stratum']==g['key'])
            lines.append(f"- {_source_name(first['source'])} {g['key'][1]} {g['key'][2]}：{d['n']} 次零点轻触，后续任一指相对最初三次均值的最大质心变化 {d['max_later_finger_drift_px']:.2f} px。包括瓶位、接触、材料、扣零和噪声的共同影响，不能换算成已知瓶位移动。")
    lines += ['', '## 标签与原始窗口质量', '',
              '- 剔除/缺失统计：'+json.dumps(report['exclusions'],ensure_ascii=False),
              '- 位姿或标签警示：'+json.dumps(report['label_quality_issues'],ensure_ascii=False),
              '- 原始窗口读取警示：'+json.dumps(report.get('raw_read_issues',{}),ensure_ascii=False),
              '- 无位姿警示仅表示未超配置门限；TCP 位姿不能证明瓶子未被移动。']
    lines += ['', '## 如何解释', '',
              '- 特征不足：同一对候选在质心下相似，在外框/形状/力/DD 下分开；仅为需独立验证的线索。',
              '- 噪声/漂移：查看每层重复距离与 zero_drift，不能把所有波动称为传感器噪声。',
              '- 标签问题：先核对指令与实测差异、B 重抓位姿、人工改标记录和物体是否移动。',
              '- 下一步：小范围 a/l/倾斜重复轻触，记录实测位姿；对候选状态做独立重采，再比较观测表示。', '', '## 限制', '']
    lines += ['- '+x for x in report['limitations']]
    if report['issues']:
        lines += ['', '## 输入读取问题', '']+[f"- {x['path']}: {x['reason']}" for x in report['issues']]
    plot = _plot(report,out)
    if plot:
        lines += ['', '![质心与几何偏移](centroid_screen.png)', '']
    if _plot_drift(report,out):
        lines += ['', '![零偏移质心随时间](zero_drift.png)', '']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    return out


def _source_name(source):
    path=Path(source)
    if path.parent.name=='batches':
        return path.parent.parent.name+'/batches/'+path.name
    return path.name


def _ordered_strata(report):
    return sorted(report['by_stratum'],key=lambda g:(
        -int(g['representations'].get('centroid',{}).get('status')=='screened'),
        -g['representations'].get('centroid',{}).get('candidate_pairs',0),-g['n'],g['id']))


def _plot_drift(report,out):
    groups=[g for g in _ordered_strata(report) if g['zero_drift'].get('status')=='descriptive_only']
    if not groups:
        return False
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(len(groups),1,figsize=(10,3*len(groups)),squeeze=False)
    for i,g in enumerate(groups):
        d=g['zero_drift'];X=np.asarray(d['centroids_px'])-np.asarray(d['first_three_mean_px'])
        ax=axes[i,0]
        for j,label in enumerate(('left u','left v','right u','right v')):
            ax.plot(np.arange(len(X)),X[:,j],marker='o',label=label)
        ax.set_xticks(np.arange(len(X)),d['trial_ids'],rotation=45)
        first=next(o for o in report['observations'] if o['stratum']==g['key'])
        ax.set_title(_source_name(first['source'])+' '+g['key'][1]+' '+g['key'][2]+' zero='+str(g['key'][-1])[:6])
        ax.set_xlabel('zero-trial ID in acquisition order');ax.set_ylabel('centroid minus first-three mean (px)')
        ax.axhline(0,color='black',lw=.7);ax.grid(alpha=.2);ax.legend(ncol=4)
    fig.tight_layout();fig.savefig(out/'zero_drift.png',dpi=140);plt.close(fig)
    return True


def _plot(report,out):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    groups=_ordered_strata(report)[:8]
    if not groups:
        return False
    fig,axes=plt.subplots(len(groups),2,figsize=(11,max(3,2.6*len(groups))),squeeze=False)
    for j,g in enumerate(groups):
        items=[o for o in report['observations'] if o['stratum']==g['key']]
        for finger in (0,1):
            ax=axes[j,finger]
            for coordinate,marker in [(0,'o'),(1,'x')]:
                ax.scatter([o['geometry'][0] for o in items], [o['features']['centroid'][2*finger+coordinate] for o in items],
                           s=18,marker=marker,label='uv'[coordinate])
            source=_source_name(items[0]['source'])
            ax.set_title(f"{source} {g['key'][1]} {g['key'][2]} {g['key'][3]} {'LR'[finger]} zero={str(g['key'][-1])[:6]}")
            ax.set_xlabel('a offset (mm); labels may be nominal');ax.set_ylabel('centroid (px)');ax.legend();ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(out/'centroid_screen.png',dpi=140);plt.close(fig)
    return True


def main(argv=None):
    p=argparse.ArgumentParser(description='离线观测歧义筛查，不连接机器人、不训练或修改策略。支持 session 目录和 data_snapshot.json。')
    p.add_argument('sources',nargs='+');p.add_argument('--out',required=True)
    p.add_argument('--recursive',action='store_true',help='在输入父目录递归发现 session')
    p.add_argument('--read-h5',action='store_true',help='只读 PROBE 小窗口实际 TCP 和 DD，需 h5py；不读取全程视频')
    for key in ('action_gap_mm','action_gap_deg','repeat_position_mm','repeat_angle_deg','noise_quantile'):
        p.add_argument('--'+key.replace('_','-'),type=float,default=DEFAULTS[key])
    a=p.parse_args(argv)
    settings={k:getattr(a,k) for k in ('action_gap_mm','action_gap_deg','repeat_position_mm','repeat_angle_deg','noise_quantile','read_h5')}
    try:
        report=analyze(a.sources,recursive=a.recursive,settings=settings)
        dest=write_report(report,a.out)
    except (OSError,ValueError) as exc:
        p.exit(2,'离线歧义诊断失败：'+str(exc)+'\n')
    print(json.dumps(report['summary'],ensure_ascii=False,indent=2));print('诊断报告：'+str(dest/'REPORT.md'))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
