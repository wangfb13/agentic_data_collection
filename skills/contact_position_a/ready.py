"""Measured downstream endpoints for the selectable white-bottle campaign."""
from copy import deepcopy
from math import ceil, isfinite
from pathlib import Path
import yaml
import numpy as np

HANDOFF = 'forcepolicy_handoff_l1_floor0330'
FIXED = 'fixed_squeeze_5sdk_lift_floor0330'
FIXED_DEVICE = 'fixed_squeeze_5sdk_lift_device_range'
FIXED_GOALS = (FIXED, FIXED_DEVICE)
MEASURED_GOALS = (*FIXED_GOALS, HANDOFF)
FIXED_CONSUMER = {
    'controller': 'e1e2_fixed_squeeze', 'version': 'e1e2_fixed_lift_v1',
    'params': {'closure_floor': .330, 'unit': 'sdk', 'target': 5.,
               'tolerance': .5, 'stable_s': .5, 'lift_mm': 40., 'hold_s': 2.}
}


def device_floor(cfg):
    hardware = yaml.safe_load(Path(cfg['hardware_config']).read_text())
    value = hardware['hardware']['public_ros']['raw_normalized_min']
    if type(value) not in (int, float) or not isfinite(value) or not 0 <= value < cfg['gripper']['open_normalized']:
        raise ValueError('Invalid device gripper raw_normalized_min')
    # The normalized controller commands whole thousandths. Round toward open.
    floor = ceil(value * 1000 - 1e-10) / 1000
    if floor < value or floor >= cfg['gripper']['open_normalized']:
        raise ValueError('Device gripper floor cannot be represented within its command range')
    return floor


def fixed_consumer(goal, floor):
    consumer = deepcopy(FIXED_CONSUMER)
    if goal == FIXED_DEVICE:
        consumer['version'] = 'e1e2_fixed_lift_device_range_v2'
        consumer['params']['closure_floor'] = floor
    elif goal != FIXED:
        raise ValueError('Unknown fixed squeeze protocol: ' + goal)
    return consumer


def select(spec, mode, *, device_min=None):
    original = next(c for c in spec['post_conditions'] if c['id'] == 'handoff_forcepolicy_floor0330')
    condition = deepcopy(original)
    if mode == 'forcepolicy':
        condition.update(id=HANDOFF, description='ForcePolicy 接手并确认力控就绪（L1）',
                         evaluator='skills.contact_position_a.ready:handoff_ready')
    elif mode in ('fixed5', 'fixed5_0330'):
        goal = FIXED_DEVICE if mode == 'fixed5' else FIXED
        if goal == FIXED_DEVICE:
            if device_min is None:
                raise ValueError('fixed5 requires the gripper floor from its hardware profile')
            spec['perturbation_variables']['closure_floor'].update(reference=device_min, levels=[device_min])
        condition.update(id=goal, description='双指平均法向力稳定达到 5.0 SDK 后提起 40 mm，保持 2 s 未检测到掉落',
                         evaluator='skills.contact_position_a.ready:fixed_ready',
                         consumer=fixed_consumer(goal, device_min))
    else:
        raise ValueError('Unknown Ready mode: '+str(mode))
    spec['post_conditions'] = [condition]
    spec['active_post_condition'] = condition['id']


def result(success=None, level=None, evidence=None):
    return {'evaluated': type(success) is bool, 'success': success, 'level': level,
            'evidence': [evidence] if evidence else []}


def handoff_ready(trial, window, condition):
    outcome = window.get('consumer_outcome') or {}
    expected = condition['consumer']
    if (window.get('is_terminal') and outcome.get('controller') == expected['controller']
            and outcome.get('version') and outcome['version'] == expected['version']
            and outcome.get('params', {}).get('closure_floor') == expected['params']['closure_floor']):
        for event in outcome.get('evidence', []):
            if not isinstance(event, dict):
                continue
            if outcome.get('L1') is True and event.get('event') == 'main_resumed':
                return result(True, 'L1', outcome)
            lost = event.get('event') == 'handoff_contact_lost_timeout' and event.get('phase') == 'forcepolicy'
            timeout = event.get('event') == 'fault' and str(event.get('reason', '')).startswith('ForcePolicy did not stabilize during handoff')
            if outcome.get('L1') is False and (lost or timeout):
                return result(False, 'L1', outcome)
    return result(evidence={'reason': 'no_verified_forcepolicy_handoff'})


def historical_fixed_ready(window, condition):
    """Read the original E1/E2 terminal receipt for the archived contact window."""
    import hashlib
    import json
    from pathlib import Path
    row = window['_legacy']
    missing = result(evidence={'reason': 'no_matching_completed_e1e2_endpoint'})
    if (not window.get('is_terminal') or row.get('downstream_controller') != 'e1e2_squeeze'
            or row.get('closure_floor') != condition['consumer']['params']['closure_floor']):
        return missing
    prov = row.get('provenance', {})
    records = []
    for name, checksum in [('session_meta_path','session_meta_sha256'), ('meta_path','metadata_sha256')]:
        if not prov.get(name) or not prov.get(checksum):
            return missing
        raw = Path(prov[name]).read_bytes()
        if hashlib.sha256(raw).hexdigest() != prov[checksum]:
            raise ValueError('Archived E1/E2 metadata changed: '+prov[name])
        records.append(json.loads(raw))
    session, trial = records
    if session.get('simulated') is not False or not (trial.get('recording_complete') and trial.get('complete')):
        return missing
    try:
        check_fixed_config(session['config'])
    except (KeyError,ValueError):
        return missing
    probe = trial.get('probe_final') or {}
    if not probe.get('valid') or probe.get('timestamp_ns') != row.get('timestamp_ns'):
        return missing
    stages = {s['name']:s for s in trial['stages'] if s.get('end_ns') is not None}
    if 'SQUEEZE' not in stages:
        return missing
    ev = {'source':'historical_e1e2', 'meta_path':prov['meta_path'], 'meta_sha256':prov['metadata_sha256'],
          'session_meta_path':prov['session_meta_path'], 'session_sha256':prov['session_meta_sha256'],
          'probe_timestamp_ns':probe['timestamp_ns'], 'status':trial['status']}
    if trial['status'] == 'force_not_reached' and trial.get('target_reached') is False and not trial.get('lifted'):
        return result(False, 'squeeze', ev)
    hold = stages.get('HOLD', {})
    if (trial['status'] == 'hold_complete' and trial.get('target_reached') is True
            and trial.get('lifted') is True and trial.get('hold_complete') is True
            and type(trial.get('dropped')) is bool and 'LIFT' in stages and hold
            and stages['SQUEEZE']['end_ns'] <= stages['LIFT']['start_ns']
            and stages['LIFT']['end_ns'] <= hold['start_ns']
            and hold['end_ns']-hold['start_ns'] >= condition['consumer']['params']['hold_s'] * 1e9):
        ev['hold_duration_s'] = (hold['end_ns']-hold['start_ns'])/1e9
        return result(not trial['dropped'], 'lift_hold', ev)
    return missing


def fixed_ready(trial, window, condition):
    if '_legacy' in window:
        if condition['id'] != FIXED:
            return result(evidence={'reason':'archived_fixed5_protocol'})
        return historical_fixed_ready(window, condition)
    o = window.get('consumer_outcome') or {}
    expected = condition['consumer']
    if not window.get('is_terminal') or any(o.get(k) != expected[k] for k in ('controller','version','params')):
        return result(evidence={'reason':'no_matching_fixed_squeeze_test'})
    for sample in o.get('evidence', []):
        if not isinstance(sample, dict) or sample.get('event') != 'fixed_squeeze_lift_result':
            continue
        if sample.get('target_reached') is False and sample.get('lifted') is False:
            return result(False, 'squeeze', o)
        normals = np.asarray(sample.get('normal_sdk', []), dtype=float)
        if (sample.get('target_reached') is True and normals.shape == (2,) and np.isfinite(normals).all()
                and abs(normals.mean()-expected['params']['target']) <= expected['params']['tolerance']
                and sample.get('lifted') is True and sample.get('hold_complete') is True
                and type(sample.get('dropped')) is bool):
            return result(not sample['dropped'], 'lift_hold', o)
    return result(evidence={'reason':'incomplete_fixed_squeeze_lift_evidence'})


def check_fixed_config(cfg, goal=FIXED):
    f = cfg['force']
    actual = {'closure_floor':cfg['gripper']['min_normalized'],'unit':f['unit'],'target':f['target'],
              'tolerance':f['squeeze_tolerance'],'stable_s':f['squeeze_stable_s'],
              'lift_mm':cfg['motion']['lift_mm'],'hold_s':cfg['hold_s']}
    expected = fixed_consumer(goal, device_floor(cfg) if goal == FIXED_DEVICE else None)
    if actual != expected['params']:
        raise ValueError(f'Fixed squeeze protocol differs from the selected Ready definition: {actual}')
    return expected


def fixed_squeeze_lift(backend, runner, target, goal=FIXED):
    """Called within V4's existing exclusive control lease at the terminal contact."""
    consumer = check_fixed_config(runner.cfg, goal)
    p = consumer['params']
    backend._require_supported(runner.hw.snapshot())
    window = backend.recovery.window
    sample = runner.regulate('SQUEEZE', p['target'], p['tolerance'], p['stable_s'])
    squeeze_opening = float(runner.hw.snapshot()['width']) if goal == FIXED_DEVICE else None
    runner.current.update(target_reached=sample is not None, dropped=False)
    if sample is not None:
        runner.hold_and_lower(target, sample)
    evidence = {'event':'fixed_squeeze_lift_result','target_reached':sample is not None,
                'normal_sdk':sample['normal_sdk'] if sample is not None else None,
                'lifted':runner.current.get('lifted',False),
                'hold_complete':runner.current.get('hold_complete',False),
                'dropped':runner.current['dropped']}
    if goal == FIXED_DEVICE:
        evidence['squeeze_end_opening_normalized'] = squeeze_opening
    outcome = {**consumer, 'evidence':[evidence]}
    backend._emit({'event':'state_graph_consumer_result','node_id':window['node_id'],
                   'group_id':window['group_id'],'is_terminal':True,'consumer_outcome':outcome})
    # _SkillRunner.release checks the tabletop corridor again before opening.
    runner.release()
