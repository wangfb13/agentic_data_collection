"""Offline pilot regression tests. Hardware is a virtual event recorder only."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from bottle_scan.ambiguity_collect import main, ProbeRunner, summary
from bottle_scan.ambiguity_plan import make_design, validate_design, ProbeBounds, plan_report
from bottle_scan.e1e2_core import defaults
from bottle_scan.geometry import pose_offset, pose_error
from bottle_scan.session import atomic_json
from tests.test_e1e2 import FakeHardware


def cfg():
    out = defaults()
    out['gripper']['min_normalized'] = .336
    return out


class CapturingHardware(FakeHardware):
    def probe_window(self, since):
        result = super().probe_window(since)
        self.last_probe_window = result
        return result

    def move(self, target, speed):
        assert abs(self.width-self.cfg['gripper']['open_normalized']) <= self.cfg['gripper']['open_tolerance']
        assert speed <= 5.
        super().move(target, speed)


def answer(prompt):
    return 'MOVE' if '输入 MOVE' in prompt else ''


def write_cfg(tmp_path):
    path = tmp_path/'config.yaml'
    path.write_text(yaml.safe_dump(cfg()))
    return path


def make_runner(tmp_path, *, tilt_deg=3., ask=answer):
    config = cfg()
    hardware = CapturingHardware(config)
    design = make_design(tilt_deg=tilt_deg)
    args = SimpleNamespace(out=tmp_path, step=False, cond='A', limit=None)
    runner = ProbeRunner(args, config, design, hardware, ask=ask, say=lambda _: None)
    runner.T0 = hardware.pose()
    runner.bounds = ProbeBounds(config, runner.T0, design)
    runner.last_commanded_pose = runner.T0.copy()
    runner.meta = {'schema':'ambiguity_session_v1', 'T0':runner.T0.tolist(),
                   'config':config, 'ambiguity_design':design, 'plan':design['plan']}
    runner.physical_completed = 0
    return runner, hardware, design


def test_plan_has_balanced_independent_repeats_and_zero_controls():
    design = make_design()
    assert len(design['plan']) == 33
    assert design == make_design()
    assert [design['plan'][x]['state_id'] for x in (0,11,22)] == ['s00']*3
    for state in {x['state_id'] for x in design['plan']}:
        rows = [x for x in design['plan'] if x['state_id'] == state]
        assert {x['repeat'] for x in rows} == {0,1,2}
    assert len(make_design(tilt_deg=0)['plan']) == 27
    tampered = copy.deepcopy(design)
    tampered['plan'][1]['perturbation']['a_mm'] += 1
    with pytest.raises(ValueError):
        validate_design(tampered)


@pytest.mark.parametrize('options', [dict(step_mm=6),dict(step_mm=float('nan')),
                                      dict(tilt_deg=4),dict(tilt_deg=-1),dict(repeats=2)])
def test_design_rejects_limits_and_nonfinite(options):
    with pytest.raises(ValueError):
        make_design(**options)


def test_targets_use_original_local_axes_and_tcp_pivot(tmp_path):
    runner, hw, design = make_runner(tmp_path)
    bounds = runner.bounds
    point = {'perturbation':{'a_mm':4.,'l_mm':-4.,'tilt_deg':3.}}
    target = bounds.target(point)
    np.testing.assert_allclose(target[:3,3]-runner.T0[:3,3], 4*bounds.a-4*bounds.l)
    assert pose_error(runner.T0,target)[1] == pytest.approx(3.)
    assert bounds.check(target) is None
    assert bounds.check(bounds.pre(target),'retreat',target) is None
    np.testing.assert_allclose(bounds.pre(target)[:3,3]-target[:3,3],
                               -target[:3,:3]@np.array(runner.cfg['axes_tcp']['a'])*40)
    # Rotation alone does not secretly hold an assumed gel center fixed.
    only_rotation = bounds.target({'perturbation':{'a_mm':0.,'l_mm':0.,'tilt_deg':3.}})
    np.testing.assert_allclose(only_rotation[:3,3],runner.T0[:3,3])


def test_bounds_reject_other_axis_c_translation_height_and_large_rotation(tmp_path):
    r, _, _ = make_runner(tmp_path)
    b = r.bounds
    assert b.check(pose_offset(r.T0,[0,1,0],axes=r.cfg['axes_tcp']))
    assert b.check(pose_offset(r.T0,[0,0,0],'a',1,r.cfg['axes_tcp']))
    assert b.check(pose_offset(r.T0,[0,0,0],'c',4,r.cfg['axes_tcp']))
    assert b.check(pose_offset(r.T0,[9,0,0],axes=r.cfg['axes_tcp']))
    low = r.T0.copy(); low[2,3] -= 11
    assert '高度' in b.check(low)


def test_plan_command_is_offline_and_does_not_invent_base_frame(tmp_path):
    def forbidden(_):
        pytest.fail('plan constructed hardware')
    output = tmp_path/'plan.json'
    assert main(['plan','--out',str(output)],make_hardware=forbidden,output=lambda _:None) == 0
    value = json.loads(output.read_text())
    assert value['has_measured_T0'] is False
    assert all('target_tcp_mm' not in row for row in value['trials'])
    assert 'rospy' not in __import__('sys').modules


def test_collect_without_explicit_real_never_constructs_hardware(tmp_path):
    with pytest.raises(SystemExit) as error:
        main(['collect','--out',str(tmp_path/'session')],make_hardware=lambda _:pytest.fail('hardware'))
    assert error.value.code == 2


def test_trial_is_probe_only_full_raw_export_and_open_moves(tmp_path):
    r, hw, design = make_runner(tmp_path)
    item = next(x for x in design['plan'] if x['perturbation']['tilt_deg'] == 3.)
    r.trial(item)
    meta = json.loads((tmp_path/'trials'/item['id']/'meta.json').read_text())
    assert meta['complete'] and meta['recording_complete']
    assert meta['probe_initial']['valid'] and meta['probe_initial']['stable_probe']
    assert meta['training_eligible'] is False
    assert meta['lifted'] is False
    assert meta['perturbation']['tilt_deg'] == 3.
    assert (tmp_path/'trials'/item['id']/'probe_observation.npz').exists()
    with np.load(tmp_path/'trials'/item['id']/'probe_observation.npz') as f:
        assert set(f.files) >= {'dd_left','dd_right','reference_left','reference_right','wrench_zeroed'}
    stages = [e[1] for e in hw.events if e[0] == 'stage']
    assert 'PROBE' in stages and 'SQUEEZE' not in stages and 'LIFT' not in stages and 'HOLD' not in stages
    commands = [e for e in hw.events if e[0] == 'force']
    assert commands and all(e[1] == r.cfg['force']['target']*r.cfg['force']['probe_fraction'] for e in commands)
    moves = [e[1] for e in hw.events if e[0] == 'move']
    # Neutral retreat precedes reorientation, then approach, then retreat.
    np.testing.assert_allclose(moves[0], r.bounds.pre(r.T0))
    np.testing.assert_allclose(moves[1], r.bounds.pre(r.bounds.target(item)))
    np.testing.assert_allclose(moves[2], r.bounds.target(item))
    np.testing.assert_allclose(moves[-1], r.bounds.pre(r.bounds.target(item)))


def test_small_tilt_also_requires_retracted_pose(tmp_path):
    r, hw, _ = make_runner(tmp_path,tilt_deg=.5)
    target = r.bounds.target({'perturbation':{'a_mm':0.,'l_mm':0.,'tilt_deg':.5}})
    with pytest.raises(RuntimeError, match='退让'):
        r.move(r.bounds.pre(target),'retreat',target)
    assert not any(e[0] == 'move' for e in hw.events)


def test_closed_gripper_cannot_move(tmp_path):
    r, hw, _ = make_runner(tmp_path)
    hw.width = .5
    with pytest.raises(RuntimeError, match='全开'):
        r.move(r.T0)
    assert not any(e[0] == 'move' for e in hw.events)


def test_invalid_height_trial_skips_without_open_close_or_movement(tmp_path):
    r, hw, design = make_runner(tmp_path)
    # a points up, so the 40 mm retreat violates the configured z floor.
    r.T0 = np.eye(4); r.T0[2,3] = 500
    hw.pose_value = r.T0.copy()
    r.bounds = ProbeBounds(r.cfg,r.T0,design)
    r.trial(design['plan'][0])
    assert r.current['status'] == 'skipped_bound'
    assert not any(e[0] in ('move','open','close','force') for e in hw.events)


def test_fault_stops_then_opens_without_lift(tmp_path):
    r, hw, design = make_runner(tmp_path)
    def fail(target,speed):
        hw.events.append(('move_failed',))
        raise RuntimeError('injected move fault')
    hw.move = fail
    with pytest.raises(RuntimeError, match='injected'):
        r.trial(design['plan'][0])
    names = [x[0] for x in hw.events]
    assert 'stop_arm' in names and 'recovery_open' in names
    assert names.index('stop_arm') < names.index('recovery_open')
    assert r.current['complete'] is False
    assert r.current['recording_complete'] is False


def test_done_resume_never_constructs_hardware(tmp_path):
    config = cfg(); design = make_design()
    atomic_json(tmp_path/'session_meta.json',{'schema':'ambiguity_session_v1','config':config,'ambiguity_design':design})
    for item in design['plan']:
        atomic_json(tmp_path/'trials'/item['id']/'meta.json',{**item,'complete':True,'status':'contact','label':'normal','recording_complete':True})
    assert main(['collect','--real','--out',str(tmp_path)],make_hardware=lambda _:pytest.fail('hardware'),output=lambda _:None) == 0


def test_changed_plan_and_foreign_schema_fail_before_hardware(tmp_path):
    design = make_design()
    atomic_json(tmp_path/'session_meta.json',{'schema':'ambiguity_session_v1','config':cfg(),'ambiguity_design':design})
    assert main(['collect','--real','--step-mm','2','--out',str(tmp_path)],make_hardware=lambda _:pytest.fail('hardware'),output=lambda _:None) == 2
    atomic_json(tmp_path/'session_meta.json',{'schema':'e1e2_session_v1'})
    assert main(['collect','--real','--out',str(tmp_path)],make_hardware=lambda _:pytest.fail('hardware'),output=lambda _:None) == 2


def test_full_entry_limit_and_q_label(tmp_path):
    config_file = write_cfg(tmp_path)
    hardware = CapturingHardware(cfg())
    labels = iter(['q','1'])
    def ask(prompt):
        if '回车/1=' in prompt:
            return next(labels)
        return answer(prompt)
    session = tmp_path/'session'
    result = main(['collect','--real','--config',str(config_file),'--out',str(session),'--limit','2'],
                  make_hardware=lambda _:hardware,input_fn=ask,output=lambda _:None)
    assert result == 0
    rows = summary(session)
    assert len(rows) == 1 and rows[0]['complete'] and rows[0]['label'] == 'normal'
    assert hardware.events[-1][0] == 'disconnect'


def test_pending_label_resume_does_not_repeat_physical_trial(tmp_path):
    config = cfg(); design = make_design()
    atomic_json(tmp_path/'session_meta.json',{'schema':'ambiguity_session_v1','config':config,'ambiguity_design':design})
    for item in design['plan']:
        atomic_json(tmp_path/'trials'/item['id']/'meta.json',{**item,'complete':True,'status':'contact','label':'normal','recording_complete':True})
        atomic_json(tmp_path/'trials'/item['id']/'recording_status.json',{'complete':True})
    first = tmp_path/'trials'/'00001'/'meta.json'
    pending = json.loads(first.read_text()); pending.update(complete=False,physical_finished=True,label=None)
    atomic_json(first,pending)
    assert main(['collect','--real','--out',str(tmp_path)],make_hardware=lambda _:pytest.fail('unlabelled completed trial repeated'),
                input_fn=lambda _:'1',output=lambda _:None) == 0
    fixed = json.loads(first.read_text())
    assert fixed['complete'] and fixed['label'] == 'normal'
    assert not (tmp_path/'interrupted').exists()


def test_initial_bad_retreat_preflight_precedes_any_actuator(tmp_path):
    config_file = write_cfg(tmp_path)
    hardware = CapturingHardware(cfg())
    hardware.pose_value = np.eye(4); hardware.pose_value[2,3] = 500
    messages = []
    result = main(['collect','--real','--config',str(config_file),'--out',str(tmp_path/'session')],
                  make_hardware=lambda _:hardware,input_fn=answer,output=messages.append)
    assert result == 2
    assert any('初始退让位' in m for m in messages)
    assert not any(e[0] in ('move','open','close','force','recovery_open') for e in hardware.events)
