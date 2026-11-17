"""Retries across V4 process exits, operator review and restarts."""
import json
import os
import sys
from pathlib import Path

import pytest

from bottle_scan.state_graph.campaign import (
    RetryTrial, all_rows, atomic, collect_pending, deferred_trials, queue_plan, read, release_deferred, retry_archive,
)
from skills.contact_position_a.campaign import recover_trial, run_with_diagnostics
from tests.test_graph_campaign_v4 import make_campaign, synthetic_executor

ERROR = 'TimeoutError: WristFisheyeX5Camera(192.168.9.10:5552) timed out waiting for a new frame.'


def seed(tmp_path):
    out, spec = make_campaign(tmp_path)
    spec['adapters']['campaign_recover'] = 'skills.contact_position_a.campaign:recover_trial'
    state = read(out/'state.json')
    queue_plan(out, state, spec, {'cells': [{'settings': {'x': 0}, 'reps': 2}]}, 'exploration')
    return out, spec, state


@pytest.mark.parametrize('old_receipt', [False, True])
def test_camera_before_episode_reruns_exact_request_once(tmp_path, old_receipt):
    out, spec, state = seed(tmp_path)
    first, later = state['trials']
    request = read(Path(first['directory'])/'request.json')
    calls = []

    def fail(request, directory, *args, **kwargs):
        calls.append(request['trial_id'])
        atomic(directory/'launch.json', {'request': request})
        if old_receipt:
            (directory/'episodes').mkdir()
        proc = {'returncode': 1, 'wall_seconds': 4}
        if not old_receipt:
            proc['technical_error'] = ERROR
        atomic(directory/'process.json', proc)
        if old_receipt:
            return None    # A run started with the older scheduler.
        return recover_trial(request, directory, spec, {}, interactive=False)

    collect_pending(out, state, spec, {}, executor=fail)
    if old_receipt:
        assert state['phase'] == 'awaiting_result'
        state = read(out/'state.json')
        collect_pending(out, state, spec, {}, real=False)
    assert calls == [request['trial_id']]
    assert state['spent'] == {'trials': 0, 'actions': 0, 'failures': 0, 'seconds': 0}
    assert state['phase'] == 'ready_to_collect'
    first, later = state['trials']
    assert first['status'] == later['status'] == 'queued'
    assert read(Path(first['directory'])/'request.json') == request
    archive = Path(first['incomplete_attempts'][0]['archive'])
    assert (archive/'process.json').exists() and all_rows(out, state) == []
    collect_pending(out, state, spec, {}, executor=synthetic_executor)
    assert state['spent']['trials'] == 2 and len(all_rows(out, state)) == 4
    assert first['trial_id'] == request['trial_id']


def test_mistyped_confirmation_and_crash_during_archive(tmp_path):
    out, spec, state = seed(tmp_path)
    trial = state['trials'][0]
    d = Path(trial['directory'])
    trial.update(status='started', started_ns=42)
    state['spent']['trials'] = 1
    archive = retry_archive(out, trial)
    archive.parent.mkdir(parents=True)
    atomic(d/'retry.json', {'trial_id': trial['trial_id'], 'reason': '未确认'})
    d.rename(archive)   # interruption before republishing the request and counters
    collect_pending(out, state, spec, {}, real=False)
    collect_pending(out, state, spec, {}, real=False)
    assert state['spent']['trials'] == 0 and len(trial['incomplete_attempts']) == 1
    assert read(d/'request.json') == trial['request']


def test_live_error_classification_and_preservation_of_review(tmp_path):
    out, spec, state = seed(tmp_path)
    req = state['trials'][0]['request']
    d = Path(state['trials'][0]['directory'])
    log = d/'episodes/episode_1/events.jsonl'
    log.parent.mkdir(parents=True)
    selected = {'event': 'graph_runtime_selected', 'model_id': 'planned_exploration', 'request': req}
    fault = {'event': 'fault', 'reason': ERROR}
    log.write_text('\n'.join(map(json.dumps, [selected, fault])))
    assert recover_trial(req, d, spec, {}, interactive=False) is None
    atomic(d/'process.json', {'returncode': 1, 'wall_seconds': 2})
    with pytest.raises(RetryTrial, match='WristFisheye'):
        recover_trial(req, d, spec, {}, interactive=False)
    outcome = {'event': 'state_graph_consumer_result', 'consumer_outcome': {'L1': True, 'L2': None}}
    log.write_text('\n'.join(map(json.dumps, [selected, outcome])))
    assert recover_trial(req, d, spec, {}, interactive=False) is None


def test_non_move_or_missing_exit_receipt(tmp_path):
    _, spec, state = seed(tmp_path)
    req = state['trials'][0]['request']
    d = Path(state['trials'][0]['directory'])
    with pytest.raises(RetryTrial, match='尚未确认'):
        recover_trial(req, d, spec, {}, interactive=False)
    atomic(d/'launch.json', {'request': req})
    assert recover_trial(req, d, spec, {}, interactive=False) is None


def test_child_stderr_and_exit_code_are_preserved(capsys):
    command = [sys.executable, '-c', 'import sys; print(sys.argv[1], file=sys.stderr); sys.exit(1)', ERROR]
    assert run_with_diagnostics(command, os.environ) == (1, ERROR)
    assert ERROR in capsys.readouterr().err


@pytest.mark.parametrize('exception', [RetryTrial(ERROR), RuntimeError('TCP child disconnected')])
def test_defer_then_run_rest_and_explicitly_retry_original_request(tmp_path, exception):
    out, spec, state = seed(tmp_path)
    first, second = state['trials']
    calls = []

    def interrupted(request, folder, *args, **kwargs):
        calls.append(request['trial_id'])
        if request['trial_id'] == first['trial_id']:
            atomic(folder/'actual_move.json', {'applied': True})
            raise exception
        return synthetic_executor(request, folder, *args, **kwargs)

    collect_pending(out, state, spec, {}, executor=interrupted, on_interruption=lambda *_: True)
    assert calls == [first['trial_id'], second['trial_id']]
    assert state['spent']['trials'] == 1 and state['spent']['failures'] == 0
    assert state['phase'] == 'deferred_review' and second['status'] == 'complete'
    archive = Path(first['incomplete_attempts'][0]['archive'])
    assert read(archive/'actual_move.json')['applied'] and not (Path(first['directory'])/'result.json').exists()
    assert read(archive/'retry.json')['reason'] == str(exception)
    if not isinstance(exception, RetryTrial):
        assert str(exception) in read(archive/'exception.json')['traceback']
    assert len(all_rows(out, state)) == 2
    state = read(out/'state.json')
    collect_pending(out, state, spec, {}, executor=interrupted)
    assert len(calls) == 2 and not (out/'certificate.json').exists()
    with pytest.raises(ValueError):
        release_deferred(out, [first['trial_id'], second['trial_id']])
    assert [t['trial_id'] for t in deferred_trials(read(out/'state.json'))] == [first['trial_id']]
    assert release_deferred(out, ['all']) == [first['trial_id']]
    assert release_deferred(out, ['all']) == []
    state = read(out/'state.json')
    assert first['request'] == state['trials'][0]['request']
    def record(request, *args, **kwargs):
        calls.append(request['trial_id'])
        return synthetic_executor(request, *args, **kwargs)
    collect_pending(out, state, spec, {}, executor=record)
    assert calls == [first['trial_id'], second['trial_id'], first['trial_id']]
    assert state['pending'] is None and state['spent']['trials'] == 2 and len(all_rows(out,state)) == 4


@pytest.mark.parametrize('archived',[False, True])
def test_restart_uses_durable_skip_receipt_once(tmp_path, archived):
    out, spec, state = seed(tmp_path)
    first, second = state['trials']
    d = Path(first['directory'])
    first.update(status='started',started_ns=101)
    state['spent']['trials']=1
    atomic(out/'state.json',state)
    atomic(d/'retry.json',dict(trial_id=first['trial_id'], disposition='deferred',reason=ERROR))
    if archived:
        archive=retry_archive(out,first)
        archive.parent.mkdir(parents=True)
        d.rename(archive)
    state=read(out/'state.json')
    collect_pending(out,state,spec,{},executor=synthetic_executor)
    assert state['phase']=='deferred_review' and state['spent']['trials']==1
    collect_pending(out,state,spec,{},executor=synthetic_executor)
    assert len(state['trials'][0]['incomplete_attempts'])==1
    assert state['trials'][0]['status']=='deferred' and state['trials'][1]['status']=='complete'


def test_recover_unknown_outcome_can_defer_without_reissuing_motion(tmp_path):
    out, spec, state = seed(tmp_path)
    def killed(*args, **kwargs):
        raise KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):
        collect_pending(out,state,spec,{},executor=killed,
                        on_interruption=lambda *_: pytest.fail('interrupt must stop'))
    first, second = state['trials']
    atomic(Path(first['directory'])/'launch.json',{'request':first['request']})
    state=read(out/'state.json')
    collect_pending(out,state,spec,{},executor=synthetic_executor,on_interruption=lambda *_: True)
    assert [t['status'] for t in state['trials']]==['deferred','complete']
    assert state['spent']['trials']==1


def test_completed_unsuccessful_contact_is_an_outcome_not_a_skip(tmp_path):
    out,spec,state=seed(tmp_path)
    def no_contact(request,*args,**kwargs):
        return dict(trial_id=request['trial_id'],status='failed',nodes=[],attempt_reason='no_bilateral_contact')
    collect_pending(out,state,spec,{},executor=no_contact,
                    on_interruption=lambda *_: pytest.fail('outcome is already known'))
    assert state['spent']['failures']==2 and state['pending'] is None
    assert not deferred_trials(state)


def test_operator_can_pause_then_use_cli_list_and_retry(tmp_path, monkeypatch, capsys):
    from skills.contact_position_a import campaign_ah
    out,spec,state=seed(tmp_path)
    # Keep the real CLI dispatch while using only a tmp campaign and synthetic hardware.
    spec['current_placement']='P3'
    atomic(out/'spec.json',spec)
    first,second=state['trials']
    def error(request,folder,*args,**kwargs):
        atomic(folder/'launch.json',{'request':request})
        raise ValueError('interrupt at reprobe')
    monkeypatch.setattr('builtins.input',lambda _: '')
    collect_pending(out,state,spec,{},executor=error,on_interruption=campaign_ah.offer_defer)
    assert first['status']=='started' and second['status']=='queued'
    monkeypatch.setattr('builtins.input',lambda _: 'skip')
    collect_pending(out,state,spec,{},executor=synthetic_executor,on_interruption=campaign_ah.offer_defer)
    before=(out/'state.json').read_bytes()
    campaign_ah.main(['--out',str(out),'--list-deferred'])
    assert (out/'state.json').read_bytes()==before
    output=capsys.readouterr().out
    assert first['trial_id'] in output and 'interrupt at reprobe' in output
    campaign_ah.main(['--out',str(out),'--retry-deferred',first['trial_id']])
    updated=read(out/'state.json')
    assert [t['status'] for t in updated['trials']]==['queued','complete']
    assert updated['spent']['trials']==1
