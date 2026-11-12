"""Leakage-resistant, hardware-free replay of bottle skill learning.

This is an observational replay: new policy actions are NOT counterfactually
executed. Geometric error and invocation/exit proxies are separate from recorded
legacy B outcomes. Historical frozen B validation never guides collection.
"""
from __future__ import annotations
import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from .agentic_learning import _load, _compatibility, _observations, _measured_probe, _success, _hash
from .contact_policy import (ContactPolicy, train_contact_policy, feature_vector,
                             FINAL_ROLES, _predict)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def discover(root):
    """Only canonical data locations; backup copies are never extra evidence."""
    root = Path(root)
    candidates = [root / 'e1e2_A_formal01', root / 'e1e2_B_formal01']
    candidates += [p.parent for p in sorted((root / 'rsi_contact01' / 'batches').glob('*/session_meta.json'))]
    return [str(p.resolve()) for p in candidates if (p / 'session_meta.json').exists()]


def provenance(row):
    return {'trial_path': row['trial_path'], 'metadata_sha256': row['metadata_sha256'],
            'session': row['session'], 'id': row.get('id')}


def eligible_A(rows, reference, settings):
    obs, excluded = _observations(rows, reference, settings)
    rowmap = {r['trial_path']: r for r in rows}
    kept = []
    for o in obs:
        r = rowmap[o['trial_path']]
        reason = None
        if r.get('label') in ('tilted', 'tipped', 'pushed', 'invalid'):
            reason = 'object_moved_or_invalid'
        elif not o['measurement']['source'].startswith('measured_tcp'):
            reason = 'no_measured_supervision'
        elif any(d.get('touches_edge') for d in o['descriptors']):
            reason = 'edge_contact_rule_not_learned'
        else:
            try:
                feature_vector(o['descriptors'])
            except (ValueError, TypeError, KeyError):
                reason = 'invalid_feature'
        if reason:
            excluded[reason] = excluded.get(reason, 0) + 1
        else:
            kept.append(o)
    return kept, excluded


def split_observations(obs, seed):
    """Freeze whole session if feasible; otherwise whole repeated-trial block.

    A group is the existing learner's session+bottle+repeat block, not a frame.
    The selection sees identities and counts only, never outcomes/features.
    """
    rng = np.random.default_rng(seed)
    sessions = sorted({o['session'] for o in obs})
    groups = sorted({o['group'] for o in obs})
    counts = Counter(o['session'] for o in obs)
    # Require enough source training for three equal-budget replay checkpoints.
    if len(sessions) >= 4:
        order = [sessions[i] for i in rng.permutation(len(sessions))]
        small = [s for s in order if counts[s] <= len(obs) * .25]
        if len(small) >= 2:
            final, dev = {small[0]}, set(small[1:3])
            if len(obs) - sum(counts[s] for s in final | dev) >= 24 and sum(counts[s] for s in dev) >= 4:
                return {o['trial_path']: ('independent' if o['session'] in final else
                         'development' if o['session'] in dev else 'pool') for o in obs}, 'whole_session'
    order = [groups[i] for i in rng.permutation(len(groups))]
    count = max(1, int(np.ceil(len(groups)*.2)))
    final, dev = set(order[:count]), set(order[count:2*count])
    roles = {o['trial_path']: ('independent' if o['group'] in final else
                              'development' if o['group'] in dev else 'pool') for o in obs}
    if sum(v == 'development' for v in roles.values()) < 4:
        raise ValueError('insufficient independent development groups')
    return roles, 'whole_session_bottle_repeat_group'


def stage_training(out, selected, dev, rowmap):
    """Metadata-only views with symlinks to immutable raw trials."""
    roots = {}
    for o in selected + dev:
        row = rowmap[o['trial_path']]
        source = row['session']
        if source not in roots:
            dest = Path(out) / hashlib.sha256(source.encode()).hexdigest()[:12]
            (dest / 'trials').mkdir(parents=True, exist_ok=True)
            meta = deepcopy(row['session_meta'])
            meta['skill_replay_source_session'] = source
            write_json(dest / 'session_meta.json', meta)
            roots[source] = dest.resolve()
        target = roots[source] / 'trials' / row['id']
        if not target.exists():
            target.symlink_to(Path(row['trial_path']))
    held = []
    for o in dev:
        # Match ContactPolicy._observations group key in staged session.
        row = rowmap[o['trial_path']]
        suffix = row.get('repeat') if row.get('repeat') is not None else 'monitor_' + str(row.get('id'))
        held.append(f"{roots[row['session']]}|{row.get('bottle')}|{suffix}")
    return [str(p) for p in roots.values()], sorted(set(held))


def raw_prediction(model, descriptors):
    """Ungated geometric estimate for error auditing, never a runtime action."""
    X = (feature_vector(descriptors) - np.asarray(model['scaling']['mean'])) / model['scaling']['scale']
    values = [_predict(n, X[None])[0] * model['scaling']['y_scale'] for n in model['networks']]
    return float(np.mean(values))


def observation_from_probe(row, probe):
    """Deployment fields only. Experimental delta and measured TCP are excluded."""
    stamp = probe.get('timestamp_ns')
    stable_start = probe.get('stable_start_ns')
    stable = ((stamp-stable_start)/1e9 if stamp is not None and stable_start is not None else None)
    descriptors = probe.get('descriptors', [])
    force = probe.get('normal_sdk')
    # Legacy probe.valid means the existing force-controlled PROBE completed.
    return {'status': 'contact' if probe.get('valid') else row.get('initial_status', 'unknown'),
            'descriptors': descriptors,
            'bilateral': bool(probe.get('valid') and len(descriptors) == 2 and all(d and d.get('contact') for d in descriptors)),
            'timestamp_ns': stamp, 'first_touch_complete': bool(probe),
            'force_stable': probe.get('force_stable'),
            'stable_duration_s': stable, 'context': {}, 'safety_ok': True}


def geometry_evaluation(model, observations, tolerance):
    report = {'n': len(observations), 'prediction_n': 0, 'runtime_accepted_n': 0,
              'mae_mm': None, 'p90_mm': None, 'rejected': 0}
    if not model.get('networks'):
        report['reason'] = 'no_trained_weights'
        return report
    errors = []
    runtime = ContactPolicy(model, {'tol_mm': tolerance, 'max_delta_mm': 15.,
                  'no_contact_delta_mm': 8., 'stall_delta_mm': -8.,
                  'root_edge_delta_mm': -6., 'tip_edge_delta_mm': 6.}) if model.get('usable_for_B') else None
    confusion = Counter()
    details = []
    for o in observations:
        pred = raw_prediction(model, o['descriptors'])
        errors.append(abs(pred + o['x_mm']))
        p = runtime.predict(o['descriptors']) if runtime else {'valid': False, 'reason': 'model_not_approved'}
        truth = abs(o['x_mm']) > tolerance
        if p['valid']:
            call = abs(p['action_mm']) > tolerance
            confusion['TP' if call and truth else 'FP' if call else 'FN' if truth else 'TN'] += 1
        else:
            confusion['unknown'] += 1
        details.append({'trial_path': o['trial_path'], 'target_action_mm': -o['x_mm'],
                        'prediction_mm': pred, 'runtime_prediction': p,
                        'needs_recovery_geometry_proxy': truth})
    report.update(prediction_n=len(errors), runtime_accepted_n=sum(v for k,v in confusion.items() if k != 'unknown'),
                  rejected=confusion['unknown'], mae_mm=float(np.mean(errors)),
                  p90_mm=float(np.quantile(errors,.9)), invocation=dict(confusion), details=details,
                  label_definition='negative measured TCP displacement from shared T0; proxy, not true bottle-relative contact goal')
    return report


def evaluate_B(model, rows, reference, settings, tolerance):
    """Audit recorded trajectories; never assign their outcome to a new action."""
    invocation, exit_counts, legacy_exit_counts = Counter(), Counter(), Counter()
    initial_missing, final_missing = Counter(), Counter()
    details = []
    for row in rows:
        if row.get('kind') != 'scan' or row.get('archived_attempt'):
            continue
        initial, final = row.get('probe_initial', {}), row.get('probe_final', {})
        initm = _measured_probe(row, reference, settings)
        finalm = _measured_probe({**row, 'probe_initial': final}, reference, settings)
        goal_initial = (abs(initm['x_mm']) <= tolerance if initm['x_mm'] is not None and
                        initm['source'].startswith('measured_tcp') else None)
        goal_final = (abs(finalm['x_mm']) <= tolerance if finalm['x_mm'] is not None and
                      finalm['source'].startswith('measured_tcp') else None)
        decisions = []
        if model.get('usable_for_B'):
            from .recovery_skill import RecoverySkill
            skill = RecoverySkill(model, row['session_meta']['config']['servo'], mode='development')
            probes = [initial] + [(r.get('probe') or {}) for r in row.get('probe_regrasps', [])]
            # Do not duplicate final snapshot: repeated timestamp is not fresh evidence.
            if final and all(final.get('timestamp_ns') != p.get('timestamp_ns') for p in probes):
                probes.append(final)
            for probe in probes:
                decisions.append(skill.evaluate(observation_from_probe(row, probe)))
        first = decisions[0]['invocation'] if decisions else {}
        need = first.get('needs_recovery')
        truth = not goal_initial if goal_initial is not None else None
        if need is None or truth is None:
            invocation['unknown'] += 1
        else:
            invocation['TP' if need and truth else 'FP' if need else 'FN' if truth else 'TN'] += 1
        last_exit = decisions[-1].get('exit', {}) if decisions else {}
        initial_missing[initm.get('reason') or 'available'] += 1
        final_missing[finalm.get('reason') or 'available'] += 1
        # Legacy trajectories did not execute this skill's stateful confirmation.
        # Missing windows are unknown, never a demonstrated false negative.
        exit_counts['unknown_missing_prospective_confirmation'] += 1
        history = row.get('correction_history', [])
        legacy_accepted = history[-1].get('accepted') if history else None
        if goal_final is None or legacy_accepted is None:
            legacy_exit_counts['unknown'] += 1
        elif legacy_accepted:
            legacy_exit_counts['true_completion_proxy' if goal_final else 'false_success_proxy'] += 1
        else:
            legacy_exit_counts['missed_completion_proxy' if goal_final else 'correct_incomplete_proxy'] += 1
        details.append({**provenance(row), 'legacy_success': _success(row),
                        'legacy_regrasps': row.get('regrasps'), 'legacy_controller': row.get('correction_history', []),
                        'initial_measurement': initm, 'final_measurement': finalm,
                        'goal_initial_geometry_proxy': goal_initial, 'goal_final_geometry_proxy': goal_final,
                        'skill_replay': decisions, 'runtime_context_missing': True})
    n = len(details)
    return {'n': n, 'invocation': dict(invocation), 'exit': dict(exit_counts), 'legacy_exit_geometry_proxy': dict(legacy_exit_counts),
            'initial_measurement_status': dict(initial_missing), 'final_measurement_status': dict(final_missing),
            'observed_legacy_B_success': sum(d['legacy_success'] for d in details),
            'observed_legacy_B_success_rate': sum(d['legacy_success'] for d in details)/n if n else None,
            'new_skill_prospective_recovery_success': None,
            'new_skill_prospective_exit_accuracy': None,
            'new_skill_completion_evaluable_n': 0,
            'proxy_only': True, 'details': details,
            'limitations': ['Recorded actions belong to legacy controllers, not the new skill.',
                           'TCP displacement is geometric proxy; bottle movement is not observed.',
                           'Legacy PROBE stores one stable snapshot, not two independent post-action windows.',
                           'Current runtime context and target contact labels were not collected.']}


def choose_batch(pool, selected, count, mode, feedback_rows, model, seed, tolerance):
    remaining = [o for o in pool if o['trial_path'] not in {x['trial_path'] for x in selected}]
    rng = np.random.default_rng(seed)
    rng.shuffle(remaining)
    if mode == 'fixed' or not selected:
        return remaining[:count], 'seeded fixed acquisition order; no feedback'
    difficulty = []
    for row in feedback_rows:
        if row.get('kind') != 'scan':
            continue
        weight = 1. + (not _success(row)) * 2.
        if model and model.get('networks') and row.get('probe_initial', {}).get('valid'):
            try:
                predicted = raw_prediction(model, row['probe_initial']['descriptors'])
                weight += min(abs(predicted), 15.) / 5.
            except (ValueError, KeyError, TypeError):
                pass
        difficulty.append((float(row['delta_mm']), weight))
    def score(o):
        if not difficulty:
            return 0.
        return max(w / (1. + abs(o['planned_delta_mm'] - x)) for x,w in difficulty)
    if mode == 'simple_feedback':
        return sorted(remaining, key=score, reverse=True)[:count], 'repeat conditions nearest unsuccessful or large-correction development B'
    chosen = []
    for i in range(min(count, len(remaining))):
        available = [o for o in remaining if o not in chosen]
        # Interleave reference repeats, B-informed hard points, coverage checks.
        if i % 3 == 0:
            item = min(available, key=lambda o: abs(o['planned_delta_mm']))
        elif i % 3 == 1:
            item = max(available, key=score)
        else:
            have = selected + chosen
            item = max(available, key=lambda o: min(abs(o['planned_delta_mm'] - x['planned_delta_mm']) for x in have))
        chosen.append(item)
    return chosen, 'evidence rule: alternate zero reference, B-informed hard condition, and undercovered a location; no held-out feedback'


def cost_record(row):
    stages = row.get('stages', [])
    durations = [(s['end_ns']-s['start_ns'])/1e9 for s in stages if
                 isinstance(s.get('end_ns'), int) and isinstance(s.get('start_ns'), int) and s['end_ns'] >= s['start_ns']]
    wall = ((row['finished_unix_ns']-row['started_unix_ns'])/1e9 if
            row.get('finished_unix_ns') and row.get('started_unix_ns') else None)
    failed = bool(not row.get('complete') or row.get('label') not in ('normal',) or row.get('status') == 'skipped_bound' or (row.get('kind') == 'scan' and not _success(row)))
    return {**provenance(row), 'phase': 'source_experience',
            'historical_role': ('B_independent' if row.get('session_role') in FINAL_ROLES else
                                'B_development' if row.get('cond') == 'B' else 'A_exploration'),
            'attempts': 1, 'failed': failed, 'wall_seconds': wall,
            'recorded_stage_seconds': sum(durations) if durations else None,
            'robot_motion_seconds': None, 'reset_seconds': None,
            'cost_note': 'stage duration includes sensing/wait; not motion-only time. Unrecorded reset/applicability costs unknown.'}


def run(roots, out, *, seed=20261105, budgets=(24,48,72), epochs=500):
    started = time.monotonic()
    out = Path(out).resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError('use a fresh output directory; frozen split and reports cannot be overwritten')
    out.mkdir(parents=True, exist_ok=True)
    settings = {'contact_policy': {'seed': seed, 'epochs': epochs},
                'learning': {'min_levels': 3}}
    sessions, rows, issues = _load(roots, settings)
    compatibility, evidence = _compatibility(sessions, settings)
    if issues or compatibility:
        write_json(out/'blocked.json', {'issues': issues+compatibility, 'compatibility': evidence})
        raise ValueError('input data incompatible; see blocked.json')
    reference = sessions[0]['meta']
    obs, excluded = eligible_A(rows, reference, settings)
    roles, split_method = split_observations(obs, seed)
    pool = [o for o in obs if roles[o['trial_path']] == 'pool']
    dev = [o for o in obs if roles[o['trial_path']] == 'development']
    final = [o for o in obs if roles[o['trial_path']] == 'independent']
    Bdev = [r for r in rows if r.get('cond') == 'B' and r['session_role'] not in FINAL_ROLES]
    Bfinal = [r for r in rows if r.get('cond') == 'B' and r['session_role'] in FINAL_ROLES]
    rowmap = {r['trial_path']: r for r in rows}
    budgets = sorted(set(min(int(n), len(pool)) for n in budgets))
    if not budgets or budgets[0] < 12:
        raise ValueError('insufficient replay budget / eligible pool')
    manifest = {'schema': 'skill_replay_split_v1', 'seed': seed, 'method': split_method,
                'A_roles': roles, 'B_development': [provenance(r) for r in Bdev],
                'B_independent': [provenance(r) for r in Bfinal],
                'independent_B_definition': 'historical validation sessions withheld from THIS replay selection; historical methods may already have used their outcomes, so not fresh blind evaluation',
                'A_evidence': [{**provenance(rowmap[o['trial_path']]), 'group': o['group']} for o in obs],
                'source_sessions': [{k:s[k] for k in ('path','metadata_sha256','role')} for s in sessions],
                'excluded_A_counts': excluded, 'compatibility': evidence,
                'budgets': budgets, 'learner': 'unchanged ContactPolicy numpy_tanh_mlp_ensemble',
                'action_space': 'same bounded scalar a correction / release-retreat-regrasp executor',
                'collection_interpretation': 'offline availability-limited replay, no new physical interaction'}
    write_json(out/'split_manifest.json', manifest)
    rounds, terminal = [], {}
    tolerance = float(reference['config']['servo']['tol_mm'])
    for mode in ('fixed','simple_feedback','agent_evidence'):
        selected, model = [], None
        for index, budget in enumerate(budgets, 1):
            batch, rationale = choose_batch(pool, selected, budget-len(selected), mode, Bdev,
                                            model, seed+index, tolerance)
            selected += batch
            destination = out/mode/f'round_{index:02d}'
            training_roots, held = stage_training(destination/'training_view', selected, dev, rowmap)
            cfg = deepcopy(settings)
            cfg['contact_policy']['frozen_validation_groups'] = held
            before = time.monotonic()
            model = train_contact_policy(training_roots, cfg, out=destination/'contact_policy.json')
            train_seconds = time.monotonic()-before
            development = geometry_evaluation(model, dev, tolerance)
            # Development only is visible to selection. Final evaluation occurs after every branch freezes.
            Bdevelopment = evaluate_B(model, Bdev, reference, settings, tolerance)
            result = {'round': index, 'mode': mode, 'budget_selected_A_trials': len(selected),
                      'actual_training_count': model.get('training_count'),
                      'shared_A_development_trials': len(dev), 'shared_B_development_trials': len(Bdev),
                      'model_id': model['model_id'], 'usable_for_B': model['usable_for_B'],
                      'blocking_reasons': model['blocking_reasons'],
                      'problem': 'invocation coverage, action prediction, and exit evidence gaps in development data',
                      'experiment_choice': [provenance(rowmap[o['trial_path']]) for o in batch],
                      'reason': rationale, 'development_geometry': development,
                      'B_development': Bdevelopment,
                      'skill_update': {'recovery_model': str(destination/'contact_policy.json'),
                                       'deployment_certificate': 'not_issued_no_prospective_skill_validation',
                                       'invocation_and_exit': 'unchanged preregistered runtime rules'},
                      'cost': {'new_robot_attempts': 0, 'replayed_A_acquisition_trials': len(selected),
                               'source_experience_charged_separately': True,
                               'offline_training_wall_seconds': train_seconds,
                               'source_reset_seconds': None}}
            write_json(destination/'round.json', result)
            rounds.append(result)
        terminal[mode] = model
    independent = {}
    for mode, model in terminal.items():
        from .recovery_skill import RecoverySkill
        if model.get('usable_for_B'):
            skill = RecoverySkill(model, reference['config']['servo'])
            write_json(out/mode/'skill.json', skill.artifact())
        independent[mode] = {'A_geometry': geometry_evaluation(model, final, tolerance),
                             'B_recorded_replay': evaluate_B(model, Bfinal, reference, settings, tolerance)}
    learning_curve = []
    for r in rounds:
        model_path = Path(r['skill_update']['recovery_model'])
        checkpoint_model = json.loads(model_path.read_text())
        learning_curve.append({'mode': r['mode'], 'round': r['round'],
            'selected_A_trials': r['budget_selected_A_trials'],
            'shared_development_trials': len(dev) + len(Bdev),
            'independent_A': geometry_evaluation(checkpoint_model, final, tolerance),
            'physical_new_interactions': 0})
    write_json(out/'learning_curve.json', learning_curve)
    efficiency=sample_efficiency_summary(learning_curve,source_attempts=len(rows),shared_development_trials=len(dev)+len(Bdev))
    write_json(out/'sample_efficiency.json',efficiency)
    costs = [cost_record(r) for r in rows]
    with (out/'source_costs.jsonl').open('w') as f:
        for record in costs:
            f.write(json.dumps(record, ensure_ascii=False, allow_nan=False)+'\n')
    report = {'schema': 'skill_offline_report_v1', 'status': 'offline_validation_complete_deployment_unverified',
              'source_trials': len(rows), 'eligible_A': len(obs), 'A_pool': len(pool),
              'A_development': len(dev), 'A_independent': len(final),
              'B_development': len(Bdev), 'B_independent': len(Bfinal),
              'sample_efficiency':efficiency,
              'rounds': [{k:r[k] for k in ('round','mode','budget_selected_A_trials','model_id','usable_for_B','blocking_reasons','cost')} for r in rounds],
              'independent': independent,
              'cost': {'source_experience_attempts': len(costs),
                       'source_failed_attempts': sum(c['failed'] for c in costs),
                       'source_recorded_stage_seconds': sum(c['recorded_stage_seconds'] or 0 for c in costs),
                       'source_missing_stage_costs': sum(c['recorded_stage_seconds'] is None for c in costs),
                       'source_reset_seconds': None, 'new_robot_attempts': 0,
                       'offline_wall_seconds': time.monotonic()-started},
              'gaps': ['Prospective B with the new three-output skill and independent goal labels is still required.',
                       'No-contact, already-normal and out-of-scope deployment conditions need balanced coverage.',
                       'Legacy final PROBE snapshots cannot validate the two-fresh-window exit rule.',
                       'B_formal01 actual cross-axis displacement exceeds original 2mm one-dimensional supervision gate; its geometric labels stay unknown.',
                       'Contact height and probing-force effects have no controlled evidence in these a-only sessions.'],
              'certificate': {'issued': False, 'reason': 'no prospective invocation/recovery/exit validation for this model'},
              'interpretation': ['A geometry accuracy is not grasp recovery success.',
                                 'Legacy B outcomes are reported only for the actual recorded controller.',
                                 'Three acquisition modes share learner, seeds, action space and selected-trial budgets.',
                                 'Replay comparisons are availability-limited, not causal estimates of real collection efficiency.',
                                 'Historical RSI validation may have guided source methods and later A collection. Withheld from THIS replay only; not a fresh independent prospective test.'],
              'evidence_refs': [str(out/'split_manifest.json'), str(out/'source_costs.jsonl'), str(out/'learning_curve.json')]}
    write_json(out/'report.json', report)
    text = ['# 药瓶 skill 离线回放验证', '',
            '状态：离线验证完成；新 skill 的真实部署适用范围仍未验证。没有启动机器人。',
            f'读取 {len(rows)} 次历史尝试；合格 A={len(obs)}，采样池={len(pool)}，A 开发={len(dev)}，A 独立={len(final)}；B 开发={len(Bdev)}，B 历史保留回放={len(Bfinal)}。',
            '', '|模式|最终 A 预算|A 独立 MAE / P90 (mm)|模型 B 门禁|',
            '|---|---:|---|---|']
    for mode, model in terminal.items():
        g=independent[mode]['A_geometry']
        text.append(f"|{mode}|{budgets[-1]}|{g.get('mae_mm')} / {g.get('p90_mm')}|{model['usable_for_B']}|")
    text += ['', '相同现有 ContactPolicy 学习器、随机种子、动作空间和采样预算；开发集成本是各组共同开销。Agent evidence 是可审计本地证据规则，并非伪称调用了远程 LLM。',
             '几何标签来自 PROBE 时实测 TCP；模型仅输入触觉。几何精度不等于真实瓶位或恢复成功。',
             '旧 B 使用旧控制器，不能反事实证明新模型动作有效。RSI 历史 validation 可能曾指导源方法或后续 A；本次禁止其指导补采，但不能将其当作从未曝光的前瞻盲测。',
             f'A 保留集为 {len(final)} 条；小样本不能证明方法统计优越性。最终独立实采还需冻结新 skill 后进行。退出要求两个新鲜观测窗口，旧数据多数只保存一次 PROBE；缺失证据不会补造。', '',
             '逐轮详情：各模式 round_XX/round.json；模型：contact_policy.json；拆分和原始证据：split_manifest.json；历史逐尝试成本：source_costs.jsonl。',
             '实际新增机器人尝试为 0；历史复位、方法适用性检查等未记录成本保持 null。']
    (out/'REPORT_zh.md').write_text('\n'.join(text)+'\n')
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', default='outputs')
    parser.add_argument('--session', action='append', help='explicit compatible canonical session; repeatable')
    parser.add_argument('--out', required=True)
    parser.add_argument('--seed', type=int, default=20261105)
    parser.add_argument('--budgets', type=int, nargs='+', default=[24,48,72])
    parser.add_argument('--epochs', type=int, default=500)
    args=parser.parse_args(argv)
    report=run(args.session or discover(args.data_root), args.out, seed=args.seed,
               budgets=args.budgets, epochs=args.epochs)
    print(json.dumps({k:report[k] for k in ('status','source_trials','eligible_A','cost')}, indent=2))
    return 0



def _binary_counts(counter, predicted, truth):
    if type(predicted) is not bool or type(truth) is not bool:
        counter['unknown'] += 1
    else:
        counter['TP' if predicted and truth else 'FP' if predicted else 'FN' if truth else 'TN'] += 1



def _load_skill_evaluation_rows(roots):
    """Read skill B contracts without making extension data eligible for a training."""
    from .agentic_learning import _signature
    sessions, rows, issues = [], [], []
    seen=set()
    for value in roots:
        root=Path(value).expanduser().resolve()
        if str(root) in seen:
            continue
        seen.add(str(root))
        try:
            raw=(root/'session_meta.json').read_bytes()
            meta=json.loads(raw)
        except (OSError,ValueError) as exc:
            issues.append(f'{root}: {exc}')
            continue
        if meta.get('schema')=='e1e2_session_v1':
            loaded, records, errors=_load([root],{})
            sessions.extend(loaded); rows.extend(records); issues.extend(errors)
            continue
        if meta.get('schema')!='ambiguity_session_v1':
            issues.append(f'{root}: unsupported skill evaluation schema')
            continue
        contract=meta.get('skill_extension_B')
        role=meta.get('evaluation_role')
        if (meta.get('simulated',False) or meta.get('cond')!='B' or
            role not in ('development','independent') or not isinstance(contract,dict) or
            contract.get('evaluation_role')!=role or not isinstance(meta.get('fit'),dict) or
            contract.get('fit')!=meta['fit'] or
            not isinstance(contract.get('settings'),dict) or
            contract['settings']!=meta.get('config',{}).get('recovery_skill')):
            issues.append(f'{root}: invalid or inconsistent extension B evaluation contract')
            continue
        eval_role='final_validation' if role=='independent' else 'development'
        session={'path':str(root),'meta':meta,'signature':_signature(meta),
                 'metadata_sha256':hashlib.sha256(raw).hexdigest(),'role':eval_role}
        sessions.append(session)
        for folder in ('trials','interrupted'):
            for path in sorted((root/folder).glob('*/meta.json')):
                try:
                    raw=path.read_bytes()
                    row=json.loads(raw)
                    if row.get('cond')!='B' or row.get('kind')!='skill_extension_probe' or row.get('evaluation_role')!=role:
                        raise ValueError('extension B trial kind/condition/evaluation role differs from frozen session')
                    status=path.parent/'recording_status.json'
                    verified=bool(row.get('recording_complete') and status.exists() and json.loads(status.read_text()).get('complete'))
                    row.update(session=str(root),trial_path=str(path.parent),
                        metadata_sha256=hashlib.sha256(raw).hexdigest(),
                        archived_attempt=folder=='interrupted',recording_verified=verified,
                        session_meta=meta,session_role=eval_role)
                    rows.append(row)
                except (OSError,ValueError,TypeError) as exc:
                    issues.append(f'{path}: invalid extension B evidence: {exc}')
    return sessions,rows,issues

def evaluate_skill_sessions(roots, *, role, out=None):
    """Evaluate recorded, actually executed three-output B skills.

    Labels must be separately collected operator contact-state annotations.
    Experimental delta/TCP are never labels here. Development and independent
    sources cannot be pooled. No certificate or method promotion is issued.
    """
    if role not in ('development', 'independent'):
        raise ValueError('role must be development or independent')
    from .agentic_learning import _signature
    sessions, rows, issues = _load_skill_evaluation_rows(roots)
    allowed = {'development', 'exploration'} if role == 'development' else {'final_validation', 'frozen_validation'}
    if issues or not sessions:
        raise ValueError('cannot evaluate sessions: ' + '; '.join(issues or ['no sessions']))
    for session in sessions:
        if session['role'] not in allowed:
            raise ValueError(f"{session['path']}: role {session['role']} cannot enter {role}")
    signatures = sorted({_hash(_signature(s['meta'])) for s in sessions})
    if len(signatures) != 1:
        raise ValueError('mixed hardware/contact protocols must be evaluated separately')
    calls, applicability, support, exits, recovery, actual_calls = (Counter() for _ in range(6))
    coverage, actions, conditions = Counter(), [], {}
    rejection_reasons = Counter()
    legacy_reason_corrections = 0
    details, seen = [], set()
    model_ids = set()
    for row in rows:
        if row.get('cond') != 'B' or row.get('kind') not in ('scan','skill_extension_probe'):
            continue
        canonical = str(Path(row['trial_path']).resolve())
        if canonical in seen:
            raise ValueError('duplicated physical trial in evaluation')
        seen.add(canonical)
        events = row.get('skill_observations', [])
        decisions = [event for event in events if isinstance(event.get('skill_outputs'), dict)]
        first = decisions[0]['skill_outputs'] if decisions else {}
        last = row.get('skill_outputs') or (decisions[-1]['skill_outputs'] if decisions else {})
        invocation = first.get('invocation', {})
        handoff = row.get('skill_handoff') or last.get('exit', {})
        decision = (first.get('recovery') or {}).get('decision') or {}
        recorded_reason = handoff.get('reason')
        diagnostic_reason = recorded_reason
        # Preserve the historical handoff, but diagnose the legacy reason bug.
        if (recorded_reason == 'missing_or_mismatched_validation_certificate'
                and not invocation.get('reasons') and decision.get('rejected') is True
                and decision.get('reason')):
            diagnostic_reason = decision['reason']
            legacy_reason_corrections += 1
        if handoff.get('completed') is False and diagnostic_reason:
            rejection_reasons[diagnostic_reason] += 1
        model_ids.update(x['skill_outputs'].get('model_id') for x in decisions
                         if x['skill_outputs'].get('model_id'))
        gt = row.get('skill_ground_truth') or {}
        source_valid = (gt.get('source') == 'operator_review_of_observed_contact' and
                        gt.get('independent_of_experimental_offset') is True)
        valid_recording = row.get('recording_verified') is True and not row.get('archived_attempt')
        label = gt.get('initial_class') if source_valid and valid_recording else None
        if label not in ('needs_recovery', 'already_normal', 'not_applicable'):
            label = None
        target = gt.get('final_target_reached') if source_valid and valid_recording else None
        target = target if type(target) is bool else None
        coverage['trials'] += 1
        coverage['with_skill_decisions' if decisions else 'no_skill_decisions'] += 1
        coverage['recording_verified' if valid_recording else 'incomplete_recording'] += 1
        coverage['initial_' + (label or 'unlabelled')] += 1
        coverage['final_labelled' if target is not None else 'final_unlabelled'] += 1
        coverage['verified_scope_claims'] += invocation.get('verified') is True
        coverage['verified_claims_without_refs'] += bool(invocation.get('verified') is True and not invocation.get('evidence_refs'))
        for event in events:
            coverage['phase_' + str(event.get('phase', 'unknown'))] += 1
        needs = True if label == 'needs_recovery' else False if label == 'already_normal' else None
        _binary_counts(calls, invocation.get('needs_recovery'), needs)
        _binary_counts(actual_calls, invocation.get('invoke'), True if label == 'needs_recovery' else False if label is not None else None)
        domain = False if label == 'not_applicable' else True if label is not None else None
        _binary_counts(applicability, invocation.get('applicable'), domain)
        _binary_counts(support, invocation.get('candidate_supported'), domain)
        calls['unsafe_or_inapplicable_invocation'] += int(label == 'not_applicable' and invocation.get('invoke') is True)
        predicted_exit = handoff.get('completed') if handoff.get('hand_back') is True else None
        _binary_counts(exits, predicted_exit, target)
        exits['false_success'] = exits['FP']
        exits['missed_completion'] = exits['FN']
        exits['unknown_no_terminal_handoff'] += predicted_exit is None
        receipts = [x.get('receipt') for x in row.get('probe_regrasps', [])]
        history = row.get('correction_history', [])
        applied_count = 0
        for index, action in enumerate(history):
            if action.get('terminal') is True:
                coverage['terminal_history_entries'] += 1
                continue
            command = action.get('delta_mm')
            measured = action.get('applied_delta_mm')
            receipt = receipts[applied_count] if not action.get('blocked') and applied_count < len(receipts) else None
            if not action.get('blocked'):
                applied_count += 1
            valid = (isinstance(receipt, dict) and receipt.get('applied') is True and
                     type(command) in (float, int) and type(measured) in (float, int) and
                     np.isfinite([command, measured]).all() and
                     type(receipt.get('action_mm')) in (float, int) and
                     np.isfinite(receipt['action_mm']) and
                     abs(float(receipt['action_mm']) - measured) <= 1e-6 and
                     type(receipt.get('completed_ns')) is int)
            probe = row.get('probe_regrasps', [])[applied_count-1].get('probe') if valid else None
            fresh = bool(valid and probe and type(probe.get('timestamp_ns')) is int and
                         probe['timestamp_ns'] > receipt['completed_ns'])
            actions.append({'trial_path': row['trial_path'], 'index': index,
                            'command_mm': command, 'measured_action_mm': measured,
                            'blocked': bool(action.get('blocked')),
                            'axis':((action.get('skill_decision') or {}).get('recovery',{}).get('action') or {}).get('axis','unknown'),
                            'receipt_expected':not bool(action.get('blocked')), 'receipt_verified': valid,
                            'fresh_post_action_observation': fresh,
                            'tracking_error_mm': abs(command-measured) if valid else None})
        recovered = any(a['trial_path'] == row['trial_path'] and a['receipt_verified'] and a['fresh_post_action_observation'] for a in actions)
        if needs is True:
            recovery['n'] += 1
            if recovered and target is True:
                recovery['successes'] += 1
            elif target is False:
                recovery['failures'] += 1
            else:
                recovery['unknown'] += 1
            recovery['rejected'] += invocation.get('invoke') is not True
        # Planned condition is used only for experiment-analysis cells.
        perturb = row.get('perturbation') or {}
        force = row['session_meta'].get('config', {}).get('force', {})
        condition = {'a_mm': row.get('delta_mm'),
                     'contact_height_l_mm': perturb.get('l_mm', 0.),
                     'probe_force': row.get('probe_force_target',force.get('target', 0)*force.get('probe_fraction', 0))}
        key = json.dumps(condition, sort_keys=True)
        cell = conditions.setdefault(key, {'conditions': condition, 'n': 0, 'failures': 0,
                                           'missing_labels': 0, 'evidence_refs': []})
        cell['n'] += 1
        cell['failures'] += int(target is False or (label == 'not_applicable' and invocation.get('invoke') is True))
        cell['missing_labels'] += int(label is None or target is None)
        cell['evidence_refs'].append(str(Path(row['trial_path'])/'meta.json'))
        details.append({**provenance(row), 'label_source_verified': source_valid,
                        'initial_class': label, 'final_target_reached': target,
                        'invocation': invocation, 'terminal_exit': handoff,
                        'initial_model_decision': decision,
                        'diagnostic_terminal_reason': diagnostic_reason,
                        'skill_cost': row.get('skill_cost'), 'model_id': first.get('model_id')})
    for counter in (calls, applicability, support, exits):
        counter['n'] = coverage['trials']
        counter['labelled_n'] = sum(counter[k] for k in ('TP','TN','FP','FN'))
    errors = [a['tracking_error_mm'] for a in actions if a['tracking_error_mm'] is not None]
    recovery = {**{k:0 for k in ('n','successes','failures','unknown','rejected')}, **dict(recovery)}
    recovery.update(action_attempts=len(actions), measured_actions=len(errors),
                    blocked_actions=sum(a['blocked'] for a in actions),
                    missing_receipts=sum(a['receipt_expected'] and not a['receipt_verified'] for a in actions),
                    missing_fresh_post_action=sum(a['receipt_expected'] and not a['fresh_post_action_observation'] for a in actions),
                    tracking_mae_mm=float(np.mean(errors)) if errors else None,
                    tracking_p90_mm=float(np.quantile(errors, .9)) if errors else None)
    evidence = [str(Path(s['path'])/'session_meta.json') for s in sessions]
    evidence += [str(Path(d['trial_path'])/'meta.json') for d in details]
    gaps = []
    if coverage['initial_unlabelled'] or coverage['final_unlabelled']:
        gaps.append('Missing independent operator contact-state labels; do not substitute planned offsets.')
    if recovery['missing_receipts'] or recovery['missing_fresh_post_action']:
        gaps.append('Missing measured action receipt or fresh post-action tactile observation.')
    if not coverage['initial_not_applicable']:
        gaps.append('No labelled temporarily-inapplicable contacts; invocation range not tested.')
    if not coverage['initial_already_normal']:
        gaps.append('No labelled already-normal contacts; unnecessary-call rate not tested.')
    if len(model_ids) > 1:
        gaps.append('Multiple frozen models present; pooled counts are descriptive only and cannot certify one model.')
    report = {'schema':'skill_observed_B_evaluation_v1', 'role':role,
              'label_definition':'independent operator review of observed initial/final contact, never experimental delta',
              'hardware_signature':signatures[0], 'simulated':False,
              'source_sessions':[{k:s[k] for k in ('path','metadata_sha256','role')} for s in sessions],
              'model_ids':sorted(model_ids), 'coverage':dict(coverage),
              'invocation':{**dict(calls), 'actual_invoke_confusion':dict(actual_calls)}, 'applicability':dict(applicability),
              'candidate_support':dict(support), 'recovery':dict(recovery), 'exit':dict(exits),
              'rejection_diagnostics':{'terminal_reasons':dict(rejection_reasons),
                  'legacy_reason_corrections':legacy_reason_corrections,
                  'note':'Diagnostic only; original terminal_exit remains unchanged.'},
              'actions':actions, 'cells':list(conditions.values()), 'details':details,
              'gaps':gaps, 'evidence_refs':evidence, 'certificate_issued':False,
              'applicability_interpretation':'applicable means verified deployable scope; candidate_support measures physical candidate domain. Development mode may conservatively report applicable=false.',
              'independence':'source role enforced; operator label independence is declared by recorded protocol, not externally proven'}
    if out is not None:
        if Path(out).exists():
            raise ValueError('evaluation output is immutable; choose a fresh path')
        write_json(out,report)
    return report


def feedback_cli(argv=None):
    """Evaluate real recorded B, or propose next experiments from development only."""
    parser = argparse.ArgumentParser(description=feedback_cli.__doc__)
    sub=parser.add_subparsers(dest='command', required=True)
    ev=sub.add_parser('evaluate')
    ev.add_argument('--session',action='append',required=True)
    ev.add_argument('--role',choices=['development','independent'],required=True)
    ev.add_argument('--out',required=True)
    cert=sub.add_parser('certificate')
    cert.add_argument('--report',required=True)
    cert.add_argument('--fit',required=True)
    cert.add_argument('--config',default='config.skill.bottle.yaml')
    cert.add_argument('--out',required=True)
    cert.add_argument('--minimum-per-cell',type=int,default=30)
    cert.add_argument('--minimum-lower-bound',type=float,default=.8)
    fb=sub.add_parser('feedback')
    fb.add_argument('--report',required=True)
    fb.add_argument('--config',default='config.skill.bottle.yaml')
    fb.add_argument('--exploration-config')
    fb.add_argument('--out',required=True)
    fb.add_argument('--round-id',default='next')
    fb.add_argument('--mode',choices=['fixed','simple_feedback','agent_evidence'],default='agent_evidence')
    fb.add_argument('--stage',choices=['a_closed_loop','height_controlled','force_controlled','small_combinations'],default='a_closed_loop')
    fb.add_argument('--budget',type=int,default=12)
    fb.add_argument('--seed',type=int,default=20261105)
    fb.add_argument('--memory')
    fb.add_argument('--history-off',action='store_true')
    args=parser.parse_args(argv)
    if args.command == 'evaluate':
        report=evaluate_skill_sessions(args.session,role=args.role,out=args.out)
        print(json.dumps({k:report[k] for k in ('role','coverage','invocation','recovery','exit','rejection_diagnostics')},indent=2))
        return 0
    import yaml
    if args.command == 'certificate':
        spec_path=Path(args.config).resolve()
        spec=yaml.safe_load(spec_path.read_text())
        base=yaml.safe_load((spec_path.parent/spec['base_config']).read_text())
        report=json.loads(Path(args.report).read_text())
        model=json.loads(Path(args.fit).read_text())
        certificate=build_skill_certificate(report,model,base['servo'],spec.get('runtime'),
            minimum_per_cell=args.minimum_per_cell,minimum_lower_bound=args.minimum_lower_bound,
            certification=spec.get('certification',{}))
        if Path(args.out).exists():
            raise ValueError('certificate output is immutable')
        write_json(args.out,certificate)
        print(json.dumps({'status':certificate['status'],'validated_cells':len(certificate['observed_cells']),
                          'reasons':certificate['reasons']},indent=2))
        return 0
    from .skill_exploration import (default_exploration_config, choose_experiments,
        record_round, memory_context, persist_method_experience)
    from .rsi_methods import MethodStore, baseline_method
    report=json.loads(Path(args.report).read_text())
    if report.get('schema') != 'skill_observed_B_evaluation_v1' or report.get('role') != 'development':
        raise ValueError('feedback requires an observed development-only report; independent reports cannot guide updates')
    # Revalidate source roles and labels instead of trusting an edited report.
    roots=[str(Path(ref).parent) for ref in report['evidence_refs'] if Path(ref).name=='session_meta.json']
    checked=evaluate_skill_sessions(roots,role='development')
    if _hash(checked) != _hash(report):
        raise ValueError('report no longer matches original source evidence')
    root=Path(args.out)
    if root.exists() and any(root.iterdir()):
        raise ValueError('feedback output is immutable; use a fresh directory')
    spec_path=Path(args.config).resolve()
    spec=yaml.safe_load(spec_path.read_text())
    cfg=yaml.safe_load(Path(args.exploration_config).read_text()) if args.exploration_config else default_exploration_config()
    if not args.exploration_config:
        base=yaml.safe_load((spec_path.parent/spec['base_config']).read_text())
        cfg['variables']['probe_force']['reference']=base['force']['target']*base['force']['probe_fraction']
        cfg['variables']['probe_force']['levels']=[cfg['variables']['probe_force']['reference']]
        cfg['variables']['a_mm']['levels']=spec['collection']['offsets_mm']
        cfg['action_space']['a_mm']=[-base['servo']['max_delta_mm'],base['servo']['max_delta_mm']]
    cfg['history_enabled']=not args.history_off
    selection=choose_experiments(report,cfg,mode=args.mode,seed=args.seed,
                                 remaining_trials=args.budget,stage=args.stage)
    root.mkdir(parents=True,exist_ok=True)
    write_json(root/'selection.json',selection)
    record=record_round(root,args.round_id,feedback=report,selection=selection,
        result={'role':'development','source_report':str(Path(args.report).resolve()),
                'new_selection_execution':'planned_unexecuted'},
        skill_update={'status':'pending_new_experiment','uses_independent_evaluation':False},
        evidence_paths=[args.report]+report['evidence_refs'])
    store=MethodStore(args.memory or spec.get('memory',{}).get('path','outputs/rsi_method_memory'))
    context={'simulated':False,'hardware_signature':report['hardware_signature']}
    write_json(root/'retrieved_memory.json',memory_context(store,context,history_enabled=not args.history_off))
    experience=persist_method_experience(store,baseline_method(),task_id=cfg['task_id'],context=context,
        round_record=record,simulated=False,comparison=None,
        applicability_check={'status':'pending','note':'development observations guide hypotheses; new selection unexecuted; no method promotion'})
    write_json(root/'memory_record.json',experience)
    print(json.dumps({'selection':str(root/'selection.json'),'planned_trials':selection['planned_trial_attempts'],
                     'memory':experience,'hardware_connected':False},indent=2))
    return 0


def sample_efficiency_summary(curve, *, source_attempts, shared_development_trials, mae_mm=1., p90_mm=2.):
    """First observed geometric checkpoint; no monotonicity interpolation."""
    summaries={}
    for mode in sorted({r['mode'] for r in curve}):
        rows=sorted((r for r in curve if r['mode']==mode),key=lambda r:r['selected_A_trials'])
        passed=[r for r in rows if r['independent_A'].get('mae_mm') is not None and
                r['independent_A'].get('p90_mm') is not None and
                r['independent_A']['mae_mm']<=mae_mm and r['independent_A']['p90_mm']<=p90_mm]
        first=passed[0]['selected_A_trials'] if passed else None
        summaries[mode]={'first_observed_A_budget':first,
            'selected_plus_shared_development_trials': first+shared_development_trials if first is not None else None,
            'censored':first is None,'maximum_observed_A_budget':rows[-1]['selected_A_trials'],
            'full_skill_equivalent_capability':'not_established',
            'full_skill_first_budget':None}
    return {'schema':'skill_replay_sample_efficiency_v1',
        'common_geometric_thresholds':{'mae_mm':mae_mm,'p90_mm':p90_mm},
        'branches':summaries,'shared_development_trials':shared_development_trials,
        'historical_source_attempts_separate':source_attempts,'new_physical_interactions':0,
        'interpretation':['First observed checkpoint only; improvement is not assumed monotonic.',
                          'Replay budgets plus shared development are not additional robot interactions.',
                          'Source history overlaps replayed trials and must not be added a second time.',
                          'Full skill equivalence requires prospective invocation, recovery and exit evidence; not established here.']}



def _certificate_cells(spec, required_context):
    cells=spec.get('observation_cells',[])
    if not isinstance(cells,list):
        raise ValueError('certification.observation_cells must be a list')
    seen=set()
    for cell in cells:
        if not isinstance(cell,dict) or set(cell)!={'cell_id','status','context','centroid_bounds_px'}:
            raise ValueError('observation cell requires only cell_id/status/context/centroid_bounds_px')
        if not isinstance(cell['cell_id'],str) or not cell['cell_id'] or cell['cell_id'] in seen:
            raise ValueError('observation cell_id must be unique')
        seen.add(cell['cell_id'])
        if cell['status']!='contact' or not isinstance(cell['context'],dict):
            raise ValueError('this certificate supports contact cells with explicit observed context')
        if any(cell['context'].get(k)!=v for k,v in required_context.items()):
            raise ValueError('cell context must include all frozen required context')
        if any(k in cell['context'] for k in ('delta_mm','planned_delta_mm','trial_id','bottle_truth')):
            raise ValueError('experiment assignments cannot define deployable observation cells')
        bounds=np.asarray(cell['centroid_bounds_px'])
        if bounds.shape!=(2,2,2) or bounds.dtype.kind not in 'fi' or not np.isfinite(bounds).all() or np.any(bounds[:,:,0]>bounds[:,:,1]):
            raise ValueError('centroid bounds must be finite [finger][uv][lower,upper]')
    for i,cell in enumerate(cells):
        for other in cells[:i]:
            contexts_compatible=all(other['context'].get(k,v)==v for k,v in cell['context'].items())
            if contexts_compatible and all(max(cell['centroid_bounds_px'][f][j][0],other['centroid_bounds_px'][f][j][0]) <=
                min(cell['centroid_bounds_px'][f][j][1],other['centroid_bounds_px'][f][j][1])
                for f in range(2) for j in range(2)):
                raise ValueError('preregistered observation cells overlap')
    return cells

def build_skill_certificate(report, model, servo, config=None, *, minimum_per_cell=30, minimum_lower_bound=.8, certification=None):
    """Certificate for preregistered observation cells from independent prospective B.

    Without predeclared cells, exact repeated points are the only fallback.
    Cells must be frozen in every source session BEFORE evaluation. Each cell needs
    repeated independently labelled correct invocation/recovery/exit episodes,
    fresh measured action receipts and two stable final observation windows.
    Sparse old data naturally produces an explicit empty observed_cells.
    """
    from .recovery_skill import RecoverySkill
    if report.get('role') != 'independent' or report.get('schema') != 'skill_observed_B_evaluation_v1':
        raise ValueError('certificate requires prospective independent observed B evaluation')
    if type(minimum_per_cell) is not int or minimum_per_cell < 30 or not .8 <= minimum_lower_bound < 1.:
        raise ValueError('certificate requires >=30 trials per exact cell and >=0.8 Wilson lower bound')
    roots=[str(Path(ref).parent) for ref in report['evidence_refs'] if Path(ref).name=='session_meta.json']
    verified=evaluate_skill_sessions(roots,role='independent')
    if _hash(verified)!=_hash(report):
        raise ValueError('independent report does not match raw evidence')
    skill=RecoverySkill(model,servo,config)
    certification=deepcopy(certification or {})
    declared=_certificate_cells(certification,skill.config['required_context'])
    certificate={'schema':'skill_validation_certificate_v1','model_id':skill.model_id,
        'model_sha256':skill.model_sha256,'config_sha256':skill.config_sha256,
        'observed_cells':[],'status':'insufficient_evidence','reasons':[],
        'evidence_refs':report['evidence_refs'],
        'statistical_rule':{'minimum_per_cell':minimum_per_cell,
                            'wilson_95_lower_bound':minimum_lower_bound},
        'certification_sha256':_hash(certification),
        'scope_semantics':('preregistered observation bins under the tested sampling protocol; no guarantee for every unobserved state' if declared else 'exact observed tactile centroid points and exact deployable context; no interpolation')}
    for root in roots:
        meta=json.loads((Path(root)/'session_meta.json').read_text())
        if _hash(meta.get('config',{}).get('recovery_skill',{}).get('certification',{})) != _hash(certification):
            certificate['reasons'].append('preregistered observation cells differ from the frozen independent session specification')
            return certificate
        recorded_model=meta.get('fit')
        recorded_cfg=meta.get('config',{})
        if not isinstance(recorded_model,dict) or _hash(recorded_model)!=skill.model_sha256:
            certificate['reasons'].append('session does not bind the exact frozen model artifact')
            return certificate
        try:
            recorded_skill=RecoverySkill(recorded_model,recorded_cfg['servo'],
                                          recorded_cfg['recovery_skill'].get('runtime'))
        except (KeyError,TypeError,ValueError):
            certificate['reasons'].append('session lacks complete frozen runtime/servo provenance')
            return certificate
        if recorded_skill.config_sha256!=skill.config_sha256:
            certificate['reasons'].append('session runtime/servo configuration differs from requested certificate')
            return certificate
    if report['model_ids'] != [skill.model_id]:
        certificate['reasons'].append('independent evidence does not exclusively match this frozen model')
        return certificate
    groups={}
    for detail in report['details']:
        row=json.loads((Path(detail['trial_path'])/'meta.json').read_text())
        events=[e for e in row.get('skill_observations',[]) if isinstance(e.get('skill_outputs'),dict)]
        if not events:
            continue
        first=events[0]['observation']; desc=first.get('descriptors',[])
        try:
            centers=[[float(x) for x in d['centroid_px']] for d in desc]
            if len(centers)!=2 or any(len(c)!=2 for c in centers) or not np.isfinite(centers).all():
                continue
        except (KeyError,TypeError,ValueError):
            continue
        context=first.get('context') or {}
        if any(context.get(k)!=v for k,v in skill.config['required_context'].items()):
            continue
        if first.get('status')!='contact':
            continue
        if declared:
            matched=[cell for cell in declared if cell['status']==first.get('status') and
                all(context.get(k)==v for k,v in cell['context'].items()) and
                all(cell['centroid_bounds_px'][i][j][0]<=centers[i][j]<=cell['centroid_bounds_px'][i][j][1]
                    for i in range(2) for j in range(2))]
            if not matched:
                continue
            cell=matched[0]
            key=cell['cell_id']
            bounds=cell['centroid_bounds_px']
            cell_context=cell['context']
        else:
            key=json.dumps({'centers':centers,'context':context},sort_keys=True)
            bounds=[[[v,v] for v in c] for c in centers]
            cell_context=context
        group=groups.setdefault(key,{'bounds':bounds,'context':cell_context,'n':0,'correct':0,'refs':[],
                                     'cell_id':key if declared else 'exact_'+_hash(key)[:12]})
        group['n']+=1
        group['refs'].append(str(Path(detail['trial_path'])/'meta.json'))
        label=detail['initial_class']
        inv=detail['invocation']
        correct_call=(label in ('needs_recovery','already_normal') and
            inv.get('needs_recovery')==(label=='needs_recovery') and inv.get('candidate_supported') is True)
        actions=[a for a in report['actions'] if a['trial_path']==detail['trial_path']]
        action_ok=(label=='already_normal' and not actions) or (label=='needs_recovery' and actions and
            all(a['receipt_verified'] and a['fresh_post_action_observation'] and
                a['tracking_error_mm']<=skill.config['tracking_tolerance_mm'] for a in actions))
        end=detail['terminal_exit']
        required_windows=skill.config['target'].get('confirmation_windows',2)
        windows=[e['observation'] for e in events[-required_windows:]]
        stamps=[o.get('timestamp_ns') for o in windows]
        stable=(len(windows)==required_windows and all(type(t) is int for t in stamps) and all(b-a>=skill.config['target']['min_stable_duration_s']*1e9 for a,b in zip(stamps,stamps[1:])) and all(
            o.get('status')=='contact' and o.get('bilateral') is True and o.get('force_stable') is True and
            type(o.get('now_ns')) is int and 0<=o['now_ns']-o['timestamp_ns']<=skill.config.get('max_observation_age_s',2.)*1e9 and
            type(o.get('stable_duration_s')) in (float,int) and
            o['stable_duration_s']>=skill.config['target']['min_stable_duration_s'] and
            len(o.get('descriptors',[]))==2 and all(d.get('valid') is True and not d.get('touches_edge')
                                                   for d in o['descriptors']) for o in windows))
        receipts=[r.get('receipt') for r in row.get('probe_regrasps',[]) if isinstance(r.get('receipt'),dict)]
        if receipts:
            stable=bool(stable and type(receipts[-1].get('completed_ns')) is int and stamps[0]>receipts[-1]['completed_ns'])
        final_ok=detail['final_target_reached'] is True and end.get('completed') is True and end.get('hand_back') is True
        group['correct']+=int(bool(correct_call and action_ok and stable and final_ok))
    for group in groups.values():
        n=group['n']; p=group['correct']/n; z=1.959963984540054
        lower=(p+z*z/(2*n)-z*np.sqrt(p*(1-p)/n+z*z/(4*n*n)))/(1+z*z/n)
        if n<minimum_per_cell or lower<minimum_lower_bound:
            continue
        certificate['observed_cells'].append({'status':'contact','context':group['context'],
            'centroid_bounds_px':group['bounds'],'cell_id':group['cell_id'],
            'invocation_validated':True,'recovery_validated':True,'exit_validated':True,
            'evidence_refs':group['refs'],'n':n,'joint_correct':group['correct'],'wilson_lower':float(lower)})
    if certificate['observed_cells']:
        certificate['status']='validated_preregistered_observation_cells' if declared else 'validated_exact_observation_cells'
    else:
        certificate['reasons'].append('no preregistered (or exact fallback) cell has enough independently labelled, fresh, measured joint successes')
    return certificate

if __name__ == "__main__":
    raise SystemExit(main())
