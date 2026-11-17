"""Independent trial starts for tactile a/h collection."""
from pathlib import Path

from copy import deepcopy
import shutil
from tempfile import TemporaryDirectory, mkdtemp

from bottle_scan.state_graph import campaign
from .ah_geometry import COORDS, position_limits, validate_cell

MODE = 'per_trial_current_tcp_ah_v2'
PHASE = ('每次由操作者对准本次瓶身，从当前位置轻触。组名仅用于归档，不指明世界位置或朝向。'
         '起始 a/h 偏移未知，不继承历史起点。一次试次内瓶子保持静止。'
         '一次可仅轻触，或沿动作开始时 TCP a 或基座 h 移动一次并重触；优先 +h。'
         '相对本次起点，a 允许 −10 到 +10 mm，h 允许 −3 到 +45 mm。可在范围内选任意实数幅度。')


def set_trial_protocol(spec, execution):
    execution.update(start_mode=MODE,phase=PHASE)
    execution['sampling_preference']['instruction']=PHASE
    space=spec['task']['action_space']
    space.pop('candidate_actions', None)
    space.pop('candidate_levels', None)
    space['continuous_displacement_variables'] = list(COORDS)
    for key in COORDS:
        spec['perturbation_variables'][key].pop('nominal_step', None)


def continuous_rule(rule, spec):
    bounds = {k: spec['perturbation_variables'][k]['envelope'] for k in COORDS}
    if all(rule['variables'][k]['range'] == v for k, v in bounds.items()):
        return rule
    updated = deepcopy(rule)
    for key, value in bounds.items():
        updated['variables'][key]['range'] = list(value)
    parent = rule['rule_id']
    updated.update(rule_id='rule_'+campaign.digest([parent, 'continuous_displacements', bounds])[:16],
                   parent_rule_id=parent, reason=PHASE, evidence_refs=[])
    return updated


def upgrade(out, previous):
    # Do not mutate a running, interrupted or frozen experimental protocol.
    cfg=campaign.read(out/'config.json')
    execution=cfg['execution']
    state=campaign.read(out/'state.json')
    spec=campaign.read(out/'spec.json')
    if state.get('frozen') or state.get('complete'):
        return
    if execution['start_mode'] != MODE:
        if execution['start_mode'] != 'current_contact_ah_v1':
            raise ValueError('未知自由起点协议版本')
        if any(t['status']=='started' for t in state['trials']):
            raise ValueError('旧运动协议有未结算试次；请先结算')
        for t in state['trials']:
            if t['status']=='queued':
                req=t['request']
                validate_cell(dict(settings=req['settings'],actions=req['actions'],blocks=[req['block']]),spec,'',{})
        old_basis=previous/'placement_basis.json'
        if old_basis.is_file():
            z=campaign.read(old_basis)['measured_pose'][2][3]
            inherited=(z+position_limits(spec)[1]).tolist()
            if execution.get('table_tcp_z_mm')==inherited:
                execution.pop('table_tcp_z_mm')
    set_trial_protocol(spec,execution)
    state['rule'] = continuous_rule(state['rule'], spec)
    for name,value in [('instance',spec),('spec',spec),('input',cfg),('config',cfg),('state',state)]:
        path = out/(name+'.json')
        if campaign.read(path) != value:
            campaign.atomic(path,value)


def initialize(out, placement, previous):
    out, previous = Path(out).resolve(), Path(previous).resolve()
    if placement in ('P1', 'P2') or out == previous:
        raise ValueError('新摆放需要独立组名和输出目录')
    with campaign.ownership(out):
        if (out/'state.json').exists():
            spec = campaign.read(out/'spec.json')
            if spec.get('current_placement') != placement:
                raise ValueError('该目录已有其他采集组或协议')
            upgrade(out, previous)
            return
        with campaign.ownership(previous):
            spec = campaign.read(previous/'spec.json')
            cfg = campaign.read(previous/'config.json')
            prior = campaign.read(previous/'state.json')
            inherited = campaign.read(previous/'history_windows.json')
            for t in prior['trials']:
                if t['status'] == 'complete' and t['request']['role'] == 'exploration':
                    path = Path(t['directory'])/'windows.json'
                    if path.is_file():
                        inherited.extend(campaign.read(path))
            source = dict(campaign=str(previous), spent=prior['spent'], imported_trials=len(inherited))
        spec.pop('placement_geometry')
        spec['current_placement'] = placement
        remaining = spec['budget']['max_new_trials'] - prior['spent']['trials']
        if remaining <= 0:
            raise ValueError('原采集预算已用尽')
        spec['budget']['max_new_trials'] = remaining
        for key in COORDS:
            spec['perturbation_variables'][key].pop('reference', None)
        spec['perturbation_variables']['approach_pose'].pop('reference', None)
        spec['perturbation_variables']['l_offset_mm'].pop('reference', None)
        execution = cfg['execution']
        execution.pop('center_file')
        execution['executable_variables']=['mass_class']
        execution['fixed_settings']={'closure_floor':.33}
        execution['placement']=placement
        set_trial_protocol(spec,execution)
        cfg.update(spec=str(out/'instance.json'), sources=[str(out/'history_windows.json')])
        for name, value in [('history_windows', inherited), ('instance', spec), ('input', cfg),
                            ('imported_exploration', source)]:
            campaign.atomic(out/(name+'.json'), value)
        campaign.initialize(out/'input.json', out)
        upgrade(out, previous)



def preview_plan(out, placement, *, provider=None):
    """Take an atomic state snapshot; never recover, dispatch or replace a pending batch."""
    out = Path(out).resolve()
    state = campaign.read(out/'state.json')
    spec = campaign.read(out/'spec.json')
    cfg = campaign.read(out/'config.json')
    if (spec.get('current_placement') != placement or cfg['execution']['start_mode'] != MODE
            or state.get('complete') or state.get('frozen')):
        raise ValueError('预览仅适用于未冻结的自由起点探索')
    set_trial_protocol(spec, cfg['execution'])
    state['rule'] = continuous_rule(state['rule'], spec)
    pending_ids = set((state.get('pending') or {}).get('trial_ids', []))
    pending = [t for t in state['trials'] if t['trial_id'] in pending_ids and t['status'] in ('queued','started','deferred')]
    # Started trials are already charged in spent; queued trials still need reservations.
    reserved = sum(t['status'] in ('queued','deferred') for t in pending)
    remaining = spec['budget']['max_new_trials'] - state['spent']['trials'] - reserved
    if remaining <= 0:
        raise ValueError('为当前批次预留次数后，没有下一批可用预算')
    completed = [t['trial_id'] for t in state['trials'] if t['status']=='complete']
    with TemporaryDirectory() as work:
        snapshot = Path(work)
        shutil.copyfile(out/'seed_nodes.json', snapshot/'seed_nodes.json')
        campaign.next_decision(snapshot, state, spec, cfg, False)
        ctx = campaign.read(snapshot/'director_context.json')
    ctx['budget_left'] = remaining
    ctx['planning_scenario'] = 'conditional_next_batch_preview'
    ctx['preview_assumption'] = ('The current batch finishes and its results do not materially change the '
                                 'gaps shown. Actual pending outcomes and amplitudes remain unknown. This '
                                 'proposal is for review only. Reassess when the results arrive.')
    ctx['pending_at_snapshot'] = [dict(trial_id=t['trial_id'],status=t['status'],block=t['request']['block'],
                                       planned_actions=t['request']['actions']) for t in pending]
    previews = out/'plan_previews'
    previews.mkdir(exist_ok=True)
    dest = Path(mkdtemp(prefix='continuous_', dir=previews))
    answer = campaign.decide(ctx, dest, allow_agent=True, provider=provider,
                             provider_settings=cfg.get('provider_settings'))
    if answer['status'] != 'accepted':
        raise ValueError(f'agent 未产生通过校验的预览计划：{dest}')
    report = dict(preview_only=True, robot_executed=False, source_campaign=str(out),
                  graph_version=ctx['graph_version'], completed_trials=completed,
                  pending=ctx['pending_at_snapshot'], available_new_trials=remaining,
                  decision=answer['answer'])
    campaign.atomic(dest/'plan.json', report)
    return dest/'plan.json'
