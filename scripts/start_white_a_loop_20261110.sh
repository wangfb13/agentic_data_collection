#!/usr/bin/env bash
# One command for the operator; preparation/check mode never opens hardware.
set -euo pipefail
case "${1:-}" in
  --check-only) start_mode=check ;;
  --confirm-current-scene) start_mode=run ;;
  *)
    echo 'Usage: bash scripts/start_white_a_loop_20261110.sh --check-only|--confirm-current-scene'
    echo '确认含义：沿用20261110这轮T0、底盘/桌子/瓶位；开始时全开空载；本轮闭合下限为0.330。'
    echo '连续执行A/联合训练/B/agent补采/独立验证；保留放瓶、扣零、标签和故障停止。'
    exit 2 ;;
esac
if [[ $# != 1 ]]; then echo 'Unexpected arguments' >&2; exit 2; fi
root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root_dir"
bootstrap_python="$root_dir/.venv/bin/python"
if [[ -e /.dockerenv || -e /run/.containerenv ]]; then
  bootstrap_python="$root_dir/.venv_robot/bin/python"
fi
if [[ ! -x "$bootstrap_python" ]]; then echo "找不到环境: $bootstrap_python" >&2; exit 2; fi
loop_out=outputs/white_a_director_live_0330_20261110
mkdir -p outputs
exec 9>outputs/white_a_director_live_20261110.launch.lock
flock -n 9 || { echo '同一一键入口已经运行，请使用原终端。' >&2; exit 2; }
echo '核对历史证据、参考T0和新目标定义；此阶段不连接机器人。' >&2
"$bootstrap_python" - "$start_mode" <<'PY'
import json
from pathlib import Path
import socket
import sys
import yaml
from bottle_scan import skill_loop

root = Path.cwd()
out = root/'outputs/white_a_director_live_0330_20261110'
cfgpath = root/'config.white_bottle_a_20261110.live_0330_director.loop.yaml'
prior = root/'outputs/white_a_director_force_outcome_20261110'
# Historical data/decisions remain evidence after a code upgrade. Never resume
# their old implementation/config or rewrite their frozen protocol.
previous_state = skill_loop.read(prior/'state.json')
previous_config = skill_loop.read(prior/'config.json')
assert skill_loop.digest(previous_config) == previous_state['config_hash']
assert skill_loop.source_snapshot(previous_config['reference_session']) == previous_config['reference_snapshot']
for source in previous_state['training_sources']:
    assert skill_loop.source_snapshot(source['path']) == source
for sealed in previous_state['sealed_reports']:
    assert skill_loop.digest(skill_loop.read(sealed['path'])) == sealed['hash']
assert previous_state['new_attempts'] == 0, '历史准备轮已有真机数据，请重新核对启动方案'
seal = previous_state['director_decisions'][-1]
decision = skill_loop.read(seal['path'])
assert skill_loop.digest(decision) == seal['hash']
assert decision['validation']['accepted'] and decision['answer']['action'] == 'collect_A'
config = yaml.safe_load((root/'config.white_bottle_a_20261110.force_outcome_director.loop.yaml').read_text())
assert config['target_definition'] == previous_config['target_definition']
assert config['operator_feedback'] == previous_config['operator_feedback']
assert previous_config['base_config']['gripper']['min_normalized'] == .336
config['physical_setup_confirmed'] = True
config['execution_overrides'] = {'gripper': {'min_normalized': .330}}
config['operator_feedback']['execution_protocol_change'] = {
    'operator_request': '闭合下限我改成0.330吧',
    'old_closure_floor': .336, 'current_closure_floor': .330,
    'meaning': '旧22次和旧agent计划使用0.336；本轮按用户明确要求使用0.330。'
               '历史未达力统计不能直接当作新下限的结果；请按当前协议重新规划。'}
config['operator_feedback']['previous_director_decision'] = {
    'path': seal['path'], 'hash': seal['hash'], 'answer': decision['answer'],
    'meaning': '上一轮真实agent在0.336下限下的补采建议，仅作历史背景；本轮0.330由agent重新决定，不能伪装成新调用。'}
config['operator_feedback']['training_provenance_note'] = (
    '旧22次保留为历史证据与实测T0参考，因其位置标签使用旧目标定义，不混入本轮v2训练。'
    '本轮训练源只加入按新目标定义标注的新A及开发B；11/11结果分组仍是研究假设。')
config['operator_launch_confirmation'] = {
    'method': 'operator executes --confirm-current-scene',
    'reference_scene': str(previous_config['reference_session']),
    'closure_floor': .330,
    'meaning': '操作者执行明确确认参数，沿用本轮T0场景、初始开爪空载和已声明下限；不是自动传感器验收。'}
skill_loop.validate_config(config)
if sys.argv[1] == 'check':
    if (out/'state.json').exists():
        _,state,frozen = skill_loop._load(out)
        assert frozen['physical_setup_confirmed'] is True
        assert frozen['base_config']['gripper']['min_normalized'] == .330
        assert frozen['execution_protocol']['gripper']['min_normalized'] == .330
    else:
        state = {'phase': 'not_initialized'}
    print(json.dumps({'check_passed': True, 'phase': state['phase'],
        'new_training_seeds': [], 'historical_trials_retained_in_agent_context': 22,
        'prior_agent_plan': decision['answer']['plan']['cells'],
        'closure_floor': .330, 'reference_recorded_floor': .336, 'hardware_started': False,
        'external_llm_called': False, 'campaign': str(out)}, ensure_ascii=False))
    raise SystemExit(0)
try:
    with socket.create_connection(('127.0.0.1', 18082), timeout=3):
        pass
except OSError:
    raise SystemExit('Agent代理未连接（127.0.0.1:18082）；请先恢复代理，再运行此命令。')
if not (out/'state.json').exists():
    if out.exists():
        raise ValueError('初始化目录不完整；保留现场记录，请检查 '+str(out))
    if cfgpath.exists():
        assert yaml.safe_load(cfgpath.read_text()) == config, '启动配置已有不同内容'
    else:
        with cfgpath.open('x') as f:
            yaml.safe_dump(config, f, allow_unicode=True, sort_keys=False)
    skill_loop.init_campaign(cfgpath, out, seed_sessions=[])
else:
    _,state,frozen = skill_loop._load(out)
    assert frozen['target_definition'] == config['target_definition']
    assert frozen['operator_launch_confirmation'] == config['operator_launch_confirmation']
    assert frozen['physical_setup_confirmed'] is True
_, _, frozen = skill_loop._load(out)
assert frozen['base_config']['gripper']['min_normalized'] == .330
assert frozen['execution_protocol']['gripper']['min_normalized'] == .330
print('进入0.330下限连续闭环；复用本版本进度，不重置已采数据。', file=sys.stderr)
PY
if [[ "$start_mode" == check ]]; then exit 0; fi
exec bash "$root_dir/run_on_5090.sh" skill-loop run \
  --real --allow-agent --auto-stages --out "$loop_out"
