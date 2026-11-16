"""One isolated fixed5 trial at the nominal contact. No director or automatic retry."""
import argparse
from datetime import datetime
from pathlib import Path
from time import time_ns

from bottle_scan.state_graph.campaign import atomic, collect_pending, initialize, queue_plan, read
from .campaign import launch_request


def prepare_once(out):
    root = Path(__file__).resolve().parents[2]
    cfg = read(root/'config.state_graph.campaign_v4.json')
    cfg.update(ready_mode='fixed5', sources=[])
    atomic(out/'config.json', cfg)
    state = initialize(out/'config.json', out)
    spec = read(out/'spec.json')
    cfg = read(out/'config.json')
    plan = {'cells':[{'settings':{'a_offset_mm':0.0}, 'blocks':[{'mass_class':'heavy'}],
                      'actions':[], 'reps':1}]}
    queue_plan(out, state, spec, plan, 'exploration')
    return state, spec, cfg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--real', action='store_true', help='现场确认后执行一次；默认只生成计划')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    parent = root/'outputs/state_graph/fixed5_once'
    out = parent / f"{datetime.now():%Y%m%d_%H%M%S}_{time_ns()}"
    out.mkdir(parents=True, exist_ok=False)
    state, spec, cfg = prepare_once(out)
    trial, = state['trials']
    directory = Path(trial['directory'])
    params = spec['post_conditions'][0]['consumer']['params']
    print(f"重白瓶，原 T0，0 mm，无位置纠正，仅 1 次。设备配置下限 {params['closure_floor']:.3f}；"
          f"双指均值 {params['target']} ± {params['tolerance']} SDK 持续 {params['stable_s']} s，"
          f"提起 {params['lift_mm']:g} mm，保持 {params['hold_s']:g} s。", flush=True)
    print(f"本次记录目录 {out}", flush=True)
    if not args.real:
        launch_request(trial['request'], directory, spec, cfg)
        print('离线预览结束。')
        return 0
    collect_pending(out, state, spec, cfg, real=True, interactive=True)
    result_path = directory/'result.json'
    if not result_path.exists():
        print('本次未取得实测终点。')
        return 2
    trial_result = read(result_path)
    for node in trial_result['nodes']:
        post = node['post'][spec['active_post_condition']]
        if post['evaluated']:
            sample = post['evidence'][0]['evidence'][0]
            print('达力={}，夹紧结束开度={}，提起={}，保持完成={}，检测掉落={}'.format(
                sample['target_reached'], sample['squeeze_end_opening_normalized'], sample['lifted'],
                sample['hold_complete'], sample['dropped']))
            return 0 if post['success'] else 1
    print('本次未取得实测终点。')
    return 2


if __name__ == '__main__':
    raise SystemExit(main())
