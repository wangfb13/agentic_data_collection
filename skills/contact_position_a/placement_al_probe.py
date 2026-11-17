"""Five controlled contact measurements around an existing placement center; no learned actions."""
import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from bottle_scan.geometry import validate_transform
from bottle_scan.hardware_arm import pose_error
from . import placement_probe as single

# Alternating axes keeps the nominal longest move at sqrt(3**2 + 3**2) mm.
POINTS = [('plus', 3, 0), ('l_plus', 0, 3), ('minus', -3, 0), ('l_minus', 0, -3), ('center_repeat', 0, 0)]


def contact(path, ref, model, serials):
    row = single.read(path)
    if (row['schema'] != single.LABEL or row['placement'] != path.parent.parent.name
            or row['trial'] != path.parent.name or row['model_id'] != model['model_id']
            or row['reference_digest'] != ref['reference_digest'] or row.get('finger_serials') != serials
            or row['state'] != 'released' or row.get('status') != 'contact' or not row.get('bottle_static')
            or not (row.get('probe') or {}).get('valid')):
        raise ValueError(f'轻触缺失、瓶位未确认或实验配置不一致，请查看 {path}')
    fb = row['window_pose']
    if fb['base_frame'] != 'base_link' or fb['tcp_frame'] != 'right_gripper_base_link':
        raise ValueError('TCP frame differs from recorded center')
    return validate_transform(fb['T_base_tcp_mm']), row


def child(*args, **kwargs):
    subprocess.run([sys.executable, *map(str,args)], check=True, **kwargs)


def run(args, invoke=child):
    cfg, ref, model, serials = single.assets()
    replay = single.load_replay()
    folder = args.run / args.placement
    center, _ = contact(folder/'center'/'observation.json', ref, model, serials)
    basis = np.column_stack([cfg['axes_tcp'][k] for k in ('a','c','l')])

    for name, a, l in POINTS:
        obs = folder/name/'observation.json'
        local = basis @ [a, 0, l]
        label = f'{name} (a={a:+d}, l={l:+d} mm)'
        if not obs.exists():
            if obs.parent.exists():
                raise ValueError(f'请先核对此未完成的试次 {obs.parent}')
            if not args.real:
                print('待执行',label)
                continue
            print(label, '确认瓶子仍在本组原位、手已全开，当前夹爪整条路径离桌面与瓶壁有间隙。', flush=True)
            prev = args.run/'moves'
            prev.mkdir(parents=True,exist_ok=True)
            with tempfile.TemporaryDirectory() as tmp:
                measurement = Path(tmp)/'pose.json'
                invoke('record_pose.py','--arm','right','--label','pre_move','--output',measurement,stdout=subprocess.DEVNULL)
                feedback = single.read(measurement)
            g = feedback['gripper']
            if not (g['fresh'] and g['valid'] and abs(g['normalized']-cfg['gripper']['open_normalized']) <= cfg['gripper']['open_tolerance']):
                raise ValueError('右爪未全开')
            actual = validate_transform(feedback['T_base_tcp_mm'])
            if pose_error(center,actual)[1] > 1:
                raise ValueError('当前朝向与本组首次轻触相差超过1°')
            # Translation only; the placement origin is used solely for controlled perturbations.
            xyz = center[:3,3] + actual[:3,:3] @ local
            logs = Path(tempfile.mkdtemp(dir=prev,prefix=args.placement+'_'+name+'_'))
            invoke('move_tcp.py','--arm','right','--xyz-mm',*xyz,'--speed-mm-s',2,'--max-distance-mm',6.1,
                   '--pose-tolerance-mm',2,'--log-dir',logs,'--real')
            receipts = list(logs.glob('tcp_move_*.json'))
            if len(receipts) != 1 or single.read(receipts[0])['result'] != 'reached_by_measured_feedback':
                print('移动未确认到位，本次停在轻触前。')
                return 1
            invoke('-m','skills.contact_position_a.placement_probe','probe','--run',args.run,'--placement',
                   args.placement,'--trial',name,'--real')
            if not obs.exists():
                print('本次已取消。')
                return 1
        pose, row = contact(obs,ref,model,serials)
        da, dc, dl = basis.T @ center[:3,:3].T @ (pose[:3,3]-center[:3,3])
        pred = replay(model,row['probe']['descriptors'])
        print(f'{label}  首触时实测 Δa={da:+.2f}  Δl={dl:+.2f}  Δc={dc:+.2f} mm；'
              f'原模型原始 a 估计={pred["raw_offset_a_mm"]:+.2f} mm，{pred["sample_gate"]}',flush=True)
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--placement',type=single.tag,default='P2')
    p.add_argument('--real',action='store_true')
    return run(p.parse_args())


if __name__ == '__main__':
    raise SystemExit(main())
