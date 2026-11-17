"""Replay P3 lightweight-contact archives for the representation study (no robot interfaces)."""
import argparse,json
from pathlib import Path
import numpy as np
from .common import write
from .v5_profile import configure,protect_output
from .modalities.tactile_embedding import FrozenEncoder,compose,spatial_readout
from .representation_eval import evaluate


def recorded_frames(row):
    import h5py
    path=Path(row['provenance']['source_file'])
    probe=json.loads(path.read_text())['probe']['window']
    h5=path.parent/'record'/'streams.h5';arrays={}
    with h5py.File(h5) as handle:
        for i,side in enumerate(('left','right')):
            g=handle['streams/dd_encoded_'+side];expected=probe[i]['frame_ns']
            stamps=np.asarray(g['monotonic_ns'])
            if len(set(expected))!=5:raise ValueError('missing distinct frame stamps')
            indices=np.searchsorted(stamps,expected)
            if max(indices)>=len(stamps) or list(stamps[indices])!=list(expected):raise ValueError('no exact frame stamp match')
            arrays[side]=g['data'][indices]
    return arrays,str(h5)


def encode_rows(rows,settings,cfg):
    import torch
    methods=('mean','attention','grid','soft_argmax')
    base=FrozenEncoder(settings);base._load()
    versions={m:FrozenEncoder({**settings,'readout':m,'grid':cfg['tac_grid'],'temperature':cfg['tac_temperature']}).version for m in methods}
    evidence=[]
    for row in rows:
        try:arrays,path=recorded_frames(row)
        except (OSError,KeyError,ValueError) as exc:
            evidence.append({'node_id':row['node_id'],'status':'missing','reason':str(exc)});continue
        count=len(arrays['left']);x=base.prepare(np.concatenate([arrays['left'],arrays['right']])).to(base.device)
        with torch.inference_mode():
            tokens,_,restore=base.model.forward_encoder(x,mask_ratio=0.)
            patches=tokens[:,1:].gather(1,restore[:,:,None].expand(-1,-1,tokens.shape[-1]))
            for m in methods:
                z=spatial_readout(x,patches,base.model.config['patch_size'],m,cfg['tac_grid'],cfg['tac_temperature']).cpu().numpy()
                value=compose(np.median(z[:count],axis=0),np.median(z[count:],axis=0),base.representation)
                row.setdefault('representations',{})[m]=value.tolist()
                row.setdefault('representation_versions',{})[m]=versions[m]
        evidence.append({'node_id':row['node_id'],'status':'computed','source_file':path,'frames_per_side':count})
    return evidence


def evidence_note(rows,out,report):
    lookup={r['node_id']:r for r in rows}
    from .representation_eval import raw_candidate
    edges=[]
    for target in rows:
        source=lookup.get(target.get('previous_node_id'));receipt=target.get('u_receipt') or {}
        if not source or not receipt.get('applied'):continue
        response={}
        for name in ('legacy_geometry','force','mean','attention','grid','soft_argmax'):
            a=raw_candidate(source,name);b=raw_candidate(target,name)
            if a is not None and b is not None:response[name]=float(np.linalg.norm(np.subtract(b,a)))
        edges.append(dict(source=source['node_id'],target=target['node_id'],executed_mm=target['u_executed'],distance=response))
    write(out/'edge_responses.json',{'scope':'raw within-representation L2 change; different dimensions and units, not comparable between readouts', 'edges':edges})
    text=['# P3 触觉表征离线复核','',f"同一白瓶任务的 {report['group_count']} 个试次组产生 {len(rows)} 个窗口。其中 {report['raw_encoded_windows']} 个窗口按轻触时间戳精确找回了双指各五帧编码 DD。冻结 MAE 的 mean、attention、grid 和 soft-argmax 读出均由实际数据计算。",'',
          f"记录中有 {len(edges)} 条有执行回执的状态转移。逐边保存的原始特征差使用各自表征的单位和维数，仅供同一表征内部复核。",'',
          '本批缺少可匹配的严格重复轻触和独立 r/h 位置标签，不能计算动作分辨率与重复噪声之比或毫米预测误差。跨角度同位姿缺配对证据。下游未评估，AUC 无法估计。按组留出的训练集无法标定可用距离，系统对已有表征的留出窗口均无支撑（unsupported = 100%）。未取得有效特征的候选保留缺失值。','',
          '指面毫米标定及瓶肩场景标签仍缺失。当前数据未使任何候选表征满足覆盖全部场景的切换条件。']
    (out/'README_zh.md').write_text('\n'.join(text)+'\n')


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--campaign',required=True)
    p.add_argument('--artifact',required=True);p.add_argument('--out',required=True)
    p.add_argument('--device',default='cpu');p.add_argument('--threads',type=int,default=2)
    a=p.parse_args(argv)
    import torch
    torch.set_num_threads(a.threads);campaign=Path(a.campaign);out=Path(a.out)
    protect_output(out,kind='p3_representation_eval')
    if (out/'ranking.json').exists():raise FileExistsError(out/'ranking.json')
    spec=configure(json.loads((campaign/'spec.json').read_text()));obs=spec['observation']
    rows=[r for r in json.loads((campaign/'seed_nodes.json').read_text()) if r['session']=='P3']
    gids={r['node_id']:r for r in rows}
    for trial in json.loads((campaign/'state.json').read_text())['trials']:
        if trial['status']!='complete':continue
        path=Path(trial['directory'])/'result.json';result=json.loads(path.read_text())
        for r in result.get('nodes',[]):
            if r['session']=='P3' and not r.get('evaluation_only'):gids[r['node_id']]=r
    rows=list(gids.values())
    settings={'artifact':a.artifact,'cache_dir':str(out/'cache'),'device':a.device,'representation':'concat'}
    spec['tactile_encoder']=settings
    evidence=encode_rows(rows,settings,obs)
    # Only an observed a-only motion identifies an a-axis trial. Height within
    # the body/shoulder and matching offsets across tilts need independent annotations.
    a_groups={r['group_id'] for r in rows if len(r.get('u_executed') or [])==2 and r['u_executed'][0] and not r['u_executed'][1]}
    for r in rows:
        if r['group_id'] in a_groups:r['representation_scenes']=['a_axis']
    write(out/'feature_sources.json',evidence)
    with (out/'features.jsonl').open('w') as h:
        for r in rows:h.write(json.dumps({k:r.get(k) for k in ('node_id','representations','representation_scenes','representation_versions')},separators=(',',':'))+'\n')
    report=evaluate(rows,spec)
    report.update(node_count=len(rows),group_count=len(set(r['group_id'] for r in rows)),source_campaign=str(campaign),
                  absent_evidence=['finger-plane millimetre calibration','matched independent r/h labels at multiple tilt angles',
                                   'height labels distinguishing bottle body and shoulder','independent repeat settings for here-at-contact P3'],
                  raw_encoded_windows=sum(e['status']=='computed' for e in evidence))
    write(out/'ranking.json',report)
    evidence_note(rows,out,report)
    print(json.dumps({k:report[k] for k in ('node_count','group_count','raw_encoded_windows','selectable')},ensure_ascii=False))
    return 0


if __name__=='__main__':raise SystemExit(main())
