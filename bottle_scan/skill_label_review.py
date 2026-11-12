"""Human review of recorded contact windows; never creates physical trials.

prepare emits only unknown labels, recorded tactile images and descriptor diagrams.
apply requires explicit human annotations and writes a metadata-only view with
raw HDF5 symlinks. Original metadata/raw recordings are never modified.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import html
import json
import math
from pathlib import Path
import shutil
import tempfile
import time

import yaml

REVIEW_SCHEMA='a_skill_label_review_v1'
ANNOTATION_SCHEMA='a_skill_label_annotations_v1'
VIEW_SCHEMA='a_skill_label_view_v1'
TRIAL_SCHEMA='a_skill_label_view_trial_v1'
SOURCE='operator_review_of_observed_contact'
FINAL_ROLES={'validation','independent','final_validation','frozen_validation'}


def _read(path):return json.loads(Path(path).read_text())

def _hash(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def _file_hash(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def _write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')


def _role(meta):
    from .agentic_learning import _session_role
    return _session_role(meta)


def _definition(path):
    value=yaml.safe_load(Path(path).read_text())
    target=value.get('target_definition')
    if not isinstance(target,dict) or any(not isinstance(target.get(k),str) or not target[k].strip()
            for k in ('id','description','normal','needs_recovery','completion')):
        raise ValueError('explicit target_definition requires id, description, normal, needs_recovery, completion')
    return target


def _windows(row):
    result={}
    for index,event in enumerate(row.get('skill_observations') or []):
        obs=event.get('observation') or {};stamp=obs.get('timestamp_ns')
        if type(stamp) is not int or stamp<0:continue
        if stamp in result:
            if result[stamp]['observation']!=obs:
                raise ValueError('conflicting recorded observations share one physical timestamp')
            result[stamp]['phases'].append(event.get('phase'));continue
        result[stamp]={'timestamp_ns':stamp,'event_index':index,'phases':[event.get('phase')],
                       'observation':deepcopy(obs),'observation_sha256':_hash(obs)}
    return list(result.values())


def _raw_proof(folder):
    from .skill_loop import raw_tcp_snapshot
    path=Path(folder)/'streams.h5'
    if not path.exists():return None
    return {'path':str(path.resolve()),'size_bytes':path.stat().st_size,
            'mtime_ns':path.stat().st_mtime_ns,'consumed_tcp_stream':raw_tcp_snapshot(path)}


def _trial_manifest(path):
    path=Path(path)
    if path.parent.is_symlink() or path.is_symlink():raise ValueError('review source must contain original physical metadata, not replay links')
    row=_read(path)
    if row.get('skill_label_review') or row.get('skill_replay_source_trial'):
        raise ValueError('nested trial review forbidden')
    windows=_windows(row);status=path.parent/'recording_status.json'
    return {'id':path.parent.name,'source_trial':str(path.parent.resolve()),
        'source_metadata_sha256':_file_hash(path),
        'recording_status_sha256':_file_hash(status) if status.exists() else None,
        'raw':_raw_proof(path.parent),'windows':windows,
        'initial_timestamp_ns':next((w['timestamp_ns'] for w in windows if 'first_touch' in w['phases']),None),
        'final_contact_timestamp_ns':windows[-1]['timestamp_ns'] if windows else None,
        'recording_reviewable':row.get('complete') is True and row.get('recording_complete') is True and
                               status.exists() and _read(status).get('complete') is True}


def _source_manifest(session,target,reference_trial=None):
    root=Path(session).expanduser().resolve();meta=_read(root/'session_meta.json')
    if meta.get('schema')!='e1e2_session_v1' or _role(meta) not in ('exploration','development'):
        raise ValueError('only original exploration/development A/B sessions may be reviewed for training')
    if meta.get('cond') not in ('A','B') or (meta['cond']=='B' and _role(meta)!='development'):
        raise ValueError('B review requires development evidence')
    if meta.get('skill_label_review') or meta.get('skill_replay_source_session'):
        raise ValueError('review-of-review/replay is forbidden; use the original physical session')
    trials=[]
    for path in sorted((root/'trials').glob('*/meta.json')):
        trials.append(_trial_manifest(path))
    if not trials or not any(t['windows'] for t in trials):raise ValueError('no timestamp-bound contact windows available')
    if reference_trial is not None and reference_trial not in {t['id'] for t in trials}:
        raise ValueError('reference trial must be explicitly selected from this session')
    return {'schema':REVIEW_SCHEMA,'source_session':str(root),'source_session_metadata_sha256':_file_hash(root/'session_meta.json'),
            'source_role':_role(meta),'target_definition':deepcopy(target),'trials':trials,
            'reference_trial':reference_trial,'reference_is_operator_candidate_not_ground_truth':True,
            'label_rule':'Review observed contact only. Planned offsets, TCP displacement, model predictions and lift success are not contact labels.',
            'physical_trials_added':0}


def _diagram(window,reference=None):
    # Explicit geometry plot, no inferred contact mask or synthetic raw texture.
    def points(o):
        values=[]
        for d in o.get('descriptors',[]):
            for key in ('centroid_px','bbox_xyxy_px'):
                v=d.get(key)
                if isinstance(v,list):values.extend(x for x in v if type(x) in (int,float) and math.isfinite(x))
        return values
    bound=max([128.]+points(window['observation'])+(points(reference['observation']) if reference else []))+8.
    scale=200./bound
    parts=['<svg xmlns="http://www.w3.org/2000/svg" width="600" height="280" viewBox="0 0 600 280">',
           '<rect width="600" height="280" fill="white"/>',
           '<text x="12" y="18" font-size="12">Recorded descriptor geometry only — NOT raw DD or contact-mask image</text>']
    for i,side in enumerate(('left','right')):
        x0=35+i*290;y0=45
        parts.append(f'<text x="{x0}" y="35" font-size="12">{side}: u right, v down (pixels)</text>')
        parts.append(f'<path d="M{x0} {y0} v200 h200" stroke="#aaa" fill="none"/>')
        ds=window['observation'].get('descriptors') or []
        d=ds[i] if i<len(ds) else {}
        bbox=d.get('bbox_xyxy_px')
        if isinstance(bbox,list) and len(bbox)==4 and all(type(v) in (int,float) and math.isfinite(v) for v in bbox):
            x,y,x1,y1=bbox
            parts.append(f'<rect x="{x0+x*scale}" y="{y0+y*scale}" width="{max(0,x1-x)*scale}" height="{max(0,y1-y)*scale}" stroke="#1684a0" stroke-dasharray="4,2" fill="none"/>')
        centroid=d.get('centroid_px')
        if isinstance(centroid,list) and len(centroid)==2 and all(type(v) in (int,float) and math.isfinite(v) for v in centroid):
            u,v=centroid
            parts.append(f'<circle cx="{x0+u*scale}" cy="{y0+v*scale}" r="4" fill="#1684a0"/>')
            parts.append(f'<text x="{x0}" y="260" font-size="11">centroid=({u:.2f}, {v:.2f})</text>')
        if reference:
            rd=reference['observation'].get('descriptors') or []
            c=rd[i].get('centroid_px') if i<len(rd) else None
            if isinstance(c,list) and len(c)==2 and all(type(v) in (int,float) and math.isfinite(v) for v in c):
                x=x0+c[0]*scale;y=y0+c[1]*scale
                parts.append(f'<path d="M{x-5} {y-5} l10 10 m-10 0 l10 -10" stroke="#777"/>')
    parts.append('</svg>');return '\n'.join(parts)


def _review_form_script(annotations):
    # Browser Numbers cannot exactly represent every monotonic-ns integer. Keep
    # timestamps as decimal strings until serialization into the downloaded JSON.
    browser=deepcopy(annotations)
    for trial in browser['trials']:
        if trial['final_contact_timestamp_ns'] is not None:
            trial['final_contact_timestamp_ns']=str(trial['final_contact_timestamp_ns'])
        for window in trial['windows']:window['timestamp_ns']=str(window['timestamp_ns'])
    payload=json.dumps(browser,ensure_ascii=False).replace('<',r'\u003c')
    script=r"""<script>
'use strict';
const template = PAYLOAD;
function answer(value) { return value === 'yes' ? true : value === 'no' ? false : null; }
document.getElementById('download').addEventListener('click', () => {
  const status = document.getElementById('review-status');
  try {
    const result = JSON.parse(JSON.stringify(template));
    result.reviewer = document.getElementById('reviewer').value.trim();
    if (!result.reviewer) throw new Error('请填写审核人。');
    document.querySelectorAll('.attestation').forEach(input => {
      result.attestation[input.dataset.key] = input.checked;
      if (!input.checked) throw new Error('请完成三项审核声明。');
    });
    document.querySelectorAll('section.trial').forEach(section => {
      const trial = result.trials.find(item => item.id === section.dataset.trial);
      trial.reviewed = section.querySelector('.reviewed').checked;
      if (!trial.reviewed) throw new Error('Trial '+trial.id+' 尚未确认审核。未知标签可以保留。');
      const initial = section.querySelector('.initial').value;
      trial.initial_class = initial === 'unknown' ? null : initial;
      section.querySelectorAll('.window-label').forEach(input => {
        trial.windows[Number(input.dataset.index)].target_reached = answer(input.value);
      });
      trial.final_target_reached = answer(section.querySelector('.final-label').value);
      const firstIndex = Number(section.dataset.firstIndex);
      const first = firstIndex >= 0 ? trial.windows[firstIndex].target_reached : null;
      if ((trial.initial_class === 'needs_recovery' && first === true) ||
          (trial.initial_class === 'already_normal' && first === false)) {
        throw new Error('Trial '+trial.id+'：首次调用标签与首次窗口目标标签矛盾，请复核。');
      }
      const last = trial.windows.length ? trial.windows[trial.windows.length-1].target_reached : null;
      if (trial.final_target_reached !== null && last !== trial.final_target_reached) {
        throw new Error('Trial '+trial.id+'：最后接触标签必须与最后记录窗口一致，也可保留未知。');
      }
    });
    let text = JSON.stringify(result, null, 2);
    text = text.replace(/("(?:timestamp_ns|final_contact_timestamp_ns)"\s*:\s*)"([0-9]+)"/g, '$1$2');
    const blob = new Blob([text+'\n'], {type:'application/json'});
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a'); link.href = url; link.download = 'annotations.json';
    document.body.appendChild(link); link.click(); link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
    status.textContent = '已生成下载。原始数据没有改动；把该 JSON 交给 apply 检查后才会生成训练视图。';
  } catch (error) { status.textContent = error.message; }
});
</script>""".replace('PAYLOAD',payload)
    return ['<section><button id=download type=button>下载人工审核 annotations.json</button><p id=review-status role=status></p><p>全程离线，无网络请求；此按钮只下载新文件，不写原记录。</p></section>',script]


def _contact_images_html(evidence):
    """Show original pixels first, with their exact recorded probe association."""
    cards=[]
    for side,title in (('left','左指'),('right','右指')):
        item=(evidence or {}).get(side) or {}
        cards.append('<figure><figcaption><b>'+title+' · 原始触觉图</b></figcaption>')
        if item.get('status')=='available':
            filename=html.escape(item['filename'],quote=True)
            cards.append('<a href="'+filename+'" target="_blank" rel="noopener"><img class=raw-tactile loading=lazy decoding=async src="'+filename+'" alt="'+title+'该接触窗口关联的原始触觉帧"></a>')
            cards.append('<p class=media-note>原尺寸无损 PNG；点击查看原尺寸。未增强、未翻转、未叠加接触框。<br>原图时间戳：'+str(item['frame_timestamp_ns'])+
                '<br>相对接触观测：'+format(item['observation_delta_ms'],'+.2f')+' ms</p>')
        else:
            cards.append('<p class=missing-image>该窗口没有可核验的对应原图，不能用其他阶段图像代替。</p><details><summary>缺失原因</summary><pre>'+
                html.escape(str(item.get('reason','没有对应图像记录')))+'</pre></details>')
        cards.append('</figure>')
    skew=(evidence or {}).get('pair_skew_ms')
    note='左右原图不是同一曝光时刻；描述子来自一段 DD 窗口，原图来自该次轻触关联的关键帧。'
    if skew is not None:note+=' 左右帧时间差：'+format(skew,'.2f')+' ms。'
    return '<div class=tactile-pair>'+''.join(cards)+'</div><p class=media-note>'+note+'</p>'


def prepare_review(session,config,out,*,reference_trial=None):
    out=Path(out).expanduser().resolve()
    if out.exists():raise FileExistsError('review packages are immutable; choose a new output')
    manifest=_source_manifest(session,_definition(config),reference_trial)
    annotations={'schema':ANNOTATION_SCHEMA,'manifest_sha256':_hash(manifest),
                 'target_definition_id':manifest['target_definition']['id'],'reviewer':'',
                 'attestation':{'reviewed_observed_contact':False,'independent_of_experimental_offset':False,'target_definition_understood':False},
                 'trials':[]}
    out.mkdir(parents=True,exist_ok=False);_write(out/'manifest.json',manifest)
    reference=next((t['windows'][0] for t in manifest['trials'] if t['id']==reference_trial and t['windows']),None)
    from .skill_review_images import export_contact_images
    media={'schema':'tactile_review_media_v1','manifest_sha256':_hash(manifest),'trials':[]}
    body=['<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>接触标签审核 · 触觉原图</title>',
          '<h1>接触标签人工审核 · 触觉原图</h1><p>每个窗口优先展示其记录关联的左右胶面原图；缺失时明确标注。图像保持原始像素和方向，点击可查看原尺寸。<b>图片本身不自动判定目标是否达标。</b></p>',
          '<p>折叠区域里的质心与包围盒只是描述子示意，<b>不是原始 DD 图像</b>。灰色叉是人工指定的候选参考，不自动算正确。</p>',
          '<p>不要按计划偏移、TCP、旧标签、模型结论或提起结果填写。无法判断请保留 null。填写 annotations.json；本页不执行机器人操作。</p>',
          '<style>body{font-family:sans-serif;max-width:1100px;margin:24px auto;padding:16px;background:#f5f7fa;color:#17212b}section{background:white;padding:20px;margin:18px 0;border:1px solid #dce3eb}label{display:block;margin:10px 0}select,input{font:inherit;padding:5px}pre{white-space:pre-wrap;font-size:12px}img{max-width:100%}button{padding:12px 20px;font-size:16px}#review-status,.missing-image{color:#923331}.tactile-pair{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:20px}.tactile-pair figure{margin:10px 0}.raw-tactile{display:block;width:100%;height:auto;margin-top:8px}.media-note{font-size:13px;color:#52606b;line-height:1.6;overflow-wrap:anywhere}.descriptor-geometry{margin:18px 0;padding:10px;border:1px solid #dce3eb}summary{cursor:pointer}@media(max-width:650px){.tactile-pair{grid-template-columns:1fr}}</style>',
          '<pre>'+html.escape(json.dumps(manifest['target_definition'],ensure_ascii=False,indent=2))+'</pre>',
          '<section><label>审核人 <input id=reviewer autocomplete=off></label>'+
          ''.join('<label><input type=checkbox class=attestation data-key="'+key+'"> '+label+'</label>' for key,label in (
            ('reviewed_observed_contact','我审核了实际记录的接触窗口，不能判断的保留未知'),
            ('independent_of_experimental_offset','我没有按实验偏移或提起结果代填接触标签'),
            ('target_definition_understood','我理解上述目标接触定义；参考图仅是候选')))+'</section>']
    for trial in manifest['trials']:
        images=export_contact_images(trial,out)
        media['trials'].append({'id':trial['id'],'windows':images})
        image_by_time={item['timestamp_ns']:item for item in images}
        item={'id':trial['id'],'reviewed':False,'initial_class':None,
              'final_contact_timestamp_ns':trial['final_contact_timestamp_ns'],'final_target_reached':None,
              'windows':[{'timestamp_ns':w['timestamp_ns'],'target_reached':None} for w in trial['windows']]}
        annotations['trials'].append(item)
        first_index=next((i for i,w in enumerate(trial['windows']) if w['timestamp_ns']==trial['initial_timestamp_ns']),-1)
        body.append('<section class=trial data-trial="'+html.escape(trial['id'],quote=True)+'" data-first-index="'+str(first_index)+'"><h2>Trial '+html.escape(trial['id'])+'</h2>')
        body.append('<label>首次接触：<select class=initial><option value=unknown>未知</option><option value=already_normal>接触已符合要求，无需调整位置</option><option value=needs_recovery>需要沿 a 轴恢复位置</option><option value=not_applicable>当前不适用</option></select></label>')
        body.append('<p>完整录制可用于训练：'+str(trial['recording_reviewable'])+'</p>')
        for window_index,window in enumerate(trial['windows']):
            filename=f"trial_{trial['id']}_{window['timestamp_ns']}.svg"
            (out/filename).write_text(_diagram(window,reference))
            obs=window['observation'];visible={k:obs.get(k) for k in ('timestamp_ns','status','bilateral','force_stable','stable_duration_s')}
            visible['descriptors']=[{k:d.get(k) for k in ('valid','centroid_px','bbox_xyxy_px','area_px','touches_edge','edge_sides')} for d in obs.get('descriptors',[])]
            body.append('<p>窗口 '+str(window['timestamp_ns'])+'，'+html.escape(', '.join(str(p) for p in window['phases']))+'</p>')
            body.append(_contact_images_html(image_by_time.get(window['timestamp_ns'])))
            body.append('<details class=descriptor-geometry><summary>查看描述子示意及数值（不是原图）</summary><img loading=lazy src="'+html.escape(filename)+'"><pre>'+html.escape(json.dumps(visible,ensure_ascii=False,indent=2))+'</pre></details><label>这个窗口是否已符合交给力控模块的接触要求？<select class=window-label data-index="'+str(window_index)+'"><option value=unknown>未知</option><option value=yes>是</option><option value=no>否</option></select></label>')
        body.append('<label>最后记录的接触窗口达到目标了吗？<select class=final-label><option value=unknown>未知（可保留）</option><option value=yes>是</option><option value=no>否</option></select></label><p>这不是提起/整任务是否成功；没有独立观察依据请选未知。</p><label><input type=checkbox class=reviewed> 我已审核本条；无法判断的字段保持未知</label></section>')
    body.extend(_review_form_script(annotations))
    (out/'index.html').write_text('\n'.join(body))
    _write(out/'annotations.json',annotations)
    _write(out/'image_evidence.json',media)
    available=sum(w[side].get('status')=='available' for t in media['trials'] for w in t['windows'] for side in ('left','right'))
    return {'review_directory':str(out),'index':str(out/'index.html'),'annotations':str(out/'annotations.json'),
            'trials':len(manifest['trials']),'all_labels_unknown':True,'physical_trials_added':0,
            'original_images_available':available,'image_evidence':str(out/'image_evidence.json')}


def _checked_annotations(manifest,annotations):
    if annotations.get('schema')!=ANNOTATION_SCHEMA or annotations.get('manifest_sha256')!=_hash(manifest):
        raise ValueError('annotations must bind to the exact immutable review manifest')
    if annotations.get('target_definition_id')!=manifest['target_definition']['id']:
        raise ValueError('annotation target definition changed')
    if not isinstance(annotations.get('reviewer'),str) or not annotations['reviewer'].strip():raise ValueError('human reviewer identity required')
    att=annotations.get('attestation') or {}
    if any(att.get(k) is not True for k in ('reviewed_observed_contact','independent_of_experimental_offset','target_definition_understood')):
        raise ValueError('explicit independent human contact review attestation required')
    items=annotations.get('trials')
    if not isinstance(items,list) or len(items)!=len(manifest['trials']):raise ValueError('every source trial requires an explicit reviewed or unknown annotation')
    mapping={}
    for item in items:
        identifier=item.get('id')
        if identifier in mapping:raise ValueError('duplicate physical trial annotation')
        mapping[identifier]=item
    if set(mapping)!={t['id'] for t in manifest['trials']}:raise ValueError('annotation trial identities changed')
    for trial in manifest['trials']:
        item=mapping[trial['id']]
        if item.get('reviewed') is not True:raise ValueError('each trial must be explicitly reviewed; unknown labels may remain null')
        initial=item.get('initial_class');final=item.get('final_target_reached')
        if initial not in (None,'already_normal','needs_recovery','not_applicable'):raise ValueError('invalid initial contact class')
        if final is not None and type(final) is not bool:raise ValueError('final contact label must be true/false/null')
        if item.get('final_contact_timestamp_ns')!=trial['final_contact_timestamp_ns']:raise ValueError('final label must bind to the last recorded contact window')
        windows=item.get('windows')
        if not isinstance(windows,list) or len(windows)!=len(trial['windows']):raise ValueError('every recorded window must be explicitly present')
        values={}
        for label in windows:
            stamp=label.get('timestamp_ns');truth=label.get('target_reached')
            if type(stamp) is not int or stamp in values:raise ValueError('duplicate or invalid annotation timestamp')
            if truth is not None and type(truth) is not bool:raise ValueError('window label must be true/false/null')
            values[stamp]=truth
        if set(values)!={w['timestamp_ns'] for w in trial['windows']}:raise ValueError('annotation timestamps do not match observed contact windows')
        first=values.get(trial['initial_timestamp_ns'])
        if initial=='needs_recovery' and first is True or initial=='already_normal' and first is False:
            raise ValueError('contradictory initial invocation/contact-target labels')
        if initial is not None and trial['initial_timestamp_ns'] is None:raise ValueError('no first-touch window to support initial label')
        if final is not None and (trial['final_contact_timestamp_ns'] is None or values.get(trial['final_contact_timestamp_ns']) is not final):
            raise ValueError('final target label must agree with the observed final contact window, never task success')
        if not trial['recording_reviewable'] and (initial is not None or final is not None or any(v is not None for v in values.values())):
            raise ValueError('incomplete recordings must keep contact labels unknown')
    return mapping


def _view_trial(original,trial,item,manifest,annotations,applied_ns):
    shared={'source':SOURCE,'target_definition_id':manifest['target_definition']['id'],
            'independent_of_experimental_offset':True,'reviewer':annotations['reviewer'],'reviewed_unix_ns':applied_ns}
    row=deepcopy(original)
    row['skill_ground_truth']={**shared,'initial_class':item['initial_class'],'final_target_reached':item['final_target_reached'],
                               'final_contact_timestamp_ns':item['final_contact_timestamp_ns']}
    row['skill_window_labels']=[{**shared,**label} for label in item['windows']]
    row['skill_replay_source_trial']=trial['source_trial']
    row['skill_label_review']={'schema':TRIAL_SCHEMA,'source_session':manifest['source_session'],
        'source_trial':trial['source_trial'],'source_metadata_sha256':trial['source_metadata_sha256'],
        'recording_status_sha256':trial['recording_status_sha256'],'raw':trial['raw'],
        'previous_skill_ground_truth':deepcopy(original.get('skill_ground_truth')),
        'previous_skill_window_labels':deepcopy(original.get('skill_window_labels')),
        'applied_unix_ns':applied_ns,'physical_trials_added':0}
    return row


def apply_review(review,annotations,out):
    review=Path(review).expanduser().resolve();out=Path(out).expanduser().resolve()
    if out.exists():raise FileExistsError('label views are immutable; choose a new output')
    manifest=_read(review/'manifest.json');answers=_read(annotations)
    if manifest.get('schema')!=REVIEW_SCHEMA:raise ValueError('unsupported review manifest')
    current=_source_manifest(manifest['source_session'],manifest['target_definition'],manifest.get('reference_trial'))
    if current!=manifest:raise ValueError('source evidence changed since review preparation')
    items=_checked_annotations(manifest,answers)
    source=Path(manifest['source_session']);meta=_read(source/'session_meta.json');applied_ns=time.time_ns()
    meta['skill_replay_source_session']=str(source)
    meta.setdefault('config',{}).setdefault('recovery_skill',{})['target_definition']=deepcopy(manifest['target_definition'])
    meta['skill_label_review']={'schema':VIEW_SCHEMA,'source_session':str(source),
        'source_session_metadata_sha256':manifest['source_session_metadata_sha256'],'source_role':manifest['source_role'],
        'manifest_sha256':_hash(manifest),'annotations_sha256':_hash(answers),'applied_unix_ns':applied_ns,'physical_trials_added':0}
    out.parent.mkdir(parents=True,exist_ok=True);stage=Path(tempfile.mkdtemp(prefix='.'+out.name+'.',dir=out.parent))
    try:
        _write(stage/'review_manifest.json',manifest);_write(stage/'review_annotations.json',answers)
        _write(stage/'session_meta.json',meta)
        for trial in manifest['trials']:
            origin=Path(trial['source_trial']);folder=stage/'trials'/trial['id'];folder.mkdir(parents=True)
            row=_view_trial(_read(origin/'meta.json'),trial,items[trial['id']],manifest,answers,applied_ns)
            _write(folder/'meta.json',row)
            for child in origin.iterdir():
                if child.name=='meta.json':continue
                (folder/child.name).symlink_to(child.resolve(),target_is_directory=child.is_dir())
        if out.exists():raise FileExistsError('view destination appeared during apply')
        stage.rename(out)
    except BaseException:
        shutil.rmtree(stage,ignore_errors=True);raise
    return {'view_session':str(out),'source_session':str(source),'physical_trials_added':0,
            'note':'Use this view instead of its original session; loading both or multiple views is blocked.'}


def verify_view_row(root,meta,row,path):
    """Loader hook: source hashes, roles, annotations and physical identity."""
    root=Path(root);proof=meta.get('skill_label_review') or {}
    if proof.get('schema')!=VIEW_SCHEMA:raise ValueError('unsupported contact review view')
    manifest=_read(root/'review_manifest.json');annotations=_read(root/'review_annotations.json')
    if _hash(manifest)!=proof.get('manifest_sha256') or _hash(annotations)!=proof.get('annotations_sha256'):
        raise ValueError('contact review manifest/annotations changed')
    source=Path(proof['source_session']).resolve();source_meta=_read(source/'session_meta.json')
    if source_meta.get('skill_label_review') or source_meta.get('skill_replay_source_session'):
        raise ValueError('review source must be original physical evidence')
    if _file_hash(source/'session_meta.json')!=proof.get('source_session_metadata_sha256'):
        raise ValueError('review source session metadata changed')
    if (manifest.get('schema')!=REVIEW_SCHEMA or manifest.get('source_session')!=str(source) or
            manifest.get('source_session_metadata_sha256')!=proof.get('source_session_metadata_sha256') or
            manifest.get('source_role')!=_role(source_meta)):
        raise ValueError('review manifest does not match original physical session')
    if _role(source_meta) not in ('exploration','development') or _role(meta)!=_role(source_meta) or proof.get('source_role')!=_role(source_meta):
        raise ValueError('review cannot change source roles or relabel final evidence')
    if meta.get('simulated')!=source_meta.get('simulated') or meta.get('cond')!=source_meta.get('cond'):
        raise ValueError('review cannot change simulation or experiment condition')
    expected_meta=deepcopy(source_meta)
    expected_meta['skill_replay_source_session']=str(source)
    expected_meta.setdefault('config',{}).setdefault('recovery_skill',{})['target_definition']=deepcopy(manifest['target_definition'])
    expected_meta['skill_label_review']=proof
    if meta!=expected_meta:raise ValueError('review changed non-label session metadata')
    items=_checked_annotations(manifest,annotations);identifier=Path(path).parent.name
    trial=next((t for t in manifest['trials'] if t['id']==identifier),None)
    if trial is None:raise ValueError('trial not included in reviewed physical evidence')
    origin=(source/'trials'/identifier).resolve()
    if str(origin)!=trial['source_trial'] or Path(row.get('skill_replay_source_trial','')).resolve()!=origin:
        raise ValueError('review trial physical identity mismatch')
    if _file_hash(origin/'meta.json')!=trial['source_metadata_sha256']:
        raise ValueError('review source trial metadata changed')
    status=origin/'recording_status.json'
    if (_file_hash(status) if status.exists() else None)!=trial['recording_status_sha256'] or _raw_proof(origin)!=trial['raw']:
        raise ValueError('review recording or raw TCP evidence changed')
    if trial!=_trial_manifest(origin/'meta.json'):
        raise ValueError('review window manifest differs from original physical evidence')
    if row!=_view_trial(_read(origin/'meta.json'),trial,items[identifier],manifest,annotations,proof['applied_unix_ns']):
        raise ValueError('review trial differs from original evidence plus approved contact annotations')
    view_status=Path(path).parent/'recording_status.json'
    if status.exists() and (not view_status.is_symlink() or view_status.resolve()!=status.resolve()):
        raise ValueError('review recording status must reuse original evidence')
    if not status.exists() and view_status.exists():raise ValueError('review invented recording status')
    raw=Path(path).parent/'streams.h5'
    if trial['raw'] is not None and (not raw.is_symlink() or raw.resolve()!=origin/'streams.h5'):
        raise ValueError('review raw data must reuse the original recording')
    return str(source),str(origin)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    q=sub.add_parser('prepare');q.add_argument('--session',required=True);q.add_argument('--config',required=True);q.add_argument('--out',required=True);q.add_argument('--reference-trial')
    q=sub.add_parser('apply');q.add_argument('--review',required=True);q.add_argument('--annotations',required=True);q.add_argument('--out',required=True)
    a=p.parse_args(argv)
    result=prepare_review(a.session,a.config,a.out,reference_trial=a.reference_trial) if a.command=='prepare' else apply_review(a.review,a.annotations,a.out)
    print(json.dumps(result,ensure_ascii=False,indent=2));return 0
