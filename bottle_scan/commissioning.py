"""Operator-reviewed commissioning and reusable hardware evidence, offline only."""
from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path
from .session import Session, atomic_json, fingerprint
from .force_control import force_value, force_unit


def hardware_binding(cfg):
    selected = {key:copy.deepcopy(cfg.get(key)) for key in
        ('hardware','tool','gripper','force','tactile','cameras','motion','recording','record','park_guard','zero_guard','close','probe','hold','lower')}
    if isinstance(selected.get('force'),dict):
        selected['force'].get('validation',{}).pop('report_path',None)
    if isinstance(selected.get('recording'),dict):
        selected['recording'].pop('stress',None)  # offline synthetic load does not change hardware behavior
    selected['gripper_geometry'] = cfg.get('geometry',{}).get('gripper')
    selected['axes'] = cfg.get('axes')
    root = Path(__file__).resolve().parents[1]
    sources = {}
    for name in ('bridge_ros_public.py','bottle_scan/hardware_arm.py','bottle_scan/hardware_gripper.py',
                 'bottle_scan/hardware_sensors.py','bottle_scan/hardware_observer.py','bottle_scan/zero_guard.py',
                 'bottle_scan/runtime.py','bottle_scan/existing_force.py','bottle_scan/command_ownership.py',
                 'bottle_scan/force_control.py','bottle_scan/geometry.py','bottle_scan/design.py',
                 'bottle_scan/park_guard.py','bottle_scan/recording.py','bottle_scan/config.py',
                 'bottle_scan/commissioning.py'):
        sources[name] = hashlib.sha256((root/name).read_bytes()).hexdigest()
    return {'config_sha256':fingerprint(selected),'source_sha256':fingerprint(sources)}


def require_prior_session(cfg, path, *, simulated=False):
    if not path:
        raise ValueError('Provide --verified-session with a prior full three-point PASS session')
    prior = Session(path)
    approval = prior.meta.get('commissioning_acceptance',{})
    if bool(prior.meta['simulated']) != bool(simulated):
        raise ValueError('Simulated commissioning cannot authorize a real session')
    if not prior.progress.get('commissioning_passed') or approval.get('level') != 'full' or approval.get('operator_confirmation') != 'PASS' or approval.get('passed') is not True:
        raise ValueError('Prior session requires full three-point operator PASS evidence')
    if approval.get('binding') != hardware_binding(cfg) or approval.get('binding') != hardware_binding(prior.cfg):
        raise ValueError('Hardware/bridge configuration or code changed: full three-point commissioning required')
    return {'path':str(prior.path.resolve()), 'binding':approval['binding']}


def require_current_commissioning(session, task):
    """A boolean left by a legacy run is not reusable measured evidence."""
    keys = ['commissioning_acceptance']
    if task == 'visual':
        keys.append('visual_commissioning_acceptance')
    for key in keys:
        report = session.meta.get(key,{})
        if (report.get('passed') is True and report.get('operator_confirmation') == 'PASS'
                and report.get('simulated') is False and report.get('binding') == hardware_binding(session.cfg)):
            if report.get('level') in ('lite','visual') and not report.get('prior'):
                continue
            return report
    raise ValueError('This session lacks a matching current operator-PASS commissioning report; repeat commissioning')


def build_acceptance(session, records, *, level='full', prior=None):
    expected = {'full':3,'lite':1,'visual':8}[level]
    errors = []
    if len(records) != expected:
        errors.append(f'Expected {expected} complete records, found {len(records)}')
    rows = []
    for record in records:
        auto = record.get('auto',{})
        windows = record.get('descriptor_windows',{})
        kind = record.get('kind','')
        row = {'trial_id':record['trial_id'],'kind':kind,'status':record.get('status'),
            'max_force':auto.get('max_force'), 'F_max':force_value(session.cfg,'max'),
            'contact_peak':auto.get('contact_peak'),'probe_not_settled':auto.get('probe_not_settled'),
            'probe_snr':auto.get('probe_snr'), 'probe_descriptors':windows.get('PROBE'),
            'above_cap_no_contact':None if 'above_cap' not in kind else not auto.get('bilateral_contact',True),
            'approach_translation_error_mm':auto.get('approach_translation_error_mm'),
            'approach_rotation_error_deg':auto.get('approach_rotation_error_deg'),
            'control_loop_periods':auto.get('control_loop_periods'),
            'checks':record.get('position_checks',[])}
        rows.append(row)
        peak = auto.get('max_force')
        if peak is not None and any(float(value) > row['F_max'] for value in peak):
            errors.append(record['trial_id']+': measured F_max exceeded')
        if record.get('status') != 'completed' or auto.get('exceptions') or auto.get('dropped') or auto.get('slip_suspected') or record.get('operator',{}).get('label'):
            errors.append(record['trial_id']+': incomplete or flagged trial')
        if level != 'visual':
            if 'above_cap' in kind:
                if auto.get('bilateral_contact') or not (auto.get('no_bilateral_contact') or auto.get('safety_closure_limit')):
                    errors.append(record['trial_id']+': above-cap point did not take no-contact branch')
            elif not auto.get('bilateral_contact') or not auto.get('force_reached') or auto.get('probe_not_settled',True):
                errors.append(record['trial_id']+': contact/force/PROBE not stable')
        if any(not check.get('passed') for check in row['checks']):
            # Failed first attempts are evidence of a reset, not an accepted CHECK.
            if not row['checks'] or not row['checks'][-1].get('passed'):
                errors.append(record['trial_id']+': final bottle CHECK failed')
    return {'schema':'bottle_commissioning_v2','level':level,'simulated':session.meta['simulated'],
        'binding':hardware_binding(session.cfg),'prior':prior,'force_unit':force_unit(session.cfg),
        'rows':rows,'failures':errors,'measurements_passed':not errors,'passed':False,'operator_confirmation':None}


def acceptance_page(report):
    """Compact operator page; full per-frame evidence remains in JSON/HDF5."""
    def value(data):
        if data is None: return '未提供'
        if isinstance(data, bool): return '是' if data else '否'
        if isinstance(data, (float,int)): return f'{data:.4g}'
        if isinstance(data, (list,tuple)): return '/'.join(value(x) for x in data)
        return str(data).replace('|','/')
    def snr(row):
        data=row.get('probe_snr') or {}
        return value([data.get('left'),data.get('right')])
    def centroids(row):
        result=[]
        for side in ('left','right'):
            median=(row.get('probe_descriptors') or {}).get(side,{}).get('median',{})
            result.append(side+': '+value([median.get('centroid_u_px'),median.get('centroid_v_px')]))
        return '; '.join(result)
    def loops(row):
        output=[]
        for name,stats in (row.get('control_loop_periods') or {}).items():
            nums=[stats.get(k) for k in ('mean_s','p95_s','max_s')]
            output.append(name+': '+value([None if x is None else x*1000 for x in nums]))
        return '; '.join(output) or '未提供'
    def checks(row):
        data=row.get('checks') or []
        if not data: return '未提供'
        latest=data[-1]
        distances=[latest.get('cameras',{}).get(camera,{}).get('offset_px') for camera in ('left_wrist','right_wrist')]
        return ('PASS' if latest.get('passed') else 'FAIL')+'; '+value(distances)+' px'
    columns=report['rows']
    lines=['# 现场验收摘要', '', f"级别：{report['level']}；力单位：{report['force_unit']}；"+
           ('模拟数据，不能放行实机。' if report['simulated'] else '实机数据，等待操作员核对。'), '',
           '| 项目 | '+' | '.join(row['trial_id'] for row in columns)+' |',
           '| --- | '+' | '.join('---' for _ in columns)+' |']
    metrics=[('状态',lambda r:r['status']),('最大力（左/右）',lambda r:value(r['max_force'])),
             ('每指 F_max',lambda r:value(r['F_max'])),('接触后峰值（左/右）',lambda r:value(r['contact_peak'])),
             ('PROBE 未稳定',lambda r:value(r['probe_not_settled'])),('轻触 SNR（左/右）',snr),
             ('PROBE 质心 u/v（px）',centroids),('越瓶盖点无接触分支',lambda r:value(r['above_cap_no_contact'])),
             ('到位误差 mm / deg',lambda r:value([r['approach_translation_error_mm'],r['approach_rotation_error_deg']])),
             ('循环 mean/p95/max（ms）',loops),('最终 CHECK（两腕位移）',checks)]
    for name,render in metrics:
        lines.append('| '+name+' | '+' | '.join(render(row) for row in columns)+' |')
    lines+=['', '数据检查：'+('通过' if report['measurements_passed'] else '未通过'),
            '失败项：'+('; '.join(report['failures']) or '无')]
    return '\n'.join(lines)+'\n'


def accept_commissioning(session, records, *, level='full', prior=None, confirm=input):
    report = build_acceptance(session, records, level=level, prior=prior)
    filename = session.path / ('commissioning_'+level+'_summary.json')
    atomic_json(filename,report)
    page=acceptance_page(report)
    filename.with_suffix('.md').write_text(page)
    print(page)
    if report['measurements_passed']:
        answer = 'PASS' if session.meta['simulated'] else confirm('请核对验收摘要与现场结果；输入 PASS 才登记验证通过：').strip()
        report['operator_confirmation'] = answer if answer == 'PASS' else None
        report['passed'] = answer == 'PASS'
    atomic_json(filename,report)
    session.meta['commissioning_acceptance' if level != 'visual' else 'visual_commissioning_acceptance'] = report
    session.progress['commissioning_passed' if level != 'visual' else 'visual_commissioning_passed'] = report['passed']
    session.save_meta(); session.save_progress()
    return report
