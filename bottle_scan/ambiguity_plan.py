"""Small, deterministic contact-identifiability experiment. Pure geometry only."""
from __future__ import annotations

import math
import numpy as np

from .geometry import pose_offset, validate_transform, pose_error
from .session import fingerprint


def make_design(step_mm=4., tilt_deg=3., repeats=3, seed=20261105, bottle='bottle'):
    """Three independently randomized blocks, each starting at zero.

    Tilt is about the TCP, NOT an assumed sensor/grasp center. Pairwise a/l
    combinations expose coupling without a large full factorial sweep.
    """
    if not math.isfinite(step_mm) or not 0 < step_mm <= 5:
        raise ValueError('step-mm 必须在 (0,5] mm')
    if not math.isfinite(tilt_deg) or not 0 <= tilt_deg <= 3:
        raise ValueError('tilt-deg 必须在 [0,3] 度；0 表示仅 a/l')
    if isinstance(repeats, bool) or not isinstance(repeats, int) or not 3 <= repeats <= 10:
        raise ValueError('repeats 必须为 3–10；至少三次独立重接触才能估计重复波动')
    if not bottle or '/' in bottle or '\\' in bottle:
        raise ValueError('需要一个非空物体名称')
    points = [(0., 0., 0.), (-step_mm, 0., 0.), (step_mm, 0., 0.),
              (0., -step_mm, 0.), (0., step_mm, 0.)]
    if tilt_deg:
        points += [(0., 0., -tilt_deg), (0., 0., tilt_deg)]
    points += [(a, l, 0.) for a in (-step_mm, step_mm) for l in (-step_mm, step_mm)]
    rng = np.random.default_rng(seed)
    rows = []
    for repeat in range(repeats):
        order = [0] + list(rng.permutation(np.arange(1, len(points))))
        for point in order:
            a, l, theta = points[point]
            rows.append({'id': f'{len(rows)+1:05d}', 'index': len(rows)+1,
                         'cond': 'A', 'bottle': bottle, 'repeat': repeat,
                         'kind': 'ambiguity_probe', 'state_id': f's{point:02d}',
                         'delta_mm': a, 'perturbation': {'a_mm': a, 'l_mm': l, 'tilt_deg': theta}})
    design = {'schema': 'ambiguity_design_v1', 'step_mm': step_mm, 'tilt_deg': tilt_deg,
              'repeats': repeats, 'seed': seed, 'bottle': bottle, 'tilt_axis': 'c',
              'rotation_pivot': 'tcp', 'coordinates': 'original_T0_local_axes',
              'max_translation_mm': 8., 'max_tilt_deg': 3.,
              'probe_only': True, 'states': len(points), 'plan': rows,
              'label_semantics': 'geometric perturbation, not verified optimal recovery action'}
    design['content_sha256'] = fingerprint(design)
    return design


def validate_design(design):
    if design.get('schema') == 'skill_extension_probe_design_v1':
        from .skill_extension import validate_probe_design
        return validate_probe_design(design)
    rebuilt = make_design(design['step_mm'], design['tilt_deg'], design['repeats'],
                          design['seed'], design['bottle'])
    if rebuilt != design:
        raise ValueError('实验计划已被修改或哈希不符；请重新生成计划')
    return design


class ProbeBounds:
    """Point/path endpoint bounds, not a tool envelope or collision certificate."""
    def __init__(self, cfg, T0, design):
        self.cfg, self.design = cfg, validate_design(design)
        self.T0 = validate_transform(T0).copy()
        self.a = self.T0[:3, :3] @ np.asarray(cfg['axes_tcp']['a'])
        self.l = self.T0[:3, :3] @ np.asarray(cfg['axes_tcp']['l'])
        self.c = self.T0[:3, :3] @ np.asarray(cfg['axes_tcp']['c'])

    def target(self, item):
        p = item['perturbation']
        return pose_offset(self.T0, [p['a_mm'], 0., p['l_mm']], 'c',
                           p['tilt_deg'], self.cfg['axes_tcp'])

    def pre(self, target):
        # Retraction uses the CURRENT tilted finger direction. Rotation happens
        # between these open, retracted poses, never during contact.
        p = validate_transform(target).copy()
        p[:3, 3] -= p[:3, :3] @ np.asarray(self.cfg['axes_tcp']['a']) * self.cfg['motion']['pre_mm']
        return p

    def check(self, pose, role='grasp', anchor=None):
        try:
            pose = validate_transform(pose)
        except ValueError as exc:
            return str(exc)
        if pose[2, 3] < self.T0[2, 3] - self.cfg['motion']['z_below_T0_mm'] - 1e-7:
            return 'TCP z 低于 T0 高度下限'
        relative_R = self.T0[:3, :3].T @ pose[:3, :3]
        c = np.asarray(self.cfg['axes_tcp']['c'])
        if not np.allclose(relative_R @ c, c, atol=1e-6):
            return '只允许绕原 T0 的局部 c 轴旋转'
        angle = pose_error(self.T0, pose)[1]
        if angle > self.design['max_tilt_deg'] + 1e-6:
            return '倾斜超过3度'
        if role == 'grasp':
            delta = pose[:3, 3] - self.T0[:3, 3]
            if abs(delta @ self.c) > 1e-6:
                return '只允许 a/l 平移'
            if np.linalg.norm(delta) > self.design['max_translation_mm'] + 1e-6:
                return '接触目标平移超过8 mm'
            return None
        if role != 'retreat' or anchor is None:
            return '未知运动角色或缺少接触锚点'
        error = self.check(anchor)
        if error:
            return error
        if not np.allclose(pose, self.pre(anchor), atol=1e-6):
            return '退让必须沿该接触位姿的 -a，保持朝向'
        return None


def plan_report(design, cfg, T0=None):
    """Omitted T0 produces an explicitly local preview, never fake base poses."""
    validate_design(design)
    rows = []
    bounds = ProbeBounds(cfg, T0, design) if T0 is not None else None
    for item in design['plan']:
        row = dict(item)
        if bounds:
            target = bounds.target(item)
            pre = bounds.pre(target)
            row.update(target_tcp_mm=target.tolist(), pre_tcp_mm=pre.tolist(),
                       delta_base_xyz_mm=(target[:3, 3] - bounds.T0[:3, 3]).tolist(),
                       skip_reason=bounds.check(target) or bounds.check(pre, 'retreat', target))
        rows.append(row)
    return {'schema': 'ambiguity_plan_preview_v1', 'design': design,
            'has_measured_T0': T0 is not None, 'trials': rows,
            'limits': {'translation_mm': 8., 'tilt_deg': 3.,
                       'retreat_mm': cfg['motion']['pre_mm'], 'lift': False,
                       'angular_speed_deg_s': 1., 'collision_check': False},
            'note': '倾斜绕 TCP 原点；胶面中心会随之平移。实机开始前按实测 T0 打印目标并检查高度。'}
