"""Hand displacement components [current TCP a, base_link +Z], millimetres."""
import numpy as np
from bottle_scan.geometry import validate_transform

KIND = 'translation_tcp_a_base_h'
COORDS = ('a_offset_mm', 'h_offset_mm')


def basis(pose, a_tcp):
    pose = validate_transform(pose)
    a = np.asarray(a_tcp, float)
    if a.shape != (3,) or not np.isclose(np.linalg.norm(a), 1):
        raise ValueError('Invalid TCP a axis')
    M = np.column_stack((pose[:3, :3] @ a, [0., 0., 1.]))
    if np.linalg.cond(M) > 10:
        raise ValueError('当前 a 和 h 近共线，此姿态无法可靠区分两轴')
    return M


def measured(origin, end, frame, axis):
    origin, end = validate_transform(origin), validate_transform(end)
    M = basis(frame, axis)
    delta = end[:3, 3] - origin[:3, 3]
    u = np.linalg.lstsq(M, delta, rcond=None)[0]
    return u, float(np.linalg.norm(delta - M @ u))


def target(origin, frame, u, axis):
    origin, frame = validate_transform(origin), validate_transform(frame)
    out = frame.copy()
    out[:3, 3] = origin[:3, 3] + basis(frame, axis) @ np.asarray(u, float)
    return out


def position_limits(spec):
    return np.asarray([spec['perturbation_variables'][key]['envelope'] for key in COORDS],float)


def position_allowed(pos, spec, *, base_height=None, tolerance=0.):
    pos=np.asarray(pos,float)
    bounds=position_limits(spec)
    if pos.shape!=(2,) or not np.isfinite(pos).all() or not np.all((bounds[:,0]-tolerance<=pos)&(pos<=bounds[:,1]+tolerance)):
        return False
    floor,ceiling=bounds[1]
    if base_height is None:
        if 'placement_geometry' not in spec:
            return True  # Original 3 mm plans have component-only bounds.
        base_height=pos[1]+spec['placement_geometry']['tcp_a_base_z']*pos[0]
    return floor-tolerance-1e-9<=base_height<=ceiling+tolerance+1e-9


def validate_cell(cell, spec, op, args):
    settings = cell.get('settings', {})
    current = bool(spec.get('current_placement'))
    if current and (set(settings)-{'mass_class','closure_floor'} or cell.get('source_node_id')):
        raise ValueError('从当前接触采集，不能指定起始 a/h 坐标或复用历史起点')
    if set(settings) - set(COORDS) - {'closure_floor','approach_pose','l_offset_mm'}:
        raise ValueError('Undeclared a/h placement setting')
    if settings.get('l_offset_mm',0) or settings.get('closure_floor',.33)!=.33 or settings.get('approach_pose','pose_0')!='pose_0':
        raise ValueError('This collection profile supports a and h only')
    if any(set(b) != {'mass_class'} or b['mass_class'] not in ('light','heavy') for b in cell.get('blocks',[])):
        raise ValueError('Unknown bottle block')
    pos = np.asarray([settings.get(k,0.) for k in COORDS],float)
    step_limits=np.asarray(spec['task']['action_space']['max_step'],float)
    if not current and not position_allowed(pos,spec):
        raise ValueError('起点超出本组采集范围')
    actions = cell.get('actions',[])
    if len(actions)>1 or (op=='collect_seed' and actions):
        raise ValueError('一次探索只比较首触和一次移位后的重触')
    for action in actions:
        u = np.asarray(action['u'],float)
        if action['kind']!=KIND or u.shape!=(2,) or not np.isfinite(u).all() or np.count_nonzero(u)!=1 or np.any(np.abs(u)>step_limits):
            raise ValueError(f'每步只动一个轴；a/h 单步上限分别为 {step_limits.tolist()} mm')
        if current:
            bounds=position_limits(spec)
            if np.any(u<bounds[:,0]) or np.any(u>bounds[:,1]):
                raise ValueError('所选动作超出本次起点允许的 a/h 位移')
        if args.get('axis'):
            i = {'a':0,'h':1}[args['axis']]
            if np.sign(u[i]) != {'positive':1,'negative':-1}[args['side']]:
                raise ValueError('动作与导演所选轴或方向不一致')
        pos += u
        if not current and not position_allowed(pos,spec):
            raise ValueError('移位后超出本组采集范围')
    if op=='collect_transitions' and not actions:
        raise ValueError('需要真实的前后移动')
