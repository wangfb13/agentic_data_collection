"""Hand displacement components [current TCP a, base_link +Z], millimetres."""
from copy import deepcopy
from pathlib import Path
import json
import numpy as np
from bottle_scan.geometry import validate_transform

KIND = 'translation_tcp_a_base_h'
COORDS = ('a_offset_mm', 'h_offset_mm')


RH_KIND='translation_horizontal_r_base_h'
RH_COORDS=('r_offset_mm','h_offset_mm')


def horizontal(spec):
    return bool((spec or {}).get('geometry',{}).get('horizontal_basis'))


def coord_names(spec):
    return RH_COORDS if horizontal(spec) else COORDS


def action_kind(spec):
    return RH_KIND if horizontal(spec) else KIND


def basis(pose, a_tcp, spec=None):
    pose = validate_transform(pose)
    a = np.asarray(a_tcp, float)
    if a.shape != (3,) or not np.isclose(np.linalg.norm(a), 1):
        raise ValueError('Invalid TCP a axis')
    direction=pose[:3,:3]@a
    if horizontal(spec):
        direction[2]=0
        norm=np.linalg.norm(direction)
        if norm<=spec['geometry']['minimum_horizontal_projection']:
            raise ValueError('TCP a has no reliable horizontal direction at this orientation')
        direction/=norm
    M=np.column_stack((direction,[0.,0.,1.]))
    if np.linalg.cond(M) > 10:
        raise ValueError('当前 a 和 h 近共线，此姿态无法可靠区分两轴')
    return M


def measured(origin, end, frame, axis, spec=None):
    origin, end = validate_transform(origin), validate_transform(end)
    M = basis(frame, axis, spec)
    delta = end[:3, 3] - origin[:3, 3]
    u = np.linalg.lstsq(M, delta, rcond=None)[0]
    return u, float(np.linalg.norm(delta - M @ u))


def target(origin, frame, u, axis, spec=None):
    origin, frame = validate_transform(origin), validate_transform(frame)
    out = frame.copy()
    out[:3, 3] = origin[:3, 3] + basis(frame, axis, spec) @ np.asarray(u, float)
    return out


def position_limits(spec):
    return np.asarray([spec['perturbation_variables'][key]['envelope'] for key in coord_names(spec)],float)


def position_allowed(pos, spec, *, base_height=None, tolerance=0.):
    pos=np.asarray(pos,float)
    bounds=position_limits(spec)
    if pos.shape!=(2,) or not np.isfinite(pos).all() or not np.all((bounds[:,0]-tolerance<=pos)&(pos<=bounds[:,1]+tolerance)):
        return False
    floor,ceiling=bounds[1]
    if base_height is None:
        if horizontal(spec) or 'placement_geometry' not in spec:
            return True  # Original 3 mm plans have component-only bounds.
        base_height=pos[1]+spec['placement_geometry']['tcp_a_base_z']*pos[0]
    return floor-tolerance-1e-9<=base_height<=ceiling+tolerance+1e-9


def validate_cell(cell, spec, op, args):
    settings = cell.get('settings', {})
    current = bool(spec.get('current_placement'))
    witnessed=spec.get('facts',{}).get('measured_placement_coordinates',False)
    if current and not witnessed and (set(settings)-{'mass_class','closure_floor'} or cell.get('source_node_id')):
        raise ValueError('从当前接触采集，不能指定起始 a/h 坐标或复用历史起点')
    if set(settings) - set(coord_names(spec)) - {'closure_floor','approach_pose','l_offset_mm'}:
        raise ValueError('Undeclared a/h placement setting')
    if settings.get('l_offset_mm',0) or settings.get('closure_floor',.33)!=.33 or settings.get('approach_pose','pose_0')!='pose_0':
        raise ValueError('This collection profile supports a and h only')
    if any(set(b) != {'mass_class'} or b['mass_class'] not in ('light','heavy') for b in cell.get('blocks',[])):
        raise ValueError('Unknown bottle block')
    pos = np.asarray([settings.get(k,0.) for k in coord_names(spec)],float)
    step_limits=np.asarray(spec['task']['action_space']['max_step'],float)
    if (not current or witnessed) and not position_allowed(pos,spec):
        raise ValueError('起点超出本组采集范围')
    actions = cell.get('actions',[])
    if len(actions)>spec.get('runtime',{}).get('max_steps',1) or (op=='collect_seed' and actions):
        raise ValueError('计划超过本轮修正步数或在无动作试次中指定了动作')
    for action in actions:
        u = np.asarray(action['u'],float)
        if action['kind']!=action_kind(spec) or u.shape!=(2,) or not np.isfinite(u).all() or np.count_nonzero(u)!=1 or np.any(np.abs(u)>step_limits):
            raise ValueError(f'每步只动一个轴；a/h 单步上限分别为 {step_limits.tolist()} mm')
        if current and not witnessed:
            bounds=position_limits(spec)
            if np.any(u<bounds[:,0]) or np.any(u>bounds[:,1]):
                raise ValueError('所选动作超出本次起点允许的 a/h 位移')
        if args.get('axis'):
            i = {coord_names(spec)[0].split('_')[0]:0,'h':1}[args['axis']]
            if np.sign(u[i]) != {'positive':1,'negative':-1}[args['side']]:
                raise ValueError('动作与导演所选轴或方向不一致')
        pos += u
        if not position_allowed(pos,spec):
            raise ValueError('移位后超出本组采集范围')
    if op=='collect_transitions' and not actions:
        raise ValueError('需要真实的前后移动')


def convert_legacy_row(row,spec):
    """Re-express measured motion; keep old labels, receipts and vectors as provenance."""
    from bottle_scan.state_graph.skill_interface import task_id,repeat_key
    out=deepcopy(row)
    if horizontal(spec):
        p=out.setdefault('provenance',{})
        receipt=out.get('u_receipt') or {}
        old=out.get('plan_coords',{})
        p['legacy_plan_coords']=deepcopy(old)
        if 'a_offset_mm' in old:
            origin=p.get('placement_reference_pose')
            az=p.get('placement_tcp_a_base_z')
            if origin is None and p.get('coordinate_schema')=='measured_placement_relative_ah_v1' and p.get('source_file'):
                source=Path(p['source_file'])
                sibling=source.parent.parent/'center'/'observation.json'
                # Two known layouts: saved center probe in the same placement or
                # frozen placement_basis.json attached to the originating campaign.
                choices=[sibling]
                choices.extend(parent/'placement_basis.json' for parent in source.parents if (parent/'spec.json').is_file())
                for path in choices:
                    if not path.is_file():continue
                    info=json.loads(path.read_text())
                    if not p.get('reference_digest') or info.get('reference_digest')!=p['reference_digest']:continue
                    if path==sibling:
                        if info.get('bottle_static') is not True or not (info.get('probe') or {}).get('valid'):continue
                        origin=info.get('window_pose',{}).get('T_base_tcp_mm')
                    elif info.get('placement')==row.get('session'):origin=info.get('measured_pose')
                    if origin is not None:
                        p['legacy_coordinate_basis_source']=str(path)
                        break
            if origin is not None:
                az=float(np.asarray(origin)[2,2])
            if az is not None and np.isfinite(az) and abs(az)<=1 and 'h_offset_mm' in old:
                a,h=old['a_offset_mm'],old['h_offset_mm']
                out['plan_coords']={**old,'r_offset_mm':float(a*np.sqrt(1-az*az)),'h_offset_mm':float(h+a*az)}
            else:
                out['plan_coords']={}
                p['coordinate_conversion']='missing_legacy_measurement_basis'
        if receipt and receipt.get('kind')==KIND:
            p['legacy_u_receipt']=deepcopy(receipt)
            p['legacy_u_executed']=out.get('u_executed')
            frame=receipt.get('start_pose')
            if frame is None:
                out.update(u_executed=None,u_receipt=None)
            else:
                a=np.asarray(frame)[:3,2]
                M=basis(frame,[0,0,1],spec)
                delta=receipt.get('net_displacement_base_mm')
                if delta is None:
                    ua,uh=receipt['net_contact_displacement_mm']
                    delta=a*ua+np.array([0.,0.,uh])
                d=np.asarray(delta,dtype=float)
                u=M.T@d
                receipt.update(kind=RH_KIND,semantics=['horizontal_tcp_a_at_start','base_link_positive_z'],
                        raw_tcp_a_projection_mm=float(d@a),net_contact_displacement_mm=u.tolist(),
                        net_displacement_base_mm=d.tolist(),residual_mm=float(np.linalg.norm(d-M@u)),
                        commanded_axis='h' if receipt.get('commanded_axis')=='h' else None)
                out.update(u_executed=u.tolist(),u_receipt=receipt)
        out['task_id']=task_id(spec)
        out['repeat_key']=repeat_key(out,spec)
    if spec.get('consumer',{}).get('dual_paired'):
        for c in spec['post_conditions']:
            if c['id'] not in out.get('post',{}):
                out.setdefault('post',{})[c['id']]={'evaluated':False,'success':None,'level':None,'evidence':[]}
                for prev in row.get('post',{}).values():
                    for e in prev.get('evidence',[]):
                        if isinstance(e,dict) and e.get('version') and e.get('version')==c['consumer'].get('version') and e.get('controller')==c['consumer']['controller']:
                            out['post'][c['id']]=deepcopy(prev)
    return out
