"""Offline definitions for the independent E1/E2 protocol. No hardware imports."""
from __future__ import annotations
import copy
import math
from pathlib import Path
import numpy as np
import yaml
from .sensing import dd_descriptors
from .session import fingerprint, plain

DEFAULT_OFFSETS = [-20., -15., -10., -5., 0., 5., 10., 15., 20.]
LABELS = {'': 'normal', '1': 'normal', '2': 'tilted', '3': 'pushed',
          '4': 'dropped', '5': 'slipped', '0': 'invalid'}


def opening_within_tolerance(normalized, gripper):
    """Use inclusive endpoints so the 999/1000 readback passes a one-count tolerance."""
    target, tolerance = gripper['open_normalized'], gripper['open_tolerance']
    return target - tolerance <= normalized <= target + tolerance


def defaults():
    return {
        'schema': 'e1e2_config_v1', 'arm': 'right',
        'axes_tcp': {'a': [0., 0., 1.], 'c': [0., -1., 0.], 'l': [1., 0., 0.]},
        'force': {'unit': 'sdk', 'target': 5., 'max': 15., 'min_contact': .05,
                  'normal_axis': 2, 'normal_signs': [-1., -1.], 'probe_fraction': .3,
                  'probe_tolerance': .25, 'squeeze_tolerance': .5,
                  'probe_stable_s': .2, 'squeeze_stable_s': .5, 'timeout_s': 30.,
                  'step_counts': 1, 'control_period_s': .05, 'command_settle_s': .10},
        'gripper': {'open_normalized': 1., 'min_normalized': None,
                    'body_contact_record': 'station_profile_setup/width_body/calibration.json',
                    'contact_margin_normalized': .03, 'close_speed_normalized_s': .025,
                    'open_speed_normalized_s': .10, 'stall_window_s': 2.0,
                    'fast_close_until_normalized': .5, 'fast_close_speed_normalized_s': .20,
                    'stall_epsilon_normalized': .0002, 'close_start_timeout_s': 1.0,
                    'command_response_s': .5, 'open_tolerance': .001,
                    'position_feedback_tolerance_counts': 1,
                    'position_max_tracking_error_counts': 2,
                    'timeout_s': 240.},
        'motion': {'free_speed_mm_s': 20., 'near_speed_mm_s': 5., 'near_distance_mm': 50.,
                   'max_grasp_offset_mm': 25., 'pre_mm': 40., 'lift_mm': 40.,
                   'max_lift_mm': 60., 'z_below_T0_mm': 10., 'pose_tolerance_mm': 2.,
                   'pose_tolerance_deg': 1., 'timeout_margin_s': 10.,
                   'bounds_policy': 'separate'},
        'tactile': {'depth_channel': 0, 'dd_threshold': .022, 'edge_pixels': 3,
                    'probe_frames': 5, 'min_area_px': 4},
        'servo': {'fit': None, 'tol_mm': 2., 'max_regrasps': 3, 'max_delta_mm': 15.,
                  'retreat_mm': 15., 'no_contact_delta_mm': 8.,
                  'stall_delta_mm': -8., 'root_edge_delta_mm': -6.,
                  'tip_edge_delta_mm': 6., 'lift_without_contact': False},
        'design': {'offsets_mm': DEFAULT_OFFSETS, 'reps': 10, 'seed': 20261029,
                   'bottles': ['light', 'heavy'], 'zero_every': 10,
                   'drift_mm': 1.5, 'step_first_trials': 0},
        'zero': {'frames': 60, 'timeout_s': 12.}, 'hold_s': 2.,
        'lower': {'unload_ratio': .2, 'unload_abs_sdk': .5, 'min_descent_mm': 1.,
                  'mapping_min_cosine': .7, 'frames': 3},
        'hardware_config': 'config.ros.yaml',
        'recording': {'writer': {'queue_bytes': 134217728, 'queue_items': 4096,
                                'startup_timeout_s': 15., 'close_timeout_s': 30.}},
    }


def merge(base, extra):
    out = copy.deepcopy(base)
    for k, v in extra.items():
        out[k] = merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


def load_settings(path):
    cfg = merge(defaults(), yaml.safe_load(Path(path).read_text()) or {})
    validate(cfg, require_resolved=False)
    return cfg


def validate(cfg, *, require_resolved=True):
    def number(v, name, *, zero=False):
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or (v < 0 if zero else v <= 0):
            raise ValueError(name + ' 必须是有限' + ('非负数' if zero else '正数'))
    for section, names in {
        'force': ['target','max','min_contact','probe_fraction','probe_tolerance','squeeze_tolerance',
                  'probe_stable_s','squeeze_stable_s','timeout_s','step_counts','control_period_s','command_settle_s'],
        'gripper': ['open_normalized','close_speed_normalized_s','open_speed_normalized_s','stall_window_s',
                    'stall_epsilon_normalized','open_tolerance','timeout_s'],
        'motion': ['free_speed_mm_s','near_speed_mm_s','near_distance_mm','max_grasp_offset_mm','pre_mm',
                   'lift_mm','max_lift_mm','pose_tolerance_mm','pose_tolerance_deg','timeout_margin_s'],
        'servo': ['tol_mm','max_delta_mm','retreat_mm'],
        'zero': ['frames','timeout_s'], 'lower': ['unload_ratio','unload_abs_sdk','min_descent_mm','mapping_min_cosine','frames'],
        'tactile': ['probe_frames','min_area_px'],
    }.items():
        for k in names: number(cfg[section][k], section+'.'+k)
    for sec, key in [('gripper','contact_margin_normalized'),('motion','z_below_T0_mm'),
                     ('tactile','dd_threshold'),('tactile','edge_pixels')]: number(cfg[sec][key],sec+'.'+key,zero=True)
    for key in ['close_start_timeout_s','command_response_s']:
        number(cfg['gripper'].get(key,defaults()['gripper'][key]),'gripper.'+key)
    for key in ['fast_close_until_normalized','fast_close_speed_normalized_s']:
        number(cfg['gripper'].get(key,defaults()['gripper'][key]),'gripper.'+key)
    feedback_counts=cfg['gripper'].get('position_feedback_tolerance_counts',1)
    max_error_counts=cfg['gripper'].get('position_max_tracking_error_counts',2)
    if type(feedback_counts) is not int or feedback_counts!=1 or type(max_error_counts) is not int or max_error_counts!=2:
        raise ValueError('慢闭合反馈容差固定为1个归一化刻度，跟随误差上限为2个刻度')
    if cfg['gripper'].get('fast_close_until_normalized',.5)>1.:
        raise ValueError('快闭切换开度必须在(0,1]内')
    number(cfg['hold_s'],'hold_s')
    for sec,key in [('design','reps'),('design','zero_every'),('design','step_first_trials'),
                    ('servo','max_regrasps'),('zero','frames'),('tactile','probe_frames'),
                    ('tactile','edge_pixels'),('lower','frames'),('force','step_counts')]:
        v=cfg[sec][key]
        if type(v) is not int or v < (0 if key in ('zero_every','step_first_trials','edge_pixels','max_regrasps') else 1):
            raise ValueError(sec+'.'+key+' 必须是合法整数')
    if cfg['zero']['frames']<2:raise ValueError('空载标准差至少需要2个独立样本')
    if cfg['arm'] not in ('right','left') or cfg['force']['unit']!='sdk': raise ValueError('需选择左右臂和 SDK 力单位')
    if cfg['force']['normal_axis'] not in (0,1,2) or len(cfg['force']['normal_signs'])!=2 or any(v not in (-1,1) for v in cfg['force']['normal_signs']):
        raise ValueError('需逐指指定法向轴和 ±1 符号')
    f=cfg['force']; probe=f['probe_fraction']*f['target']
    if not 0<f['probe_fraction']<1 or not f['min_contact']<probe-f['probe_tolerance'] or f['target']+f['squeeze_tolerance']>=f['max']:
        raise ValueError('需 contact_min < PROBE 力带下限，且目标力带低于 F_max')
    if f['step_counts']>2: raise ValueError('归一化力控单步最多 0.002')
    g=cfg['gripper']; floor=g['min_normalized']
    if not 0<g['open_normalized']<=1 or (floor is not None and not 0<=floor<g['open_normalized']):
        raise ValueError('归一化下限必须小于张开值，且位于 [0,1]')
    if require_resolved and floor is None: raise ValueError('尚未输入归一化闭合下限')
    axes=np.column_stack([cfg['axes_tcp'][k] for k in ('a','c','l')]).astype(float)
    if axes.shape!=(3,3) or not np.isfinite(axes).all() or not np.allclose(axes.T@axes,np.eye(3),atol=1e-5):
        raise ValueError('a/c/l 需为正交单位向量')
    m=cfg['motion']
    if m['pre_mm']>40 or cfg['servo']['retreat_mm']>40:raise ValueError('退让目标最多40mm')
    if m['max_grasp_offset_mm']>25 or m['max_lift_mm']>60 or m['lift_mm']>m['max_lift_mm']:
        raise ValueError('抓取平移最多25 mm；提起最多60 mm')
    if m['bounds_policy'] not in (None,'separate','strict25') or (require_resolved and m['bounds_policy'] is None):
        raise ValueError('需明确距离冲突的处理策略')
    if cfg['servo']['lift_without_contact'] not in (None,True,False) or (require_resolved and cfg['servo']['lift_without_contact'] is None):
        raise ValueError('需明确耗尽重抓后的处理策略')
    if len(cfg['design']['bottles'])!=2 or len(set(cfg['design']['bottles']))!=2 or cfg['design']['reps']%2:
        raise ValueError('轻重瓶分块，各档 reps 必须为偶数')
    if not cfg['design']['offsets_mm'] or any(isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x) or abs(x)>25 for x in cfg['design']['offsets_mm']):
        raise ValueError('offsets 需有限，且每个绝对值不超过25 mm')
    if len(set(cfg['design']['offsets_mm']))!=len(cfg['design']['offsets_mm']): raise ValueError('偏移列表不能重复')
    if not 0<cfg['lower']['unload_ratio']<1 or not 0<cfg['lower']['mapping_min_cosine']<=1: raise ValueError('卸载参数不合法')
    if not 0<cfg['servo']['max_delta_mm']<=15 or not 0<=cfg['servo']['max_regrasps']<=3: raise ValueError('最多重抓3次，每次修正最多15mm')
    number(cfg['design']['drift_mm'],'design.drift_mm')


def plan_trials(cfg, cond):
    d=cfg['design']; rng=np.random.default_rng(d['seed']); rows=[]; count=0
    for bottle in d['bottles']:
        block=[{'bottle':bottle,'delta_mm':float(x),'repeat':r,'kind':'scan','cond':cond}
               for r in range(d['reps']//2) for x in d['offsets_mm']]
        rng.shuffle(block)
        for row in block:
            rows.append(row);count+=1
            if d['zero_every'] and count%d['zero_every']==0:
                rows.append({'bottle':bottle,'delta_mm':0.,'repeat':None,'kind':'zero_monitor','cond':cond})
    for i,row in enumerate(rows): row.update(id=f'{i+1:05d}',index=i+1)
    return rows


def target_pose(T0, axis_base, delta):
    p=np.asarray(T0,float).copy();p[:3,3]+=np.asarray(axis_base)*float(delta);return p


class MotionBounds:
    """Translation-only point bounds; deliberately no collision/workspace model."""
    def __init__(self,cfg,T0):
        self.cfg=cfg;self.T0=np.asarray(T0,float).copy()
        self.a=self.T0[:3,:3]@np.asarray(cfg['axes_tcp']['a'],float)
        self.l=self.T0[:3,:3]@np.asarray(cfg['axes_tcp']['l'],float)
    def grasp(self,delta):return target_pose(self.T0,self.a,delta)
    def pre(self,grasp,retreat=None):return target_pose(grasp,self.a,-(self.cfg['motion']['pre_mm'] if retreat is None else retreat))
    def lift(self,grasp):
        p=np.asarray(grasp,float).copy();p[2,3]+=self.cfg['motion']['lift_mm'];return p
    def check(self,pose,role='grasp',anchor=None):
        p=np.asarray(pose,float);m=self.cfg['motion']
        if p.shape!=(4,4) or not np.isfinite(p).all() or not np.allclose(p[3],[0,0,0,1]) or not np.allclose(p[:3,:3],self.T0[:3,:3],atol=1e-7):return '非有限位姿或含旋转'
        if p[2,3]<self.T0[2,3]-m['z_below_T0_mm']-1e-7:return 'TCP z 低于 T0 高度下限'
        if m['bounds_policy']=='strict25' or role=='grasp':
            return None if np.linalg.norm(p[:3,3]-self.T0[:3,3])<=m['max_grasp_offset_mm']+1e-7 else '目标相对 T0 超过25 mm'
        if anchor is None:return '退让／提起缺少已检查的抓取锚点'
        error=self.check(anchor,'grasp')
        if error:return error
        v=p[:3,3]-np.asarray(anchor)[:3,3]
        if role=='retreat':
            along=float(v@self.a)
            return None if -m['pre_mm']-1e-7<=along<=1e-7 and np.linalg.norm(v-along*self.a)<1e-6 else '退让超过单独行程限制'
        if role in ('lift','lower'):
            return None if np.linalg.norm(v[:2])<1e-6 and -1e-7<=v[2]<=m['max_lift_mm']+1e-7 else '提起超过竖直行程限制'
        return '未知运动角色'


def describe(dd,reference,cfg):
    t=cfg['tactile'];out=dd_descriptors(dd,reference,depth_channel=t['depth_channel'],
        contact_threshold=t['dd_threshold'],edge_width_px=t['edge_pixels'])
    v=np.asarray(dd,float);r=np.asarray(reference,float)
    if v.ndim==3:v=v[...,t['depth_channel']]
    if r.ndim==3:r=r[...,t['depth_channel']]
    mask=v-r>t['dd_threshold'];yy,xx=np.nonzero(mask);h,w=mask.shape;edge=t['edge_pixels']
    bbox=None if not len(xx) else [int(xx.min()),int(yy.min()),int(xx.max()),int(yy.max())]
    sides={'left':False,'right':False,'top':False,'bottom':False}
    if bbox is not None:
        sides=dict(left=bbox[0]<edge,right=bbox[2]>=w-edge,top=bbox[1]<edge,bottom=bbox[3]>=h-edge)
    angle=None
    if out['second_moment_px2'] is not None:
        eig,vec=np.linalg.eigh(out['second_moment_px2']);major=vec[:,np.argmax(eig)]
        angle=float((np.degrees(np.arctan2(major[1],major[0]))+90)%180-90) if eig[-1]-eig[0]>1e-5*max(eig[-1],1.) else None
    out.update(bbox_xyxy_px=bbox,edge_sides=sides,touches_edge=any(sides.values()),
        principal_angle_deg=angle,shape_hw=[h,w],valid=out['area_px']>=t['min_area_px'] and out['centroid_px'] is not None)
    # Compact optional ContactPolicy input, with no change to existing contact
    # geometry or threshold decisions. Store numerical means, not image bytes.
    if min(h,w)>=4:
        positive=np.maximum(v-r,0.)
        out['dd_pool_4x4']=[[float(block.mean()) for block in np.array_split(row,4,axis=1)]
                            for row in np.array_split(positive,4,axis=0)]
    return out


class Correction:
    """Inputs intentionally exclude planned delta and robot target coordinates."""
    def __init__(self,fit,settings):
        self.fit=fit;self.cfg=settings
        if fit.get('schema')!='e1e2_fit_v1' or fit.get('usable_for_B') is not True:raise ValueError('E1 拟合不可用于 B')
        if len(fit.get('fingers',[]))!=2:raise ValueError('需两指 E1 拟合')
        for f in fit['fingers']:
            if f.get('coordinate') not in ('u','v') or not np.isfinite([f.get('slope_px_per_mm',np.nan),f.get('intercept_px',np.nan)]).all() or abs(f['slope_px_per_mm'])<1e-6 or f.get('direction_sign')!=int(np.sign(f['slope_px_per_mm'])):
                raise ValueError('E1 方向符号或斜率无效')
    def estimate(self,descriptors):
        vals=[]
        for desc,fit in zip(descriptors,self.fit['fingers']):
            if not desc or not desc.get('valid'):return None
            idx=0 if fit['coordinate']=='u' else 1
            vals.append((desc['centroid_px'][idx]-fit['intercept_px'])/fit['slope_px_per_mm'])
        return float(np.mean(vals))
    def decide(self,status,descriptors,bilateral):
        e=self.estimate(descriptors) if descriptors else None
        edge=[]
        for d,f in zip(descriptors or [],self.fit['fingers']):
            if not d:continue
            axis=f['coordinate'];positive='right' if axis=='u' else 'bottom';negative='left' if axis=='u' else 'top'
            root=positive if f['direction_sign']>0 else negative
            tip=negative if f['direction_sign']>0 else positive
            if d.get('edge_sides',{}).get(root):edge.append('root')
            if d.get('edge_sides',{}).get(tip):edge.append('tip')
            if d.get('touches_edge') and not (d['edge_sides'].get(root) or d['edge_sides'].get(tip)):edge.append('transverse')
        if bilateral and e is not None and abs(e)<=self.cfg['tol_mm'] and not any(d.get('touches_edge') for d in descriptors):
            return dict(accepted=True,e_mm=e,delta_mm=0.,edges=edge,reason='within_tolerance')
        if status=='no_contact':delta=self.cfg['no_contact_delta_mm'];reason='no_contact_forward'
        elif status=='stall_no_contact':delta=self.cfg['stall_delta_mm'];reason='stall_back'
        elif 'root' in edge and 'tip' not in edge:delta=self.cfg['root_edge_delta_mm'];reason='root_edge'
        elif 'tip' in edge and 'root' not in edge:delta=self.cfg['tip_edge_delta_mm'];reason='tip_edge'
        elif e is not None:delta=-e;reason='centroid_error'
        else:delta=0.;reason='unobservable_contact'
        return dict(accepted=False,e_mm=e,delta_mm=float(np.clip(delta,-self.cfg['max_delta_mm'],self.cfg['max_delta_mm'])),edges=edge,reason=reason)


def config_identity(cfg,cond,fit):
    return fingerprint({'config':plain(cfg),'cond':cond,'fit':fit})
