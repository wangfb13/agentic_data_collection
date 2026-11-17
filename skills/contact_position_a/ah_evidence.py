"""Convert saved bilateral probes and measured translations into V4 observations."""
from pathlib import Path
import h5py
import numpy as np
from bottle_scan.hardware_arm import pose_error
from bottle_scan.state_graph.facts import fact
from . import placement_probe as probe
from .placement_al_probe import contact
from .ah_geometry import COORDS, KIND, measured, coord_names, action_kind, horizontal, basis


def iter_trials(sources,spec):
    for source in sources:
        yield from probe.read(source)


def wrench_pair(path, row):
    stamp = row['probe']['timestamp_ns']
    zero = np.asarray(row['unloaded_zero']['offset_raw'])
    values, times = [], []
    with h5py.File(path.parent/'record'/'streams.h5','r') as stream:
        for i, side in enumerate(('left','right')):
            group = stream['streams/wrench_raw_'+side]
            stamps = group['monotonic_ns'][:]
            j = int(np.argmin(abs(stamps-stamp)))
            times.append(int(stamps[j]))
            values.extend(group['data'][j]-zero[i])
    if max(abs(t-stamp) for t in times)>100_000_000 or abs(times[0]-times[1])>100_000_000:
        raise ValueError('轻触窗口与六维力记录时间不匹配')
    return values, times


def window(path, identity, center, group, index, nominal, mass, placement):
    ref, model, serials = identity
    pose, row = contact(path,ref,model,serials)
    return captured_window(path, identity, center, group, index, nominal, mass, placement, pose, row)


def captured_window(path, identity, center, group, index, nominal, mass, placement, pose, row):
    ref, model, serials = identity
    observation = row['probe']
    stamp = observation['timestamp_ns']
    fb = row['window_pose']
    if max(abs(s['frame_ns'][-1]-fb['received_monotonic_ns']) for s in observation['window'])>100_000_000:
        raise ValueError('TCP 与接触窗口不同步')
    if (center is not None and pose_error(center,pose)[1]>1) or pose_error(row['before_pose'],row['after_pose'])[0]>1 or pose_error(row['before_pose'],row['after_pose'])[1]>1:
        raise ValueError('轻触姿态超出本组测量条件')
    axis = ref['protocol']['axes_tcp']['a']
    residual=None
    coords={}
    context=[placement,ref['reference_digest'],serials]
    schema='current_contact_unlabelled_ah_v1'
    if center is not None:
        u,residual=measured(center,pose,center,axis)
        coords=dict(zip(COORDS,u)) if residual<=1 else {}
        schema='measured_placement_relative_ah_v1'
    else:
        context.append(group)  # Free starts are not independently witnessed noise repeats.
    wrench,times=wrench_pair(path,row)
    contacts=[dict(u_px=d['centroid_px'][0],v_px=d['centroid_px'][1],area_px=d['area_px']) for d in observation['descriptors']]
    return dict(node_id=group+':'+str(index),session=placement,trial_id=group,group_id=group,window_index=index,
                phase='PROBE',t_ns=stamp,contacts=contacts,wrench=wrench,preconditions={'bilateral_contact':True},
                attrs={'mass_class':mass,'closure_floor':.33,'approach_pose':'pose_0' if center is not None else None,'gripper_opening':row['touch_report']['width_normalized'],
                       'sensor_version':serials,'protocol_id':ref['reference_digest']},
                repeat_settings=nominal,plan_coords=coords,is_reference=False,is_terminal=False,
                provenance={'source_file':str(path),'coordinate_schema':schema,
                            'repeat_context':context,
                            'placement_residual_mm':residual,'measured_pose':pose.tolist(),
                            'force_frame_ns':times,'reference_digest':ref['reference_digest']})


def link(first,last,receipts,axis,*,command=None,spec=None):
    if not receipts or first['group_id']!=last['group_id']:
        return
    records=[probe.read(p) for p in receipts]
    if not all(r.get('result')=='reached_by_measured_feedback' and r.get('arm')=='right' and r.get('real') is True
               and r.get('base_frame')=='base_link' and r.get('tcp_frame')=='right_gripper_base_link' for r in records):
        return
    before=first['provenance']['measured_pose']
    after=last['provenance']['measured_pose']
    frame=records[0]['execution_start_mm']
    # Both ends of each intervening physical move must agree with the next observed pose.
    for a,b in zip([before]+[r['end_measured_mm'] for r in records],
                   [r['execution_start_mm'] for r in records]+[after]):
        translation,rotation=pose_error(a,b)
        if translation>1 or rotation>1:return
    if any(pose_error(frame,p)[1]>1 for r in records for p in (r['execution_start_mm'],r['end_measured_mm'])):
        return
    details={}
    delta=np.asarray(after)[:3,3]-np.asarray(before)[:3,3]
    a_base=np.asarray(frame)[:3,:3]@axis
    if command is None:
        u,residual=measured(before,after,frame,axis,spec)
    else:
        # The executed command selects one direction. A 2-axis inverse is
        # unnecessary even when the tool happens to point along base Z.
        k=int(np.flatnonzero(command)[0])
        rec=records[0]
        if (len(records)!=1 or rec.get('frame_semantics')!='action_start_v1'
                or rec.get('relative_frame')!=(('horizontal_tcp_a' if horizontal(spec) else 'tcp'),'base')[k]):
            return
        a_base=np.asarray(frame)[:3,:3]@axis
        direction=(basis(frame,axis,spec)[:,0] if horizontal(spec) else a_base) if k==0 else np.array([0.,0.,1.])
        delta=np.asarray(after)[:3,3]-np.asarray(before)[:3,3]
        distance=float(delta@direction)
        if distance*command[k]<=0:return
        residual=float(np.linalg.norm(delta-distance*direction))
        u=np.zeros(2)
        u[k]=distance
        details=dict(measurement='projection_on_executed_axis',commanded_axis=coord_names(spec)[k].split('_')[0],
                     net_displacement_base_mm=delta.tolist(),tcp_a_base_z=float(a_base[2]))
    if residual>(spec or {}).get('geometry',{}).get('receipt_tolerance_mm',1):return
    if horizontal(spec):
        details.update(raw_tcp_a_projection_mm=float(delta@a_base),out_of_plane_residual_mm=float(np.linalg.norm(delta-basis(frame,axis,spec)@(basis(frame,axis,spec).T@delta))),net_displacement_base_mm=delta.tolist())
    last.update(previous_node_id=first['node_id'],u_executed=u.tolist(),
                u_receipt={'applied':True,'kind':action_kind(spec),'semantics':['horizontal_tcp_a_at_start' if horizontal(spec) else 'tcp_a_at_start','base_link_positive_z'],
                           'start_pose':frame,'net_contact_displacement_mm':u.tolist(),'residual_mm':residual,
                           'move_logs':[str(p) for p in receipts],**details})


def historical(run, identity):
    axis=identity[0]['protocol']['axes_tcp']['a']
    moves=sorted((probe.read(p)['created_unix_ns'],p) for p in (run/'moves').glob('tcp_move_*.json'))
    trials=[]
    for placement in ('P1','P2'):
        directory=run/placement
        center,_=contact(directory/'center'/'observation.json',*identity)
        paths=sorted(directory.glob('*/observation.json'),key=lambda p:probe.read(p)['started_ns'])
        group='historical_'+run.name+'_'+placement
        windows=[]
        for index,path in enumerate(paths):
            w=window(path,identity,center,group,index,{},None,placement)
            if index:
                start=probe.read(paths[index-1])['started_ns']
                end=probe.read(path)['started_ns']
                link(windows[-1],w,[p for t,p in moves if start<t<end and probe.read(p)['finished_unix_ns']<=end],axis)
            windows.append(w)
        trials.append(dict(trial_id=group,windows=windows))
    return trials


def amplitude_coverage(edges, bounds, placement, index, sign,spec=None):
    """Open spaces between distinct measurements, partitioned only by real observations."""
    axis=coord_names(spec)[index].split('_')[0]
    lower, upper = (0., bounds[1]) if sign > 0 else (bounds[0], 0.)
    samples = []
    for edge in edges:
        receipt = edge.get('receipt') or {}
        if receipt.get('kind') != action_kind(spec) or receipt.get('applied') is not True:
            continue
        net = receipt.get('net_contact_displacement_mm')
        if net is None or len(net) != 2 or not np.isfinite(net).all():
            continue
        value, cross = net[index], net[1-index]
        if sign * value <= 0 or abs(cross) > 1:
            continue
        commanded_axis = receipt.get('commanded_axis')
        if commanded_axis is None:
            # Older receipts decomposed both components without naming a command axis.
            if abs(value) <= abs(cross):
                continue
        elif commanded_axis != axis:
            continue
        samples.append({'edge_id': edge['edge_id'], 'measured_mm': value})

    points = sorted({lower, upper, *(s['measured_mm'] for s in samples
                                      if lower <= s['measured_mm'] <= upper)})
    return fact('action_amplitude_coverage', [placement, axis, 'positive' if sign > 0 else 'negative'],
                [s['edge_id'] for s in samples], {
                    'measured_transitions': samples,
                    'signed_bounds_mm': [lower, upper],
                    'unsampled_intervals_mm': [[a, b] for a, b in zip(points, points[1:])],
                    'measurement': 'Measured contact-to-contact displacements. Each interval interior contains no '
                                   'measured point. A nonzero envelope endpoint is untested unless it appears '
                                   'in measured_transitions. Zero is the boundary between directions.',
                })


def exploration_context(rows,graph,facts,spec):
    ops=[{'op':'pause','bindings':{},'premises_satisfied_by':[]}]
    facts=list(facts)
    edges=[e for e in graph['edges'] if e['kind']=='observed']
    placement=spec.get('current_placement')
    if placement:
        current_ids={r['node_id'] for r in rows if r.get('session')==placement}
        edges=[e for e in edges if e['source'] in current_ids and e['target'] in current_ids]
        if not current_ids:
            missing=fact('current_placement_unsampled',[placement],[],{'start_offset_known':False})
            facts.append(missing)
            ops.append(dict(op='collect_seed',bindings={'post_condition':spec['active_post_condition']},
                            premises_satisfied_by=[missing['id']]))
    cid=spec['active_post_condition']
    capability=spec['capabilities'][action_kind(spec)]
    available=[]
    for i,key in enumerate(coord_names(spec)):
        axis=key.split('_')[0]
        if capability['kind']!='vector' or capability['dim']!=2 or capability['max_action'][i]<=0:
            continue
        cap=fact('translation_available',[axis],[],{'kind':action_kind(spec),'frame':['horizontal_tcp_a' if horizontal(spec) else 'current_tcp_a','base_link_Z'][i],
                 'unit':'mm','max_command_mm':capability['max_action'][i],'scope':placement or 'declared_for_P2_exploration'})
        available.append(cap['id'])
        facts.append(cap)
        for sign,side in ((1,'positive'),(-1,'negative')):
            if placement:
                bounds=spec['perturbation_variables'][coord_names(spec)[i]]['envelope']
                coverage=amplitude_coverage(edges,bounds,placement,i,sign,spec)
                facts.append(coverage)
                matches=coverage['evidence']
            else:
                matches=[e['edge_id'] for e in edges if sign*e['u'][i]>=1 and abs(e['u'][1-i])<=1]
            evidence=fact('direction_sampled' if matches else 'direction_sample_missing',[axis,side],matches,
                          {'edges':len(matches),'unit':'mm','cross_component_tolerance_mm':1})
            facts.append(evidence)
            premises=[cap['id'],evidence['id']]
            if placement:
                premises.append(coverage['id'])
            ops.append(dict(op='collect_transitions',bindings={'axis':axis,'side':side,'post_condition':cid},
                            premises_satisfied_by=premises))
    if len(available)==2:
        # A mixed batch compares directions, with one axis per physical trial.
        ops.append(dict(op='collect_transitions',bindings={'post_condition':cid},premises_satisfied_by=available))
    if 'placement_geometry' in spec and spec.get('graph_schema')!='state_graph_v5':
        observed=[(r['node_id'],r['plan_coords']['h_offset_mm']) for r in rows if r.get('session')=='P2'
                  and 'h_offset_mm' in r.get('plan_coords',{}) and r['window_valid']]
        ceiling=spec['perturbation_variables']['h_offset_mm']['envelope'][1]
        highest=max((h for _,h in observed),default=None)
        if highest is None or highest+2 < ceiling:
            gap=fact('upper_h_contact_coverage_gap',['P2'],[k for k,_ in observed],
                     {'highest_valid_contact_h_mm':highest,'allowed_upper_h_mm':ceiling,
                      'meaning':'Upper range has no validated contact windows; bottle part and recovery outcome are unknown.'})
            facts.append(gap)
            ops.append(dict(op='collect_seed',bindings={'post_condition':cid,'var':'h_offset_mm'},
                            premises_satisfied_by=[gap['id']]))
    for f in facts:
        if f['pred']=='repeat_calibration_missing' and f['value']:
            ops.append(dict(op='collect_seed',bindings={'post_condition':cid},premises_satisfied_by=[f['id']]))
    return facts,ops
