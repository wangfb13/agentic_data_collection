"""Placement continuity qualifies coordinate/transition evidence, not contact validity."""
from bottle_scan.state_graph.campaign import atomic


def unlabelled_contact(raw, confirmation, *, original_observation, refreshed):
    if raw is None:return None
    raw['plan_coords']={}
    raw['is_reference']=False
    p=raw['provenance']
    p.update(coordinate_source=None,placement_reference_valid=False,
             placement_continuity=False,placement_confirmation=confirmation,
             original_observation=str(original_observation),fresh_after_placement_confirmation=bool(refreshed),
             repeat_context=[raw['group_id'],raw['node_id'],'placement_unwitnessed'])
    for key in ('coordinate_evidence_id','placement_reference_pose','calibration_epoch'):
        p.pop(key,None)
    return raw


def reference_unavailable(io, request, reason, raw=None):
    """Continue from current contact; never invent a new standard or execute an old-origin target."""
    io.invalidate_reference()
    report=dict(schema='p3_optional_coordinate_witness_v1',status='unavailable',reason=reason,
                requested_settings=request['settings'],requested_initial_coordinates_executed=False,
                actual_initial_coordinates=None,robot_movement_sent=False,
                source_file=(raw or {}).get('provenance',{}).get('source_file'))
    io.preparation_report=report
    atomic(io.directory/'preparation.json',report)
    print('[P3] 本次不建立位置零点；从当前接触继续，毫米坐标留空，不执行旧参考位置的准备移动。',flush=True)
    return report
