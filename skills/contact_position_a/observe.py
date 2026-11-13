"""Historical E1/E2/RSI parsing and current tactile-window adaptation."""
from collections import defaultdict
from pathlib import Path
import json
from .legacy_v2.provenance import prepare_nodes


def iter_trials(sources,spec):
    groups=defaultdict(list)
    for source in sources:
        path=Path(source)
        rows=json.loads(path.read_text()) if path.suffix=='.json' else [json.loads(line) for line in path.read_text().splitlines() if line]
        # A scheduler episode can terminate before its first valid graph window.
        # It is still an event log, not historical rows with session/trial keys.
        if path.name=='events.jsonl' or any(r.get('event') in ('state_graph_window','graph_runtime_selected') for r in rows):
            from .scheduler import trials_from_events
            yield from trials_from_events(rows,spec);continue
        for row in rows:groups[(row['session'],row['trial_id'])].append(row)
    for (session,trial),rows in sorted(groups.items()):
        yield {'session':session,'trial_id':trial,'rows':sorted(rows,key=lambda r:r['window_index'])}


def probe_window(trial,spec):
    if 'windows' in trial:
        yield from trial['windows'];return
    for row in trial['rows']:
        obj=row['object_id'];category='white_bottle' if obj.startswith('white') else 'green_bottle'
        mass='heavy' if obj.endswith('heavy') else 'light'
        de=row.get('downstream_evidence') or {}
        terminal=de.get('evaluated') is True
        attrs={k:row.get(k) for k in spec['observation']['record_only']}
        attrs.update(mass_class=mass,a_offset_mm=row.get('planned_delta_mm'),l_offset_mm=row.get('tcp_offset_l'),
            approach_pose=row.get('approach_pose') or 'unrecorded',
            unloaded_reference_id=(row.get('unloaded_reference') or {}).get('sha256') if isinstance(row.get('unloaded_reference'),dict) else None)
        result={k:row.get(k) for k in ('node_id','session','trial_id','window_index','phase','group_id','previous_node_id','evaluation_only','evaluation_scope','frozen_independent','rule_id')}
        result.update(task_id=category+'|'+spec['task']['action_space']['id'],t_ns=row.get('timestamp_ns'),
            preconditions={'bilateral_contact':row.get('bilateral_contact')},is_terminal=terminal,
            contacts=[row['sensor_contact_left'],row['sensor_contact_right']],wrench=row['wrench_left']+row['wrench_right'],visual=row['z_vis'],tactile_frames=row.get('frames'),
            extractor_versions={k:v['version'] for k,v in spec['observation']['modalities'].items()},attrs=attrs,
            plan_coords={'a_offset_mm':row.get('tcp_offset_a'),'l_offset_mm':row.get('tcp_offset_l')},
            repeat_settings={'a_offset_mm':row.get('planned_delta_mm'),'l_offset_mm':0.,'closure_floor':row.get('closure_floor'),'approach_pose':attrs['approach_pose']},
            is_reference=row.get('planned_delta_mm')==0,
            provenance={**row.get('provenance',{}),'source_type':'historical_collector','trial_started_unix_ns':row.get('trial_started_unix_ns'),'original_source_role':row.get('original_source_role',row.get('source_role')),
                        'frames':row.get('frames'),'wrist_frame':row.get('wrist_frame'),'T0_mm':row.get('T0_mm'),'a_axis_base':row.get('a_axis_base')},
            legacy_labels=row.get('original_skill_ground_truth') or {},risk={'normal_force_peak':max(row.get('normal_sdk') or [0]),'contact_lost':False},
            deprecated_aliases={'downstream_success':row.get('downstream_success'),'tcp_offset_a':row.get('tcp_offset_a')},_legacy=row)
        yield result


def wrench_zeroed(window,spec=None):return window['wrench']
def wrist_dinov2(window,spec=None):return window['visual']


def extract_raw_sessions(sources,out,config=None):
    # Original parsing lives in this instance; new generic extraction does not
    # know collector stage names, source file layout or controller conventions.
    from .legacy_v2.extract_nodes import extract_sessions
    return extract_sessions(sources,out_dir=out,config=config)
