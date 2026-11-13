"""Install the opt-in graph decision hook in V4; never connect to hardware."""
import argparse
import ast
from pathlib import Path
import json
import shutil
import time


def install_probe_window_observation(root,bundle):
    """Use completed PROBE evidence for graph windows; retain live safety/contact."""
    root=Path(root).resolve();bundle=Path(bundle).resolve()
    target=bundle/'runtime/bottle_scan/contact_recovery_backend.py'
    source=target.read_text()
    marker='        # Graph features and force stability refer to the same completed PROBE.'
    if marker in source:return {'changed':False,'path':str(target)}
    old='''        stable = bool(p.get("valid") and status == "contact" and bilateral
                      and abs(float(np.mean(sample["normal"])) -
                              force["target"] * force["probe_fraction"]) <= force["probe_tolerance"]
                      and self.policy._completed_stable_probe(p))'''
    new='''        # Graph features and force stability refer to the same completed PROBE.
        # _sample() still checks live force/freshness; bilateral above is live.
        # A later force tick must not relabel the already recorded window.
        normal = np.asarray(p.get("normal_sdk", [np.nan, np.nan])
                            if self.config.get("state_graph_execution") else sample["normal"], dtype=float)
        stable = bool(p.get("valid") and status == "contact" and bilateral
                      and normal.shape == (2,) and np.isfinite(normal).all()
                      and np.all(normal >= self.hw.zero["contact_thresholds_sdk"])
                      and abs(float(np.mean(normal)) -
                              force["target"] * force["probe_fraction"]) <= force["probe_tolerance"]
                      and self.policy._completed_stable_probe(p))'''
    if source.count(old)!=1:raise ValueError('Unexpected recovery observation implementation')
    updated=source.replace(old,new);ast.parse(updated)
    backup=root/'outputs/state_graph/v4_install_backups'/str(time.time_ns());backup.mkdir(parents=True)
    shutil.copy2(target,backup/target.name);target.write_text(updated)
    return {'changed':True,'path':str(target),'backup':str(backup/target.name)}


def install(root,bundle):
    root=Path(root).resolve();bundle=Path(bundle).resolve()
    target=bundle/'runtime/bottle_scan/contact_scheduler.py';text=target.read_text()
    marker='        # Explicit graph V4 decision adapter (legacy entries unchanged).'
    if marker not in text:
        needle='        backend.set_cancel_check(adapter.stop_requested)'
        if text.count(needle)!=1:raise ValueError('Unexpected V4 scheduler hook location')
        insertion='''        # Explicit graph V4 decision adapter (legacy entries unchanged).
        graph_execution = config.get("state_graph_execution")
        if graph_execution:
            import sys
            import bottle_scan
            from pathlib import Path
            graph_root = Path(graph_execution["repository"])
            if str(graph_root / "bottle_scan") not in bottle_scan.__path__:
                bottle_scan.__path__.append(str(graph_root / "bottle_scan"))
            if str(graph_root) not in sys.path:
                sys.path.append(str(graph_root))
            from skills.contact_position_a.runtime import install_backend
            install_backend(backend, graph_execution)
'''
        # Do not import Path inside build_runtime: it would shadow its earlier use.
        insertion=insertion.replace('            from pathlib import Path\n','')
        updated=text.replace(needle,insertion+needle);ast.parse(updated)
        backup=root/'outputs/state_graph/v4_install_backups'/str(time.time_ns())
        backup.mkdir(parents=True);shutil.copy2(target,backup/target.name);target.write_text(updated)
    # The old controlled B entry only allowed 0 or |offset| in [3,10].
    # Graph exploration must also execute small boundary probes such as 2.5 mm.
    # Its explicit config uses the already-declared physical motion envelope.
    scheduler=target.read_text()
    if 'graph_config=None' not in scheduler:
        scheduler=scheduler.replace('def experiment_b_offset_from_environment(*, allow_zero=False):',
            'def experiment_b_offset_from_environment(*, allow_zero=False, graph_config=None):')
        old='    if not math.isfinite(offset) or not (3. <= abs(offset) <= 10. or (allow_zero and offset == 0.)):'
        new='    allowed = (abs(offset) <= graph_config["execution_config"]["motion"]["max_grasp_offset_mm"]) if graph_config and graph_config.get("state_graph_execution") else (3. <= abs(offset) <= 10. or (allow_zero and offset == 0.))\n    if not math.isfinite(offset) or not allowed:'
        if scheduler.count(old)!=1:raise ValueError('Unexpected offset parser implementation')
        scheduler=scheduler.replace(old,new).replace('experiment_b_offset_from_environment(allow_zero=bool(config.get("recovery_skill_bundle")))',
            'experiment_b_offset_from_environment(allow_zero=bool(config.get("recovery_skill_bundle")), graph_config=config)')
        ast.parse(scheduler)
        backup=root/'outputs/state_graph/v4_install_backups'/str(time.time_ns());backup.mkdir(parents=True)
        shutil.copy2(target,backup/target.name);target.write_text(scheduler)
    physical=bundle/'runtime/bottle_scan/contact_skill_backend.py';source=physical.read_text()
    if 'graph_offset_allowed' not in source:
        old="        if not np.isfinite(offset) or not -10. <= offset <= 10. or (abs(offset) < 3. and not (offset == 0. and self.config.get('recovery_skill_bundle'))):"
        new="        graph_offset_allowed = abs(offset) <= self.cfg['motion']['max_grasp_offset_mm'] if self.config.get('state_graph_execution') else (-10. <= offset <= 10. and (abs(offset) >= 3. or (offset == 0. and self.config.get('recovery_skill_bundle'))))\n        if not np.isfinite(offset) or not graph_offset_allowed:"
        if source.count(old)!=1:raise ValueError('Unexpected physical prelude implementation')
        source=source.replace(old,new);ast.parse(source)
        backup=root/'outputs/state_graph/v4_install_backups'/str(time.time_ns());backup.mkdir(parents=True)
        shutil.copy2(physical,backup/physical.name);physical.write_text(source)
    messages={target:('raise ValueError("CONTACT_EXPERIMENT_B_OFFSET_MM must be between -10 and +10 and at least 3 mm from zero")',
                      'raise ValueError("Graph offset exceeds configured motion envelope" if graph_config and graph_config.get("state_graph_execution") else "CONTACT_EXPERIMENT_B_OFFSET_MM must be between -10 and +10 and at least 3 mm from zero")'),
              physical:("raise ValueError('Test offset must be between -10 and +10 mm and at least 3 mm from zero')",
                        "raise ValueError('Graph offset exceeds configured motion envelope' if self.config.get('state_graph_execution') else 'Test offset must be between -10 and +10 mm and at least 3 mm from zero')")}
    for file,(old,new) in messages.items():
        before=file.read_text();after=before.replace(old,new)
        if before!=after:
            ast.parse(after);backup=root/'outputs/state_graph/v4_install_backups'/str(time.time_ns());backup.mkdir(parents=True)
            shutil.copy2(file,backup/file.name);file.write_text(after)
    probe_fix=install_probe_window_observation(root,bundle)
    config=json.loads((bundle/'contact_scheduler.state_graph_v3_0330.json').read_text())
    config['state_graph_recording']={'enabled':True,'adapter_path':str(root/'skills/contact_position_a/scheduler.py'),'wrist_camera_key':'right_wrist'}
    config['single_correction_open_handoff']=False
    destination=root/'config.state_graph.v4_scheduler.json';destination.write_text(json.dumps(config,ensure_ascii=False,indent=2)+'\n')
    return {'hook':str(target),'base_config':str(destination),'probe_observation':probe_fix,'robot_started':False}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',default=str(Path(__file__).resolve().parents[2]));p.add_argument('--bundle',default='/home/dm/DM_das/contact_scheduler_20261104_experiment_b_v4_single_open_20261110')
    a=p.parse_args();print(json.dumps(install(a.root,a.bundle),ensure_ascii=False))
if __name__=='__main__':main()
