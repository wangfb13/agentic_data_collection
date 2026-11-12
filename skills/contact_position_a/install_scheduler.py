"""Install an explicit v3 recording entry into the existing V4 bundle; never run it."""
from pathlib import Path
import json
import shutil
import ast

ROOT=Path('/home/dm/DM_das/bottle_grasp_experiment')
BUNDLE=Path('/home/dm/DM_das/contact_scheduler_20261104_experiment_b_v4_single_open_20261110')
ADAPTER=ROOT/'skills/contact_position_a/scheduler.py'
BACKUP=ROOT/'outputs/state_graph/20261112_symbolic_v3/scheduler_code_before'


def install():
    BACKUP.mkdir(parents=True,exist_ok=True)
    backend=BUNDLE/'runtime/bottle_scan/contact_recovery_backend.py'
    scheduler=BUNDLE/'runtime/bottle_scan/contact_scheduler.py'
    for p in (backend,scheduler):
        q=BACKUP/p.name
        if not q.exists():shutil.copy2(p,q)
    b=backend.read_text()
    if 'def _state_graph_window' not in b:
        method='''    def _state_graph_window(self, probe, folder, index, previous, receipt):
        settings = self.config.get("state_graph_recording", {})
        if not settings.get("enabled"):
            return None
        if not hasattr(self, "_graph_adapter"):
            import importlib.util
            definition = importlib.util.spec_from_file_location("contact_graph_evidence", settings["adapter_path"])
            self._graph_adapter = importlib.util.module_from_spec(definition)
            definition.loader.exec_module(self._graph_adapter)
            self._graph_consumer = self._graph_adapter.consumer_identity()
        window = self._graph_adapter.capture_window(self, probe, self.hw.snapshot(), folder, index, previous, receipt)
        self._emit(window)
        return window["node_id"]

'''
        b=b.replace('    def execute(self, *, caller_paused=False, retain_probe_for_handoff=False):',method+'    def execute(self, *, caller_paused=False, retain_probe_for_handoff=False):')
        b=b.replace('            while True:\n                result["observations"]', '            graph_previous = None\n            graph_pending_receipt = None\n            while True:\n                result["observations"]',1)
        needle='                self._emit({"event": "recovery_skill_decision", "observation": observation, "skill_outputs": outputs})'
        replacement='''                graph_previous = self._state_graph_window(probe, folder, len(result["observations"])-1,
                                                          graph_previous, graph_pending_receipt) or graph_previous
                graph_pending_receipt = None
'''+needle
        assert needle in b;b=b.replace(needle,replacement,1)
        b=b.replace('                    target, status, probe, row = self._recover_once(runner, target, action, row)',
                    '                    target, status, probe, row = self._recover_once(runner, target, action, row)\n                    graph_pending_receipt = copy.deepcopy(row.get("receipt"))',1)
    s=scheduler.read_text()
    if 'graph_recording_adapter = None' not in s:
        needle='    def emit(event):\n        from .session import plain\n'
        replacement='''    graph_recording_adapter = None
    graph_recording_state = {"window": None, "consumer": None}
    recording_settings = config.get("state_graph_recording", {})
    if recording_settings.get("enabled"):
        import importlib.util
        definition = importlib.util.spec_from_file_location("contact_graph_lifecycle", recording_settings["adapter_path"])
        graph_recording_adapter = importlib.util.module_from_spec(definition)
        definition.loader.exec_module(graph_recording_adapter)

    def emit(event):
        from .session import plain
'''
        assert needle in s;s=s.replace(needle,replacement,1)
        needle='        name = event.get("event")\n'
        replacement='''        if graph_recording_adapter is not None:
            if event.get("event") == "state_graph_window":
                graph_recording_state.update(window=event, consumer=event.get("consumer"))
            extra = graph_recording_adapter.lifecycle_event(event, graph_recording_state["window"], graph_recording_state["consumer"])
            if extra is not None:
                log.write(json.dumps(plain(extra), ensure_ascii=False, allow_nan=False) + "\\n")
'''+needle
        # Only the emit closure in build_runtime owns the log stream.
        start=s.index('    graph_recording_adapter = None');prefix=s[:start];tail=s[start:]
        assert needle in tail;s=prefix+tail.replace(needle,replacement,1)
    # Update an already-installed recording hook as well as a fresh install.
    b=b.replace('def _state_graph_window(self, probe, folder, index, previous, receipt):',
                'def _state_graph_window(self, probe, folder, index, previous, receipt, observation=None):')
    b=b.replace('folder, index, previous, receipt)\n        self._emit(window)',
                'folder, index, previous, receipt, observation=observation)\n        self._emit(window)')
    b=b.replace('graph_previous, graph_pending_receipt) or graph_previous',
                'graph_previous, graph_pending_receipt, observation=observation) or graph_previous')
    ast.parse(b);ast.parse(s);backend.write_text(b);scheduler.write_text(s)
    source=BUNDLE/'contact_scheduler.single_open_0330.json';cfg=json.loads(source.read_text())
    cfg['single_correction_open_handoff']=False
    cfg['state_graph_recording']={'enabled':True,'adapter_path':str(ADAPTER),'wrist_camera_key':'right_wrist','schema':'state_graph_recording_v3'}
    dest=BUNDLE/'contact_scheduler.state_graph_v3_0330.json';dest.write_text(json.dumps(cfg,ensure_ascii=False,indent=2)+'\n')
    script=BUNDLE/'run_state_graph_v3_0330.sh'
    script.write_text('''#!/usr/bin/env bash
set -euo pipefail
MODE="${1:-skill-check}"
case "$MODE" in skill-check|skill-run|task-check|task-run) ;; *) echo "Expected skill-check, skill-run, task-check or task-run" >&2; exit 2;; esac
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CONTACT_SCHEDULER_CONFIG="$DIR/contact_scheduler.state_graph_v3_0330.json"
export CONTACT_SKILL_AFTER=handoff CONTACT_SKILL_RECOVERY=enabled
export RIGHT_GRIPPER_CLOSE_LIMIT=0.330 RIGHT_FORCE_POSITION_MIN=0.330
export RIGHT_FORCE_APPROACH_TIMEOUT_S="${RIGHT_FORCE_APPROACH_TIMEOUT_S:-120}"
echo "[state-graph-v3] correction -> fresh probe -> closed-loop exit -> existing force controller"
exec bash "$DIR/run_contact_261010.sh" "$MODE"
''')
    script.chmod(0o755)
    manifest={'backend':str(backend),'scheduler':str(scheduler),'entry':str(script),'config':str(dest),'robot_started':False,'closed_loop':True,
              'recovery_model':'Existing V4 model for collecting new consumer evidence; v3 active consumer model is not yet trained.',
              'L1':'Logged from actual existing force-ready event','L2_L3':'Require actual outcome producer or reviewed evidence; never inferred from resumed policy.'}
    (ROOT/'outputs/state_graph/20261112_symbolic_v3/scheduler_integration.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(manifest,ensure_ascii=False))

if __name__=='__main__':install()
