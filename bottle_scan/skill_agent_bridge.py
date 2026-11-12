"""Read-only bridge into the existing LLM RSI proposer. No provider call here."""
import json
from pathlib import Path
from .agentic_learning import _hash
from .skill_offline import evaluate_skill_sessions
from .skill_exploration import diagnose_feedback

def skill_planning_reference(path):
    report=json.loads(Path(path).read_text())
    if report.get('role') != 'development':
        raise ValueError('only development skill evidence may guide the RSI Agent')
    roots=[entry['path'] for entry in report.get('source_sessions',[])]
    checked=evaluate_skill_sessions(roots,role='development')
    if _hash(report)!=_hash(checked):
        raise ValueError('skill feedback or raw evidence changed; regenerate a versioned report')
    return {'report_path':str(Path(path).resolve()),'report_sha256':_hash(checked),
            'role':'development','invocation':checked['invocation'],
            'recovery':checked['recovery'],'exit':checked['exit'],
            'applicability':checked['applicability'],'coverage':checked['coverage'],
            'findings':diagnose_feedback(checked),'evidence_refs':checked['evidence_refs'],
            'constraint':'Hypotheses may guide the next method; never claim new validated scope from missing evidence.'}
