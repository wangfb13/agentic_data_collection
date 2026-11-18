"""Requeue a known interrupted P3 attempt after operator-confirmed scene reset.

No hardware imports/connections. Original airborne and partial-result evidence
stays in the attempt archive; physical dispatch costs are never refunded.
"""
from pathlib import Path
import time
from bottle_scan.state_graph import campaign

KNOWN_ERRORS=(
    'Raw finger wrench time skew',
    'ForcePolicy reference lost during lift/return',
    'ForcePolicy reference lost during hold',
    'Recording message queue full; acquisition incomplete',
    'Recording byte queue full;',
)


def resume_interrupted(out, *, scene_reset_confirmed=False):
    if scene_reset_confirmed is not True:
        raise ValueError('先停臂、将瓶放稳并张开夹爪，再明确确认恢复现场')
    out=Path(out)
    with campaign.ownership(out):
        state=campaign.read(out/'state.json')
        pending=set((state.get('pending') or {}).get('trial_ids',[]))
        selected=[t for t in state['trials'] if t['trial_id'] in pending and t['status']=='started']
        validated=[]
        for trial in selected:
            directory=Path(trial['directory'])
            if (directory/'result.json').exists() or (directory/'completed_capture.json').exists():
                raise ValueError('已有结果，请按正常恢复流程结算，不能重做 '+trial['trial_id'])
            archive=campaign.retry_archive(out,trial)
            evidence=next((p/'exception.json' for p in (directory,archive) if (p/'exception.json').exists()),None)
            if evidence is None:raise ValueError('中断原因无记录，不能自动重试 '+trial['trial_id'])
            reason=campaign.read(evidence)['reason']
            if not any(reason.startswith(x) for x in KNOWN_ERRORS):
                raise ValueError('尚未归类的中断，需先核对证据：'+reason)
            validated.append((trial,directory,reason))
        for trial,directory,reason in validated:
            if directory.exists():
                campaign.atomic(directory/'operator_reset.json',dict(
                    scene_reset_confirmed=True,source='explicit_resume_interrupted_cli',
                    confirmed_unix_ns=time.time_ns(),robot_command_sent=False,
                    note='Operator confirms stopped arm, supported bottle and open unloaded gripper. Original evidence is retained.'))
            campaign.requeue_attempt(out,state,trial,reason,defer=False)
        return [t['trial_id'] for t,_,_ in validated]
