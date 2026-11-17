"""Prospectively record an externally operated named expert. Never connects to hardware.
The other terminal/recorder must supply actual device evidence for each new attempt.
"""
import json
import os
from pathlib import Path
import time
from bottle_scan.max_expert_demonstration_finder.evidence import require,encode
from .frontier import verify_files
from .local_motion import command,support_error

class WitnessRecorder:
    environment='physical'
    def __init__(self,m):
        self.m=m;self.expert_identity=m['expert'];self.base=None;self.folder=None;self.deadline=None
    def verify(self,m):
        verify_files(m)
        require(m.get('expert') and isinstance(m['station'],dict) and m.get('expert_exchange'),'named expert, exchange directory and surveyed station required')
        self.base=Path(m['expert_exchange'])
        att=json.loads((self.base/'station_identity.json').read_text())
        require(att=={'survey_id':m['station']['survey_id'],'reference_digest':m['reference_digest'],'sensor_version':m['sensor_version'],
                 'tool_version':m['tool_version'],'consumer':m['post_condition']['consumer'],'condition':m['condition'],'expert':m['expert']},
                'external recorder identity mismatch')
    def wait_json(self,path,timeout):
        end=time.monotonic()+timeout
        while time.monotonic()<end:
            if path.is_file():return json.loads(path.read_text())
            time.sleep(.1)
        raise TimeoutError('No durable recorder reply; inspect attempt, never repeat automatically')
    def perform(self,req):
        self.folder=self.base/req['attempt_id'];self.folder.mkdir(parents=True,exist_ok=True)
        key=req['command_id'].split('/',1)[1];ticket=self.folder/(key+'.request.json')
        # Tickets are recording authorizations, not an actuator interface.
        with ticket.open('x') as f:
            f.write(encode({'request':req,'expert':self.expert_identity,'limits':self.m['limits'],'manifest_reference':self.m['reference_digest']}));f.flush();os.fsync(f.fileno())
        print('待采真实记录：',ticket,flush=True)
        if key=='touch':self.deadline=time.monotonic()+self.m['limits']['seconds']
        t=req.get('remaining_seconds',req.get('deadline_seconds',self.m['limits']['seconds']))
        envelope=self.wait_json(self.folder/(key+'.reply.json'),t)
        response=envelope['reply']
        # Raw source is this immutable reply sidecar plus the acquisition files it names.
        return dict(response,raw_source=str((self.folder/(key+'.reply.json')).resolve()))
    def expert_decision(self,obs,step,expert):
        p=self.folder/(f'decision_{step}.json');print('等待指定专家决定：',p,flush=True)
        d=self.wait_json(p,max(0,self.deadline-time.monotonic()))
        require(d['expert_id']==expert['id'] and d['expert_version']==expert['version'] and d['observation_id']==obs['id']
                and d['attempt_id']==self.folder.name,'unbound expert decision')
        if d['state']!='NeedAction':
            require(d['state'] in ('Unknown','Ready'),'invalid expert state')
            return dict(state=d['state'],reason=d.get('reason'),raw_source=str(p))
        import numpy as np
        T=np.array(obs['tcp_mm'],float);delta=d['delta_a_mm'];direction=T[:3,:3]@self.m['axis_tcp']
        for u in (0,-self.m['reject_rules']['servo']['retreat_mm'],delta-self.m['reject_rules']['servo']['retreat_mm'],delta):
            target=T.copy();target[:3,3]+=u*direction
            require(support_error(target,self.m['station']) is None,'expert translation path outside surveyed station')
        return dict(state='NeedAction',command=command(delta,dict(observation_id=obs['id'],timestamp_ns=obs['timestamp_ns']),
                    self.m['target_definition_id'],'expert'),raw_source=str(p))

