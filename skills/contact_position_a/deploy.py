"""Export and optionally run a graph model through V4's natural main-policy entry."""
import argparse
from pathlib import Path
import json
import os
import subprocess
from bottle_scan.state_graph.campaign import atomic,read
from bottle_scan.state_graph.skill_interface import load_spec
from .campaign import launch_request


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--campaign',required=True);p.add_argument('--candidate',action='store_true')
    p.add_argument('--entry',choices=['task','skill'],default='task');p.add_argument('--offset-mm',type=float,default=0.)
    p.add_argument('--real',action='store_true');p.add_argument('--out')
    a=p.parse_args(argv);campaign=Path(a.campaign).resolve();state=read(campaign/'state.json');cfg=read(campaign/'config.json');spec=load_spec(campaign/'spec.json')
    model=state.get('model') if a.candidate else state.get('frozen')
    if not model or not model.get('trained'):raise ValueError('No trained selected model. Complete collection first; --candidate selects development only.')
    out=Path(a.out).resolve() if a.out else campaign/'deployment';out.mkdir(parents=True,exist_ok=True)
    from .ready import FIXED_GOALS
    if a.entry=='task' and spec['active_post_condition'] in FIXED_GOALS:
        raise ValueError('Natural main-policy handoff requires a ForcePolicy post-condition; fixed squeeze uses skill entry')
    if a.entry=='task' and a.offset_mm:raise ValueError('Natural main-policy entry does not inject offsets')
    request={'schema':'graph_trial_request_v4','trial_id':'deployment','campaign':str(campaign),'spec':str(campaign/'spec.json'),
             'role':'deployment','post_condition':spec['active_post_condition'],'mode':'learned','model_id':model['model_id'],'model_bundle':model['bundle'],
             'settings':{} if a.entry=='task' else {'a_offset_mm':a.offset_mm},
             'block':{},'actions':[],'post_freeze':not a.candidate,'frozen_model_id':None if a.candidate else model['model_id']}
    atomic(out/'request.json',request);command,env=launch_request(request,out,spec,cfg)
    if a.entry=='task':env.pop('CONTACT_EXPERIMENT_B_OFFSET_MM',None)
    command[-1]=a.entry+'-run'
    atomic(out/'deployment.json',{'command':command,'config':env['CONTACT_SCHEDULER_CONFIG'],'model_id':model['model_id'],
                                  'candidate':a.candidate,'robot_started':a.real,'fresh_observation_before_exit':True})
    print(json.dumps({'model_id':model['model_id'],'request':str(out/'request.json'),'command':command,'robot_started':a.real},ensure_ascii=False))
    if not a.real:return 0
    print('确认瓶子放稳、夹爪全开空载、机器人已停止且无人遥操。')
    if input('输入 MOVE 运行：').strip()!='MOVE':return 1
    env['CONTACT_TABLETOP_CONFIRMED']='1'
    return subprocess.run(command,env=env,check=False).returncode
if __name__=='__main__':raise SystemExit(main())
