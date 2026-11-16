"""Replay recorded PROBE windows without the robot; descriptive, in-sample diagnostic."""
import argparse
from collections import Counter
import json
from pathlib import Path
from types import SimpleNamespace

import joblib
from bottle_scan.contact_policy import ContactPolicy
from bottle_scan.state_graph.campaign import atomic,read
from bottle_scan.state_graph.skill import SymbolicRecoverySkill
from .local_decision import decide_local


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--assets',type=Path,default=Path('outputs/local_contact_a_v1'))
    p.add_argument('--campaign',type=Path,default=Path('outputs/state_graph/campaign_white_a_v4_forcepolicy'))
    args=p.parse_args(argv)
    graph=SymbolicRecoverySkill(joblib.load(args.assets/'graph/models/bundle.joblib'))
    cfg=read('config.state_graph.v4_scheduler.json')['execution_config']
    ref=read('skills/contact_position_a/white_reference.json')
    backend=SimpleNamespace(cfg=cfg,contact_reference=ref,offset_policy=ContactPolicy(read(args.assets/'offset.json'),cfg['servo']),
        local_settings={'gain':1.,'max_steps':cfg['servo']['max_regrasps'],'offset_deadband_mm':cfg['servo']['tol_mm'],'observation_max_age_s':.5})
    sources=read(args.campaign/'config.json')['sources']
    windows={}
    for path in sources:
        for line in Path(path).read_text().splitlines():
            e=json.loads(line)
            if e['event']=='state_graph_window':windows[e['node_id']]=e
    counters={};logs=[]
    for source in ('offset_servo','graph_policy'):
        backend.action_source=source;counter=Counter()
        for row in graph.bundle['support']:
            e=windows[row['node_id']];probe=e['probe'];ok=e['preconditions']['bilateral_contact'] is True
            o={'bilateral':ok,'force_stable':ok,'safety_ok':True,'timestamp_ns':probe['timestamp_ns'],
               'observation_id':row['node_id'],'descriptors':probe['descriptors']}
            backend.hw=SimpleNamespace(now=lambda:o['timestamp_ns'])
            state,reason,command,pred=decide_local(backend,o,row,graph,e['window_index'])
            logs.append({'observation_id':row['node_id'],'action_source':source,'state':state,'reason':reason,
                         'offset_estimate':pred['offset_estimate'],'action':command,'offset_ground_truth_mm':None})
            counter[state]+=1
        counters[source]=dict(counter)
    report={'counts':counters,'in_sample':True,'robot_started':False,
            'offset_mae_for_these_windows':None,'windows':logs}
    atomic(args.assets/'recorded_window_replay.json',report)
    print(json.dumps(counters))


if __name__=='__main__':main()
