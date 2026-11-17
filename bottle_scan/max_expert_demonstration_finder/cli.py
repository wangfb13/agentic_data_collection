"""Persistent, independently frozen a-axis frontier experiments."""
import argparse
import json
from pathlib import Path
from .ledger import Ledger,costs
from .search import report,offers,validate_choice
from .engine import execute,reconcile
from .evidence import require,sha
from .compare import compare
from bottle_scan.state_graph.operators import frontier_context
from bottle_scan.state_graph.common import write
from skills.contact_position_a.frontier import manifest,BoundExperiment

def read(f):return json.loads(Path(f).read_text())
def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--db',default='outputs/frontier_a/ledger.sqlite')
    p.add_argument('--config',default='skills/contact_position_a/frontier_config.json')
    sub=p.add_subparsers(dest='cmd',required=True)
    sub.add_parser('init')
    sub.add_parser('audit').add_argument('--out',required=True)
    sub.add_parser('demo').add_argument('--out',required=True)
    iv=sub.add_parser('invalidate-basis');iv.add_argument('basis_id');iv.add_argument('--reason',required=True)
    cp=sub.add_parser('compare');cp.add_argument('older');cp.add_argument('newer')
    for cmd in ('next','status','report','run-one','confirm','close','reconcile','retrain'):
        q=sub.add_parser(cmd);q.add_argument('round')
        if cmd=='next':q.add_argument('--search-without-expert',action='store_true');q.add_argument('--agent',action='store_true');q.add_argument('--out')
        if cmd=='report':q.add_argument('--out',required=True)
        if cmd=='run-one':q.add_argument('--choice',required=True);q.add_argument('--real',action='store_true',required=True)
        if cmd=='confirm':q.add_argument('--points',required=True,type=float,nargs='+')
        if cmd=='reconcile':q.add_argument('attempt');q.add_argument('--station-replies');q.add_argument('--reason',required=True)
        if cmd=='retrain':
            q.add_argument('--out',required=True);q.add_argument('--seed-campaign',default='outputs/state_graph/campaign_white_a_v4_forcepolicy')
    a=p.parse_args(argv);cfg=read(a.config)
    if a.cmd=='demo':
        from .demo import run_demo
        result=run_demo(a.out);print(json.dumps({'out':a.out,'environment':'simulation','comparison':result['comparison']}));return
    if a.cmd=='audit':
        from .audit import audit
        result=audit(manifest(cfg),Path.cwd());write(a.out,result);print(json.dumps(result,ensure_ascii=False));return
    db=Ledger(a.db)
    if a.cmd=='invalidate-basis':
        from .ledger import now
        require(len(a.basis_id)==64 and a.reason.strip(),'basis digest and reason required')
        with db.tx():db.db.execute('insert or ignore into invalid_basis values(?,?,?)',(a.basis_id,a.reason,now()))
        return
    if a.cmd=='init':print(db.new(manifest(cfg)));return
    if a.cmd=='compare':print(json.dumps(compare(db,a.older,a.newer)));return
    r=db.round(a.round);m=r['manifest'];at=db.attempts(a.round)
    if a.cmd in ('confirm','retrain'):
        key={'confirm':'validate_recovery_frontier','retrain':'retrain_from_frontier'}[a.cmd]
        require(any(v['op']==key for v in frontier_context(db,a.round)['lifecycle_choices']),'no evidence for lifecycle operation: '+key)
    if a.cmd=='confirm':db.confirm(a.round,a.points)
    elif a.cmd=='close':db.close(a.round)
    elif a.cmd=='reconcile':
        require(db.get(a.attempt)['rid']==a.round,'wrong round')
        reconcile(db,a.attempt,read(a.station_replies) if a.station_replies else [],a.reason)
    elif a.cmd=='retrain':
        from .train import retrain
        print(json.dumps(retrain(db,a.round,a.out,cfg,a.seed_campaign)));return
    elif a.cmd=='next':
        ctx=frontier_context(db,a.round);choices=ctx['collection_choices']
        need_provider=bool(r['stage']=='explore' and not m['expert'] and any(f['pred']=='recovery_seed_missing' for f in ctx['facts']))
        if need_provider and not a.search_without_expert:choices=[]
        choice=choices[0] if choices else None
        if a.agent and choices:
            require(r['stage']=='explore','confirmation uses a fixed schedule')
            from bottle_scan.agentic_provider import AgentProvider
            agent=AgentProvider(read('config.state_graph.campaign_v4.json')['provider_settings'])
            try:
                ctx={'facts':report(r,at)['facts'],'budget':m['limits'],'applicable_choices':choices,'instruction':'Choose one exact choice by its zero-based index; return JSON with index and reason. No actions or labels.'}
                idx=agent.frontier_select(ctx)['index']
                require(type(idx) is int and 0<=idx<len(choices),'invalid frontier selection');choice=validate_choice(choices[idx],r,at)
            finally:agent.close()
        if a.out and choice:write(a.out,choice)
        print(json.dumps({'choice':choice,'choices':choices,'pending':db.pending(),'expert_provider':m['expert'],
              'expert_requests':ctx.get('expert_requests',[]) if 'expert_requests' in ctx else frontier_context(db,a.round)['expert_requests'],'expert_request_pending':need_provider},ensure_ascii=False));return
    elif a.cmd=='run-one':
        choice=validate_choice(read(a.choice),r,at);require(m['environment']=='physical','not a physical round')
        if input('本次工位与瓶子已核对，准备及复位将计入账本。输入 MOVE：').strip()!='MOVE':return
        port=BoundExperiment(m,choice['role'])
        try:print(execute(db,a.round,choice,port,port))
        finally:port.close()
        return
    result=report(db.round(a.round),db.attempts(a.round))
    allrows=[db.get(x[0]) for x in db.db.execute('select id from attempts')]
    result['ledger_lifetime_cost']=costs(allrows)
    if a.cmd=='report':write(a.out,result)
    print(json.dumps({k:result[k] for k in ('round_id','phase','model_id','facts','cost','ledger_lifetime_cost')},ensure_ascii=False))

if __name__=='__main__':main()

