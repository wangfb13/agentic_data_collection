"""SQLite write-ahead intents. Unknown dispatch is charged and never reissued."""
from contextlib import contextmanager
from pathlib import Path
import json
import sqlite3
import time
import uuid

from .evidence import encode,sha,num,require,ROLES

def now():return time.time_ns()
def uid():return uuid.uuid4().hex

class Ledger:
    def __init__(self,path):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        self.db=sqlite3.connect(path,timeout=15,isolation_level=None)
        self.db.row_factory=sqlite3.Row
        self.db.executescript("""
        PRAGMA journal_mode=WAL; PRAGMA synchronous=FULL;
        CREATE TABLE IF NOT EXISTS rounds(id TEXT PRIMARY KEY, manifest TEXT, digest TEXT, stage TEXT, plan TEXT, parent TEXT);
        CREATE TABLE IF NOT EXISTS attempts(id TEXT PRIMARY KEY, rid TEXT, role TEXT, x REAL, phase TEXT, start INTEGER, end INTEGER, reason TEXT);
        CREATE UNIQUE INDEX IF NOT EXISTS one_pending ON attempts((1)) WHERE end IS NULL;
        CREATE TABLE IF NOT EXISTS events(aid TEXT,k TEXT,t INTEGER,data TEXT,PRIMARY KEY(aid,k));
        CREATE TABLE IF NOT EXISTS invalid_basis(id TEXT PRIMARY KEY, reason TEXT, t INTEGER);
        CREATE TABLE IF NOT EXISTS spent_inputs(token TEXT PRIMARY KEY, aid TEXT);
        """)
    @contextmanager
    def tx(self):
        self.db.execute('BEGIN IMMEDIATE')
        try:yield
        except BaseException:self.db.rollback();raise
        else:self.db.commit()
    def round(self,rid):
        row=self.db.execute('select * from rounds where id=?',(rid,)).fetchone()
        if row is None:raise ValueError('unknown frontier round')
        m=json.loads(row['manifest']);require(sha(m)==row['digest'],'round manifest changed')
        charged=self.db.execute("select count(distinct aid) from events where k like 'io/%'").fetchone()[0]
        return dict(id=rid,project_remaining=m['limits']['project_physical_attempts']-charged,manifest=m,digest=row['digest'],stage=row['stage'],plan=json.loads(row['plan']),parent=row['parent'])
    def new(self,m,parent=None):
        from .search import validate_manifest
        validate_manifest(m)
        with self.tx():
            prior=self.db.execute('select id from rounds limit 1').fetchone()
            if prior:
                first=self.round(prior[0])['manifest']
                require(m['environment']==first['environment'] and m['limits']['project_physical_attempts']==first['limits']['project_physical_attempts'],'ledger environment and total physical budget are immutable')
            if parent:
                old=self.round(parent)
                require(old['stage']=='closed' and m['model_id']!=old['manifest']['model_id'],'new version requires closed parent and changed model')
            rid='fr_'+uid()
            self.db.execute('insert into rounds values(?,?,?,?,?,?)',(rid,encode(m),sha(m),'explore','null',parent))
        return rid
    def pending(self):
        r=self.db.execute('select id from attempts where end is null').fetchone()
        return r[0] if r else None
    def attempts(self,rid):
        return [self.get(r[0]) for r in self.db.execute('select id from attempts where rid=? order by start',(rid,))]
    def get(self,aid):
        r=self.db.execute('select * from attempts where id=?',(aid,)).fetchone()
        if not r:raise ValueError('unknown attempt')
        events={x['k']:json.loads(x['data']) for x in self.db.execute('select k,data from events where aid=? order by rowid',(aid,))}
        return {**dict(r),'events':events}
    def begin(self,rid,role,x,*,op=None,proposal=None):
        from .search import offers
        with self.tx():
            rr=self.round(rid);require(not self.pending(),'pending attempt: reconcile before retry')
            candidates=offers(rr,self.attempts(rid))
            if proposal is not None:require(proposal in candidates,'stale choice revision')
            require(any(v['role']==role and v['x']==x and (op is None or v['op']==op) for v in candidates),'not an applicable frontier operation')
            aid=uid()
            self.db.execute('insert into attempts values(?,?,?,?,?,?,NULL,NULL)',(aid,rid,role,x,rr['stage'],now()))
        return aid
    def _event(self,aid,key,payload):
        encoded=encode(payload)
        row=self.db.execute('select data from events where aid=? and k=?',(aid,key)).fetchone()
        if row:
            require(encoded==row[0],'conflicting duplicate event');return False
        require(self.get(aid)['end'] is None,'attempt closed')
        self.db.execute('insert into events values(?,?,?,?)',(aid,key,now(),encoded));return True
    def event(self,aid,key,payload):
        with self.tx():return self._event(aid,key,payload)
    def dispatch(self,aid,key,request):
        # Commit *before* calling hardware. Existing intent is not permission to retry,
        # including after process death between commit and the driver's return.
        with self.tx():
            a=self.get(aid);r=self.round(a['rid']);all_a=self.attempts(a['rid'])
            require(a['end'] is None and not any(k.startswith('io/') and 'reply/'+k[3:] not in a['events'] for k in a['events']),'unresolved I/O: reconcile from device records')
            if not any(k.startswith('io/') for k in a['events']):
                require(r['project_remaining']>0 and costs(all_a)['physical_attempts_charged']<r['manifest']['limits']['physical_attempts'],'physical budget exhausted')
            require('io/'+key not in a['events'],'physical command already dispatched')
            self._event(aid,'io/'+key,request)
    def reply(self,aid,key,data):
        with self.tx():
            a=self.get(aid);require('io/'+key in a['events'],'reply without dispatch')
            # A real device recording must not be replayed as a second independent trial.
            require(data.get('attempt_id')==aid and data.get('command_id')==aid+'/'+key and data.get('raw_source'),'unbound device reply')
            import copy
            source_digest=sha(data)
            old=a['events'].get('reply/'+key)
            if old:
                require(old.get('source_digest')==source_digest,'conflicting duplicate reply');return False
            data=copy.deepcopy(data); data['source_digest']=source_digest
            require(type(data.get('physical_interaction')) is bool,'missing physical cost acknowledgement')
            m=self.round(a['rid'])['manifest']
            if m['environment']=='physical':
                from .proof import check_reply
                data['evidence_verified']=check_reply({k:v for k,v in data.items() if k!='source_digest'})
            sample=data if key=='prepare' else data.get('observation',{})
            label=sample.get('measurement') or {};basis=label.get('record',{}).get('basis_id')
            if label and m['environment']=='physical':
                from .proof import check_measurement
                stamp=sample.get('timestamp_ns')
                if (not stamp or not check_measurement(label,m,'preparation' if key=='prepare' else 'first_contact',stamp)
                    or (basis and self.db.execute('select 1 from invalid_basis where id=?',(basis,)).fetchone())):
                    sample['rejected_measurement']=label;sample['measurement']=None
            if data.get('object_moved') or sample.get('object_moved'):
                bases=[basis]+[(v.get('measurement') or {}).get('record',{}).get('basis_id') for v in a['events'].values()]
                for bid in filter(None,bases):self.db.execute('insert or ignore into invalid_basis values(?,?,?)',(bid,aid,now()))
            activities=data.get('activity_counts',{})
            require(isinstance(activities,dict) and all(k in ('standard_measurements','preparation_moves','resets','touches','actions','expert_interventions') and type(v) is int and v>=0 for k,v in activities.items()),'invalid activity costs')
            if key=='prepare' and data.get('independent_trial_unit'):
                unit='unit:'+str(data['independent_trial_unit'])
                who=self.db.execute('select aid from spent_inputs where token=?',(unit,)).fetchone()
                if who and who[0]!=aid:data['independent_trial_unit']=None
                else:self.db.execute('insert or ignore into spent_inputs values(?,?)',(unit,aid))
            token=str(data['raw_source'])
            owner=self.db.execute('select aid from spent_inputs where token=?',(token,)).fetchone()
            require(not owner or owner[0]==aid,'same device evidence used by another attempt')
            self.db.execute('insert or ignore into spent_inputs values(?,?)',(token,aid))
            return self._event(aid,'reply/'+key,data)
    def finish(self,aid,reason):
        require(isinstance(reason,str) and reason,'termination reason required')
        with self.tx():
            a=self.get(aid)
            if a['end'] is not None:
                require(reason==a['reason'],'conflicting terminal result');return
            self.db.execute('update attempts set end=?,reason=? where id=?',(now(),reason,aid))
    def confirm(self,rid,points):
        from .search import grid
        with self.tx():
            r=self.round(rid);m=r['manifest'];g=grid(m)
            require(r['stage']=='explore' and not self.pending(),'freeze once, with no pending attempt')
            require(points and len(set(points))==len(points) and all(x in g for x in points),'unvalidated confirmation points')
            sampled={(a['role'],a['x']) for a in self.attempts(rid) if a['end'] and a['phase']=='explore'}
            require(all(('policy_frontier',x) in sampled for x in points),'confirm only explored policy targets')
            p={'points':sorted(points),'roles':['direct_baseline','policy_frontier'],'n':m['statistics']['confirmation_n'],
               'family_size':2*len(points),'frozen_ns':now(),'method':'two-sided exact CP, Bonferroni across role x position'}
            self.db.execute("update rounds set stage='confirm',plan=? where id=?",(encode(p),rid))
    def close(self,rid):
        with self.tx():
            self.round(rid);require(not self.pending(),'pending attempt')
            self.db.execute("update rounds set stage='closed' where id=?",(rid,))

def costs(attempts):
    ev=[(a,k,v) for a in attempts for k,v in a['events'].items() if k.startswith('io/')]
    known=[(a,k,v) for a,k,v in ev if 'reply/'+k[3:] in a['events']]
    from collections import Counter
    actual=Counter()
    for a,k,_ in known:actual.update(a['events']['reply/'+k[3:]].get('activity_counts',{}))
    return dict(acknowledged_activity_counts=dict(actual),physical_cost_detail_unreported=sum('activity_counts' not in a['events']['reply/'+k[3:]] for a,k,_ in known),planned_cells=len({(a['role'],a['phase'],a['x']) for a in attempts}),attempts_started=len(attempts),
        physical_attempts_charged=len({a['id'] for a,_,_ in ev}),physical_attempts_acknowledged=len({a['id'] for a,k,_ in known
        if a['events']['reply/'+k[3:]].get('physical_interaction') is True}),
        unresolved_dispatches=len(ev)-len(known),dispatches=len(ev),preparations=sum(k=='io/prepare' for _,k,_ in ev),
        touches=sum(v['operation'] in ('touch','correct_retouch') for _,_,v in ev),actions=sum(v['operation']=='correct_retouch' for _,_,v in ev),
        resets=sum(v['operation']=='reset' for _,_,v in ev),expert_actions=sum(v['operation']=='correct_retouch' and a['role']=='expert_recovery' for a,_,v in ev),
        measurements=sum(bool(a['events'].get('reply/'+k[3:],{}).get('measurement')) for a,k,_ in ev),
        faults=sum((a['reason'] or '').startswith(('technical','interrupted')) for a in attempts),
        wall_seconds=sum((a['end']-a['start'])/1e9 for a in attempts if a['end'] is not None))

