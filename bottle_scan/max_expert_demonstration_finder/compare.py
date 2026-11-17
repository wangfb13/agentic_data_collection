"""Compare complete fixed-sample confirmation cells, not the extrema of the training data."""
from .search import report,interval
from .evidence import require

def compare(db,old,new):
    a,b=db.round(old),db.round(new);require(old!=new and a['manifest']['model_id']!=b['manifest']['model_id'],'need different frozen versions')
    for k in ('environment','reference_digest','condition','protocol','axis_tcp','post_condition','limits','statistics','station','tool_version','sensor_version','expert','action_source'):
        require(a['manifest'][k]==b['manifest'][k],'incomparable '+k)
    ra,rb=[report(r,db.attempts(r['id'])) for r in (a,b)]
    result={'older':old,'newer':new,'confirmed_improvement_cells':[],'unresolved':True,'verified_connected_expansion':None,
            'scope':'same measured grid, independent frozen plans; no claim between sampled positions'}
    if not (a['plan'] and b['plan']):return result
    if a['plan']['points']!=b['plan']['points'] or a['plan']['n']!=b['plan']['n']:return result
    st=a['manifest']['statistics'];N=a['plan']['n'];family=a['plan']['family_size']+b['plan']['family_size']
    alpha=(1-st['confidence'])/family;maps=[];lower_old=set()
    for j,r in enumerate((ra,rb)):
        supported=set()
        for c in r['policy_recovery_frontier']['cells']:
            cc=c['confirmation']
            if cc['interval'] is None:continue
            l,h=interval(cc['successes'],N,alpha)
            if l>=st['target']:supported.add(c['x'])
            if j==0 and h<st['target']:lower_old.add(c['x'])
        maps.append(supported)
    improvements=sorted(maps[1]&lower_old);result['confirmed_improvement_cells']=improvements
    res=a['manifest']['limits']['resolution'];ext={}
    for sign,side in ((-1,'negative'),(1,'positive')):
        chains=[]
        for group in maps:
            x=0;chain=[]
            while x in group:
                chain.append(x);x+=res*sign
            chains.append(chain)
        ext[side]=dict(old=chains[0],new=chains[1],verified_new_cells=[x for x in chains[1] if x in improvements])
    result['verified_connected_expansion']=ext;result['unresolved']=not bool(improvements)
    return result

