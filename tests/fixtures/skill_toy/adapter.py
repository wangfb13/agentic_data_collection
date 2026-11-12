"""Synthetic two-dimensional skill with two distinct post conditions."""
import numpy as np


def iter_trials(sources,spec):
    rng=np.random.default_rng(42);trial=0
    for kind in ('A','B'):
        for x in (-1.,0.,1.):
            for y in (-1.,0.,1.):
                for block in ('soft','hard'):
                    for repeat in range(4):
                        trial+=1;group=f'toy-{trial}';session='toy_context_'+str(repeat%2)
                        initial=np.array([x,y])+rng.normal(0,.018,2);windows=[]
                        trajectory=[initial] if kind=='A' else [initial,initial*.25+rng.normal(0,.008,2)]
                        for index,point in enumerate(trajectory):
                            window={'node_id':group+'-'+str(index),'session':session,'trial_id':str(trial),'group_id':group,'window_index':index,
                              'phase':'A' if kind=='A' else ('B_pre' if index==0 else 'B_post'),'t_ns':trial*100+index,
                              'preconditions':{'observable':True},'is_terminal':index==len(trajectory)-1,'contacts':point.tolist(),
                              'load':(point*np.array([.3,.4])).tolist(),'image':[float(point.sum()),float(point[0]-point[1])],
                              'attrs':{'payload_class':block,'context':'context_'+str(repeat%2),'controller_setting':'standard'},
                              'repeat_settings':{'x':x,'y':y,'context':'context_'+str(repeat%2)},
                              'plan_coords':{'x':float(point[0]),'y':float(point[1])},'is_reference':x==y==0,
                              'previous_node_id':group+'-0' if index else None,'u_executed':(-.75*initial).tolist() if index else None,
                              'u_receipt':{'measured':True,'source':'synthetic_ground_truth'} if index else None,
                              'risk':{},'rule_id':'toy_initial','point':point.tolist()}
                            windows.append(window)
                        yield {'session':session,'trial_id':str(trial),'windows':windows}


def windows(trial,spec):return trial['windows']
def geometry(window,spec):return window['contacts']
def load(window,spec):return window['load']
def image(window,spec):return window['image']
def action(trial,window,spec):return {'u':window['u_executed'],'receipt':window['u_receipt']}
def post(trial,window,condition):
    if not window['is_terminal']:return {'evaluated':False,'success':None,'level':None,'evidence':[]}
    x,y=window['point'];success=x*x+y*y<.45**2 if condition['id']=='centered' else x+y<.5
    return {'evaluated':True,'success':bool(success),'level':'synthetic','evidence':[{'point':window['point']}]}
def executor(plan,spec):return {'executed':False,'settings':plan}
