"""Offline integration coverage of the P3 loop. No transport, robot or external agent."""
from contextlib import contextmanager
from pathlib import Path
import pytest

import joblib
import numpy as np

from bottle_scan.state_graph import campaign
from skills.contact_position_a import p3_loop as loop, p3_trial as trial
from skills.contact_position_a.ah_geometry import KIND


class Plant:
    def __init__(self,req,spec,tag,fail_lift=False):
        self.req=req
        self.spec=spec
        self.tag=tag
        self.fail_lift=fail_lift
        self.action=None

    def confirm(self,req):
        pass

    @contextmanager
    def contact(self,index):
        # Distinct tactile signatures for a/height corrections and direct handoff.
        x={'ready':(40.,40.),'a':(15.,40.),'h':(40.,15.)}[self.tag]
        gid=self.req['trial_id']
        yield dict(node_id=f'{gid}:{index}',session='P3',group_id=gid,trial_id=gid,
                   phase='PROBE',t_ns=100+index,window_index=index,preconditions={'bilateral_contact':True},
                   contacts=[dict(u_px=x[0],v_px=x[1],area_px=100.) for _ in range(2)],
                   wrench=[x[0]/40,x[1]/40,1.,0.,0.,0.]*2,
                   attrs={'mass_class':self.req['block']['mass_class']},plan_coords={},repeat_settings={},
                   provenance={'simulated':True,'repeat_context':['fixture_repeat',self.tag]})

    def move(self,action,first):
        self.action=action
        if self.tag=='a' and action['u']==[5.,0.] or self.tag=='h' and action['u']==[0.,8.]:
            self.tag='ready'
        return 'synthetic_move_receipt'

    def link(self,first,last,receipt,u):
        last.update(previous_node_id=first['node_id'],u_executed=u,
            u_receipt=dict(kind=KIND,applied=True,net_contact_displacement_mm=u,
                           commanded_axis='a' if u[0] else 'h',simulated=True))

    def consume(self):
        return dict(controller='existing_forcepolicy',version='simulated_only',params={'closure_floor':.33},
                    L1=True,L2=self.tag=='ready' and not self.fail_lift,evidence=['fake_lift_hold'],reason='fake_lift_hold')


class Director:
    def __init__(self,failed_stage):
        self.failed_stage=failed_stage
        self.confirmed=0
        self.ops=[]
        self.dev_episodes=0
        self.axes=[]
        self.rejected_candidate=None

    def symbolic_direct(self,ctx):
        choices=ctx['applicable_operators']
        ops={o['op'] for o in choices}
        plan=None
        if 'complete' in ops:
            op='complete'
        elif 'freeze' in ops:
            op='freeze'
        elif 'validate_B' in ops:
            op='validate_B'
            plan=ctx.get('independent_plan') or self.plan([None,None])
        elif 'collect_outcomes' in ops and self.ops.count('collect_outcomes')<3:
            op='collect_outcomes'
            plan=self.plan([None,None])
        elif 'collect_transitions' in ops:
            op='collect_transitions'
            plan=self.plan([[5.,0.],[0.,8.]])
        else:
            raise AssertionError(ops)
        self.ops.append(op)
        choice=next(c for c in choices if c['op']==op and 'axis' not in c['bindings'])
        return dict(operator=op,args=choice['bindings'],premises=choice['premises_satisfied_by'],
                    reason='离线夹持模拟',expected='测量交接结果',plan=plan,hypotheses_update=[],discriminates=[])

    @staticmethod
    def plan(actions):
        return {'cluster_classes':['Ready','NeedAction'],'cells':[
            dict(settings={},blocks=[{'mass_class':'light'},{'mass_class':'heavy'}],reps=1,
                 actions=[dict(kind=KIND,u=u)] if u else []) for u in actions]}

    def execute(self,req,directory,spec,cfg,**kw):
        n=int(req['trial_id'].split('_')[-1])-1
        if req['role']=='exploration':
            if req['actions']:
                tag='a' if req['actions'][0]['u'][0] else 'h'
            else:
                tag=('ready','a','h')[self.ops.count('collect_outcomes')-1]
        else:
            tag=('ready','a','ready','h')[n%4]
        if req['role']=='development':
            self.dev_episodes+=1
            if self.rejected_candidate is None:
                self.rejected_candidate=req['model_id']
        # A deliberately bad first development consumer, independently observable.
        if req['role']=='independent':
            self.confirmed+=1
        fail=(self.dev_episodes==1 if self.failed_stage=='development' else self.confirmed==1)
        plant=Plant(req,spec,tag,req['role']==self.failed_stage and fail)
        result=trial.run_trial(req,directory,spec,cfg,plant)
        if req['role']!='exploration' and plant.action is not None:
            self.axes.append(plant.action['u'])
        return result


@pytest.mark.parametrize('failed_stage',['development','independent'])
def test_full_p3_fake_loop(tmp_path,failed_stage):
    from skills.contact_position_a import campaign_ah, ah_current
    src=tmp_path/'source'
    center=np.eye(4)
    center[:3,:3]=[[0.,0.,1.],[0.,1.,0.],[-1.,0.,0.]]
    spec,cfg=campaign_ah.configuration(src,center=center)
    spec.pop('placement_geometry')
    spec['current_placement']='P3'
    for name in ('a_offset_mm','h_offset_mm','l_offset_mm','approach_pose'):
        spec['perturbation_variables'][name].pop('reference',None)
    cfg['execution'].update(placement='P3',executable_variables=['mass_class'],fixed_settings={'closure_floor':.33})
    ah_current.set_trial_protocol(spec,cfg['execution'])
    campaign.atomic(src/'instance.json',spec)
    cfg.update(spec=str(src/'instance.json'),sources=[])
    campaign.atomic(src/'input.json',cfg)
    campaign.initialize(src/'input.json',src)
    out=tmp_path/'simulation'
    loop.initialize(src,out,extra_trials=80,consumer={'version':'simulated_only'})
    agent=Director(failed_stage)
    state=campaign.run(out,provider=agent,executor=agent.execute)
    assert state['complete'], (agent.ops,state['phase'],out)
    assert state['robot_started'] is False
    assert agent.ops.count('collect_transitions')>=2 and 'validate_B' in agent.ops
    first_b=agent.ops.index('validate_B')
    assert 'collect_transitions' in agent.ops[first_b:] and agent.ops.index('freeze')>first_b
    if failed_stage=='independent':
        retired=state['retired_candidates']
        assert len(retired)==1 and state['frozen']['model_id'] not in retired
        assert not campaign.read(out/'confirmations'/retired[0]/'acceptance.json')['passed']
    assert set(map(tuple,agent.axes))=={(5.,0.),(0.,8.)}
    independent=campaign.read(out/'independent_acceptance.json')
    assert independent['passed'] and independent['metrics']['trials']==12
    bundle=joblib.load(state['frozen']['bundle'])
    sealed={n['node_id'] for t in state['trials'] if t['request']['role']=='independent'
            for n in campaign.read(Path(t['directory'])/'result.json')['nodes']}
    assert sealed.isdisjoint(bundle['graph']['node_ids'])
    print('SIMULATED',state['spent'],agent.ops,independent['metrics'])

