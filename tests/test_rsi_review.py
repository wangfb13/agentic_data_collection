"""Offline adversarial regression checks from independent RSI workflow review."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from bottle_scan.rsi import RSICampaign
from bottle_scan.rsi_methods import MethodStore, baseline_method, validate_method
from bottle_scan.session import fingerprint


class RSIReviewTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name)
        campaign=object.__new__(RSICampaign)
        campaign.root=self.root
        campaign.path=self.root/'rsi.json'
        campaign.say=lambda *_:None
        campaign.ask=lambda *_:''
        campaign.first_motion=True
        campaign.started=None
        campaign.check_cycle_origin=lambda *_:None
        campaign.executor=None
        campaign.proposer=None
        campaign.settings={'simulate':True,'scope':{'min_mm':-10,'max_mm':10,'step_mm':5},
            'seed':1,'memory':{'mode':'recursive','retrieval_limit':5},
            'comparison':{'collection_trials_per_branch':12,'offsets_mm':[0.], 'reps':4,
                          'min_trials':20,'min_effect':.05,'batch_scan_trials':2},
            'contact_policy':{'epochs':100,'seed':1}}
        campaign.cfg={'design':{'bottles':['light','heavy'],'zero_every':10}}
        campaign.store=MethodStore(self.root/'memory')
        old_id=campaign.store.register(baseline_method())
        new=baseline_method(); new.pop('method_id'); new['name']='new'; new['parent_id']=old_id
        new_id=campaign.store.register(validate_method(new))
        protocol={'task_id':'t1','initial_dataset_hash':'same_data','initial_policy_hash':'initial_policy',
            'training_budget':{'epochs':100,'seed':1},'collection_budget':12,'simulated':True,
            'evaluation_cells':[{'offset_mm':0.,'bottle':b,'trials':2} for b in ('light','heavy')],
            'preregistered':True,'evaluation_independent':True,'min_trials':20,'min_effect':.05}
        cycle={'id':1,'phase':'collect','base_sessions':['base'],'base_policy_id':'base_policy',
               'protocol':protocol,'protocol_sha256':fingerprint(protocol),'decision':{'reason':'test'},
               'branches':{key:{'method_id':value,'model_id':key+'_policy','collect':[],
                                'evaluation':[],'scan_count':12,'training_manifest':{}}
                           for key,value in [('old',old_id),('new',new_id)]},
               'schedule':[],'evaluation_schedule':[]}
        campaign.state={'task_id':'t1','status':'running','phase':'comparison','hardware_signature':'hw',
                        'base_sessions':['base'],'current_policy_id':'base_policy',
                        'incumbent_method_id':old_id,'cycles':[cycle],'batches':[],
                        'context':{},'models':{}}
        self.campaign,self.cycle=campaign,cycle

    def tearDown(self): self.temp.cleanup()

    def test_partial_evaluation_schedule_resume_is_idempotent(self):
        """A stop between branch slots must not add a second evaluation plan."""
        campaign=self.campaign
        calls=[0]
        def fake_batch(plan,**kwargs):
            calls[0]+=1
            if calls[0]==2:
                raise KeyboardInterrupt('operator stopped during preparation')
            row={'id':str(len(campaign.state['batches'])+1),'plan':copy.deepcopy(plan),
                 'path':'unused','status':'planned',**kwargs}
            campaign.state['batches'].append(row)
            return row
        campaign.batch=fake_batch
        try:
            campaign.prepare_evaluation(self.cycle)
        except KeyboardInterrupt:
            pass
        # Match run()'s exception/finally checkpoint and process restart.
        campaign.save()
        campaign.state=json.loads(campaign.path.read_text())
        self.cycle=campaign.state['cycles'][0]
        campaign.prepare_evaluation(self.cycle)
        schedule=self.cycle['evaluation_schedule']
        self.assertEqual(len(schedule),4,'resume duplicated fixed evaluation slots')
        self.assertEqual(len(set(schedule)),4)
        self.assertEqual(sum(len(b['plan']) for b in campaign.state['batches'] if b['id'] in schedule),8)

    def test_invalid_comparison_must_not_commit_unverified_new_baseline(self):
        """Incomplete/equal-budget-invalid experiments must pause, not become history."""
        campaign=self.campaign
        campaign.spent=lambda *_:12
        invalid={'successes':0,'trials':0,'evaluation_cells':[
            {'offset_mm':0.,'bottle':b,'trials':0,'successes':0} for b in ('light','heavy')],
            'records_complete':False,'mean_regrasps':None}
        original_policy=campaign.state['current_policy_id']
        original_data=copy.deepcopy(campaign.state['base_sessions'])
        with patch('bottle_scan.rsi.summarize',return_value=invalid), patch('bottle_scan.agentic.source_manifest',return_value={}):
            try:
                campaign.finish_cycle(self.cycle)
            except (ValueError,RuntimeError):
                pass
        self.assertEqual(campaign.state['current_policy_id'],original_policy)
        self.assertEqual(campaign.state['base_sessions'],original_data)
        self.assertNotEqual(campaign.state['phase'],'propose')


if __name__=='__main__': unittest.main()
