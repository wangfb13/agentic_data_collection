"""No hardware: method memory, numerical selection and promotion gates."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from bottle_scan.rsi_methods import (MethodStore, baseline_method, validate_method,
    cell_features, select_cells, evaluate_comparison, learning_curve)


def child(parent=None):
    method=baseline_method()
    method.pop('method_id')
    method['name']='Conflict-aware diversity'
    method['parent_id']=parent
    method['score']={'op':'add','args':[{'op':'feature','name':'action_conflict'},
                                     {'op':'feature','name':'novelty'}]}
    method['sampling'].update(step_mm=1.,max_reps_per_cell=4,diversity_mm=1.)
    return validate_method(method)


def comparison(old_id='old',new_id='new',*,new_success=100,old_success=20,n=100,task='task1',simulated=True):
    cells=[{'offset_mm':0.,'bottle':'light','trials':n//2},
           {'offset_mm':0.,'bottle':'heavy','trials':n//2}]
    protocol={'task_id':task,'initial_dataset_hash':'dataset_hash','initial_policy_hash':'policy_hash',
              'training_budget':{'epochs':50,'ensemble':3,'seed':1},'collection_budget':12,
              'evaluation_cells':cells,'simulated':simulated,'preregistered':True,
              'evaluation_independent':True,'min_trials':20,'min_effect':.05}
    groups=[]
    for identity,k in ((old_id,old_success),(new_id,new_success)):
        group={k:copy.deepcopy(v) for k,v in protocol.items() if k not in ('preregistered','evaluation_independent','min_trials','min_effect')}
        group.update(method_id=identity,spent_trials=12,successes=k,trials=n,records_complete=True)
        group['evaluation_cells'][0]['successes']=k//2
        group['evaluation_cells'][1]['successes']=k-k//2
        groups.append(group)
    return {'old':groups[0],'new':groups[1],'protocol':protocol}


def observation(i,x,px,group=None,measured=True):
    desc={'valid':True,'centroid_px':[float(px),40.]}
    return {'id':str(i),'session':'session','group':group or str(i),'planned_delta_mm':float(x),'x_mm':float(x),
            'bottle':'light' if i%2 else 'heavy','descriptors':[copy.deepcopy(desc),copy.deepcopy(desc)],
            'measurement':{'source':'measured_tcp_probe_end' if measured else 'planned_delta_fallback'}}


class MethodTests(unittest.TestCase):
    def test_hash_and_lineage_validated(self):
        base=baseline_method()
        self.assertEqual(base,validate_method(base))
        changed=copy.deepcopy(base); changed['name']='a'
        with self.assertRaisesRegex(ValueError,'hash'): validate_method(changed)
        value=child(base['method_id'])
        with self.assertRaisesRegex(ValueError,'parent'): validate_method(value,parent_ids=set())
        self.assertEqual(value,validate_method(value,parent_ids={base['method_id']}))

    def test_no_code_or_control_escape(self):
        for patch in ({'F_max':99},{'score':{'op':'eval','code':'1'}},
                      {'score':{'op':'feature','name':'delta_truth'}},
                      {'score':{'op':'constant','value':float('nan')}},
                      {'score':{'op':'constant','value':100}}):
            method=baseline_method(); method.pop('method_id'); method.update(patch)
            with self.assertRaises(ValueError): validate_method(method)

    def test_AST_depth_and_node_count_bounded(self):
        method=baseline_method(); method.pop('method_id')
        ast={'op':'feature','name':'novelty'}
        for _ in range(10): ast={'op':'add','args':[ast,{'op':'constant','value':1}]}
        method['score']=ast
        with self.assertRaisesRegex(ValueError,'depth'): validate_method(method)
        ast={'op':'add','args':[{'op':'constant','value':1}]*8}
        method['score']={'op':'add','args':[{'op':'add','args':[copy.deepcopy(ast)]*8}]*8}
        with self.assertRaisesRegex(ValueError,'nodes'): validate_method(method)

    def test_schema_ref_validation(self):
        method=baseline_method()
        with self.assertRaisesRegex(ValueError,'references'): validate_method(method,known_refs=set())
        self.assertEqual(method,validate_method(method,known_refs={'baseline_protocol'}))
        for name,value in [('step_mm',.1),('diversity_mm',-1),('max_reps_per_cell',3),('extra_offsets_mm',[0,.1])]:
            bad=copy.deepcopy(method); bad.pop('method_id'); bad['sampling'][name]=value
            with self.assertRaises(ValueError): validate_method(bad)

    def test_conflicts_are_measured_independent_geometric_proxy(self):
        obs=[observation(1,-4,50),observation(2,4,50.2),observation(3,0,80),observation(4,8,50.1,measured=False)]
        report={'evidence':{'observations':obs}}
        result=cell_features(report,scope={'min_mm':-10,'max_mm':10},step_mm=1)
        self.assertEqual(len(result['conflict_examples']),1)
        self.assertEqual(result['excluded_unmeasured_from_conflict'],1)
        scores={c['offset_mm']:c['features']['action_conflict'] for c in result['cells']}
        self.assertGreater(scores[-4],.9)
        self.assertEqual(scores[8],0.)
        self.assertIn('not evidence',result['conflict_definition'])
        report['evidence']['observations']+=copy.deepcopy(obs)
        duplicate=cell_features(report,scope={'min_mm':-10,'max_mm':10},step_mm=1)
        self.assertEqual(result,duplicate)
        obs[1]['group']=obs[0]['group']
        self.assertEqual(cell_features({'evidence':{'observations':obs}},scope={'min_mm':-10,'max_mm':10})['conflict_examples'],[])

    def test_balanced_exact_reproducible_plan_and_no_silent_override(self):
        method=child(); scope={'min_mm':-10,'max_mm':10}
        cells=select_cells(method,{},scope=scope,trial_budget=12,seed=2)
        self.assertEqual(cells,select_cells(method,{},scope=scope,trial_budget=12,seed=2))
        self.assertEqual(sum(c['reps'] for c in cells),12)
        self.assertTrue(all(c['reps']%2==0 and c['reps']<=4 for c in cells))
        for budget in (0,1,3):
            with self.assertRaises(ValueError): select_cells(method,{},scope=scope,trial_budget=budget)
        impossible=copy.deepcopy(method); impossible.pop('method_id'); impossible['sampling']['diversity_mm']=50
        with self.assertRaisesRegex(ValueError,'cannot fill'): select_cells(impossible,{},scope=scope,trial_budget=6)
        outside=copy.deepcopy(method); outside.pop('method_id'); outside['sampling']['extra_offsets_mm']=[20.]
        with self.assertRaisesRegex(ValueError,'scope'): select_cells(outside,{},scope=scope,trial_budget=6)

    def test_blocked_evidence_not_silently_used(self):
        with self.assertRaisesRegex(ValueError,'blocked'):
            select_cells(baseline_method(),{'quality':{'blocking_reasons':['bad hardware']}},scope={'min_mm':-10,'max_mm':10},trial_budget=12)

    def test_selection_uses_conflict_scores(self):
        method=child(); method.pop('method_id'); method['score']={'op':'feature','name':'action_conflict'}
        obs=[observation(1,-4,50),observation(2,4,50)]
        cells=select_cells(method,{'evidence':{'observations':obs}},scope={'min_mm':-10,'max_mm':10},trial_budget=4)
        self.assertEqual({c['offset_mm'] for c in cells},{-4.,4.})


class ComparisonTests(unittest.TestCase):
    def test_large_separation_promotes_with_expected_fields(self):
        result=evaluate_comparison(**comparison())
        self.assertEqual(result['outcome'],'new_better')
        self.assertEqual(result['winner'],'new')
        self.assertGreater(result['difference_interval_conservative'][0],.05)

    def test_small_samples_or_ties_never_promote(self):
        for value in (comparison(new_success=4,old_success=0,n=4),comparison(new_success=90,old_success=90)):
            result=evaluate_comparison(**value)
            self.assertEqual(result['outcome'],'inconclusive')
            self.assertIsNone(result['winner'])

    def test_worse_method_is_kept_as_negative_evidence(self):
        self.assertEqual(evaluate_comparison(**comparison(new_success=20,old_success=100))['outcome'],'old_better')

    def test_origin_budget_complete_and_simulation_checks(self):
        fields={'initial_dataset_hash':'other','initial_policy_hash':'other','training_budget':{},
                'collection_budget':10,'spent_trials':13,'simulated':False,'records_complete':False,
                'trials':102,'successes':99}
        for key,value in fields.items():
            test=comparison(); test['new'][key]=value
            result=evaluate_comparison(**test)
            self.assertEqual(result['outcome'],'invalid',key)
            self.assertIsNone(result['winner'])

    def test_predeclared_cells_and_independence(self):
        for change in ('offset','count','missing','not_independent','not_preregistered'):
            test=comparison()
            if change=='offset': test['new']['evaluation_cells'][0]['offset_mm']=1.
            elif change=='count': test['new']['evaluation_cells'][0]['trials']=60
            elif change=='missing': test['new']['evaluation_cells'].pop()
            elif change=='not_independent': test['protocol']['evaluation_independent']=False
            else: test['protocol']['preregistered']=False
            self.assertEqual(evaluate_comparison(**test)['outcome'],'invalid',change)

    def test_descriptive_learning_curve(self):
        curve=learning_curve([{'spent_trials':0,'success_rate':.2},{'spent_trials':10,'success_rate':.8}])
        self.assertAlmostEqual(curve['normalized_auc'],.5)
        self.assertIn('not independent',curve['interpretation'])


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.store=MethodStore(self.tmp.name)
        self.old=self.store.register(baseline_method()); self.new=self.store.register(child(self.old))
    def tearDown(self): self.tmp.cleanup()
    def record(self,task='task1',simulated=True,problem=None,explicit_problem=True,created_unix_ns=None,**kwargs):
        compare=comparison(self.old,self.new,task=task,simulated=simulated,**kwargs)
        context={'hardware_signature':'hw1','issue':'contact ambiguity'}
        if problem is not None:
            context.update(problem_signature=problem,explicit_problem=explicit_problem)
        record={'method_id':self.new,'task_id':task,'simulated':simulated,
            'context':context,'diagnosis':'centroid conflict',
            'comparison':compare}
        if created_unix_ns is not None:
            record['created_unix_ns']=created_unix_ns
        return self.store.record_experience(record)
    def query(self,simulated=True):
        return {r['method']['method_id']:r for r in self.store.retrieve({'simulated':simulated,'hardware_signature':'hw1'})}

    def test_idempotent_versions_and_lineage(self):
        self.assertEqual(self.new,self.store.register(child(self.old)))
        self.assertEqual(self.record(),self.record())
        self.assertEqual(len(list((Path(self.tmp.name)/'experiences').glob('*.json'))),1)
        self.assertEqual(self.store.get(self.new)['parent_id'],self.old)
        orphan=child('not_known')
        with self.assertRaises(ValueError): self.store.register(orphan)

    def test_renamed_campaigns_do_not_establish_cross_problem_transfer(self):
        self.record()
        self.assertEqual(self.query()[self.new]['status'],'current_task_valid')
        self.record()
        self.assertEqual(self.query()[self.new]['status'],'current_task_valid')
        self.record(task='task2')
        result=self.query()[self.new]
        self.assertEqual(result['status'],'current_task_valid')
        self.assertEqual(result['winning_task_ids'],['task1','task2'])
        self.assertEqual(result['winning_problem_ids'],[])
        self.assertEqual(self.query(False)[self.new]['status'],'pending')
        self.record(task='task3',new_success=20,old_success=100)
        self.assertTrue(self.query()[self.new]['counterexample_ids'])
        self.assertEqual(self.query()[self.new]['status'],'contested')

    def test_distinct_explicit_problem_signatures_required_for_transfer(self):
        self.record(task='task1',problem='problem1')
        self.record(task='task2',problem='problem1')
        self.assertEqual(self.query()[self.new]['status'],'current_task_valid')
        self.record(task='task3',problem='problem2',explicit_problem=False)
        self.record(task='task4',problem='   ')
        self.assertEqual(self.query()[self.new]['winning_problem_ids'],['problem1'])
        self.assertEqual(self.query()[self.new]['status'],'current_task_valid')
        self.record(task='task5',problem='problem2')
        result=self.query()[self.new]
        self.assertEqual(result['status'],'cross_task_valid')
        self.assertEqual(result['winning_problem_ids'],['problem1','problem2'])
        self.record(task='task6',problem='problem3',new_success=20,old_success=100)
        self.assertEqual(self.query()[self.new]['status'],'contested')

    def test_experience_recency_uses_fixed_timestamp_not_content_hash(self):
        older=self.record(task='legacy')
        identifiers=[self.record(task=f'ordered_{i}',created_unix_ns=i+1) for i in range(8)]
        self.assertNotEqual(identifiers,sorted(identifiers),'fixture must exercise opposite hash ordering')
        self.assertEqual([r['experience_id'] for r in self.store._experiences()],[older]+identifiers)
        recent=self.query()[self.new]['past_experiences'][-6:]
        self.assertEqual([r['created_unix_ns'] for r in recent],list(range(3,9)))
        self.assertEqual(self.record(task='ordered_7',created_unix_ns=8),identifiers[-1])

    def test_legacy_timestamp_not_synthesized_and_supplied_timestamp_validated(self):
        identifier=self.record()
        self.assertEqual(self.record(),identifier)
        self.assertNotIn('created_unix_ns',self.store._experiences()[0])
        for bad in (0,-1,True,1.5,'yesterday'):
            with self.subTest(timestamp=bad),self.assertRaisesRegex(ValueError,'created_unix_ns'):
                self.record(created_unix_ns=bad)

    def test_hardware_scope_never_borrows_simulated_wins(self):
        self.record(simulated=False)
        self.assertEqual(self.query(False)[self.new]['status'],'current_task_valid')
        self.assertEqual(self.query(True)[self.new]['status'],'pending')
        other={r['method']['method_id']:r for r in self.store.retrieve({'simulated':False,'hardware_signature':'hw2'})}
        self.assertEqual(other[self.new]['status'],'pending')
        with self.assertRaisesRegex(ValueError,'signature'): self.store.retrieve({'simulated':False})

    def test_unverified_claims_rejected_and_pending_retained(self):
        entry={'method_id':self.new,'task_id':'task1','simulated':True,'context':{'hardware_signature':'hw1'},
               'notes':'This is claimed to be the greatest method but no evidence yet'}
        self.store.record_experience(entry)
        self.assertEqual(self.query()[self.new]['status'],'pending')
        for key in ('status','promoted','winner','evaluation','outcome'):
            bad=copy.deepcopy(entry); bad[key]='passed'
            with self.assertRaises(ValueError): self.store.record_experience(bad)
        self.record(new_success=20,old_success=100)
        self.assertEqual(self.query()[self.new]['status'],'not_supported')

    def test_corruption_and_path_traversal_fail(self):
        identity=self.record()
        path=Path(self.tmp.name)/'experiences'/(identity+'.json')
        data=json.loads(path.read_text()); data['evaluation']['winner']=self.old
        path.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError,'hash'): self.store.retrieve({'simulated':True,'hardware_signature':'hw1'})
        with self.assertRaises(ValueError): self.store.get('../../something')


if __name__=='__main__': unittest.main()
