"""Numerical and evidence-boundary tests for offline budget diagnostics."""
import unittest
from pathlib import Path
import numpy as np
from bottle_scan.contact_policy import DEFAULTS,_network,_predict
from bottle_scan.skill_budget_diagnosis import fit_predictions,stats,source_ids

class BudgetDiagnosisTests(unittest.TestCase):
    def test_exact_existing_learner_parity(self):
        X=np.random.default_rng(5).normal(size=(24,20));y=X[:,0]*3-X[:,3]
        cfg={**DEFAULTS,'epochs':8}
        pred,sc=fit_predictions(X,y,cfg,11,8,{'test':X[:5]})
        mean,scale,ys=sc;rng=np.random.default_rng(11)
        nets=[_network((X-mean)/scale,y/ys,cfg,rng) for _ in range(cfg['ensemble_size'])]
        expected=np.asarray([_predict(n,(X[:5]-mean)/scale)*ys for n in nets]).mean(0)
        np.testing.assert_allclose(pred['test'],expected,atol=0,rtol=0)

    def test_evaluation_values_cannot_change_normalization_or_other_predictions(self):
        X=np.arange(24*20,dtype=float).reshape(24,20);y=np.linspace(-5,5,24)
        cfg={**DEFAULTS,'epochs':4}
        p1,s1=fit_predictions(X,y,cfg,12,4,{'test':X[:2]})
        p2,s2=fit_predictions(X,y,cfg,12,4,{'test':X[:2],'another':X[2:]+1e7})
        np.testing.assert_array_equal(p1['test'],p2['test'])
        for a,b in zip(s1,s2):np.testing.assert_array_equal(a,b)

    def test_fixed_train_subset_scaling_does_not_use_test(self):
        X=np.random.default_rng(6).normal(size=(30,20));y=X[:,0]
        cfg={**DEFAULTS,'epochs':4};sc=(X[:12].mean(0),X[:12].std(0),1.)
        _,out=fit_predictions(X,y,cfg,1,4,{'test':X+1e6},scaling=sc)
        for a,b in zip(sc,out):np.testing.assert_array_equal(a,b)

    def test_stats_includes_large_errors(self):
        r=stats(np.array([0.,0.,10.]),np.zeros(3))
        self.assertEqual(r['n'],3);self.assertAlmostEqual(r['mae_mm'],10/3)
        self.assertEqual(r['max_mm'],10)

    def test_provenance_respects_declared_indices(self):
        m={'source_trials':[{'trial_path':'/tmp/diagnostic-trial-a'},{'trial_path':'/tmp/diagnostic-trial-b'}],
           'split':{'training_indices':[0],'validation_indices':[1]}}
        self.assertEqual(source_ids(m,'training'),['/tmp/diagnostic-trial-a'])
        self.assertFalse(set(source_ids(m,'training')) & set(source_ids(m,'validation')))
