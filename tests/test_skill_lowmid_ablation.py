import unittest
from collections import Counter
from bottle_scan.skill_lowmid_ablation import (select_additions, canonical_metadata,
    initial_and_fixed_order, COMPONENTS)

class AblationTests(unittest.TestCase):
    def candidates(self):
        return [{"trial_path":f"p{i}","planned_delta_mm":float((i%9)-4),
                 "session":f"s{i%5}"} for i in range(80)]
    def test_metadata_allowlist_rejects_candidate_truth(self):
        c=self.candidates();c[0]["x_mm"]=999.
        with self.assertRaises(ValueError):
            select_additions(c,[x["trial_path"] for x in c[:24]],8,"offset_session_balanced",[],2)
    def test_projected_metadata_ignores_features_truth(self):
        c=self.candidates()
        self.assertEqual(canonical_metadata(c),canonical_metadata([{**x,"x_mm":999.,"descriptors":"secret"} for x in c]))
    def test_no_duplicate_equal_budget_all_modes(self):
        c=self.candidates();initial=[x["trial_path"] for x in c[:24]]
        order=[x["trial_path"] for x in c[24:]]
        for mode in ("fixed","simple_feedback","agent_evidence","offset_session_balanced","no_reference","no_B_hard","no_coverage"):
            selected=initial[:]
            for start in (0,8,16):
                new,log=select_additions(c,selected,8,mode,[(0.,3.)],100+start,start_added=start,fixed_order=order)
                self.assertEqual(len(new),8);self.assertFalse(set(new)&set(selected));selected+=new
            self.assertEqual(len(set(selected)),48)
    def test_full_continuous_cycle_equals_eight_each(self):
        c=self.candidates();selected=[x["trial_path"] for x in c[:24]];all_components=[]
        for start in (0,8,16):
            new,log=select_additions(c,selected,8,"agent_evidence",[(1.,4.)],10,start_added=start)
            all_components += [x["component"] for x in log];selected+=new
        self.assertEqual(Counter(all_components),Counter(reference=8,B_hard=8,coverage=8))
    def test_ablation_balanced_remaining_components(self):
        c=self.candidates();initial=[x["trial_path"] for x in c[:24]]
        for mode,components in COMPONENTS.items():
            if mode=="agent_evidence":continue
            _,log=select_additions(c,initial,8,mode,[(1.,4.)],10)
            self.assertEqual(Counter(x["component"] for x in log),Counter({x:4 for x in components}))
    def test_no_B_hard_ignores_B_feedback(self):
        c=self.candidates();p=[x["trial_path"] for x in c[:24]]
        a=select_additions(c,p,8,"no_B_hard",[(1.,999.)],7)
        b=select_additions(c,p,8,"no_B_hard",[(-4.,1.)],7)
        self.assertEqual(a,b)
    def test_balanced_ignores_B_feedback(self):
        c=self.candidates();p=[x["trial_path"] for x in c[:24]]
        self.assertEqual(select_additions(c,p,8,"offset_session_balanced",[],5),
                         select_additions(c,p,8,"offset_session_balanced",[(0.,999.)],5))
    def test_fixed_prefix_is_stable(self):
        pool=self.candidates()
        initial,order=initial_and_fixed_order(pool,1101)
        selected=initial[:]
        for start in (0,8,16):
            new,_=select_additions(pool,selected,8,"fixed",[],200,start_added=start,fixed_order=order)
            selected+=new
        self.assertEqual(selected,initial+order[:24])
    def test_insufficient_budget_is_not_clipped(self):
        c=self.candidates()
        with self.assertRaises(ValueError):select_additions(c,[x["trial_path"] for x in c[:75]],8,"fixed",[],1)
