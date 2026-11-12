import tempfile
from pathlib import Path
import unittest
from bottle_scan.skill_repeated_benchmark import validate_split, monitor_blocks

class SplitTests(unittest.TestCase):
    def fixtures(self):
        obs=[{'trial_path':str(i),'group':f'g{i//2}','session':'s'} for i in range(10)]
        roles={str(i):'pool' if i<6 else 'development' if i<8 else 'independent' for i in range(10)}
        return obs,roles,{o['trial_path']:o['group'] for o in obs}
    def test_complete_disjoint_groups(self):
        o,r,b=self.fixtures();validate_split(o,r,b,[2,6])
    def test_budget_cannot_clip(self):
        o,r,b=self.fixtures()
        with self.assertRaises(ValueError):validate_split(o,r,b,[2,8])
    def test_same_repeat_never_crosses_roles(self):
        o,r,b=self.fixtures();r['5']='development'
        with self.assertRaises(ValueError):validate_split(o,r,b,[2])
    def test_monitor_attachment_cannot_leak(self):
        o,r,b=self.fixtures();b['8']='g2'
        with self.assertRaises(ValueError):validate_split(o,r,b,[2])
    def test_monitor_attaches_same_bottle_preceding_repeat(self):
        o=[{'trial_path':'p1','group':'g1','session':'s'},
           {'trial_path':'p2','group':'g2','session':'s'},
           {'trial_path':'p3','group':'mon','session':'s'}]
        rows={'p1':{'id':'1','bottle':'light','repeat':0},
              'p2':{'id':'2','bottle':'heavy','repeat':1},
              'p3':{'id':'3','bottle':'light','repeat':None}}
        self.assertEqual(monitor_blocks(o,rows)['p3'],'g1')
    def test_missing_trial_rejected(self):
        o,r,b=self.fixtures();del r['0']
        with self.assertRaises(ValueError):validate_split(o,r,b,[2])
