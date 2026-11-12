"""Task targets are labelled supported state sets; no hardware/network used."""
import copy
from unittest.mock import Mock
import numpy as np

from bottle_scan.acceptable_contact_set import match_features, choose_action, validate, _id
from bottle_scan.learned_skill import LearnedSkillModel, train_learned_skill, _seal
from bottle_scan.recovery_skill import RecoverySkill
from tests.test_learned_skill import LearnedSkillFixture, learned_settings, TARGET_ID, SOURCE
from tests.test_agentic_learning import desc, servo


class AcceptableSetTests(LearnedSkillFixture):
    def train_set(self, **kwargs):
        root=self.make_learned(**kwargs)
        cfg=learned_settings();cfg['target_mode']='acceptable_set'
        model=train_learned_skill([root],cfg)
        self.assertTrue(model['usable_for_B'],model['blocking_reasons'])
        return root,cfg,model

    def observation(self,x,ts=1_000_000_000):
        return {'status':'contact','bilateral':True,'descriptors':[desc(60-3*x),desc(55-3*x)],
            'context':{'target_definition_id':TARGET_ID},'timestamp_ns':ts,'now_ns':ts,
            'first_touch_complete':True,'safety_ok':True,'force_stable':True,'stable_duration_s':.2}

    def test_nonzero_target_exits_without_motion_and_reports_set_state(self):
        _,_,model=self.train_set()
        self.assertEqual(model['schema'],'learned_a_skill_set_v1')
        self.assertIn(2.,model['acceptable_set']['anchor_positions_mm'])
        skill=RecoverySkill(model,{**servo(),'max_regrasps':3},mode='development')
        skill.policy.predict=Mock(side_effect=AssertionError('acceptable contact needs no recentering'))
        first=skill.start(self.observation(2.))
        self.assertFalse(first['invocation']['needs_recovery'])
        self.assertIsNone(first['recovery']['action'])
        last=skill.advance(self.observation(2.,2_000_000_000))
        self.assertTrue(last['exit']['completed'])
        state=last['exit']['achieved_state']
        self.assertEqual(state['set_id'],model['acceptable_set']['set_id'])
        self.assertEqual(state['target_definition_id'],TARGET_ID)
        self.assertEqual(state['timestamp_ns'],2_000_000_000)
        self.assertEqual(state['contact_features'],[54.,40.,49.,40.])
        self.assertTrue(state['anchor_ids'])
        skill.policy.predict.assert_not_called()

    def test_recovery_goes_to_nearest_nonzero_target_not_T0(self):
        _,_,model=self.train_set()
        skill=RecoverySkill(model,{**servo(),'max_regrasps':3},mode='development')
        skill.policy.predict=Mock(return_value={'valid':True,'action_mm':-5.,'uncertainty_mm':.1})
        first=skill.start(self.observation(6.))
        self.assertEqual(first['recovery']['action']['delta_mm'],-3.)
        self.assertEqual(first['recovery']['decision']['target_a_mm'],2.)
        self.assertFalse(first['exit']['completed'])
        following=skill.advance(self.observation(2.,3_000_000_000),
            applied_action={'applied':True,'action_mm':-3.,'completed_ns':2_000_000_000})
        self.assertFalse(following['exit']['completed'])
        last=skill.advance(self.observation(2.,4_000_000_000))
        self.assertTrue(last['exit']['completed'])

    def test_supported_union_does_not_fill_unobserved_gaps(self):
        _,_,model=self.train_set()
        target=model['acceptable_set']
        self.assertTrue(match_features(target,[54.,40.,49.,40.])['supported'])
        # +1 lies between labelled 0 and +2 but outside either narrow observed
        # neighborhood. It cannot be accepted merely because a hull includes it.
        self.assertFalse(match_features(target,[57.,40.,52.,40.])['supported'])
        state=LearnedSkillModel(model).exit(self.observation(1.))
        self.assertFalse(state['valid'])
        self.assertEqual(state['reason'],'exit_outside_supported_acceptable_set')

    def test_geometry_and_state_holdouts_do_not_supply_anchors(self):
        root,cfg,model=self.train_set()
        aset=model['acceptable_set'];anchors=aset['anchors']
        groups=model['exit_head']['split']['groups']
        geometry_train={model['recovery_model']['source_trials'][i]['trial_path']
            for i in model['recovery_model']['split']['training_indices']}
        for anchor in anchors:
            self.assertIn(anchor['source']['group'],groups['train'])
            self.assertNotIn(anchor['source']['group'],groups['calibration']+groups['validation'])
            self.assertIn(anchor['source']['trial_path'],geometry_train)
        held=set(groups['validation'])
        def mutate_validation(row):
            if f"{root}|{row['bottle']}|{row['repeat']}" in held:
                for event in row['skill_observations']:
                    event['observation']['descriptors'][0]['centroid_px'][1]+=1000
        self.mutate(root,mutate_validation)
        changed=train_learned_skill([root],cfg)
        self.assertEqual(anchors,changed['acceptable_set']['anchors'])
        self.assertFalse(changed['usable_for_B'])

    def test_no_measured_positive_anchors_never_falls_back_to_T0(self):
        root,cfg,model=self.train_set()
        def erase_positive_position(row):
            if row['skill_ground_truth']['initial_class']=='already_normal':
                row['probe_initial'].pop('measured_tcp_mm',None)
        self.mutate(root,erase_positive_position)
        changed=train_learned_skill([root],cfg)
        self.assertFalse(changed['usable_for_B'])
        self.assertFalse(changed['acceptable_set']['usable'])
        self.assertEqual(changed['acceptable_set']['anchors'],[])

    def test_all_normal_still_needs_explicit_negative_boundary_evidence(self):
        root=self.make_learned()
        def all_normal(row):
            row['skill_ground_truth']['initial_class']='already_normal'
            for label in row['skill_window_labels']:label['target_reached']=True
        self.mutate(root,all_normal)
        cfg=learned_settings();cfg['target_mode']='acceptable_set'
        model=train_learned_skill([root],cfg)
        self.assertFalse(model['usable_for_B'])
        self.assertTrue(model['data_gaps'])
        self.assertFalse(model['acceptable_set']['usable'])

    def test_disjoint_positive_modes_keep_rejected_center_outside_set(self):
        root=self.make_learned(offsets=(-6.,-4.,0.,4.,6.))
        def disjoint(row):
            first=row['skill_observations'][0]['observation']
            # Synthetic independent semantic truth: two accepted regions with
            # a central forbidden state. Production code never uses offsets.
            u=first['descriptors'][0]['centroid_px'][0]
            target=u<=48 or u>=72
            row['skill_ground_truth']['initial_class']='already_normal' if target else 'needs_recovery'
            row['skill_observations'][1]['observation']['descriptors']=copy.deepcopy(first['descriptors'])
            for label in row['skill_window_labels']:label['target_reached']=target
        self.mutate(root,disjoint)
        cfg=learned_settings();cfg['target_mode']='acceptable_set'
        model=train_learned_skill([root],cfg)
        self.assertTrue(model['usable_for_B'],model['blocking_reasons'])
        aset=model['acceptable_set'];state=LearnedSkillModel(model)
        self.assertTrue(state.exit(self.observation(-4.))['positive'])
        self.assertTrue(state.exit(self.observation(4.))['positive'])
        self.assertFalse(state.exit(self.observation(0.))['positive'])
        self.assertFalse(match_features(aset,[60.,40.,55.,40.])['supported'])
        action=choose_action(aset,{'valid':True,'action_mm':-1.,'uncertainty_mm':.1})
        self.assertEqual(action['target_a_mm'],4.)
        self.assertEqual(action['action_mm'],3.)

    def test_unknown_target_modes_and_resealed_bad_anchors_rejected(self):
        root,cfg,model=self.train_set()
        bad=copy.deepcopy(model);bad['acceptable_set']['anchors'][0]['a_mm']=float('nan')
        with self.assertRaises(ValueError):validate(bad['acceptable_set'],TARGET_ID)
        bad=copy.deepcopy(model);bad['acceptable_set']['anchors'][0]['support_groups']=[]
        bad['acceptable_set']['set_id']=_id(bad['acceptable_set'])
        with self.assertRaisesRegex(ValueError,'anchor'):LearnedSkillModel(_seal(bad))
        legacy=copy.deepcopy(model);legacy['schema']='learned_a_skill_v1'
        with self.assertRaisesRegex(ValueError,'schema'):LearnedSkillModel(_seal(legacy))
        cfg['target_mode']='arbitrary'
        with self.assertRaisesRegex(ValueError,'target_mode'):train_learned_skill([root],cfg)
