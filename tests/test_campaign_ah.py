import subprocess
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch
import h5py
import numpy as np
from bottle_scan.state_graph import campaign
from bottle_scan.state_graph.graph import observed_edges
from bottle_scan.state_graph.operators import validate_decision
from skills.contact_position_a import campaign_ah as ah
from skills.contact_position_a import ah_geometry as geo, ah_evidence as ev, placement_probe as probe


class MixedFrameCollection(unittest.TestCase):
    def setUp(self):
        tmp=tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root=Path(tmp.name)
        self.pose=np.eye(4)
        theta=.8
        self.pose[:3,:3]=[[np.cos(theta),0,np.sin(theta)],[0,1,0],[-np.sin(theta),0,np.cos(theta)]]
        self.center=self.pose.copy()
        self.spec,self.cfg=ah.configuration(self.root,center=self.center)
        self.axis=[0,0,1]
        self.ref={'reference_digest':'r','protocol':{'axes_tcp':{'a':self.axis}}}
        self.identity=(self.ref,{'model_id':'m'},{'left':'L','right':'R'})
        self.hw={'axes_tcp':{'a':self.axis},'gripper':{'open_normalized':1.,'open_tolerance':.01}}
        center_file=self.root/'history/P2/center/observation.json'
        self.capture(center_file)
        self.cfg['execution']['center_file']=str(center_file)
        campaign.atomic(self.root/'placement_basis.json',dict(measured_pose=self.center.tolist(),reference_digest='r'))
        self.moves=[]
        self.static=True

    def test_resume_adds_sampling_preference_without_replacing_pending_plan(self):
        out=self.root/'campaign'
        campaign.atomic(out/'placement_basis.json',dict(measured_pose=self.center.tolist()))
        spec,cfg=ah.configuration(out)
        old=deepcopy(cfg)
        del old['execution']['sampling_preference']
        state={'range_version':ah.RANGE_VERSION,'pending':{'trial_ids':['already_queued']},'spent':{'trials':1}}
        for name,value in [('input',old),('config',old),('spec',spec),('instance',spec),('state',state)]:
            campaign.atomic(out/(name+'.json'),value)
        ah.initialize(out)
        ah.initialize(out)
        self.assertEqual(probe.read(out/'input.json'),cfg)
        self.assertEqual(probe.read(out/'config.json')['execution']['sampling_preference']['base_h_negative'],'optional')
        self.assertEqual(probe.read(out/'state.json'),state)

    def capture(self,path):
        t=10**9
        fb=dict(tcp_frame='right_gripper_base_link',base_frame='base_link',
                T_base_tcp_mm=self.pose.tolist(),received_monotonic_ns=t)
        sample={'normal_sdk':[1.5,1.5],'descriptors':[dict(centroid_px=[30,20],area_px=5)]*2,
                'window':[{'frame_ns':[t]}]*2,'timestamp_ns':t,'valid':True}
        row=dict(schema=probe.LABEL,placement=path.parent.parent.name,trial=path.parent.name,state='released',status='contact',
                 bottle_static=getattr(self,'static',True),model_id='m',finger_serials=self.identity[-1],reference_digest='r',
                 window_pose=fb,before_pose=self.pose.tolist(),after_pose=self.pose.tolist(),
                 touch_report={'width_normalized':.4},probe=sample,unloaded_zero={'offset_raw':[[1]*6]*2})
        campaign.atomic(path,row)
        parent=path.parent/'record'
        parent.mkdir()
        with h5py.File(parent/'streams.h5','w') as f:
            for side in ('left','right'):
                g=f.require_group('streams/wrench_raw_'+side)
                g['monotonic_ns']=[t];g['data']=[[3]*6]

    def invoke(self,command,*args,**kwargs):
        get=lambda flag:args[args.index(flag)+1]
        if command=='record_pose.py':
            campaign.atomic(get('--output'),dict(T_base_tcp_mm=self.pose.tolist(),pose_source={'arms':{'right':{'tcp_frame':'right_gripper_base_link'}},'base_frame':'base_link'},gripper=dict(fresh=True,valid=True,normalized=1.)))
        elif command=='move_tcp.py':
            start=self.pose.copy()
            if '--xyz-mm' in args:
                i=args.index('--xyz-mm')+1
                self.pose[:3,3]=args[i:i+3]
            else:
                amount=float(get('--distance-mm'))*(1 if get('--direction')=='z+' else -1)
                frame=get('--frame')
                self.moves.append((frame,amount))
                self.pose[:3,3]+=amount*(self.pose[:3,2] if frame=='tcp' else np.array([0,0,1]))
            extra=dict(relative_frame=get('--frame'),frame_semantics='action_start_v1') if '--relative-at-confirm' in args else {}
            campaign.atomic(Path(get('--log-dir'))/'tcp_move_fake.json',dict(arm='right',real=True,base_frame='base_link',**extra,
                            tcp_frame='right_gripper_base_link',execution_start_mm=start.tolist(),end_measured_mm=self.pose.tolist(),
                            result='reached_by_measured_feedback'))
        elif command=='-m':
            self.capture(Path(get('--run'))/get('--placement')/get('--trial')/'observation.json')
        else:self.fail(command)

    def exercise(self,u,invoke=None):
        folder=self.root/'t'
        folder.mkdir()
        req=dict(trial_id='t',campaign=str(self.root),role='exploration',mode='planned',settings=dict(zip(geo.COORDS,[0,0])),
                 actions=[dict(kind=geo.KIND,u=u)],block={'mass_class':'light'})
        with patch.object(ah,'check_assets',return_value=(self.hw,self.identity)):
            return ah.execute_trial(req,folder,self.spec,self.cfg,real=True,interactive=True,ask=lambda x:'MOVE',invoke=invoke or self.invoke)

    def test_initial_open_feedback_includes_one_count_boundary(self):
        self.hw['gripper']['open_tolerance']=.001
        for counts,fresh,valid,ok in [(1000,True,True,True),(999,True,True,True),
                                      (998,True,True,False),(999,False,True,False),(999,True,False,False)]:
            with self.subTest(counts=counts,fresh=fresh,valid=valid):
                folder=self.root/f'initial_{counts}_{fresh}_{valid}'
                def read_only(command,*args,**kwargs):
                    self.assertEqual(command,'record_pose.py')
                    self.invoke(command,*args,**kwargs)
                    path=Path(args[args.index('--output')+1])
                    data=probe.read(path)
                    data['gripper']=dict(normalized=counts/1000,fresh=fresh,valid=valid)
                    campaign.atomic(path,data)
                if ok:
                    np.testing.assert_array_equal(ah.current_pose(folder,'prepare_pose_0',self.hw,read_only),self.pose)
                    self.assertFalse((folder/'initial_open_rejection.json').exists())
                else:
                    with self.assertRaises(campaign.RetryTrial):
                        ah.current_pose(folder,'prepare_pose_0',self.hw,read_only)
                    with self.assertRaises(campaign.RetryTrial):
                        ah.recover_trial({},folder,{}, {},interactive=False)
                    receipt=probe.read(folder/'interruption.json')
                    self.assertEqual(receipt['stage'],'prepare_pose_0')
                    self.assertEqual(receipt['move_receipts_with_execution_start'],[])

    def test_normal_cli_resumes_same_free_run_and_fixed_comparison_is_explicit(self):
        state=dict(phase='ready_to_collect',spent={},pending=None)
        with patch.object(ah.ah_current,'initialize') as free,patch.object(ah,'initialize') as fixed, \
                patch.object(ah.campaign,'run',return_value=state):
            ah.main([])
            free.assert_called_once_with(ah.DEFAULT_OUT.with_name('campaign_white_ah_here_P3'),'P3',ah.DEFAULT_OUT)
            fixed.assert_not_called()
            ah.main(['--fixed-p2'])
            fixed.assert_called_once_with(ah.DEFAULT_OUT)

    def test_new_bottle_placement_collects_at_current_hand_and_moves_only_selected_axis(self):
        prior=self.root/'p2'
        prior.mkdir()
        for name,value in [('history_windows',[]),('spec',self.spec),('config',self.cfg),
                           ('placement_basis',dict(measured_pose=self.center.tolist())),
                           ('state',dict(trials=[],spent={'trials':19}))]:
            campaign.atomic(prior/(name+'.json'),value)
        out=self.root/'here'
        ah.ah_current.initialize(out,'P3',prior)
        spec=probe.read(out/'spec.json')
        cfg=probe.read(out/'config.json')
        self.assertEqual(spec['budget']['max_new_trials'],5)
        self.assertNotIn('center_file',cfg['execution'])
        # Same run id, fresh pose per trial, even when a aligns exactly with base Z.
        (out/'travel_origin.json').write_text('old archive deliberately unreadable')
        yaw=np.array([[0,-1,0],[1,0,0],[0,0,1]])
        for n,u in enumerate((None,[3,0],[10,0],[0,45],[0,-3],[-10,0])):
            self.pose=np.eye(4)
            self.pose[:3,:3]=np.linalg.matrix_power(yaw,n)@self.center[:3,:3] if n%2 else np.eye(3)
            self.pose[:3,3]=[120*n,-70*n,80*n]
            start=self.pose.copy()
            actions=[] if u is None else [dict(kind=geo.KIND,u=u)]
            req=dict(trial_id=f't{n}',campaign=str(out),role='exploration',mode='planned',settings={},
                     actions=actions,block={'mass_class':'light'})
            directory=out/req['trial_id']
            directory.mkdir()
            with patch.object(ah,'check_assets',return_value=(self.hw,self.identity)):
                result=ah.execute_trial(req,directory,spec,cfg,real=True,interactive=True,
                                        ask=lambda _: 'MOVE',invoke=self.invoke)
            nodes=result['nodes']
            self.assertEqual(result['status'],'complete')
            np.testing.assert_allclose(nodes[0]['provenance']['measured_pose'],start)
            self.assertFalse(any(directory.glob('prepare_move_*')))
            for node in nodes:
                self.assertEqual(node['plan_coords'],{})
                self.assertIsNone(node['attrs']['approach_pose'])
                self.assertEqual(node['session'],'P3')
            if u:
                np.testing.assert_allclose(nodes[-1]['u_executed'],u,atol=1e-10)
                np.testing.assert_allclose(self.pose[:3,3]-start[:3,3],start[:3,2]*u[0]+np.array([0,0,u[1]]),atol=1e-10)
                edge=observed_edges(nodes,spec)[0]
                ev.exploration_context(nodes,{'edges':[{**edge,'edge_id':'e'}]},[],spec)
        for settings in ({'h_offset_mm':15},{'a_offset_mm':0}):
            with self.assertRaises(ValueError):
                geo.validate_cell({'settings':settings,'actions':[]},spec,'collect_seed',{})
        for u in ([0,46],[0,-4],[-11,0]):
            with self.assertRaises(ValueError):
                geo.validate_cell({'actions':[dict(kind=geo.KIND,u=u)]},spec,'collect_transitions',{})
        self.assertEqual((out/'travel_origin.json').read_text(),'old archive deliberately unreadable')

        # Resuming an old run must migrate just the inherited P2 sampling band.
        state=probe.read(out/'state.json')
        for bounds,keep in (([-3.,45.],False),([-50.,100.],True)):
            cfg['execution'].update(start_mode='current_contact_ah_v1',table_tcp_z_mm=bounds)
            campaign.atomic(out/'config.json',cfg)
            ah.ah_current.initialize(out,'P3',prior)
            ah.ah_current.initialize(out,'P3',prior)
            execution=probe.read(out/'config.json')['execution']
            self.assertEqual(execution['start_mode'],ah.ah_current.MODE)
            self.assertEqual('table_tcp_z_mm' in execution,keep)
            self.assertEqual(probe.read(out/'state.json'),state)

    def test_height_end_to_end(self):
        result=self.exercise([0,-3])
        self.assertEqual(self.moves,[('base',-3)])
        self.assertEqual(result['status'],'complete')
        self.assertEqual(result['nodes'][0]['obs']['force'],[2]*12)
        edges=observed_edges(result['nodes'],self.spec)
        np.testing.assert_allclose(edges[0]['u'],[0,-3],atol=1e-12)
        self.assertFalse(result['nodes'][-1]['post'][self.spec['active_post_condition']]['evaluated'])
        self.assertEqual(ah.recover_trial({},self.root/'t',{}, {},interactive=False),result)

    def test_a_end_to_end(self):
        result=self.exercise([3,0])
        self.assertEqual(self.moves,[('tcp',3)])
        np.testing.assert_allclose(result['nodes'][-1]['u_executed'],[3,0],atol=1e-12)

    def test_l_unchanged_and_degenerate(self):
        for key,s in [('l_offset_mm',2),('closure_floor',.3)]:
            with self.assertRaises(ValueError):geo.validate_cell({'settings':{key:s}},self.spec,'',{})
        with self.assertRaises(ValueError):geo.basis(np.eye(4),[0,0,1])
        with self.assertRaises(ValueError):geo.validate_cell({'settings':{'h_offset_mm':45},'actions':[{'kind':geo.KIND,'u':[0,1]}]},self.spec,'',{})

    def test_new_workspace_endpoints_and_single_axis_execution(self):
        result=self.exercise([0,45])
        self.assertEqual(self.moves,[('base',45)])
        np.testing.assert_allclose(result['nodes'][-1]['u_executed'],[0,45],atol=1e-10)
        for h,u,ok in [(45,-45,True),(10,-13,True),(0,-4,False),(44,2,False),(0,46,False)]:
            cell={'settings':{'h_offset_mm':h},'actions':[{'kind':geo.KIND,'u':[0,u]}]}
            if ok:geo.validate_cell(cell,self.spec,'',{})
            else:
                with self.assertRaises(ValueError):geo.validate_cell(cell,self.spec,'',{})
        for u in (10,-10):
            geo.validate_cell({'settings':{'h_offset_mm':15},'actions':[{'kind':geo.KIND,'u':[u,0]}]},self.spec,'',{})
        for u in ([11,0],[0,46],[5,5]):
            with self.assertRaises(ValueError):
                geo.validate_cell({'settings':{},'actions':[{'kind':geo.KIND,'u':u}]},self.spec,'',{})

    def test_actual_world_height_and_motion_from_high_start(self):
        # At this synthetic orientation +a also raises base Z.
        with self.assertRaises(ValueError):
            geo.validate_cell({'settings':{'a_offset_mm':10,'h_offset_mm':44}},self.spec,'',{})
        self.pose=geo.target(self.center,self.center,[0,45],self.axis)
        ah.prepare_position(self.root,self.center,[0,-3],self.hw,self.spec,self.invoke)
        np.testing.assert_allclose(self.pose[:3,3]-self.center[:3,3],[0,0,-3],atol=1e-10)
        shifted=geo.target(self.center,self.center,[0,1],self.axis)
        np.testing.assert_allclose(ah.issued_action(self.center,shifted,np.array([0.,45]),self.axis,self.spec),[0,44])

    def test_upgrade_preserves_trials_and_retry_replans_only_unstarted(self):
        out=self.root/'campaign'
        campaign.atomic(out/'placement_basis.json',dict(measured_pose=self.center.tolist()))
        old_spec,old_cfg=ah.configuration(out,legacy=True)
        for name,value in [('input',old_cfg),('config',old_cfg),('spec',old_spec),('instance',old_spec)]:
            campaign.atomic(out/(name+'.json'),value)
        ids=['done','started','retry','unused']
        state={'phase':'collect','rule':campaign.initial_rule([],old_spec),'trials':[],'spent':{'trials':2},
               'pending':{'role':'exploration','trial_ids':ids},'history':[{'original':True}]}
        for trial_id,status in zip(ids,['complete','started','queued','queued']):
            directory=out/trial_id
            campaign.atomic(directory/'request.json',{'trial_id':trial_id})
            rec=dict(directory=str(directory),trial_id=trial_id,status=status,request=probe.read(directory/'request.json'))
            if trial_id=='retry':rec['incomplete_attempts']=[{'reason':'sensor fault'}]
            state['trials'].append(rec)
        campaign.atomic(out/'state.json',state)
        ah.initialize(out);s=probe.read(out/'state.json')
        self.assertEqual(s['pending']['trial_ids'],ids[:-1])
        self.assertEqual(s['pending']['superseded_trial_ids'],['unused'])
        self.assertEqual(s['trials'][2]['status'],'queued')
        self.assertEqual(s['spent'],state['spent'])
        self.assertEqual(s['history'],state['history'])
        self.assertEqual(s['rule']['variables']['h_offset_mm']['range'],[-3,45])
        ah.initialize(out)
        self.assertEqual(probe.read(out/'state.json'),s)

    def test_no_edge_if_bottle_moved(self):
        self.static=False
        r=self.exercise([0,3])
        self.assertEqual(r['status'],'failed')
        self.assertEqual(r['nodes'],[])
        self.assertEqual(self.moves,[])

    def test_director_selects_each_sign_and_rejects_wrong_action(self):
        facts,ops=ev.exploration_context([],{'edges':[]},[],self.spec)
        ctx=dict(spec=self.spec,facts=facts,applicable_operators=ops,execution=self.cfg['execution'])
        for op in [op for op in ops if 'axis' in op['bindings']]:
            args=op['bindings']
            u=[0,0];u[['a','h'].index(args['axis'])]=2*(1 if args['side']=='positive' else -1)
            cell=dict(settings={},reps=1,blocks=[{'mass_class':'light'}],actions=[dict(kind=geo.KIND,u=u)])
            ans=dict(operator=op['op'],args=args,premises=op['premises_satisfied_by'],plan={'cells':[cell]},
                     hypotheses_update=[],discriminates=[],expected='direction',reason='compare')
            validate_decision(ans,ctx)
            bad=deepcopy(ans);bad['plan']['cells'][0]['actions'][0]['u']=list(-np.asarray(u))
            with self.assertRaises(ValueError):validate_decision(bad,ctx)

    def test_free_start_amplitudes_count_receipts_by_group_axis_sign_and_distance(self):
        spec=deepcopy(self.spec)
        spec.pop('placement_geometry')
        spec['current_placement']='P3'
        ah.ah_current.set_trial_protocol(spec,self.cfg['execution'])
        rows=[]
        edges=[]
        def add(name, net, axis, *, session='P3', kind='observed', applied=True):
            first,last=name+':0',name+':1'
            rows.extend({'node_id':n,'session':session} for n in (first,last))
            edges.append(dict(edge_id=name,source=first,target=last,kind=kind,u=net,
                              receipt=dict(kind=geo.KIND,applied=applied,commanded_axis=axis,
                                           net_contact_displacement_mm=net)))
        add('h3',[0,2.324364434],'h')
        add('h3_repeat',[0,3.1],'h')
        add('halfway',[0,4],'h')
        add('between',[0,8],'h')
        add('a_negative',[-5.2,0],'a')
        add('a_sub_mm',[.6,0],'a')
        add('old',[0,15],'h',session='P2')
        add('inferred',[0,30],'h',kind='inferred')
        add('unapplied',[0,45],'h',applied=False)
        add('mixed',[2,10],None)
        # Candidate labels and plans cannot replace the physical measurement.
        edges[0]['u']=[0,15]
        facts,ops=ev.exploration_context(rows,{'edges':edges},[],spec)
        coverage={tuple(f['args'][1:]):f for f in facts if f['pred']=='action_amplitude_coverage'}
        self.assertEqual(len(coverage),4)
        positive=coverage['h','positive']
        metrics=positive['metrics']
        self.assertEqual(metrics['unsampled_intervals_mm'],[[0,2.324364434],[2.324364434,3.1],
                                                           [3.1,4],[4,8],[8,45]])
        actual={s['edge_id']:s['measured_mm'] for s in metrics['measured_transitions']}
        self.assertEqual(actual,{'h3':2.324364434,'h3_repeat':3.1,'halfway':4,'between':8})
        self.assertEqual(coverage['a','negative']['metrics']['unsampled_intervals_mm'],[[-10,-5.2],[-5.2,0]])
        self.assertEqual(coverage['a','positive']['metrics']['measured_transitions'],
                         [{'edge_id':'a_sub_mm','measured_mm':.6}])
        self.assertEqual(coverage['h','negative']['metrics']['unsampled_intervals_mm'],[[-3,0]])
        self.assertTrue(any(f['pred']=='direction_sampled' and f['args']==['a','positive'] for f in facts))
        from bottle_scan.state_graph.skill_interface import action_grid
        self.assertNotIn('candidate_actions',spec['task']['action_space'])
        for key in geo.COORDS:
            self.assertNotIn('nominal_step',spec['perturbation_variables'][key])
        self.assertEqual(action_grid(spec).shape,(0,2))
        self.assertEqual(action_grid(spec,[edges[0]]).tolist(),[[0,2.324364434]])
        geo.validate_cell({'actions':[{'kind':geo.KIND,'u':[0,12.713]}]},spec,'collect_transitions',{})
        self.assertFalse(coverage['h','negative']['evidence'])
        for op in ops:
            args=op['bindings']
            if 'axis' in args:
                self.assertIn(coverage[args['axis'],args['side']]['id'],op['premises_satisfied_by'])

    def test_continuous_preview_keeps_live_requests_and_reserves_their_budget(self):
        out=self.root/'preview_test'
        spec=deepcopy(self.spec)
        spec.pop('placement_geometry')
        spec['current_placement']='P3'
        cfg={**self.cfg,'spec':str(out/'instance.json'),'sources':[]}
        cfg['execution']['start_mode']=ah.ah_current.MODE
        campaign.atomic(out/'instance.json',spec)
        campaign.atomic(out/'input.json',cfg)
        state=campaign.initialize(out/'input.json',out)
        old_plan={'cells':[dict(settings={},reps=2,blocks=[{'mass_class':'light'}],
                                actions=[{'kind':geo.KIND,'u':[3,0]}])]}
        campaign.queue_plan(out,state,spec,old_plan,'exploration')
        state['trials'][0]['status']='started'
        state['spent']['trials']=1
        campaign.atomic(out/'state.json',state)
        original={k:k.read_bytes() for k in out.rglob('*.json')}
        class PreviewDirector:
            def symbolic_direct(self,ctx):
                op=next(o for o in ctx['applicable_operators'] if o['bindings'].get('axis')=='h'
                        and o['bindings'].get('side')=='positive')
                plan={'cells':[dict(settings={},reps=1,blocks=[{'mass_class':'light'}],
                                    actions=[{'kind':geo.KIND,'u':[0,12.713]}])]}
                return dict(operator=op['op'],args=op['bindings'],premises=op['premises_satisfied_by'],
                            plan=plan,hypotheses_update=[],discriminates=[],expected='test',reason='preview')
        report=probe.read(ah.ah_current.preview_plan(out,'P3',provider=PreviewDirector()))
        self.assertEqual(report['available_new_trials'],spec['budget']['max_new_trials']-2)
        self.assertEqual(report['decision']['plan']['cells'][0]['actions'][0]['u'],[0,12.713])
        self.assertEqual({k:k.read_bytes() for k in original},original)
        ah.ah_current.upgrade(out,self.root/'unused')
        self.assertEqual(probe.read(out/'state.json')['pending'],state['pending'])
        self.assertEqual(probe.read(out/'spec.json'),probe.read(out/'instance.json'))
        self.assertNotIn('nominal_step',probe.read(out/'spec.json')['perturbation_variables']['h_offset_mm'])

    def test_declined_move_or_incomplete_receipt_not_edge(self):
        r=self.exercise([0,2])
        f,t=deepcopy(r['nodes'])
        t.pop('previous_node_id');t.pop('u_receipt')
        path=Path(r['move_receipts'][0])
        receipt=probe.read(path);receipt['result']='failed';campaign.atomic(path,receipt)
        ev.link(f,t,[path],self.axis)
        self.assertNotIn('u_receipt',t)
        self.assertIsNone(ah.recover_trial({},self.root,{}, {},interactive=False))

    def test_feedback_fault_after_move_requeues_same_trial_live_and_on_restart(self):
        fault='慢闭合指令与反馈相差超过2个刻度，暂停250毫秒后仍未连续恢复：指令=670，反馈=673；已停止闭合'
        def failed_probe(command,*args,**kw):
            if command=='-m' and args[args.index('--trial')+1]=='reprobe':
                path=self.root/'t/P2/reprobe/observation.json'
                campaign.atomic(path,{'state':'fault','error':fault})
                raise subprocess.CalledProcessError(1,['placement_probe','probe'])
            self.invoke(command,*args,**kw)
        with self.assertRaisesRegex(campaign.RetryTrial,'指令=670'):
            self.exercise([-3,0],invoke=failed_probe)
        self.assertEqual(self.moves,[('tcp',-3)])
        d=self.root/'t'
        req={'trial_id':'t','settings':{'a_offset_mm':0,'h_offset_mm':0},'block':{'mass_class':'light'},
             'actions':[{'kind':geo.KIND,'u':[-3,0]}]}
        trial={'trial_id':'t','request':req,'directory':str(d),'status':'started','started_ns':17}
        state={'pending':{'trial_ids':['t']},'trials':[trial],'spent':{'trials':1,'actions':0,'failures':0,'seconds':0}}
        for _ in range(2):
            campaign.collect_pending(self.root,state,self.spec,self.cfg,real=False,interactive=False)
        self.assertEqual(state['phase'],'ready_to_collect')
        self.assertEqual(trial['status'],'queued')
        self.assertEqual(state['spent'],{'trials':0,'actions':0,'failures':0,'seconds':0})
        self.assertEqual(probe.read(d/'request.json'),req)
        self.assertEqual(len(trial['incomplete_attempts']),1)
        archive=Path(trial['incomplete_attempts'][0]['archive'])
        receipt=probe.read(archive/'interruption.json')
        self.assertEqual(receipt['stage'],'reprobe')
        self.assertEqual(receipt['reached_move_receipts'],['action_move/tcp_move_fake.json'])
        self.assertTrue((archive/'P2/initial/observation.json').is_file())
        self.assertEqual([p.name for p in d.iterdir()],['request.json'])

    def test_reprobe_wbc_stamp_fault_retains_actual_move_and_request_on_retry(self):
        directory=self.root/'wbc'
        path=directory/'P3/reprobe/observation.json'
        fault='left WBC timestamp is stale or clocks disagree: -0.250206 s'
        campaign.atomic(path,{'state':'fault','error':fault})
        move=directory/'action_move/tcp_move_saved.json'
        campaign.atomic(move,dict(result='reached_by_measured_feedback',execution_start_mm=self.pose.tolist()))
        request={'trial_id':'wbc','actions':[{'kind':geo.KIND,'u':[0,11.5]}]}
        campaign.atomic(directory/'request.json',request)
        trial={'trial_id':'wbc','request':request,'directory':str(directory),'status':'started','started_ns':135}
        state={'pending':{'trial_ids':['wbc']},'trials':[trial],'spent':{'trials':1,'actions':0,'failures':0,'seconds':0}}
        campaign.collect_pending(self.root,state,self.spec,self.cfg,real=False,interactive=False)
        self.assertEqual((trial['status'],state['phase'],state['spent']['failures']),('queued','ready_to_collect',0))
        self.assertEqual(probe.read(directory/'request.json'),request)
        archive=Path(trial['incomplete_attempts'][0]['archive'])
        interruption=probe.read(archive/'interruption.json')
        self.assertEqual((interruption['stage'],interruption['reason']),('reprobe',fault))
        self.assertEqual(interruption['reached_move_receipts'],['action_move/tcp_move_saved.json'])
        self.assertEqual(probe.read(archive/'P3/reprobe/observation.json')['error'],fault)

    def test_read_only_clock_fault_queues_original_trial_on_restart(self):
        diagnostic=('Traceback (most recent call last):\n'
                    'bottle_scan.hardware_arm.ArmObservationError: Read-only arm connection timed out: '
                    'left WBC timestamp is stale or clocks disagree: -0.376609 s')
        calls=[]
        def clock_error(command,*args,**kwargs):
            calls.append(command)
            raise subprocess.CalledProcessError(1,[command,*args],stderr=diagnostic)
        with self.assertRaisesRegex(campaign.RetryTrial,'-0.376609'):
            self.exercise([0,3],invoke=clock_error)
        self.assertEqual(calls,['record_pose.py'])
        d=self.root/'t'
        req={'trial_id':'t'}
        trial=dict(trial_id='t',directory=str(d),request=req,status='started',started_ns=11)
        state=dict(pending={'trial_ids':['t']},trials=[trial],
                   spent=dict(trials=1,actions=0,failures=0,seconds=0))
        for _ in range(2):
            campaign.collect_pending(self.root,state,self.spec,self.cfg,real=False,interactive=False)
        self.assertEqual((state['phase'],trial['status'],state['spent']['trials']),('ready_to_collect','queued',0))
        self.assertEqual(len(trial['incomplete_attempts']),1)
        archive=Path(trial['incomplete_attempts'][0]['archive'])
        self.assertEqual(probe.read(archive/'pose_clock_failure.json')['stderr'],diagnostic)
        record=probe.read(archive/'interruption.json')
        self.assertEqual(record['stage'],'prepare_pose_0')
        self.assertEqual(record['move_receipts_with_execution_start'],[])
        self.assertFalse((archive/'P2').exists())
        self.assertEqual(probe.read(d/'request.json'),req)

    def test_unclassified_pose_read_error_stays_pending(self):
        def bad_reader(*args,**kwargs):
            raise subprocess.CalledProcessError(1,args,stderr='TypeError: unexpected argument')
        with self.assertRaises(subprocess.CalledProcessError):
            self.exercise([3,0],invoke=bad_reader)
        self.assertIsNone(ah.recover_trial({},self.root/'t',{}, {},interactive=False))

    def test_terminal_interpolation_fault_retries_from_prepare_receipt(self):
        d=self.root/'t'
        path=d/'prepare_move_0/tcp_move_fake.json'
        receipt=dict(error='机械臂反馈/控制故障：invalid rotation axis',result='failed',
                     finished_unix_ns=13,execution_start_mm=self.pose.tolist(),
                     software_hold='fresh_measured_pose_published')
        for override in ({'finished_unix_ns':None},{'software_hold':'unavailable'},
                         {'error':'机械臂反馈/控制故障：unknown'}):
            campaign.atomic(path,{**receipt,**override})
            self.assertIsNone(ah.recover_trial({},d,{}, {},interactive=False))
        campaign.atomic(path,receipt)
        with self.assertRaises(campaign.RetryTrial):
            ah.recover_trial({},d,{}, {},interactive=False)
        interruption=probe.read(d/'interruption.json')
        self.assertEqual(interruption['move_receipts_with_execution_start'],[str(path.relative_to(d))])
        self.assertEqual(interruption['reached_move_receipts'],[])

    def test_tech_retries_do_not_claim_unknown_or_force_limit(self):
        p=self.root/'t/P2/initial/observation.json'
        for row in ({'state':'started'}, {'state':'fault','error':'逐指法向力超过 F_max，已停止闭合并执行张开'}):
            campaign.atomic(p,row)
            self.assertIsNone(ah.recover_trial({},self.root/'t',{}, {},interactive=False))
        for message in ('Raw finger wrench time skew','WristFisheyeX5Camera timed out waiting for a new frame'):
            campaign.atomic(p,{'state':'fault','error':message})
            with self.assertRaises(campaign.RetryTrial):
                ah.recover_trial({},self.root/'t',{}, {},interactive=False)
        campaign.atomic(self.root/'t/completed_capture.json',{'trial_id':'t','status':'complete'})
        self.assertEqual(ah.recover_trial({},self.root/'t',{}, {},interactive=False)['status'],'complete')

    def test_v4_loop_queues_executes_rebuilds_and_pauses(self):
        out=self.root/'campaign'
        out.mkdir()
        path=out/'input_spec.json'
        campaign.atomic(path,self.spec)
        cfg={**self.cfg,'spec':str(path),'sources':[]}
        campaign.atomic(out/'input.json',cfg)
        campaign.atomic(out/'placement_basis.json',probe.read(self.root/'placement_basis.json'))
        campaign.initialize(out/'input.json',out)
        class Director:
            def symbolic_direct(self,ctx):
                if ctx['spent']['trials']:
                    op=next(op for op in ctx['applicable_operators'] if op['op']=='pause')
                    plan=None
                else:
                    op=next(op for op in ctx['applicable_operators'] if op['bindings'].get('axis')=='h'
                            and op['bindings'].get('side')=='negative')
                    plan={'cells':[{'settings':{},'reps':1,'blocks':[{'mass_class':'light'}],
                                   'actions':[{'kind':geo.KIND,'u':[0,-2]}]}]}
                return dict(operator=op['op'],args=op['bindings'],premises=op['premises_satisfied_by'],plan=plan,
                            hypotheses_update=[],discriminates=[],expected='check signed movement',reason='offline verification')
        def fake(req,directory,spec,cfg,**unused):
            return ah.execute_trial(req,directory,spec,cfg,real=True,interactive=True,ask=lambda _: 'MOVE',invoke=self.invoke)
        with patch.object(ah,'check_assets',return_value=(self.hw,self.identity)):
            state=campaign.run(out,provider=Director(),allow_agent=True,executor=fake)
        self.assertEqual(state['phase'],'pause')
        self.assertEqual(state['spent']['actions'],1)
        ctx=probe.read(out/'director_context.json')
        self.assertTrue(any(f['pred']=='direction_sampled' and f['args']==['h','negative'] for f in ctx['facts']))
        self.assertEqual(ctx['spent']['trials'],1)
