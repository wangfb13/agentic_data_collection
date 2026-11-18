from copy import deepcopy
import numpy as np
import pytest
from bottle_scan.state_graph.v5_profile import configure
from bottle_scan.state_graph.v5_values import resolution,solve
from bottle_scan.state_graph.sink import threshold
from bottle_scan.state_graph.noise import fit_noise,pairwise_distances
from bottle_scan.state_graph.modalities.contact_geometry import calibrated_geometry
from bottle_scan.state_graph.v5_metric import fit_metric
from tests.fixtures.skill_toy.testing import data


def spec():
    return configure(data()[1])


def test_resolution_noise_and_smallest_action_have_different_neighbors():
    cfg=spec();cfg['graph'].update(resolution_min_edges=2,observable_axes_only=False)
    nodes=[{'node_id':n} for n in 'abc']
    edges=[dict(source='a',target='b',u=[3,0]),dict(source='b',target='c',u=[-3,0])]
    D=np.array([[0,.42,.03],[.42,0,.46],[.03,.46,0]])
    scale,radii=resolution(nodes,edges,D,cfg)
    assert scale['by_axis']['dx']['scale']==pytest.approx(.44)
    assert D[0,2]<radii['similar_radius']<D[0,1]
    assert scale['missing_axes']==['dy']
    assert all(v<=1 for v in radii.values())


def test_ready_lower_bound_and_legacy_switch():
    cfg=spec();c={'ready_p':.8}
    assert not threshold(3,3,c,cfg)[2] and threshold(30,30,c,cfg)[2]
    assert threshold(3,3,c,{})[2]


def edge(s,t,u,kind='observed'):
    return dict(source=s,target=t,edge_id=s+t,kind=kind,action={'kind':'translation_xy','u':u},u=u,cost=1)


def test_success_prefers_high_probability_ready_and_borrows_without_teleporting():
    cfg=spec();cfg['runtime']['max_steps']=2
    ids=list('abcdefg')
    E=[edge('b','c',[1,0]),edge('d','e',[0,1]),edge('d','f',[0,-1]),edge('g','f',[-1,0],'inferred')]
    D=np.full((len(ids),len(ids)),3.)
    np.fill_diagonal(D,0)
    for i,j in [(0,1),(2,3),(5,6)]:D[i,j]=D[j,i]=.1
    probs={k:{'ready':k in ('e','f'),'p_hat':.95 if k=='f' else .81} for k in ids}
    values,path=solve(ids,E,probs,D,.2,cfg,False)
    assert [v['edge_id'] for v in path['a']]==['bc','df']
    assert [v['action_source'] for v in path['a']]==['borrowed','borrowed']
    assert values['a']==pytest.approx(.95*((1-cfg['graph']['step_penalty'])*(1-cfg['graph']['similar_penalty']))**2)
    assert values['g']==0
    _,hyp=solve(ids,E,probs,D,.2,cfg,True)
    assert hyp['g'][0]['action_source']=='inferred'


def test_inferred_only_graph_proposals_never_become_training_labels(monkeypatch):
    from bottle_scan.state_graph.graph import build_graph
    from bottle_scan.state_graph.labels import generate_labels
    rows=[deepcopy(r) for r in data()[0][:2]];cfg=spec()
    for r in rows:r['previous_node_id']=None
    a,b=[r['node_id'] for r in rows]
    p={c['id']:{a:{'ready':False,'p_hat':.1},b:{'ready':True,'p_hat':.95}} for c in cfg['post_conditions']}
    monkeypatch.setattr('bottle_scan.state_graph.graph.estimate_sinks',lambda *args,**kw:(p,[]))
    monkeypatch.setattr('bottle_scan.state_graph.dynamics.inferred_edges',lambda *a_,**kw:[edge(a,b,[1,0],'inferred')])
    graph=build_graph(rows,data()[2],cfg,{'status':'trained'})
    labels=generate_labels(graph,rows,cfg)
    assert not any(x['status']=='recoverable' for x in labels['nodes'])
    assert all(graph['model_reachability'][c['id']][a]['status']=='model_recoverable' for c in cfg['post_conditions'])


def test_calibrated_band_translation_deskews_finger_orientation():
    pose=np.eye(4)
    cal={'calibration_id':'x','sensor_version':'sx','axes_tcp':{'a':[1,0,0]},
         'fingers':{s:{'tcp_mm_per_px':[[1,0],[0,0],[0,1]],'origin_tcp_mm':[0,0,0]} for s in ('left','right')}}
    mask=np.zeros((30,30),bool);mask[7:20,11:14]=True
    win={'finger_masks':dict(left=mask,right=mask),'provenance':{'sensor_version':'sx','measured_pose':pose}}
    a=np.array(calibrated_geometry(win,cal))
    win['finger_masks']={s:np.roll(mask,2,axis=1) for s in ('left','right')}
    b=np.array(calibrated_geometry(win,cal))
    assert np.allclose(b-a,[2,0,0,0,0,0,0]*2)
    # Rotate the sensor and mask by a quarter turn in opposite senses.
    R=np.array([[0,0,1],[0,1,0],[-1,0,0]])
    pose[:3,:3]=R
    cal['axes_tcp']['a']=[0,0,1]
    for c in cal['fingers'].values():c['origin_tcp_mm']=[-29,0,0]
    win['finger_masks']={s:np.rot90(mask,-1) for s in ('left','right')}
    assert np.allclose(calibrated_geometry(win,cal),a)


def test_all_spatial_readouts_preserve_shift():
    torch=pytest.importorskip('torch')
    from bottle_scan.state_graph.modalities.tactile_embedding import spatial_readout
    x=torch.zeros(2,3,8,8);x[0,:,1:3,1:3]=8;x[1,:,5:7,5:7]=8
    tokens=torch.nn.functional.avg_pool2d(x,2).flatten(2).transpose(1,2)
    def change(mode):
        z=spatial_readout(x,tokens,2,mode,(2,2))
        return float(torch.linalg.vector_norm(z[0]-z[1]))
    assert change('mean')==0 and min(map(change,('soft_argmax','attention','grid')))>.5


def test_metric_uses_only_measured_displacements_and_repeats():
    cfg=spec();cfg['observation']['metric'].update(input_modality='embed',a_axis_tcp=[1,0,0])
    rows=[];rng=np.random.default_rng(11)
    for i in range(28):
        b=np.eye(4);a=b.copy();a[:3,3]=[i%3+1,0,(i%4)-2]
        for j,t in enumerate([b,a]):
            y=t[:3,3][[0,2]]
            rows.append(dict(session='s',task_id='t',group_id=str(i),node_id=f'{i}:{j}',window_index=j,window_valid=True,
                             repeat_key='k',previous_node_id=f'{i}:0' if j else None,u_receipt={'applied':True} if j else None,
                             obs={'embed':list(y+rng.normal(0,.0001,2))},provenance={'measured_pose':t.tolist()}))
    result=fit_metric(rows,cfg)
    assert result['status']=='ready' and result['action_error_mm']<.01 and result['repeat_distance_mm']<.01
    for r in rows:r['post']={'made_up_label':{'success':True}}
    assert fit_metric(rows,cfg)==result


def test_joint_noise_without_repeat_support_is_excluded_and_neighbor_fraction_stable():
    rng=np.random.default_rng(1)
    cfg=spec();cfg['observation'].update(modalities={'x':{},'jitter':{},'unfit':{}},tac_metric=False)
    rows=[];cfg['noise']['pca_dim']=256
    for i in range(120):
        rows.append(dict(node_id=str(i),group_id=str(i),task_id='t',session='s',repeat_key='all',window_index=0,
                         window_valid=True,obs={'x':[rng.normal()],'jitter':[rng.normal()],'unfit':rng.normal(size=80).tolist()}))
    noise=fit_noise(rows,cfg);D=pairwise_distances(rows,noise)
    assert noise['active_modalities']==['x','jitter'] and noise['modalities']['unfit']['status']=='insufficient_repeats'
    del cfg['observation']['modalities']['jitter'];D0=pairwise_distances(rows,fit_noise(rows,cfg))
    assert abs(np.mean(D<=1)-np.mean(D0<=1))<cfg['noise']['max_neighbor_fraction_change']


def test_representation_ranking_needs_all_axes_scenes_and_outcomes():
    from bottle_scan.state_graph.representation_eval import evaluate,director_evidence
    rows,_,_=data();cfg=spec();cfg['observation']['representations'].update(candidates=['legacy_geometry','force'],bootstrap_reps=10)
    report=evaluate(deepcopy(rows[:12]),cfg)
    assert not report['selectable'] and not director_evidence(report,cfg)[1]
    for m in report['scenes']['force'].values():
        assert {'resolution_noise_ratio','mae_r','mae_h','angle_distance','unsupported_fraction','auc_centered'}<=set(m)


def test_same_model_grouped_action_ablation_is_reported():
    from bottle_scan.state_graph.dynamics import fit_dynamics,evaluate_dynamics
    rows,base,_=data();cfg=spec();cfg['dynamics'].update(ensemble=2,max_iter=25)
    noise=fit_noise(rows,cfg);model=fit_dynamics(rows,noise,cfg)
    assert np.isfinite(model['action_ablation']['relative_gain'])
    evaluation=evaluate_dynamics(model,rows,cfg)
    assert np.isfinite(evaluation['action_gain']) and 'same_model_no_action' in evaluation['samples'][0]['baseline_absolute_error']


def test_ranking_requires_all_scenes_noninferior_and_uses_group_pairs():
    from bottle_scan.state_graph.representation_eval import compare,director_evidence
    cfg=spec();settings=cfg['observation']['representations'];settings['bootstrap_reps']=30
    def make(boost):
        return [dict(id=f'{i}:{y}',group=str(i),distance_action=[2+boost],distance_repeat=[1],
                     angle_distance=[],unsupported=None,errors={'r':1-.3*boost,'h':1-.3*boost},
                     outcomes={'c':[y,(.1 if y==0 else .8)]}) for i in range(10) for y in (0,1)]
    a=make(1);b=make(0)
    stats=compare(a,b,['c'],settings)
    assert stats['eligible'] and stats['improved']
    rank={'schema':'representation_ranking_v5','current':'legacy_geometry','source_id':'x','evidence':['archive-test'],
          'comparisons':{'force':dict.fromkeys(settings['required_scenes'],stats)},'selectable':['force']}
    facts,ok=director_evidence(rank,cfg);assert ok==['force'] and facts[0]['metrics']['selectable']==ok
    del rank['comparisons']['force']['different_angles']
    assert director_evidence(rank,cfg)[1]==[]


def test_metric_used_in_noise_graph_learning_and_live_projection():
    from bottle_scan.state_graph.noise import feature_vector,raw
    from bottle_scan.state_graph.representation_apply import apply_representation
    cfg=spec();cfg['observation'].update(modalities={'embed':{'encoder_features':True}},tac_embed_readout='grid')
    cfg['observation']['metric'].update(input_modality='embed',a_axis_tcp=[1,0,0])
    cfg['tactile_encoder']={'artifact':'frozen-test'}
    rng=np.random.default_rng(6);rows=[]
    for i in range(32):
        delta=np.array([1+(i%3), (i%4)-2.])
        for j in (0,1):
            p=np.eye(4);p[:3,3]=[delta[0]*j,0,delta[1]*j];v=(delta*j+rng.normal(0,.02,2)).tolist()
            rows.append(dict(node_id=f'{i}:{j}',window_index=j,task_id='t',session='s',group_id=str(i),
                             previous_node_id=f'{i}:0' if j else None,window_valid=True,
                             obs={'embed':v},representations={'grid':v},representation_versions={'grid':'frozen_1'},
                             extractor_versions={'embed':'frozen_1'},repeat_key='center',risk={},
                             u_receipt={'applied':True} if j else None,u_executed=[1,0] if j else None,
                             provenance={'measured_pose':p.tolist()}))
    selected,cfg=apply_representation(rows,cfg,'tac_metric')
    noise=fit_noise(selected,cfg)
    assert noise['active_modalities']==['tac_metric'] and 'metric_resolution' in noise
    assert noise['modalities']['embed']['status']=='projected_only'
    live={'obs':selected[2]['obs'],'window_valid':True,'task_id':'t'}
    assert np.allclose(raw(live,'tac_metric',noise),raw(selected[2],'tac_metric',noise))
    assert len(feature_vector(live,noise))==2
