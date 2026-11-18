"""Continue the P3 free-start experiment with measured ForcePolicy endpoints."""
import argparse
from pathlib import Path

from bottle_scan.state_graph import campaign
from bottle_scan.state_graph.common import independent
from bottle_scan.state_graph.facts import fact
from . import campaign_ah, ah_evidence, p3_force
from .ah_geometry import validate_cell

PARENT = Path('outputs/state_graph/campaign_white_ah_here_P3')
OUTPUT = PARENT.with_name(PARENT.name + '_forcepolicy')
PROTOCOL = 'p3_ah_forcepolicy_lift_v1'


def initialize(parent, out, *, extra_trials=60, consumer=None, version=4, overrides=None, ranking_report=None):
    parent, out = Path(parent).resolve(), Path(out).resolve()
    if out == parent:
        raise ValueError('The seed campaign must remain separate')
    with campaign.ownership(out):
        if version==5:
            from bottle_scan.state_graph.v5_profile import protect_output
            protect_output(out,kind='P3-rounds')
        if (out/'state.json').exists():
            manifest = campaign.read(out/'continuation.json')
            if version==4 and (out/'v5_output.json').exists():raise ValueError('This directory contains a v5 continuation')
            if version==5 and overrides:
                from bottle_scan.state_graph.v5_profile import merge
                declared=manifest.get('graph_overrides',campaign.read(out/'spec.json'))
                if merge(declared,overrides)!=declared:
                    raise ValueError('The requested overrides differ from the frozen campaign; choose a new directory')
            if version==5 and ranking_report and campaign.read(ranking_report)!=campaign.read(out/'spec.json')['observation'].get('ranking_report'):
                raise ValueError('Representation evidence differs from the frozen campaign; choose a new directory')
            if manifest['source'] != str(parent) or manifest['protocol'] != PROTOCOL:
                raise ValueError('Continuation directory belongs to another experiment')
            return
        from .ah_transition_v5 import read_source
        with (read_source(parent) if version==5 else campaign.ownership(parent)):
            state = campaign.read(parent/'state.json')
            spec = campaign.read(parent/'spec.json')
            cfg = campaign.read(parent/'config.json')
            if spec.get('current_placement') != 'P3' or state['pending'] or any(
                    t['status'] not in ('complete','superseded') for t in state['trials']):
                raise ValueError('Source P3 has unsettled trials or another start mode')
            rows = [r for r in campaign.all_rows(parent,state) if not independent(r)]
            origins = [{k:t[k] for k in ('trial_id','directory','outcome_status')} for t in state['trials'] if t['status']=='complete']
        path = campaign.read('config.state_graph.campaign_v4.json')['execution']['environment']['RIGHT_FORCE_POLICY_PATH']
        physics = consumer or p3_force.identity(path)
        condition = next(c for c in spec['post_conditions'] if c['id'] == spec['active_post_condition'])
        condition['consumer'].update(version=physics['version'])
        condition['consumer']['params'].update(lift_mm=40, hold_s=2)
        spec['adapters'].update(campaign_executor='skills.contact_position_a.p3_trial:execute_trial',
            campaign_menu='skills.contact_position_a.p3_loop:menu',
            after_independent_gap='skills.contact_position_a.p3_loop:after_independent_gap',
            exploration_context='skills.contact_position_a.p3_loop:exploration_context',
            validate_collection_cell='skills.contact_position_a.p3_loop:validate_plan')
        count = spec['acceptance']['independent_B']['min_trials']
        if extra_trials < count + spec['acceptance']['development_B']['min_trials'] + 1:
            raise ValueError('Continuation budget cannot fit both B sets and A')
        spec['budget'].update(max_new_trials=extra_trials,max_batch_trials=count)
        cfg.update(spec=str(out/'instance.json'), sources=[], consumer=physics,
                   independent_plan={'cluster_classes':['Ready','NeedAction'],'cells':[
                       {'settings':{},'blocks':[{'mass_class':'light'},{'mass_class':'heavy'}],
                        'reps':count//2,'actions':[]}]})
        if len(campaign.expand_plan(cfg['independent_plan'],spec)) != count:
            raise ValueError('Independent plan requires balanced mass blocks')
        cfg['execution']['phase'] = ('A：每条有效末次轻触交给指定 ForcePolicy，实测提起保持。'
            '先建立不纠正的对照，再用动作与结果证据补采；h 优先向上。'
            'B：计划不填动作，冻结的位置模型选不动或单轴纠正。')
        if version==5:
            from bottle_scan.state_graph.v5_profile import configure
            from . import ah_v5_config, ah_geometry
            from bottle_scan.state_graph.skill_interface import task_id
            spec=configure(spec,overrides)
            ah_v5_config.configure(spec,cfg)
            if ranking_report:
                ranking=campaign.read(ranking_report)
                permitted={r['node_id'] for r in rows}
                if ranking.get('schema')!='representation_ranking_v5' or not ranking.get('evidence') or not set(ranking['evidence'])<=permitted:
                    raise ValueError('Representation report must use eligible source-campaign windows')
                spec['observation']['ranking_report']=ranking
            cfg['execution'].pop('phase',None)
            cfg['execution'].pop('sampling_preference',None)
            cfg.update(graph_version=5,development_plan=cfg['independent_plan'])
            if spec['geometry']['horizontal_basis']:
                validation=campaign.read(Path(__file__).with_name('rh_validation_plan.json'))
                for cell in validation['cells']:
                    validate_cell(cell,spec,'validate_B',{})
                cfg.update(development_plan=validation,independent_plan=validation)
            rows=[ah_geometry.convert_legacy_row(row,spec) for row in rows]
            # A seed has no new consumer outcome until it has actually been evaluated.
            for row in rows:
                row['task_id']=task_id(spec)
                for post in spec['post_conditions']:
                    row.setdefault('post',{}).setdefault(post['id'],dict(evaluated=False,success=None,evidence=[]))
        else:
            cfg['execution']['sampling_preference']['instruction'] += cfg['execution']['phase']
        campaign.atomic(out/'continuation.json',dict(protocol=PROTOCOL,source=str(parent),
                      trials=origins,source_spent=state['spent'],seed_nodes=len(rows),
                      source_state_digest=campaign.digest(state),new_budget=extra_trials,**({'graph_overrides':overrides or {}} if version==5 else {})))
        campaign.atomic(out/'instance.json',spec)
        campaign.atomic(out/'input.json',cfg)
        campaign.initialize(out/'input.json',out,seed_rows=rows)


def validate_plan(cell,spec,op,args):
    validate_cell(cell,spec,op,args)
    if op in ('collect_outcomes','validate_B') and cell.get('actions'):
        raise ValueError('Direct baseline and learned B cannot contain scripted movements')


def exploration_context(rows,graph,facts,spec):
    facts, options = ah_evidence.exploration_context(rows,graph,facts,spec)
    cid = spec['active_post_condition']
    direct = [r for r in rows if r.get('session') == 'P3' and
              r['provenance'].get('direct_A') and r['post'][cid]['evaluated']]
    summary = fact('p3_direct_consumer_coverage',[cid],[r['node_id'] for r in direct],
                   {'success':sum(r['post'][cid]['success'] for r in direct),
                    'failure':sum(not r['post'][cid]['success'] for r in direct)})
    summary['metrics']['by_mass']={mass:{'success':sum(r['post'][cid]['success'] is True for r in direct if r['attrs']['mass_class']==mass),
        'failure':sum(r['post'][cid]['success'] is False for r in direct if r['attrs']['mass_class']==mass)}
        for mass in ('light','heavy')}
    facts.append(summary)
    axes=('r','h') if spec.get('geometry',{}).get('horizontal_basis') else ('a','h')
    for k,axis in enumerate(axes):
        outcomes=[r for r in rows if r.get('session')=='P3' and (r.get('u_receipt') or {}).get('commanded_axis')==axis
                  and r['post'][cid]['evaluated'] and r.get('u_executed')]
        facts.append(fact('p3_measured_action_outcomes',[cid,axis],[r['node_id'] for r in outcomes],
                          {'samples':[dict(node=r['node_id'],mass=r['attrs']['mass_class'],mm=r['u_executed'][k],
                                           success=r['post'][cid]['success']) for r in outcomes]}))
    options.append(dict(op='collect_outcomes',bindings={'post_condition':cid},premises_satisfied_by=[summary['id']]))
    return facts, options


def after_independent_gap(out,state,spec,result):
    frozen=state['frozen']
    dest=Path(out)/'confirmations'/frozen['model_id']
    campaign.atomic(dest/'acceptance.json',result)
    campaign.atomic(dest/'frozen.json',frozen)
    left=spec['budget']['max_new_trials']-state['spent']['trials']
    minimum=sum(spec['acceptance'][r+'_B']['min_trials'] for r in ('development','independent'))
    if left<minimum+1:
        return False
    state.setdefault('retired_candidates',[]).append(frozen['model_id'])
    state.pop('candidate_to_freeze',None)
    state.update(frozen=None,model=None,last_batch={},phase='analyze',needs_fresh_A=True)
    campaign.atomic(Path(out)/'state.json',state)
    return True


def menu(ctx, options, state, training):
    generic = ctx['applicable_operators']
    if state['frozen']:
        ctx['applicable_operators'] = [op for op in generic if op['op']==
             ('complete' if any(f['pred']=='indep_B_pass' for f in ctx['facts']) else 'validate_B')]
        return
    reserve = ctx['spec']['acceptance']['independent_B']['min_trials']
    ctx['budget_left'] = max(0,ctx['budget_left']-reserve)
    ctx['spec']['budget']['max_batch_trials'] = 4
    direct = next(f for f in ctx['facts'] if f['pred']=='p3_direct_consumer_coverage')
    gap = next((f for f in ctx['facts'] if f['pred']=='dev_B_gap'),None)
    if state.get('needs_fresh_A') and (state.get('last_batch') or {}).get('role')=='exploration':
        state['needs_fresh_A']=False
    retired=(state.get('model') or {}).get('model_id') in state.get('retired_candidates',[])
    if (state.get('needs_fresh_A') or retired) and ctx['budget_left']:
        choices=[x for x in options if x['op'] in ('collect_outcomes','collect_transitions')]
    elif any(x['op']=='freeze' for x in generic):
        choices=[x for x in generic if x['op']=='freeze']
    elif not ctx['budget_left']:
        choices=[x for x in generic if x['op']=='pause']
    elif not direct['evidence']:
        choices=[x for x in options if x['op']=='collect_outcomes']
    elif (training['active_model_trained'] and any(x['op']=='validate_B' for x in generic)
          and (not gap or gap['metrics']['trials'] < ctx['spec']['acceptance']['development_B']['min_trials'])):
        choices=[x for x in generic if x['op']=='validate_B']
    else:
        choices=[x for x in options if x['op'] in ('collect_transitions','collect_outcomes')]
    ctx['applicable_operators']=choices


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--graph-version',type=int, choices=(4,5),default=5)
    p.add_argument('--out',type=Path)
    p.add_argument('--director-mode',choices=('llm','fixed_priority'))
    p.add_argument('--ranking-report',type=Path)
    p.add_argument('--overrides',type=Path,help='JSON v5 spec overrides for a fresh campaign, including measured calibrations')
    p.add_argument('--real',action='store_true')
    p.add_argument('--allow-agent',action='store_true')
    mode=p.add_mutually_exclusive_group()
    mode.add_argument('--check',action='store_true')
    mode.add_argument('--list-deferred',action='store_true')
    mode.add_argument('--retry-deferred',nargs='+',metavar='ID')
    mode.add_argument('--resume-interrupted',action='store_true',help='确认已停臂、瓶已放稳且夹爪张开空载：归档已知技术中断，保留费用，重排原试次后继续')
    args=p.parse_args(argv)
    out=OUTPUT.with_name(OUTPUT.name+'_v5') if args.graph_version==5 else OUTPUT
    out=args.out or out
    if (args.director_mode or args.overrides or args.ranking_report) and args.graph_version==4:
        p.error('Director, calibration and ranking overrides require v5')
    overrides=campaign.read(args.overrides) if args.overrides else {}
    if args.director_mode:overrides.setdefault('director',{})['mode']=args.director_mode
    initialize(PARENT,out,version=args.graph_version,ranking_report=args.ranking_report,overrides=overrides)
    cfg=campaign.read(out/'config.json')
    p3_force.load(cfg['consumer'])  # CPU-only preflight, before any robot connection
    if args.check:
        print(campaign.read(out/'continuation.json'))
    elif args.list_deferred:
        campaign_ah.show_deferred(campaign.read(out/'state.json'))
    elif args.retry_deferred:
        print(campaign.release_deferred(out,args.retry_deferred))
    else:
        if args.resume_interrupted:
            from .p3_resume import resume_interrupted
            print('[graph-loop] 恢复原试次：',resume_interrupted(out,scene_reset_confirmed=True))
        options={}
        if args.graph_version==5:
            policy=campaign.read(out/'spec.json')
            if policy.get('loop',{}).get('run_through_budget'):
                # Six stages per round, with room for a final round and independent B.
                # The trial/dispatch budgets and all explicit pause/stop states still apply.
                options['max_decisions']=6*(policy['budget']['max_new_trials']+1)+1
        state=campaign.run(out,real=args.real,allow_agent=args.allow_agent,on_interruption=campaign_ah.offer_defer,**options)
        print({k:state[k] for k in ('phase','spent','complete','pending')})
        campaign_ah.show_deferred(state)


if __name__=='__main__':
    main()

