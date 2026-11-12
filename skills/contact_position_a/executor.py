"""Translate a reviewed generic plan to existing collector fields; no execution."""
from copy import deepcopy


def translate_plan(plan,spec):
    translated=[]
    for cell in plan['cells']:
        settings=cell['settings'];config={}
        for name,value in settings.items():
            definition=spec['perturbation_variables'][name]
            destination=definition.get('executor')
            if destination:
                parts=destination.split('.');node=config
                for part in parts[:-1]:node=node.setdefault(part,{})
                node[parts[-1]]=value
        for block in cell.get('blocks',[{}]):
            trial=deepcopy(config);trial.setdefault('trial',{}).update(bottle=block.get('mass_class'),reps=cell['reps'])
            trial['record_attributes']=dict(block)
            requested=next(c for c in spec['post_conditions'] if c['id']==plan.get('post_condition',spec['active_post_condition']))
            floor=settings.get('closure_floor',requested['consumer']['params'].get('closure_floor'))
            matched=next(c for c in spec['post_conditions'] if c['consumer']['controller']==requested['consumer']['controller'] and c['consumer']['params'].get('closure_floor')==floor)
            trial['post_condition']=matched['id'];trial['consumer']=matched['consumer']
            translated.append(trial)
    return {'schema':'existing_collector_plan_v3','trials':translated,'executed':False}
