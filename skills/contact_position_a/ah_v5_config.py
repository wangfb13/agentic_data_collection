"""Instance configuration for a new r/h campaign. Existing campaigns retain their own spec."""
from copy import deepcopy
from pathlib import Path
import json
from bottle_scan.state_graph.common import digest
from .ah_geometry import KIND, RH_KIND, RH_COORDS, horizontal


def configure(spec, cfg=None):
    preferences=json.loads(Path(__file__).with_name('rh_sampling_priors.json').read_text())
    if preferences and not horizontal(spec):
        preferences['a']=preferences.pop('r')
    if preferences:
        prior=spec.setdefault('priors',{})
        prior.setdefault('placement_sampling',preferences)
    # Use one documented source for the director. Remove legacy prose hints from its shared context.
    if cfg is not None:
        cfg['execution'].pop('sampling_preference',None)
        cfg['execution'].pop('phase',None)
    observation=spec.get('observation',{})
    from bottle_scan.state_graph.calibrations import missing
    if observation.get('tac_version')=='calibrated_rh' and 'observation.tac_calibration' in missing(spec):
        observation.update(tac_version='legacy_pixel',tac_version_requested='calibrated_rh',
                           tac_calibration_status='unavailable: finger-plane metric calibration absent')
    if observation.get('tac_version')=='calibrated_rh' and 'tac' in observation.get('modalities',{}):
        observation['modalities']['tac']['version']='calibrated_rh:'+digest(observation['tac_calibration'])
    if horizontal(spec):
        geom=spec['geometry']
        step=geom['max_step_mm']
        variables=spec['perturbation_variables']
        base=variables.pop('a_offset_mm',variables.get('r_offset_mm',{}))
        cross=spec.get('facts',{}).get('boundary_cross_tolerance_mm')
        if isinstance(cross,dict) and 'a_offset_mm' in cross:
            cross.setdefault('r_offset_mm',cross.pop('a_offset_mm'))
        for key in RH_COORDS:
            var=deepcopy(base if key==RH_COORDS[0] else variables[key])
            lo,hi=geom['placement_envelopes_mm'][key]
            var.update(envelope=[lo,hi],initial_range=[lo,hi],executor='placement.'+key)
            variables[key]=var
        space=spec['task']['action_space']
        space.update(id=RH_KIND,names=['delta_horizontal_r_mm','delta_base_h_mm'],max_step=step,
                     frame='[horizontal(current TCP a), base_link Z]')
        space.pop('candidate_actions',None)
        space.pop('candidate_levels',None)
        space['continuous_displacement_variables']=list(RH_COORDS)
        cap=spec['capabilities'].pop(KIND,spec['capabilities'].get(RH_KIND,{}))
        spec['capabilities'][RH_KIND]={**cap,'max_action':step}
        spec['safety_envelope']['max_action']=step
        if cfg is not None:
            cfg['execution'].update(executable_actions=[RH_KIND],executable_variables=[*RH_COORDS,'mass_class'])
    if spec.get('facts',{}).get('measured_placement_coordinates') and cfg is not None:
        from .ah_geometry import coord_names
        cfg['execution']['executable_variables']=[*coord_names(spec),'mass_class']
    if 'geometry' in spec:
        spec['geometry'].setdefault('axes_tcp',{'a':[0.,0.,1.]})  # verified by check_assets before physical collection
    options=spec.get('consumer',{})
    if options.get('dual_paired') and not any(c['id']=='fixed_force_lift' for c in spec['post_conditions']):
        prev=next(c for c in spec['post_conditions'] if c['id']==spec['active_post_condition'])
        fp=deepcopy(prev)
        fp.update(id='forcepolicy_lift',evaluator='skills.contact_position_a.post:live_lift')
        fixed=deepcopy(fp)
        fixed.update(id='fixed_force_lift')
        params={k:options[k] for k in ('fixed_force_sdk','force_tolerance_sdk','stable_force_s','lift_mm','hold_s')}
        params['closure_floor']=options['fixed_closure_floor']
        fixed['consumer']={'controller':'fixed_force','version':'p3_fixed_'+digest(params)[:20],'params':params}
        spec['post_conditions']=[fixed,fp]
        spec['active_post_condition']=fp['id']
    return spec
