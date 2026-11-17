"""Independent transition-only v5 instance; downstream validation uses p3_loop."""
from pathlib import Path
from contextlib import contextmanager
import fcntl
from bottle_scan.state_graph import campaign
from bottle_scan.state_graph.common import independent
from bottle_scan.state_graph.v5_profile import configure,protect_output
from . import ah_v5_config, campaign_ah
from .ah_geometry import convert_legacy_row


@contextmanager
def read_source(directory):
    """Coordinate with the writer's flock without opening historical files for writing."""
    path=Path(directory)/'campaign.lock'
    if not path.exists():raise ValueError('Missing source campaign lock: '+str(path))
    with path.open('rb') as handle:
        fcntl.flock(handle,fcntl.LOCK_SH|fcntl.LOCK_NB)
        yield


def initialize(parent,out,placement):
    out,parent=Path(out).resolve(),Path(parent).resolve()
    if out==parent:raise ValueError('Use a new v5 directory')
    with campaign.ownership(out):
        protect_output(out,kind='rh_transitions_only')
        if (out/'state.json').exists():
            manifest=campaign.read(out/'seed_manifest.json')
            if manifest['placement']!=placement or manifest['source']!=str(parent):
                raise ValueError('Output already belongs to a different placement or data source')
            return
        with read_source(parent):
            saved=campaign.read(parent/'state.json')
            spec=campaign.read(parent/'spec.json')
            cfg=campaign.read(parent/'config.json')
            seeds=[r for r in campaign.all_rows(parent,saved) if not independent(r)]
        # This entry acquires probe transitions only. A matched consumer and B
        # validation are available via the separate P3 closed-loop instance.
        spec=configure(spec,{'loop':{'rounds':False,'termination_rules':False},
                             'consumer':{'dual_paired':False},'campaign':{'integrated_frontier':False}})
        ah_v5_config.configure(spec,cfg)
        spec['current_placement']=placement
        spec['adapters']['campaign_executor']='skills.contact_position_a.p3_trial:execute_trial'
        spec.pop('placement_geometry',None)
        for key in ('phase','sampling_preference','center_file'):
            cfg['execution'].pop(key,None)
        cfg['execution'].update(scope='transitions_only',placement=placement,
             allowed_operations=['collect_seed','collect_transitions','recalibrate_reference','pause'])
        cfg.pop('consumer',None)
        cfg.update(graph_version=5,spec=str(out/'instance.json'),sources=[])
        campaign.atomic(out/'seed_manifest.json',dict(source=str(parent),source_digest=campaign.digest(saved),placement=placement))
        campaign.atomic(out/'instance.json',spec)
        campaign.atomic(out/'input.json',cfg)
        campaign.initialize(out/'input.json',out,seed_rows=[convert_legacy_row(r,spec) for r in seeds])


def entry(opts):
    if opts.fixed_p2:
        raise ValueError('Original fixed-P2 geometry is available with --graph-version 4 --fixed-p2')
    placement=opts.from_current or 'P3'
    previous=campaign_ah.DEFAULT_OUT.with_name('campaign_white_ah_here_'+placement)
    if not (previous/'state.json').is_file():
        previous=campaign_ah.DEFAULT_OUT.with_name('campaign_white_ah_here_P3')
    out=opts.out or campaign_ah.DEFAULT_OUT.with_name('campaign_white_rh_'+placement+'_transitions_v5')
    if opts.preview_plan and opts.real:raise ValueError('Preview is offline')
    initialize(previous,out,placement)
    if opts.list_deferred:
        campaign_ah.show_deferred(campaign.read(out/'state.json'))
    elif opts.retry_deferred:
        print(campaign.release_deferred(out,opts.retry_deferred))
    else:
        state=campaign.run(out,real=opts.real and not opts.preview_plan,allow_agent=opts.allow_agent,
                           on_interruption=campaign_ah.offer_defer)
        print({k:state.get(k) for k in ('phase','spent','pending')})
    return 0
