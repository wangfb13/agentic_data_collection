"""Precompute optional tactile embeddings from saved frames; no robot imports."""
import argparse
from pathlib import Path
import json
import time
from .skill_interface import load_spec,load_callable
from .common import write
from .extract_nodes import extract_nodes
from .modalities.tactile_embedding import _ENCODERS,embedding


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True);p.add_argument('--device',default='cuda');p.add_argument('--limit',type=int)
    a=p.parse_args(argv);config=json.loads(Path(a.config).read_text());spec=load_spec(config['spec'])
    spec['tactile_encoder']={**spec['tactile_encoder'],'device':a.device,'cache_only':False}
    start=time.monotonic();count=0
    iterate=load_callable(spec['adapters']['trials']);observe=load_callable(spec['observation']['window']['extractor'])
    for trial in iterate(config.get('sources',spec['data']['sources']),spec):
        for window in observe(trial,spec):
            if not all(window.get('preconditions',{}).get(k) is True for k in spec['observation']['window'].get('requires',[])):continue
            embedding(window,spec);count+=1
            if count%25==0:print(json.dumps({'windows':count,'elapsed_s':round(time.monotonic()-start,1)}),flush=True)
            if a.limit and count>=a.limit:break
        if a.limit and count>=a.limit:break
    evidence={'schema':'tactile_precompute_v1','windows':count,'seconds':time.monotonic()-start,'limited_run':a.limit is not None,
              'encoders':[{'version':e.version,'cache_hits':e.cache_hits,'computed_windows':e.computed_windows,'recipe':e.recipe} for e in _ENCODERS.values()],
              'model_training_started':False,'robot_started':False}
    output=Path(config['out']);output.mkdir(parents=True,exist_ok=True)
    write(output/('precompute_smoke.json' if a.limit else 'precompute.json'),evidence)
    print(json.dumps({k:v for k,v in evidence.items() if k!='encoders'},ensure_ascii=False),flush=True)
    return 0

if __name__=='__main__':raise SystemExit(main())
