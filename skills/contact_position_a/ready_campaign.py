"""Create/resume a goal-specific campaign using the recorded exploration episodes."""
import argparse
import fcntl
from pathlib import Path
from bottle_scan.state_graph.campaign import atomic, initialize, ownership, read


def open_campaign(mode, source, out, config):
    source, out = Path(source).resolve(), Path(out).resolve()
    if source == out:
        raise ValueError('Use a separate campaign directory for the selected Ready goal')
    with ownership(out):
        if (out/'state.json').exists():
            saved = read(out/'config.json')
            if saved.get('ready_mode') != mode or saved.get('seed_campaign') != str(source):
                raise ValueError('Campaign belongs to a different goal or seed campaign')
            return
        cfg = read(config)
        if not (source/'state.json').is_file():raise ValueError(f'Seed campaign not found: {source}')
        with (source/'campaign.lock').open('r') as source_lock:
            fcntl.flock(source_lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
            state = read(source/'state.json')
            trials = [t for t in state['trials'] if t['status']=='complete' and t['request']['role']=='exploration' and not t['request']['post_freeze']]
            sources = []
            for t in trials:
                d = Path(t['directory'])
                if read(d/'result.json')['nodes']:
                    episode, = d.glob('episodes/episode_*/events.jsonl')
                    sources.append(str(episode))
            cfg.update(ready_mode=mode,seed_campaign=str(source),seed_trial_count=len(trials),sources=sources)
            cfg['execution']['outcomes'] = ('ForcePolicy L1 handoff from actual events; task outcome separately recorded.'
                                            if mode=='forcepolicy' else 'Fixed 5.0 SDK mean normal force; measured lift and hold from E1/E2.')
            atomic(out/'config.json',cfg)
            # Seed evidence carries neither learned labels nor prior B assessments.
            initialize(out/'config.json',out)
            atomic(out/'seed_origin.json',{'source':str(source),'revision':state['revision'],
                   'trials':[t['trial_id'] for t in trials],'event_logs':sources})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=['fixed5','forcepolicy'],required=True)
    p.add_argument('--source',required=True);p.add_argument('--out',required=True)
    p.add_argument('--config',default='config.state_graph.campaign_v4.json')
    a=p.parse_args();open_campaign(a.mode,a.source,a.out,a.config)


if __name__=='__main__':
    main()
