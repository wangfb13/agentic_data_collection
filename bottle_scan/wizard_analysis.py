"""Time plot of zero-offset PROBE centroids. No hardware IO."""
import csv
import json
from pathlib import Path
import numpy as np


def write_zero_drift_plot(session,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    session=Path(session);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    meta=json.loads((session/'session_meta.json').read_text())
    state=meta.get('zero_drift',{});samples=state.get('samples',[])
    threshold=state.get('threshold_mm',1.5)
    rows=[];fig,axes=plt.subplots(2,1,figsize=(9,6),sharex=True)
    for side,ax in zip(('left','right'),axes):
        stamps=np.asarray([s['monotonic_ns'] for s in samples],dtype=np.float64)
        time=(stamps-stamps[0])/1e9 if len(stamps) else stamps
        values=np.asarray([s['centroid_al_mm'][side] for s in samples],float)
        if len(values):
            baseline=np.asarray(state.get('baseline',{}).get(side,values[:3].mean(axis=0)))
            for i,axis in enumerate(('a','l')):
                residual=values[:,i]-baseline[i]
                ax.plot(time,residual,'o-',label=axis)
                for sample,t,v in zip(samples,time,residual):
                    rows.append({'trial_id':sample['trial_id'],'monotonic_ns':sample['monotonic_ns'],
                        'elapsed_s':float(t),'finger':side,'axis':axis,'centroid_equivalent_mm':float(sample['centroid_al_mm'][side][i]),
                        'residual_equivalent_mm':float(v),'baseline_ready':len(samples)>=3,
                        'baseline_member':sample['trial_id'] in state.get('baseline_trial_ids',[]),'alert':sample.get('alert',False)})
            ax.legend()
        else:ax.text(.5,.5,'No valid zero-offset PROBE samples',ha='center',va='center',transform=ax.transAxes)
        ax.axhline(threshold,color='r',ls='--');ax.axhline(-threshold,color='r',ls='--')
        ax.set_ylabel(side+' residual (equiv. mm)');ax.grid(alpha=.25)
    axes[-1].set_xlabel('Time since first zero-offset PROBE (s)')
    fig.suptitle('Zero-offset centroid drift'+(' [SIMULATION ONLY]' if meta.get('simulated') else ''))
    fig.tight_layout();plot=output/'zero_offset_centroid_drift.png';fig.savefig(plot,dpi=150);plt.close(fig)
    with (output/'zero_offset_centroid_drift.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]) if rows else ['trial_id','finger','axis'])
        writer.writeheader();writer.writerows(rows)
    return {'plot':str(plot.resolve()),'samples':len(samples),'threshold_mm':threshold}
