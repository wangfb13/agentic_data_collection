"""Standalone figures for grouped diagnostics; no model fitting or data edits."""
from pathlib import Path
import textwrap
import numpy as np


def render_plots(reports,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
    names=[r['task_id'].split('|')[0] for r in reports]
    values=[r['V1']['fraction_within_noise'] for r in reports]
    axes[0].bar(names,[v or 0 for v in values],color=['#176d8f','#ea9750'][:len(names)])
    axes[0].set_ylim(0,1.15);axes[0].set_ylabel('Fraction within fitted repeat noise')
    axes[0].set_title('V1: reference repeatability (in sample)')
    for i,(value,r) in enumerate(zip(values,reports)):
        axes[0].text(i,(value or 0)+.03,f'{value:.1%}\nn={r["V1"]["pairs"]} pairs' if value is not None else 'Unavailable',ha='center')
    modalities=sorted(set(k for r in reports for k in r['V2']['probes']))
    coordinates=sorted(set(k for r in reports for p in r['V2']['probes'].values() for k,v in p['coordinates'].items()
                           if v['predictions'] and np.std([s['actual'] for s in v['predictions']])>1e-6))
    # One panel per recorded coordinate is unnecessary for constant coordinates.
    if coordinates:
        coordinate=coordinates[0];width=.8/max(1,len(reports))
        for j,r in enumerate(reports):
            numbers=[r['V2']['probes'].get(m,{}).get('coordinates',{}).get(coordinate,{}).get('mae',np.nan) for m in modalities]
            axes[1].bar(np.arange(len(modalities))+(j-(len(reports)-1)/2)*width,numbers,width,label=names[j])
        axes[1].set_xticks(range(len(modalities)),modalities);axes[1].legend()
        axes[1].set_ylabel('Held-out MAE: '+coordinate)
    else:axes[1].text(.5,.5,'No varying coordinate',ha='center',transform=axes[1].transAxes)
    axes[1].set_title('V2: grouped coordinate probes')
    fig.savefig(out/'repeatability_and_probes.png',dpi=170);plt.close(fig)

    fig,axes=plt.subplots(1,max(1,len(reports)),figsize=(6*max(1,len(reports)),5.8),squeeze=False,layout='constrained')
    for ax,r in zip(axes[0],reports):
        ax.plot([0,1],[0,1],'--',color='.65',label='Ideal calibration')
        available=0
        for cid,cal in r['V3'].items():
            bins=[b for b in cal['bins'] if b['n']]
            if not bins:continue
            available+=1
            label='\n'.join(textwrap.wrap(cid,38))+f' (n={cal["n"]})'
            ax.plot([b['predicted'] for b in bins],[b['observed'] for b in bins],'o-',label=label)
            for b in bins:ax.annotate(str(b['n']),(b['predicted'],b['observed']),xytext=(4,-12),textcoords='offset points',fontsize=8)
        ax.set(xlim=(-.03,1.03),ylim=(-.07,1.07),xlabel='Predicted probability (training neighbors)',ylabel='Held-out observed success',title=r['task_id'].split('|')[0])
        ax.legend(loc='upper center',bbox_to_anchor=(.5,-.18),fontsize=8)
        if not available:ax.text(.5,.5,'No assessed terminal evidence',ha='center')
    fig.suptitle('V3: grouped reliability; point labels are held-out counts')
    fig.savefig(out/'reliability.png',dpi=170);plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
    available=[r for r in reports if r['V6']['samples']]
    for r in available:
        values=r['V6'];label=r['task_id'].split('|')[0]
        sigma=np.array([s['sigma'] for s in values['samples']]);error=np.array([s['absolute_error'] for s in values['samples']])
        ks=np.linspace(0,4,41);coverage=[float(np.mean(error<=k*sigma)) for k in ks]
        axes[0].plot(ks,coverage,label=f'{label}, n={len(error)} transitions')
        measures=[values['endpoint_within_noise'],values['variance_augmented_within_noise'],values['sigma_2x_coverage'],values['p_off_auc']]
        axes[1].bar(range(4),[x if x is not None else np.nan for x in measures],label=label)
        for i,v in enumerate(measures):
            if v is not None:axes[1].text(i,v+.02,f'{v:.3f}',ha='center')
    axes[0].axhline(.95,ls='--',color='.65',label='95% reference (not assumed)')
    axes[0].set(xlabel='Multiple of ensemble standard deviation',ylabel='Measured component coverage',ylim=(0,1.05),title='V6: empirical spread calibration');axes[0].legend(fontsize=8)
    axes[1].set_xticks(range(4),['Endpoint\nwithin noise','Noise +\nvariance','2-sigma\ncoverage','Off-label\nAUC']);axes[1].set_ylim(0,1.12)
    axes[1].set_title('V6: grouped transition diagnostics')
    if not available:axes[1].text(.5,.5,'No observed transitions',ha='center',transform=axes[1].transAxes)
    fig.savefig(out/'dynamics.png',dpi=170);plt.close(fig)
    return ['repeatability_and_probes.png','reliability.png','dynamics.png']
