"""Plot all D/E epochs; markers use the frozen global-mIoU-selected checkpoint."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
from pathlib import Path
import sys
import json
import subprocess
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from export_repository_summary import ROOT, EXP, digest, dump
from export_followup_summary import read
sys.path.insert(0,str(EXP))


def main():
    report=read(ROOT/'results/de_review.json')
    assert report['status']=='pass' and report['exploratory_seed_only']
    rows={r['variant']:r for r in report['results']}
    fig,axes=plt.subplots(1,3,figsize=(12.8,4.0),layout='constrained')
    metrics=[('val_iou_sidewalk_global','best_sidewalk_iou_pct','Sidewalk IoU (%)'),
        ('val_boundary_iou_author_fg_mean','best_boundary_iou_pct','Boundary IoU, 26px (%)')]
    inputs={}
    for variant,color in [('D','#2166ac'),('E','#b35806')]:
        row=rows[variant]; path=ROOT/row['final_checkpoint']
        assert digest(path)==row['final_checkpoint_sha256']
        # Keep PyTorch and this Windows matplotlib build in separate processes:
        # their bundled OpenMP DLLs conflict when loaded together.
        extract="""import json,sys,torch
torch.set_num_threads(1)
cp=torch.load(sys.argv[1],map_location='cpu',weights_only=False)
keys=['epoch','val_iou_sidewalk_global','val_boundary_iou_author_fg_mean','alpha_after']
print(json.dumps(dict(best_epoch=cp['best_epoch'],records=[{k:r[k] for k in keys} for r in cp['records']])))
"""
        cp=json.loads(subprocess.check_output([sys.executable,'-c',extract,str(path)],text=True))
        assert digest(path)==row['final_checkpoint_sha256']
        records=cp['records']; assert [r['epoch'] for r in records]==list(range(1,101))
        epoch=row['best_epoch']; assert epoch==cp['best_epoch']
        xs=[r['epoch'] for r in records]
        for ax,(field,selected_metric,label) in zip(axes,metrics):
            ys=[100*r[field] for r in records]
            assert abs(ys[epoch-1]-row[selected_metric])<1e-12
            ax.plot(xs,ys,color=color,linewidth=1.3,label=variant)
            ax.scatter([epoch],[ys[epoch-1]],color=color,s=45,zorder=4,marker='o')
            ax.set_ylabel(label)
        axes[2].plot(xs,[r['alpha_after'] for r in records],color=color,linewidth=1.6,label=variant)
        inputs[str(path.relative_to(ROOT))]=digest(path)
    for ax,(_,selected_metric,_) in zip(axes,metrics):
        ax.axhline(rows['C'][selected_metric],color='#666666',linestyle='--',linewidth=1,label='C selected')
    axes[2].set_ylabel('Mixture coefficient alpha')
    axes[2].set_ylim(.08,.23)
    for ax in axes:
        ax.set_xlabel('Completed epoch'); ax.set_xlim(1,100)
        ax.grid(alpha=.18); ax.spines[['top','right']].set_visible(False)
        ax.legend(fontsize=8,loc='lower right')
    fig.suptitle('Seed42 exploration: complete histories; dots mark global-mIoU selection',fontsize=12)
    fig.savefig(ROOT/'results/de_exploration.png',dpi=180)
    fig.savefig(ROOT/'results/de_exploration.svg')
    plt.close(fig)
    dump(ROOT/'results/de_exploration_plot.json',dict(status='pass',source_checkpoints=inputs,
        review_sha256=digest(ROOT/'results/de_review.json'),script_sha256=digest(Path(__file__)),
        png_sha256=digest(ROOT/'results/de_exploration.png'),svg_sha256=digest(ROOT/'results/de_exploration.svg'),
        scope='All100 epochs, one exploratory seed, C selected checkpoint reference; not three-seed confirmation.'))
    print('Saved full-history D/E figure without changing metrics or selection.')


if __name__=='__main__': main()
