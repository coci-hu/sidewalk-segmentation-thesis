"""Publication figure of every paired seed; no significance or independent-test claim."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]


def main():
    rows=json.loads((ROOT/'results/abc_summary.json').read_text(encoding='utf-8'))['results']
    assert len(rows)==9
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'svg.hashsalt':'sidewalk-abc-v1'})
    fig,axes=plt.subplots(1,3,figsize=(11,4.3))
    colors={42:'#3366aa',43:'#d57726',44:'#16826b'}
    metrics=[('best_global_miou_pct','Global mIoU (%)'),('best_sidewalk_iou_pct','Sidewalk IoU (%)'),
        ('best_boundary_iou_pct','Boundary IoU (%)')]
    for ax,(key,label) in zip(axes,metrics):
        for seed in (42,43,44):
            selected={r['variant']:r[key] for r in rows if r['seed']==seed}
            ax.plot([0,1,2],[selected[v] for v in 'ABC'],marker='o',color=colors[seed],label=f'Seed {seed}',linewidth=1.4)
        ax.set_xticks([0,1,2],['A\nENet','B\nCA ReLU','C\nCA h-swish'])
        ax.set_ylabel(label)
        ax.spines[['top','right']].set_visible(False)
        ax.grid(axis='y',alpha=.2)
        ax.margins(x=.15,y=.18)
    axes[0].legend(frameon=False,fontsize=9)
    fig.suptitle('Controlled comparison: three paired seeds, 100 epochs per run',y=.97,fontsize=13)
    fig.text(.5,.025,'Cityscapes validation (used for checkpoint selection); same selected weight for all metrics.\nBoundary width: 26 px. Lines connect paired seeds; truncated vertical axes show variation.',
        ha='center',va='bottom',fontsize=8.5,color='#444444')
    fig.subplots_adjust(left=.07,right=.985,top=.85,bottom=.24,wspace=.35)
    fig.savefig(ROOT/'results/abc_three_seeds.png',dpi=180)
    fig.savefig(ROOT/'results/abc_three_seeds.svg',metadata={'Date':None})
    plt.close(fig)
    print('Rendered 9 observed runs; no inferred statistical significance.')


if __name__=='__main__':
    main()
