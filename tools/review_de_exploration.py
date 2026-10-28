"""Reconstruct the frozen D/E exploration decision from the completed checkpoints."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import argparse
import datetime
from pathlib import Path
import sys
from export_repository_summary import ROOT, EXP, digest, dump
from export_followup_summary import read, sanitize
sys.path.insert(0,str(EXP))
import run_mixed_pair as supervisor


def main(a):
    supervisor.core.torch.set_num_threads(1)
    job=a.job.resolve()
    proof=read(job/'final_verification.json')
    assert proof['status']=='pass' and proof['all_100_epochs'] and proof['historical_artifacts_unchanged']
    plan=read(job/'job_plan.json')
    supervisor.integrity(job,plan,original=True)
    raw=read(job/'seed42_ABCDE_results.json')
    indexed={r['variant']:r for r in raw['results']}
    assert len(raw['results'])==5 and set(indexed)==set('ABCDE')
    rows=[r for r in read(ROOT/'results/abc_summary.json')['results'] if r['seed']==42]
    metrics=['best_global_miou_pct','best_sidewalk_iou_pct','best_boundary_iou_pct']
    for r in rows:
        assert all(r[k]==indexed[r['variant']][k] for k in metrics+['best_epoch','final_checkpoint_sha256'])
    for entry in plan['entries']:
        row=indexed[entry['variant']]
        checkpoint=Path(entry['remaining'])/'checkpoints/epoch_100.pth'
        assert row['epochs']==100 and digest(checkpoint)==row['final_checkpoint_sha256']
        cp=supervisor.core.torch.load(checkpoint,map_location='cpu',weights_only=False)
        selected=max(cp['records'],key=lambda r:r['val_miou_global'])
        assert cp['completed_epoch']==100 and row['best_epoch']==cp['best_epoch']==selected['epoch']
        fields=['val_miou_global','val_iou_sidewalk_global','val_boundary_iou_author_fg_mean']
        assert all(abs(row[k]-100*selected[v])<1e-12 for k,v in zip(metrics,fields))
        assert row['alpha_at_selected_epoch']==selected['alpha_after']
        assert row['final_alpha']==cp['records'][-1]['alpha_after']
        assert digest(Path(row['best_weights']))==row['best_weights_sha256']
        weights=supervisor.core.torch.load(row['best_weights'],map_location='cpu',weights_only=True)
        assert supervisor.equal_tree(weights,cp['best_model_state'])
        rows.append(row)
    baseline=indexed['C']
    passing=[v for v in 'DE' if all(indexed[v][k]>=baseline[k] for k in metrics[1:])
        and any(indexed[v][k]>baseline[k] for k in metrics[1:])]
    assert raw['candidates_passing_screen']==passing and raw['promotion_screen_passed']==bool(passing)
    comparisons={v+'_minus_'+other:{k:indexed[v][k]-indexed[other][k] for k in metrics}
        for v in 'DE' for other in ('A','B','C')}
    comparisons['E_minus_D']={k:indexed['E'][k]-indexed['D'][k] for k in metrics}
    report=dict(status='pass',reviewed_at=datetime.datetime.now().astimezone().isoformat(timespec='seconds'),
        results=rows,paired_percentage_points=comparisons,candidates_passing_screen=passing,
        decision='review_and_repeat_both_candidates' if passing else 'stop_DE_branch_and_continue_baseline_deployment',
        repeat_training_started=False,exploratory_seed_only=True,training_parameters_changed=False,
        rule='At least one candidate must improve C42 on sidewalk or Boundary IoU and be no worse on the other; otherwise stop this bounded branch.',
        interpretation='E improves region IoU over C42 but loses boundary IoU; do not substitute global mIoU for the predeclared two-metric screen.' if not passing else 'Screening only, not proof of stable benefit.',
        deployment_baselines=['A42','B42','C42'],
        baseline_rationale='Retain B because it has the strongest existing three-seed means; engineering comparison uses the fixed seed42 for all references.',
        limits='Validation was used for checkpoint selection. D/E each have one seed; E versus D improvements are exploratory, not stable superiority or significance.',
        sources={str(p.relative_to(ROOT)):digest(p) for p in (job/'final_verification.json',job/'seed42_ABCDE_results.json',job/'job_plan.json',ROOT/'results/abc_summary.json',Path(__file__))})
    dump(ROOT/'results/de_review.json',sanitize(report))
    print(report['decision'],comparisons['E_minus_C'])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--job',type=Path,default=EXP/'step_07_2026-10-26/DE_seed42_training_001')
    main(p.parse_args())
