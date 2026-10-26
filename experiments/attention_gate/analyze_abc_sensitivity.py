"""Reconstruct all sensitivity metrics from saved per-image evidence, without GPU."""
import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
GROUPS = ('zero', '(0,1%]', '(1%,5%]', '>5%')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def score(matrix):
    tn, fp = matrix[0]
    fn, tp = matrix[1]
    fg = tp / (tp + fp + fn) if tp + fp + fn else 0.
    bg = tn / (tn + fp + fn) if tn + fp + fn else 0.
    return dict(confusion=matrix, sidewalk_iou_pct=100 * fg, global_miou_pct=50 * (fg + bg),
        precision_pct=100*tp/(tp+fp) if tp+fp else None,
        recall_pct=100*tp/(tp+fn) if tp+fn else None,
        false_positive_pixels=fp, false_negative_pixels=fn)


def sum_conf(rows, policy):
    return [[sum(r['policies'][policy]['confusion'][i][j] for r in rows) for j in range(2)] for i in range(2)]


def group(row):
    fraction = row['foreground_pixels'] / (768 * 1024)
    return 'zero' if fraction == 0 else '(0,1%]' if fraction <= .01 else '(1%,5%]' if fraction <= .05 else '>5%'


def close(actual, expected):
    if expected is None:
        assert actual is None
    else:
        assert math.isclose(actual, expected, rel_tol=0, abs_tol=1e-9), (actual, expected)


def main(a):
    job, out = a.job.resolve(), a.output.resolve()
    assert out.is_relative_to(HERE) and not out.exists()
    verification = read(job / 'final_verification.json')
    assert verification['status'] == 'pass' and verification['full_core_metrics_reproduced']
    assert verification['models'] == 9 and verification['images_per_model'] == 500
    raw = read(job / 'results.json')['results']
    assert {(r['variant'], r['seed']) for r in raw} == {(v, s) for v in 'ABC' for s in (42,43,44)}
    inputs = {str((job / 'final_verification.json').relative_to(PROJECT)): digest(job / 'final_verification.json')}
    records, partition, valid_support = [], None, None
    for r in raw:
        path = job / f"{r['variant']}_seed{r['seed']}_per_image.json"
        images = read(path)
        inputs[str(path.relative_to(PROJECT))] = digest(path)
        assert len(images) == 500 and len({p['id'] for p in images}) == 500
        current_partition = [(p['id'], p['city'], p['foreground_pixels'], p['ignored_pixels'], group(p)) for p in images]
        if partition is None:
            partition = current_partition
        assert partition == current_partition
        support = sum(768*1024-p['ignored_pixels'] for p in images)
        if valid_support is None:
            valid_support = support
        assert valid_support == support
        for policy in ('core', 'valid_pixels'):
            matrix = sum_conf(images, policy)
            assert matrix == r['policies'][policy]['confusion']
            assert sum(map(sum, matrix)) == (500*768*1024 if policy == 'core' else support)
            values = score(matrix)
            for metric in ('sidewalk_iou_pct', 'global_miou_pct', 'precision_pct', 'recall_pct'):
                close(values[metric], r['policies'][policy][metric])
            for city, reported in r['cities'].items():
                assert sum_conf([p for p in images if p['city'] == city], policy) == reported[policy]['confusion']
        for width in ('6', '13', '26'):
            values = [p['core_boundary_iou'][width] for p in images if p['core_boundary_iou'][width] is not None]
            assert len(values) == r['core_boundary'][width]['valid_images']
            close(100*statistics.mean(values), r['core_boundary'][width]['mean_pct'])
        groups = {}
        for name in GROUPS:
            selected = [p for p in images if group(p) == name]
            groups[name] = dict(images=len(selected), policies={p:score(sum_conf(selected,p)) if selected else None for p in ('core','valid_pixels')})
        records.append(dict(variant=r['variant'], seed=r['seed'], selected_epoch=r['selected_epoch'],
            weights_sha256=r['weights_sha256'], policies=r['policies'], core_boundary=r['core_boundary'],
            cities=r['cities'], foreground_groups=groups))
    metrics = {'core_sidewalk_iou_pct': lambda r:r['policies']['core']['sidewalk_iou_pct'],
        'valid_pixels_sidewalk_iou_pct':lambda r:r['policies']['valid_pixels']['sidewalk_iou_pct'],
        **{f'boundary_{w}px_pct':lambda r,w=w:r['core_boundary'][w]['mean_pct'] for w in ('6','13','26')}}
    summary, paired = {}, {}
    for variant in 'ABC':
        summary[variant] = {}
        for metric, getter in metrics.items():
            values = [getter(r) for r in records if r['variant'] == variant]
            summary[variant][metric] = dict(mean=statistics.mean(values), sample_std_ddof1=statistics.stdev(values))
    for left, right in (('B','A'),('C','A'),('C','B')):
        paired[left+'_minus_'+right] = {}
        for metric, getter in metrics.items():
            differences = {}
            for seed in (42,43,44):
                indexed = {r['variant']:r for r in records if r['seed'] == seed}
                differences[str(seed)] = getter(indexed[left])-getter(indexed[right])
            values = list(differences.values())
            paired[left+'_minus_'+right][metric] = dict(per_seed=differences, mean=statistics.mean(values),
                sample_std_ddof1=statistics.stdev(values), positive_seeds=sum(v>0 for v in values), negative_seeds=sum(v<0 for v in values))
    out.mkdir(parents=True)
    addendum = HERE / 'step_07_2026-10-24/literature_plan_001/plan_addendum.json'
    write(out / 'results.json', dict(results=records, summary=summary, paired_percentage_points=paired,
        grouping_predeclared_in=str(addendum.relative_to(PROJECT)), grouping_plan_sha256=digest(addendum),
        group_image_counts={name:sum(p[-1]==name for p in partition) for name in GROUPS},
        total_pixels=500*768*1024, valid_pixels=valid_support, ignored_fraction=1-valid_support/(500*768*1024),
        inputs=inputs, independent_test=False, interpretation='Separate same-weight sensitivity, no retraining or checkpoint reselection'))
    write(out / 'verification.json', dict(status='pass', script_sha256=digest(__file__), inputs=inputs,
        images_per_model=500, models=9, per_image_and_city_confusion_reconstructed=True,
        boundary_means_and_excluded_counts_reconstructed=True, identical_image_partition_and_valid_support=True))
    print(json.dumps(dict(summary=summary, paired=paired, group_counts={name:sum(p[-1]==name for p in partition) for name in GROUPS}), ensure_ascii=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--job', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    main(p.parse_args())
