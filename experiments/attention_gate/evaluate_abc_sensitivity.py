"""Frozen-weight sensitivity evaluation; train/selection rules are not changed.

Full mode waits for verified C repeats, then evaluates all nine selected weights.
Pilot mode checks metrics and two real images each for A42/C42, never trains.
"""
import argparse
from collections import defaultdict
import json
import msvcrt
import os
from pathlib import Path
import subprocess
import time
import traceback

import numpy as np
from PIL import Image
import common_runner as core
import common_runner_v2 as fast
import common_runner_c as c
from ca_hswish_control import build_c
from metrics_v1 import boundary_iou_author, confusion_matrix_numpy, scores_from_confusion
import safe_progress_io as io
from evaluate_existing import timestamp

VALID_IDS = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def masked_confusion(pred, target, valid):
    truth = np.where(valid, target, -1)
    return confusion_matrix_numpy(pred, truth)


def describe(matrix):
    tn, fp, fn, tp = map(int, np.asarray(matrix).ravel())
    scores = scores_from_confusion(matrix)
    return {'confusion': np.asarray(matrix).tolist(), 'global_miou_pct': 100 * scores['miou'],
        'sidewalk_iou_pct': 100 * scores['iou'][1],
        'precision_pct': 100 * tp / (tp + fp) if tp + fp else None,
        'recall_pct': 100 * tp / (tp + fn) if tp + fn else None,
        'false_positive_pixels': fp, 'false_negative_pixels': fn}


class Dataset(core.SidewalkDataset):
    def __getitem__(self, index):
        image, binary = super().__getitem__(index)
        with Image.open(self.pairs[index][1]) as source:
            labels = np.array(source.resize((self.width, self.height), Image.NEAREST))
        assert np.array_equal(binary.numpy(), labels == 8)
        return image, binary, core.torch.from_numpy(np.isin(labels, VALID_IDS)), index


def checks():
    gt = np.array([[0, 1], [0, 1]])
    pred = np.array([[0, 0], [1, 1]])
    valid = np.array([[True, True], [True, False]])
    assert np.array_equal(masked_confusion(pred, gt, valid), [[1, 1], [1, 0]])
    assert masked_confusion(pred, gt, np.zeros_like(valid)).sum() == 0
    assert masked_confusion(pred, gt, np.ones_like(valid)).sum() == 4
    empty = np.zeros((64, 64), dtype=bool)
    obj = empty.copy(); obj[20:40, 20:40] = True
    for width in (6, 13, 26):
        assert boundary_iou_author(obj, obj, width_pixels=width) == 1
        assert boundary_iou_author(empty, empty, width_pixels=width) is None
        assert boundary_iou_author(empty, obj, width_pixels=width) == 0
    assert np.isin(np.arange(256), VALID_IDS).sum() == 19


def evaluate(row, output, limit=None):
    pinned = c.sources()
    weights = Path(row['best_weights'])
    before = core.digest(weights)
    core.configure(row['seed'])
    model = build_c(row['seed']) if row['variant'] == 'C' else core.build_paired_model(row['variant'], row['seed'])
    model.load_state_dict(core.torch.load(weights, map_location='cpu', weights_only=True), strict=True)
    model.cuda().eval()
    data = Dataset(fast.protocol(), 'val')
    selected = data if limit is None else core.Subset(data, list(range(limit)))
    loader = core.DataLoader(selected, batch_size=1, shuffle=False, num_workers=4,
        pin_memory=True, persistent_workers=False, worker_init_fn=core.seed_worker,
        generator=core.torch.Generator().manual_seed(70042))
    aggregate = {p: np.zeros((2, 2), dtype=np.int64) for p in ('core', 'valid_pixels')}
    cities = defaultdict(lambda: {p: np.zeros((2, 2), dtype=np.int64) for p in aggregate})
    boundaries = {w: [] for w in (6, 13, 26)}
    image_records = []
    started = time.perf_counter()
    with core.torch.inference_mode():
        for image, truth, valid, index in loader:
            logits = model(image.cuda(non_blocking=True))
            assert core.torch.isfinite(logits).all()
            pred = logits.argmax(1)[0].cpu().numpy()
            target, allowed = truth[0].numpy(), valid[0].numpy()
            i = int(index.item())
            city = data.pairs[i][0].parent.name
            matrices = {'core': confusion_matrix_numpy(pred, target), 'valid_pixels': masked_confusion(pred, target, allowed)}
            biou = {str(w): boundary_iou_author(target == 1, pred == 1, width_pixels=w) for w in boundaries}
            for policy, matrix in matrices.items():
                aggregate[policy] += matrix
                cities[city][policy] += matrix
            for w in boundaries:
                if biou[str(w)] is not None:
                    boundaries[w].append(biou[str(w)])
            image_records.append({'index': i, 'id': data.pairs[i][0].stem, 'city': city,
                'foreground_pixels': int((target == 1).sum()), 'ignored_pixels': int((~allowed).sum()),
                'policies': {p: describe(m) for p, m in matrices.items()}, 'core_boundary_iou': biou})
            if len(image_records) % 50 == 0 or len(image_records) == len(selected):
                io.atomic_json(output / 'job_status.json', {'status': 'running', 'pid': os.getpid(),
                    'phase': 'frozen_weight_sensitivity', 'variant': row['variant'], 'seed': row['seed'],
                    'images_completed': len(image_records), 'images_total': len(selected), 'training_started': False})
                print(row['variant'], row['seed'], len(image_records), '/', len(selected), flush=True)
    assert len(image_records) == len(selected)
    assert aggregate['core'].sum() == len(selected) * 768 * 1024
    assert core.digest(weights) == before and c.sources() == pinned
    summary = {'variant': row['variant'], 'seed': row['seed'], 'images': len(selected),
        'weights': str(weights), 'weights_sha256': before, 'selected_epoch': row['best_epoch'],
        'policies': {p: describe(m) for p, m in aggregate.items()},
        'cities': {city: {p: describe(m) for p, m in by_policy.items()} for city, by_policy in cities.items()},
        'core_boundary': {str(w): {'mean_pct': float(100 * np.mean(v)) if v else None,
            'valid_images': len(v), 'both_empty_excluded': len(selected) - len(v)} for w, v in boundaries.items()},
        'wall_seconds': time.perf_counter() - started}
    if limit is None:
        assert abs(summary['policies']['core']['global_miou_pct'] - row['best_global_miou_pct']) < 1e-7
        assert abs(summary['policies']['core']['sidewalk_iou_pct'] - row['best_sidewalk_iou_pct']) < 1e-7
        assert abs(summary['core_boundary']['26']['mean_pct'] - row['best_boundary_iou_pct']) < 1e-7
    core.json_write(output / f'{row["variant"]}_seed{row["seed"]}_per_image.json', image_records)
    core.json_write(output / f'{row["variant"]}_seed{row["seed"]}_summary.json', summary)
    del model
    core.torch.cuda.empty_cache()
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=('pilot', 'wait-and-evaluate'), required=True)
    p.add_argument('--source-job', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--pilot-proof', type=Path)
    a = p.parse_args()
    output = a.output.resolve()
    assert output.is_relative_to(core.HERE) and not output.exists()
    output.mkdir(parents=True)
    core.torch.set_num_threads(2)
    checks()
    pinned = c.sources()
    script_sha = core.digest(__file__)
    try:
        if a.mode == 'pilot':
            rows = [r for r in read(a.source_job / 'c_seed42_results.json')['results'] if r['variant'] in ('A', 'C')]
            limit = 2
        else:
            pilot = read(a.pilot_proof)
            assert pilot['status'] == 'pass' and pilot['script_sha256'] == script_sha and pilot['source_hashes'] == pinned
            began = time.monotonic()
            while True:
                assert core.digest(__file__) == script_sha and c.sources() == pinned
                if (output / 'STOP_AFTER_MODEL').exists() or (a.source_job / 'STOP_AFTER_EPOCH').exists():
                    io.atomic_json(output / 'job_status.json', {'status': 'stopped_by_user', 'training_started': False})
                    return
                status = read(a.source_job / 'job_status.json')
                if status['status'] == 'complete':
                    assert status['verified'] and read(a.source_job / 'final_verification.json')['status'] == 'pass'
                    break
                if status['status'] in ('failed', 'stopped_at_epoch_boundary'):
                    raise RuntimeError('Prerequisite training failed or paused; no evaluation started')
                if time.monotonic() - began > 72 * 3600:
                    raise TimeoutError('C repeats not complete within72h; no retry')
                io.atomic_json(output / 'job_status.json', {'status': 'waiting_for_C_repeats', 'pid': os.getpid(),
                    'prerequisite': str(a.source_job), 'training_started': False, 'at': timestamp()})
                time.sleep(30)
            rows = read(a.source_job / 'three_seed_ABC_results.json')['results']
            assert sorted((r['variant'], r['seed']) for r in rows) == [(v, s) for v in ('A', 'B', 'C') for s in (42, 43, 44)]
            limit = None
        core.json_write(output / 'evaluation_plan.json', {'mode': a.mode, 'source_job': str(a.source_job),
            'source_hashes': pinned, 'script_sha256': script_sha, 'rows': rows, 'limit': limit,
            'valid_original_ids': VALID_IDS, 'boundary_widths': [6, 13, 26],
            'boundary_policy': 'core binary only; no artificial masked-boundary metric',
            'selection': 'reuse saved global-mIoU-best weights, no reselection', 'training_started': False})
        with (core.HERE / 'seed42_training.lock').open('a+b') as lock:
            lock.seek(0); msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            try:
                with fast.keep_awake():
                    summaries = []
                    for row in rows:
                        if (output / 'STOP_AFTER_MODEL').exists():
                            io.atomic_json(output / 'job_status.json', {'status': 'stopped_by_user', 'completed_models': len(summaries)})
                            return
                        summaries.append(evaluate(row, output, limit))
            finally:
                lock.seek(0); msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
        assert c.sources() == pinned and core.digest(__file__) == script_sha
        core.json_write(output / 'results.json', {'results': summaries, 'independent_test': False,
            'masked_results_are_separate_sensitivity_not_retrained_models': True})
        core.json_write(output / 'final_verification.json', {'status': 'pass', 'script_sha256': script_sha,
            'source_hashes': pinned, 'metric_edge_cases_pass': True, 'models': len(summaries),
            'images_per_model': limit or 500, 'full_core_metrics_reproduced': limit is None, 'training_started': False})
        io.atomic_json(output / 'job_status.json', {'status': 'complete', 'verified': True,
            'completed_models': len(summaries), 'completed': timestamp(), 'training_started': False,
            'next_step': 'review paired label/boundary sensitivity and perform fixed-sample gate diagnosis before D/E'})
    except BaseException:
        io.atomic_json(output / 'job_status.json', {'status': 'failed', 'traceback': traceback.format_exc(),
            'training_started': False, 'partial_outputs_preserved': True})
        raise


if __name__ == '__main__':
    main()
