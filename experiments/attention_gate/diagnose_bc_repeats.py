"""Repeat frozen interventions on B/C paired weights and the original 54 TRAIN images."""
import argparse
import gc
import json
import msvcrt
import os
import statistics
import time
import traceback
from pathlib import Path

import common_runner as core
import common_runner_c as c
import common_runner_v2 as fast
import diagnose_attention as diag
import safe_progress_io as io
from ca_hswish_control import build_c

MANIFEST = core.HERE / 'step_04_2026-10-14/run_001/fixed_train_manifest.json'
PROTOCOL = core.HERE / 'diagnostic_protocol_v1.json'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sources():
    result = c.sources()
    for path in (Path(__file__), Path(diag.__file__), PROTOCOL, MANIFEST):
        result[str(path.relative_to(core.PROJECT))] = core.digest(path)
    return result


def summarize(rows, stats, conditions):
    result = {}
    for condition in conditions:
        selected = [r for r in rows if r['condition'] == condition]
        matrix = diag.np.array([[sum(r['conf_00'] for r in selected), sum(r['conf_01'] for r in selected)],
                                [sum(r['conf_10'] for r in selected), sum(r['conf_11'] for r in selected)]])
        score = diag.scores_from_confusion(matrix)
        biou = [r['boundary_iou_image'] for r in selected if r['boundary_iou_image'] is not None]
        result[condition] = dict(images=len(selected), confusion=matrix.tolist(),
            sidewalk_iou_pct=100 * score['iou'][1], miou_pct=100 * score['miou'],
            boundary_iou_pct=100 * statistics.mean(biou) if biou else None,
            boundary_valid_images=len(biou),
            feature_l2_retention_mean=statistics.mean(r['feature_l2_retention_under_intervention'] for r in selected))
    for condition in conditions:
        result[condition]['sidewalk_delta_from_original_pp'] = result[condition]['sidewalk_iou_pct'] - result['original']['sidewalk_iou_pct']
    gate = {}
    for key in stats[0]:
        if key in ('model', 'sample_id', 'city'):
            continue
        values = [r[key] for r in stats if r[key] is not None]
        gate[key] = dict(mean=statistics.mean(values) if values else None, images=len(values))
    return dict(conditions=result, gate_statistics=gate)


def run(a):
    output = a.output.resolve()
    assert output.is_relative_to(core.HERE) and output != core.HERE and not output.exists()
    evaluation = read(a.sensitivity_job / 'final_verification.json')
    assert evaluation['status'] == 'pass' and evaluation['models'] == 9 and evaluation['images_per_model'] == 500
    assert evaluation['full_core_metrics_reproduced'] is True
    raw = read(a.source_job / 'three_seed_ABC_results.json')
    assert read(a.source_job / 'final_verification.json')['status'] == 'pass'
    models = [r for r in raw['results'] if r['variant'] in ('B', 'C') and (a.mode == 'full' or r['seed'] == 42)]
    protocol = read(PROTOCOL)
    conditions = [r['id'] for r in protocol['interventions']]
    core.torch.set_num_threads(2)
    dataset = core.SidewalkDataset(fast.protocol(), 'train')
    manifest = diag.select_manifest(dataset, protocol)
    assert manifest == read(MANIFEST) and len(manifest) == 54
    active = manifest if a.mode == 'full' else manifest[:2]
    pinned = sources()
    output.mkdir(parents=True)
    core.json_write(output / 'plan.json', dict(mode=a.mode, source_hashes=pinned,
        sensitivity_verification_sha256=core.digest(a.sensitivity_job / 'final_verification.json'),
        models=models, conditions=conditions, images=len(active), fixed_manifest=manifest,
        scope='Train-image inference interventions only; not a training ablation or independent test'))
    began = time.perf_counter()
    summaries, checks = [], []
    try:
        with (core.HERE / 'seed42_training.lock').open('a+b') as lock:
            lock.seek(0); msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            try:
                with fast.keep_awake():
                    samples = diag.prepare_samples(dataset, active, .02)
                    for model_row in models:
                        if (output / 'STOP_AFTER_MODEL').exists():
                            io.atomic_json(output / 'job_status.json', dict(status='stopped_by_user', completed_models=len(summaries)))
                            return
                        seed, variant = model_row['seed'], model_row['variant']
                        core.configure(seed)
                        model = build_c(seed) if variant == 'C' else core.build_paired_model('B', seed)
                        weights = Path(model_row['best_weights'])
                        weight_hash = core.digest(weights)
                        model.load_state_dict(core.torch.load(weights, map_location='cpu', weights_only=True), strict=True)
                        model = model.cuda().eval()
                        source_state = core.state_digest(model.state_dict())
                        module = model.ca
                        wrapper = diag.InterventionCA(module, protocol['controls']['energy_matching_max_channel_relative_error_tolerance']).eval()
                        rows, stats, energy_max = [], [], 0.0
                        name = f'{variant}_seed{seed}'
                        with core.torch.inference_mode():
                            for position, sample in enumerate(samples):
                                tensor = dataset.transform(diag.Image.fromarray(sample['rgb']))[None].cuda()
                                original_logits = model(tensor)
                                wrapper.begin(sample['id'])
                                model.ca = wrapper
                                assert not any(m.training for m in model.modules())
                                original_prediction = None
                                for condition in conditions:
                                    wrapper.mode = condition
                                    logits = model(tensor)
                                    if condition == 'original':
                                        assert core.torch.equal(logits, original_logits)
                                        stats.append(dict(model=name, sample_id=sample['id'], city=sample['city'],
                                            **diag.feature_statistics(wrapper, sample, protocol)))
                                    row, pred = diag.metric_row(name, condition, sample, logits, original_prediction)
                                    if original_prediction is None:
                                        original_prediction = pred.copy()
                                    row.update(feature_l2_retention_under_intervention=wrapper.output_retention,
                                        channel_energy_max_relative_error=wrapper.max_energy_error)
                                    rows.append(row)
                                    if condition.startswith('energy_'):
                                        energy_max = max(energy_max, wrapper.max_energy_error)
                                    del logits
                                assert wrapper.calls == len(conditions)
                                model.ca = module
                                del tensor, original_logits
                                if (position + 1) % 9 == 0 or position + 1 == len(samples):
                                    io.atomic_json(output / 'job_status.json', dict(status='running', pid=os.getpid(),
                                        model=name, images_completed=position+1, images_total=len(samples), completed_models=len(summaries)))
                                    print(name, position+1, '/', len(samples), flush=True)
                        assert core.state_digest(model.state_dict()) == source_state
                        assert core.digest(weights) == weight_hash and sources() == pinned
                        diag.csv_write(output / f'{name}_interventions.csv', rows)
                        diag.csv_write(output / f'{name}_gate_statistics.csv', stats)
                        summaries.append(dict(variant=variant, seed=seed, weights_sha256=weight_hash,
                            **summarize(rows, stats, conditions)))
                        checks.append(dict(model=name, original_logits_bitwise_equal=True,
                            parameters_and_buffers_unchanged=True, feature_identity_and_energy_checks_pass=True,
                            maximum_energy_error=energy_max, images=len(samples), predictions=len(rows)))
                        del model, wrapper, module, rows, stats
                        gc.collect(); core.torch.cuda.empty_cache()
            finally:
                lock.seek(0); msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
        for row in manifest:
            assert core.digest(core.PROJECT / row['image']) == row['image_sha256']
            assert core.digest(core.PROJECT / row['label']) == row['label_sha256']
        assert sources() == pinned
        core.json_write(output / 'results.json', dict(results=summaries, independent_test=False,
            intervention_causality_limit='Decoder/BN input distributions may shift even under energy matching'))
        core.json_write(output / 'final_verification.json', dict(status='pass', mode=a.mode,
            models=len(models), images_per_model=len(active), conditions_per_model=len(conditions),
            source_hashes=pinned, checks=checks, fixed_manifest_unchanged=True, training_started=False))
        io.atomic_json(output / 'job_status.json', dict(status='complete', verified=True, completed_models=len(models),
            wall_seconds=time.perf_counter()-began))
    except Exception:
        io.atomic_json(output / 'job_status.json', dict(status='failed', traceback=traceback.format_exc()))
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=('pilot', 'full'), required=True)
    p.add_argument('--source-job', type=Path, required=True)
    p.add_argument('--sensitivity-job', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    run(p.parse_args())
