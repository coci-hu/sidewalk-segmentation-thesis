"""Read-only audit of completed A/B/C results and the local label convention.

Writes only into a new review directory. Does not train or change old artifacts.
The label inventory is a prospective baseline, not proof of historical pixel hashes.
"""
import argparse
import io
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image
import common_runner as core
import run_c_exploration as runner
from check_step03 import equal_tree
from ca_hswish_control import build_c

VALID_IDS = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
LABEL_SOURCE = 'https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    assert output.is_relative_to(core.HERE) and not output.exists()
    output.mkdir(parents=True)
    plan = runner.read(args.job / 'job_plan.json')
    runner.integrity(args.job, plan)
    c_checkpoint = runner.audit(Path(plan['stage2']), 100, plan)
    ab_rows = runner.read(Path(plan['ab_job']) / 'three_seed_results.json')['results']
    c_result = runner.read(args.job / 'c_seed42_results.json')['results'][-1]
    rows = []
    for source in ab_rows + [c_result]:
        cp = c_checkpoint if source['variant'] == 'C' else core.torch.load(
            source['final_checkpoint'], map_location='cpu', weights_only=False)
        assert core.digest(source['final_checkpoint']) == source['final_checkpoint_sha256']
        assert cp['completed_epoch'] == 100
        assert cp['settings']['variant'] == source['variant']
        assert cp['settings']['seed'] == source['seed']
        best = max(cp['records'], key=lambda x: x['val_miou_global'])
        assert best['epoch'] == cp['best_epoch'] == source['best_epoch']
        weights = core.torch.load(source['best_weights'], map_location='cpu', weights_only=True)
        assert equal_tree(weights, cp['best_model_state'])
        model = build_c(source['seed']) if source['variant'] == 'C' else core.build_paired_model(source['variant'], source['seed'])
        model.load_state_dict(weights, strict=True)
        matrix = np.asarray(best['val_confusion'], dtype=np.int64)
        assert matrix.sum() == 500 * 768 * 1024
        tn, fp, fn, tp = map(int, matrix.ravel())
        assert abs(100 * tp / (tp + fp + fn) - source['best_sidewalk_iou_pct']) < 1e-10
        rows.append({**source, 'parameters': sum(p.numel() for p in model.parameters()),
            'sidewalk_precision_pct': 100 * tp / (tp + fp),
            'sidewalk_recall_pct': 100 * tp / (tp + fn),
            'sidewalk_dice_pct': 200 * tp / (2 * tp + fp + fn),
            'confusion_at_selected_epoch': matrix.tolist(),
            'best_weights_sha256': core.digest(source['best_weights']),
            'checkpoint_selection': 'earliest maximum global validation mIoU; all metrics from same epoch'})
        print('checkpoint verified', source['variant'], source['seed'], flush=True)
    core.json_write(output / 'verified_results.json', {'status': 'pass', 'results': rows,
        'new_inference_performed': False,
        'metric_provenance': 'precision/recall/Dice derived from saved global confusion; not a new held-out evaluation'})

    manifest = runner.read(Path(plan['stage2']) / 'sample_manifest.json')
    _, _, current = core.prepare('train', plan['expected_settings']['protocol'])
    assert current == manifest, 'Dataset metadata differs from completed training'
    inventory, summaries = [], {}
    ids, cities = {}, {}
    for split, expected in [('train', 2975), ('val', 500)]:
        samples = manifest[split]
        assert len(samples) == expected
        raw_counts = np.zeros(256, dtype=np.int64)
        resized_counts = np.zeros(256, dtype=np.int64)
        shapes, empty = Counter(), 0
        ids[split], cities[split] = set(), set()
        for index, item in enumerate(samples):
            path = core.PROJECT / item['label']
            blob = path.read_bytes()
            import hashlib
            sha = hashlib.sha256(blob).hexdigest()
            with Image.open(io.BytesIO(blob)) as image:
                raw = np.asarray(image)
                assert raw.ndim == 2 and raw.dtype == np.uint8
                resized = np.asarray(image.resize((1024, 768), Image.NEAREST))
            shapes[str(tuple(raw.shape))] += 1
            raw_counts += np.bincount(raw.ravel(), minlength=256)
            resized_counts += np.bincount(resized.ravel(), minlength=256)
            empty += int(not np.any(resized == 8))
            sample_id = path.name.replace('_gtFine_labelIds.png', '')
            assert sample_id not in ids[split]
            ids[split].add(sample_id)
            cities[split].add(path.parent.name)
            inventory.append({'split': split, 'sample_id': sample_id, 'label': item['label'], 'sha256': sha})
            if (index + 1) % 500 == 0 or index + 1 == expected:
                print('labels inventoried', split, index + 1, '/', expected, flush=True)
        total = int(resized_counts.sum())
        unknown = [i for i in range(256) if raw_counts[i] and i not in list(range(34)) + [255]]
        assert not unknown, f'Unknown original label IDs: {unknown}'
        summaries[split] = {'images': expected, 'cities': sorted(cities[split]),
            'raw_shapes': dict(shapes), 'resized_pixels': total,
            'empty_sidewalk_images_at_core_resolution': empty,
            'sidewalk_pixels_pct': float(100 * resized_counts[8] / total),
            'official_ignored_pixels_pct_at_core_resolution': float(100 * (total - resized_counts[VALID_IDS].sum()) / total),
            'raw_label_histogram': {str(i): int(n) for i, n in enumerate(raw_counts) if n},
            'resized_label_histogram': {str(i): int(n) for i, n in enumerate(resized_counts) if n}}
    assert not ids['train'] & ids['val'] and not cities['train'] & cities['val']
    assert len({x['sha256'] for x in inventory if x['split'] == 'train'} &
               {x['sha256'] for x in inventory if x['split'] == 'val'}) == 0
    # Recheck metadata after scanning. This does not claim images were content-hashed.
    assert core.prepare('train', plan['expected_settings']['protocol'])[2] == manifest
    core.json_write(output / 'label_inventory.json', {'source': LABEL_SOURCE,
        'official_nonignored_original_ids': VALID_IDS, 'files': inventory})
    core.json_write(output / 'label_audit.json', {'status': 'pass', 'splits': summaries,
        'split_sample_ids_disjoint': True, 'split_cities_disjoint': True,
        'no_exact_label_file_duplicates_across_splits': True,
        'image_contents_hashed': False, 'semantic_near_duplicate_check_performed': False,
        'historical_label_pixel_identity_claimed': False,
        'dataset_metadata_matches_completed_training': True,
        'current_core_policy': 'all original labels other than 8 map to background',
        'required_followup': 'evaluate SAME selected weights with official ignored pixels excluded; separate sensitivity table; do not overwrite core results'})
    runner.integrity(args.job, plan)
    first = runner.read(Path(plan['stage1']) / 'status.json')
    last = runner.read(Path(plan['stage2']) / 'status.json')
    core.json_write(output / 'review_verification.json', {'status': 'pass',
        'checked_at': datetime.now().astimezone().isoformat(), 'verified_completed_runs': len(rows),
        'C_full_audit_rechecked': True, 'old_artifact_integrity_rechecked': True,
        'C_training_process_wall_hours': (first['wall_seconds'] + last['wall_seconds']) / 3600,
        'script_sha256': core.digest(__file__), 'no_training_started': True})
    print(json.dumps({'status': 'pass', 'splits': {s: {k: v for k, v in r.items() if k.endswith('_pct') or k.endswith('_resolution')} for s, r in summaries.items()}}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
