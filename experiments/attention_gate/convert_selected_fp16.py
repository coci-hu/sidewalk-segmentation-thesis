"""CPU graph conversion only; preserve verified FP32 graphs and selected weights."""
import argparse
import collections
import datetime
import hashlib
import json
import os
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import onnx
import onnxruntime as ort
from onnxruntime.transformers import float16
from onnxruntime.transformers.onnx_model import OnnxModel

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def write(p, obj):
    Path(p).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def main(a):
    protocol_path = HERE / 'deployment_fp16_protocol.json'
    protocol = read(protocol_path)
    manifest = read(a.models)
    accepted_path = ROOT / 'results/deployment_cuda_fp32_preserved_bn.json'
    accepted = read(accepted_path)
    assert accepted['status'] == 'pass' and accepted['full500_acceptance_passed']
    assert ort.__version__ == protocol['conversion']['required_onnxruntime_version']
    assert {m['variant'] for m in manifest['models']} == set('ABC') and len(manifest['models']) == 3
    assert manifest['execution']['disabled_optimizers'] == protocol['execution']['disabled_optimizers']
    out = a.output.resolve()
    assert out.is_relative_to(HERE) and out != HERE and not out.exists()
    import inspect
    sources = {str(p): digest(p) for p in (Path(__file__), protocol_path, Path(float16.__file__),
               Path(inspect.getfile(OnnxModel)), a.models, accepted_path)}
    out.mkdir(parents=True)
    write(out / 'plan.json', dict(created_at=datetime.datetime.now().astimezone().isoformat(),
          protocol=protocol, sources=sources, converter_default_block_list=float16.DEFAULT_OP_BLOCK_LIST,
          deterministic_topological_sort=True, CUDA_inference=False, training_modified=False))
    entries = []
    for item in manifest['models']:
        original = Path(item['onnx'])
        row = next(r for r in accepted['results'] if r['variant'] == item['variant'])
        assert row['seed'] == item['seed'] == 42 and row['selected_epoch'] == item['selected_epoch']
        assert row['weights_sha256'] == item['weights_sha256'] == digest(item['weights'])
        assert row['onnx_sha256'] == item['onnx_sha256'] == digest(original)
        graph = onnx.load(str(original))
        converted = float16.convert_float_to_float16(graph, keep_io_types=True,
            min_positive_val=5.96e-8, max_finite_val=65504.0, disable_shape_infer=False,
            op_block_list=list(float16.DEFAULT_OP_BLOCK_LIST), force_fp16_initializers=False)
        # This installed converter appends IO casts after consumers. Sorting
        # changes serialization order only, not graph edges, tensors or operators.
        before_sort = sorted(n.SerializeToString() for n in converted.graph.node)
        OnnxModel(converted).topological_sort(is_deterministic=True)
        assert sorted(n.SerializeToString() for n in converted.graph.node) == before_sort
        onnx.checker.check_model(converted)
        target = out / (item['name'] + '_fp16_mixed.onnx')
        onnx.save(converted, str(target))
        dtypes = collections.Counter(onnx.TensorProto.DataType.Name(t.data_type) for t in converted.graph.initializer)
        assert dtypes['FLOAT16'] > 0
        entry = {key: item[key] for key in ('name', 'variant', 'seed', 'selected_epoch', 'weights', 'weights_sha256')}
        entry.update(onnx=str(target), onnx_sha256=digest(target),
            fp32_onnx=str(original), fp32_onnx_sha256=item['onnx_sha256'],
            initializer_types=dict(dtypes), graph_bytes=target.stat().st_size,
            fp32_graph_bytes=original.stat().st_size,
            cast_nodes=sum(n.op_type == 'Cast' for n in converted.graph.node))
        entries.append(entry)
        assert digest(original) == item['onnx_sha256']
        print(item['name'], 'converted; inference not yet checked', flush=True)
    for path, value in sources.items():
        assert digest(path) == value
    result = {key: manifest[key] for key in ('height', 'width', 'mean', 'std', 'execution')}
    result.update(models=entries, protocol_sha256=digest(protocol_path),
                  precision='FP16 mixed internal graph, FP32 IO',
                  scope='Structural conversion only; no model execution, accuracy or FPS claim')
    write(out / 'models.json', result)
    write(out / 'verification.json', dict(status='pass', models=3, source_hashes=sources,
          checker_passed=True, CUDA_inference=False, full500_acceptance_passed=False,
          original_graphs_and_weights_unchanged=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args())
