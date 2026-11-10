"""Copy verified A/L0 artifacts into a portable folder; never modify originals."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[2]

def package(source, target):
    config=json.loads(source.read_text(encoding='utf-8'))
    accepted=json.loads((ROOT/'results/compact_deployment_validation_20261107.json').read_text(encoding='utf-8'))
    assert accepted['status']=='pass'
    assert {m['variant'] for m in config['models']}=={'A','L0'}
    copies=[]
    for item in config['models']:
        row=next(r for r in accepted['results'] if r['variant']==item['variant'])
        assert item['seed']==42 and item['selected_epoch']==row['selected_epoch']
        for kind in ('onnx','weights'):
            origin=Path(item[kind])
            if not origin.is_absolute():origin=source.parent/origin
            digest=hashlib.sha256(origin.read_bytes()).hexdigest()
            assert digest==item[kind+'_sha256']==row[kind+'_sha256'], str(origin)
            name=item['variant']+'_seed42'+('.onnx' if kind=='onnx' else '.pth')
            copies.append((origin,name,digest));item[kind]=name
    target.mkdir(parents=True,exist_ok=False)
    for origin,name,digest in copies:
        shutil.copyfile(origin,target/name)
        assert hashlib.sha256((target/name).read_bytes()).hexdigest()==digest
    config['scope']='Portable copy of A/L0 seed42 FP32 models; full500 CUDA acceptance in results/compact_deployment_validation_20261107.json.'
    (target/'models.json').write_text(json.dumps(config,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('Packaged verified A/L0 assets:',target)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,default=Path(__file__).resolve().parent/'models')
    a=p.parse_args();package(a.source.resolve(),a.output.resolve())
