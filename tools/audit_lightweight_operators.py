"""Audit operators excluded from the declared Conv/Linear FLOPs convention.

This inventory does not silently assign zero FLOPs to unsupported arithmetic.
It complements, and never replaces, the independently cross-checked profile.
"""
import argparse
import collections
import hashlib
import json
import sys
from pathlib import Path
import torch
from torch.utils._python_dispatch import TorchDispatchMode

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'code'),str(ROOT/'experiments/attention_gate')]
from model_enet import ENet
from model_enet_ca import ENetCA
from compact_candidate import build_compact


class Inventory(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.counts=collections.Counter()
    def __torch_dispatch__(self,func,types,args=(),kwargs=None):
        self.counts[str(func)]+=1
        return func(*args,**(kwargs or {}))


def main(output):
    torch.set_num_threads(1)
    assert not output.exists()
    models={'A':ENet(2),'B':ENetCA(2),'L0':build_compact('L0'),'L1':build_compact('L1')}
    results={}
    for name,model in models.items():
        model.eval()
        x=torch.zeros(1,3,768,1024)
        with torch.no_grad(),Inventory() as audit:
            y=model(x)
        assert list(y.shape)==[1,2,768,1024] and torch.isfinite(y).all()
        results[name]=dict(sorted(audit.counts.items()))
    profile=ROOT/'results/lightweight_profile_20261101_v2.json'
    value=dict(status='operator_inventory_complete',input=[1,3,768,1024],eval=True,
        device='cpu',torch=torch.__version__,source_profile_sha256=hashlib.sha256(profile.read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),operators=results,
        candidate_sha256=hashlib.sha256((ROOT/'experiments/attention_gate/compact_candidate.py').read_bytes()).hexdigest(),
        full_operator_flops_claim=False,
        counted='Conv2d, ConvTranspose2d, Linear only; 1 MAC = 2 FLOPs, bias addition excluded',
        interpretation='Inventory counts operator invocations, not arithmetic operations or runtime. Excluded arithmetic is uncounted, not free. All four models use identical accounting.',
        excluded_arithmetic=['batch normalization','PReLU/ReLU/sigmoid','pooling and reductions','residual addition','attention elementwise multiplication','bilinear interpolation'],
        movement_not_arithmetic=['concatenation','padding','split','transpose/permute','view/alias','allocation'],
        thesis_table_label='GFLOPs (Conv/ConvTranspose/Linear, 2 FLOPs per MAC)')
    with output.open('x',encoding='utf-8') as f: json.dump(value,f,ensure_ascii=False,indent=2)
    print(json.dumps(results,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True,type=Path)
    main(p.parse_args().output)
