"""Untrained width-reduction candidates; original ENet/CA files stay unchanged.

L0 uses widths (16,48,96,48,16); L1 adds the existing ReLU CA to L0.
Width scaling and CA are existing techniques, not claimed as new operators.
"""
import sys
from pathlib import Path
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code'))
from model_enet import InitialBlock, Bottleneck
from model_enet_ca import ENetCA
from components.ca_attention import CoordAtt


class CompactENet(ENetCA):
    def __init__(self, attention=False):
        nn.Module.__init__(self)
        self.initial = InitialBlock()
        self.stage1 = nn.Sequential(
            Bottleneck(16, 48, stride=2, dropout_prob=.01),
            *[Bottleneck(48, 48, dropout_prob=.01) for _ in range(4)])
        self.stage2 = nn.Sequential(
            Bottleneck(48, 96, stride=2, dropout_prob=.1),
            Bottleneck(96, 96, dilation=2, dropout_prob=.1),
            Bottleneck(96, 96, asymmetric=True, dropout_prob=.1),
            Bottleneck(96, 96, dilation=4, dropout_prob=.1))
        self.stage3 = nn.Sequential(
            Bottleneck(96, 96, dropout_prob=.1),
            Bottleneck(96, 96, dilation=2, dropout_prob=.1),
            Bottleneck(96, 96, asymmetric=True, dropout_prob=.1),
            Bottleneck(96, 96, dilation=4, dropout_prob=.1),
            Bottleneck(96, 96, dropout_prob=.1))
        self.ca = CoordAtt(channels=96, reduction=32) if attention else nn.Identity()
        self.up4 = nn.ConvTranspose2d(96, 48, 2, stride=2, bias=False)
        self.bn_up4 = nn.BatchNorm2d(48)
        self.prelu_up4 = nn.PReLU(48)
        self.stage4 = nn.Sequential(*[Bottleneck(48, 48, dropout_prob=.1) for _ in range(2)])
        self.up5 = nn.ConvTranspose2d(48, 16, 2, stride=2, bias=False)
        self.bn_up5 = nn.BatchNorm2d(16)
        self.prelu_up5 = nn.PReLU(16)
        self.stage5 = nn.Sequential(Bottleneck(16, 16, dropout_prob=.01))
        self.classifier = nn.Conv2d(16, 2, 1)


def build_compact(variant, seed=42):
    if variant not in ('L0', 'L1'):
        raise ValueError(variant)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        baseline = CompactENet(False)
        if variant == 'L0':
            return baseline
        torch.manual_seed(seed + 10000)
        model = CompactENet(True)
    state = model.state_dict()
    for name, tensor in baseline.state_dict().items():
        assert tensor.shape == state[name].shape
        state[name] = tensor.clone()
    model.load_state_dict(state, strict=True)
    return model
