"""Prospective D/E candidates, not yet authorized as validated training runners.

D fixes alpha=.1, E learns one sigmoid-logit scalar initialized at the same .1.
Shared CA tensors/layout/h-swish are identical to C. No training source is edited.
"""
import math
import torch
from torch import nn
from ca_hswish_control import build_c


class MixedCA(nn.Module):
    def __init__(self, reference, learned=False, alpha=0.1):
        super().__init__()
        if not 0 < alpha < 1:
            raise ValueError('Initialization must be strictly between0and1')
        for name in ('conv1', 'bn1', 'act', 'conv_h', 'conv_w'):
            setattr(self, name, getattr(reference, name))
        self.learned = learned
        if learned:
            self.beta = nn.Parameter(torch.tensor(math.log(alpha / (1 - alpha)), dtype=torch.float32))
        else:
            # Match the same float32 logit->sigmoid rounding at initialization.
            value = torch.tensor(math.log(alpha / (1 - alpha)), dtype=torch.float32).sigmoid()
            self.register_buffer('fixed_alpha', value)

    def alpha(self):
        return self.beta.sigmoid() if self.learned else self.fixed_alpha

    def forward(self, x):
        h, w = x.shape[2:]
        xh = x.mean(dim=2, keepdim=True)
        xw = x.mean(dim=3, keepdim=True).permute(0, 1, 3, 2)
        y = self.act(self.bn1(self.conv1(torch.cat([xh, xw], dim=3))))
        yh, yw = torch.split(y, [w, h], dim=3)
        aw = self.conv_w(yw.permute(0, 1, 3, 2)).sigmoid()
        ah = self.conv_h(yh).sigmoid()
        standard = x * aw * ah
        a = self.alpha()
        return (1 - a) * x + a * standard


def build_candidate(variant, seed):
    if variant not in ('D', 'E'):
        raise ValueError('Expected D or E')
    model = build_c(seed)
    model.ca = MixedCA(model.ca, learned=variant == 'E')
    return model


def parameter_groups(model, weight_decay=0.01):
    """Explicitly exclude the E logit scalar from decay toward alpha=.5."""
    ordinary = [p for name, p in model.named_parameters() if name != 'ca.beta']
    groups = [{'params': ordinary, 'weight_decay': weight_decay}]
    if model.ca.learned:
        groups.append({'params': [model.ca.beta], 'weight_decay': 0.0})
    return groups
