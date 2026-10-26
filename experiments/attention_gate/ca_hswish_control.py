"""Step 6 C: isolate the paper-form h-swish activation in the existing CA.

Keep the legacy W-then-H mean layout and every trainable state name. Local
conv_h acts along width, whereas the author's conv_h acts along height; an
axis-aware weight swap is needed when comparing against the official code.
This naming difference alone is not evidence of a broken attention formula.

Reference: houqb/CoordAttention, coordatt.py at
a3f3490ceba779a8443a231d0f792866bd24436d (CVPR 2021).
No frozen model or training source is edited by this module.
"""
import torch.nn as nn
import common_runner as core


class PaperHardSigmoid(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU6(inplace=True)

    def forward(self, x):
        return self.relu(x + 3) / 6


class PaperHardSwish(nn.Module):
    def __init__(self):
        super().__init__()
        self.sigmoid = PaperHardSigmoid()

    def forward(self, x):
        return x * self.sigmoid(x)


def build_c(seed):
    """Same paired B initialization, with only its parameter-free act replaced.

    This constructs a new model; it does not load a trained B checkpoint.
    The original constructor already preserves the calling CPU RNG state.
    """
    if seed not in (42, 43, 44):
        raise ValueError('Seed outside the fixed comparison')
    model = core.build_paired_model('B', seed)
    model.ca.act = PaperHardSwish()
    return model


def author_axis_state(local_state):
    """Map local width/height branch names to the official height/width names."""
    mapped = {}
    for name, value in local_state.items():
        if name.startswith('conv_h.'):
            key = name.replace('conv_h.', 'conv_w.', 1)
        elif name.startswith('conv_w.'):
            key = name.replace('conv_w.', 'conv_h.', 1)
        else:
            key = name
        mapped[key] = value.detach().clone()
    return mapped
