"""Separate, prospectively declared FP16 gates; never alter FP32 tolerances."""
import json
from pathlib import Path
import numpy as np
from deployment_equivalence import Agreement as FP32Agreement

PROTOCOL = Path(__file__).with_name('deployment_fp16_protocol.json')
LIMITS = json.loads(PROTOCOL.read_text(encoding='utf-8'))['acceptance_limits']


class Agreement(FP32Agreement):
    def add(self, reference, exported, target, image_id):
        # Inherits finite-output, duplicate-ID, shape and binary-target guards.
        super().add(reference, exported, target, image_id)
        row = self.records[-1]
        row['logits_allclose_fp32_diagnostic'] = row['logits_allclose']
        row['logits_allclose'] = bool(np.allclose(
            exported, reference, rtol=LIMITS['logit_rtol'], atol=LIMITS['logit_atol']))

    def summary(self):
        report = super().summary()
        # Semantic thresholds deliberately equal the frozen FP32 values.
        from deployment_equivalence import LIMITS as fp32_limits
        for key in ('argmax_disagreement_fraction', 'metric_difference_pp'):
            assert LIMITS[key] == fp32_limits[key]
        report['limits'] = dict(LIMITS)
        report['protocol_id'] = json.loads(PROTOCOL.read_text(encoding='utf-8'))['protocol_id']
        report['signed_metric_difference_pp'] = {
            key: report['metrics']['onnx'][key] - report['metrics']['pytorch'][key]
            if report['metrics']['onnx'][key] is not None and report['metrics']['pytorch'][key] is not None else None
            for key in ('sidewalk_iou_pct', 'boundary_iou_pct')}
        return report
