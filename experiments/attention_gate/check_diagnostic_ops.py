"""Small meaningful controls for intervention math and attention displays."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import common_runner as cr
from components.ca_attention import CoordAtt
from diagnose_attention import InterventionCA, norm_channels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(cr.HERE) or output.exists():
        raise ValueError("Use a fresh experiment output")
    cr.configure(42)
    module = CoordAtt(channels=8).eval()
    wrapper = InterventionCA(module, 2e-5).eval()
    conditions = [item["id"] for item in json.loads((cr.HERE / "diagnostic_protocol_v1.json").read_text(encoding="utf-8"))["interventions"]]
    details = []
    with torch.inference_mode():
        for case, x in (("rectangular", torch.randn(1, 8, 7, 11)), ("zero_features", torch.zeros(1, 8, 7, 11))):
            wrapper.begin("fixed_synthetic_case")
            for condition in conditions:
                wrapper.mode = condition
                y = wrapper(x)
                assert y.shape == x.shape and bool(torch.isfinite(y).all())
                if condition == "original":
                    assert torch.equal(y, module(x))
                    original_norm = norm_channels(y)
                if condition.startswith("energy_"):
                    assert torch.allclose(norm_channels(y), original_norm, rtol=2e-5, atol=1e-7)
                details.append({"case": case, "condition": condition, "finite": True,
                    "energy_relative_error": wrapper.max_energy_error})
            one = wrapper.shuffled_gate(wrapper.horizontal, wrapper.vertical, 42)
            repeated = wrapper.shuffled_gate(wrapper.horizontal, wrapper.vertical, 42)
            two = wrapper.shuffled_gate(wrapper.horizontal, wrapper.vertical, 43)
            assert torch.equal(one, repeated)
            assert torch.equal(torch.sort(wrapper.gate.flatten(2), dim=2).values, torch.sort(one.flatten(2), dim=2).values)
            if case == "rectangular":
                assert not torch.equal(one, two)
    horizontal = np.array([0.9, 0.1])[:, None, None]
    vertical = np.array([0.9, 0.1])[:, None, None]
    correct = float((horizontal * vertical).mean())
    legacy = float(horizontal.mean() * vertical.mean())
    assert abs(correct - 0.41) < 1e-12 and abs(legacy - 0.25) < 1e-12
    cr.json_write(output, {"status": "pass", "conditions": details,
        "shuffle_distribution_exact_and_seed_reproducible": True,
        "heatmap_counterexample": {"mean_of_channel_products": correct, "product_of_axis_means": legacy},
        "source_sha256": {"diagnose_attention.py": cr.digest(cr.HERE / "diagnose_attention.py"),
                          "diagnostic_protocol_v1.json": cr.digest(cr.HERE / "diagnostic_protocol_v1.json")}})
    print("Diagnostic operation checks passed")


if __name__ == "__main__":
    main()
