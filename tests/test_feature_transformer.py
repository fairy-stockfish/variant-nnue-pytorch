"""
Checks the device-agnostic (CPU) feature transformer against a naive reference
implementation, for both the forward pass and the gradients.

Run with: python tests/test_feature_transformer.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from feature_transformer import DoubleFeatureTransformerSlice  # noqa: E402

BATCH_SIZE = 16
NUM_INPUTS = 32
NUM_OUTPUTS = 64
MAX_ACTIVE_FEATURES = 48
MAX_ERROR = 1e-4


def make_indices():
    # Active features, followed by a -1 terminator and then arbitrary (possibly
    # uninitialized) values, mirroring what the C++ data loader produces.
    active = MAX_ACTIVE_FEATURES * 3 // 4
    return torch.cat([
        (torch.rand(BATCH_SIZE, active) * NUM_INPUTS).to(dtype=torch.int32),
        torch.full((BATCH_SIZE, 1), -1, dtype=torch.int32),
        torch.randint(-5, 10**9, (BATCH_SIZE, MAX_ACTIVE_FEATURES - active - 1), dtype=torch.int32),
    ], dim=1)


def reference(indices, values, weight, bias):
    inputs = torch.zeros(BATCH_SIZE, NUM_INPUTS)
    for i in range(BATCH_SIZE):
        for j in range(MAX_ACTIVE_FEATURES):
            if indices[i, j] < 0:
                break
            inputs[i, indices[i, j]] += values[i, j]
    return inputs @ weight + bias


def main():
    torch.manual_seed(0)
    layer = DoubleFeatureTransformerSlice(NUM_INPUTS, NUM_OUTPUTS)
    weight = layer.weight.detach().clone().requires_grad_()
    bias = layer.bias.detach().clone().requires_grad_()

    indices0, indices1 = make_indices(), make_indices()
    values0 = torch.rand(BATCH_SIZE, MAX_ACTIVE_FEATURES)
    values1 = torch.rand(BATCH_SIZE, MAX_ACTIVE_FEATURES)

    out0, out1 = layer(indices0, values0, indices1, values1)
    ref0 = reference(indices0, values0, weight, bias)
    ref1 = reference(indices1, values1, weight, bias)
    assert (out0 - ref0).abs().max() < MAX_ERROR
    assert (out1 - ref1).abs().max() < MAX_ERROR

    (out0.sum() + 2 * out1.sum()).backward()
    (ref0.sum() + 2 * ref1.sum()).backward()
    assert (layer.weight.grad - weight.grad).abs().max() < MAX_ERROR
    assert (layer.bias.grad - bias.grad).abs().max() < MAX_ERROR

    print('Feature transformer tests passed.')


if __name__ == '__main__':
    main()
