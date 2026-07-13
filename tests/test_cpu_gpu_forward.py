import gc
import pytest
import torch

import feature_transformer
import features
import model
from feature_transformer import DoubleFeatureTransformerSlice, FeatureTransformerSlice


CUDA_KERNEL_AVAILABLE = torch.cuda.is_available() and feature_transformer.cp is not None


def test_portable_feature_transformer_preserves_sparse_semantics():
    layer = FeatureTransformerSlice(5, 3)
    with torch.no_grad():
        layer.weight.copy_(torch.arange(15, dtype=torch.float32).reshape(5, 3) / 10)
        layer.bias.copy_(torch.tensor([0.1, -0.2, 0.3]))

    indices = torch.tensor([[1, 1, -1, 2], [3, 0, 2, -1]], dtype=torch.int32)
    values = torch.tensor([[2.0, 0.5, 99.0, 7.0], [1.0, -2.0, 0.25, 0.0]], dtype=torch.float32)
    output = layer(indices, values)

    dense = torch.tensor(
        [[0.0, 2.5, 0.0, 0.0, 0.0], [-2.0, 0.0, 0.25, 1.0, 0.0]],
        dtype=torch.float32,
    )
    expected = dense @ layer.weight + layer.bias
    assert torch.allclose(output, expected)

    output.sum().backward()
    expected_weight_grad = dense.sum(dim=0).unsqueeze(1).expand_as(layer.weight)
    assert torch.allclose(layer.weight.grad, expected_weight_grad)
    assert torch.equal(layer.bias.grad, torch.full_like(layer.bias, 2.0))


def synthetic_batch(feature_count, device):
    indices = torch.tensor([[0, 1, 2, -1, -1, -1]], dtype=torch.int32, device=device)
    other_indices = torch.tensor([[3, 4, 5, -1, -1, -1]], dtype=torch.int32, device=device)
    values = torch.tensor([[1.0, 1.0, 0.5, 0.0, 0.0, 0.0]], dtype=torch.float32, device=device)
    assert int(indices.max()) < feature_count
    us = torch.ones((1, 1), dtype=torch.float32, device=device)
    them = torch.zeros((1, 1), dtype=torch.float32, device=device)
    bucket = torch.zeros(1, dtype=torch.long, device=device)
    return us, them, indices, values, other_indices, values.clone(), bucket, bucket.clone()


def run_full_nnue_forward_backward(device):
    feature_set = features.get_feature_set_from_name('HalfKAv2^')
    network = model.NNUE(feature_set=feature_set).to(device)
    batch = synthetic_batch(feature_set.num_features, device)
    output = network(*batch)
    loss = output.square().mean()
    loss.backward()

    assert output.shape == (1, 1)
    assert torch.isfinite(output).all()
    assert network.input.weight.grad is not None
    assert torch.isfinite(network.input.weight.grad).all()
    assert torch.count_nonzero(network.input.weight.grad).item() > 0
    del loss, output, batch, network
    gc.collect()


def test_full_nnue_forward_backward_on_cpu():
    run_full_nnue_forward_backward(torch.device('cpu'))


@pytest.mark.cuda_gate
@pytest.mark.skipif(
    not CUDA_KERNEL_AVAILABLE,
    reason='CUDA and matching CuPy are required for the optimized-kernel smoke test',
)
def test_full_nnue_forward_backward_on_cuda():
    run_full_nnue_forward_backward(torch.device('cuda'))
    torch.cuda.synchronize()
    torch.cuda.empty_cache()


@pytest.mark.cuda_gate
@pytest.mark.skipif(
    not CUDA_KERNEL_AVAILABLE,
    reason='CUDA and matching CuPy are required for portable/custom parity',
)
def test_custom_cuda_transformer_matches_portable_torch_path():
    torch.manual_seed(20260711)
    cpu_layer = FeatureTransformerSlice(7, 4)
    cuda_layer = FeatureTransformerSlice(7, 4).cuda()
    cuda_layer.load_state_dict(cpu_layer.state_dict())

    indices = torch.tensor([[1, 1, 4, -1], [6, 0, -1, -1]], dtype=torch.int32)
    values = torch.tensor([[1.0, 0.25, -0.5, 0.0], [2.0, 0.75, 0.0, 0.0]])

    cpu_output = cpu_layer(indices, values)
    cuda_output = cuda_layer(indices.cuda(), values.cuda())
    assert torch.allclose(cuda_output.cpu(), cpu_output, atol=1e-5, rtol=1e-5)

    cpu_output.sum().backward()
    cuda_output.sum().backward()
    assert torch.allclose(cuda_layer.weight.grad.cpu(), cpu_layer.weight.grad, atol=1e-5, rtol=1e-5)
    assert torch.allclose(cuda_layer.bias.grad.cpu(), cpu_layer.bias.grad, atol=1e-5, rtol=1e-5)


@pytest.mark.cuda_gate
@pytest.mark.skipif(
    not CUDA_KERNEL_AVAILABLE,
    reason='CUDA and matching CuPy are required for double-transformer parity',
)
def test_custom_cuda_double_transformer_matches_portable_outputs_and_gradients():
    torch.manual_seed(20260712)
    cpu_layer = DoubleFeatureTransformerSlice(7, 4)
    cuda_layer = DoubleFeatureTransformerSlice(7, 4).cuda()
    cuda_layer.load_state_dict(cpu_layer.state_dict())

    indices0 = torch.tensor(
        [[1, 1, -1, 4], [6, 0, 2, -1]], dtype=torch.int32
    )
    values0 = torch.tensor(
        [[1.0, 0.25, 99.0, 7.0], [2.0, -0.5, 0.75, 0.0]], dtype=torch.float32
    )
    indices1 = torch.tensor(
        [[3, 3, 4, -1], [5, -1, 1, 2]], dtype=torch.int32
    )
    values1 = torch.tensor(
        [[0.5, 1.5, -0.25, 0.0], [1.25, 0.0, 8.0, 9.0]], dtype=torch.float32
    )

    cpu_outputs = cpu_layer(indices0, values0, indices1, values1)
    cuda_outputs = cuda_layer(
        indices0.cuda(), values0.cuda(), indices1.cuda(), values1.cuda()
    )
    for cuda_output, cpu_output in zip(cuda_outputs, cpu_outputs):
        assert torch.allclose(cuda_output.cpu(), cpu_output, atol=1e-5, rtol=1e-5)

    grad0 = torch.tensor(
        [[1.0, -0.5, 0.25, 2.0], [-1.0, 0.75, 0.5, -0.25]], dtype=torch.float32
    )
    grad1 = torch.tensor(
        [[0.5, 1.0, -1.5, 0.25], [2.0, -0.25, 0.75, 1.5]], dtype=torch.float32
    )
    ((cpu_outputs[0] * grad0).sum() + (cpu_outputs[1] * grad1).sum()).backward()
    ((cuda_outputs[0] * grad0.cuda()).sum() + (cuda_outputs[1] * grad1.cuda()).sum()).backward()

    assert torch.allclose(cuda_layer.weight.grad.cpu(), cpu_layer.weight.grad, atol=1e-5, rtol=1e-5)
    assert torch.allclose(cuda_layer.bias.grad.cpu(), cpu_layer.bias.grad, atol=1e-5, rtol=1e-5)
