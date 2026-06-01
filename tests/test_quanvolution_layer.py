"""Tests for the quantum convolution layer."""

import torch

from quanvolution_layer import QuanvolutionLayer


def test_quanvolution_layer_shape_smoke():
    layer = QuanvolutionLayer(image_size=4)
    inputs = torch.rand(2, 1, 4, 4)

    outputs = layer(inputs)

    assert outputs.shape == (2, 4, 2, 2)


def test_quanvolution_layer_backward_smoke():
    layer = QuanvolutionLayer(image_size=4)
    inputs = torch.rand(1, 1, 4, 4, requires_grad=True)

    outputs = layer(inputs)
    loss = outputs.mean()
    loss.backward()

    assert inputs.grad is not None
    assert layer.quantum_layer.weight.grad is not None
