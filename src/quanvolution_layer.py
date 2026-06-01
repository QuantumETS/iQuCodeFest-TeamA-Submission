"""Quantum convolution layers built with Qiskit and PyTorch."""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor, nn
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.connectors.torch_connector import TorchConnector
from qiskit_machine_learning.neural_networks.estimator_qnn import EstimatorQNN


def _build_observables(num_qubits: int) -> list[SparsePauliOp]:
    """Create the output observables for the quantum convolution layer.

    Args:
        num_qubits: Number of qubits in the circuit and output channels.

    Returns:
        A list of Pauli-Z observables, one per qubit.
    """

    observables: list[SparsePauliOp] = []
    for qubit in range(num_qubits):
        pauli_label = ["I"] * num_qubits
        pauli_label[num_qubits - qubit - 1] = "Z"
        observables.append(SparsePauliOp.from_list([("".join(pauli_label), 1.0)]))
    return observables


def _build_quantum_circuit(num_qubits: int):
    """Build the parameterized circuit used by the quantum convolution layer.

    Args:
        num_qubits: Number of qubits used for patch encoding and measurement.

    Returns:
        A tuple containing the quantum circuit, the input parameters, and the
        trainable weight parameters.
    """

    input_params = ParameterVector("x", num_qubits)
    weight_params = ParameterVector("theta", 2 * num_qubits)

    circuit = QuantumCircuit(num_qubits)

    for qubit in range(num_qubits):
        circuit.ry(input_params[qubit], qubit)

    for qubit in range(num_qubits):
        circuit.ry(weight_params[qubit], qubit)

    for qubit in range(num_qubits - 1):
        circuit.cx(qubit, qubit + 1)
    circuit.cx(num_qubits - 1, 0)

    for qubit in range(num_qubits):
        circuit.rz(weight_params[num_qubits + qubit], qubit)

    return circuit, input_params, weight_params


class QuanvolutionLayer(nn.Module):
    """Quantum convolution layer for grayscale image tensors.

    The default contract matches a 2x2 scan over 256x256 grayscale images and
    returns 4 quantum feature channels at half the input resolution.

    Example:
        >>> import torch
        >>> layer = QuanvolutionLayer(image_size=256)
        >>> inputs = torch.rand(1, 1, 256, 256)
        >>> outputs = layer(inputs)
        >>> outputs.shape
        torch.Size([1, 4, 128, 128])
    """

    def __init__(
        self,
        image_size: int = 256,
        patch_size: int = 2,
        stride: int = 2,
        num_qubits: int | None = None,
        initial_weights: np.ndarray | Tensor | None = None,
    ) -> None:
        """Initialize the quantum convolution layer.

        Args:
            image_size: Height and width of the input image tensor.
            patch_size: Height and width of each square quantum patch.
            stride: Step size used while scanning the image.
            num_qubits: Optional override for the number of qubits.
            initial_weights: Optional initial parameter tensor for the quantum
                kernel.

        Raises:
            ValueError: If the geometry or qubit count is inconsistent.
        """
        super().__init__()

        if image_size < patch_size:
            raise ValueError("image_size must be greater than or equal to patch_size")
        if (image_size - patch_size) % stride != 0:
            raise ValueError(
                "image_size, patch_size, and stride must produce an even scan"
            )

        resolved_num_qubits = (
            patch_size * patch_size if num_qubits is None else num_qubits
        )
        if resolved_num_qubits != patch_size * patch_size:
            raise ValueError("num_qubits must match patch_size * patch_size")

        self.image_size = image_size
        self.patch_size = patch_size
        self.stride = stride
        self.num_qubits = resolved_num_qubits
        self.output_channels = resolved_num_qubits
        self.output_height = (image_size - patch_size) // stride + 1
        self.output_width = self.output_height

        circuit, input_params, weight_params = _build_quantum_circuit(self.num_qubits)
        observables = _build_observables(self.num_qubits)

        self.qnn = EstimatorQNN(
            circuit=circuit,
            input_params=list(input_params),
            weight_params=list(weight_params),
            observables=observables,
            input_gradients=True,
        )

        if initial_weights is None:
            initial_weights = np.zeros(self.qnn.num_weights, dtype=np.float32)
        elif isinstance(initial_weights, torch.Tensor):
            initial_weights = initial_weights.detach().cpu().numpy()

        self.quantum_layer = TorchConnector(self.qnn, initial_weights=initial_weights)

    def _validate_input(self, inputs: Tensor) -> Tensor:
        """Validate and normalize input tensors.

        Args:
            inputs: Input tensor shaped as `[H, W]`, `[B, H, W]`, or
                `[B, 1, H, W]`.

        Returns:
            A batch-first tensor with shape `[B, 1, H, W]`.

        Raises:
            TypeError: If the tensor is not floating point.
            ValueError: If the tensor shape does not match the layer contract.
        """

        if not torch.is_floating_point(inputs):
            raise TypeError("inputs must be a floating point tensor")

        if inputs.dim() == 2:
            inputs = inputs.unsqueeze(0).unsqueeze(0)
        elif inputs.dim() == 3:
            inputs = inputs.unsqueeze(1)
        elif inputs.dim() != 4:
            raise ValueError(
                "inputs must have shape [H, W], [B, H, W], or [B, 1, H, W]"
            )

        if inputs.shape[1] != 1:
            raise ValueError("QuanvolutionLayer expects a single input channel")

        if inputs.shape[2] != self.image_size or inputs.shape[3] != self.image_size:
            raise ValueError(
                f"Expected spatial size {self.image_size}x{self.image_size}, "
                f"got {inputs.shape[2]}x{inputs.shape[3]}"
            )

        return inputs

    def forward(self, inputs: Tensor) -> Tensor:
        """Apply the quanvolution kernel to an image batch.

        Args:
            inputs: Batch of grayscale image tensors.

        Returns:
            A tensor with shape `[B, 4, image_size / 2, image_size / 2]` when
            using the default 2x2 stride-2 configuration.
        """

        inputs = self._validate_input(inputs)
        batch_size = inputs.shape[0]

        patches = inputs.unfold(2, self.patch_size, self.stride).unfold(
            3, self.patch_size, self.stride
        )
        patches = patches.contiguous().view(
            batch_size, 1, self.output_height, self.output_width, -1
        )
        patches = patches[:, 0].reshape(
            batch_size * self.output_height * self.output_width, -1
        )

        quantum_dtype = self.quantum_layer.weight.dtype
        patch_outputs = self.quantum_layer(patches.to(dtype=quantum_dtype))
        patch_outputs = patch_outputs.to(dtype=inputs.dtype)

        patch_outputs = patch_outputs.view(
            batch_size, self.output_height, self.output_width, self.output_channels
        )
        patch_outputs = patch_outputs.permute(0, 3, 1, 2).contiguous()
        return patch_outputs


__all__ = ["QuanvolutionLayer"]
