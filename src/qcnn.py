"""QCNN circuit construction utilities.

The saved quanvolution features in ``data/preprocessed`` have shape
``(samples, height, width, channels)``. By default, ``build_qcnn_estimator``
uses the channel count as the QNN input dimension, which matches the four
channels produced by the current quanvolution setup.
"""

from pathlib import Path
from typing import Sequence

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import z_feature_map, zz_feature_map
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN


DEFAULT_FEATURE_PATH = Path("data/preprocessed/q_train_images.npy")


def _pool_circuit(params: Sequence) -> QuantumCircuit:
    """Create the two-qubit pooling unit used for each source/sink pair."""
    if len(params) != 3:
        raise ValueError("A pooling unit requires exactly three parameters.")

    circuit = QuantumCircuit(2)
    circuit.rz(-np.pi / 2, 1)
    circuit.cx(1, 0)
    circuit.rz(params[0], 0)
    circuit.ry(params[1], 1)
    circuit.cx(0, 1)
    circuit.ry(params[2], 1)
    return circuit


def pool_layer(
    sources: Sequence[int],
    sinks: Sequence[int],
    param_prefix: str = "p",
    *,
    insert_barriers: bool = False,
) -> QuantumCircuit:
    """Build a quantum pooling layer from source qubits into sink qubits.

    The returned circuit acts on ``len(sources) + len(sinks)`` local qubits.
    After this layer, callers should continue composing later layers only on
    the sink qubits, which is how the circuit dimension is reduced for QCNNs.
    """
    if len(sources) != len(sinks):
        raise ValueError("sources and sinks must contain the same number of qubits.")
    if not sources:
        raise ValueError("pool_layer requires at least one source/sink pair.")

    qubits = list(sources) + list(sinks)
    if sorted(qubits) != list(range(len(qubits))):
        raise ValueError(
            "sources and sinks must be local qubit indices covering 0..N-1."
        )

    num_qubits = len(qubits)
    params = ParameterVector(param_prefix, length=len(sources) * 3)
    layer = QuantumCircuit(num_qubits, name="Pooling Layer")

    param_index = 0
    for source, sink in zip(sources, sinks):
        layer.compose(
            _pool_circuit(params[param_index : param_index + 3]),
            [source, sink],
            inplace=True,
        )
        if insert_barriers:
            layer.barrier()
        param_index += 3

    instruction = layer.to_instruction()
    circuit = QuantumCircuit(num_qubits)
    circuit.append(instruction, range(num_qubits))
    return circuit


def _quanvolution_unit(params: Sequence) -> QuantumCircuit:
    """Create the two-qubit trainable unit used by the QCNN conv layers."""
    if len(params) != 3:
        raise ValueError("A quanvolution unit requires exactly three parameters.")

    circuit = QuantumCircuit(2)
    circuit.rz(-np.pi / 2, 1)
    circuit.cx(1, 0)
    circuit.rz(params[0], 0)
    circuit.ry(params[1], 1)
    circuit.cx(0, 1)
    circuit.ry(params[2], 1)
    circuit.cx(1, 0)
    circuit.rz(np.pi / 2, 0)
    return circuit


def _quanvolution_layer(
    num_qubits: int,
    param_prefix: str,
    *,
    insert_barriers: bool = False,
) -> QuantumCircuit:
    """Build the trainable two-local quanvolution layer for active qubits."""
    if num_qubits < 2:
        raise ValueError("A quanvolution layer requires at least two qubits.")

    qubits = list(range(num_qubits))
    params = ParameterVector(param_prefix, length=num_qubits * 3)
    layer = QuantumCircuit(num_qubits, name="Quanvolution Layer")

    param_index = 0
    for q1, q2 in zip(qubits[0::2], qubits[1::2]):
        layer.compose(
            _quanvolution_unit(params[param_index : param_index + 3]),
            [q1, q2],
            inplace=True,
        )
        if insert_barriers:
            layer.barrier()
        param_index += 3

    for q1, q2 in zip(qubits[1::2], qubits[2::2] + [0]):
        layer.compose(
            _quanvolution_unit(params[param_index : param_index + 3]),
            [q1, q2],
            inplace=True,
        )
        if insert_barriers:
            layer.barrier()
        param_index += 3

    instruction = layer.to_instruction()
    circuit = QuantumCircuit(num_qubits)
    circuit.append(instruction, qubits)
    return circuit


def _infer_num_inputs(feature_path: Path) -> int:
    """Infer the QNN input dimension from a saved quanvolution feature tensor."""
    features = np.load(feature_path, mmap_mode="r")
    if features.ndim < 2:
        raise ValueError(
            f"Expected a feature tensor with at least 2 dimensions, got {features.shape}."
        )
    return int(features.shape[-1])


def _validate_power_of_two(num_qubits: int) -> None:
    if num_qubits < 2:
        raise ValueError("num_inputs must be at least 2.")
    if num_qubits & (num_qubits - 1):
        raise ValueError("num_inputs must be a power of two for repeated pooling.")


def _feature_map(num_inputs: int, kind: str) -> QuantumCircuit:
    if kind == "z":
        return z_feature_map(num_inputs)
    if kind == "zz":
        return zz_feature_map(feature_dimension=num_inputs, reps=1)
    raise ValueError('feature_map must be either "z" or "zz".')


def build_qcnn_estimator(
    num_inputs: int | None = None,
    *,
    feature_path: Path | str = DEFAULT_FEATURE_PATH,
    feature_map: str = "z",
    estimator: StatevectorEstimator | None = None,
    input_gradients: bool = False,
    insert_barriers: bool = False,
) -> EstimatorQNN:
    """Compose alternating quanvolution/pooling layers into an EstimatorQNN.

    If ``num_inputs`` is omitted, it is inferred from the final axis of
    ``feature_path``. For the existing ``q_train_images.npy`` file, this
    creates a four-input QCNN that reduces to one measured output qubit.
    """
    feature_path = Path(feature_path)
    if num_inputs is None:
        num_inputs = _infer_num_inputs(feature_path)
    _validate_power_of_two(num_inputs)

    fmap = _feature_map(num_inputs, feature_map)
    ansatz = QuantumCircuit(num_inputs, name="QCNN Ansatz")
    active_qubits = list(range(num_inputs))
    layer_index = 1

    while len(active_qubits) > 1:
        ansatz.compose(
            _quanvolution_layer(
                len(active_qubits),
                f"c{layer_index}",
                insert_barriers=insert_barriers,
            ),
            active_qubits,
            inplace=True,
        )

        half = len(active_qubits) // 2
        sources = list(range(half))
        sinks = list(range(half, len(active_qubits)))
        ansatz.compose(
            pool_layer(
                sources,
                sinks,
                f"p{layer_index}",
                insert_barriers=insert_barriers,
            ),
            active_qubits,
            inplace=True,
        )

        active_qubits = active_qubits[half:]
        layer_index += 1

    circuit = QuantumCircuit(num_inputs)
    circuit.compose(fmap, range(num_inputs), inplace=True)
    circuit.compose(ansatz, range(num_inputs), inplace=True)

    observable = SparsePauliOp.from_sparse_list(
        [("Z", [active_qubits[0]], 1.0)],
        num_qubits=num_inputs,
    )

    return EstimatorQNN(
        circuit=circuit.decompose(),
        estimator=estimator or StatevectorEstimator(),
        observables=observable,
        input_params=tuple(fmap.parameters),
        weight_params=tuple(ansatz.parameters),
        input_gradients=input_gradients,
    )


if __name__ == "__main__":
    from argparse import ArgumentParser
    from types import SimpleNamespace

    arg_parser = ArgumentParser(description="Build a QCNN from quanvolution features.")
    arg_parser.add_argument(
        "--samples",
        type=int,
        default=4,
        help="Number of samples to use for training/testing.",
    )
    arg_parser.add_argument(
        "--preprocess",
        action="store_true",
        help="Run quanvolutional feature extraction instead of using saved features.",
    )
    arg_parser.add_argument(
        "--feature-map",
        choices=("z", "zz"),
        default="z",
        help="Feature map to use in the QCNN.",
    )
    arg_parser.add_argument(
        "--plot",
        action="store_true",
        help="Save the QCNN circuit visualization to disk.",
    )

    args = arg_parser.parse_args()

    if args.preprocess:
        from quanvolution_layer import RANDOM_SEED, extract_quantum_feature_maps

        extract_quantum_feature_maps(
            SimpleNamespace(
                preprocess=True,
                samples=args.samples,
                seed=RANDOM_SEED,
            )
        )

    qnn = build_qcnn_estimator(feature_map=args.feature_map)

    if args.plot:
        Path("figures").mkdir(parents=True, exist_ok=True)
        qnn.circuit.draw("mpl", filename="figures/qcnn_circuit.png")

    print(
        "QCNN built with "
        f"{qnn.circuit.num_qubits} qubits, "
        f"{len(qnn.input_params)} input parameters, "
        f"{len(qnn.weight_params)} weight parameters."
    )
