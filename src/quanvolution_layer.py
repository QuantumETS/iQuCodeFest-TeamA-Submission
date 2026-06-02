import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator
from qiskit.circuit.library import n_local
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def conv_circuit(n_qubit) -> QuantumCircuit:
    """Create a parameterized quantum circuit for convolutional feature extraction.

    Args:
        n_qubit: Number of qubits in the circuit (should match the number of features).

    Returns:
        A parameterized QuantumCircuit with the specified number of qubits.
    """

    qc = n_local(
        num_qubits=n_qubit,
        rotation_blocks=["ry"],
        entanglement_blocks="cz",
        entanglement="linear",
        reps=1,
        skip_final_rotation_layer=True,
    )
    logger.debug(f"\n{qc.draw()}")

    return qc


def estimate_expectations(qc: QuantumCircuit, input_data: list) -> list:
    """Simulate the quantum circuit and estimate expectation values for each qubit.

    Args:
        qc: The parameterized QuantumCircuit to simulate.
        input_data: A list of input values to encode into the circuit (length should match n_qubit).

    Returns:
        A list of expectation values for each qubit after running the circuit.
    """
    estimator = StatevectorEstimator()

    circuit = qc.assign_parameters(input_data)
    results = []
    for i in range(qc.num_qubits):
        observable_i = SparsePauliOp.from_sparse_list(
            [("Z", [i], 1.0)], num_qubits=qc.num_qubits
        )
        pubs = (circuit, observable_i)
        primitive_result = estimator.run(pubs=[pubs]).result()
        results.append(primitive_result[0].data.evs)
    return [res for res in results]


def quanv(image):
    """Convolves the input image with many applications of the same quantum circuit."""
    out = np.zeros((128, 128, 4))

    # Loop over the coordinates of the top-left pixel of 2X2 squares
    for j in range(0, 256, 2):
        for k in range(0, 256, 2):
            # Process a squared 2x2 region of the image with a quantum circuit
            q_results = estimate_expectations(
                conv_circuit(4),
                [
                    image[j, k, 0],
                    image[j, k + 1, 0],
                    image[j + 1, k, 0],
                    image[j + 1, k + 1, 0],
                ],
            )
            if k % 32 == 0:
                logger.info(
                    f"Processed block starting at ({j}, {k}), quantum results: {q_results}"
                )
            # Assign expectation values to different channels of the output pixel (j/2, k/2)
            for c in range(4):
                out[j // 2, k // 2, c] = q_results[c]
    return out


def mock_image():
    """Creates a mock 256x256 grayscale image for testing."""
    return np.random.rand(256, 256, 1)  # Shape (256, 256, 1) for grayscale


quanv_image = quanv(mock_image())
