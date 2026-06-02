import numpy as np
from typing import List, Tuple, Union
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator
from qiskit.circuit.library import n_local, zz_feature_map
import logging
import matplotlib.pyplot as plt
from pathlib import Path
from dataset_loader import load_brain_tumor_dataset

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

PREPROCESS = True
SAVE_PATH = "data/preprocessed/"

# Hyperparameters: configurable kernel size and stride
# KERNEL_SIZE may be an int (square) or a tuple (kh, kw)
# STRIDE may be an int or a tuple (sh, sw)
KERNEL_SIZE: Union[int, Tuple[int, int]] = (2, 2)
STRIDE: Union[int, Tuple[int, int]] = (2, 2)
# Padding mode: 'valid' (no padding) or 'same' (pad to preserve spatial dims for stride=1)
PADDING: str = "same"


def _normalize_pair(value: Union[int, Tuple[int, int]]) -> Tuple[int, int]:
    """Normalize an int or tuple into a (h, w) tuple.

    Args:
        value: Either an int (square) or a (h, w) tuple.

    Returns:
        A tuple (h, w).
    """
    if isinstance(value, int):
        return (value, value)
    if isinstance(value, tuple) and len(value) == 2:
        return value
    raise ValueError("value must be int or tuple of length 2")


def _pad_same(
    image: np.ndarray, kernel: Tuple[int, int]
) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Apply same padding to an image for a given kernel size.

    Args:
        image: Input image array with shape ``(H, W, C)``.
        kernel: Kernel size as a ``(height, width)`` tuple.

    Returns:
        A tuple containing:
        - The padded image array.
        - The top/left padding applied as a ``(pad_h, pad_w)`` tuple.
    """
    kh, kw = kernel
    ih, iw = image.shape[:2]

    pad_h_total = max(kh - 1, 0)
    pad_w_total = max(kw - 1, 0)

    pad_h = pad_h_total // 2
    pad_w = pad_w_total // 2

    padded = np.pad(
        image,
        ((pad_h, pad_h_total - pad_h), (pad_w, pad_w_total - pad_w), (0, 0)),
        mode="constant",
    )
    return padded, (pad_h, pad_w)


def conv_circuit_n_local(n_qubit: int) -> Tuple[QuantumCircuit, List, List]:
    """Create a parameterized quantum circuit for convolutional feature extraction.

    This function composes a ZZ feature map with an ``n_local`` ansatz that
    uses ``ry`` rotations and linear ``cz`` entanglement.

    Args:
        n_qubit: Number of qubits in the circuit. This should match the number
            of features encoded into the feature map.

    Returns:
        A tuple containing:
        - The composed quantum circuit.
        - The ordered feature-map parameters.
        - The ordered ansatz parameters.

    Example:
        >>> qc, x_params, theta_params = conv_circuit_n_local(4)
        >>> isinstance(qc, QuantumCircuit)
        True
    """
    feature_map = zz_feature_map(
        feature_dimension=n_qubit,
        reps=1,
        parameter_prefix="x",
    )

    ansatz = n_local(
        num_qubits=n_qubit,
        rotation_blocks=["ry"],
        entanglement_blocks="cz",
        entanglement="linear",
        reps=1,
        skip_final_rotation_layer=True,
        parameter_prefix="θ",
    )

    qc = QuantumCircuit(n_qubit)
    qc.compose(feature_map, inplace=True)
    qc.compose(ansatz, inplace=True)

    x_params = list(feature_map.parameters)
    theta_params = list(ansatz.parameters)
    logger.debug(f"\n{qc.draw()}")

    return qc, x_params, theta_params


def estimate_expectations(
    qc: QuantumCircuit, x_params: List, theta_params: List, input_data: List[float]
) -> List[float]:
    """Run the given parameterized circuit and estimate qubit expectation values.

    The function assigns the provided `input_data` as parameters to `qc`, then
    uses a `StatevectorEstimator` to measure the expectation value of the
    Pauli-Z observable on each qubit.

    Args:
        qc: Parameterized quantum circuit produced by
            :func:`conv_circuit_n_local`.
        x_params: Feature-map parameters to bind from ``input_data``.
        theta_params: Ansatz parameters to bind to zero values.
        input_data: Feature values used to bind the feature-map parameters.

    Returns:
        A list of expectation values, one per qubit, in circuit order.

    Raises:
        ValueError: If ``input_data`` does not match the feature-map parameter
            count or if the zero-valued ansatz binding does not match the ansatz
            parameter count.

    Example:
        >>> qc, x_params, theta_params = conv_circuit_n_local(4)
        >>> estimate_expectations(qc, x_params, theta_params, [0.0, 0.1, 0.2, 0.3])
        [
        ...
        ]
    """
    estimator = StatevectorEstimator()

    def bind_patch_and_filter(qc, x_params, theta_params, patch, theta_values):
        if len(patch) != len(x_params):
            raise ValueError(
                f"Patch has {len(patch)} values, but feature map expects {len(x_params)}."
            )

        if len(theta_values) != len(theta_params):
            raise ValueError(
                f"Filter has {len(theta_values)} values, but ansatz expects {len(theta_params)}."
            )

        parameter_values = {}

        parameter_values.update({param: value for param, value in zip(x_params, patch)})

        parameter_values.update(
            {param: value for param, value in zip(theta_params, theta_values)}
        )

        return qc.assign_parameters(parameter_values)

    circuit = bind_patch_and_filter(
        qc, x_params, theta_params, input_data, [0.0] * len(theta_params)
    )
    results: List[float] = []
    for i in range(qc.num_qubits):
        observable_i = SparsePauliOp.from_sparse_list(
            [("Z", [i], 1.0)], num_qubits=qc.num_qubits
        )
        pubs = (circuit, observable_i)
        primitive_result = estimator.run(pubs=[pubs]).result()
        results.append(primitive_result[0].data.evs)
    return [res for res in results]


def quanv(image: np.ndarray) -> np.ndarray:
    """Apply a quantum convolution to a single grayscale image with flexible kernel.

    The function slides a kernel window across the input image, encodes the
    flattened patch values into a quantum circuit with `n_qubits = kh * kw`
    and stores the per-qubit expectation values as feature channels in the
    output tensor.

    Args:
        image: A numpy array with shape `(H, W, C)` where `C >= 1`. Only the
            first channel is used for encoding.

    Returns:
        A numpy array with shape `(out_h, out_w, n_qubits)` containing the
        quantum-processed feature maps.

    Example:
        >>> image = mock_image(4, 4)
        >>> features = quanv(image)
        >>> features.ndim
        3
    """
    kh, kw = _normalize_pair(KERNEL_SIZE)
    sh, sw = _normalize_pair(STRIDE)

    # Use only the first channel for encoding
    img = image[..., 0]

    # Optionally pad for 'same' behavior
    if PADDING == "same":
        img_padded, (pad_h, pad_w) = _pad_same(img[..., np.newaxis], (kh, kw))
        img = img_padded[..., 0]

    H, W = img.shape

    out_h = (H - kh) // sh + 1
    out_w = (W - kw) // sw + 1

    n_qubits = kh * kw
    out = np.zeros((out_h, out_w, n_qubits))

    # Loop over top-left coordinates of patches
    for i_out, j in enumerate(range(0, H - kh + 1, sh)):
        for i_out_w, k in enumerate(range(0, W - kw + 1, sw)):
            # Extract patch and flatten row-major
            patch = img[j : j + kh, k : k + kw].reshape(-1).tolist()
            qc, x_params, theta_params = conv_circuit_n_local(n_qubits)
            q_results = estimate_expectations(qc, x_params, theta_params, patch)

            if j % max(1, W // 8) == 0 and k == 0:
                logger.info(
                    f"Processed block starting at ({j}, {k}), quantum results: {q_results}"
                )

            for c in range(n_qubits):
                out[i_out, i_out_w, c] = q_results[c]

    return out


def mock_image(height: int = 256, width: int = 256) -> np.ndarray:
    """Create a mock grayscale image with given height and width.

    Args:
        height: Image height in pixels.
        width: Image width in pixels.

    Returns:
        A numpy array of shape `(height, width, 1)` with random floats in [0, 1).

    Example:
        >>> image = mock_image(8, 8)
        >>> image.shape
        (8, 8, 1)
    """
    return np.random.rand(height, width, 1)


def extract_quantum_feature_maps(args):
    """Extract and display quantum feature maps for a dataset subset.

    Args:
        args: Parsed command-line arguments with ``preprocess`` and ``samples``
            attributes.

    Returns:
        None. The function optionally saves preprocessed feature maps and then
        renders a matplotlib figure comparing input images with their quantum
        feature maps.

    Example:
        >>> from argparse import Namespace
        >>> extract_quantum_feature_maps(Namespace(preprocess=False, samples=1))
    """
    train_images = load_brain_tumor_dataset("data/archive/Data")[0][
        : args.samples
    ]  # Load a small subset of images for testing

    if args.preprocess:
        n_qubit = (
            KERNEL_SIZE
            if isinstance(KERNEL_SIZE, int)
            else KERNEL_SIZE[0] * KERNEL_SIZE[1]
        )

        qc, x_params, theta_params = conv_circuit_n_local(n_qubit=n_qubit)

        q_train_images = []
        for idx, img in enumerate(train_images):
            logger.info(f"Processing image {idx + 1}/{len(train_images)}")
            q_train_images.append(quanv(img))
        q_train_images = np.asarray(q_train_images)

        if q_train_images.shape[-1] >= 2:
            channel_diff = np.abs(
                q_train_images[0, :, :, 0] - q_train_images[0, :, :, 1]
            )
            logger.info(
                "Pre-save channel comparison (sample 0, ch. 0 vs ch. 1): max diff=%s, mean diff=%s",
                float(channel_diff.max()),
                float(channel_diff.mean()),
            )
        else:
            logger.info(
                "Pre-save channel comparison skipped: fewer than 2 output channels."
            )

        q_test_images = []
        q_test_images = np.asarray(q_test_images)

        # Save pre-processed images
        Path(SAVE_PATH).mkdir(parents=True, exist_ok=True)
        np.save(SAVE_PATH + "q_train_images.npy", q_train_images)

    # Load pre-processed images
    q_train_images = np.load(SAVE_PATH + "q_train_images.npy")

    n_samples = len(train_images)
    kh, kw = _normalize_pair(KERNEL_SIZE)
    n_channels = kh * kw
    fig, axes = plt.subplots(1 + n_channels, n_samples, figsize=(10, 10), squeeze=False)
    for k in range(n_samples):
        axes[0, 0].set_ylabel("Input")
        if k != 0:
            axes[0, k].yaxis.set_visible(False)
        axes[0, k].imshow(train_images[k, :, :, 0], cmap="gray")

        # Plot all output channels
        for c in range(n_channels):
            axes[c + 1, 0].set_ylabel("Output [ch. {}]".format(c))
            if k != 0:
                axes[c + 1, k].yaxis.set_visible(False)
            axes[c + 1, k].imshow(q_train_images[k, :, :, c], cmap="gray")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    from argparse import ArgumentParser

    arg_parser = ArgumentParser(
        description="Run quantum feature extraction on brain tumor dataset."
    )
    arg_parser.add_argument(
        "--preprocess", action="store_true", help="Whether to run preprocessing."
    )
    arg_parser.add_argument(
        "--samples",
        type=int,
        default=4,
        help="Number of samples to process for testing.",
    )

    args = arg_parser.parse_args()

    extract_quantum_feature_maps(args)
