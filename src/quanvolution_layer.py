import logging
from collections import Counter
from pathlib import Path
from typing import List, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.library import n_local, zz_feature_map
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp

from dataset_loader import CLASSES, load_brain_tumor_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

SAVE_PATH = "data/preprocessed/"
DEFAULT_FEATURE_FILENAME = "q_train_images.npy"
DEFAULT_LABEL_FILENAME = "q_train_labels.npy"
DEFAULT_CLASS_FILENAME = "q_train_class_names.npy"
DEFAULT_IMAGE_SIZE = 256

# Hyperparameters: configurable kernel size and stride.
# KERNEL_SIZE may be an int (square) or a tuple (kh, kw).
# STRIDE may be an int or a tuple (sh, sw).
KERNEL_SIZE: Union[int, Tuple[int, int]] = (2, 2)
STRIDE: Union[int, Tuple[int, int]] = (4, 4)

# Padding mode: "valid" (no padding) or "same".
PADDING: str = "same"

# A 2x2 patch gives 4 input values and therefore 4 qubits.
# The number of output channels is a modelling choice. By default, use one
# channel per qubit, but each channel gets its own ansatz/filter parameters.
NUM_OUTPUT_CHANNELS: int | None = None

# Image values are usually normalized in [0, 1]. Scaling them to [0, pi]
# makes the angle encoding less likely to collapse into a tiny-angle regime.
DATA_ANGLE_SCALE: float = np.pi

RANDOM_SEED = 42


def _normalize_pair(value: Union[int, Tuple[int, int]]) -> Tuple[int, int]:
    """Normalize an int or tuple into a ``(height, width)`` tuple."""
    if isinstance(value, int):
        return (value, value)
    if isinstance(value, tuple) and len(value) == 2:
        return value
    raise ValueError("value must be an int or a tuple of length 2")


def _pad_same(
    image: np.ndarray,
    kernel: Tuple[int, int],
) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Apply same-style zero padding to an ``(H, W, C)`` image."""
    kh, kw = kernel

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


def conv_circuit_n_local(
    n_qubits: int,
) -> Tuple[QuantumCircuit, List[Parameter], List[Parameter]]:
    """Create the local quantum filter circuit.

    The circuit has two distinct parameter groups:
    - ``x_params``: data-encoding parameters bound to the image patch values;
    - ``theta_params``: ansatz/filter parameters, analogous to convolution weights.

    For a 2x2 kernel, ``n_qubits == 4`` and the feature map expects exactly
    4 input values. The ansatz may add extra parameters; those are not extra
    pixels, they are filter parameters.
    """
    feature_map = zz_feature_map(
        feature_dimension=n_qubits,
        reps=1,
        parameter_prefix="x",
    )

    # This is intentionally a little richer than the original ry/cz block.
    # It gives the random filters more freedom while keeping the circuit shallow.
    ansatz = n_local(
        num_qubits=n_qubits,
        rotation_blocks=["ry", "rz"],
        entanglement_blocks="cx",
        entanglement="linear",
        reps=1,
        skip_final_rotation_layer=False,
        parameter_prefix="theta",
    )

    qc = QuantumCircuit(n_qubits)
    qc.compose(feature_map, inplace=True)
    qc.compose(ansatz, inplace=True)

    x_params = list(feature_map.parameters)
    theta_params = list(ansatz.parameters)

    logger.debug("\n%s", qc.draw())
    logger.info(
        "Circuit created with %s data parameters and %s filter parameters.",
        len(x_params),
        len(theta_params),
    )

    return qc, x_params, theta_params


def bind_patch_and_filter(
    qc: QuantumCircuit,
    x_params: Sequence[Parameter],
    theta_params: Sequence[Parameter],
    patch: Sequence[float],
    theta_values: Sequence[float],
) -> QuantumCircuit:
    """Bind image-patch values and quantum-filter values separately."""
    if len(patch) != len(x_params):
        raise ValueError(
            f"Patch has {len(patch)} values, but feature map expects {len(x_params)}."
        )

    if len(theta_values) != len(theta_params):
        raise ValueError(
            f"Filter has {len(theta_values)} values, but ansatz expects {len(theta_params)}."
        )

    parameter_values = {
        **{param: value for param, value in zip(x_params, patch)},
        **{param: value for param, value in zip(theta_params, theta_values)},
    }
    return qc.assign_parameters(parameter_values)


def estimate_channel_expectations(
    qc: QuantumCircuit,
    x_params: Sequence[Parameter],
    theta_params: Sequence[Parameter],
    patch: Sequence[float],
    theta_values_by_channel: np.ndarray,
    estimator: StatevectorEstimator,
) -> List[float]:
    """Estimate one output value per channel for a single image patch.

    Each channel uses the same encoded patch but its own ansatz/filter
    parameter vector. This is closer to the classical idea that each output
    channel has its own convolutional filter.
    """
    n_qubits = qc.num_qubits
    results: List[float] = []

    for channel, theta_values in enumerate(theta_values_by_channel):
        circuit = bind_patch_and_filter(
            qc=qc,
            x_params=x_params,
            theta_params=theta_params,
            patch=patch,
            theta_values=theta_values,
        )

        observable_qubit = channel % n_qubits
        observable = SparsePauliOp.from_sparse_list(
            [("Z", [observable_qubit], 1.0)],
            num_qubits=n_qubits,
        )

        primitive_result = estimator.run(pubs=[(circuit, observable)]).result()
        results.append(float(primitive_result[0].data.evs))

    return results


def quanv(
    image: np.ndarray,
    qc: QuantumCircuit,
    x_params: Sequence[Parameter],
    theta_params: Sequence[Parameter],
    theta_values_by_channel: np.ndarray,
    estimator: StatevectorEstimator,
) -> np.ndarray:
    """Apply a quanvolutional feature extractor to one grayscale image.

    The flattened patch values are bound only to the feature-map parameters.
    The ansatz parameters are bound separately using fixed random values.
    """
    kh, kw = _normalize_pair(KERNEL_SIZE)
    sh, sw = _normalize_pair(STRIDE)

    img = image[..., 0]

    if PADDING == "same":
        img_padded, _ = _pad_same(img[..., np.newaxis], (kh, kw))
        img = img_padded[..., 0]
    elif PADDING != "valid":
        raise ValueError('PADDING must be either "same" or "valid"')

    height, width = img.shape
    out_h = (height - kh) // sh + 1
    out_w = (width - kw) // sw + 1
    num_output_channels = theta_values_by_channel.shape[0]

    out = np.zeros((out_h, out_w, num_output_channels), dtype=float)

    for i_out, row in enumerate(range(0, height - kh + 1, sh)):
        for j_out, col in enumerate(range(0, width - kw + 1, sw)):
            patch = img[row : row + kh, col : col + kw].reshape(-1)
            patch_angles = (patch * DATA_ANGLE_SCALE).tolist()

            q_results = estimate_channel_expectations(
                qc=qc,
                x_params=x_params,
                theta_params=theta_params,
                patch=patch_angles,
                theta_values_by_channel=theta_values_by_channel,
                estimator=estimator,
            )

            out[i_out, j_out, :] = q_results

        if i_out % max(1, out_h // 8) == 0:
            logger.info("Processed output row %s/%s", i_out + 1, out_h)

    return out


def mock_image(height: int = 256, width: int = 256) -> np.ndarray:
    """Create a mock grayscale image with shape ``(height, width, 1)``."""
    return np.random.rand(height, width, 1)


def reduced_image_size(reduction_factor: int) -> tuple[int, int]:
    """Return the square image size produced by a resolution reduction factor."""
    if reduction_factor < 1:
        raise ValueError("--resolution-reduction must be at least 1.")
    if DEFAULT_IMAGE_SIZE % reduction_factor != 0:
        raise ValueError(
            f"--resolution-reduction must evenly divide {DEFAULT_IMAGE_SIZE}."
        )

    size = DEFAULT_IMAGE_SIZE // reduction_factor
    return (size, size)


def log_channel_statistics(q_train_images: np.ndarray) -> None:
    """Log per-channel value ranges for debugging visualization issues."""
    for channel in range(q_train_images.shape[-1]):
        values = q_train_images[..., channel]
        logger.info(
            "channel %s: min=%.6f max=%.6f mean=%.6f std=%.6f",
            channel,
            float(values.min()),
            float(values.max()),
            float(values.mean()),
            float(values.std()),
        )

    if q_train_images.shape[-1] >= 2:
        channel_diff = np.abs(q_train_images[0, :, :, 0] - q_train_images[0, :, :, 1])
        logger.info(
            "Sample 0, channel 0 vs 1: max diff=%.6f, mean diff=%.6f",
            float(channel_diff.max()),
            float(channel_diff.mean()),
        )


def _balanced_indices(
    labels: np.ndarray,
    *,
    classes: Sequence[str],
    samples_per_class: int,
    seed: int,
) -> np.ndarray:
    """Return image indices balanced across the requested raw dataset classes."""
    if samples_per_class < 1:
        raise ValueError("--samples-per-class must be at least 1.")

    rng = np.random.default_rng(seed)
    selected_indices: list[int] = []

    for class_name in classes:
        if class_name not in CLASSES:
            raise ValueError(
                f"Unknown class {class_name!r}. Available classes: {tuple(CLASSES)}"
            )

        class_label = CLASSES[class_name]
        class_indices = np.flatnonzero(labels == class_label)
        if len(class_indices) == 0:
            present_labels = tuple(int(label) for label in np.unique(labels))
            raise ValueError(
                f"No images found for class {class_name!r} with label {class_label}. "
                f"Present labels are {present_labels}. If dataset classes were "
                "filtered before this step, keep the original CLASSES label values "
                "so training and inference agree."
            )

        sample_count = min(samples_per_class, len(class_indices))
        if sample_count < samples_per_class:
            logger.warning(
                "Class %s has only %s images; requested %s.",
                class_name,
                sample_count,
                samples_per_class,
            )

        selected = rng.choice(class_indices, size=sample_count, replace=False)
        selected_indices.extend(int(index) for index in selected)

    rng.shuffle(selected_indices)
    return np.asarray(selected_indices, dtype=int)


def extract_quantum_feature_maps(args) -> None:
    """Extract, save, load, and display quanvolutional feature maps."""
    image_size = reduced_image_size(args.resolution_reduction)
    classes = tuple(args.classes)
    all_images, all_labels = load_brain_tumor_dataset(
        args.dataset_path,
        image_size=image_size,
    )

    samples_per_class = args.samples_per_class
    if samples_per_class is None:
        samples_per_class = max(1, args.samples // len(classes))

    selected_indices = _balanced_indices(
        all_labels,
        classes=classes,
        samples_per_class=samples_per_class,
        seed=args.seed,
    )
    train_images = all_images[selected_indices]
    train_labels = all_labels[selected_indices]
    label_counts = Counter(int(label) for label in train_labels)
    logger.info(
        "Selected %s balanced images across classes=%s label_counts=%s.",
        len(train_images),
        classes,
        dict(label_counts),
    )

    kh, kw = _normalize_pair(KERNEL_SIZE)
    n_qubits = kh * kw
    num_output_channels = NUM_OUTPUT_CHANNELS or n_qubits

    qc, x_params, theta_params = conv_circuit_n_local(n_qubits=n_qubits)

    rng = np.random.default_rng(args.seed)
    theta_values_by_channel = rng.uniform(
        low=0.0,
        high=2 * np.pi,
        size=(num_output_channels, len(theta_params)),
    )

    if args.preprocess:
        estimator = StatevectorEstimator()
        q_train_images = []

        for idx, img in enumerate(train_images):
            logger.info("Processing image %s/%s", idx + 1, len(train_images))
            q_train_images.append(
                quanv(
                    image=img,
                    qc=qc,
                    x_params=x_params,
                    theta_params=theta_params,
                    theta_values_by_channel=theta_values_by_channel,
                    estimator=estimator,
                )
            )

        q_train_images = np.asarray(q_train_images)
        log_channel_statistics(q_train_images)

        Path(SAVE_PATH).mkdir(parents=True, exist_ok=True)
        np.save(Path(SAVE_PATH) / DEFAULT_FEATURE_FILENAME, q_train_images)
        np.save(Path(SAVE_PATH) / DEFAULT_LABEL_FILENAME, train_labels)
        np.save(Path(SAVE_PATH) / DEFAULT_CLASS_FILENAME, np.asarray(classes))
    else:
        q_train_images = np.load(Path(SAVE_PATH) / DEFAULT_FEATURE_FILENAME)

    n_samples = len(train_images)
    n_channels = q_train_images.shape[-1]

    fig, axes = plt.subplots(
        1 + n_channels,
        n_samples,
        figsize=(2.5 * n_samples, 2.5 * (1 + n_channels)),
        squeeze=False,
    )

    for sample_idx in range(n_samples):
        axes[0, 0].set_ylabel("Input")
        if sample_idx != 0:
            axes[0, sample_idx].yaxis.set_visible(False)
        axes[0, sample_idx].imshow(train_images[sample_idx, :, :, 0], cmap="gray")

        for channel in range(n_channels):
            axes[channel + 1, 0].set_ylabel(f"Output [ch. {channel}]")
            if sample_idx != 0:
                axes[channel + 1, sample_idx].yaxis.set_visible(False)
            axes[channel + 1, sample_idx].imshow(
                q_train_images[sample_idx, :, :, channel],
                cmap="gray",
                vmin=-1,
                vmax=1,
            )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    from argparse import ArgumentParser

    arg_parser = ArgumentParser(
        description="Run quantum feature extraction on a brain tumor dataset subset."
    )
    arg_parser.add_argument(
        "--preprocess",
        action="store_true",
        help="Run preprocessing instead of loading the saved feature maps.",
    )
    arg_parser.add_argument(
        "--dataset-path",
        type=Path,
        default=Path("data/archive/Data"),
        help="Path to the raw dataset class folders.",
    )
    arg_parser.add_argument(
        "--samples",
        type=int,
        default=4,
        help=(
            "Legacy total sample hint. Used only when --samples-per-class is not set."
        ),
    )
    arg_parser.add_argument(
        "--samples-per-class",
        type=int,
        default=None,
        help="Number of images to preprocess from each requested class.",
    )
    arg_parser.add_argument(
        "--classes",
        nargs="+",
        default=["normal", "meningioma_tumor"],
        help="Dataset class folders to sample for balanced preprocessing.",
    )
    arg_parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help="Random seed for fixed quanvolutional filter parameters.",
    )
    arg_parser.add_argument(
        "--resolution-reduction",
        type=int,
        default=1,
        help=(
            "Factor used to reduce dataset image resolution before quanvolution; "
            "2 converts 256x256 images to 128x128."
        ),
    )

    extract_quantum_feature_maps(arg_parser.parse_args())
