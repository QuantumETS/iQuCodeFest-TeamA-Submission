"""Checkpoint-based inference helpers for the QCNN tumor classifier."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from PIL import Image
from qiskit.primitives import StatevectorEstimator

from qcnn import build_qcnn_estimator
from quanvolution_layer import (
    KERNEL_SIZE,
    NUM_OUTPUT_CHANNELS,
    RANDOM_SEED,
    conv_circuit_n_local,
    quanv,
    _normalize_pair,
)


LOGGER = logging.getLogger(__name__)

DEFAULT_CHECKPOINT_PATH = Path("checkpoints/qcnn_weights.npz")
DEFAULT_IMAGE_SIZE = (256, 256)
INFERENCE_BATCH_SIZE = 256
NORMAL_LABEL = "All is good, no tumor!"
TUMOR_LABEL = "Oh no, you have a tumor!"


def _checkpoint_image_size(checkpoint: np.lib.npyio.NpzFile) -> tuple[int, int]:
    """Return the image size stored in a checkpoint, or the legacy default."""
    if "image_size" not in checkpoint:
        return DEFAULT_IMAGE_SIZE

    image_size = np.asarray(checkpoint["image_size"], dtype=int).reshape(-1)
    if len(image_size) != 2:
        raise ValueError(
            f"Checkpoint image_size must contain 2 values, got {image_size}."
        )

    height, width = (int(image_size[0]), int(image_size[1]))
    if height < 1 or width < 1:
        raise ValueError(f"Checkpoint image_size must be positive, got {image_size}.")

    return (height, width)


def _load_image(image_path: str | Path, image_size: tuple[int, int]) -> np.ndarray:
    """Load an image in the same shape/range used for quanvolution."""
    image = Image.open(image_path).convert("L").resize(image_size)
    image_array = np.asarray(image, dtype=np.float32) / 255.0
    return image_array[..., np.newaxis]


def _build_quanvolution_filter() -> tuple:
    """Recreate the fixed quanvolution filter used during preprocessing."""
    kh, kw = _normalize_pair(KERNEL_SIZE)
    n_qubits = kh * kw
    num_output_channels = NUM_OUTPUT_CHANNELS or n_qubits

    qc, x_params, theta_params = conv_circuit_n_local(n_qubits=n_qubits)
    rng = np.random.default_rng(RANDOM_SEED)
    theta_values_by_channel = rng.uniform(
        low=0.0,
        high=2 * np.pi,
        size=(num_output_channels, len(theta_params)),
    )
    return qc, x_params, theta_params, theta_values_by_channel


def _extract_spatial_vectors(image: np.ndarray) -> np.ndarray:
    """Run quanvolution and flatten all spatial positions into QCNN inputs."""
    qc, x_params, theta_params, theta_values_by_channel = _build_quanvolution_filter()
    LOGGER.info("Starting quanvolution for image shape=%s.", image.shape)
    feature_map = quanv(
        image=image,
        qc=qc,
        x_params=x_params,
        theta_params=theta_params,
        theta_values_by_channel=theta_values_by_channel,
        estimator=StatevectorEstimator(),
    )
    LOGGER.info("Finished quanvolution with feature map shape=%s.", feature_map.shape)
    return feature_map.reshape(-1, feature_map.shape[-1])


def _forward_in_batches(
    qnn,
    spatial_vectors: np.ndarray,
    weights: np.ndarray,
    batch_size: int = INFERENCE_BATCH_SIZE,
) -> np.ndarray:
    """Run QCNN inference in batches so debug logs can show progress."""
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")

    outputs = []
    total_vectors = len(spatial_vectors)
    total_batches = (total_vectors + batch_size - 1) // batch_size

    LOGGER.info(
        "Starting QCNN forward pass for %s spatial vectors in %s batches.",
        total_vectors,
        total_batches,
    )

    for batch_index, start in enumerate(range(0, total_vectors, batch_size), start=1):
        end = min(start + batch_size, total_vectors)
        LOGGER.info(
            "Running QCNN batch %s/%s: vectors %s-%s.",
            batch_index,
            total_batches,
            start,
            end - 1,
        )
        batch_outputs = qnn.forward(spatial_vectors[start:end], weights)
        outputs.append(np.asarray(batch_outputs, dtype=float).reshape(-1))
        LOGGER.info("Finished QCNN batch %s/%s.", batch_index, total_batches)

    return np.concatenate(outputs)


def classify_image(
    image_path: str | Path,
    checkpoint_path: Path = DEFAULT_CHECKPOINT_PATH,
) -> str:
    """Classify one MRI image as tumor or non-tumor using a saved QCNN checkpoint."""
    image_path = Path(image_path)
    checkpoint_path = Path(checkpoint_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    LOGGER.info("Loading checkpoint from %s.", checkpoint_path)
    with np.load(checkpoint_path) as checkpoint:
        weights = np.asarray(checkpoint["weights"], dtype=float)
        image_size = _checkpoint_image_size(checkpoint)
    LOGGER.info(
        "Loaded checkpoint weights=%s image_size=%s.",
        len(weights),
        image_size,
    )

    spatial_vectors = _extract_spatial_vectors(_load_image(image_path, image_size))
    LOGGER.info("Extracted spatial vectors with shape=%s.", spatial_vectors.shape)

    LOGGER.info(
        "Building QCNN estimator with num_inputs=%s.", spatial_vectors.shape[-1]
    )
    qnn = build_qcnn_estimator(num_inputs=spatial_vectors.shape[-1])
    LOGGER.info("Finished building QCNN estimator.")

    outputs = _forward_in_batches(qnn, spatial_vectors, weights)
    mean_score = float(outputs.mean())
    LOGGER.info("Finished inference with mean_score=%.6f.", mean_score)

    return NORMAL_LABEL if mean_score >= 0.0 else TUMOR_LABEL
