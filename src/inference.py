"""Checkpoint-based inference helpers for the QCNN tumor classifier."""

from __future__ import annotations

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


DEFAULT_CHECKPOINT_PATH = Path("checkpoints/qcnn_weights.npz")
IMAGE_SIZE = (256, 256)
NORMAL_LABEL = "Vous n'avez pas de tumeur"
TUMOR_LABEL = "Vous avez une tumeur!"


def _load_image(image_path: str | Path) -> np.ndarray:
    """Load an image in the same shape/range used for quanvolution."""
    image = Image.open(image_path).convert("L").resize(IMAGE_SIZE)
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
    feature_map = quanv(
        image=image,
        qc=qc,
        x_params=x_params,
        theta_params=theta_params,
        theta_values_by_channel=theta_values_by_channel,
        estimator=StatevectorEstimator(),
    )
    return feature_map.reshape(-1, feature_map.shape[-1])


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

    weights = np.asarray(np.load(checkpoint_path)["weights"], dtype=float)
    spatial_vectors = _extract_spatial_vectors(_load_image(image_path))

    qnn = build_qcnn_estimator(num_inputs=spatial_vectors.shape[-1])
    outputs = np.asarray(qnn.forward(spatial_vectors, weights), dtype=float).reshape(-1)
    mean_score = float(outputs.mean())

    return NORMAL_LABEL if mean_score >= 0.0 else TUMOR_LABEL
