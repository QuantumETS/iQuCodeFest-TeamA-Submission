"""Train and evaluate the QCNN classifier on saved quanvolution features."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Callable

import numpy as np
from qiskit_machine_learning.optimizers import ADAM, COBYLA, L_BFGS_B, SPSA

from dataset_loader import load_brain_tumor_dataset
from qcnn import DEFAULT_FEATURE_PATH, build_qcnn_estimator


LOGGER = logging.getLogger(__name__)

DEFAULT_CHECKPOINT_PATH = Path("checkpoints/qcnn_weights.npz")
DEFAULT_DATASET_PATH = Path("data/archive/Data")
DEFAULT_IMAGE_SIZE = 256
DEFAULT_LABEL_PATH = Path("data/preprocessed/q_train_labels.npy")


def configure_logging() -> None:
    """Configure command-line logging for training progress."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and test the QCNN on saved quanvolution features."
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=50,
        help="Maximum number of optimizer iterations.",
    )
    parser.add_argument(
        "--optimizer",
        choices=("cobyla", "spsa", "l-bfgs-b", "adam"),
        default="cobyla",
        help="Optimizer to use for training.",
    )
    parser.add_argument(
        "--train-size",
        type=int,
        default=6,
        help="Number of expanded spatial QCNN samples to use for training.",
    )
    parser.add_argument(
        "--test-size",
        type=int,
        default=2,
        help="Number of expanded spatial QCNN samples to use for testing.",
    )
    parser.add_argument(
        "--train-images",
        type=int,
        default=None,
        help=(
            "Number of preprocessed images to reserve for training before patch "
            "expansion. Defaults to a stratified 75/25 image split."
        ),
    )
    parser.add_argument(
        "--test-images",
        type=int,
        default=None,
        help=(
            "Number of preprocessed images to reserve for testing before patch "
            "expansion. Defaults to a stratified 75/25 image split."
        ),
    )
    parser.add_argument(
        "--use-initial",
        action="store_true",
        help="Initialize training from the checkpoint at --checkpoint-path.",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=DEFAULT_CHECKPOINT_PATH,
        help="Path used to load/save QCNN weight checkpoints.",
    )
    parser.add_argument(
        "--feature-path",
        type=Path,
        default=DEFAULT_FEATURE_PATH,
        help="Path to saved quanvolution feature maps.",
    )
    parser.add_argument(
        "--label-path",
        type=Path,
        default=None,
        help=(
            "Path to labels saved with the quanvolution feature maps. Defaults to "
            "a sidecar inferred from --feature-path."
        ),
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help="Path to the original dataset, used to recover labels.",
    )
    parser.add_argument(
        "--resolution-reduction",
        type=int,
        default=1,
        help=(
            "Factor used to reduce dataset image resolution before quanvolution; "
            "2 converts 256x256 images to 128x128."
        ),
    )
    parser.add_argument(
        "--feature-map",
        choices=("z", "zz"),
        default="z",
        help="Feature map used when building the QCNN.",
    )
    parser.add_argument(
        "--positive-label",
        type=int,
        default=2,
        help="Dataset label treated as the positive binary class.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling data and initializing weights.",
    )
    return parser.parse_args()


def _expand_spatial_features(
    features: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert saved quanvolution maps into QCNN-ready feature vectors.

    A saved tensor shaped ``(images, height, width, channels)`` contains many
    channel vectors per image. Each spatial vector is a valid QCNN input, so
    expanding them avoids being limited to the number of preprocessed images.
    """
    if features.ndim == 2:
        return features.astype(float), labels
    if features.ndim < 2:
        raise ValueError(
            f"Expected at least 2 feature dimensions, got {features.shape}."
        )

    image_count = features.shape[0]
    spatial_vectors_per_image = int(np.prod(features.shape[1:-1]))
    expanded_features = features.reshape(image_count * spatial_vectors_per_image, -1)
    expanded_labels = np.repeat(labels[:image_count], spatial_vectors_per_image)

    LOGGER.info(
        "Expanded %s preprocessed images into %s spatial QCNN samples.",
        image_count,
        len(expanded_features),
    )
    return expanded_features.astype(float), expanded_labels


def _default_label_path(feature_path: Path) -> Path:
    """Return the conventional sidecar label path for a feature tensor."""
    if feature_path.name == "q_train_images.npy":
        return feature_path.with_name("q_train_labels.npy")
    return feature_path.with_name(f"{feature_path.stem}_labels.npy")


def _load_feature_labels(
    *,
    feature_path: Path,
    label_path: Path | None,
    dataset_path: Path,
    image_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Load preprocessed feature maps and their image-level labels."""
    raw_features = np.load(feature_path)
    candidate_label_path = label_path or _default_label_path(feature_path)

    if candidate_label_path.exists():
        labels = np.load(candidate_label_path)
        LOGGER.info("Loaded preprocessed labels from %s.", candidate_label_path)
    else:
        LOGGER.warning(
            "Label sidecar not found at %s; falling back to legacy dataset-order labels.",
            candidate_label_path,
        )
        _, labels = load_brain_tumor_dataset(dataset_path, image_size=image_size)
        if len(raw_features) != len(labels):
            raise FileNotFoundError(
                "Preprocessed label sidecar is required because feature and raw "
                f"dataset label counts differ: features={len(raw_features)} "
                f"raw_labels={len(labels)}. Re-run quanvolution preprocessing to "
                f"create {candidate_label_path}."
            )

    image_count = min(len(raw_features), len(labels))
    if image_count == 0:
        raise ValueError("No preprocessed features and labels are available.")
    if len(raw_features) != len(labels):
        LOGGER.warning(
            "Feature/label count mismatch: features=%s labels=%s; using first %s.",
            len(raw_features),
            len(labels),
            image_count,
        )

    return raw_features[:image_count], np.asarray(labels[:image_count], dtype=int)


def _split_image_indices(
    *,
    labels: np.ndarray,
    train_images: int | None,
    test_images: int | None,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Split image indices before patch expansion to avoid same-image leakage."""
    image_count = len(labels)
    unique_labels = np.unique(labels)
    class_count = len(unique_labels)

    if train_images is None and test_images is None:
        test_images = max(class_count, int(round(image_count * 0.25)))
        train_images = image_count - test_images
    elif train_images is None:
        if test_images is None:
            raise ValueError("Internal split error: test_images was not resolved.")
        train_images = image_count - test_images
    elif test_images is None:
        test_images = image_count - train_images

    if train_images < 1 or test_images < 1:
        raise ValueError("--train-images and --test-images must both be at least 1.")

    requested = train_images + test_images
    if requested > image_count:
        raise ValueError(
            f"Requested {requested} preprocessed images, but only {image_count} are available."
        )

    rng = np.random.default_rng(seed)
    if len(unique_labels) > 1 and requested >= len(unique_labels) * 2:
        split_labels = rng.permutation(unique_labels)
        train_parts: list[np.ndarray] = []
        test_parts: list[np.ndarray] = []
        train_base = train_images // len(unique_labels)
        train_remainder = train_images % len(unique_labels)
        test_base = test_images // len(unique_labels)
        test_remainder = test_images % len(unique_labels)

        for class_offset, label in enumerate(split_labels):
            class_indices = rng.permutation(np.flatnonzero(labels == label))
            class_train_size = train_base + int(class_offset < train_remainder)
            class_test_size = test_base + int(class_offset < test_remainder)
            class_requested = class_train_size + class_test_size

            if class_requested > len(class_indices):
                raise ValueError(
                    "Not enough preprocessed images for label "
                    f"{label}: requested {class_requested}, found {len(class_indices)}."
                )

            train_parts.append(class_indices[:class_train_size])
            test_parts.append(
                class_indices[class_train_size : class_train_size + class_test_size]
            )

        train_indices = rng.permutation(np.concatenate(train_parts))
        test_indices = rng.permutation(np.concatenate(test_parts))
    else:
        indices = rng.permutation(image_count)[:requested]
        train_indices = indices[:train_images]
        test_indices = indices[train_images:]

    for split_name, split_indices in (
        ("train", train_indices),
        ("test", test_indices),
    ):
        unique_targets = np.unique(labels[split_indices])
        if len(unique_targets) == 1:
            LOGGER.warning(
                "%s image split contains one raw class only; metrics may be misleading.",
                split_name,
            )

    return train_indices, test_indices


def _sample_spatial_examples(
    *,
    features: np.ndarray,
    targets: np.ndarray,
    sample_size: int,
    seed: int,
    split_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample expanded patch examples while preserving class balance where possible."""
    if sample_size < 1:
        raise ValueError(f"--{split_name}-size must be at least 1.")
    if sample_size > len(features):
        raise ValueError(
            f"Requested {sample_size} {split_name} spatial samples, but only "
            f"{len(features)} are available after image split expansion."
        )

    rng = np.random.default_rng(seed)
    unique_targets = np.unique(targets)
    if len(unique_targets) > 1 and sample_size >= len(unique_targets):
        selected_parts: list[np.ndarray] = []
        base = sample_size // len(unique_targets)
        remainder = sample_size % len(unique_targets)

        for target_offset, target in enumerate(rng.permutation(unique_targets)):
            target_indices = rng.permutation(np.flatnonzero(targets == target))
            target_sample_size = base + int(target_offset < remainder)
            if target_sample_size > len(target_indices):
                raise ValueError(
                    f"Not enough {split_name} spatial samples for target {target}: "
                    f"requested {target_sample_size}, found {len(target_indices)}."
                )
            selected_parts.append(target_indices[:target_sample_size])

        selected_indices = rng.permutation(np.concatenate(selected_parts))
    else:
        selected_indices = rng.permutation(len(features))[:sample_size]

    return features[selected_indices], targets[selected_indices]


def load_qcnn_data(
    *,
    feature_path: Path,
    label_path: Path | None = None,
    dataset_path: Path,
    train_size: int,
    test_size: int,
    train_images: int | None = None,
    test_images: int | None = None,
    positive_label: int,
    seed: int,
    image_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load saved features, split images, then sample expanded patches."""
    raw_features, labels = _load_feature_labels(
        feature_path=feature_path,
        label_path=label_path,
        dataset_path=dataset_path,
        image_size=image_size,
    )

    train_indices, test_indices = _split_image_indices(
        labels=labels,
        train_images=train_images,
        test_images=test_images,
        seed=seed,
    )
    train_labels = labels[train_indices]
    test_labels = labels[test_indices]

    if len(np.unique(labels[np.concatenate([train_indices, test_indices])])) == 1:
        LOGGER.warning(
            "Selected image data contains one raw class only; accuracy may be misleading."
        )

    train_inputs, train_image_labels = _expand_spatial_features(
        raw_features[train_indices],
        train_labels,
    )
    test_inputs, test_image_labels = _expand_spatial_features(
        raw_features[test_indices],
        test_labels,
    )

    train_targets = np.where(train_image_labels == positive_label, 1.0, -1.0)
    test_targets = np.where(test_image_labels == positive_label, 1.0, -1.0)

    train_inputs, train_targets = _sample_spatial_examples(
        features=train_inputs,
        targets=train_targets,
        sample_size=train_size,
        seed=seed,
        split_name="train",
    )
    test_inputs, test_targets = _sample_spatial_examples(
        features=test_inputs,
        targets=test_targets,
        sample_size=test_size,
        seed=seed + 1,
        split_name="test",
    )

    LOGGER.info(
        "Image split ready: train_images=%s test_images=%s.",
        len(train_indices),
        len(test_indices),
    )

    return train_inputs, train_targets, test_inputs, test_targets


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


def build_optimizer(name: str, iterations: int):
    """Create a Qiskit optimizer from the CLI name."""
    if iterations < 1:
        raise ValueError("--iterations must be at least 1.")

    if name == "cobyla":
        return COBYLA(maxiter=iterations)
    if name == "spsa":
        return SPSA(maxiter=iterations)
    if name == "l-bfgs-b":
        return L_BFGS_B(maxiter=iterations)
    if name == "adam":
        return ADAM(maxiter=iterations)
    raise ValueError(f"Unsupported optimizer: {name}")


def load_initial_weights(
    *,
    checkpoint_path: Path,
    num_weights: int,
    use_initial: bool,
    rng: np.random.Generator,
) -> np.ndarray:
    """Load checkpoint weights or create a fresh random initial point."""
    if use_initial:
        if checkpoint_path.exists():
            checkpoint = np.load(checkpoint_path)
            weights = np.asarray(checkpoint["weights"], dtype=float)
            if len(weights) != num_weights:
                raise ValueError(
                    "Checkpoint has "
                    f"{len(weights)} weights, but the QCNN expects {num_weights}."
                )
            LOGGER.info("Loaded initial weights from %s", checkpoint_path)
            return weights

        LOGGER.warning(
            "--use-initial was set, but %s does not exist; using random weights.",
            checkpoint_path,
        )

    return rng.uniform(-0.1, 0.1, size=num_weights)


def predict(qnn, inputs: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Return binary predictions from QCNN expectation values."""
    raw_outputs = np.asarray(qnn.forward(inputs, weights), dtype=float).reshape(-1)
    return np.where(raw_outputs >= 0.0, 1.0, -1.0)


def accuracy(
    qnn, inputs: np.ndarray, targets: np.ndarray, weights: np.ndarray
) -> float:
    predictions = predict(qnn, inputs, weights)
    return float(np.mean(predictions == targets))


def make_objective(
    *,
    qnn,
    train_inputs: np.ndarray,
    train_targets: np.ndarray,
    test_inputs: np.ndarray,
    test_targets: np.ndarray,
) -> Callable[[np.ndarray], float]:
    """Create a logging objective for optimizer-driven QCNN training."""
    evaluations = 0
    best_loss = float("inf")

    def objective(weights: np.ndarray) -> float:
        nonlocal evaluations, best_loss

        outputs = np.asarray(qnn.forward(train_inputs, weights), dtype=float).reshape(
            -1
        )
        loss = float(np.mean((outputs - train_targets) ** 2))
        evaluations += 1

        if loss < best_loss:
            best_loss = loss

        train_acc = accuracy(qnn, train_inputs, train_targets, weights)
        test_acc = accuracy(qnn, test_inputs, test_targets, weights)
        LOGGER.info(
            "eval=%03d loss=%.6f best=%.6f train_acc=%.3f test_acc=%.3f",
            evaluations,
            loss,
            best_loss,
            train_acc,
            test_acc,
        )
        return loss

    return objective


def train(args: argparse.Namespace) -> np.ndarray:
    configure_logging()
    rng = np.random.default_rng(args.seed)
    image_size = reduced_image_size(args.resolution_reduction)

    LOGGER.info("Loading training data from %s", args.feature_path)
    train_inputs, train_targets, test_inputs, test_targets = load_qcnn_data(
        feature_path=args.feature_path,
        label_path=args.label_path,
        dataset_path=args.dataset_path,
        train_size=args.train_size,
        test_size=args.test_size,
        train_images=args.train_images,
        test_images=args.test_images,
        positive_label=args.positive_label,
        seed=args.seed,
        image_size=image_size,
    )

    LOGGER.info(
        "Dataset ready: train=%s test=%s input_dim=%s",
        len(train_inputs),
        len(test_inputs),
        train_inputs.shape[1],
    )

    qnn = build_qcnn_estimator(
        num_inputs=train_inputs.shape[1],
        feature_path=args.feature_path,
        feature_map=args.feature_map,
    )
    optimizer = build_optimizer(args.optimizer, args.iterations)
    initial_weights = load_initial_weights(
        checkpoint_path=args.checkpoint_path,
        num_weights=len(qnn.weight_params),
        use_initial=args.use_initial,
        rng=rng,
    )

    LOGGER.info(
        "Starting training with optimizer=%s iterations=%s weights=%s",
        args.optimizer,
        args.iterations,
        len(initial_weights),
    )

    objective = make_objective(
        qnn=qnn,
        train_inputs=train_inputs,
        train_targets=train_targets,
        test_inputs=test_inputs,
        test_targets=test_targets,
    )
    result = optimizer.minimize(fun=objective, x0=initial_weights)
    final_weights = np.asarray(result.x, dtype=float)

    final_train_acc = accuracy(qnn, train_inputs, train_targets, final_weights)
    final_test_acc = accuracy(qnn, test_inputs, test_targets, final_weights)
    LOGGER.info(
        "Training complete: objective=%.6f train_acc=%.3f test_acc=%.3f",
        float(result.fun),
        final_train_acc,
        final_test_acc,
    )

    args.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.checkpoint_path,
        weights=final_weights,
        optimizer=args.optimizer,
        iterations=args.iterations,
        train_accuracy=final_train_acc,
        test_accuracy=final_test_acc,
        resolution_reduction=args.resolution_reduction,
        image_size=np.asarray(image_size, dtype=int),
    )
    LOGGER.info("Saved checkpoint to %s", args.checkpoint_path)
    return final_weights


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
