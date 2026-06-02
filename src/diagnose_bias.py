"""Command-line diagnostics for QCNN tumor-classifier prediction bias."""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from inference import (
    DEFAULT_CHECKPOINT_PATH,
    _checkpoint_image_size,
    _extract_spatial_vectors,
    _forward_in_batches,
    _load_image,
)
from qcnn import DEFAULT_FEATURE_PATH, build_qcnn_estimator


LOGGER = logging.getLogger(__name__)

DEFAULT_DATASET_PATH = Path("data/archive/Data")
DEFAULT_LABEL_PATH = Path("data/preprocessed/q_train_labels.npy")
DEFAULT_CLASSES = (
    "normal",
    "meningioma_tumor",
)
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")


@dataclass(frozen=True)
class ImageDiagnostic:
    """Bias diagnostics for one image."""

    class_name: str
    image_path: Path
    expected_binary: str
    predicted_binary: str
    mean_score: float
    threshold_margin: float
    normal_patch_percent: float
    tumor_patch_percent: float
    min_score: float
    max_score: float
    std_score: float

    @property
    def is_correct(self) -> bool:
        return self.expected_binary == self.predicted_binary


def configure_logging(verbose: bool) -> None:
    """Configure logs for optional pipeline progress."""
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logging.getLogger().setLevel(logging.INFO if verbose else logging.WARNING)
    if not verbose:
        logging.getLogger("qiskit_machine_learning").setLevel(logging.ERROR)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose binary prediction bias in the checkpoint-based QCNN "
            "tumor classifier."
        )
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help="Path to the raw dataset class folders.",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=DEFAULT_CHECKPOINT_PATH,
        help="Path to the QCNN weight checkpoint.",
    )
    parser.add_argument(
        "--feature-path",
        type=Path,
        default=DEFAULT_FEATURE_PATH,
        help="Path to saved quanvolution features used for size warnings.",
    )
    parser.add_argument(
        "--label-path",
        type=Path,
        default=DEFAULT_LABEL_PATH,
        help="Path to labels saved with the quanvolution feature maps.",
    )
    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=5,
        help="Number of images to sample from each class folder.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic balanced sampling.",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        default=list(DEFAULT_CLASSES),
        help="Class folders to include in the diagnostic run.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed quanvolution and QCNN progress logs.",
    )
    return parser.parse_args()


def binary_target_for_class(class_name: str) -> str:
    """Map raw dataset class folders to the app's binary output space."""
    return "normal" if class_name == "normal" else "tumor"


def binary_prediction_from_score(mean_score: float) -> str:
    """Return the binary prediction used by inference.classify_image."""
    return "normal" if mean_score >= 0.0 else "tumor"


def image_files_for_class(dataset_path: Path, class_name: str) -> list[Path]:
    """Return sorted image files for one raw dataset class."""
    class_path = dataset_path / class_name
    if not class_path.is_dir():
        raise FileNotFoundError(f"Dataset class folder not found: {class_path}")

    image_paths = sorted(
        path
        for path in class_path.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_paths:
        raise FileNotFoundError(
            f"No images found in dataset class folder: {class_path}"
        )

    return image_paths


def sample_dataset_images(
    dataset_path: Path,
    class_names: Iterable[str],
    samples_per_class: int,
    seed: int,
) -> list[tuple[str, Path]]:
    """Sample a balanced set of raw dataset images."""
    if samples_per_class < 1:
        raise ValueError("--samples-per-class must be at least 1.")

    rng = np.random.default_rng(seed)
    selected: list[tuple[str, Path]] = []

    for class_name in class_names:
        image_paths = image_files_for_class(dataset_path, class_name)
        sample_count = min(samples_per_class, len(image_paths))
        if sample_count < samples_per_class:
            LOGGER.warning(
                "Class %s has only %s images; requested %s.",
                class_name,
                sample_count,
                samples_per_class,
            )

        indices = rng.choice(len(image_paths), size=sample_count, replace=False)
        for index in sorted(indices):
            selected.append((class_name, image_paths[int(index)]))

    return selected


def load_checkpoint_metadata(
    checkpoint_path: Path,
) -> tuple[np.ndarray, tuple[int, int], dict[str, object]]:
    """Load checkpoint weights, image size, and printable metadata."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    with np.load(checkpoint_path) as checkpoint:
        weights = np.asarray(checkpoint["weights"], dtype=float)
        image_size = _checkpoint_image_size(checkpoint)
        metadata = {
            key: checkpoint[key].item()
            if np.asarray(checkpoint[key]).shape == ()
            else np.asarray(checkpoint[key]).tolist()
            for key in checkpoint.files
            if key != "weights"
        }

    return weights, image_size, metadata


def diagnose_image(
    *,
    class_name: str,
    image_path: Path,
    image_size: tuple[int, int],
    qnn,
    weights: np.ndarray,
) -> ImageDiagnostic:
    """Run QCNN diagnostics for one image."""
    spatial_vectors = _extract_spatial_vectors(_load_image(image_path, image_size))
    outputs = _forward_in_batches(qnn, spatial_vectors, weights)

    mean_score = float(outputs.mean())
    normal_votes = int(np.count_nonzero(outputs >= 0.0))
    total_votes = len(outputs)
    normal_patch_percent = 100.0 * normal_votes / total_votes
    tumor_patch_percent = 100.0 - normal_patch_percent
    predicted_binary = binary_prediction_from_score(mean_score)

    return ImageDiagnostic(
        class_name=class_name,
        image_path=image_path,
        expected_binary=binary_target_for_class(class_name),
        predicted_binary=predicted_binary,
        mean_score=mean_score,
        threshold_margin=abs(mean_score),
        normal_patch_percent=normal_patch_percent,
        tumor_patch_percent=tumor_patch_percent,
        min_score=float(outputs.min()),
        max_score=float(outputs.max()),
        std_score=float(outputs.std()),
    )


def diagnose_images(
    selected_images: Iterable[tuple[str, Path]],
    checkpoint_path: Path,
) -> tuple[list[ImageDiagnostic], dict[str, object]]:
    """Run diagnostics for a balanced image sample."""
    weights, image_size, metadata = load_checkpoint_metadata(checkpoint_path)
    selected_images = list(selected_images)

    if not selected_images:
        return [], metadata

    first_image = selected_images[0]
    first_array = _load_image(first_image[1], image_size)
    first_vectors = _extract_spatial_vectors(first_array)
    qnn = build_qcnn_estimator(num_inputs=first_vectors.shape[-1])

    first_outputs = _forward_in_batches(qnn, first_vectors, weights)
    first_mean = float(first_outputs.mean())
    first_normal_votes = int(np.count_nonzero(first_outputs >= 0.0))
    first_total_votes = len(first_outputs)

    diagnostics = [
        ImageDiagnostic(
            class_name=first_image[0],
            image_path=first_image[1],
            expected_binary=binary_target_for_class(first_image[0]),
            predicted_binary=binary_prediction_from_score(first_mean),
            mean_score=first_mean,
            threshold_margin=abs(first_mean),
            normal_patch_percent=100.0 * first_normal_votes / first_total_votes,
            tumor_patch_percent=100.0
            - (100.0 * first_normal_votes / first_total_votes),
            min_score=float(first_outputs.min()),
            max_score=float(first_outputs.max()),
            std_score=float(first_outputs.std()),
        )
    ]

    for class_name, image_path in selected_images[1:]:
        diagnostics.append(
            diagnose_image(
                class_name=class_name,
                image_path=image_path,
                image_size=image_size,
                qnn=qnn,
                weights=weights,
            )
        )

    return diagnostics, metadata


def warn_about_preprocessed_features(
    feature_path: Path,
    label_path: Path,
    dataset_path: Path,
) -> None:
    """Print warnings for feature tensors that are too small for reliable training."""
    if not feature_path.exists():
        print(f"Warning: preprocessed feature file not found: {feature_path}")
        return

    features = np.load(feature_path, mmap_mode="r")
    feature_images = int(features.shape[0]) if features.ndim >= 1 else 0
    raw_counts = {
        class_name: len(image_files_for_class(dataset_path, class_name))
        for class_name in DEFAULT_CLASSES
        if (dataset_path / class_name).is_dir()
    }
    raw_total = sum(raw_counts.values())

    print("Preprocessed feature check:")
    print(f"  feature_path: {feature_path}")
    print(f"  feature_shape: {tuple(features.shape)}")
    print(f"  raw_dataset_images_counted: {raw_total}")

    if label_path.exists():
        labels = np.load(label_path)
        unique_labels, counts = np.unique(labels, return_counts=True)
        label_counts = {
            int(label): int(count)
            for label, count in zip(unique_labels.tolist(), counts.tolist())
        }
        print(f"  label_path: {label_path}")
        print(f"  saved_label_counts: {label_counts}")
        if len(unique_labels) == 1:
            print(
                "  warning: saved preprocessed labels contain one class only; "
                "training cannot learn a useful binary boundary."
            )
    else:
        print(f"  label_path: {label_path} (missing)")
        print(
            "  warning: saved labels are missing; training will need a legacy "
            "dataset-order fallback and may mismatch features to labels."
        )

    if feature_images < len(DEFAULT_CLASSES) * 5:
        print(
            "  warning: very few preprocessed images are saved; training may report "
            "high accuracy while seeing little image-level diversity."
        )
        print(
            "  warning: with sequential preprocessing, this may also be class-skewed "
            "toward the first dataset folders."
        )

    if raw_total and feature_images < raw_total * 0.05:
        print(
            "  warning: saved features cover less than 5% of counted raw images; "
            "this is a likely generalization risk."
        )


def print_checkpoint_metadata(metadata: dict[str, object]) -> None:
    print("Checkpoint metadata:")
    for key in (
        "optimizer",
        "iterations",
        "train_accuracy",
        "test_accuracy",
        "resolution_reduction",
        "image_size",
    ):
        if key in metadata:
            print(f"  {key}: {metadata[key]}")


def print_image_rows(diagnostics: list[ImageDiagnostic]) -> None:
    print("Per-image diagnostics:")
    print(
        "  class                expected predicted mean_score margin  "
        "normal% tumor%  min      max      std     image"
    )
    for item in diagnostics:
        print(
            f"  {item.class_name:<20} "
            f"{item.expected_binary:<8} "
            f"{item.predicted_binary:<9} "
            f"{item.mean_score:>+10.6f} "
            f"{item.threshold_margin:>7.6f} "
            f"{item.normal_patch_percent:>7.2f} "
            f"{item.tumor_patch_percent:>6.2f} "
            f"{item.min_score:>+7.3f} "
            f"{item.max_score:>+7.3f} "
            f"{item.std_score:>7.4f} "
            f"{item.image_path.name}"
        )


def print_class_summary(diagnostics: list[ImageDiagnostic]) -> None:
    print("Class summary:")
    for class_name in sorted({item.class_name for item in diagnostics}):
        class_items = [item for item in diagnostics if item.class_name == class_name]
        normal_predictions = sum(
            item.predicted_binary == "normal" for item in class_items
        )
        tumor_predictions = len(class_items) - normal_predictions
        accuracy = sum(item.is_correct for item in class_items) / len(class_items)
        mean_score = float(np.mean([item.mean_score for item in class_items]))
        mean_normal_votes = float(
            np.mean([item.normal_patch_percent for item in class_items])
        )
        print(
            f"  {class_name:<20} n={len(class_items):<3} "
            f"normal_pred={normal_predictions:<3} tumor_pred={tumor_predictions:<3} "
            f"binary_acc={accuracy:.3f} avg_score={mean_score:+.6f} "
            f"avg_normal_patches={mean_normal_votes:.2f}%"
        )


def print_confusion_summary(diagnostics: list[ImageDiagnostic]) -> None:
    labels = ("normal", "tumor")
    print("Binary confusion summary:")
    print("  expected -> predicted counts")
    for expected in labels:
        row_items = [item for item in diagnostics if item.expected_binary == expected]
        counts = {
            predicted: sum(item.predicted_binary == predicted for item in row_items)
            for predicted in labels
        }
        print(
            f"  {expected:<6} -> normal={counts['normal']:<3} "
            f"tumor={counts['tumor']:<3}"
        )


def print_wrong_examples(diagnostics: list[ImageDiagnostic], limit: int = 5) -> None:
    wrong = [item for item in diagnostics if not item.is_correct]
    wrong.sort(key=lambda item: item.threshold_margin, reverse=True)

    print("Most confidently wrong:")
    if not wrong:
        print("  none")
        return

    for item in wrong[:limit]:
        print(
            f"  {item.class_name:<20} expected={item.expected_binary:<6} "
            f"predicted={item.predicted_binary:<6} score={item.mean_score:+.6f} "
            f"margin={item.threshold_margin:.6f} image={item.image_path.name}"
        )


def print_bias_hint(diagnostics: list[ImageDiagnostic]) -> None:
    if not diagnostics:
        return

    tumor_predictions = sum(item.predicted_binary == "tumor" for item in diagnostics)
    tumor_rate = tumor_predictions / len(diagnostics)
    normal_items = [item for item in diagnostics if item.expected_binary == "normal"]
    normal_tumor_rate = (
        sum(item.predicted_binary == "tumor" for item in normal_items)
        / len(normal_items)
        if normal_items
        else 0.0
    )

    print("Bias hint:")
    print(f"  tumor_prediction_rate: {tumor_rate:.3f}")
    if normal_items:
        print(f"  normal_images_predicted_as_tumor_rate: {normal_tumor_rate:.3f}")
    if normal_items and normal_tumor_rate == 1.0:
        print("  warning: every sampled normal image was classified as tumor.")
    elif tumor_rate >= 0.8:
        print("  warning: sampled predictions are heavily skewed toward tumor.")


def run(args: argparse.Namespace) -> int:
    configure_logging(args.verbose)

    selected_images = sample_dataset_images(
        dataset_path=args.dataset_path,
        class_names=args.classes,
        samples_per_class=args.samples_per_class,
        seed=args.seed,
    )
    diagnostics, metadata = diagnose_images(selected_images, args.checkpoint_path)

    print_checkpoint_metadata(metadata)
    print()
    warn_about_preprocessed_features(
        args.feature_path,
        args.label_path,
        args.dataset_path,
    )
    print()
    print_image_rows(diagnostics)
    print()
    print_class_summary(diagnostics)
    print()
    print_confusion_summary(diagnostics)
    print()
    print_wrong_examples(diagnostics)
    print()
    print_bias_hint(diagnostics)
    return 0


def main() -> None:
    raise SystemExit(run(parse_args()))


if __name__ == "__main__":
    main()
