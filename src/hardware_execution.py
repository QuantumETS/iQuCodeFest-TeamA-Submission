import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, List, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.library import n_local, zz_feature_map
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService, Session, EstimatorV2 as Estimator
from qiskit_ibm_runtime.options import EstimatorOptions

from dataset_loader import load_brain_tumor_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

IBM_BACKEND_NAME = "ibm_quebec"
IBM_TOKEN_ENV = "TOKEN"
IBM_INSTANCE_ENV = "CRN"

SAVE_PATH = "data/preprocessed/"
IBM_JOB_CHECKPOINT_PATH = Path(SAVE_PATH) / "ibm_quebec_jobs.json"
IBM_CIRCUIT_PRINT_PATH = Path(SAVE_PATH) / "ibm_quebec_circuits"

# Hyperparameters: configurable kernel size and stride.
# KERNEL_SIZE may be an int (square) or a tuple (kh, kw).
# STRIDE may be an int or a tuple (sh, sw).
KERNEL_SIZE: Union[int, Tuple[int, int]] = (2, 2)
STRIDE: Union[int, Tuple[int, int]] = (2, 2)

# Padding mode: "valid" (no padding) or "same".
PADDING: str = "same"

# A 2x2 patch gives 4 input values and therefore 4 qubits.
# The number of output channels is a modelling choice. By default, use one
# channel per qubit, but each channel gets its own ansatz/filter parameters.
NUM_OUTPUT_CHANNELS: int | None = 4

# Image values are usually normalized in [0, 1]. Scaling them to [0, pi]
# makes the angle encoding less likely to collapse into a tiny-angle regime.
DATA_ANGLE_SCALE: float = np.pi

RANDOM_SEED = 42
IBM_RESULT_RETRY_DELAY_SECONDS = 60
IBM_SESSION_MAX_TIME = "8h"
TRANSPILER_OPTIMIZATION_LEVEL = 3
RUNTIME_MITIGATION_PROFILE = {
    "default_shots": 4096,
    "resilience_level": 2,
    "measure_mitigation": True,
    "zne_mitigation": True,
    "zne_amplifier": "gate_folding",
    "zne_noise_factors": [1, 3, 5],
    "zne_extrapolator": ["linear", "exponential"],
    # IBM Runtime does not allow PEC and ZNE to be enabled simultaneously.
    "pec_mitigation": False,
    "pec_max_overhead": 100,
    "twirling_enable_gates": True,
    "twirling_enable_measure": True,
    "twirling_num_randomizations": "auto",
    "twirling_shots_per_randomization": "auto",
    "twirling_strategy": "all",
    "dynamical_decoupling": True,
    "dynamical_decoupling_sequence": "XY4",
    "dynamical_decoupling_slack_distribution": "middle",
    "dynamical_decoupling_scheduling": "alap",
}


def _load_local_dotenv(dotenv_path: Path = Path(".env")) -> None:
    """Load simple KEY=VALUE pairs from a local .env file if not already set."""
    if not dotenv_path.exists():
        return

    for line in dotenv_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue

        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'\"")

        if key and key not in os.environ:
            os.environ[key] = value


def create_estimator_options() -> EstimatorOptions:
    """Configure IBM Runtime error mitigation and suppression options."""
    options = EstimatorOptions()
    options.default_shots = RUNTIME_MITIGATION_PROFILE["default_shots"]
    options.resilience_level = RUNTIME_MITIGATION_PROFILE["resilience_level"]

    options.resilience.measure_mitigation = RUNTIME_MITIGATION_PROFILE[
        "measure_mitigation"
    ]
    options.resilience.zne_mitigation = RUNTIME_MITIGATION_PROFILE["zne_mitigation"]
    options.resilience.zne.amplifier = RUNTIME_MITIGATION_PROFILE["zne_amplifier"]
    options.resilience.zne.noise_factors = RUNTIME_MITIGATION_PROFILE[
        "zne_noise_factors"
    ]
    options.resilience.zne.extrapolator = RUNTIME_MITIGATION_PROFILE[
        "zne_extrapolator"
    ]
    options.resilience.pec_mitigation = RUNTIME_MITIGATION_PROFILE["pec_mitigation"]
    if RUNTIME_MITIGATION_PROFILE["pec_mitigation"]:
        options.resilience.pec.max_overhead = RUNTIME_MITIGATION_PROFILE[
            "pec_max_overhead"
        ]

    options.twirling.enable_gates = RUNTIME_MITIGATION_PROFILE[
        "twirling_enable_gates"
    ]
    options.twirling.enable_measure = RUNTIME_MITIGATION_PROFILE[
        "twirling_enable_measure"
    ]
    options.twirling.num_randomizations = RUNTIME_MITIGATION_PROFILE[
        "twirling_num_randomizations"
    ]
    options.twirling.shots_per_randomization = RUNTIME_MITIGATION_PROFILE[
        "twirling_shots_per_randomization"
    ]
    options.twirling.strategy = RUNTIME_MITIGATION_PROFILE["twirling_strategy"]

    options.dynamical_decoupling.enable = RUNTIME_MITIGATION_PROFILE[
        "dynamical_decoupling"
    ]
    options.dynamical_decoupling.sequence_type = RUNTIME_MITIGATION_PROFILE[
        "dynamical_decoupling_sequence"
    ]
    options.dynamical_decoupling.extra_slack_distribution = RUNTIME_MITIGATION_PROFILE[
        "dynamical_decoupling_slack_distribution"
    ]
    options.dynamical_decoupling.scheduling_method = RUNTIME_MITIGATION_PROFILE[
        "dynamical_decoupling_scheduling"
    ]
    return options


def create_ibm_quebec_estimator() -> Tuple[Estimator, Any, QiskitRuntimeService, Session]:
    """Create an IBM Runtime estimator that submits jobs to IBM Quebec.

    Credentials are loaded from ``TOKEN`` and ``CRN`` environment variables,
    with a local ``.env`` file used as a fallback source.
    """
    _load_local_dotenv()

    token = os.environ.get(IBM_TOKEN_ENV)
    instance = os.environ.get(IBM_INSTANCE_ENV)

    if not token or not instance:
        raise RuntimeError(
            f"IBM credentials are required. Set {IBM_TOKEN_ENV} and "
            f"{IBM_INSTANCE_ENV} in the environment or local .env file."
        )

    service = QiskitRuntimeService(
        channel="ibm_quantum_platform",
        token="YOUR TOKEN", # to be manually inputted because UV and load_env issues
        instance="YOUR CRN INSTANCE",
    )
    backend = service.backend(IBM_BACKEND_NAME)
    logger.info("Using IBM Quantum backend: %s", backend.name)
    session = Session(backend=backend, max_time=IBM_SESSION_MAX_TIME)
    logger.info("Opened IBM Runtime session with max_time=%s", IBM_SESSION_MAX_TIME)
    return Estimator(mode=session, options=create_estimator_options()), backend, service, session


def load_job_checkpoint(path: Path = IBM_JOB_CHECKPOINT_PATH) -> dict[str, Any]:
    """Load saved IBM Runtime job IDs and completed patch results."""
    if not path.exists():
        return {"backend": IBM_BACKEND_NAME, "patches": {}}

    with path.open("r", encoding="utf-8") as checkpoint_file:
        checkpoint = json.load(checkpoint_file)

    checkpoint.setdefault("backend", IBM_BACKEND_NAME)
    checkpoint.setdefault("patches", {})
    return checkpoint


def save_job_checkpoint(
    checkpoint: dict[str, Any],
    path: Path = IBM_JOB_CHECKPOINT_PATH,
) -> None:
    """Persist IBM Runtime job IDs and patch results atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(".tmp")
    temp_path.write_text(
        json.dumps(checkpoint, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temp_path.replace(path)


def sanitize_run_id(run_id: str) -> str:
    """Return a filesystem-safe run ID for independent hardware runs."""
    safe_run_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", run_id.strip())
    if not safe_run_id:
        raise ValueError("--run-id must contain at least one valid character")
    return safe_run_id


def hardware_run_paths(run_id: str | None) -> dict[str, Path]:
    """Build output paths for the default run or an isolated named run."""
    if run_id is None:
        base_path = Path(SAVE_PATH)
    else:
        base_path = Path(SAVE_PATH) / "hardware_runs" / sanitize_run_id(run_id)

    return {
        "base": base_path,
        "checkpoint": base_path / "ibm_quebec_jobs.json",
        "circuits": base_path / "ibm_quebec_circuits",
        "partial": base_path / "q_train_images_partial.npy",
        "final": base_path / "q_train_images.npy",
    }


def validate_job_checkpoint_config(
    checkpoint: dict[str, Any],
    expected_config: dict[str, Any],
    checkpoint_path: Path = IBM_JOB_CHECKPOINT_PATH,
) -> None:
    """Prevent reusing IBM job results from an incompatible preprocessing run."""
    if not checkpoint.get("patches"):
        return

    mismatches = [
        key
        for key, expected_value in expected_config.items()
        if key in checkpoint and checkpoint.get(key) != expected_value
    ]
    if mismatches:
        mismatch_text = ", ".join(mismatches)
        raise RuntimeError(
            "Existing IBM job checkpoint does not match this run "
            f"({mismatch_text}). Delete {checkpoint_path} to start a fresh "
            "hardware run, or use a different --run-id."
        )


def _get_runtime_job_id(job: Any) -> str:
    """Return a Runtime job ID across qiskit-ibm-runtime API variants."""
    job_id = getattr(job, "job_id", None)
    if callable(job_id):
        return str(job_id())
    if job_id is not None:
        return str(job_id)
    raise RuntimeError("IBM Runtime job did not expose a job_id.")


def write_job_circuits(
    checkpoint_key: str,
    pubs: Sequence[tuple[QuantumCircuit, SparsePauliOp]],
    circuit_print_path: Path = IBM_CIRCUIT_PRINT_PATH,
) -> list[Path]:
    """Write optimized backend-ISA circuit PNGs for a job before submission."""
    circuit_print_path.mkdir(parents=True, exist_ok=True)
    safe_name = checkpoint_key.replace("/", "__")
    metadata_path = circuit_print_path / f"{safe_name}.json"
    metadata = {
        "checkpoint_key": checkpoint_key,
        "backend": IBM_BACKEND_NAME,
        "transpiler_optimization_level": TRANSPILER_OPTIMIZATION_LEVEL,
        "runtime_mitigation_profile": RUNTIME_MITIGATION_PROFILE,
        "channels": [],
    }
    png_paths = []

    for channel, (circuit, observable) in enumerate(pubs):
        png_path = circuit_print_path / f"{safe_name}__channel_{channel:02d}.png"
        figure = circuit.draw(output="mpl", idle_wires=False)
        figure.savefig(png_path, bbox_inches="tight", dpi=200)
        plt.close(figure)

        png_paths.append(png_path)
        metadata["channels"].append(
            {
                "channel": channel,
                "observable": str(observable),
                "png": str(png_path),
            }
        )

    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return png_paths


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


def build_channel_pubs(
    qc: QuantumCircuit,
    x_params: Sequence[Parameter],
    theta_params: Sequence[Parameter],
    patch: Sequence[float],
    theta_values_by_channel: np.ndarray,
    pass_manager: Any | None = None,
) -> list[tuple[QuantumCircuit, SparsePauliOp]]:
    """Build estimator publications for every output channel of one patch."""
    n_qubits = qc.num_qubits
    pubs = []

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

        if pass_manager is not None:
            circuit = pass_manager.run(circuit)
            observable = observable.apply_layout(circuit.layout)

        pubs.append((circuit, observable))

    return pubs


def _estimator_result_values(primitive_result: Any) -> List[float]:
    return [float(pub_result.data.evs) for pub_result in primitive_result]


def is_transient_ibm_error(exc: Exception) -> bool:
    """Return true for IBM/network errors that should be retried."""
    error_text = str(exc)
    transient_markers = (
        "Read timed out",
        "Max retries exceeded",
        "ConnectionError",
        "ReadTimeoutError",
        "handshake operation timed out",
        "quantum.cloud.ibm.com",
    )
    return any(marker in error_text for marker in transient_markers)


def estimate_channel_expectations(
    qc: QuantumCircuit,
    x_params: Sequence[Parameter],
    theta_params: Sequence[Parameter],
    patch: Sequence[float],
    theta_values_by_channel: np.ndarray,
    estimator: StatevectorEstimator | Estimator,
    pass_manager: Any | None = None,
) -> List[float]:
    """Estimate one output value per channel for a single image patch.

    Each channel uses the same encoded patch but its own ansatz/filter
    parameter vector. This is closer to the classical idea that each output
    channel has its own convolutional filter.
    """
    pubs = build_channel_pubs(
        qc=qc,
        x_params=x_params,
        theta_params=theta_params,
        patch=patch,
        theta_values_by_channel=theta_values_by_channel,
        pass_manager=pass_manager,
    )
    primitive_result = estimator.run(pubs=pubs).result()
    return _estimator_result_values(primitive_result)


def estimate_channel_expectations_with_checkpoint(
    qc: QuantumCircuit,
    x_params: Sequence[Parameter],
    theta_params: Sequence[Parameter],
    patch: Sequence[float],
    theta_values_by_channel: np.ndarray,
    estimator: Estimator,
    service: QiskitRuntimeService,
    checkpoint: dict[str, Any],
    checkpoint_key: str,
    pass_manager: Any,
    checkpoint_path: Path = IBM_JOB_CHECKPOINT_PATH,
    circuit_print_path: Path = IBM_CIRCUIT_PRINT_PATH,
    submit_missing: bool = True,
) -> List[float]:
    """Submit or resume a hardware estimator job for a single image patch."""
    patch_entries = checkpoint.setdefault("patches", {})
    entry = patch_entries.setdefault(checkpoint_key, {})

    if "results" in entry:
        return [float(value) for value in entry["results"]]

    if "job_id" in entry:
        job_id = entry["job_id"]
        logger.info("Resuming IBM Runtime job %s for %s", job_id, checkpoint_key)
        job = service.job(job_id)
    else:
        if not submit_missing:
            raise RuntimeError(
                f"No saved IBM job for {checkpoint_key}; not submitting because "
                "--collect-only was requested."
            )

        pubs = build_channel_pubs(
            qc=qc,
            x_params=x_params,
            theta_params=theta_params,
            patch=patch,
            theta_values_by_channel=theta_values_by_channel,
            pass_manager=pass_manager,
        )
        circuit_paths = write_job_circuits(
            checkpoint_key=checkpoint_key,
            pubs=pubs,
            circuit_print_path=circuit_print_path,
        )
        logger.info(
            "Wrote %s optimized circuit PNG(s) for %s to %s",
            len(circuit_paths),
            checkpoint_key,
            circuit_print_path,
        )
        job = estimator.run(pubs=pubs)
        job_id = _get_runtime_job_id(job)
        logger.info("Submitted IBM Runtime job %s for %s", job_id, checkpoint_key)

        entry["job_id"] = job_id
        entry["status"] = "submitted"
        save_job_checkpoint(checkpoint, checkpoint_path)

    while True:
        try:
            results = _estimator_result_values(job.result())
            break
        except Exception as exc:
            if not is_transient_ibm_error(exc):
                entry["status"] = "error"
                entry["error"] = str(exc)
                save_job_checkpoint(checkpoint, checkpoint_path)
                raise

            entry["status"] = "waiting_for_result_retry"
            entry["last_transient_error"] = str(exc)
            save_job_checkpoint(checkpoint, checkpoint_path)
            logger.warning(
                "Transient IBM result retrieval error for job %s. "
                "Will retry in %s seconds. Error: %s",
                job_id,
                IBM_RESULT_RETRY_DELAY_SECONDS,
                exc,
            )
            time.sleep(IBM_RESULT_RETRY_DELAY_SECONDS)

    entry["results"] = results
    entry["status"] = "completed"
    entry.pop("error", None)
    entry.pop("last_transient_error", None)
    save_job_checkpoint(checkpoint, checkpoint_path)
    return results


def quanv(
    image: np.ndarray,
    qc: QuantumCircuit,
    x_params: Sequence[Parameter],
    theta_params: Sequence[Parameter],
    theta_values_by_channel: np.ndarray,
    estimator: StatevectorEstimator | Estimator,
    pass_manager: Any | None = None,
    service: QiskitRuntimeService | None = None,
    checkpoint: dict[str, Any] | None = None,
    sample_idx: int | None = None,
    checkpoint_path: Path = IBM_JOB_CHECKPOINT_PATH,
    circuit_print_path: Path = IBM_CIRCUIT_PRINT_PATH,
    sample_partial_path: Path | None = None,
    submit_missing: bool = True,
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
    if not submit_missing:
        out.fill(np.nan)

    for i_out, row in enumerate(range(0, height - kh + 1, sh)):
        for j_out, col in enumerate(range(0, width - kw + 1, sw)):
            patch = img[row : row + kh, col : col + kw].reshape(-1)
            patch_angles = (patch * DATA_ANGLE_SCALE).tolist()

            if service is not None and checkpoint is not None and sample_idx is not None:
                checkpoint_key = (
                    f"sample_{sample_idx:06d}/row_{i_out:04d}/col_{j_out:04d}"
                )
                try:
                    q_results = estimate_channel_expectations_with_checkpoint(
                        qc=qc,
                        x_params=x_params,
                        theta_params=theta_params,
                        patch=patch_angles,
                        theta_values_by_channel=theta_values_by_channel,
                        estimator=estimator,
                        service=service,
                        checkpoint=checkpoint,
                        checkpoint_key=checkpoint_key,
                        pass_manager=pass_manager,
                        checkpoint_path=checkpoint_path,
                        circuit_print_path=circuit_print_path,
                        submit_missing=submit_missing,
                    )
                except RuntimeError as exc:
                    if submit_missing:
                        raise
                    logger.info("%s", exc)
                    q_results = out[i_out, j_out, :].tolist()
            else:
                q_results = estimate_channel_expectations(
                    qc=qc,
                    x_params=x_params,
                    theta_params=theta_params,
                    patch=patch_angles,
                    theta_values_by_channel=theta_values_by_channel,
                    estimator=estimator,
                    pass_manager=pass_manager,
                )

            out[i_out, j_out, :] = q_results
            if sample_partial_path is not None:
                sample_partial_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(sample_partial_path, out)

        if i_out % max(1, out_h // 8) == 0:
            logger.info("Processed output row %s/%s", i_out + 1, out_h)

    return out


def mock_image(height: int = 256, width: int = 256) -> np.ndarray:
    """Create a mock grayscale image with shape ``(height, width, 1)``."""
    return np.random.rand(height, width, 1)


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


def extract_quantum_feature_maps(args) -> None:
    """Extract, save, load, and display quanvolutional feature maps."""
    run_paths = hardware_run_paths(args.run_id)
    logger.info("Starting hardware_execution.py")
    logger.info("preprocess=%s samples=%s seed=%s run_id=%s", args.preprocess, args.samples, args.seed, args.run_id)
    logger.info("checkpoint path: %s", run_paths["checkpoint"])
    logger.info("circuit print path: %s", run_paths["circuits"])
    logger.info("partial output path: %s", run_paths["partial"])
    logger.info("final output path: %s", run_paths["final"])

    logger.info("Loading dataset from data/archive/Data")
    train_images = load_brain_tumor_dataset("data/archive/Data")[0][: args.samples]
    logger.info("Loaded %s image(s) for this run", len(train_images))

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
        run_paths["base"].mkdir(parents=True, exist_ok=True)
        logger.info("Creating IBM Quebec estimator with Runtime mitigation options")
        estimator, backend, service, session = create_ibm_quebec_estimator()
        try:
            logger.info(
                "Generating preset pass manager with optimization level %s",
                TRANSPILER_OPTIMIZATION_LEVEL,
            )
            pass_manager = generate_preset_pass_manager(
                backend=backend,
                optimization_level=TRANSPILER_OPTIMIZATION_LEVEL,
            )
            checkpoint = load_job_checkpoint(run_paths["checkpoint"])
            checkpoint_config = {
                "backend": IBM_BACKEND_NAME,
                "run_id": args.run_id,
                "seed": args.seed,
                "kernel_size": [kh, kw],
                "stride": list(_normalize_pair(STRIDE)),
                "padding": PADDING,
                "num_output_channels": num_output_channels,
                "transpiler_optimization_level": TRANSPILER_OPTIMIZATION_LEVEL,
                "runtime_mitigation_profile": RUNTIME_MITIGATION_PROFILE,
                "ibm_runtime_session_max_time": IBM_SESSION_MAX_TIME,
            }
            validate_job_checkpoint_config(
                checkpoint=checkpoint,
                expected_config=checkpoint_config,
                checkpoint_path=run_paths["checkpoint"],
            )
            checkpoint.update(checkpoint_config)
            checkpoint["samples"] = args.samples
            save_job_checkpoint(checkpoint, run_paths["checkpoint"])

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
                        pass_manager=pass_manager,
                        service=service,
                        checkpoint=checkpoint,
                        sample_idx=idx,
                        checkpoint_path=run_paths["checkpoint"],
                        circuit_print_path=run_paths["circuits"],
                        sample_partial_path=(
                            run_paths["base"]
                            / f"q_train_image_sample_{idx:06d}_partial.npy"
                        ),
                        submit_missing=not args.collect_only,
                    )
                )
                np.save(run_paths["partial"], np.asarray(q_train_images))

            q_train_images = np.asarray(q_train_images)
            if np.isnan(q_train_images).any():
                logger.info(
                    "Partial results saved to %s and per-sample partial files under %s. "
                    "Final output was not saved because some patches are missing.",
                    run_paths["partial"],
                    run_paths["base"],
                )
            else:
                log_channel_statistics(q_train_images)
                np.save(run_paths["final"], q_train_images)
                logger.info("Final output saved to %s", run_paths["final"])
        finally:
            logger.info("Closing IBM Runtime session")
            session.close()

    else:
        q_train_images = np.load(run_paths["final"])

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
        "--samples",
        type=int,
        default=1,
        help="Number of samples to process for testing.",
    )
    arg_parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help="Random seed for fixed quanvolutional filter parameters.",
    )
    arg_parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help=(
            "Optional isolated hardware run ID. When set, checkpoint, circuits, "
            "partial output, and final output are stored under "
            "data/preprocessed/hardware_runs/<run-id>/."
        ),
    )
    arg_parser.add_argument(
        "--collect-only",
        action="store_true",
        help=(
            "Only retrieve results for saved IBM job IDs. Do not submit new "
            "jobs for missing patches; missing patch values are saved as NaN "
            "in partial outputs."
        ),
    )

    extract_quantum_feature_maps(arg_parser.parse_args())
