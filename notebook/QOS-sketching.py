from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

try:
    import pygame
except ImportError as exc:
    raise SystemExit("pygame is required. Install it with: pip install pygame") from exc

try:
    import tkinter as tk
    from tkinter import filedialog
except ImportError:
    tk = None
    filedialog = None

try:
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import RYGate
    from qiskit.quantum_info import Statevector
except ImportError as exc:
    raise SystemExit("qiskit is required. Install it with: pip install qiskit") from exc


ROOT = Path(__file__).resolve().parent
WINDOW_SIZE = (1240, 800)
PANEL_BG = (26, 29, 35)
TEXT = (233, 237, 242)
MUTED = (165, 174, 188)
ACCENT = (88, 169, 255)
BUTTON = (48, 56, 68)
BUTTON_HOVER = (62, 72, 87)
IMAGE_SIZE = 32
DEFAULT_K = 12
MAX_K = IMAGE_SIZE * IMAGE_SIZE

DEFAULT_IMAGES = [
    ROOT
    / "iqucodefest-2026-main"
    / "side_quests"
    / "challenge_03_qml_image_classification"
    / "lab3.png",
    ROOT
    / "iqucodefest-2026-main"
    / "side_quests"
    / "challenge_02_generative_quantum_art"
    / "images"
    / "panda_colored.png",
    ROOT
    / "iqucodefest-2026-main"
    / "side_quests"
    / "challenge_02_generative_quantum_art"
    / "images"
    / "hello_kitty.png",
]


@dataclass
class SketchResult:
    path: Path
    original: np.ndarray
    reconstructed: np.ndarray
    error: np.ndarray
    circuit: QuantumCircuit
    indices: list[tuple[int, int]]
    decoded_coefficients: np.ndarray
    mae: float
    mse: float
    k: int
    index_qubits: int
    padded_slots: int
    max_abs_coeff: float


def clamp01(arr: np.ndarray) -> np.ndarray:
    return np.clip(arr, 0.0, 1.0)


def load_grayscale(path: Path, size: int = IMAGE_SIZE) -> np.ndarray:
    img = Image.open(path).convert("L")
    img = img.resize((size, size), Image.Resampling.LANCZOS)
    return np.asarray(img, dtype=np.float64) / 255.0


def reconstruct_from_sampled_pixels(
    shape: tuple[int, int], indices: list[tuple[int, int]], values: np.ndarray
) -> np.ndarray:
    if not indices:
        return np.zeros(shape, dtype=np.float64)

    sample_coords = np.array(indices, dtype=np.int64)
    sample_y = sample_coords[:, 0]
    sample_x = sample_coords[:, 1]
    grid_y, grid_x = np.indices(shape)
    distances = (grid_y[..., None] - sample_y) ** 2 + (grid_x[..., None] - sample_x) ** 2
    nearest = np.argmin(distances, axis=2)
    return values[nearest]


def qos_inspired_encode_decode(path: Path, k: int) -> SketchResult:
    original = load_grayscale(path)
    flat_pixels = original.ravel()

    k = min(max(1, k), flat_pixels.size)
    rng = np.random.default_rng()
    sampled_flat_indices = rng.choice(flat_pixels.size, size=k, replace=False)
    indices = [
        tuple(int(part) for part in np.unravel_index(int(idx), original.shape))
        for idx in sampled_flat_indices
    ]

    selected = np.array([original[idx] for idx in indices], dtype=np.float64)
    max_abs = float(np.max(selected)) if selected.size else 1.0
    if max_abs < 1e-12:
        max_abs = 1.0

    index_qubits = max(1, math.ceil(math.log2(k)))
    padded_slots = 2**index_qubits

    # Sparse FRQI/QOS-inspired sketch:
    # - each encoding shot draws fresh random pixel positions
    # - sampled pixel coordinates are classical oracle metadata
    # - an index register selects one sampled pixel
    # - one value qubit stores the normalized pixel intensity by controlled RY
    # - decoding estimates value probabilities per sampled index, not every image pixel
    scaled_magnitudes = np.clip(selected / max_abs, 0.0, 1.0)
    value_qubit = index_qubits
    circuit = QuantumCircuit(index_qubits + 1, name="random_pixel_qos_sketch")
    for qubit in range(index_qubits):
        circuit.h(qubit)

    for slot, value in enumerate(scaled_magnitudes):
        theta = 2.0 * math.asin(math.sqrt(float(value)))
        if index_qubits == 0:
            circuit.ry(theta, value_qubit)
            continue

        ctrl_state = format(slot, f"0{index_qubits}b")
        gate = RYGate(theta).control(index_qubits, ctrl_state=ctrl_state)
        circuit.append(gate, list(range(index_qubits)) + [value_qubit])

    state = Statevector.from_instruction(circuit)
    decoded_scaled = decode_indexed_value_probabilities(state, index_qubits, k)

    decoded_coefficients = decoded_scaled * max_abs

    reconstructed = clamp01(
        reconstruct_from_sampled_pixels(original.shape, indices, decoded_coefficients)
    )
    error = np.abs(original - reconstructed)
    mae = float(np.mean(error))
    mse = float(np.mean(error * error))

    return SketchResult(
        path=path,
        original=original,
        reconstructed=reconstructed,
        error=error,
        circuit=circuit,
        indices=[(int(y), int(x)) for y, x in indices],
        decoded_coefficients=decoded_coefficients,
        mae=mae,
        mse=mse,
        k=k,
        index_qubits=index_qubits,
        padded_slots=padded_slots,
        max_abs_coeff=max_abs,
    )


def decode_indexed_value_probabilities(
    state: Statevector, index_qubits: int, k: int
) -> np.ndarray:
    data = np.asarray(state.data)
    probabilities = np.abs(data) ** 2
    value_qubit = index_qubits
    numerators = np.zeros(k, dtype=np.float64)
    denominators = np.zeros(k, dtype=np.float64)

    for basis_index, probability in enumerate(probabilities):
        slot = basis_index & ((1 << index_qubits) - 1)
        if slot >= k:
            continue

        denominators[slot] += float(probability)
        if (basis_index >> value_qubit) & 1:
            numerators[slot] += float(probability)

    return np.divide(
        numerators,
        denominators,
        out=np.zeros_like(numerators),
        where=denominators > 1e-15,
    )


def to_rgb_surface_data(gray: np.ndarray) -> np.ndarray:
    gray = (clamp01(gray) * 255).astype(np.uint8)
    return np.repeat(gray[..., None], 3, axis=2)


def error_surface_data(error: np.ndarray) -> np.ndarray:
    normalized = error / max(float(np.max(error)), 1e-9)
    rgb = np.zeros((*error.shape, 3), dtype=np.uint8)
    rgb[..., 0] = (normalized * 255).astype(np.uint8)
    rgb[..., 1] = ((1.0 - normalized) * 42).astype(np.uint8)
    rgb[..., 2] = ((1.0 - normalized) * 52).astype(np.uint8)
    return rgb


def surface_from_array(arr: np.ndarray, size: tuple[int, int]) -> pygame.Surface:
    surface = pygame.image.frombuffer(arr.tobytes(), (arr.shape[1], arr.shape[0]), "RGB")
    surface = surface.convert()
    return pygame.transform.scale(surface, size)


def draw_text_wrapped(
    screen: pygame.Surface,
    font: pygame.font.Font,
    text: str,
    color: tuple[int, int, int],
    x: int,
    y: int,
    width: int,
    line_height: int,
) -> int:
    words = text.split()
    line = ""
    for word in words:
        candidate = word if not line else f"{line} {word}"
        if font.size(candidate)[0] <= width:
            line = candidate
        else:
            if line:
                screen.blit(font.render(line, True, color), (x, y))
                y += line_height
            line = word
    if line:
        screen.blit(font.render(line, True, color), (x, y))
        y += line_height
    return y


def choose_file() -> Optional[Path]:
    if tk is None or filedialog is None:
        return None

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    selected = filedialog.askopenfilename(
        title="Select an image",
        filetypes=[
            ("Image files", "*.png *.jpg *.jpeg *.bmp *.gif"),
            ("All files", "*.*"),
        ],
    )
    root.destroy()
    return Path(selected) if selected else None


def draw_button(
    screen: pygame.Surface,
    font: pygame.font.Font,
    rect: pygame.Rect,
    label: str,
    mouse_pos: tuple[int, int],
) -> None:
    color = BUTTON_HOVER if rect.collidepoint(mouse_pos) else BUTTON
    pygame.draw.rect(screen, color, rect, border_radius=6)
    pygame.draw.rect(screen, (78, 88, 104), rect, width=1, border_radius=6)
    label_surface = font.render(label, True, TEXT)
    label_rect = label_surface.get_rect(center=rect.center)
    screen.blit(label_surface, label_rect)


def draw_result(
    screen: pygame.Surface,
    result: SketchResult,
    fonts: dict[str, pygame.font.Font],
    k: int,
) -> None:
    title_font = fonts["title"]
    body_font = fonts["body"]
    small_font = fonts["small"]

    image_size = (300, 300)
    x_positions = [50, 370, 690]
    labels = ["Input image", "Decoded reconstruction", "Absolute error"]
    arrays = [
        to_rgb_surface_data(result.original),
        to_rgb_surface_data(result.reconstructed),
        error_surface_data(result.error),
    ]

    for x, label, arr in zip(x_positions, labels, arrays):
        screen.blit(body_font.render(label, True, TEXT), (x, 132))
        screen.blit(surface_from_array(arr, image_size), (x, 165))

    y = 500
    screen.blit(
        title_font.render("QOS-inspired sketch circuit", True, TEXT),
        (50, y),
    )
    y += 42
    details = [
        f"Image: {result.path.name}",
        f"Working resolution: {IMAGE_SIZE}x{IMAGE_SIZE} grayscale",
        f"Random sampled pixels: {result.k} stored in {result.padded_slots} index slots",
        f"Index qubits: {result.index_qubits}",
        "Value qubit: 1",
        f"Circuit width: {result.circuit.num_qubits} qubits",
        f"Circuit depth: {result.circuit.depth()}",
        f"MAE: {result.mae:.4f}",
        f"MSE: {result.mse:.4f}",
    ]
    for detail in details:
        screen.blit(body_font.render(detail, True, MUTED), (50, y))
        y += 27

    explanation = (
        "This is not the full Quantum Oracle Sketching paper implementation. "
        "It combines the useful QOS idea with an FRQI-style value register: each "
        "encoding pass randomly samples a compact set of pixels instead of loading "
        "the full image, puts sampled pixel IDs in an index register, encodes each "
        "pixel intensity into one value qubit with controlled RY rotations, keeps "
        "true image coordinates as classical oracle metadata, then decodes the "
        "compact sketch."
    )
    draw_text_wrapped(screen, small_font, explanation, TEXT, 690, 510, 470, 22)

    coeff_preview = ", ".join([f"({y},{x})" for y, x in result.indices[:8]])
    draw_text_wrapped(
        screen,
        small_font,
        f"Random sampled pixel positions: {coeff_preview}",
        MUTED,
        690,
        625,
        470,
        22,
    )

    controls = "Controls: click Select Image, 1/2/3 load sample images, +/- changes k, PageUp/PageDown jumps k by 16, R reruns, Esc exits."
    draw_text_wrapped(screen, small_font, controls, MUTED, 50, 730, 1080, 22)


def main() -> None:
    pygame.init()
    pygame.display.set_caption("Sparse FRQI / QOS-inspired image sketching")
    screen = pygame.display.set_mode(WINDOW_SIZE)
    clock = pygame.time.Clock()
    fonts = {
        "title": pygame.font.SysFont("Segoe UI", 26, bold=True),
        "body": pygame.font.SysFont("Segoe UI", 18),
        "small": pygame.font.SysFont("Segoe UI", 15),
    }

    available_defaults = [path for path in DEFAULT_IMAGES if path.exists()]
    current_path = available_defaults[0] if available_defaults else None
    k = DEFAULT_K
    result: Optional[SketchResult] = None
    status = "Select an image to create a random-pixel FRQI / QOS-inspired sketch."

    if current_path is not None:
        result = qos_inspired_encode_decode(current_path, k)
        status = f"Loaded sample: {current_path.name}"

    select_button = pygame.Rect(50, 70, 160, 42)
    rerun_button = pygame.Rect(225, 70, 120, 42)
    running = True

    while running:
        mouse_pos = pygame.mouse.get_pos()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if select_button.collidepoint(event.pos):
                    selected = choose_file()
                    if selected is not None:
                        try:
                            current_path = selected
                            result = qos_inspired_encode_decode(current_path, k)
                            status = f"Loaded: {current_path.name}"
                        except Exception as exc:
                            status = f"Could not load image: {exc}"
                elif rerun_button.collidepoint(event.pos) and current_path is not None:
                    result = qos_inspired_encode_decode(current_path, k)
                    status = f"Recomputed with k={k}"
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key in (pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS):
                    k = min(MAX_K, k + 1)
                    if current_path is not None:
                        result = qos_inspired_encode_decode(current_path, k)
                        status = f"Recomputed with k={k}"
                elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    k = max(2, k - 1)
                    if current_path is not None:
                        result = qos_inspired_encode_decode(current_path, k)
                        status = f"Recomputed with k={k}"
                elif event.key == pygame.K_PAGEUP:
                    k = min(MAX_K, k + 16)
                    if current_path is not None:
                        result = qos_inspired_encode_decode(current_path, k)
                        status = f"Recomputed with k={k}"
                elif event.key == pygame.K_PAGEDOWN:
                    k = max(2, k - 16)
                    if current_path is not None:
                        result = qos_inspired_encode_decode(current_path, k)
                        status = f"Recomputed with k={k}"
                elif event.key == pygame.K_r and current_path is not None:
                    result = qos_inspired_encode_decode(current_path, k)
                    status = f"Recomputed with k={k}"
                elif event.key in (pygame.K_1, pygame.K_2, pygame.K_3):
                    sample_index = event.key - pygame.K_1
                    if sample_index < len(available_defaults):
                        current_path = available_defaults[sample_index]
                        result = qos_inspired_encode_decode(current_path, k)
                        status = f"Loaded sample: {current_path.name}"

        screen.fill(PANEL_BG)
        screen.blit(
            fonts["title"].render(
                "Sparse FRQI / QOS-inspired image encoding and decoding", True, TEXT
            ),
            (50, 28),
        )
        draw_button(screen, fonts["body"], select_button, "Select Image", mouse_pos)
        draw_button(screen, fonts["body"], rerun_button, "Rerun", mouse_pos)
        screen.blit(fonts["body"].render(f"k = {k}", True, ACCENT), (370, 80))
        screen.blit(fonts["small"].render(status, True, MUTED), (430, 84))

        if result is not None:
            draw_result(screen, result, fonts, k)
        else:
            draw_text_wrapped(
                screen,
                fonts["body"],
                "No image loaded. Click Select Image or place the sample challenge images in the expected folders.",
                TEXT,
                50,
                145,
                850,
                28,
            )

        pygame.display.flip()
        clock.tick(30)

    pygame.quit()


if __name__ == "__main__":
    main()
