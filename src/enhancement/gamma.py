"""BGR uint8 reference generation, grayscale MSE search and Gamma LUT."""
from dataclasses import dataclass
import cv2
import numpy as np
import torch


@dataclass
class GammaConfig:
    global_min: float = 1.1
    global_max: float = 3.0  # inclusive
    step: float = 0.1
    local_radius: float = 0.5
    local_step: float = 0.1
    samples: int = 15
    image_size: int = 224
    window_seconds: float = 4.0
    fps: float = 90.0

    def __post_init__(self):
        if (self.global_min <= 0 or self.global_max < self.global_min or self.step <= 0
                or self.local_radius < 0 or self.local_step <= 0 or self.samples < 1
                or self.image_size < 1 or self.window_seconds <= 0 or self.fps <= 0
                or self.window_seconds * self.fps < 1):
            raise ValueError('Invalid Gamma/window configuration')


def gamma_grid(config, previous=None):
    if previous is None:
        low, high, step = config.global_min, config.global_max, config.step
    else:
        low, high, step = max(0.1, previous-config.local_radius), previous+config.local_radius, config.local_step
    return np.round(np.arange(low, high + step * 0.5, step), 6).tolist()


def gamma_correction(image, gamma):
    if gamma <= 0:
        raise ValueError('gamma must be positive')
    table = np.array([(i / 255.0) ** (1.0 / gamma) * 255 for i in range(256)], dtype=np.uint8)
    return cv2.LUT(image, table)


@torch.inference_mode()
def reference_images(images, model, device, image_size=224):
    if not images:
        raise ValueError('A window must contain images')
    tensors = []
    for image in images:
        rgb = cv2.cvtColor(cv2.resize(image, (image_size, image_size)), cv2.COLOR_BGR2RGB)
        tensors.append(torch.from_numpy(rgb.copy()).permute(2, 0, 1).float() / 255.0)
    _, enhanced, _ = model(torch.stack(tensors).to(device))
    arrays = enhanced.clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy()
    return [cv2.cvtColor((a * 255).astype(np.uint8), cv2.COLOR_RGB2BGR) for a in arrays]


def search_gamma(originals, references, grid):
    if not originals or len(originals) != len(references) or not len(grid):
        raise ValueError('Provide paired images and a nonempty Gamma grid')
    resized = [cv2.resize(image, (ref.shape[1], ref.shape[0])) for image, ref in zip(originals, references)]
    gray_refs = [cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY).astype(np.float64) for ref in references]
    scores = []
    for gamma in grid:
        scores.append(float(np.mean([
            np.mean((cv2.cvtColor(gamma_correction(image, gamma), cv2.COLOR_BGR2GRAY).astype(np.float64)-ref)**2)
            for image, ref in zip(resized, gray_refs)
        ])))
    best = int(np.argmin(scores))
    return float(grid[best]), scores[best]


def optimize_window(frames, model, device, config, previous=None):
    if not frames:
        raise ValueError('Empty window')
    indices = np.linspace(0, len(frames)-1, min(config.samples, len(frames)), dtype=int)
    samples = [frames[i] for i in indices]
    refs = reference_images(samples, model, device, config.image_size)
    return search_gamma(samples, refs, gamma_grid(config, previous))
