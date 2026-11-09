"""Small shared helpers for the command-line entry points."""
import json
from pathlib import Path
import torch

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}


def read_config(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def get_device(name='auto'):
    return torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if name == 'auto' else name)


def load_weights(model, path, device):
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True), strict=True)
    return model.to(device).eval()


def image_paths(path):
    path = Path(path)
    paths = [path] if path.is_file() else sorted(p for p in path.rglob('*') if p.suffix.lower() in IMAGE_EXTENSIONS)
    if not paths:
        raise ValueError(f'No images found: {path}')
    return paths
