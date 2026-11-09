import argparse
from pathlib import Path
import cv2
import numpy as np
from src.common import get_device, load_weights, image_paths, read_config, write_json
from src.enhancement.model import EnhanceNetLight
from src.enhancement.gamma import GammaConfig, reference_images, optimize_window, gamma_correction


def main():
    parser = argparse.ArgumentParser(description='Enhance images independently; reference or full Gamma path')
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--weights', required=True)
    parser.add_argument('--mode', choices=['gamma', 'reference'], default='gamma')
    parser.add_argument('--config', default='configs/gamma_realtime.json')
    parser.add_argument('--device', default='auto')
    args = parser.parse_args()
    device = get_device(args.device)
    model = load_weights(EnhanceNetLight(), args.weights, device)
    config = GammaConfig(**read_config(args.config))
    source, target = Path(args.input), Path(args.output)
    if source.resolve() == target.resolve() or source.resolve() in target.resolve().parents:
        raise ValueError('Output must be outside the input directory')
    paths = image_paths(source)
    records = []
    for path in paths:
        image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f'Cannot read image: {path}')
        if args.mode == 'gamma':
            gamma, mse = optimize_window([image], model, device, config)
            result = gamma_correction(image, gamma)
        else:
            result = reference_images([image], model, device, config.image_size)[0]
            gamma, mse = None, None
        relative = path.relative_to(source) if source.is_dir() else Path(path.name)
        output = target / relative.with_name(relative.name + '.png')
        output.parent.mkdir(parents=True, exist_ok=True)
        ok, encoded = cv2.imencode('.png', result)
        if not ok:
            raise RuntimeError(f'Cannot encode: {output}')
        encoded.tofile(output)
        records.append({'input': str(path), 'output': str(output), 'gamma': gamma, 'mse': mse})
    write_json(target/'enhancement.json', {'mode': args.mode, 'weights': args.weights, 'config': vars(config), 'images': records})


if __name__ == '__main__':
    main()
