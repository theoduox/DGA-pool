import argparse
from pathlib import Path
import cv2
import numpy as np
from src.common import image_paths, read_config, write_json
from src.preprocessing.roi import ROIConfig, NoRegionError, extract_roi


def main():
    parser = argparse.ArgumentParser(description='Extract contour/HSV melt-pool ROIs from an image or directory')
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--config', default='configs/roi_historical.json')
    args = parser.parse_args()
    source, target = Path(args.input), Path(args.output)
    if source.resolve() == target.resolve() or source.resolve() in target.resolve().parents:
        raise ValueError('Output must be outside the input directory')
    config = ROIConfig(**read_config(args.config))
    records = []
    for path in image_paths(source):
        image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f'Cannot read image: {path}')
        try:
            roi, info = extract_roi(image, config)
        except NoRegionError as error:
            records.append({'input': str(path), 'status': 'no_roi', 'reason': str(error)})
            continue
        relative = path.relative_to(source) if source.is_dir() else Path(path.name)
        output = target / relative.with_name(relative.name + '.png')
        output.parent.mkdir(parents=True, exist_ok=True)
        ok, encoded = cv2.imencode('.png', roi)
        if not ok:
            raise RuntimeError(f'Cannot encode: {output}')
        encoded.tofile(output)
        records.append({'input': str(path), 'output': str(output), 'status': 'ok', **info})
    write_json(target/'roi.json', {'config': vars(config), 'images': records})
    successes = sum(record['status']=='ok' for record in records)
    print(f'ROI: {successes}/{len(records)} images; no contour: {len(records)-successes}')


if __name__ == '__main__':
    main()
