"""Run from the repository root: python -m scripts.train_enhancement --help."""
import argparse
from src.common import read_config, write_json
from src.enhancement.training import TrainConfig, train
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description='Train the zero-reference enhancement network')
    parser.add_argument('--config', default='configs/enhancement_train.json')
    parser.add_argument('--images', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()
    values = read_config(args.config)
    values.update(lowlight_images_path=args.images, snapshots_folder=args.output)
    if args.cpu:
        values['cpu'] = True
    cfg = TrainConfig(**values)
    write_json(Path(args.output)/'config.json', values)
    train(cfg)


if __name__ == '__main__':
    main()
