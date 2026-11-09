import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image
import torch
from src.classification.core import labeled_images
from scripts import train_classifier, evaluate_classifier


class DatasetSplitTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.images = self.root / 'images'
        for label in ['excessive', 'insufficient', 'normal']:
            for index in range(2):
                path = self.images / label / f'{label}_{index}.png'
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.new('RGB', (256, 256), (40, 80, 120)).save(path)
        self.split = self.root / 'split.csv'
        self.rows = [{'image_id': f'{label}_{index}', 'original_subset': 'train' if index == 0 else 'val'}
                     for label in ['excessive', 'insufficient', 'normal'] for index in range(2)]
        self.write_split(self.rows)

    def write_split(self, rows):
        with self.split.open('w', encoding='utf-8-sig', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=['image_id', 'original_subset'])
            writer.writeheader()
            writer.writerows(rows)

    @staticmethod
    def tiny_model(*args, **kwargs):
        return torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(), torch.nn.Linear(3, 3))

    def test_lowercase_and_historical_class_order(self):
        current = labeled_images(self.images)
        self.assertEqual(current.targets, [0, 0, 1, 1, 2, 2])
        for path in list(self.images.iterdir()):
            path.rename(path.with_name(path.name.capitalize()))
        historical = labeled_images(self.images)
        self.assertEqual(historical.targets, current.targets)

    def test_subset_keeps_samples_targets_and_transforms_aligned(self):
        dataset = labeled_images(self.images, split_file=self.split, subset='val')
        self.assertEqual(len(dataset), 3)
        self.assertEqual(dataset.targets, [0, 1, 2])
        self.assertIs(dataset.imgs, dataset.samples)
        for index in range(3):
            image, label = dataset[index]
            self.assertEqual(tuple(image.shape), (3, 224, 224))
            self.assertEqual(label, index)
            self.assertTrue(Path(dataset.samples[index][0]).stem.endswith('_1'))

    def test_missing_or_unknown_subset_is_reported(self):
        with self.assertRaisesRegex(ValueError, 'No images'):
            labeled_images(self.images, split_file=self.split, subset='other')
        self.write_split([{'image_id': 'missing', 'original_subset': 'val'}])
        with self.assertRaisesRegex(ValueError, 'missing from'):
            labeled_images(self.images, split_file=self.split, subset='val')

    def test_duplicate_ids_and_unpaired_options_are_reported(self):
        self.write_split([self.rows[0], self.rows[0]])
        with self.assertRaisesRegex(ValueError, 'unique'):
            labeled_images(self.images, split_file=self.split, subset='train')
        with self.assertRaisesRegex(ValueError, 'both'):
            labeled_images(self.images, split_file=self.split)
        with self.assertRaisesRegex(ValueError, 'both'):
            labeled_images(self.images, subset='train')

    def test_training_shared_directory_and_evaluation_select_csv_subsets(self):
        config = self.root / 'config.json'
        config.write_text(json.dumps(dict(epochs=1, batch_size=3, lr=0.01, momentum=0.9,
            step_size=7, lr_gamma=0.1, preprocessing='center_crop', num_workers=0, seed=42)))
        output = self.root / 'training'
        argv = ['train_classifier', '--train', str(self.images), '--val', str(self.images),
                '--split-file', str(self.split), '--train-subset', 'train', '--val-subset', 'val',
                '--config', str(config), '--output', str(output), '--init', 'random', '--device', 'cpu']
        with patch.object(sys, 'argv', argv), patch.object(train_classifier, 'create_classifier', self.tiny_model):
            train_classifier.main()
        self.assertTrue((output / 'best.pth').exists())
        metrics_dir = self.root / 'evaluation'
        argv = ['evaluate_classifier', '--images', str(self.images), '--split-file', str(self.split),
                '--subset', 'val', '--weights', str(output / 'best.pth'), '--output', str(metrics_dir),
                '--preprocessing', 'center_crop', '--device', 'cpu']
        with patch.object(sys, 'argv', argv), patch.object(evaluate_classifier, 'create_classifier', self.tiny_model):
            evaluate_classifier.main()
        metrics = json.loads((metrics_dir / 'metrics.json').read_text())
        self.assertEqual(metrics['support'], [1, 1, 1])
        self.assertEqual(metrics['subset'], 'val')
        with (metrics_dir / 'predictions.csv').open() as stream:
            predictions = list(csv.DictReader(stream))
        self.assertEqual(len(predictions), 3)
        self.assertTrue(all(Path(row['path']).stem.endswith('_1') for row in predictions))
        argv = ['train_classifier', '--train', str(self.images), '--val', str(self.images),
                '--split-file', str(self.split), '--train-subset', 'train', '--val-subset', 'train',
                '--config', str(config), '--output', str(output), '--init', 'random', '--device', 'cpu']
        with patch.object(sys, 'argv', argv), self.assertRaisesRegex(ValueError, 'overlap'):
            train_classifier.main()


if __name__ == '__main__':
    unittest.main()
