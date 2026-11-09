"""ConvNeXt-Tiny, historical transforms and three-class metrics."""
import csv
from pathlib import Path
import numpy as np
import timm
from torchvision import transforms, datasets

CLASSES = ['Excessive', 'Insufficient', 'Normal']


def create_classifier(pretrained=False):
    return timm.create_model('convnext_tiny', pretrained=pretrained, num_classes=3)


def preprocessing(training=False, mode='center_crop'):
    if mode not in ('center_crop', 'resize'):
        raise ValueError(f'Unknown preprocessing: {mode}')
    operations = [transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip()] if training else (
        [transforms.Resize(256), transforms.CenterCrop(224)] if mode == 'center_crop' else [transforms.Resize((224, 224))])
    return transforms.Compose(operations + [transforms.ToTensor(), transforms.Normalize(
        [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


def labeled_images(path, training=False, mode='center_crop', split_file=None, subset=None):
    dataset = datasets.ImageFolder(path, preprocessing(training, mode))
    if [name.lower() for name in dataset.classes] != [name.lower() for name in CLASSES]:
        raise ValueError(f'Expected folders {CLASSES}; found {dataset.classes}')
    if (split_file is None) != (subset is None):
        raise ValueError('Provide both split_file and subset')
    if split_file is not None:
        with open(split_file, encoding='utf-8-sig', newline='') as stream:
            reader = csv.DictReader(stream)
            if not {'image_id', 'original_subset'} <= set(reader.fieldnames or []):
                raise ValueError('Split CSV requires image_id and original_subset columns')
            rows = list(reader)
        ids = [row['image_id'] for row in rows]
        if len(ids) != len(set(ids)) or any(not value for value in ids):
            raise ValueError('Split CSV image_id values must be nonempty and unique')
        wanted = {row['image_id'] for row in rows if row['original_subset'] == subset}
        if not wanted:
            raise ValueError(f'No images in subset {subset!r}')
        samples = [sample for sample in dataset.samples if Path(sample[0]).stem in wanted]
        found = {Path(sample[0]).stem for sample in samples}
        if found != wanted:
            raise ValueError(f'{len(wanted - found)} subset images are missing from {path}')
        if len(samples) != len(found):
            raise ValueError('Selected image IDs occur in multiple class folders')
        dataset.samples = samples
        dataset.imgs = samples
        dataset.targets = [label for _, label in samples]
    return dataset


def classification_metrics(labels, predictions):
    matrix = np.zeros((3, 3), dtype=np.int64)
    for label, prediction in zip(labels, predictions):
        matrix[label, prediction] += 1
    tp = matrix.diagonal().astype(float)
    precision = np.divide(tp, matrix.sum(0), out=np.zeros(3), where=matrix.sum(0)!=0)
    recall = np.divide(tp, matrix.sum(1), out=np.zeros(3), where=matrix.sum(1)!=0)
    f1 = np.divide(2*precision*recall, precision+recall, out=np.zeros(3), where=(precision+recall)!=0)
    return {'classes': CLASSES, 'confusion_matrix': matrix.tolist(),
            'accuracy': float(tp.sum()/matrix.sum()) if matrix.sum() else 0.0,
            'precision': precision.tolist(), 'recall': recall.tolist(), 'f1': f1.tolist(),
            'support': matrix.sum(1).tolist(), 'macro_f1': float(f1.mean())}
