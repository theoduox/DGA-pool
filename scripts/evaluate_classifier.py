import argparse
import csv
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from src.common import get_device, load_weights, write_json
from src.classification.core import create_classifier, labeled_images, classification_metrics, CLASSES


def main():
    parser = argparse.ArgumentParser(description='Evaluate one fixed classifier on an existing labeled directory')
    parser.add_argument('--images', required=True)
    parser.add_argument('--split-file', help='Optional CSV with image_id and original_subset')
    parser.add_argument('--subset', help='Original subset name from the split CSV')
    parser.add_argument('--weights', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--preprocessing', choices=['center_crop', 'resize'], required=True)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--device', default='auto')
    args = parser.parse_args()
    device = get_device(args.device)
    model = load_weights(create_classifier(), args.weights, device)
    dataset = labeled_images(args.images, mode=args.preprocessing,
                             split_file=args.split_file, subset=args.subset)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    labels_all, predictions = [], []
    with (output/'predictions.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['path', 'label', 'prediction']+['p_'+name for name in CLASSES])
        offset = 0
        with torch.inference_mode():
            for images, labels in loader:
                probabilities = model(images.to(device)).softmax(1).cpu()
                predicted = probabilities.argmax(1).tolist()
                for i, (label, prediction, probs) in enumerate(zip(labels.tolist(), predicted, probabilities.tolist())):
                    writer.writerow([dataset.samples[offset+i][0], label, prediction]+probs)
                labels_all.extend(labels.tolist())
                predictions.extend(predicted)
                offset += len(labels)
    metrics = classification_metrics(labels_all, predictions)
    metrics.update(weights=args.weights, images=args.images, preprocessing=args.preprocessing)
    if args.split_file is not None:
        metrics.update(split_file=args.split_file, subset=args.subset)
    write_json(output/'metrics.json', metrics)
    print(f"accuracy={metrics['accuracy']:.6f} macro_f1={metrics['macro_f1']:.6f}")


if __name__ == '__main__':
    main()
