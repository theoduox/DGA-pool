import argparse
import csv
from pathlib import Path
import random
import torch
from torch.utils.data import DataLoader
from src.common import get_device, read_config, write_json
from src.classification.core import create_classifier, labeled_images


def main():
    parser = argparse.ArgumentParser(description='Train one independent classifier per image treatment')
    parser.add_argument('--config', default='configs/classification_historical.json')
    parser.add_argument('--train', required=True, help='Existing labeled training directory')
    parser.add_argument('--val', required=True, help='Existing labeled validation directory')
    parser.add_argument('--split-file', help='Optional CSV shared by training and validation')
    parser.add_argument('--train-subset', help='Original training subset name from the CSV')
    parser.add_argument('--val-subset', help='Original validation subset name from the CSV')
    parser.add_argument('--output', required=True)
    parser.add_argument('--init', choices=['random', 'imagenet'], required=True)
    parser.add_argument('--device', default='auto')
    args = parser.parse_args()
    if args.split_file is None and (args.train_subset is not None or args.val_subset is not None):
        raise ValueError('Subset selection requires --split-file')
    if args.split_file is not None and (args.train_subset is None or args.val_subset is None):
        raise ValueError('Provide --train-subset and --val-subset with --split-file')
    if args.split_file is None and Path(args.train).resolve() == Path(args.val).resolve():
        raise ValueError('Training and validation directories must differ')
    cfg = read_config(args.config)
    random.seed(cfg['seed'])
    torch.manual_seed(cfg['seed'])
    device = get_device(args.device)
    datasets = {'train': labeled_images(args.train, True, split_file=args.split_file, subset=args.train_subset),
                'val': labeled_images(args.val, mode=cfg['preprocessing'],
                                      split_file=args.split_file, subset=args.val_subset)}
    if args.split_file is not None:
        train_ids = {Path(path).stem for path, _ in datasets['train'].samples}
        val_ids = {Path(path).stem for path, _ in datasets['val'].samples}
        if train_ids & val_ids:
            raise ValueError('Training and validation subsets overlap')
    loaders = {phase: DataLoader(ds, batch_size=cfg['batch_size'], shuffle=phase=='train',
                                num_workers=cfg['num_workers']) for phase, ds in datasets.items()}
    model = create_classifier(pretrained=args.init=='imagenet').to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg['lr'], momentum=cfg['momentum'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg['step_size'], gamma=cfg['lr_gamma'])
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output/'config.json', {**cfg, 'initialization': args.init, 'train': args.train,
                                     'val': args.val, 'classes': datasets['train'].classes,
                                     'split_file': args.split_file, 'train_subset': args.train_subset,
                                     'val_subset': args.val_subset})
    criterion = torch.nn.CrossEntropyLoss()
    best = -1.0
    with (output/'training.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['epoch', 'phase', 'loss', 'accuracy', 'lr'])
        for epoch in range(1, cfg['epochs']+1):
            for phase in ['train', 'val']:
                model.train(phase=='train')
                total_loss, correct = 0.0, 0
                for images, labels in loaders[phase]:
                    images, labels = images.to(device), labels.to(device)
                    optimizer.zero_grad(set_to_none=True)
                    with torch.set_grad_enabled(phase=='train'):
                        logits = model(images)
                        loss = criterion(logits, labels)
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
                    total_loss += loss.item()*len(labels)
                    correct += (logits.argmax(1)==labels).sum().item()
                accuracy = correct/len(datasets[phase])
                writer.writerow([epoch, phase, total_loss/len(datasets[phase]), accuracy, optimizer.param_groups[0]['lr']])
                stream.flush()
                print(f'{epoch} {phase}: loss={total_loss/len(datasets[phase]):.6f} accuracy={accuracy:.6f}')
                if phase == 'val' and accuracy > best:
                    best = accuracy
                    torch.save(model.state_dict(), output/'best.pth')
            scheduler.step()
        torch.save(model.state_dict(), output/'last.pth')


if __name__ == '__main__':
    main()
