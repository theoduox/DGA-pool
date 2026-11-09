"""Run verified historical evaluations through the shared classifier entry point."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(description='Reproduce a known classification result using existing source files')
    parser.add_argument('--config', default='configs/classification_reproductions.json')
    parser.add_argument('--list', action='store_true', help='List cases without loading images or models')
    parser.add_argument('--case', help='Case ID, or all')
    parser.add_argument('--source-root', help='Root containing the historical data/ directory and weight files')
    parser.add_argument('--output-root', default='outputs/classification_reproductions')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--workers', type=int, default=0)
    args = parser.parse_args(argv)
    cases = json.loads(Path(args.config).read_text(encoding='utf-8-sig'))['cases']
    if args.list:
        for case in cases:
            print(f"{case['id']}: {case['images']} | {case['weights']} | {case['preprocessing']}")
        return
    if not args.source_root or not args.case:
        parser.error('--source-root and --case are required unless --list is used')
    selected = cases if args.case == 'all' else [case for case in cases if case['id'] == args.case]
    if not selected:
        parser.error(f'Unknown case: {args.case}')
    source = Path(args.source_root)
    for case in selected:
        images, weights = source/case['images'], source/case['weights']
        if not images.is_dir() or not weights.is_file():
            parser.error(f"Missing inputs for {case['id']}: {images}; {weights}")
    for case in selected:
        output = Path(args.output_root)/case['id']
        print(f"Running {case['id']}", flush=True)
        subprocess.run([sys.executable, '-m', 'scripts.evaluate_classifier',
                        '--images', str(source/case['images']), '--weights', str(source/case['weights']),
                        '--output', str(output), '--preprocessing', case['preprocessing'],
                        '--device', args.device, '--batch-size', str(args.batch_size),
                        '--workers', str(args.workers)], check=True)
        metrics = json.loads((output/'metrics.json').read_text(encoding='utf-8'))
        expected = case['expected']
        matched = all(metrics[key] == expected[key] for key in ['support', 'confusion_matrix'])
        matched = matched and all(f'{metrics[key]:.4f}' == f'{expected[key]:.4f}'
                                  for key in ['accuracy', 'macro_f1'])
        comparison = {'case': case['id'], 'matches_reference': matched, 'expected': expected,
                      'actual': {key: metrics[key] for key in expected}}
        (output/'reference_comparison.json').write_text(json.dumps(comparison, indent=2), encoding='utf-8')
        if not matched:
            raise RuntimeError(f"{case['id']} differs from the reference; see {output/'reference_comparison.json'}")
        print(f"{case['id']}: matches the reference", flush=True)


if __name__ == '__main__':
    main()
