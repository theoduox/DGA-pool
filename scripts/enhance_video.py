"""Enhance a file/camera stream, optionally replaying a file at its source FPS."""
import argparse
from dataclasses import replace
import math
import sys
from pathlib import Path
import time
import cv2
from src.common import get_device, load_weights, read_config, write_json
from src.enhancement.model import EnhanceNetLight
from src.enhancement.gamma import GammaConfig
from src.enhancement.stream import AsyncEnhancer

WINDOW = 'Original | DGA-Pool'


def positive_float(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError('must be a finite positive number')
    return number


def exit_key(key):
    key &= 0xff
    return 'q' if key == ord('q') else 'escape' if key == 27 else None


def window_closed():
    try:
        return cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1
    except cv2.error:
        return True


def wait_for_frame(deadline, show):
    """Keep Q/Esc responsive while a file waits for its next frame deadline."""
    while True:
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            return None
        if show:
            reason = exit_key(cv2.waitKey(max(1, min(20, int(remaining * 1000)))))
            if reason:
                return reason
            if window_closed():
                return 'window_closed'
        else:
            time.sleep(min(remaining, 0.02))


def comparison_frame(original, enhanced, engine, scale):
    width = max(2, round(original.shape[1] * scale) // 2 * 2)
    height = max(2, round(original.shape[0] * scale) // 2 * 2)
    left = cv2.resize(original, (width, height))
    right = cv2.resize(enhanced, (width, height))
    font = max(0.3, min(0.62, width / 1100))
    line_height = max(16, round(38 * font))
    if engine.completed:
        status = f'Active | updates={len(engine.completed)} | skipped={engine.skipped_windows}'
    elif engine.frame_index < engine.window_size:
        status = f'Waiting: first window ({engine.config.window_seconds:g} s)'
    else:
        status = 'Waiting: first optimization'
    source_time = (engine.frame_index - 1) / engine.config.fps
    for panel, lines in [(left, ['Original', f'Video time: {source_time:.2f} s']),
                         (right, [f'Enhanced | gamma={engine.gamma:.2f}', status,
                                  f'Frame {engine.frame_index} | {engine.config.fps:g} fps'])]:
        for i, text in enumerate(lines):
            point = (10, line_height * (i + 1))
            cv2.putText(panel, text, point, cv2.FONT_HERSHEY_SIMPLEX, font, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(panel, text, point, cv2.FONT_HERSHEY_SIMPLEX, font, (255, 255, 255), 1, cv2.LINE_AA)
    return cv2.hconcat([left, right])


def main(argv=None):
    parser = argparse.ArgumentParser(description='Asynchronous Gamma enhancement for a video or camera')
    parser.add_argument('--source', required=True, help='Video filename or camera index')
    parser.add_argument('--output', required=True, help='Output .mp4 file')
    parser.add_argument('--weights', required=True)
    parser.add_argument('--config', default='configs/gamma_realtime.json')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--show', action='store_true')
    parser.add_argument('--realtime', action='store_true', help='Pace a video file at its source FPS')
    parser.add_argument('--fps', type=positive_float, help='Override pacing, window and output FPS')
    parser.add_argument('--side-by-side', action='store_true', help='Save annotated comparison instead of clean full-size enhancement')
    parser.add_argument('--preview-scale', type=positive_float, default=0.5,
                        help='Scale each preview panel for display/comparison output (default: 0.5)')
    args = parser.parse_args(argv)
    camera = args.source.isdecimal()
    if args.realtime and camera:
        parser.error('--realtime is for video files; cameras already follow capture timing')
    output = Path(args.output)
    if output.suffix.lower() != '.mp4':
        parser.error('Output filename must end with .mp4')
    if output.resolve() == Path(args.source).resolve():
        parser.error('Output must differ from source')
    config = GammaConfig(**read_config(args.config))
    device = get_device(args.device)
    print(f'Python: {sys.executable}', flush=True)
    print(f'Loading enhancement model on {device}...', flush=True)
    model = load_weights(EnhanceNetLight(), args.weights, device)
    cap = cv2.VideoCapture(int(args.source) if camera else args.source)
    if not cap.isOpened():
        cap.release()
        raise ValueError(f'Cannot open source: {args.source}')
    source_fps = cap.get(cv2.CAP_PROP_FPS)
    valid_source_fps = math.isfinite(source_fps) and source_fps > 0
    effective_fps = args.fps or (source_fps if args.realtime and valid_source_fps else config.fps)
    if args.realtime and not args.fps and not valid_source_fps:
        print(f'Source FPS unavailable; using configured {config.fps:g} FPS')
    config = replace(config, fps=effective_fps)
    print(f'Video ready | {effective_fps:g} fps | realtime={args.realtime} | press Q/Esc in the preview to exit' if args.show else
          f'Video ready | {effective_fps:g} fps | realtime={args.realtime}', flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    enhancer = AsyncEnhancer(model, device, config)
    writer = None
    output_size = None
    reason = 'end_of_stream'
    start = time.perf_counter()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if args.realtime:
                requested_exit = wait_for_frame(start + enhancer.frame_index / effective_fps, args.show)
                if requested_exit:
                    reason = requested_exit
                    break
            result = enhancer.process(frame)
            preview = comparison_frame(frame, result, enhancer, args.preview_scale) if args.show or args.side_by_side else None
            saved_frame = preview if args.side_by_side else result
            if writer is None:
                output_size = [saved_frame.shape[1], saved_frame.shape[0]]
                writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*'mp4v'), effective_fps, tuple(output_size))
                if not writer.isOpened():
                    raise RuntimeError(f'Cannot create video: {output}')
            writer.write(saved_frame)
            if args.show:
                cv2.imshow(WINDOW, preview)
                requested_exit = exit_key(cv2.waitKey(1))
                if requested_exit:
                    reason = requested_exit
                    break
                if window_closed():
                    reason = 'window_closed'
                    break
    except KeyboardInterrupt:
        reason = 'interrupted'
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if args.show:
            cv2.destroyAllWindows()
        enhancer.close()
    if enhancer.frame_index == 0:
        raise ValueError('No video frames read')
    elapsed = time.perf_counter() - start
    write_json(output.with_suffix('.json'), {
        'weights': args.weights, 'source': args.source, 'config': vars(config),
        'frames': enhancer.frame_index, 'updates': enhancer.completed,
        'skipped_windows': enhancer.skipped_windows,
        'source_fps': source_fps if valid_source_fps else None,
        'effective_fps': effective_fps, 'realtime': args.realtime,
        'side_by_side': args.side_by_side, 'preview_scale': args.preview_scale,
        'output_size': output_size, 'exit_reason': reason, 'elapsed_seconds': elapsed})
    print(f'Wrote {enhancer.frame_index} frames | {effective_fps:g} fps | '
          f'updates={len(enhancer.completed)} | skipped={enhancer.skipped_windows} | exit={reason}')


if __name__ == '__main__':
    main()
