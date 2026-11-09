"""Video FPS, preview states and early-exit checks using a tiny real video."""
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import cv2
import numpy as np
import torch
from scripts import enhance_video as demo
from src.enhancement.gamma import GammaConfig
from src.enhancement.stream import AsyncEnhancer

ROOT = Path(__file__).resolve().parents[1]


class VideoDemoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.output = ROOT/'outputs/video_demo_tests'
        cls.output.mkdir(parents=True, exist_ok=True)
        cls.source = cls.output/'input.mp4'
        writer = cv2.VideoWriter(str(cls.source), cv2.VideoWriter_fourcc(*'mp4v'), 20, (64, 64))
        if not writer.isOpened():
            raise RuntimeError('MP4 encoder unavailable')
        for i in range(8):
            writer.write(np.full((64, 64, 3), 30+i, dtype=np.uint8))
        writer.release()
        cls.config = cls.output/'config.json'
        cls.config.write_text(json.dumps({'fps': 90, 'window_seconds': 0.2, 'samples': 1, 'image_size': 32}), encoding='utf-8')

    def run_case(self, name, extra, ui_key=None, visible=1):
        output = self.output/(name+'.mp4')
        engines = []
        def factory(*args):
            engine = AsyncEnhancer(*args)
            engine.close = Mock(wraps=engine.close)
            engines.append(engine)
            return engine
        args = ['--source', str(self.source), '--output', str(output), '--weights', str(ROOT/'weights/enhancement_realtime.pth'),
                '--config', str(self.config), '--device', 'cpu', *extra]
        with patch.object(demo, 'AsyncEnhancer', side_effect=factory), \
             patch.object(demo.cv2, 'imshow'), patch.object(demo.cv2, 'destroyAllWindows') as destroy, \
             patch.object(demo.cv2, 'getWindowProperty', side_effect=visible if isinstance(visible, list) else lambda *_: visible), \
             patch.object(demo.cv2, 'waitKey', side_effect=ui_key if ui_key is not None else lambda _: 0):
            demo.main(args)
            if '--show' in extra:
                destroy.assert_called_once()
        engines[0].close.assert_called_once()
        result = json.loads(output.with_suffix('.json').read_text(encoding='utf-8'))
        cap = cv2.VideoCapture(str(output))
        frames = []
        fps = cap.get(cv2.CAP_PROP_FPS)
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
        cap.release()
        self.assertEqual(len(frames), result['frames'])
        self.assertAlmostEqual(fps, result['effective_fps'])
        return result, frames

    def test_realtime_uses_source_fps_and_comparison_size(self):
        result, frames = self.run_case('paced', ['--realtime', '--side-by-side'])
        self.assertEqual(result['frames'], 8)
        self.assertEqual(result['source_fps'], 20)
        self.assertEqual(result['config']['fps'], 20)
        self.assertGreaterEqual(result['elapsed_seconds'], 7/20-0.01)
        self.assertEqual(frames[0].shape, (32, 64, 3))
        self.assertEqual(result['exit_reason'], 'end_of_stream')

    def test_offline_retains_configured_fps_and_clean_size(self):
        result, frames = self.run_case('offline', [])
        self.assertEqual(result['effective_fps'], 90)
        self.assertFalse(result['realtime'])
        self.assertEqual(frames[0].shape, (64, 64, 3))

    def test_explicit_fps_override(self):
        result, _ = self.run_case('override', ['--realtime', '--fps', '40'])
        self.assertEqual(result['effective_fps'], 40)
        self.assertEqual(result['config']['fps'], 40)

    def test_q_and_escape_close_resources(self):
        for key, reason in [(ord('q'), 'q'), (27, 'escape')]:
            result, _ = self.run_case('exit_'+reason, ['--show'], ui_key=lambda _: key)
            self.assertEqual(result['frames'], 1)
            self.assertEqual(result['exit_reason'], reason)

    def test_q_during_frame_wait(self):
        # First event is after showing frame one; second occurs while waiting for frame two.
        result, _ = self.run_case('exit_while_waiting', ['--realtime', '--show'], ui_key=[0, ord('q')])
        self.assertEqual(result['frames'], 1)
        self.assertEqual(result['exit_reason'], 'q')

    def test_window_close(self):
        result, _ = self.run_case('window_close', ['--show'], visible=0)
        self.assertEqual(result['frames'], 1)
        self.assertEqual(result['exit_reason'], 'window_closed')

    def test_window_close_during_frame_wait(self):
        result, _ = self.run_case('close_while_waiting', ['--realtime', '--show'], visible=[1, 0])
        self.assertEqual(result['frames'], 1)
        self.assertEqual(result['exit_reason'], 'window_closed')

    def test_waiting_and_active_states_even_if_gamma_is_one(self):
        image = np.zeros((256, 320, 3), dtype=np.uint8)
        engine = SimpleNamespace(completed=[], frame_index=1, window_size=4,
                                 config=GammaConfig(fps=2, window_seconds=2), gamma=1.0, skipped_windows=0)
        for index, complete, expected in [(1, [], 'Waiting: first window'), (4, [], 'Waiting: first optimization'),
                                           (5, [{'gamma': 1.0}], 'Active | updates=1')]:
            engine.frame_index = index
            engine.completed = complete
            with patch.object(demo.cv2, 'putText', wraps=cv2.putText) as draw:
                rendered = demo.comparison_frame(image, image, engine, 0.5)
                self.assertEqual(rendered.shape, (128, 320, 3))
                self.assertTrue(any(expected in call.args[1] for call in draw.call_args_list))


if __name__ == '__main__':
    unittest.main()
