"""One pending window, background reference/search, latest completed Gamma."""
from concurrent.futures import ThreadPoolExecutor
from .gamma import gamma_correction, optimize_window


class AsyncEnhancer:
    def __init__(self, model, device, config):
        self.model, self.device, self.config = model, device, config
        self.gamma = 1.0
        self.previous = None
        self.frames = []
        self.window_size = int(config.window_seconds * config.fps)
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.future = None
        self.completed = []
        self.frame_index = 0
        self.skipped_windows = 0

    def process(self, frame):
        self.frame_index += 1
        if self.future is not None and self.future.done():
            self.gamma, score = self.future.result()
            self.previous = self.gamma
            self.completed.append({'applied_from_frame': self.frame_index, 'gamma': self.gamma, 'mse': score})
            self.future = None
        self.frames.append(frame.copy())
        if len(self.frames) >= self.window_size:
            if self.future is None:
                self.future = self.executor.submit(optimize_window, self.frames, self.model,
                                                   self.device, self.config, self.previous)
            else:
                self.skipped_windows += 1
            self.frames = []
        return gamma_correction(frame, self.gamma)

    def close(self):
        self.executor.shutdown(wait=True)
        # Propagate worker failures even if the video ended before the next poll.
        if self.future is not None:
            self.future.result()
