import queue
import threading
import time
import numpy as np

class VideoCapture:
    def __init__(self, cap, name=None):
        if name is None:
            name = cap.name
        self.name = name
        self.q = queue.Queue(maxsize=1)
        self.cap = cap
        self.t = threading.Thread(target=self._reader)
        self.t.daemon = False
        self.enable = True
        self._consecutive_failures = 0
        self.t.start()

    def _reader(self):
        while self.enable:
            try:
                ret, frame = self.cap.read()
            except Exception:
                ret, frame = False, None
            if (not ret) or (frame is None):
                # 单次取帧失败不退出线程，避免相机短暂超时后直接断流。
                self._consecutive_failures += 1
                time.sleep(0.05)
                continue
            self._consecutive_failures = 0
            if self.q.full():
                try:
                    self.q.get_nowait()  # discard previous (unprocessed) frame
                except queue.Empty:
                    pass
            self.q.put(frame)

    def read(self):
        # print(self.name, self.q.qsize())
        return self.q.get(timeout=10)

    def close(self):
        self.enable = False
        self.t.join(timeout=2.0)
        self.cap.close()
