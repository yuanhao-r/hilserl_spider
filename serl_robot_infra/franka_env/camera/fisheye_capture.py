import numpy as np
import cv2


class FisheyeCapture:
    def __init__(self, name, dim=(640, 480), camera_index=2, exposure=None):
        self.name = name
        self.dim = dim
        self.camera_index = camera_index
        self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            print(f"Can NOT open fisheye camere, camera index: {camera_index}")
        else:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.dim[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.dim[1])
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            self.cap.set(cv2.CAP_PROP_FOURCC, fourcc)
            self.cap.set(cv2.CAP_PROP_FPS, 30)

            if exposure is not None:
                self.cap.set(cv2.CAP_PROP_EXPOSURE, exposure)
            # print(f"fisheye resolution: {self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)}x{self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")

    def read(self):
        ret, frame = self.cap.read()
        if ret:
            return ret, np.asarray(frame)
        else:
            return ret, None

    def close(self):
        self.cap.release()
