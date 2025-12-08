import cv2
import numpy as np

class VideoBatchLoader:
    def __init__(self, video_path, batch_size=8):
        """
        Initializes the video loader.
        :param video_path: Path to the input video file.
        :param batch_size: Number of frames per batch.
        """
        self.cap = cv2.VideoCapture(video_path)
        self.batch_size = batch_size
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def get_batches(self):
        """
        Generator that yields batches of frames.
        """
        batch = []
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            
            if not ret:
                # If there are remaining frames in the buffer, yield them
                if len(batch) > 0:
                    yield batch
                break

            batch.append(frame)

            # When batch is full, yield it and reset
            if len(batch) == self.batch_size:
                yield batch
                batch = []

    def release(self):
        self.cap.release()