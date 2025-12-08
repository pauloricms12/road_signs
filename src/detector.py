from ultralytics import YOLO
import torch

class YOLODetector:
    def __init__(self, model_path, task='detect'):
        """
        Wrapper for YOLOv11 inference.
        :param model_path: Path to .pt or .engine file.
        """
        self.model = YOLO(model_path, task=task)
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Model loaded from {model_path} on {self.device}")

    def predict_batch(self, frames):
        """
        Runs inference on a batch of frames.
        :param frames: List of numpy arrays (images).
        :return: List of results objects.
        """
        # verbose=False keeps the console clean
        results = self.model(frames, device=self.device, verbose=False)
        return results