import cv2

class Visualizer:
    def __init__(self):
        pass

    def draw_results(self, frames, results):
        """
        Draws bounding boxes on frames based on YOLO results.
        :param frames: List of original frames.
        :param results: List of YOLO result objects.
        :return: List of annotated frames.
        """
        annotated_frames = []

        for frame, result in zip(frames, results):
            annotated_frame = result.plot() 
            annotated_frames.append(annotated_frame)
            
        return annotated_frames