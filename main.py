import time
import cv2
import os
import numpy as np
import torch

from src.video_loader import VideoBatchLoader
from src.detector import YOLODetector
from src.visualizer import Visualizer

ENGINE = True  # Set to False to use .pt model

INPUT_VIDEO = "data/traffic_videos/clips.mp4"
OUTPUT_VIDEO = f"output_inference_{'engine' if ENGINE else 'pt'}.mp4"

MODEL_PATH = f"road_signs_project/yolo11n_run/weights/best{'.engine' if ENGINE else '.pt'}"
BATCH_SIZE = 16 

txt_file = f'results_{"engine" if ENGINE else "pt"}.txt'

def main():
    loader = VideoBatchLoader(INPUT_VIDEO, batch_size=BATCH_SIZE)
    detector = YOLODetector(MODEL_PATH)
    visualizer = Visualizer()

    # Setup Video Writer to save the output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, loader.fps, (loader.width, loader.height))

    print(f"Starting inference on {INPUT_VIDEO} with Batch Size {BATCH_SIZE}...")
    with open(txt_file, 'w', encoding='utf-8') as f:
        f.write(f"=== Inference Report ===\n")
        f.write(f"Model: {MODEL_PATH}\n")
        f.write(f"Batch Size: {BATCH_SIZE}\n")
        f.write(f"Video: {INPUT_VIDEO}\n")
        f.write("-" * 40 + "\n")
    
    total_start_time = time.time()
    frame_count = 0
    total_fps = 0.0

    for batch_frames in loader.get_batches():

        if ENGINE and len(batch_frames) < BATCH_SIZE:
            padding_size = BATCH_SIZE - len(batch_frames)
            blank_frame = np.zeros_like(batch_frames[0])
            batch_frames.extend([blank_frame] * padding_size)

        # Synchronize before timing
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        batch_start = time.time()

        results = detector.predict_batch(batch_frames)

        # Synchronize after inference
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        batch_time = time.time() - batch_start
        batch_fps = len(batch_frames)/batch_time
        total_fps += batch_fps
        
        annotated_frames = visualizer.draw_results(batch_frames, results)
        for frame in annotated_frames:
            out.write(frame)
        
        current_batch_size = len(batch_frames)
        frame_count += current_batch_size

        with open(txt_file, 'a', encoding='utf-8') as f:
            f.write(f"[Batch {frame_count // BATCH_SIZE}] Processed {len(batch_frames)} frames in {batch_time:.4f}s | FPS: {batch_fps:.2f}\n")
        print(f"Processed {len(batch_frames)} frames. Batch Inference Time: {batch_time:.4f}s")

    total_end_time = time.time()
    total_duration = total_end_time - total_start_time
    avg_fps = total_fps // (frame_count / BATCH_SIZE)

    loader.release()
    out.release()

    with open(txt_file, 'a', encoding='utf-8') as f:
        f.write("-" * 40 + "\n")
        f.write(f"SUMMARY:\n")
        f.write(f"Total Frames: {frame_count}\n")
        f.write(f"Total Time: {total_duration:.2f}s\n")
        f.write(f"Average FPS: {avg_fps:.2f}\n")
        f.write("=" * 40 + "\n")

    print("-" * 30)
    print("Processing Complete!")
    print(f"Total Frames: {frame_count}")
    print(f"Total Time: {total_duration:.2f}s")
    print(f"Average FPS: {avg_fps:.2f}")
    print(f"Output saved to: {os.path.abspath(OUTPUT_VIDEO)}")

if __name__ == "__main__":
    main()