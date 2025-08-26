"""
Video processing utilities for frame extraction and preprocessing
"""

import cv2
import numpy as np
from typing import List
import os

class VideoProcessor:
    @staticmethod
    def extract_frames(video_path: str, output_dir: str, 
                      target_fps: int = 5, img_size: tuple = (64, 64)) -> int:
        """Extract frames from video and save as images"""
        os.makedirs(output_dir, exist_ok=True)
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = max(1, int(fps / target_fps))
        saved_count = 0
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % frame_interval == 0:
                processed = VideoProcessor.process_frame(frame, img_size)
                cv2.imwrite(f"{output_dir}/{saved_count:06d}.png", processed)
                saved_count += 1
                
            frame_count += 1
        
        cap.release()
        return saved_count

    @staticmethod
    def process_frame(frame: np.ndarray, size: tuple = (64, 64)) -> np.ndarray:
        """Convert frame to grayscale and resize"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, size)
        return resized

    @staticmethod
    def load_frames(frames_dir: str, size: tuple = (64, 64)) -> np.ndarray:
        """Load and preprocess frames from directory"""
        frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.png')])
        frames = []
        
        for frame_file in frame_files:
            frame_path = os.path.join(frames_dir, frame_file)
            frame = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
            frame = cv2.resize(frame, size)
            frame = frame.astype(np.float32) / 255.0
            frames.append(frame)
        
        return np.array(frames)[..., np.newaxis]  # Add channel dimension