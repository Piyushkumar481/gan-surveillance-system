"""
Test different anomaly scenarios for webcam detection
"""

import cv2
import numpy as np
import time
from webcam_test import WebcamAnomalyDetector

def test_static_scenes():
    """Test detection with minimal movement"""
    print("Testing static scene detection...")
    detector = WebcamAnomalyDetector()
    detector.run_detection(duration_seconds=10)
    print("Static scene test completed\n")

def test_movement_scenes():
    """Test detection with normal movement"""
    print("Testing normal movement detection...")
    print("Please move normally in front of the camera")
    detector = WebcamAnomalyDetector()
    detector.run_detection(duration_seconds=15)
    print("Movement test completed\n")

def test_anomaly_scenarios():
    """Test specific anomaly scenarios"""
    print("Testing anomaly scenarios...")
    print("Please create unusual movements or objects")
    detector = WebcamAnomalyDetector()
    detector.run_detection(duration_seconds=20)
    print("Anomaly scenario test completed\n")

def performance_benchmark():
    """Benchmark system performance"""
    print("Running performance benchmark...")
    detector = WebcamAnomalyDetector()
    
    if not detector.initialize_camera():
        return
    
    frame_times = []
    frame_count = 100  # Test with 100 frames
    
    for i in range(frame_count):
        start_time = time.time()
        ret, frame = detector.cap.read()
        if ret:
            score, _ = detector.detect_anomaly(frame)
            end_time = time.time()
            frame_times.append((end_time - start_time) * 1000)  # Convert to ms
        
        if i % 10 == 0:
            print(f"Processed {i}/{frame_count} frames")
    
    detector.cleanup()
    
    if frame_times:
        avg_time = np.mean(frame_times)
        max_time = np.max(frame_times)
        min_time = np.min(frame_times)
        
        print(f"\nPerformance Results:")
        print(f"Average processing time: {avg_time:.2f} ms")
        print(f"Minimum processing time: {min_time:.2f} ms")
        print(f"Maximum processing time: {max_time:.2f} ms")
        print(f"Estimated FPS: {1000/avg_time:.1f}")

def main():
    """Comprehensive webcam testing suite"""
    print("Comprehensive Webcam Testing Suite")
    print("=" * 50)
    
    tests = [
        ("Static Scenes", test_static_scenes),
        ("Movement Detection", test_movement_scenes),
        ("Anomaly Scenarios", test_anomaly_scenarios),
        ("Performance Benchmark", performance_benchmark)
    ]
    
    for test_name, test_function in tests:
        print(f"\n{test_name}")
        print("-" * len(test_name))
        try:
            test_function()
        except Exception as e:
            print(f"Test failed: {e}")
    
    print("\nAll tests completed")

if __name__ == "__main__":
    main()