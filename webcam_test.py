"""
Webcam integration test for GAN anomaly detection system
"""

import cv2
import numpy as np
import time
from src.core.gan_model import GANAnomalyDetector

class WebcamAnomalyDetector:
    def __init__(self, model_path="models"):
        """Initialize webcam detector with trained model"""
        self.model = GANAnomalyDetector()
        self.model.load_model(model_path)
        self.threshold = 0.045  # Adjust based on your calibration
        self.cap = None
        
    def initialize_camera(self, camera_index=0):
        """Initialize webcam connection"""
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise RuntimeError("Cannot open webcam")
        
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        return True
    
    def preprocess_frame(self, frame):
        """Convert frame to model input format"""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Resize to model input size
        resized = cv2.resize(gray, (64, 64))
        
        # Normalize to [0, 1]
        normalized = resized.astype(np.float32) / 255.0
        
        # Add batch and channel dimensions
        processed = normalized[np.newaxis, ..., np.newaxis]
        
        return processed, resized
    
    def detect_anomaly(self, frame):
        """Detect anomaly in a single frame"""
        processed_frame, _ = self.preprocess_frame(frame)
        anomaly_score = self.model.detect_anomalies(processed_frame)[0]
        is_anomalous = anomaly_score > self.threshold
        
        return anomaly_score, is_anomalous
    
    def run_detection(self, duration_seconds=30):
        """Run real-time anomaly detection"""
        if not self.initialize_camera():
            print("Error: Webcam initialization failed")
            return
        
        print("Starting webcam anomaly detection...")
        print("Press 'q' to quit")
        print(f"Detection threshold: {self.threshold}")
        print("-" * 50)
        
        start_time = time.time()
        frame_count = 0
        anomaly_count = 0
        
        try:
            while time.time() - start_time < duration_seconds:
                ret, frame = self.cap.read()
                if not ret:
                    print("Error: Failed to capture frame")
                    break
                
                # Detect anomaly
                score, is_anomalous = self.detect_anomaly(frame)
                frame_count += 1
                
                if is_anomalous:
                    anomaly_count += 1
                    status = "ANOMALY DETECTED"
                    color = (0, 0, 255)  # Red
                else:
                    status = "Normal"
                    color = (0, 255, 0)  # Green
                
                # Display results on frame
                cv2.putText(frame, f"Score: {score:.4f}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                cv2.putText(frame, status, (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                cv2.putText(frame, f"Threshold: {self.threshold}", (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Show frame
                cv2.imshow('Anomaly Detection', frame)
                
                # Exit on 'q' press
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        finally:
            self.cleanup()
        
        # Print summary statistics
        anomaly_rate = (anomaly_count / frame_count * 100) if frame_count > 0 else 0
        print("\n" + "=" * 50)
        print("Detection Summary:")
        print(f"Total frames processed: {frame_count}")
        print(f"Anomalous frames detected: {anomaly_count}")
        print(f"Anomaly rate: {anomaly_rate:.2f}%")
        print(f"Average processing time: {((time.time() - start_time)/frame_count)*1000:.2f} ms per frame" if frame_count > 0 else "N/A")
    
    def calibrate_threshold(self, calibration_seconds=5):
        """Calibrate threshold based on normal webcam footage"""
        print("Calibrating threshold...")
        print("Please ensure normal conditions for calibration")
        
        if not self.initialize_camera():
            return self.threshold
        
        scores = []
        start_time = time.time()
        
        while time.time() - start_time < calibration_seconds:
            ret, frame = self.cap.read()
            if ret:
                processed_frame, _ = self.preprocess_frame(frame)
                score = self.model.detect_anomalies(processed_frame)[0]
                scores.append(score)
            
            # Show calibration in progress
            cv2.putText(frame, "Calibrating...", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow('Calibration', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        self.cleanup()
        
        if scores:
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            calibrated_threshold = mean_score + 2 * std_score
            
            print(f"Calibration complete:")
            print(f"Mean score: {mean_score:.6f}")
            print(f"Standard deviation: {std_score:.6f}")
            print(f"Calibrated threshold: {calibrated_threshold:.6f}")
            
            self.threshold = calibrated_threshold
            return calibrated_threshold
        
        return self.threshold
    
    def cleanup(self):
        """Release resources"""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()

def main():
    """Main function for webcam testing"""
    print("GAN Surveillance System - Webcam Integration Test")
    print("=" * 60)
    
    try:
        detector = WebcamAnomalyDetector()
        
        # Option 1: Use default threshold
        # detector.run_detection(duration_seconds=30)
        
        # Option 2: Calibrate first, then detect
        print("Starting calibration...")
        calibrated_threshold = detector.calibrate_threshold(calibration_seconds=5)
        
        print("\nStarting detection with calibrated threshold...")
        detector.threshold = calibrated_threshold
        detector.run_detection(duration_seconds=20)
        
    except Exception as e:
        print(f"Error during webcam test: {e}")
        print("Please check:")
        print("1. Webcam is connected and accessible")
        print("2. Camera drivers are installed")
        print("3. No other application is using the webcam")

if __name__ == "__main__":
    main()