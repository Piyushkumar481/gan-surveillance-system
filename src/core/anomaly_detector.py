"""
Advanced anomaly detection with multiple detection strategies and post-processing
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy import stats
import cv2
from datetime import datetime
from src.utils.logger import logger

class AdvancedAnomalyDetector:
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {
            'default_threshold': 0.1,
            'adaptive_threshold': True,
            'min_anomaly_duration': 1.0,  # seconds
            'confidence_window': 10,  # frames for confidence calculation
            'motion_weight': 0.3,
            'reconstruction_weight': 0.7
        }
        
        # State tracking
        self.anomaly_history = []
        self.frame_history = []
        self.confidence_scores = []
    
    def detect_anomalies_advanced(self, 
                                frames: np.ndarray,
                                reconstructions: np.ndarray,
                                fps: float = 5.0) -> Dict:
        """
        Advanced anomaly detection with multiple features
        
        Args:
            frames: Original frames (batch_size, height, width, channels)
            reconstructions: Reconstructed frames from generator
            fps: Frames per second for temporal analysis
            
        Returns:
            Dictionary with detailed anomaly information
        """
        if len(frames) != len(reconstructions):
            raise ValueError("Frames and reconstructions must have same length")
        
        results = {
            'frame_level': [],
            'sequence_level': [],
            'summary': {},
            'timestamps': []
        }
        
        # Calculate basic reconstruction errors
        reconstruction_errors = self._calculate_reconstruction_errors(frames, reconstructions)
        
        # Calculate motion features
        motion_features = self._calculate_motion_features(frames)
        
        # Calculate combined anomaly scores
        anomaly_scores = self._combine_features(
            reconstruction_errors, 
            motion_features
        )
        
        # Apply adaptive thresholding
        thresholds = self._calculate_adaptive_thresholds(anomaly_scores)
        
        # Detect anomalies at frame level
        for i, (score, threshold) in enumerate(zip(anomaly_scores, thresholds)):
            is_anomaly = score > threshold
            confidence = self._calculate_confidence(score, threshold, i)
            
            frame_result = {
                'frame_index': i,
                'anomaly_score': float(score),
                'reconstruction_error': float(reconstruction_errors[i]),
                'motion_score': float(motion_features[i] if i < len(motion_features) else 0),
                'threshold': float(threshold),
                'is_anomaly': bool(is_anomaly),
                'confidence': float(confidence),
                'timestamp': datetime.now().isoformat()
            }
            
            results['frame_level'].append(frame_result)
            results['timestamps'].append(datetime.now().isoformat())
        
        # Detect sequence-level anomalies
        results['sequence_level'] = self._detect_sequence_anomalies(
            results['frame_level'], 
            fps
        )
        
        # Generate summary
        results['summary'] = self._generate_summary(results['frame_level'])
        
        # Update history for adaptive learning
        self._update_detection_history(results['frame_level'])
        
        return results
    
    def _calculate_reconstruction_errors(self, 
                                       frames: np.ndarray, 
                                       reconstructions: np.ndarray) -> np.ndarray:
        """Calculate multiple reconstruction error metrics"""
        # Mean Squared Error
        mse_errors = np.mean((frames - reconstructions) ** 2, axis=(1, 2, 3))
        
        # Structural Similarity Index (SSIM)
        ssim_errors = np.array([1 - self._calculate_ssim(frames[i], reconstructions[i]) 
                              for i in range(len(frames))])
        
        # Combine errors (you can adjust weights)
        combined_errors = 0.7 * mse_errors + 0.3 * ssim_errors
        
        return combined_errors
    
    def _calculate_ssim(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        """Calculate Structural Similarity Index between two frames"""
        try:
            # Ensure frames are 2D for SSIM calculation
            if len(frame1.shape) == 3:
                frame1 = frame1.squeeze()
            if len(frame2.shape) == 3:
                frame2 = frame2.squeeze()
            
            # Simple SSIM approximation for performance
            C1 = 0.01 ** 2
            C2 = 0.03 ** 2
            
            mu1 = np.mean(frame1)
            mu2 = np.mean(frame2)
            sigma1 = np.std(frame1)
            sigma2 = np.std(frame2)
            sigma12 = np.cov(frame1.flatten(), frame2.flatten())[0, 1]
            
            ssim = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1 ** 2 + sigma2 ** 2 + C2))
            
            return max(0, min(ssim, 1))
            
        except:
            return 0.5  # Default value on error
    
    def _calculate_motion_features(self, frames: np.ndarray) -> np.ndarray:
        """Calculate motion-based features between consecutive frames"""
        if len(frames) < 2:
            return np.zeros(len(frames))
        
        motion_scores = np.zeros(len(frames))
        
        for i in range(1, len(frames)):
            # Calculate optical flow or frame difference
            frame_diff = np.abs(frames[i] - frames[i-1])
            motion_scores[i] = np.mean(frame_diff)
        
        # Normalize motion scores
        if np.max(motion_scores) > 0:
            motion_scores = motion_scores / np.max(motion_scores)
        
        return motion_scores
    
    def _combine_features(self, 
                         reconstruction_errors: np.ndarray, 
                         motion_scores: np.ndarray) -> np.ndarray:
        """Combine multiple features into final anomaly scores"""
        # Normalize reconstruction errors
        if np.max(reconstruction_errors) > 0:
            rec_errors_norm = reconstruction_errors / np.max(reconstruction_errors)
        else:
            rec_errors_norm = reconstruction_errors
        
        # Combine with weights from config
        combined_scores = (
            self.config['reconstruction_weight'] * rec_errors_norm +
            self.config['motion_weight'] * motion_scores
        )
        
        return combined_scores
    
    def _calculate_adaptive_thresholds(self, anomaly_scores: np.ndarray) -> np.ndarray:
        """Calculate adaptive thresholds based on score distribution"""
        if not self.config['adaptive_threshold']:
            return np.full_like(anomaly_scores, self.config['default_threshold'])
        
        thresholds = np.zeros_like(anomaly_scores)
        
        for i in range(len(anomaly_scores)):
            # Use recent history for adaptive threshold
            if len(self.anomaly_history) > 10:
                recent_scores = self.anomaly_history[-10:] + [anomaly_scores[i]]
            else:
                recent_scores = [anomaly_scores[i]]
            
            # Calculate dynamic threshold (mean + 2*std)
            mean_score = np.mean(recent_scores)
            std_score = np.std(recent_scores)
            
            thresholds[i] = mean_score + 2 * std_score
            
            # Ensure threshold is within reasonable bounds
            thresholds[i] = max(self.config['default_threshold'] * 0.5, 
                              min(thresholds[i], self.config['default_threshold'] * 2.0))
        
        return thresholds
    
    def _calculate_confidence(self, score: float, threshold: float, frame_index: int) -> float:
        """Calculate detection confidence score"""
        # Base confidence based on distance from threshold
        distance_ratio = (score - threshold) / threshold if threshold > 0 else 0
        base_confidence = min(1.0, max(0.0, distance_ratio))
        
        # Increase confidence for consistent anomalies
        if len(self.anomaly_history) > 0 and score > threshold:
            recent_anomalies = sum(1 for s in self.anomaly_history[-5:] if s > threshold)
            consistency_boost = min(0.3, recent_anomalies * 0.1)
            base_confidence += consistency_boost
        
        return min(1.0, base_confidence)
    
    def _detect_sequence_anomalies(self, 
                                 frame_results: List[Dict], 
                                 fps: float) -> List[Dict]:
        """Detect anomalous sequences (multiple consecutive anomalies)"""
        sequence_anomalies = []
        current_sequence = []
        min_duration_frames = max(1, int(self.config['min_anomaly_duration'] * fps))
        
        for i, result in enumerate(frame_results):
            if result['is_anomaly']:
                current_sequence.append((i, result))
            else:
                if len(current_sequence) >= min_duration_frames:
                    sequence_anomalies.append(self._analyze_sequence(current_sequence))
                current_sequence = []
        
        # Check final sequence
        if len(current_sequence) >= min_duration_frames:
            sequence_anomalies.append(self._analyze_sequence(current_sequence))
        
        return sequence_anomalies
    
    def _analyze_sequence(self, sequence: List[Tuple[int, Dict]]) -> Dict:
        """Analyze a sequence of anomalous frames"""
        frames = [item[0] for item in sequence]
        scores = [item[1]['anomaly_score'] for item in sequence]
        confidences = [item[1]['confidence'] for item in sequence]
        
        return {
            'start_frame': frames[0],
            'end_frame': frames[-1],
            'duration_frames': len(frames),
            'max_score': max(scores),
            'avg_score': sum(scores) / len(scores),
            'avg_confidence': sum(confidences) / len(confidences),
            'severity': self._calculate_severity(scores)
        }
    
    def _calculate_severity(self, scores: List[float]) -> str:
        """Calculate anomaly severity level"""
        avg_score = sum(scores) / len(scores)
        
        if avg_score > 0.3:
            return "CRITICAL"
        elif avg_score > 0.2:
            return "HIGH"
        elif avg_score > 0.15:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _generate_summary(self, frame_results: List[Dict]) -> Dict:
        """Generate summary statistics for the detection run"""
        anomaly_frames = [r for r in frame_results if r['is_anomaly']]
        normal_frames = [r for r in frame_results if not r['is_anomaly']]
        
        if not frame_results:
            return {}
        
        return {
            'total_frames': len(frame_results),
            'anomaly_frames': len(anomaly_frames),
            'anomaly_percentage': len(anomaly_frames) / len(frame_results) * 100,
            'max_anomaly_score': max(r['anomaly_score'] for r in frame_results) if frame_results else 0,
            'avg_anomaly_score': sum(r['anomaly_score'] for r in frame_results) / len(frame_results) if frame_results else 0,
            'avg_confidence': sum(r['confidence'] for r in frame_results) / len(frame_results) if frame_results else 0,
            'critical_anomalies': sum(1 for r in anomaly_frames if r['anomaly_score'] > 0.3),
            'start_time': frame_results[0]['timestamp'] if frame_results else None,
            'end_time': frame_results[-1]['timestamp'] if frame_results else None
        }
    
    def _update_detection_history(self, frame_results: List[Dict]):
        """Update detection history for adaptive learning"""
        scores = [r['anomaly_score'] for r in frame_results]
        self.anomaly_history.extend(scores)
        
        # Keep history size manageable
        if len(self.anomaly_history) > 1000:
            self.anomaly_history = self.anomaly_history[-500:]
    
    def reset_history(self):
        """Reset detection history"""
        self.anomaly_history = []
        self.frame_history = []
        self.confidence_scores = []

# Simple anomaly detector for basic use cases
class SimpleAnomalyDetector:
    @staticmethod
    def detect_anomalies(frames: np.ndarray, 
                        reconstructions: np.ndarray, 
                        threshold: float = 0.1) -> np.ndarray:
        """
        Simple anomaly detection using reconstruction error
        
        Args:
            frames: Original frames
            reconstructions: Reconstructed frames
            threshold: Anomaly detection threshold
            
        Returns:
            Array of anomaly scores
        """
        errors = np.mean((frames - reconstructions) ** 2, axis=(1, 2, 3))
        return errors