"""
Visualization utilities for anomaly detection results
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict
import cv2
import os

class ResultVisualizer:
    @staticmethod
    def plot_anomaly_scores(anomaly_scores: List[float], threshold: float = 0.1, 
                          save_path: str = None) -> plt.Figure:
        """Plot anomaly scores over time with threshold"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot anomaly scores
        ax.plot(anomaly_scores, 'b-', linewidth=1.5, label='Anomaly Score', alpha=0.7)
        
        # Add threshold line
        ax.axhline(y=threshold, color='r', linestyle='--', linewidth=2, 
                  label=f'Threshold ({threshold})', alpha=0.8)
        
        # Fill anomaly regions
        anomaly_indices = np.where(np.array(anomaly_scores) > threshold)[0]
        if len(anomaly_indices) > 0:
            for idx in anomaly_indices:
                ax.axvspan(idx-0.5, idx+0.5, color='red', alpha=0.2)
        
        # Customize plot
        ax.set_xlabel('Frame Number')
        ax.set_ylabel('Anomaly Score')
        ax.set_title('Anomaly Detection Results')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    @staticmethod
    def create_comparison_grid(original_frames: List[np.ndarray], 
                             reconstructed_frames: List[np.ndarray],
                             anomaly_scores: List[float],
                             save_path: str = None,
                             max_frames: int = 6) -> plt.Figure:
        """Create side-by-side comparison of original vs reconstructed frames"""
        # Select frames with highest anomaly scores
        if len(anomaly_scores) > max_frames:
            top_indices = np.argsort(anomaly_scores)[-max_frames:]
        else:
            top_indices = range(len(anomaly_scores))
        
        fig, axes = plt.subplots(2, min(len(top_indices), max_frames), 
                               figsize=(15, 6))
        
        if len(top_indices) == 1:
            axes = axes.reshape(2, 1)
        
        for i, idx in enumerate(top_indices):
            # Original frame
            axes[0, i].imshow(original_frames[idx].squeeze(), cmap='gray')
            axes[0, i].set_title(f'Original Frame {idx}')
            axes[0, i].axis('off')
            
            # Reconstructed frame
            axes[1, i].imshow(reconstructed_frames[idx].squeeze(), cmap='gray')
            axes[1, i].set_title(f'Reconstructed\nScore: {anomaly_scores[idx]:.3f}')
            axes[1, i].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    @staticmethod
    def generate_anomaly_report(anomaly_scores: List[float], threshold: float = 0.1) -> Dict:
        """Generate comprehensive anomaly report"""
        anomaly_scores = np.array(anomaly_scores)
        anomalies = anomaly_scores > threshold
        
        return {
            "total_frames": len(anomaly_scores),
            "anomalous_frames": int(np.sum(anomalies)),
            "anomaly_rate": float(np.mean(anomalies)),
            "max_anomaly_score": float(np.max(anomaly_scores)),
            "avg_anomaly_score": float(np.mean(anomaly_scores)),
            "std_anomaly_score": float(np.std(anomaly_scores)),
            "anomaly_threshold": threshold,
            "anomaly_frames": np.where(anomalies)[0].tolist()
        }