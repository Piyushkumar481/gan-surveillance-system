import tensorflow as tf
import numpy as np
import argparse
import sys
from src.core.gan_model import GANAnomalyDetector
from src.core.video_processor import VideoProcessor
from src.utils.logger import logger
from src.database.models import init_db, get_engine

def train_model(args):
    """Train GAN model on video data"""
    logger.info("Starting model training...")
    
    # Extract frames from video
    frame_count = VideoProcessor.extract_frames(
        args.video_path, 
        args.output_dir,
        target_fps=args.fps
    )
    logger.info(f"Extracted {frame_count} frames")
    
    # Load frames for training
    frames = VideoProcessor.load_frames(args.output_dir)
    logger.info(f"Loaded {len(frames)} frames for training")
    
    # Initialize and train model
    model = GANAnomalyDetector()
    
    # Convert to TensorFlow dataset
    dataset = tf.data.Dataset.from_tensor_slices(frames)
    dataset = dataset.shuffle(len(frames)).batch(args.batch_size)
    
    # Train model
    history = model.train(dataset, epochs=args.epochs)
    model.save_model(args.model_dir)
    
    logger.info("Training completed successfully")
    return history

def detect_anomalies(args):
    """Run anomaly detection on video"""
    logger.info("Starting anomaly detection...")
    
    # Load trained model
    model = GANAnomalyDetector()
    model.load_model(args.model_dir)
    
    # Process video
    frame_count = VideoProcessor.extract_frames(
        args.video_path,
        args.output_dir,
        target_fps=args.fps
    )
    
    frames = VideoProcessor.load_frames(args.output_dir)
    anomalies = model.detect_anomalies(frames, threshold=args.threshold)
    
    # Print results
    anomalous_frames = sum(1 for score in anomalies if score > args.threshold)
    print(f"\n=== ANOMALY DETECTION RESULTS ===")
    print(f"Total frames: {len(anomalies)}")
    print(f"Anomalous frames: {anomalous_frames}")
    print(f"Anomaly rate: {anomalous_frames/len(anomalies):.2%}")
    
    return anomalies

def main():
    """Main command-line interface"""
    parser = argparse.ArgumentParser(description="GAN Surveillance System")
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train GAN model')
    train_parser.add_argument('--video-path', required=True, help='Input video file')
    train_parser.add_argument('--output-dir', default='training_frames', help='Frame output directory')
    train_parser.add_argument('--model-dir', default='saved_models', help='Model save directory')
    train_parser.add_argument('--epochs', type=int, default=50, help='Training epochs')
    train_parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    train_parser.add_argument('--fps', type=int, default=5, help='Frames per second')
    
    # Detect command
    detect_parser = subparsers.add_parser('detect', help='Detect anomalies')
    detect_parser.add_argument('--video-path', required=True, help='Input video file')
    detect_parser.add_argument('--model-dir', required=True, help='Model directory')
    detect_parser.add_argument('--output-dir', default='detection_frames', help='Frame output directory')
    detect_parser.add_argument('--threshold', type=float, default=0.1, help='Anomaly threshold')
    detect_parser.add_argument('--fps', type=int, default=5, help='Frames per second')
    
    # Database command
    db_parser = subparsers.add_parser('init-db', help='Initialize database')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        train_model(args)
    elif args.command == 'detect':
        detect_anomalies(args)
    elif args.command == 'init-db':
        init_db(get_engine())
        logger.info("Database initialized successfully")
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()