"""
Training script for YOLO models on synthetic CIFAR-10 dataset.
Supports multiple YOLO variants and custom training configurations.
"""

import os
import argparse
from ultralytics import YOLO
import torch


def train_yolo(
    data_yaml='data.yaml',
    model='yolov8m.pt',
    epochs=25,
    batch=16,
    imgsz=640,
    device='',
    project='runs/train',
    name='exp',
    patience=10,
    save=True,
    pretrained=True,
    optimizer='auto',
    verbose=True,
    seed=42,
    **kwargs
):
    """
    Train YOLO model on the synthetic dataset.
    
    Args:
        data_yaml: Path to data.yaml configuration file
        model: Model to use (e.g., 'yolov8n.pt', 'yolov8m.pt', 'yolov8x.pt')
        epochs: Number of training epochs
        batch: Batch size
        imgsz: Image size
        device: Device to use ('' for auto, '0' for cuda:0, 'cpu' for CPU)
        project: Project directory
        name: Experiment name
        patience: Early stopping patience
        save: Save model checkpoints
        pretrained: Use pretrained weights
        optimizer: Optimizer to use
        verbose: Verbose output
        seed: Random seed
        **kwargs: Additional training arguments
    """
    
    # Set random seed for reproducibility
    torch.manual_seed(seed)
    
    print("="*60)
    print("YOLO Training Configuration")
    print("="*60)
    print(f"Data config: {data_yaml}")
    print(f"Model: {model}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch}")
    print(f"Image size: {imgsz}")
    print(f"Device: {device if device else 'auto'}")
    print(f"Project: {project}")
    print(f"Name: {name}")
    print("="*60 + "\n")
    
    # Initialize model
    model = YOLO(model)
    
    # Train the model
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        device=device,
        project=project,
        name=name,
        patience=patience,
        save=save,
        pretrained=pretrained,
        optimizer=optimizer,
        verbose=verbose,
        seed=seed,
        **kwargs
    )
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"\nBest model saved at: {model.trainer.best}")
    print(f"Results saved at: {model.trainer.save_dir}")
    
    # Print final metrics
    if hasattr(results, 'results_dict'):
        print("\nFinal Metrics:")
        metrics = results.results_dict
        if 'metrics/mAP50(B)' in metrics:
            print(f"  mAP@0.5: {metrics['metrics/mAP50(B)']:.4f}")
        if 'metrics/mAP50-95(B)' in metrics:
            print(f"  mAP@0.5:0.95: {metrics['metrics/mAP50-95(B)']:.4f}")
    
    return results


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(description='Train YOLO model on synthetic dataset')
    
    parser.add_argument('--data', type=str, default='data.yaml',
                       help='Path to data.yaml')
    parser.add_argument('--model', type=str, default='yolov8m.pt',
                       help='Model to use (yolov8n/s/m/l/x.pt, rtdetr-l/x.pt)')
    parser.add_argument('--epochs', type=int, default=25,
                       help='Number of epochs')
    parser.add_argument('--batch', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--imgsz', type=int, default=640,
                       help='Image size')
    parser.add_argument('--device', type=str, default='',
                       help='Device (e.g., 0 or 0,1,2,3 or cpu)')
    parser.add_argument('--project', type=str, default='runs/train',
                       help='Project directory')
    parser.add_argument('--name', type=str, default='exp',
                       help='Experiment name')
    parser.add_argument('--patience', type=int, default=10,
                       help='Early stopping patience')
    parser.add_argument('--optimizer', type=str, default='auto',
                       help='Optimizer (SGD, Adam, AdamW, auto)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Train the model
    train_yolo(
        data_yaml=args.data,
        model=args.model,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        project=args.project,
        name=args.name,
        patience=args.patience,
        optimizer=args.optimizer,
        seed=args.seed
    )


if __name__ == '__main__':
    main()
