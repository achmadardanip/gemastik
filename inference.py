"""
Inference script for YOLO models.
Run predictions on test images and save results.
"""

import os
import argparse
import pandas as pd
from ultralytics import YOLO
from pathlib import Path
import cv2
from tqdm import tqdm


def run_inference(
    model_path,
    source,
    conf=0.25,
    iou=0.45,
    imgsz=640,
    device='',
    save=True,
    save_txt=True,
    save_conf=True,
    project='runs/predict',
    name='exp',
    visualize=False,
    max_det=300
):
    """
    Run inference using trained YOLO model.
    
    Args:
        model_path: Path to trained model weights
        source: Path to test images directory or single image
        conf: Confidence threshold
        iou: IoU threshold for NMS
        imgsz: Image size
        device: Device to use
        save: Save results
        save_txt: Save results to txt files
        save_conf: Save confidence scores
        project: Project directory
        name: Experiment name
        visualize: Visualize results
        max_det: Maximum detections per image
    
    Returns:
        results: Prediction results
    """
    
    print("="*60)
    print("YOLO Inference Configuration")
    print("="*60)
    print(f"Model: {model_path}")
    print(f"Source: {source}")
    print(f"Confidence threshold: {conf}")
    print(f"IoU threshold: {iou}")
    print(f"Image size: {imgsz}")
    print(f"Device: {device if device else 'auto'}")
    print("="*60 + "\n")
    
    # Load model
    model = YOLO(model_path)
    
    # Run inference
    results = model.predict(
        source=source,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        device=device,
        save=save,
        save_txt=save_txt,
        save_conf=save_conf,
        project=project,
        name=name,
        visualize=visualize,
        max_det=max_det
    )
    
    print(f"\n✓ Inference complete!")
    print(f"Results saved at: {project}/{name}")
    
    return results


def results_to_csv(results, output_path='predictions.csv', class_names=None):
    """
    Convert YOLO results to CSV format for submission.
    
    Args:
        results: YOLO prediction results
        output_path: Path to save CSV file
        class_names: List of class names
    
    Returns:
        DataFrame with predictions
    """
    
    if class_names is None:
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                      'dog', 'frog', 'horse', 'ship', 'truck']
    
    predictions = []
    
    for result in results:
        image_name = Path(result.path).stem
        
        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
            confidences = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy().astype(int)
            
            for box, conf, cls_id in zip(boxes, confidences, class_ids):
                x_min, y_min, x_max, y_max = box
                predictions.append({
                    'image_id': image_name,
                    'class_id': cls_id,
                    'class_name': class_names[cls_id],
                    'confidence': conf,
                    'x_min': x_min,
                    'y_min': y_min,
                    'x_max': x_max,
                    'y_max': y_max
                })
    
    df = pd.DataFrame(predictions)
    df.to_csv(output_path, index=False)
    
    print(f"\n✓ Predictions saved to: {output_path}")
    print(f"Total detections: {len(df)}")
    
    return df


def visualize_predictions(image_path, boxes, class_names, output_path=None):
    """
    Visualize predictions on an image.
    
    Args:
        image_path: Path to image
        boxes: List of [x_min, y_min, x_max, y_max, conf, class_id]
        class_names: List of class names
        output_path: Path to save visualized image
    """
    img = cv2.imread(image_path)
    
    for box in boxes:
        x_min, y_min, x_max, y_max, conf, class_id = box
        
        # Draw bounding box
        cv2.rectangle(img, (int(x_min), int(y_min)), (int(x_max), int(y_max)), 
                     (0, 255, 0), 2)
        
        # Draw label
        label = f"{class_names[int(class_id)]}: {conf:.2f}"
        cv2.putText(img, label, (int(x_min), int(y_min) - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    if output_path:
        cv2.imwrite(output_path, img)
        print(f"Visualization saved to: {output_path}")
    
    return img


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(description='Run YOLO inference')
    
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model weights')
    parser.add_argument('--source', type=str, required=True,
                       help='Path to test images directory or single image')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.45,
                       help='IoU threshold for NMS')
    parser.add_argument('--imgsz', type=int, default=640,
                       help='Image size')
    parser.add_argument('--device', type=str, default='',
                       help='Device (e.g., 0 or 0,1,2,3 or cpu)')
    parser.add_argument('--project', type=str, default='runs/predict',
                       help='Project directory')
    parser.add_argument('--name', type=str, default='exp',
                       help='Experiment name')
    parser.add_argument('--save-csv', type=str, default=None,
                       help='Save predictions to CSV file')
    parser.add_argument('--visualize', action='store_true',
                       help='Visualize results')
    
    args = parser.parse_args()
    
    # Run inference
    results = run_inference(
        model_path=args.model,
        source=args.source,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=args.device,
        project=args.project,
        name=args.name,
        visualize=args.visualize
    )
    
    # Save to CSV if requested
    if args.save_csv:
        results_to_csv(results, args.save_csv)


if __name__ == '__main__':
    main()
