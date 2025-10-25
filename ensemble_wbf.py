"""
Ensemble predictions using Weighted Boxes Fusion (WBF).
Combines predictions from multiple models for improved accuracy.
"""

import os
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from ensemble_boxes import weighted_boxes_fusion
from ultralytics import YOLO
from tqdm import tqdm
import cv2


class EnsembleWBF:
    """Ensemble predictor using Weighted Boxes Fusion."""
    
    def __init__(self, model_paths, weights=None, iou_thr=0.5, skip_box_thr=0.0001, conf_type='avg'):
        """
        Initialize ensemble predictor.
        
        Args:
            model_paths: List of paths to trained model weights
            weights: List of weights for each model (None for equal weights)
            iou_thr: IoU threshold for WBF
            skip_box_thr: Skip boxes with confidence below this threshold
            conf_type: Confidence calculation type ('avg', 'max', 'box_and_model_avg')
        """
        self.model_paths = model_paths
        self.models = [YOLO(path) for path in model_paths]
        self.weights = weights if weights else [1] * len(model_paths)
        self.iou_thr = iou_thr
        self.skip_box_thr = skip_box_thr
        self.conf_type = conf_type
        
        print(f"Loaded {len(self.models)} models for ensemble")
        for i, path in enumerate(model_paths):
            print(f"  Model {i+1}: {path} (weight: {self.weights[i]})")
    
    def predict_single_image(self, image_path, imgsz=640, conf=0.001):
        """
        Run ensemble prediction on a single image.
        
        Args:
            image_path: Path to image
            imgsz: Image size
            conf: Minimum confidence threshold for initial predictions
        
        Returns:
            boxes, scores, labels after WBF
        """
        img = cv2.imread(image_path)
        h, w = img.shape[:2]
        
        # Collect predictions from all models
        boxes_list = []
        scores_list = []
        labels_list = []
        
        for model in self.models:
            results = model.predict(image_path, conf=conf, imgsz=imgsz, verbose=False)
            
            if results and len(results) > 0:
                result = results[0]
                
                if result.boxes is not None and len(result.boxes) > 0:
                    # Get boxes in normalized format [x1, y1, x2, y2]
                    boxes = result.boxes.xyxy.cpu().numpy()
                    boxes[:, [0, 2]] /= w  # normalize x
                    boxes[:, [1, 3]] /= h  # normalize y
                    
                    scores = result.boxes.conf.cpu().numpy()
                    labels = result.boxes.cls.cpu().numpy().astype(int)
                    
                    boxes_list.append(boxes.tolist())
                    scores_list.append(scores.tolist())
                    labels_list.append(labels.tolist())
                else:
                    # Empty prediction
                    boxes_list.append([])
                    scores_list.append([])
                    labels_list.append([])
            else:
                boxes_list.append([])
                scores_list.append([])
                labels_list.append([])
        
        # Apply WBF
        if any(len(b) > 0 for b in boxes_list):
            boxes, scores, labels = weighted_boxes_fusion(
                boxes_list,
                scores_list,
                labels_list,
                weights=self.weights,
                iou_thr=self.iou_thr,
                skip_box_thr=self.skip_box_thr,
                conf_type=self.conf_type
            )
            
            # Denormalize boxes
            boxes[:, [0, 2]] *= w
            boxes[:, [1, 3]] *= h
        else:
            boxes = np.array([])
            scores = np.array([])
            labels = np.array([])
        
        return boxes, scores, labels
    
    def predict_directory(self, image_dir, output_csv='ensemble_predictions.csv',
                         imgsz=640, conf=0.001, visualize=False, output_dir=None):
        """
        Run ensemble prediction on a directory of images.
        
        Args:
            image_dir: Directory containing images
            output_csv: Path to save predictions CSV
            imgsz: Image size
            conf: Minimum confidence threshold
            visualize: Whether to save visualized images
            output_dir: Directory to save visualized images
        
        Returns:
            DataFrame with predictions
        """
        image_paths = list(Path(image_dir).glob('*.png')) + \
                     list(Path(image_dir).glob('*.jpg')) + \
                     list(Path(image_dir).glob('*.jpeg'))
        
        print(f"\nProcessing {len(image_paths)} images...")
        
        predictions = []
        
        if visualize and output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        for image_path in tqdm(image_paths):
            boxes, scores, labels = self.predict_single_image(
                str(image_path), imgsz=imgsz, conf=conf
            )
            
            image_name = image_path.stem
            
            # Save predictions
            for box, score, label in zip(boxes, scores, labels):
                x_min, y_min, x_max, y_max = box
                predictions.append({
                    'image_id': image_name,
                    'class_id': int(label),
                    'confidence': float(score),
                    'x_min': float(x_min),
                    'y_min': float(y_min),
                    'x_max': float(x_max),
                    'y_max': float(y_max)
                })
            
            # Visualize if requested
            if visualize and output_dir and len(boxes) > 0:
                img = cv2.imread(str(image_path))
                for box, score, label in zip(boxes, scores, labels):
                    x_min, y_min, x_max, y_max = box
                    cv2.rectangle(img, (int(x_min), int(y_min)), 
                                (int(x_max), int(y_max)), (0, 255, 0), 2)
                    cv2.putText(img, f"{int(label)}: {score:.2f}", 
                              (int(x_min), int(y_min) - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                output_path = os.path.join(output_dir, f"{image_name}_ensemble.jpg")
                cv2.imwrite(output_path, img)
        
        # Create DataFrame and save
        df = pd.DataFrame(predictions)
        if len(df) > 0:
            df.to_csv(output_csv, index=False)
            print(f"\n✓ Ensemble predictions saved to: {output_csv}")
            print(f"Total detections: {len(df)}")
            print(f"Average detections per image: {len(df) / len(image_paths):.2f}")
        else:
            print("\nNo detections found!")
        
        return df


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(description='Ensemble predictions using WBF')
    
    parser.add_argument('--models', type=str, nargs='+', required=True,
                       help='Paths to trained model weights')
    parser.add_argument('--weights', type=float, nargs='+', default=None,
                       help='Weights for each model (default: equal weights)')
    parser.add_argument('--source', type=str, required=True,
                       help='Path to test images directory')
    parser.add_argument('--output', type=str, default='ensemble_predictions.csv',
                       help='Output CSV file path')
    parser.add_argument('--iou-thr', type=float, default=0.5,
                       help='IoU threshold for WBF')
    parser.add_argument('--conf', type=float, default=0.001,
                       help='Minimum confidence threshold')
    parser.add_argument('--skip-box-thr', type=float, default=0.0001,
                       help='Skip boxes below this threshold')
    parser.add_argument('--conf-type', type=str, default='avg',
                       choices=['avg', 'max', 'box_and_model_avg'],
                       help='Confidence calculation type')
    parser.add_argument('--imgsz', type=int, default=640,
                       help='Image size')
    parser.add_argument('--visualize', action='store_true',
                       help='Save visualized predictions')
    parser.add_argument('--output-dir', type=str, default='ensemble_vis',
                       help='Directory to save visualizations')
    
    args = parser.parse_args()
    
    # Validate weights
    if args.weights and len(args.weights) != len(args.models):
        raise ValueError(f"Number of weights ({len(args.weights)}) must match "
                        f"number of models ({len(args.models)})")
    
    # Create ensemble predictor
    ensemble = EnsembleWBF(
        model_paths=args.models,
        weights=args.weights,
        iou_thr=args.iou_thr,
        skip_box_thr=args.skip_box_thr,
        conf_type=args.conf_type
    )
    
    # Run predictions
    ensemble.predict_directory(
        image_dir=args.source,
        output_csv=args.output,
        imgsz=args.imgsz,
        conf=args.conf,
        visualize=args.visualize,
        output_dir=args.output_dir if args.visualize else None
    )


if __name__ == '__main__':
    main()
