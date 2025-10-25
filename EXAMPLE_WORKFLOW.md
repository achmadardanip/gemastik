# Example Workflow - Complete Pipeline

This document shows a complete example workflow from start to finish.

## Prerequisites

```bash
# Verify Python version (3.8+)
python --version

# Verify GPU (optional but recommended)
nvidia-smi

# Clone repository
git clone https://github.com/achmadardanip/gemastik.git
cd gemastik
```

## Step 1: Environment Setup

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import ultralytics; print(f'Ultralytics version: {ultralytics.__version__}')"
python -c "import torch; print(f'PyTorch version: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

**Expected output:**
```
Ultralytics version: 8.x.x
PyTorch version: 2.x.x, CUDA: True
```

## Step 2: Generate Synthetic Dataset

```bash
# Run dataset generator
python generate_synthetic_dataset.py
```

**Expected output:**
```
Loading CIFAR-10 dataset...
Downloading https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz...
Extracting...

Generating synthetic dataset:
  Canvas size: 640x640
  Objects per image: 1-5
  Training images: 10000
  Validation images: 2000

Generating training set...
100%|██████████| 10000/10000 [05:23<00:00, 30.92it/s]

Generating validation set...
100%|██████████| 2000/2000 [01:04<00:00, 31.01it/s]

✓ Dataset generated successfully at: ./synthetic_dataset
✓ data.yaml created at: ./synthetic_dataset/data.yaml

Dataset generation complete!
```

**What you get:**
- `synthetic_dataset/train/images/` - 10,000 images (640x640)
- `synthetic_dataset/train/labels/` - 10,000 label files (YOLO format)
- `synthetic_dataset/val/images/` - 2,000 images (640x640)
- `synthetic_dataset/val/labels/` - 2,000 label files (YOLO format)
- `synthetic_dataset/data.yaml` - Dataset configuration

## Step 3: Verify Dataset

```bash
# Check dataset structure
ls -R synthetic_dataset/

# View a sample label file
head synthetic_dataset/train/labels/img_000000.txt
```

**Example label format:**
```
3 0.525000 0.312500 0.050000 0.050000
7 0.725000 0.462500 0.050000 0.050000
1 0.325000 0.812500 0.050000 0.050000
```
Format: `class_id x_center y_center width height` (all normalized 0-1)

## Step 4: Train Baseline Model

```bash
# Train YOLOv8-Medium (baseline)
python train.py \
    --data synthetic_dataset/data.yaml \
    --model yolov8m.pt \
    --epochs 25 \
    --batch 16 \
    --name baseline_yolov8m \
    --patience 10
```

**Expected output:**
```
Downloading https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt...
100%|██████████| 52.0M/52.0M [00:05<00:00, 10.0MB/s]

YOLO Training Configuration
============================================================
Data config: synthetic_dataset/data.yaml
Model: yolov8m.pt
Epochs: 25
Batch size: 16
Image size: 640
Device: 0
============================================================

Epoch   GPU_mem  box_loss  cls_loss  dfl_loss  Instances  Size
  1/25    4.85G    1.234     2.456     1.789        128   640:  
        100%|██████████| 625/625 [05:23<00:00]
        Class     Images  Instances      P      R  mAP50  mAP50-95:
          all       2000       5234  0.892  0.856  0.901     0.742

...

Training complete (45.6 minutes)
Results saved to runs/train/baseline_yolov8m/
Best model: runs/train/baseline_yolov8m/weights/best.pt

Final Metrics:
  mAP@0.5: 0.9012
  mAP@0.5:0.95: 0.7421
```

**Training artifacts:**
- `runs/train/baseline_yolov8m/weights/best.pt` - Best model
- `runs/train/baseline_yolov8m/weights/last.pt` - Last epoch
- `runs/train/baseline_yolov8m/results.png` - Training curves
- `runs/train/baseline_yolov8m/confusion_matrix.png` - Confusion matrix

## Step 5: Evaluate Model

```bash
# Run validation
yolo val \
    model=runs/train/baseline_yolov8m/weights/best.pt \
    data=synthetic_dataset/data.yaml
```

**Expected output:**
```
Validating: 100%|██████████| 125/125 [00:52<00:00]

                 Class     Images  Instances      P      R  mAP50  mAP50-95
                   all       2000       5234  0.892  0.856  0.901     0.742
              airplane       2000        523  0.912  0.889  0.924     0.756
            automobile       2000        524  0.898  0.876  0.912     0.745
                  bird       2000        521  0.876  0.843  0.887     0.721
                   cat       2000        525  0.869  0.834  0.878     0.715
                  deer       2000        523  0.887  0.856  0.898     0.738
                   dog       2000        522  0.863  0.828  0.871     0.709
                  frog       2000        524  0.901  0.878  0.915     0.751
                 horse       2000        523  0.894  0.867  0.905     0.743
                  ship       2000        526  0.908  0.885  0.919     0.754
                 truck       2000        523  0.895  0.871  0.908     0.746

Speed: 0.1ms pre-process, 15.2ms inference, 1.3ms NMS per image at shape (1, 3, 640, 640)
```

## Step 6: Run Inference

```bash
# Generate predictions on validation set
python inference.py \
    --model runs/train/baseline_yolov8m/weights/best.pt \
    --source synthetic_dataset/val/images \
    --conf 0.25 \
    --save-csv baseline_predictions.csv
```

**Expected output:**
```
YOLO Inference Configuration
============================================================
Model: runs/train/baseline_yolov8m/weights/best.pt
Source: synthetic_dataset/val/images
Confidence threshold: 0.25
IoU threshold: 0.45
============================================================

Predicting: 100%|██████████| 2000/2000 [02:15<00:00, 14.77it/s]

✓ Inference complete!
Results saved at: runs/predict/exp

✓ Predictions saved to: baseline_predictions.csv
Total detections: 5234
```

**Output CSV format:**
```csv
image_id,class_id,class_name,confidence,x_min,y_min,x_max,y_max
img_000000,3,cat,0.95,336,200,368,232
img_000000,7,horse,0.89,464,296,496,328
img_000001,1,automobile,0.92,208,520,240,552
...
```

## Step 7: Train Accuracy Model (Optional)

```bash
# Train larger model for better accuracy
python train.py \
    --data synthetic_dataset/data.yaml \
    --model yolov8x.pt \
    --epochs 50 \
    --batch 8 \
    --name accuracy_yolov8x \
    --patience 15
```

**Expected improvement:**
```
Final Metrics:
  mAP@0.5: 0.9523          (+5.1%)
  mAP@0.5:0.95: 0.8512     (+10.9%)
```

## Step 8: Ensemble Models

```bash
# Combine baseline and accuracy models
python ensemble_wbf.py \
    --models runs/train/baseline_yolov8m/weights/best.pt \
             runs/train/accuracy_yolov8x/weights/best.pt \
    --weights 0.742 0.851 \
    --source synthetic_dataset/val/images \
    --output ensemble_predictions.csv \
    --iou-thr 0.5 \
    --conf 0.001
```

**Expected output:**
```
Loaded 2 models for ensemble
  Model 1: baseline_yolov8m (weight: 0.742)
  Model 2: accuracy_yolov8x (weight: 0.851)

Processing 2000 images...
100%|██████████| 2000/2000 [08:45<00:00, 3.81it/s]

✓ Ensemble predictions saved to: ensemble_predictions.csv
Total detections: 5289
Average detections per image: 2.64
```

**Expected improvement:**
- Ensemble typically improves mAP by 1-3% over single best model
- Better confidence calibration
- Fewer false positives

## Step 9: Visualize Results

```bash
# Visualize ensemble predictions
python ensemble_wbf.py \
    --models runs/train/baseline_yolov8m/weights/best.pt \
             runs/train/accuracy_yolov8x/weights/best.pt \
    --source synthetic_dataset/val/images \
    --output ensemble_predictions.csv \
    --visualize \
    --output-dir ensemble_visualizations
```

**This creates:**
- `ensemble_visualizations/` with annotated images showing all detections

## Step 10: Analyze Performance

```bash
# Open Jupyter notebook for detailed analysis
jupyter notebook object_detection_toolkit.ipynb
```

**Navigate to Phase 5 (Final Evaluation) for:**
- Per-class performance breakdown
- Confusion matrix analysis
- Failure case analysis
- Confidence distribution plots

## Performance Comparison

| Model | Parameters | mAP@0.5 | mAP@0.5:0.95 | Inference (ms) | Training Time |
|-------|-----------|---------|--------------|----------------|---------------|
| YOLOv8-N | 3.2M | 0.8523 | 0.6512 | 5.2 | 20 min |
| YOLOv8-M | 25.9M | 0.9012 | 0.7421 | 15.2 | 45 min |
| YOLOv8-X | 68.2M | 0.9523 | 0.8512 | 42.8 | 95 min |
| Ensemble | - | 0.9634 | 0.8721 | 58.0 | - |

## Complete Timeline

| Step | Time | Cumulative |
|------|------|-----------|
| 1. Setup | 5 min | 0:05 |
| 2. Generate data | 15 min | 0:20 |
| 3. Verify data | 2 min | 0:22 |
| 4. Train baseline | 45 min | 1:07 |
| 5. Evaluate | 5 min | 1:12 |
| 6. Inference | 3 min | 1:15 |
| 7. Train accuracy | 95 min | 2:50 |
| 8. Ensemble | 10 min | 3:00 |
| 9. Visualize | 10 min | 3:10 |
| 10. Analyze | 15 min | 3:25 |

**Total: ~3.5 hours for complete pipeline**

## Tips for Competition

1. **Start Early**: Begin with baseline (Steps 1-6) immediately
2. **Parallel Training**: Train multiple models simultaneously if you have resources
3. **Quick Iterations**: Use smaller dataset (1000 images) for testing
4. **Monitor Progress**: Check results after each epoch
5. **Use Ensemble**: Always ensemble for final submission
6. **Verify Outputs**: Visualize predictions before submitting

## Next Steps

- Experiment with different models (RT-DETR, YOLO-NAS)
- Tune hyperparameters (learning rate, augmentation)
- Try different ensemble weights
- Analyze failure cases and iterate
- Submit to competition leaderboard!

---

**Good luck! 🚀**
