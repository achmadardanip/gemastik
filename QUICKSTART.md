# Quick Start Guide - GEMASTIK Object Detection

## 🚀 5-Minute Quick Start

### 1. Clone and Install (2 minutes)
```bash
git clone https://github.com/achmadardanip/gemastik.git
cd gemastik
pip install -r requirements.txt
```

### 2. Open Jupyter Notebook (Recommended)
```bash
jupyter notebook object_detection_toolkit.ipynb
```

**The notebook contains everything you need!** Just run the cells sequentially.

---

## 📝 Using Command-Line Scripts

### Option A: All-in-One Commands

#### Step 1: Generate Dataset (~10-15 minutes)
```bash
python generate_synthetic_dataset.py
```

#### Step 2: Train Baseline Model (~30-45 minutes)
```bash
python train.py --data synthetic_dataset/data.yaml --model yolov8m.pt --epochs 25 --batch 16
```

#### Step 3: Run Inference
```bash
python inference.py \
    --model runs/train/exp/weights/best.pt \
    --source synthetic_dataset/val/images \
    --save-csv baseline_submission.csv
```

### Option B: Full Competition Pipeline

#### 1. Generate Data
```bash
python generate_synthetic_dataset.py
# Creates synthetic_dataset/ with 10,000 train + 2,000 val images
```

#### 2. Train Multiple Models (Parallel if you have multiple GPUs)

**Baseline Model:**
```bash
python train.py --data synthetic_dataset/data.yaml --model yolov8m.pt \
    --epochs 25 --batch 16 --name baseline_yolov8m
```

**Speed Model:**
```bash
python train.py --data synthetic_dataset/data.yaml --model yolov8n.pt \
    --epochs 25 --batch 32 --name speed_yolov8n
```

**Accuracy Model:**
```bash
python train.py --data synthetic_dataset/data.yaml --model yolov8x.pt \
    --epochs 50 --batch 8 --name accuracy_yolov8x
```

#### 3. Ensemble for Best Results
```bash
python ensemble_wbf.py \
    --models runs/train/baseline_yolov8m/weights/best.pt \
             runs/train/accuracy_yolov8x/weights/best.pt \
    --source synthetic_dataset/val/images \
    --output ensemble_submission.csv \
    --iou-thr 0.5
```

---

## 🎯 Competition Day Workflow

### Hour 1-2: Setup & Data
```bash
# Install
pip install -r requirements.txt

# Generate dataset (can reduce size for testing)
python generate_synthetic_dataset.py
```

### Hour 3-4: Baseline
```bash
# Train baseline
python train.py --data synthetic_dataset/data.yaml --model yolov8m.pt \
    --epochs 25 --batch 16 --name baseline

# Get first submission
python inference.py --model runs/train/baseline/weights/best.pt \
    --source test_images/ --save-csv submission_v1.csv
```

### Hour 5-8: Improve Models
```bash
# Train larger model for accuracy
python train.py --data synthetic_dataset/data.yaml --model yolov8x.pt \
    --epochs 50 --batch 8 --name accuracy

# Train smaller model for speed (optional)
python train.py --data synthetic_dataset/data.yaml --model yolov8n.pt \
    --epochs 25 --batch 32 --name speed
```

### Hour 9-10: Final Submission
```bash
# Ensemble best models
python ensemble_wbf.py \
    --models runs/train/baseline/weights/best.pt \
             runs/train/accuracy/weights/best.pt \
    --source test_images/ \
    --output final_submission.csv
```

---

## 🔧 Common Use Cases

### Using CPU Only
```bash
python train.py --data synthetic_dataset/data.yaml --model yolov8n.pt \
    --device cpu --batch 4 --epochs 10
```

### Smaller Dataset for Testing
Edit `generate_synthetic_dataset.py` line 206:
```python
generator.generate_dataset(
    num_train=1000,  # Reduced from 10000
    num_val=200,     # Reduced from 2000
    min_objects=1,
    max_objects=5
)
```

### Custom Configuration
```bash
python train.py \
    --data custom_data.yaml \
    --model yolov8m.pt \
    --epochs 100 \
    --batch 16 \
    --imgsz 640 \
    --patience 15 \
    --optimizer AdamW
```

### Visualize Predictions
```bash
python inference.py \
    --model runs/train/exp/weights/best.pt \
    --source synthetic_dataset/val/images \
    --visualize \
    --save-csv predictions.csv
```

---

## 📊 Expected Timeline

| Task | Time | Command |
|------|------|---------|
| Install deps | 2 min | `pip install -r requirements.txt` |
| Generate data | 15 min | `python generate_synthetic_dataset.py` |
| Train baseline | 45 min | `python train.py --model yolov8m.pt` |
| Train accuracy | 90 min | `python train.py --model yolov8x.pt` |
| Inference | 5 min | `python inference.py ...` |
| Ensemble | 10 min | `python ensemble_wbf.py ...` |

---

## 🐛 Troubleshooting

### CUDA Out of Memory
```bash
# Reduce batch size
python train.py --batch 8  # or --batch 4

# Or use smaller model
python train.py --model yolov8s.pt
```

### Slow Training
```bash
# Use smaller dataset for testing
# Edit num_train=1000, num_val=200 in generate_synthetic_dataset.py

# Or reduce epochs
python train.py --epochs 10
```

### Import Errors
```bash
# Upgrade packages
pip install -r requirements.txt --upgrade

# Or install individually
pip install ultralytics torch torchvision opencv-python
```

---

## 📁 Output Files

After running the pipeline, you'll have:

```
gemastik/
├── synthetic_dataset/
│   ├── data.yaml
│   ├── train/
│   │   ├── images/    # 10,000 images
│   │   └── labels/    # 10,000 labels
│   └── val/
│       ├── images/    # 2,000 images
│       └── labels/    # 2,000 labels
├── runs/
│   ├── train/
│   │   ├── baseline/
│   │   │   └── weights/
│   │   │       └── best.pt    # Best model
│   │   └── accuracy/
│   │       └── weights/
│   │           └── best.pt
│   └── predict/
│       └── exp/
│           └── *.png          # Visualized predictions
└── *.csv                      # Submission files
```

---

## 💡 Pro Tips

1. **Use Jupyter Notebook** for interactive exploration
2. **Start with baseline** before trying complex models
3. **Monitor training** with TensorBoard: `tensorboard --logdir runs/train`
4. **Use ensemble** for final submission (usually +2-3% mAP)
5. **Visualize predictions** to catch issues early
6. **Track experiments** - keep notes on what works

---

## 🎓 Learning More

- **YOLO Documentation**: https://docs.ultralytics.com/
- **Competition Tips**: See README.md section "Competition Strategy"
- **Jupyter Notebook**: Contains detailed explanations and visualizations
- **Code Comments**: All scripts have inline documentation

---

## 🆘 Need Help?

1. Check the **README.md** for detailed documentation
2. Open **object_detection_toolkit.ipynb** for guided walkthrough
3. Open an issue on GitHub
4. Check YOLO docs: https://docs.ultralytics.com/

---

**Good luck! 🚀**
