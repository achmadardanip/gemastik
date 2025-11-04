# GEMASTIK Object Detection Toolkit

Toolkit lengkap untuk kompetisi Object Detection menggunakan synthetic dataset dari CIFAR-10.

## 📋 Deskripsi

Repository ini berisi toolkit komprehensif untuk membangun sistem object detection dari nol, termasuk:
- Generasi synthetic dataset dari CIFAR-10
- Training multiple YOLO models (speed vs accuracy)
- Ensemble predictions menggunakan Weighted Boxes Fusion (WBF)
- Evaluation dan explainability (XAI)
- Jupyter notebook lengkap untuk semua fase kompetisi

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Clone repository
git clone https://github.com/achmadardanip/gemastik.git
cd gemastik

# Install dependencies
pip install -r requirements.txt
```

### 2. Generate Synthetic Dataset

**Option A: Using Script**
```bash
python generate_synthetic_dataset.py
```

**Option B: Using Jupyter Notebook**
```bash
jupyter notebook object_detection_toolkit.ipynb
```

### 3. Train Models

**Baseline Model (YOLOv8-M)**
```bash
python train.py --data data.yaml --model yolov8m.pt --epochs 25 --batch 16
```

**Speed Model (YOLOv8-N)**
```bash
python train.py --data data.yaml --model yolov8n.pt --epochs 25 --batch 32 --name speed_yolov8n
```

**Accuracy Model (YOLOv8-X)**
```bash
python train.py --data data.yaml --model yolov8x.pt --epochs 50 --batch 8 --name accuracy_yolov8x
```

### 4. Run Inference

```bash
python inference.py --model runs/train/baseline_yolov8m/weights/best.pt \
    --source synthetic_dataset/val/images \
    --save-csv predictions.csv
```

### 5. Ensemble Predictions (WBF)

```bash
python ensemble_wbf.py \
    --models runs/train/baseline_yolov8m/weights/best.pt \
             runs/train/accuracy_yolov8x/weights/best.pt \
    --source synthetic_dataset/val/images \
    --output ensemble_submission.csv
```

## 📁 Repository Structure

```
gemastik/
├── requirements.txt                 # Package dependencies
├── data.yaml                       # Dataset configuration template
├── generate_synthetic_dataset.py   # Dataset generation script
├── train.py                        # Model training script
├── inference.py                    # Inference script
├── ensemble_wbf.py                # Ensemble predictions with WBF
├── object_detection_toolkit.ipynb # Comprehensive Jupyter notebook
└── README.md                       # This file
```

## 📊 Competition Phases

### Phase 0: Pre-Competition Preparation
- ✅ Environment setup
- ✅ Template scripts
- ✅ Configuration files

### Phase 1: Data Generation & Preparation
- ✅ EDA on CIFAR-10
- ✅ Synthetic dataset generation (10,000 train + 2,000 val)
- ✅ YOLO format annotations
- ✅ Built-in augmentation

### Phase 2: Baseline Model
- ✅ YOLOv8-M training
- ✅ Evaluation metrics (mAP@0.5, mAP@0.5:0.95)
- ✅ First submission

### Phase 3: Core Modeling
- **Speed Track**: YOLOv8-N / YOLO-NAS-S
- **Accuracy Track**: YOLOv8-X / RT-DETR-X

### Phase 4: Ensemble
- ✅ Weighted Boxes Fusion (WBF)
- ✅ Multi-model combination
- ✅ Improved accuracy

### Phase 5: Final Evaluation
- ✅ Per-class analysis
- ✅ Confusion matrix
- ✅ XAI validation
- ✅ Final submission

## 🔧 Key Features

### 1. Synthetic Dataset Generator
- Creates 640x640 canvas from 32x32 CIFAR-10 images
- 1-5 objects per image (configurable)
- Automatic YOLO format labels
- Balanced class distribution

### 2. Multi-Model Training
- Support for all YOLO variants (v8, v9, v10, v11)
- RT-DETR transformer-based detector
- Transfer learning from COCO weights
- Early stopping and checkpointing

### 3. Ensemble WBF
- Combines predictions from multiple models
- Weighted Boxes Fusion algorithm
- Better than standard NMS
- Configurable IoU thresholds

### 4. Comprehensive Notebook
- Interactive exploration
- Visualizations
- Step-by-step guide
- All phases in one place

## 📈 Expected Performance

| Model | mAP@0.5 | mAP@0.5:0.95 | Speed (ms) |
|-------|---------|--------------|------------|
| YOLOv8-N | ~0.85 | ~0.65 | ~5 |
| YOLOv8-M | ~0.90 | ~0.75 | ~15 |
| YOLOv8-X | ~0.95 | ~0.85 | ~40 |
| Ensemble | ~0.96 | ~0.87 | Variable |

*Note: Actual performance depends on dataset quality and training configuration*

## 🎯 Competition Strategy

### Time Allocation (10 hours)
- **Hours 1-2**: Setup + Data Generation
- **Hours 3-4**: Baseline Model
- **Hours 5-8**: Parallel Model Training
- **Hours 9-10**: Ensemble + Final Submission

### Best Practices
1. **Start Simple**: Get baseline working first
2. **Iterate Quickly**: Fast experiments over complex models
3. **Monitor Metrics**: Track per-class performance
4. **Use Ensemble**: WBF significantly improves results
5. **Verify Quality**: Check predictions visually

### Common Pitfalls
- ❌ Overfitting to validation set
- ❌ Too low confidence threshold
- ❌ Ignoring class confusion
- ❌ Not using ensemble
- ❌ Poor data quality

## 📚 Dependencies

Main packages:
- `ultralytics`: YOLO models (v8, v9, v10, v11, RT-DETR)
- `super-gradients`: YOLO-NAS
- `ensemble-boxes`: Weighted Boxes Fusion
- `torch`, `torchvision`: Deep learning framework
- `opencv-python`, `Pillow`: Image processing
- `pandas`, `matplotlib`, `seaborn`: Analysis and visualization

See `requirements.txt` for complete list.

## 🔬 Model Details

### YOLOv8 Variants
- **Nano (N)**: Fastest, lowest accuracy
- **Small (S)**: Good balance for mobile
- **Medium (M)**: Recommended baseline
- **Large (L)**: Higher accuracy
- **XLarge (X)**: Best accuracy, slowest

### RT-DETR (Transformer-based)
- End-to-end detection
- No NMS required
- Excellent accuracy
- Slower inference

### YOLO-NAS (Super-Gradients)
- Neural Architecture Search
- Optimized for edge devices
- Good speed/accuracy tradeoff

## 📊 Evaluation Metrics

- **mAP@0.5**: Mean Average Precision at IoU=0.5
- **mAP@0.5:0.95**: Mean AP averaged over IoU 0.5 to 0.95 (primary metric)
- **Precision**: TP / (TP + FP)
- **Recall**: TP / (TP + FN)
- **F1-Score**: Harmonic mean of precision and recall

## 🎓 Dataset Information

### CIFAR-10 Classes
1. Airplane
2. Automobile
3. Bird
4. Cat
5. Deer
6. Dog
7. Frog
8. Horse
9. Ship
10. Truck

### Synthetic Dataset Format
- **Train**: 10,000 images (640x640)
- **Val**: 2,000 images (640x640)
- **Objects**: 32x32 pixels (from CIFAR-10)
- **Format**: YOLO format (class x_center y_center width height, normalized)

## 🔍 Troubleshooting

### Issue: CUDA out of memory
**Solution**: Reduce batch size or use smaller model
```bash
python train.py --batch 8  # or --batch 4
```

### Issue: Slow training
**Solution**: Use CPU if GPU is not available, or reduce dataset size
```bash
python train.py --device cpu
```

### Issue: Low mAP scores
**Solutions**:
1. Train longer (increase epochs)
2. Adjust learning rate
3. Use larger model
4. Improve data quality
5. Use ensemble

### Issue: Import errors
**Solution**: Install missing packages
```bash
pip install -r requirements.txt --upgrade
```

## 📝 Citation

If you use this toolkit, please cite:

```bibtex
@misc{gemastik-od-toolkit,
  title={GEMASTIK Object Detection Toolkit},
  author={Achmad Ardani P.},
  year={2025},
  url={https://github.com/achmadardanip/gemastik}
}
```

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- CIFAR-10 dataset creators
- Ultralytics team (YOLO)
- Super-Gradients team (YOLO-NAS)
- Ensemble-boxes library
- GEMASTIK competition organizers

## 📧 Contact

For questions or issues, please:
- Open an issue on GitHub
- Contact: achmadardanip@github

---

**Good luck with the GEMASTIK competition! 🚀**
