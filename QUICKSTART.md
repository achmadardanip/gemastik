# Quick Start Guide - AutoGluon CIFAR-10 Classification

## Installation

```bash
# Clone the repository
git clone https://github.com/achmadardanip/gemastik.git
cd gemastik

# Install dependencies
pip install -r requirements.txt
```

## Running Locally

```bash
# Start Jupyter
jupyter notebook

# Open autogluon_cifar10_classification.ipynb
# Run all cells sequentially
```

## Running on Modal.com (Recommended)

### Why Modal.com?
- **A100 GPU Access**: High-performance training
- **Managed Infrastructure**: No setup required
- **Cost-Effective**: Pay only for usage

### Steps:

1. **Install Modal CLI**
   ```bash
   pip install modal
   ```

2. **Authenticate**
   ```bash
   modal token new
   ```

3. **Upload Notebook**
   - Go to https://modal.com/notebooks
   - Create new notebook
   - Upload `autogluon_cifar10_classification.ipynb`
   - Select **A100 GPU** in compute settings

4. **Run**
   - Execute cells in order
   - Monitor GPU usage in Modal dashboard
   - Download outputs when complete

## Expected Runtime

With A100 GPU:
- Dataset download: ~2-3 minutes
- Data preprocessing: ~5-10 minutes
- Model training (with HPO): ~1-3 hours (varies by configuration)
- Evaluation: ~5-10 minutes

**Total: ~1.5-4 hours**

## Configuration Tips

### For Quick Testing:
```python
# Reduce number of models
supported_models = supported_models[:5]  # Test only 5 models

# Reduce HPO trials
hyperparameter_tune_kwargs = {
    'num_trials': 5,  # Down from 20
}

# Shorter time limit
time_limit = 1800  # 30 minutes
```

### For Production:
```python
# Use all models (default)
supported_models = MultiModalPredictor.list_supported_models(pretrained=True)

# More HPO trials
hyperparameter_tune_kwargs = {
    'num_trials': 50,
}

# Longer time limit
time_limit = 14400  # 4 hours
```

## Troubleshooting

### Out of Memory Error
- Reduce batch size: `'env.batch_size': [16, 32]`
- Use fewer models at once
- Increase GPU memory allocation on Modal

### Slow Training
- Verify GPU is being used
- Check Modal dashboard for GPU utilization
- Reduce number of models or HPO trials

### Dataset Download Issues
- Check internet connection
- Manually download from https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
- Place in `cifar10_data/` directory

## Expected Outputs

After running the complete notebook:

### Files:
- `model_leaderboard.csv` - Performance comparison of all models
- `leaderboard_analysis.png` - Visual leaderboard analysis
- `confusion_matrix.png` - Test set confusion matrix
- `classification_report.csv` - Detailed metrics per class
- `per_class_metrics.png` - Precision, recall, F1 charts
- `prediction_confidence_analysis.png` - Confidence by class
- `misclassified_examples.png` - Sample errors
- `autogluon_cifar10_model/` - Saved model directory

### Console Output:
- Training progress for each model
- Validation scores
- Final test accuracy (typically 85-95%)
- Per-class performance metrics

## Next Steps

1. **Analyze Results**: Review leaderboard and confusion matrix
2. **Fine-tune**: Adjust hyperparameters based on results
3. **Deploy Model**: Use saved model for inference
4. **Experiment**: Try different model architectures or augmentations

## Support

For issues or questions:
- AutoGluon: https://auto.gluon.ai/stable/index.html
- Modal.com: https://modal.com/docs
- GitHub Issues: https://github.com/achmadardanip/gemastik/issues

## Performance Benchmarks

Expected accuracy on CIFAR-10 test set:
- **Basic models**: 70-80%
- **Good models**: 80-90%
- **Best models**: 90-95%
- **State-of-the-art**: 95-99%

Your results may vary based on:
- Models selected
- HPO configuration
- Training time
- Random initialization
