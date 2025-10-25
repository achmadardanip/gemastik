# AutoGluon CIFAR-10 Image Classification

This project demonstrates using AutoGluon for image classification on the CIFAR-10 dataset with comprehensive model evaluation and hyperparameter optimization.

## Features

- 🚀 **AutoGluon MultiModalPredictor**: Automated machine learning for image classification
- 🎯 **CIFAR-10 Dataset**: 10-class image classification benchmark
- 🔍 **All Supported Backbones**: Tests all pretrained models available in AutoGluon
- ⚙️ **Hyperparameter Optimization (HPO)**: Automatic hyperparameter tuning
- 📊 **Comprehensive Analysis**:
  - Model leaderboard with performance metrics
  - Feature importance visualization
  - Confusion matrix and classification reports
  - Per-class performance metrics
  - Prediction confidence analysis
- 💾 **Model Persistence**: Save and load trained models

## Dataset

**CIFAR-10** consists of 60,000 32x32 color images in 10 classes:
- Airplane
- Automobile
- Bird
- Cat
- Deer
- Dog
- Frog
- Horse
- Ship
- Truck

Dataset source: https://www.cs.toronto.edu/~kriz/cifar.html

## Requirements

- Python 3.8+
- CUDA-capable GPU (recommended, especially A100 for Modal.com)
- 16GB+ RAM
- 10GB+ disk space

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Running Locally

Open and run the Jupyter notebook:

```bash
jupyter notebook autogluon_cifar10_classification.ipynb
```

### Running on Modal.com

This notebook is designed to run on Modal.com with A100 GPU support. Follow these steps:

1. Install Modal CLI:
```bash
pip install modal
```

2. Set up Modal authentication:
```bash
modal token new
```

3. Upload the notebook to Modal.com and run it in their notebook environment with A100 GPU configuration.

For detailed Modal.com notebook documentation: https://modal.com/docs/guide/notebooks-modal

## Notebook Structure

1. **Setup and Imports**: Install and import required packages
2. **Download CIFAR-10**: Automatic dataset download and extraction
3. **Data Preparation**: Convert CIFAR-10 to image files and create train/val/test splits
4. **List Backbones**: Display all supported pretrained models
5. **Configure HPO**: Set up hyperparameter optimization
6. **Train Models**: Train multiple models with different backbones and hyperparameters
7. **Leaderboard Analysis**: View and analyze model performance
8. **Feature Importance**: Visualize prediction confidence (proxy for feature importance in image models)
9. **Model Evaluation**: Comprehensive evaluation on test set with confusion matrix
10. **Save Model**: Persist trained models for future use
11. **Summary**: Complete experiment summary

## Outputs

The notebook generates the following outputs:

- `model_leaderboard.csv`: Performance metrics for all trained models
- `leaderboard_analysis.png`: Visual analysis of model performance
- `prediction_confidence_analysis.png`: Confidence analysis by class
- `confusion_matrix.png`: Test set confusion matrix
- `classification_report.csv`: Detailed per-class metrics
- `per_class_metrics.png`: Visualization of precision, recall, F1-score
- `misclassified_examples.png`: Examples of misclassified images
- `autogluon_cifar10_model/`: Directory containing saved model artifacts

## AutoGluon Documentation

For more information about AutoGluon:
- Official documentation: https://auto.gluon.ai/stable/index.html
- MultiModalPredictor: https://auto.gluon.ai/stable/tutorials/multimodal/index.html

## Performance Tips

1. **Time Budget**: Adjust `time_limit` parameter based on available resources
2. **HPO Trials**: Reduce `num_trials` for faster experimentation
3. **Batch Size**: Adjust based on GPU memory availability
4. **Model Selection**: Filter `supported_models` to focus on specific architectures

## Customization

You can customize the following parameters:

- **Training time**: Modify `time_limit` in the training cell
- **HPO configuration**: Adjust `hyperparameters` and `hyperparameter_tune_kwargs`
- **Train/val split**: Change `test_size` in the data split cell
- **Models to test**: Filter `supported_models` list
- **Evaluation metrics**: Add custom metrics in the evaluation cell

## License

This project uses the CIFAR-10 dataset which is freely available for research purposes.

## Acknowledgments

- CIFAR-10 dataset creators: Alex Krizhevsky, Vinod Nair, and Geoffrey Hinton
- AutoGluon team for the excellent AutoML framework
- Modal.com for providing GPU infrastructure
