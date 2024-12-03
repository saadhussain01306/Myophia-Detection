
# Myopia Detection Project

## Overview

This project implements a deep learning-based solution for detecting myopia (nearsightedness) using image classification. Built using **TensorFlow** and **Keras**, the model leverages transfer learning with the **MobileNetV2** architecture. The project includes functionality for training, validating, testing, and deploying the model, with an additional GUI feature for uploading images and obtaining predictions.

---

## Features

1. **Image Classification Model**:
   - Trained using transfer learning on the MobileNetV2 pre-trained model.
   - Fine-tuned for specific myopia classification tasks.
   
2. **Data Augmentation**:
   - Augments training data to improve model generalization.
   
3. **Performance Evaluation**:
   - Generates accuracy metrics, confusion matrix, and classification reports.
   
4. **Deployment Script**:
   - Includes a GUI-based script for predicting new images.

---

## Directory Structure

```
Myopia Detection Project/
├── train/          # Training dataset
├── valid/          # Validation dataset
├── test/           # Test dataset
├── model.ipynb     # Notebook for training, testing, and evaluation
├── model_script.py # Script for uploading and predicting new images
├── myopia_classifier_model.h5 # Saved trained model
└── README.md       # This README file
```

---

## Requirements

### Software and Libraries

- Python 3.7+
- TensorFlow 2.0+
- NumPy
- OpenCV
- Matplotlib
- Seaborn
- Scikit-learn
- Tkinter (for GUI functionality)

### Hardware

- GPU support recommended for training large datasets.

---

## Installation

1. Clone this repository or download the project files.

2. Install the required dependencies:
   ```bash
   pip install tensorflow numpy matplotlib opencv-python scikit-learn seaborn
   ```

3. Ensure your system has a GPU-enabled TensorFlow setup (optional).

---

## Dataset

### Structure
The dataset is organized into three main directories:
- `train`: Training images for model learning.
- `valid`: Validation images for hyperparameter tuning.
- `test`: Testing images for final model evaluation.

### Format
Each directory contains subfolders for the two classes:
- `Myopia`
- `Normal`

Ensure that images are cropped and resized to dimensions of **224x224 pixels** before feeding them into the model.

---

## Training the Model

1. Open the `model.ipynb` file in Jupyter Notebook or any IDE supporting notebooks.

2. Modify dataset paths in the notebook to match your dataset locations:
   ```python
   train_dir = r"Path_to_Train_Dataset"
   val_dir = r"Path_to_Validation_Dataset"
   test_dir = r"Path_to_Test_Dataset"
   ```

3. Run the cells to:
   - Load and preprocess the data.
   - Train the MobileNetV2-based model.
   - Fine-tune the model for improved accuracy.
   - Save the trained model as `myopia_classifier_model.h5`.

---

## Testing the Model

1. Evaluate the model using the test dataset:
   - Generates test accuracy, confusion matrix, and classification report.

2. Use the following script to visualize results:
   ```python
   plt.plot(history.history["accuracy"], label="Train Accuracy")
   plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
   plt.legend()
   plt.show()
   ```

---

## Predicting New Images

Use the `model_script.py` for uploading and predicting new images. 

### Steps:
1. Run the script:
   ```bash
   python model_script.py
   ```
2. A file dialog will open, allowing you to select an image.
3. The script processes the image, runs the prediction, and displays:
   - Predicted class (Myopia or Normal)
   - Confidence score

---

## Results

### Example Metrics
- **Training Accuracy**: ~98%
- **Validation Accuracy**: ~95%
- **Test Accuracy**: ~94%

### Confusion Matrix
| Predicted\Actual | Myopia | Normal |
|------------------|--------|--------|
| **Myopia**       | 100    | 5      |
| **Normal**       | 7      | 120    |

### Classification Report
| Class     | Precision | Recall | F1-Score |
|-----------|-----------|--------|----------|
| Myopia    | 0.95      | 0.94   | 0.94     |
| Normal    | 0.96      | 0.96   | 0.96     |

---

## Future Improvements

1. **Increase Dataset Size**:
   - Incorporate more diverse samples to improve robustness.

2. **Hyperparameter Tuning**:
   - Experiment with different learning rates, batch sizes, and dropout rates.

3. **Model Optimization**:
   - Implement pruning or quantization for faster inference on edge devices.

---

