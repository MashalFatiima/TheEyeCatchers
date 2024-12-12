# Diabetic Retinopathy Detection using Deep Learning

## Overview
This project implements a deep learning-based approach to detect Diabetic Retinopathy (DR) in high-resolution retinal fundus images. Using a pre-trained ResNet50 model, the system classifies retinal images as either having Diabetic Retinopathy (DR) or not (No_DR) and provides explainable predictions using Grad-CAM. The project is deployed as a user-friendly **Streamlit app**.

---

## Features
- **High Accuracy**: Achieved 96.97% test accuracy using ResNet50 pre-trained model.
- **Explainability**: Grad-CAM highlights regions in the retinal images that influenced the model's prediction, enhancing clinical interpretability.
- **User Interface**: A Streamlit app allows users to upload retinal images and view predictions along with Grad-CAM visualizations.

---

## Dataset
### Data Description
This dataset comprises a large collection of high-resolution retinal images captured under various imaging conditions. Each image is assessed by a medical professional and labeled as:

- **Diabetic Retinopathy (DR)**: 0
- **No Diabetic Retinopathy (No_DR)**: 1

#### Class-to-Index Mapping
- `{'DR': 0, 'No_DR': 1}`

### Data Split
- **Training samples**: 2076
- **Validation samples**: 531
- **Test samples**: 231

### Preprocessing
The preprocessing pipeline includes the following steps:

- **Resize:** All images are resized to 224x224 pixels to match the input size of the ResNet50 model.
- **Normalization:** Pixel values are normalized using calculated mean and standard deviation values.
- **To Tensor:** Images are converted to PyTorch tensors for compatibility with the deep learning model.

---

## Model
### Architecture
- **Base Model:** ResNet50 (pre-trained on ImageNet).
- **Feature Extraction Layers:** All pre-trained layers were frozen to retain learned features.
- **Modified Fully Connected Layers:**
  - Input: Features from ResNet50 (num_features = 2048).
  - Added a dense layer with 128 neurons and ReLU activation for better learning.
  - Dropout layer with a rate of 0.3 for regularization.
  - Output: A dense layer with 2 neurons for binary classification (DR or No_DR).
- **Optimizer:** Adam with a learning rate of 0.001, applied only to the modified fully connected layers.
- **Loss Function:** CrossEntropyLoss for multi-class classification.

### Training Results
| Epoch | Training Loss | Training Accuracy |
|-------|---------------|-------------------|
| 1     | 18.9141       | 88.10%            |
| 2     | 12.6329       | 93.45%            |
| 3     | 11.3515       | 93.83%            |
| 4     | 12.9329       | 93.16%            |
| 5     | 9.7793        | 94.85%            |
| 6     | 9.3528        | 95.57%            |
| 7     | 9.2785        | 95.81%            |
| 8     | 7.7981        | 96.19%            |
| 9     | 8.3323        | 96.15%            |
| 10    | 8.1565        | 96.05%            |

### Evaluation Metrics
- **Validation Accuracy**: 95.86%
- **Test Accuracy**: 96.97%

---

## Explainability
### Grad-CAM Implementation
- Grad-CAM was applied to generate heatmaps that highlight the regions in the retinal images influencing the model’s predictions.
- These heatmaps provide visual explanations, aiding clinicians in understanding the AI's decision-making process.

---

## Deployment
### Streamlit App
The Streamlit app provides a simple interface for:
1. **Image Upload**: Users can upload a retinal fundus image.
2. **Prediction**: The app predicts if the image shows signs of Diabetic Retinopathy.
3. **Visualization**: Displays the Grad-CAM heatmap for explainability.

### App Interface
The following screenshots demonstrate the functionality of the Streamlit app:

1. **Home Page:** The app's landing page with an option to upload a retinal image.
   
3. **Image Upload:** Users can upload a retinal fundus image for prediction.
   
4. **Prediction and Heatmap:** Displays the prediction (DR / No_DR) along with the Grad-CAM heatmap highlighting the regions influencing the prediction.
   
## Acknowledgments
- **Dataset**: [Diagnosis of Diabetic Retinopathy (Kaggle)](https://www.kaggle.com/datasets/pkdarabi/diagnosis-of-diabetic-retinopathy/data).
- **Pre-trained Model**: ResNet50 from Keras Applications.

---

## Future Enhancements
- Extend the application to classify multiple stages of Diabetic Retinopathy.
- Integrate additional explainable AI techniques.
- Deploy the application to the cloud for wider accessibility.

---

