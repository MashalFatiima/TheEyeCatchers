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
This dataset consists of a large collection of high-resolution retinal images captured under various imaging conditions. Each image is assessed by a medical professional and labeled as:

- **Diabetic Retinopathy (DR)**: 0
- **No Diabetic Retinopathy (No_DR)**: 1

#### Class-to-Index Mapping
- `{'DR': 0, 'No_DR': 1}`

### Data Split
- **Training samples**: 2076
- **Validation samples**: 531
- **Test samples**: 231

### Preprocessing
- Images resized to `224x224` pixels.
- Normalized pixel values to `[0, 1]`.
- Applied data augmentation techniques like random rotation and flipping to improve generalization.

---

## Model
### Architecture
- **Base Model**: ResNet50 (pre-trained on ImageNet).
- **Custom Layers**: Added a dense layer for binary classification with sigmoid activation.
- **Optimizer**: Adam (learning rate = 0.001).
- **Loss Function**: Binary Cross-Entropy.

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
- **Validation Loss**: 2.1362
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

### Running the App

4. Open the app in your browser at ``.


## Acknowledgments
- **Dataset**: [APTOS 2019 Blindness Detection (Kaggle)](https://www.kaggle.com/c/aptos2019-blindness-detection) or Messidor-2 dataset.
- **Pre-trained Model**: ResNet50 from Keras Applications.
- **Grad-CAM**: Selvaraju et al.'s paper on Visual Explanations from Deep Networks.

---

## Future Enhancements
- Extend the application to classify multiple stages of Diabetic Retinopathy.
- Integrate additional explainable AI techniques.
- Deploy the application to the cloud for wider accessibility.

---

## Contact
For questions or collaboration opportunities, reach out at [your-email@example.com].

