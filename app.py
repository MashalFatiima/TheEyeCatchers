import os
import streamlit as st
from PIL import Image
import torch
from model.model import load_model, predict_and_visualize
from utils.preprocess import preprocess_image

# Set Streamlit page configuration
st.set_page_config(page_title="TheEyeCatchers", layout="centered")

# Load the model
try:
    model, class_names = load_model()
except Exception as e:
    st.error(f"Failed to load the model: {e}")
    st.stop()

# App title
st.title("TheEyeCatchers: Diabetic Retinopathy Detection with Grad-CAM")

# Upload image
uploaded_file = st.file_uploader("Upload an image for prediction", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Save uploaded image
    image_path = os.path.join("uploads", uploaded_file.name)
    with open(image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Load and display the image
    image = Image.open(image_path).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    input_tensor = preprocess_image(image)

    # Make prediction and generate Grad-CAM
    try:
        predicted_label, probabilities, grad_cam_image = predict_and_visualize(model, input_tensor, image_path, class_names)

        # Display prediction results
        st.subheader("Prediction:")
        st.write(f"**Class:** {predicted_label}")
        st.write(f"**Probabilities:** {probabilities}")

        # Display Grad-CAM
        st.subheader("Grad-CAM Visualization:")
        st.image(grad_cam_image, caption="Grad-CAM Overlay", use_column_width=True)
    except Exception as e:
        st.error(f"Error during prediction: {e}")
