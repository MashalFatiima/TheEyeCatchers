import streamlit as st
from PIL import Image
import os
from src.model_loader import load_model
from src.image_preprocessor import preprocess_image
from src.predictor import predict

# Load the model once when the application starts
MODEL_PATH = r"C:\Users\Abdullah\Desktop\TheEyeCatchers\model"
st.session_state.setdefault("model", load_model(MODEL_PATH))

# Streamlit app
st.title("Diabetic Retinopathy Detection")

uploaded_file = st.file_uploader("Upload an image of the eye", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    input_tensor = preprocess_image(image)

    # Make a prediction
    model = st.session_state["model"]
    prediction = predict(model, input_tensor)

    # Display the result
    if prediction == 0:
        st.success("No Diabetic Retinopathy detected.")
    else:
        st.error("Diabetic Retinopathy detected.")
