import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image
import base64
import requests
import os

# Set Streamlit page configuration
st.set_page_config(page_title="Fracture Detection App", layout="centered")

# Title and instructions
st.title("🦴 Fracture Detection from X-ray")
st.markdown("Upload an X-ray image and the model will predict whether a fracture is detected.")

# Upload image
uploaded_file = st.file_uploader("Upload X-ray Image", type=["jpg", "jpeg", "png"])

# Load the TensorFlow Lite model
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

if uploaded_file is not None:
    # Display uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded X-ray", use_column_width=True)

    if st.button("🚦 Detect Fracture"):
        with st.spinner("Analyzing X-ray..."):
            # Preprocess image
            img_resized = img.resize((128, 128))  # adjust to your model's input shape
            img_array = image.img_to_array(img_resized)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array / 255.0
            img_array = img_array.astype(np.float32)

            # Run inference
            interpreter.set_tensor(input_details[0]['index'], img_array)
            interpreter.invoke()
            prediction = interpreter.get_tensor(output_details[0]['index'])[0][0]

            # Show result
            if prediction > 0.5:
                st.error("❌ Fracture Detected")
            else:
                st.success("✅ No Fracture Detected")

