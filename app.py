import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2

# Load model
model = tf.keras.models.load_model("saved_models/best_EfficientNetB0.keras")

st.title("ConjunctivaScan 👁️")
st.write("AI-based anemia screening (NOT diagnostic)")

uploaded_file = st.file_uploader("Upload an eye image", type=["jpg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img = np.array(image)
    img = cv2.resize(img, (224, 224)) / 255.0
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img)[0][0]

    if pred > 0.5:
        st.success(f"Normal (Confidence: {pred:.2f})")
    else:
        st.warning(f"Possible Anemia (Confidence: {1-pred:.2f})")
