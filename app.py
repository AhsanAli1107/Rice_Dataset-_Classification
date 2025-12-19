import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps

# Load Keras model
@st.cache_resource
def load_model_from_file():
    return tf.keras.models.load_model("/Users/ahsanali/Desktop/Online_Model/Rice_Data_set.keras")

model = load_model_from_file()

st.title("Rice Classification")

file = st.file_uploader("Upload rice image", type=["jpg", "png"])

def preprocess_image(image):
    image = ImageOps.fit(image, (256, 256))
    image = image.convert("RGB")
    img = np.array(image) / 255.0           # shape (256, 256, 3)
    img = np.expand_dims(img, axis=0)       # add batch dimension -> (1, 256, 256, 3)
    return img

if file:
    image = Image.open(file)
    st.image(image, use_column_width=True)

    img = preprocess_image(image)
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction, axis=1)[0]

    class_names = ['Arborio', 'Basmati', 'Ipsala', 'Jasmine', 'Karacadag']
    st.success(f"Prediction: {class_names[predicted_class]}")
