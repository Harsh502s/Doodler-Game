import streamlit as st
from streamlit_drawable_canvas import st_canvas
from keras.models import load_model
import cv2
import numpy as np
import os
from PIL import Image

# Load the model

model = load_model('vgg16_model_20.h5')

# Class labels

class_labels = [name for name in os.listdir('AugmentedData')]

# Header

st.write("""Doodler: A Deep Learning Game""")
st.write("This is a simple image classifier web app to predict the doodle drawn by the user")

# Function to predict the class from an image file with size of 224x224

def import_and_predict(image_data, model):
    size = (224,224)    
    image = cv2.resize(image_data, size)
    img = image.reshape(1,224,224,3)
    img = img/255.0
    prediction = model.predict(img).argmax()
    return prediction

# Creating a file uploader widget

uploaded_file = st.file_uploader("Choose an image...",type="jpg")

# Prediction the uploaded file using cv2 and the model

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    label = import_and_predict(np.array(image), model)
    label = class_labels[label]
    st.write('%s' % (label))
    pass

# Creating a canvas to draw the doodle

canvas_result = st_canvas(
    fill_color='#000000',
    stroke_width=4,
    stroke_color='#FFFFFF',
    background_color='#000000',
    width=700,
    height=400,
    drawing_mode="freedraw",
    key="canvas")

# Predicting the doodle drawn by the user while converting the input from the canvas to an to rgb because the input is in rgba format

if canvas_result.image_data is not None:
    img = cv2.cvtColor(canvas_result.image_data.astype('uint8'), cv2.COLOR_RGBA2RGB)
    img = cv2.resize(img, (224,224))
    st.write("Classifying...")
    label = import_and_predict(img, model)
    label = class_labels[label]
    st.write('%s' % (label))
    pass