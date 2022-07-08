import tensorflow as tf
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import numpy as np
from tensorflow.nn import softmax


MODEL_PATH = r'C:\Users\shirs\Desktop\CV Projects\Cotton Disease Prediction Project\TransferLearning_ResNet50_Model'
model = tf.keras.models.load_model(MODEL_PATH)

st.write("""
         # Cotton Disease Prediction
         """
         )

file = st.file_uploader("Please upload an image file", type=["jpg", "png"])

def import_and_predict(image_data, model):
    
    class_names = ['diseased cotton leaf', 'diseased cotton plant', 'fresh cotton leaf', 'fresh cotton plant']
    #image_array = np.array(image_data)
    x = np.expand_dims(image_data, axis=0) 
    predictions = model.predict(x)
    score = softmax( predictions )
    prediction = class_names[np.argmax(score)]
        
    return prediction
if file is None:
    st.text("Please upload an image file")
else:
    # image = Image.open(file)
    img = image.load_img(file)
    st.image(img, use_column_width=True)
    image = image.img_to_array(img)
    image = tf.keras.preprocessing.image.smart_resize( image, (256,256), interpolation='bilinear')
    
    size = (256,256)
    
    prediction = import_and_predict(image, model)
    
    st.write(prediction)