import streamlit as st
import os
from PIL import Image
import pickle
import numpy as np
from numpy.linalg import norm
from sklearn.neighbors import NearestNeighbors
import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input

feature_list = np.array(pickle.load(open('embeddings.pkl', 'rb')))
filenames = pickle.load(open('filenames.pkl','rb'))

model = ResNet50(weights = 'imagenet', include_top=False, input_shape=(224,224,3))
model.trainable = False

model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])


st.title("PicMatch")
st.write("This app uses a pre-trained model to classify images into different categories.")
st.write("Please upload an image to classify it.")


def save_uploaded_file(uploaded_file):
    try:
        with open(os.path.join("uploads", uploaded_file.name), "wb") as f: #remove wb to get an error
            f.write(uploaded_file.getbuffer())
            return 1
    except Exception as e:
        return 0


def feature_extraction(img_path,model):
    img = image.load_img(img_path, target_size=(224,224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result/norm(result)
    
    return normalized_result
    
def recommend(features,feature_list):
    neighbours = NearestNeighbors(n_neighbors=5,algorithm='brute',metric='euclidean')
    neighbours.fit(feature_list)
    distances,indices = neighbours.kneighbors([features])
    return indices

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

if uploaded_file is not None:
    # Save the file to the disk
    if save_uploaded_file(uploaded_file):
        st.success("File saved")
        display_image = Image.open(uploaded_file)
        st.image(display_image, caption='Uploaded Image',width=200) # can add width=300
        
        features = feature_extraction(os.path.join("uploads",uploaded_file.name),model)
        indices = recommend(features,feature_list)
        
        st.text("Similar Images:")
        col1,col2,col3,col4,col5 = st.columns(5)
        with col1:
            st.image(filenames[indices[0][0]])
        with col2:
            st.image(filenames[indices[0][1]])
        with col3:
            st.image(filenames[indices[0][2]])
        with col4:
            st.image(filenames[indices[0][3]])
        with col5:
            st.image(filenames[indices[0][4]])
        
    else:
        st.error("Failed to save file")
        