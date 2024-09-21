import streamlit as st
from PIL import Image
import os
from io import BytesIO
import requests

from config import config
from utils.model import CNNModel, generate_caption_beam_search
from keras.models import load_model
from pickle import load as pickle_load
from gtts import gTTS
import datetime
import numpy as np

# Load the model and tokenizer
image_model = CNNModel(config['model_type'])
caption_model = load_model(config['model_load_path'])
tokenizer = pickle_load(open(config['tokenizer_path'], 'rb'))
max_length = config['max_length']

st.title("Image Captioning and Audio Generation App")

# Function to extract features from an image
def extract_features_streamlit(image, model, model_type):
    if model_type == 'inceptionv3':
        from keras.applications.inception_v3 import preprocess_input
        target_size = (299, 299)
    elif model_type == 'vgg16':
        from keras.applications.vgg16 import preprocess_input
        target_size = (224, 224)
    image = image.resize(target_size)
    image = np.array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    features = model.predict(image, verbose=0)
    return features

# Function to generate caption and convert it to audio
def generate_caption_and_audio(image):
    features = extract_features_streamlit(image, image_model, config['model_type'])
    generated_caption = generate_caption_beam_search(caption_model, tokenizer, features, max_length, beam_index=config['beam_search_k'])
    
    # Remove startseq and endseq
    caption = ' '.join(generated_caption.split()[1:-1])
    
    # Convert the caption to audio
    tts = gTTS(text=caption, lang="en", slow=False)
    folder_path = config['audio_path']

    # This will save the generated captions with the image and the audio file in the folder that we have made.
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    audio_file_path = os.path.join(folder_path, f"output_{timestamp}.mp3")
    tts.save(audio_file_path)
    
    return caption, audio_file_path

# Function to load an image from a URL
def load_image_from_url(url):
    try:
        response = requests.get(url)
        img = Image.open(BytesIO(response.content))
        return img
    except:
        st.error("Invalid URL. Please enter a valid image URL.")
        return None
    
# Image upload options
st.header("Upload Image")
upload_option = st.radio("Choose how to upload the image:", ("Upload from system", "Enter URL"))

if upload_option == "Upload from system":
    uploaded_image = st.file_uploader("Upload an image file", type=["png", "jpg", "jpeg"])
    if uploaded_image is not None:
        img = Image.open(uploaded_image)
        st.image(img, caption="Uploaded Image", use_column_width=True)
elif upload_option == "Enter URL":
    image_url = st.text_input("Enter image URL")
    if image_url:
        img = load_image_from_url(image_url)
        if img:
            st.image(img, caption="Image from URL", use_column_width=True)

# Check if an image has been uploaded or provided via URL
if 'img' in locals() and img is not None:
    st.success("Image loaded successfully!")
    
    # Generate caption and audio
    caption, audio_file = generate_caption_and_audio(img)
    
    # Display the generated caption
    st.write(f"Generated Caption: {caption}")
    
    # Display audio file
    if audio_file:
        st.audio(audio_file)
else:
    st.warning("Please upload an image or enter a valid image URL.")
