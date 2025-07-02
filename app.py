import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image
import matplotlib.pyplot as plt
import pickle
import os
import gdown

# Streamlit page setup
st.set_page_config(page_title="AI Image Caption Generator", layout="centered")

# Helper to download file if not present
def download_file_from_drive(file_id, output_path):
    if not os.path.exists(output_path):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output_path, quiet=False)

# Download files from Google Drive if missing
os.makedirs("models", exist_ok=True)
download_file_from_drive("1IB5JjerMResGCUjYrQkP_lwZd3Iiq9Xr", "models/model.keras")
download_file_from_drive("1LKTQ82-ewO60MubpAPo10WTn8tEyr8Im", "models/feature_extractor.keras")
download_file_from_drive("1IgDi7SkKl0qLqPIQjDmPkphmxkT4OsDP", "models/tokenizer.pkl")

# Load models and tokenizer once
@st.cache_resource
def load_all_models():
    model = load_model("models/model.keras")
    feature_extractor = load_model("models/feature_extractor.keras")
    with open("models/tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    return model, feature_extractor, tokenizer

# Caption generation
def generate_caption(image_array, model, feature_extractor, tokenizer, max_length=34):
    image_array = np.expand_dims(image_array, axis=0)
    features = feature_extractor.predict(image_array, verbose=0)

    in_text = "startseq"
    for _ in range(max_length):
        seq = tokenizer.texts_to_sequences([in_text])[0]
        seq = pad_sequences([seq], maxlen=max_length)
        yhat = model.predict([features, seq], verbose=0)
        word_idx = np.argmax(yhat)
        word = tokenizer.index_word.get(word_idx)
        if word is None or word == "endseq":
            break
        in_text += " " + word
    return in_text.replace("startseq", "").replace("endseq", "").strip()

# Main app UI
def main():
    st.markdown("<h1 style='text-align:center;'>🧠 Image Caption Generator</h1>", unsafe_allow_html=True)
    st.write("Upload an image and generate a description using your trained deep learning model.")

    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_image:
        image = Image.open(uploaded_image).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        image_resized = image.resize((224, 224))
        img_array = np.array(image_resized) / 255.0

        model, fe_model, tokenizer = load_all_models()

        with st.spinner("🧠 Generating caption..."):
            caption = generate_caption(img_array, model, fe_model, tokenizer)

        st.markdown(f"<h3 style='color:#1f77b4;'>📝 Caption:</h3><p style='font-size:20px;'>{caption}</p>", unsafe_allow_html=True)
        st.success("Done!")

if __name__ == "__main__":
    main()
