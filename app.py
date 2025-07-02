import streamlit as st
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import pickle
from PIL import Image
import io

# Page configuration
st.set_page_config(page_title="AI Image Caption Generator", layout="centered")

# Load models & tokenizer once
@st.cache_resource
def load_models():
    caption_model = load_model("models/model.keras")
    feature_extractor = load_model("models/feature_extractor.keras")
    with open("models/tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    return caption_model, feature_extractor, tokenizer

# Generate caption
def generate_caption(img_array, model, fe_model, tokenizer, max_length=34):
    img_array = np.expand_dims(img_array, axis=0)
    image_features = fe_model.predict(img_array, verbose=0)
    if image_features.ndim > 1:
        image_features = image_features[0]

    in_text = "startseq"
    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([np.expand_dims(image_features, axis=0), sequence], verbose=0)
        yhat_index = np.argmax(yhat)
        word = tokenizer.index_word.get(yhat_index)
        if word is None or word == "endseq":
            break
        in_text += " " + word
    return in_text.replace("startseq", "").replace("endseq", "").strip()

# App UI
def main():
    st.markdown("<h1 style='text-align: center; color: #4A4A4A;'>🧠 AI Image Caption Generator</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 18px;'>Upload an image, and our deep learning model will describe it for you.</p>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        # Preview image
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Convert image to array
        img = image.resize((224, 224))
        img_array = np.array(img) / 255.0

        # Load models
        caption_model, feature_extractor, tokenizer = load_models()

        with st.spinner("🧠 Generating caption..."):
            caption = generate_caption(img_array, caption_model, feature_extractor, tokenizer)

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown(f"<h3 style='color: #1f77b4;'>📝 Generated Caption:</h3><p style='font-size: 20px; font-style: italic;'>{caption}</p>", unsafe_allow_html=True)

        st.success("✅ Caption generated successfully!")

if __name__ == "__main__":
    main()
