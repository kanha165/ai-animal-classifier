import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# =========================================
# PAGE CONFIG
# =========================================
st.set_page_config(
    page_title="FaunaScan AI",
    page_icon="🌿",
    layout="wide"
)

# =========================================
# CUSTOM CSS
# =========================================
st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg, #fdfbfb 0%, #ebedee 100%);
}

h1 {
    color: #2c3e50;
    font-weight: 800;
    text-align: center;
}

/* Updated Streamlit CSS selector for modern progress bars */
div[data-testid="stProgressBar"] > div > div > div {
    background-color: #2ecc71 !important;
}

.result-box {
    background-color: #f0fdf4;
    padding: 25px;
    border-radius: 15px;
    border: 1px solid #bbf7d0;
}

</style>
""", unsafe_allow_html=True)

# =========================================
# TITLE
# =========================================
st.markdown(
    "<h1>🌿 AI Animal Classifier</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p style='text-align:center; color:#7f8c8d;'>Upload an animal image for instant AI prediction.</p>",
    unsafe_allow_html=True
)

st.divider()

# =========================================
# LOAD MODEL (Thread-Safe Wrapper)
# =========================================
@st.cache_resource
def load_model():
    # Loading the model normally into memory
    return tf.keras.models.load_model("multi_animal_model.h5")

model = load_model()

# =========================================
# LOAD CLASS NAMES
# =========================================
@st.cache_resource
def load_classes():
    return np.load("class_names.npy", allow_pickle=True)

class_names = load_classes()

# =========================================
# IMAGE SETTINGS
# =========================================
IMG_SIZE = 150

# =========================================
# IMAGE PREPROCESSING
# =========================================
def preprocess_image(image):
    # Resize image
    image = image.resize((IMG_SIZE, IMG_SIZE))

    # Convert image to array
    image = np.array(image)

    # Normalize image
    image = image / 255.0

    # Add batch dimension
    image = np.expand_dims(image, axis=0)

    return image

# =========================================
# MAIN LAYOUT
# =========================================
col1, col2 = st.columns([1, 1], gap="large")

# =========================================
# LEFT SIDE
# =========================================
with col1:
    st.subheader("📸 Upload Image")

    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")

        st.image(
            image,
            caption="Uploaded Image",
            use_container_width=True
        )

# =========================================
# RIGHT SIDE
# =========================================
with col2:
    st.subheader("🎯 Prediction Result")

    if uploaded_file:
        with st.spinner("Analyzing Image..."):
            try:
                # preprocess image
                processed_image = preprocess_image(image)

                # CRITICAL FIX: Explicitly invoke model via __call__ method instead of .predict()
                # This bypasses the multi-threading graph lock tracking issues in Streamlit
                predictions_tensor = model(processed_image, training=False)
                predictions = predictions_tensor.numpy()

                # get highest prediction
                predicted_index = np.argmax(predictions)

                # confidence score
                confidence = float(np.max(predictions))

                # animal name
                animal_name = class_names[predicted_index]

                # =========================================
                # SUCCESS MESSAGE
                # =========================================
                st.success("Analysis Complete!")

                # =========================================
                # RESULT CARD
                # =========================================
                st.markdown(
                    f"""
                    <div class="result-box">
                        <h4 style="margin:0; color:#166534; font-size:0.9rem;">
                            IDENTIFIED SPECIES
                        </h4>
                        <h1 style="margin-top:10px; margin-bottom:10px; text-align:left; color:#15803d;">
                            {str(animal_name).upper()}
                        </h1>
                        <hr style="border:0.5px solid #bbf7d0;">
                        <p style="margin:0; color:#166534; font-size:18px;">
                            <b>Match Probability:</b> {confidence * 100:.2f}%
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                # progress bar
                st.write("")
                st.progress(confidence)

            except Exception as e:
                st.error(f"Prediction Error: {e}")
    else:
        st.info("Please upload an image to see prediction.")

# =========================================
# FOOTER
# =========================================
st.markdown(
    """
    <br><br>
    <p style='text-align:center; color:#bdc3c7; font-size:0.8rem;'>
    FaunaScan v2.5 • Streamlit AI Powered
    </p>
    """,
    unsafe_allow_html=True
)
