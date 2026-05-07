import streamlit as st
import requests
from PIL import Image
import io

# --- PAGE CONFIG (Professional & Clean) ---
st.set_page_config(
    page_title="Animal AI | Enterprise Classifier",
    page_icon="🐕",
    layout="centered",
)

# --- CLEAN CUSTOM CSS ---
st.markdown("""
    <style>
    /* Global Reset */
    .main {
        background-color: #f8f9fa;
    }
    
    /* Typography */
    h1, h2, h3 {
        color: #1e293b;
        font-family: 'Inter', -apple-system, sans-serif;
        font-weight: 700;
    }

    /* Standard Card Style */
    .status-card {
        background-color: white;
        padding: 24px;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }

    /* Professional Button */
    .stButton>button {
        width: 100%;
        background-color: #2563eb !important;
        color: white !important;
        border-radius: 8px !important;
        border: none !important;
        padding: 10px 0px !important;
        font-weight: 600 !important;
        transition: all 0.2s ease;
    }
    
    .stButton>button:hover {
        background-color: #1d4ed8 !important;
        box-shadow: 0 10px 15px -3px rgba(37, 99, 235, 0.2);
    }

    /* Metric Styling */
    .metric-text {
        font-size: 2rem;
        font-weight: 800;
        color: #2563eb;
    }

    /* Hide unnecessary elements */
    footer {visibility: hidden;}
    #MainMenu {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# --- SIDEBAR (Technical Info) ---
with st.sidebar:
    st.title("⚙️ System Info")
    st.info("**Model:** MobileNetV2\n\n**Accuracy:** 94.2%\n\n**Input:** 150x150 RGB")
    st.divider()
    st.markdown("### Developer\n**Kanha Patidar**\n*AI/ML Engineer*")
    st.caption("© 2026 Animal AI Systems")

# --- MAIN UI ---
st.title("🐾 Animal Classification System")
st.markdown("Upload an image for high-precision animal identification using Deep Learning.")

# Upload Section
with st.container():
    uploaded_file = st.file_uploader("Drop image here or click to browse", type=["jpg", "png", "jpeg"])

if uploaded_file:
    col1, col2 = st.columns([1, 1], gap="medium")
    
    with col1:
        img = Image.open(uploaded_file)
        st.image(img, caption="Target Image", use_container_width=True)
        predict_btn = st.button("Run Inference")

    with col2:
        if predict_btn:
            with st.spinner('Running neural network...'):
                try:
                    # API Request
                    buf = io.BytesIO()
                    img.save(buf, format='JPEG')
                    files = {"file": ("image.jpg", buf.getvalue(), "image/jpeg")}
                    
                    response = requests.post("http://127.0.0.1:9000/predict", files=files)
                    
                    if response.status_code == 200:
                        data = response.json()
                        animal = data['prediction'].capitalize()
                        conf = data['confidence']

                        # Result Display
                        st.markdown('<div class="status-card">', unsafe_allow_html=True)
                        st.subheader("Results")
                        st.markdown(f"Detected: <span class='metric-text'>{animal}</span>", unsafe_allow_html=True)
                        
                        # Confidence Bar
                        st.write(f"Confidence Score: {conf*100:.1f}%")
                        st.progress(conf)
                        
                        if conf > 0.85:
                            st.success("Verified Match")
                        else:
                            st.warning("Low Confidence Match")
                        st.markdown('</div>', unsafe_allow_html=True)
                    else:
                        st.error("Server Error: Unable to process prediction.")
                except Exception as e:
                    st.error(f"Error: {e}")
        else:
            st.info("Click 'Run Inference' to analyze the image.")

# --- FOOTER (Minimal) ---
st.divider()
st.caption("Animal AI Enterprise | Robust Computer Vision Solutions")