import streamlit as st
import requests
from PIL import Image
import io

# --- Page Configuration ---
st.set_page_config(
    page_title="FaunaScan AI | Cloud Intelligence",
    page_icon="🌿",
    layout="wide"
)

# --- Custom Styling (Soft & Elegant Theme) ---
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #fdfbfb 0%, #ebedee 100%);
    }
    .main-card {
        background: white;
        padding: 30px;
        border-radius: 20px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.05);
    }
    h1 {
        color: #2c3e50;
        font-weight: 800;
        text-align: center;
    }
    .stProgress > div > div > div > div {
        background-color: #2ecc71;
    }
    </style>
    """, unsafe_allow_html=True)


# --- Introduction ---
st.markdown("<h1>🌿 AI Animal  Classifier</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#7f8c8d;'>Upload an animal image to get instant identification from our cloud API.</p>", unsafe_allow_html=True)
st.divider()

# --- Main Layout ---
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("📸 Upload Section")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        # Convert to RGB to avoid transparency errors
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)

with col2:
    st.subheader("🎯 Prediction Result")
    
    if uploaded_file:
        with st.spinner("Analyzing with API..."):
            try:
                # --- API INTEGRATION ---
                # Placeholder URL: Replace with your actual API endpoint
                API_URL = "http://127.0.0.1:8000/predict" 
                
                # Preparing image for API
                buf = io.BytesIO()
                image.save(buf, format="JPEG")
                byte_im = buf.getvalue()
                
                files = {"file": (uploaded_file.name, byte_im, "image/jpeg")}
                response = requests.post(API_URL, files=files)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # Mapping your specific API response keys
                    animal_name = data.get("prediction", "Unknown")
                    raw_confidence = data.get("confidence", 0)
                    
                    # Convert 0-1 range to 0-100%
                    display_confidence = raw_confidence * 100

                    # --- Success UI ---
                    st.success(f"Analysis Complete!")
                    
                    st.markdown(f"""
                        <div style="background-color: #f0fdf4; padding: 25px; border-radius: 15px; border: 1px solid #bbf7d0;">
                            <h4 style="margin:0; color:#166534; font-size: 0.9rem;">IDENTIFIED SPECIES</h4>
                            <h1 style="margin:0; text-align:left; color:#15803d;">{animal_name.upper()}</h1>
                            <hr style="border: 0.5px solid #bbf7d0;">
                            <p style="margin:0; color:#166534;"><b>Match Probability:</b> {display_confidence:.2f}%</p>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # Confidence Bar
                    st.write("")
                    st.progress(raw_confidence if raw_confidence <= 1 else 1.0)
                    
                else:
                    st.error(f"API Error {response.status_code}: Could not fetch prediction.")
                    
            except Exception as e:
                st.error(f"Connection Error: {e}")
                st.info("Check if your API server is running and the URL is correct.")
    else:
        st.info("Please upload an image to see the intelligence report.")

# --- Footer ---
st.markdown("<br><br><p style='text-align:center; color:#bdc3c7; font-size:0.8rem;'>FaunaScan v2.5 • Cloud API Powered</p>", unsafe_allow_html=True)
