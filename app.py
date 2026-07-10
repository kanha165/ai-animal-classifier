import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# ─────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="FaunaScan AI",
    page_icon="🦁",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─────────────────────────────────────────
#  GLOBAL CSS  —  Warm Dark / Amber-Gold Theme
# ─────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

/* ══ Reset & Base ══ */
*, *::before, *::after { box-sizing: border-box; font-family: 'Inter', sans-serif; }

.stApp {
    background: #0d0b08;
    color: #e8dcc8;
}

#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding: 2.5rem 3.5rem 5rem 3.5rem;
    max-width: 1280px;
    margin: 0 auto;
}

/* ══ HERO ══ */
.hero {
    background: linear-gradient(160deg, #1a1208 0%, #231a0a 55%, #1a1510 100%);
    border: 1px solid #3d2e10;
    border-radius: 28px;
    padding: 56px 48px;
    text-align: center;
    margin-bottom: 40px;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -60%;
    left: 50%;
    transform: translateX(-50%);
    width: 80%;
    height: 180%;
    background: radial-gradient(ellipse, rgba(212,160,23,0.08) 0%, transparent 65%);
    pointer-events: none;
}
.hero::after {
    content: '';
    position: absolute;
    inset: 0;
    background:
        radial-gradient(circle at 10% 90%, rgba(180,83,9,0.06) 0%, transparent 40%),
        radial-gradient(circle at 90% 10%, rgba(120,53,15,0.05) 0%, transparent 40%);
    pointer-events: none;
}
.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    background: rgba(212,160,23,0.1);
    border: 1px solid rgba(212,160,23,0.28);
    color: #d4a017;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    padding: 7px 18px;
    border-radius: 100px;
    margin-bottom: 22px;
}
.hero-title {
    font-size: 3.4rem;
    font-weight: 900;
    background: linear-gradient(135deg, #f5e6c8 20%, #d4a017 60%, #b87333 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 16px 0;
    line-height: 1.1;
    letter-spacing: -1px;
}
.hero-sub {
    color: #7a6a52;
    font-size: 1rem;
    font-weight: 400;
    margin: 0;
    line-height: 1.7;
    max-width: 560px;
    margin: 0 auto;
}

/* ══ STAT CARDS ══ */
.stat-card {
    background: #13100a;
    border: 1px solid #2c2010;
    border-radius: 18px;
    padding: 22px 20px;
    text-align: center;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s;
}
.stat-card:hover { border-color: #d4a017; }
.stat-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, #d4a017, transparent);
    opacity: 0.5;
}
.stat-value {
    font-size: 1.7rem;
    font-weight: 800;
    color: #f0d898;
    line-height: 1;
    margin-bottom: 7px;
}
.stat-label {
    color: #4a3f2f;
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1.5px;
}

/* ══ SECTION LABELS ══ */
.section-label {
    color: #5a4f3a;
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    margin-bottom: 14px;
    display: flex;
    align-items: center;
    gap: 8px;
}
.section-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, #2c2010, transparent);
}

/* ══ FILE UPLOADER ══ */
[data-testid="stFileUploadDropzone"] {
    background: #13100a !important;
    border: 2px dashed #2c2010 !important;
    border-radius: 18px !important;
    padding: 32px !important;
    transition: all 0.25s ease !important;
}
[data-testid="stFileUploadDropzone"]:hover {
    border-color: #d4a017 !important;
    background: #1a1208 !important;
}
[data-testid="stFileUploadDropzone"] p,
[data-testid="stFileUploadDropzone"] span {
    color: #4a3f2f !important;
}
[data-testid="stFileUploadDropzone"] svg {
    fill: #4a3f2f !important;
}

/* ══ IMAGE ══ */
[data-testid="stImage"] img {
    border-radius: 18px !important;
    border: 1px solid #2c2010 !important;
}

/* ══ METRICS ══ */
[data-testid="stMetric"] {
    background: #13100a;
    border: 1px solid #2c2010;
    border-radius: 12px;
    padding: 12px 14px;
}
[data-testid="stMetricLabel"] { color: #4a3f2f !important; font-size: 0.72rem !important; }
[data-testid="stMetricValue"] { color: #f0d898 !important; font-size: 1rem !important; font-weight: 700 !important; }

/* ══ RESULT CARD ══ */
.result-card {
    background: linear-gradient(160deg, #1a1208, #1f1a0d);
    border: 1px solid #8B6914;
    border-radius: 22px;
    padding: 36px 32px;
    text-align: center;
    position: relative;
    overflow: hidden;
    margin-bottom: 20px;
}
.result-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, transparent 0%, #d4a017 50%, transparent 100%);
}
.result-card::after {
    content: '';
    position: absolute;
    inset: 0;
    background: radial-gradient(ellipse at 50% -10%, rgba(212,160,23,0.07) 0%, transparent 60%);
    pointer-events: none;
}
.result-tag {
    display: inline-block;
    background: rgba(212,160,23,0.1);
    border: 1px solid rgba(212,160,23,0.25);
    color: #d4a017;
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 3px;
    text-transform: uppercase;
    padding: 5px 14px;
    border-radius: 100px;
    margin-bottom: 18px;
}
.result-name {
    font-size: 3rem;
    font-weight: 900;
    background: linear-gradient(135deg, #f5e6c8, #d4a017);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.05;
    margin-bottom: 28px;
    letter-spacing: -0.5px;
}
.conf-header {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    margin-bottom: 10px;
}
.conf-label { color: #5a4f3a; font-size: 0.8rem; font-weight: 500; }
.conf-pct   { color: #d4a017; font-size: 1.1rem; font-weight: 800; }
.conf-track {
    background: #1f1a0d;
    border: 1px solid #2c2010;
    border-radius: 100px;
    height: 10px;
    overflow: hidden;
}
.conf-fill {
    height: 100%;
    border-radius: 100px;
    background: linear-gradient(90deg, #92400e, #d97706, #fbbf24);
    box-shadow: 0 0 12px rgba(217,119,6,0.4);
}

/* ══ TOP-3 BAR ══ */
.top3-item { margin-bottom: 18px; }
.top3-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 7px;
}
.top3-name { color: #c8b99a; font-size: 0.85rem; font-weight: 500; }
.top3-pct  { color: #6b5e48; font-size: 0.8rem; font-weight: 600; }
.top3-track {
    background: #1a1208;
    border-radius: 100px;
    height: 6px;
    overflow: hidden;
}

/* ══ EMPTY STATE ══ */
.empty-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 64px 24px;
    border: 2px dashed #2c2010;
    border-radius: 22px;
    background: #0d0b08;
}
.empty-icon { font-size: 4rem; opacity: 0.35; margin-bottom: 18px; }
.empty-text { color: #3d3525; font-size: 0.9rem; text-align: center; line-height: 1.6; }

/* ══ ERROR ══ */
.err-box {
    background: #150808;
    border: 1px solid #5c1a1a;
    border-radius: 16px;
    padding: 28px;
    text-align: center;
    color: #f87171;
    font-size: 0.88rem;
    line-height: 1.6;
}

/* ══ SPINNER ══ */
.stSpinner > div { border-top-color: #d4a017 !important; }

/* ══ DIVIDER ══ */
hr { border-color: #1f1a0d !important; }

/* ══ FOOTER ══ */
.footer {
    margin-top: 56px;
    padding-top: 24px;
    border-top: 1px solid #1f1a0d;
    text-align: center;
    color: #2c2010;
    font-size: 0.75rem;
    letter-spacing: 0.5px;
}
.footer strong { color: #3d3525; }

/* ══ Scrollbar ══ */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0d0b08; }
::-webkit-scrollbar-thumb { background: #2c2010; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────
#  LOAD MODEL & CLASSES
# ─────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    return tf.keras.models.load_model("multi_animal_model.h5")

@st.cache_resource(show_spinner=False)
def load_classes():
    return np.load("class_names.npy", allow_pickle=True)

model       = load_model()
class_names = load_classes()
IMG_SIZE    = 150


# ─────────────────────────────────────────
#  PREPROCESSING
# ─────────────────────────────────────────
def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.resize((IMG_SIZE, IMG_SIZE))
    arr   = np.array(image) / 255.0
    return np.expand_dims(arr, axis=0)


# ─────────────────────────────────────────
#  HERO SECTION
# ─────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-badge">� &nbsp; Deep Learning &nbsp;·&nbsp; CNN &nbsp;·&nbsp; Wildlife AI</div>
    <h1 class="hero-title">FaunaScan AI</h1>
    <p class="hero-sub">
        Drop any animal photograph and our convolutional neural network
        will identify the species instantly — with ranked confidence scores.
    </p>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────
#  STATS ROW
# ─────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(f"""<div class="stat-card">
        <div class="stat-value">{len(class_names)}</div>
        <div class="stat-label">Species Classes</div>
    </div>""", unsafe_allow_html=True)

with c2:
    st.markdown("""<div class="stat-card">
        <div class="stat-value">150 px</div>
        <div class="stat-label">Input Resolution</div>
    </div>""", unsafe_allow_html=True)

with c3:
    st.markdown("""<div class="stat-card">
        <div class="stat-value">CNN</div>
        <div class="stat-label">Architecture</div>
    </div>""", unsafe_allow_html=True)

with c4:
    st.markdown("""<div class="stat-card">
        <div class="stat-value">Real-Time</div>
        <div class="stat-label">Inference</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<div style='height:32px'></div>", unsafe_allow_html=True)


# ─────────────────────────────────────────
#  MAIN COLUMNS
# ─────────────────────────────────────────
left, right = st.columns([1, 1], gap="large")

# ──────────────
#  LEFT — Upload
# ──────────────
with left:
    st.markdown('<div class="section-label">📂 &nbsp; Input Image</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, use_container_width=True)

        w, h    = image.size
        size_kb = len(uploaded_file.getvalue()) / 1024
        m1, m2, m3 = st.columns(3)
        m1.metric("Width",  f"{w} px")
        m2.metric("Height", f"{h} px")
        m3.metric("Size",   f"{size_kb:.1f} KB")


# ───────────────
#  RIGHT — Result
# ───────────────
with right:
    st.markdown('<div class="section-label">🎯 &nbsp; Prediction Result</div>', unsafe_allow_html=True)

    if uploaded_file:
        with st.spinner("Scanning species…"):
            try:
                processed    = preprocess_image(image)
                preds_tensor = model(processed, training=False)
                preds        = preds_tensor.numpy()

                pred_idx    = int(np.argmax(preds))
                confidence  = float(np.max(preds))
                animal_name = str(class_names[pred_idx]).replace("_", " ").title()
                bar_pct     = int(confidence * 100)

                # ── Main result card ──
                st.markdown(f"""
                <div class="result-card">
                    <div class="result-tag">✦ Identified Species</div>
                    <div class="result-name">{animal_name}</div>
                    <div class="conf-header">
                        <span class="conf-label">Match Confidence</span>
                        <span class="conf-pct">{confidence*100:.1f}%</span>
                    </div>
                    <div class="conf-track">
                        <div class="conf-fill" style="width:{bar_pct}%"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # ── Top-3 breakdown ──
                st.markdown('<div class="section-label" style="margin-top:24px;">📊 &nbsp; Top 3 Predictions</div>',
                            unsafe_allow_html=True)

                top3 = np.argsort(preds[0])[::-1][:3]
                bar_colors = ["linear-gradient(90deg,#92400e,#fbbf24)",
                              "linear-gradient(90deg,#78350f,#f59e0b)",
                              "linear-gradient(90deg,#451a03,#d97706)"]
                medals = ["🥇", "🥈", "🥉"]

                for rank, idx in enumerate(top3):
                    name  = str(class_names[idx]).replace("_", " ").title()
                    score = float(preds[0][idx]) * 100
                    st.markdown(f"""
                    <div class="top3-item">
                        <div class="top3-row">
                            <span class="top3-name">{medals[rank]} &nbsp;{name}</span>
                            <span class="top3-pct">{score:.1f}%</span>
                        </div>
                        <div class="top3-track">
                            <div style="width:{int(score)}%;height:100%;border-radius:100px;
                                        background:{bar_colors[rank]};"></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            except Exception as e:
                st.markdown(f"""
                <div class="err-box">
                    ⚠️ &nbsp; Prediction failed<br>
                    <small style="opacity:0.7;">{e}</small>
                </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-icon">�</div>
            <div class="empty-text">
                Upload an animal image on the left<br>
                to see the AI prediction here.
            </div>
        </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────
#  FOOTER
# ─────────────────────────────────────────
st.markdown("""
<div class="footer">
    <strong>FaunaScan AI v3.1</strong> &nbsp;·&nbsp;
    Built with Streamlit &amp; TensorFlow &nbsp;·&nbsp;
    CNN Image Classifier
</div>
""", unsafe_allow_html=True)
