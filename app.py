import streamlit as st
import numpy as np
import tensorflow as tf
import pickle
from pathlib import Path

# =====================================================
# Page Config
# =====================================================
st.set_page_config(
    page_title="LSTM Text Generator",
    page_icon="🧠",
    layout="centered",
)

# =====================================================
# Custom CSS
# =====================================================
st.markdown(
    """
    <style>
        .main-title {
            font-size: 3rem;
            font-weight: 800;
            text-align: center;
            margin-bottom: 0.5rem;
        }
        .subtitle {
            text-align: center;
            font-size: 1.1rem;
            color: #aaaaaa;
            margin-bottom: 2rem;
        }
        .card {
            background-color: #111827;
            padding: 1.5rem;
            border-radius: 14px;
            margin-top: 1.5rem;
        }
        .footer {
            text-align: center;
            margin-top: 3rem;
            font-size: 0.9rem;
            color: #888888;
        }
        textarea {
            font-family: monospace !important;
            font-size: 0.95rem !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# =====================================================
# Header
# =====================================================
st.markdown('<div class="main-title">🧠 LSTM Text Generator</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Generate Shakespeare-style text using a trained LSTM language model</div>',
    unsafe_allow_html=True,
)

# =====================================================
# Constants & Paths
# =====================================================
MODEL_PATH = Path("lstm_text_generator.h5")
MAPPING_PATH = Path("char_mappings.pkl")
SEQ_LEN = 40
EPSILON = 1e-8

# =====================================================
# Load Model & Mappings
# =====================================================
@st.cache_resource(show_spinner="Loading model...")
def load_model_and_mappings():
    if not MODEL_PATH.exists():
        st.error("❌ Model file not found.")
        st.stop()

    if not MAPPING_PATH.exists():
        st.error("❌ Character mapping file not found.")
        st.stop()

    model = tf.keras.models.load_model(MODEL_PATH)

    with open(MAPPING_PATH, "rb") as f:
        char_to_idx, idx_to_char = pickle.load(f)

    return model, char_to_idx, idx_to_char


model, char_to_idx, idx_to_char = load_model_and_mappings()

# =====================================================
# Sampling
# =====================================================
def sample_with_temperature(preds, temperature):
    preds = np.asarray(preds).astype(np.float64)
    preds = np.clip(preds, EPSILON, 1.0)

    log_preds = np.log(preds) / temperature
    exp_preds = np.exp(log_preds)
    probs = exp_preds / np.sum(exp_preds)

    return np.random.choice(len(probs), p=probs)


# =====================================================
# Text Generation
# =====================================================
def generate_text(seed_text, length, temperature):
    seed_text = seed_text.lower()
    generated = seed_text
    current_seed = seed_text

    for _ in range(length):
        seq = [char_to_idx.get(c, 0) for c in current_seed]
        seq = np.array(seq).reshape(1, -1)

        preds = model.predict(seq, verbose=0)[0]
        next_idx = sample_with_temperature(preds, temperature)
        next_char = idx_to_char[next_idx]

        generated += next_char
        current_seed = current_seed[1:] + next_char

    return generated


# =====================================================
# Sidebar Controls
# =====================================================
with st.sidebar:
    st.header("⚙️ Controls")

    seed_text = st.text_area(
        "✍️ Seed Text",
        value="to be or not to be that is the question ",
        height=120,
    )

    text_length = st.slider(
        "📏 Generated Length",
        min_value=100,
        max_value=600,
        value=300,
        step=50,
    )

    temperature = st.slider(
        "🔥 Temperature",
        min_value=0.2,
        max_value=1.5,
        value=0.8,
        step=0.1,
        help="Lower = safer | Higher = more creative",
    )

    generate_btn = st.button("🚀 Generate Text", use_container_width=True)


# =====================================================
# Main Output Area
# =====================================================
if generate_btn:
    if len(seed_text) < SEQ_LEN:
        st.warning(f"Seed text must be at least {SEQ_LEN} characters long.")
    else:
        with st.spinner("🧠 LSTM is thinking..."):
            output = generate_text(
                seed_text=seed_text[-SEQ_LEN:],
                length=text_length,
                temperature=temperature,
            )

        st.markdown("### 📜 Generated Text")
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.text_area("", output, height=320)
        st.markdown("</div>", unsafe_allow_html=True)


# =====================================================
# Footer
# =====================================================
st.markdown(
    '<div class="footer">Built with ❤️ using Streamlit & TensorFlow</div>',
    unsafe_allow_html=True,
)
