from __future__ import annotations

import json

import numpy as np
import streamlit as st
from PIL import Image

from digit_core import (CLASS_NAMES, CNN_PATH, RESULTS_DIR, VIT_PATH, ImageRejected,
                        load_checkpoint, open_upload, predict_both, prepare_image)

st.set_page_config(page_title="Handwritten Digit Prediction", page_icon="🔢", layout="wide")


@st.cache_resource(show_spinner="Loading saved models ...")
def load_models():
    """Restore both trained models from disk (no training happens in the app)."""
    missing = [p.name for p in (CNN_PATH, VIT_PATH) if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing checkpoint(s): {', '.join(missing)}. "
                                "Run `python train.py` first.")
    cnn, cnn_ckpt = load_checkpoint(CNN_PATH)
    vit, vit_ckpt = load_checkpoint(VIT_PATH)
    return cnn, vit, cnn_ckpt, vit_ckpt


def enlarge(image01: np.ndarray, scale: int = 32) -> Image.Image:
    """Nearest-neighbour enlargement so each of the 64 pixels is visible."""
    img = Image.fromarray((image01 * 255).round().astype(np.uint8))
    return img.resize((8 * scale, 8 * scale), Image.Resampling.NEAREST)


def prob_chart(probs: np.ndarray):
    return {"digit": CLASS_NAMES, "probability": [float(p) for p in probs]}


st.title("🔢 Handwritten Digit Prediction")
st.caption("CNN vs. Vision Transformer, both trained on the 8x8 sklearn digits dataset.")

try:
    cnn, vit, cnn_ckpt, vit_ckpt = load_models()
except Exception as exc:  # missing or corrupt checkpoint: explain instead of crashing
    st.error(f"Could not load the saved models: {exc}")
    st.stop()

with st.sidebar:
    st.header("Saved models")
    for name, ck in (("CNN", cnn_ckpt), ("ViT", vit_ckpt)):
        st.markdown(f"**{name}** - input `{ck['input_shape']}`  \n"
                    f"test accuracy {ck['test_accuracy']:.3f} · macro F1 {ck['test_macro_f1']:.3f}")
    metrics_path = RESULTS_DIR / "metrics.json"
    if metrics_path.exists():
        base = json.loads(metrics_path.read_text())["results"]["Majority baseline"]["test"]
        st.markdown(f"**Majority baseline** - test accuracy {base['accuracy']:.3f}")
    if (RESULTS_DIR / "confusion_matrices.png").exists():
        with st.expander("Test confusion matrices"):
            st.image(str(RESULTS_DIR / "confusion_matrices.png"))
    st.divider()
    st.markdown("**Tips for photos**: one digit, dark pen on plain light paper, "
                "fill most of the frame, avoid shadows.")

uploaded = st.file_uploader("Upload a PNG or JPEG photo of ONE handwritten digit",
                            type=["png", "jpg", "jpeg"])

if uploaded is None:
    st.info("No image uploaded yet. Choose a PNG/JPEG file to get predictions.")
    st.stop()

try:
    photo = open_upload(uploaded.getvalue())
    image01, steps = prepare_image(photo, return_steps=True)
except ImageRejected as exc:
    st.error(f"Image rejected: {exc}")
    st.stop()
except Exception as exc:  # anything unexpected while decoding - never crash the app
    st.error(f"Image rejected: the file could not be processed ({exc.__class__.__name__}).")
    st.stop()

preds = predict_both(cnn, vit, image01)

col_photo, col_prep, col_pred = st.columns([1.2, 1, 1.3])
with col_photo:
    st.subheader("Uploaded photo")
    st.image(photo, width="stretch")
with col_prep:
    st.subheader("Prepared 8x8 input")
    st.image(enlarge(image01), caption="Exactly what both models receive (values 0-1)")
    with st.expander("Pixel values"):
        st.dataframe(np.round(image01, 2), width="stretch")
    with st.expander("Preparation steps"):
        st.image(steps["grayscale"], caption="1. Grayscale", width=180)
        st.image(steps["cleaned_crop"], caption="2. Inverted, cleaned and cropped strokes", width=180)
        st.image(steps["centred"], caption="3. Centred in a square", width=180)
with col_pred:
    st.subheader("Predictions")
    c1, c2 = st.columns(2)
    for col, name in ((c1, "CNN"), (c2, "ViT")):
        digit, probs = preds[name]
        col.metric(f"{name} predicts", CLASS_NAMES[digit], f"{probs[digit]:.0%} confident",
                   delta_color="off")
    if preds["CNN"][0] == preds["ViT"][0]:
        st.success(f"Both models agree: **{CLASS_NAMES[preds['CNN'][0]]}**")
    else:
        st.warning("The two models disagree - the photo may be ambiguous or unlike the training digits.")
    for name in ("CNN", "ViT"):
        st.caption(f"{name} class probabilities")
        st.bar_chart(prob_chart(preds[name][1]), x="digit", y="probability", height=160)
