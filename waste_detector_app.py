import streamlit as st
import torch
from PIL import Image
import numpy as np
import os
import time

# Page config
st.set_page_config(page_title="♻️ Waste Classifier", layout="wide")

# Custom CSS to center images and adjust column layout
st.markdown("""
    <style>
        .centered-image img {
            display: block;
            margin-left: auto;
            margin-right: auto;
            max-width: 100%;
            height: auto;
        }
        @media screen and (max-width: 768px) {
            .responsive-cols {
                flex-direction: column !important;
            }
        }
    </style>
""", unsafe_allow_html=True)

# Title
st.title("🗑 Waste Detection App")
st.markdown("""
Upload an image of waste, and this application will detect and classify it into categories like **General**, **Plastic**, **Glass**, **Metal**, and more using an AI model.
""")

# Load the model
@st.cache_resource
def load_model():
    return torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', force_reload=False)

model = load_model()

# Upload image
uploaded_file = st.file_uploader("📤 Upload a waste image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Read and prepare image
    original_image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(original_image)

    # Run detection
    with st.spinner("🔍 Detecting..."):
        results = model(image_np)
        results.render()
        result_image = Image.fromarray(results.ims[0])

        # Save result
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        output_path = os.path.join("runs", f"waste_detect_{timestamp}.jpg")
        os.makedirs("runs", exist_ok=True)
        result_image.save(output_path)

    # Display in responsive columns
    cols = st.columns(2)
    with cols[0]:
        st.markdown("**Original Image**")
        st.image(original_image, use_column_width=True, caption=None)
    with cols[1]:
        st.markdown("**Detected Waste Items**")
        st.image(result_image, use_column_width=True, caption=None)

    # Download button below columns
    st.success(f"✅ Processed image saved to `{output_path}`")
    with open(output_path, "rb") as f:
        st.download_button("⬇️ Download Processed Image", f, file_name=f"processed_{uploaded_file.name}", mime="image/jpeg")
