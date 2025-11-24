import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import time
import json
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))
from sod_model import create_model

# ==========================================
# 1. VERSION CONFIGURATION
# ==========================================
MODEL_VERSIONS = {
    "v1: Baseline": {
        "exp_folder": "experiments/v1_baseline",
        "size": 128,
        "config": {"use_bn": False}
    },
    "v2: HighRes": {
        "exp_folder": "experiments/v2_highres",
        "size": 224,
        "config": {"use_bn": False}
    },
    "v2: LowRes": {
        "exp_folder": "experiments/v2_lowres",
        "size": 64,
        "config": {"use_bn": False}
    },
    "v2.5: Augmentation": {
    "exp_folder": "experiments/v2.5_augmentation",
    "size": 224,
    "config": {"use_bn": False}
    },
    "v3: Batch Norm": {
        "exp_folder": "experiments/v3_batchnorm",
        "size": 224,
        "config": {"use_bn": True} # 
    }
}

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
@st.cache_resource
def load_model_and_metrics(version_name, device):
    info = MODEL_VERSIONS[version_name]
    folder = Path(info["exp_folder"])
    
    # A. Load Model
    model = create_model(**info["config"])
    ckpt_path = folder / "best_model.pt"
    
    if not ckpt_path.exists():
        return None, None, f"Checkpoint not found at {ckpt_path}"
        
    try:
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.to(device)
        model.eval()
    except Exception as e:
        return None, None, f"Error loading weights: {e}"

    # B. Load Metrics JSON
    metrics = {}
    json_path = folder / "metrics.json"
    if json_path.exists():
        with open(json_path, "r") as f:
            metrics = json.load(f)
            
    return model, metrics, None

def process_image(image, size):
    t = transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor()])
    return t(image).unsqueeze(0)

def postprocess_mask(logits, original_size):
    probs = torch.sigmoid(logits)
    mask = (probs > 0.5).float()
    mask = F.interpolate(mask, size=original_size[::-1], mode="nearest")
    return mask.squeeze().cpu().numpy()

# ==========================================
# 3. UI LAYOUT
# ==========================================
st.set_page_config(page_title="SOD Project Demo", layout="wide")
st.title("Salient Object Detection Project")

# Sidebar
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
st.sidebar.markdown(f"**Device:** `{device}`")
selected_ver = st.sidebar.selectbox("Select Version", list(MODEL_VERSIONS.keys()))

# Load Assets
model, metrics, err = load_model_and_metrics(selected_ver, device)

if err:
    st.error(err)
else:
    # --- METRICS SECTION ---
    st.markdown("### 📊 Model Performance (Test Set)")
    if metrics:
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("IoU", metrics.get("iou", "N/A"))
        c2.metric("F1-Score", metrics.get("f1", "N/A"))
        c3.metric("Precision", metrics.get("precision", "N/A"))
        c4.metric("Recall", metrics.get("recall", "N/A"))
        c5.metric("MAE", metrics.get("mae", "N/A"))
        
        with st.expander("See full training details"):
            st.json(metrics)
    else:
        st.warning("No metrics.json found for this version.")

    st.markdown("---")
    
    # --- DEMO SECTION ---
    st.markdown("### 🖼️ Live Inference")
    uploaded_file = st.sidebar.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
    
    if uploaded_file:
        img = Image.open(uploaded_file).convert("RGB")
        info = MODEL_VERSIONS[selected_ver]
        
        # Inference
        input_t = process_image(img, info["size"]).to(device)
        start = time.time()
        with torch.no_grad():
            logits = model(input_t)
        infer_time = (time.time() - start) * 1000
        
        # Visuals
        mask = postprocess_mask(logits, img.size)
        
        c1, c2, c3 = st.columns(3)
        c1.image(img, "Original", use_container_width=True)
        c2.image(mask, f"Prediction ({infer_time:.1f}ms)", use_container_width=True, clamp=True)
        
        # Red Overlay
        overlay = np.array(img)
        red = np.zeros_like(overlay); red[:,:,0] = 255
        overlay[mask == 1] = (overlay[mask == 1]*0.5 + red[mask == 1]*0.5).astype(np.uint8)
        c3.image(overlay, "Overlay", use_container_width=True)
    else:
        st.info("Upload an image in the sidebar to test the model.")