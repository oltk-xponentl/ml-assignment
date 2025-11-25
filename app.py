import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import time
import json
import pandas as pd
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))
from sod_model import create_model

# ==========================================
# 1. VERSION CONFIGURATION
# ==========================================
MODEL_VERSIONS = {
    "v1: Baseline (128px)": {
        "exp_folder": "experiments/v1_baseline",
        "size": 128,
        "config": {"use_bn": False}
    },
    "v2: HighRes (224px)": {
        "exp_folder": "experiments/v2_highres",
        "size": 224,
        "config": {"use_bn": False}
    },
    "v2: LowRes (64px)": {
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
        "config": {"use_bn": True}
    },
    "v4: U-Net": {
        "exp_folder": "experiments/v4_unet",
        "size": 224,
        "config": {"use_bn": True, "use_skip": True}
    },
    "v5: Deep U-Net": {
        "exp_folder": "experiments/v5_deep_unet",
        "size": 224,
        "config": {"use_bn": True, "use_skip": True, "deep": True}
    },
    "v5.5: Deep U-Net with Scheduler": {
        "exp_folder": "experiments/v5.5_scheduler",
        "size": 224,
        "config": {"use_bn": True, "use_skip": True, "deep": True}
    },
    "v5.5: Scheduler @ 300 Epochs": {
        "exp_folder": "experiments/v5.5_scheduler",
        "size": 224,
        "config": {"use_bn": True, "use_skip": True, "deep": True}
    }
}

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
@st.cache_resource
def load_model_and_stats(version_name, device):
    info = MODEL_VERSIONS[version_name]
    folder = Path(info["exp_folder"])
    
    # A. Load Model
    try:
        model = create_model(**info["config"])
    except TypeError:
        # Fallback if config has keys the model doesn't accept yet
        model = create_model()

    ckpt_path = folder / "best_model.pt"
    
    if not ckpt_path.exists():
        return None, None, None, None, f"Checkpoint not found at {ckpt_path}"
        
    try:
        # Try using the helper method if it exists (for v4 compatibility)
        if hasattr(model, 'load_weights'):
            model.load_weights(ckpt_path, device)
        else:
            # Fallback for older model files
            checkpoint = torch.load(ckpt_path, map_location=device)
            if "model_state" in checkpoint:
                state_dict = checkpoint["model_state"]
            else:
                state_dict = checkpoint
            model.load_state_dict(state_dict)
            
        model.to(device)
        model.eval()
    except Exception as e:
        return None, None, None, None, f"Error loading weights: {e}"

    # B. Load Metrics
    metrics = {}
    json_path = folder / "metrics.json"
    if json_path.exists():
        with open(json_path, "r") as f:
            metrics = json.load(f)

    # C. Load History (JSON)
    history = {}
    hist_path = folder / "history.json"
    if hist_path.exists():
        with open(hist_path, "r") as f:
            history = json.load(f)

    # D. Load Static Graph (PNG)
    exp_name = folder.name
    graph_path = folder / f"{exp_name}_graph.png"
    graph_image = None
    if graph_path.exists():
        graph_image = Image.open(graph_path)
            
    return model, metrics, history, graph_image, None

def process_image(image, size):
    t = transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
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
model, metrics, history, graph_img, err = load_model_and_stats(selected_ver, device)

if err:
    st.error(err)
else:
    # --- METRICS SECTION ---
    st.markdown("### 📊 Model Performance")
    if metrics:
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("IoU", metrics.get("iou", "N/A"))
        c2.metric("F1-Score", metrics.get("f1", "N/A"))
        c3.metric("Precision", metrics.get("precision", "N/A"))
        c4.metric("Recall", metrics.get("recall", "N/A"))
        c5.metric("MAE", metrics.get("mae", "N/A"))
    else:
        st.warning("No metrics.json found for this version.")

    # --- GRAPHS SECTION ---
    with st.expander("📈 Training Curves", expanded=True):
        # Priority 1: Show the static PNG (Centered and Smaller)
        if graph_img is not None:
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.image(graph_img, caption=f"Training Graph for {selected_ver}", use_container_width=True)
        
        # Priority 2: Fallback to interactive chart if JSON exists but PNG doesn't
        elif history and "train_loss" in history:
            st.info("Static graph not found. Plotting from history.json...")
            
            spacer_l, content, spacer_r = st.columns([1, 6, 1])
            
            with content:
                h_c1, h_c2 = st.columns(2)
                
                df_loss = pd.DataFrame({
                    "Train Loss": history["train_loss"],
                    "Val Loss": history["val_loss"]
                })
                h_c1.markdown("#### Loss Curve")
                h_c1.line_chart(df_loss)
                
                if "train_iou" in history:
                    df_iou = pd.DataFrame({
                        "Train IoU": history["train_iou"],
                        "Val IoU": history["val_iou"]
                    })
                    h_c2.markdown("#### IoU Curve")
                    h_c2.line_chart(df_iou)
        else:
            st.info("No training history available for this model version.")

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