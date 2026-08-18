"""
ReSAGE — AI-Based Semiconductor Image Restoration
Streamlit Demonstration Application for Project Video & Evaluation
"""

import os
import random
import numpy as np
import streamlit as st

# ============================================================
# PATH CONFIGURATION
# ============================================================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Dataset paths
TEST_NOISY_DIR = os.path.join(PROJECT_ROOT, "data", "test", "NoisyLR")
TRAIN_NOISY_DIR = os.path.join(PROJECT_ROOT, "data", "train", "NoisyLR")
GT_DIR = os.path.join(PROJECT_ROOT, "data", "train", "GT")

# Joint SAE Predictions path
PRED_DIR = os.path.join(PROJECT_ROOT, "outputs", "final_predictions")

# Custom CSS path
CSS_PATH = os.path.join(PROJECT_ROOT, "demo", "styles.css")


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="ReSAGE | AI Semiconductor Image Restoration",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# LOAD CUSTOM CSS
# ============================================================

def load_css(css_file):
    if os.path.exists(css_file):
        with open(css_file, "r") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css(CSS_PATH)


# ============================================================
# APPLICATION HEADER
# ============================================================

st.markdown(
    """
    <div class="title-container">
        <div class="main-title">ReSAGE — AI-Based Semiconductor Image Restoration</div>
        <div class="main-subtitle">
            Degraded Image → Joint SAE Feature Restoration → High-Resolution Reconstruction
        </div>
        <div>
            <span class="highlight-badge">2× Upscaling</span>
            <span class="highlight-badge">Sparse Autoencoder Latent Regularization</span>
            <span class="highlight-badge">128×128 → 256×256</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# DIRECTORY & DATASET VALIDATION
# ============================================================

# Identify available noisy directories
noisy_dir_used = None
if os.path.isdir(TRAIN_NOISY_DIR) and len(os.listdir(TRAIN_NOISY_DIR)) > 0:
    noisy_dir_used = TRAIN_NOISY_DIR
elif os.path.isdir(TEST_NOISY_DIR) and len(os.listdir(TEST_NOISY_DIR)) > 0:
    noisy_dir_used = TEST_NOISY_DIR

# Required directory checks
missing_errors = []
if noisy_dir_used is None:
    missing_errors.append(f"NoisyLR Input Directory: {TRAIN_NOISY_DIR} or {TEST_NOISY_DIR}")

if not os.path.isdir(GT_DIR):
    missing_errors.append(f"Ground Truth Directory: {GT_DIR}")

if not os.path.isdir(PRED_DIR):
    missing_errors.append(f"Joint SAE Predictions Directory: {PRED_DIR}")

if missing_errors:
    st.error("⚠️ **Required project directories missing or empty:**")
    for err in missing_errors:
        st.code(err)
    st.info("Please ensure your dataset files and Joint SAE predictions exist in `outputs/final_predictions`.")
    st.stop()


# Find matched samples across NoisyLR, Predictions, and Ground Truth
noisy_files = set(f for f in os.listdir(noisy_dir_used) if f.endswith(".npy"))
pred_files = set(f for f in os.listdir(PRED_DIR) if f.endswith(".npy"))
gt_files = set(f for f in os.listdir(GT_DIR) if f.endswith(".npy"))

# Find common filenames across all three directories
common_files = sorted(list(noisy_files & pred_files & gt_files))

# Fallback: paired noisy + prediction if GT matching filename isn't exact
fallback_files = sorted(list(noisy_files & pred_files))

if not common_files and not fallback_files:
    st.error("⚠️ **No matching `.npy` samples found across dataset and prediction directories.**")
    st.write(f"- NoisyLR files found: {len(noisy_files)}")
    st.write(f"- Prediction files found: {len(pred_files)}")
    st.write(f"- Ground Truth files found: {len(gt_files)}")
    st.stop()

valid_samples = common_files if common_files else fallback_files


# ============================================================
# SAMPLE SELECTION CONTROLS
# ============================================================

if "current_index" not in st.session_state:
    st.session_state.current_index = random.randint(0, len(valid_samples) - 1)

col_select1, col_select2, col_select3 = st.columns([2, 1, 1])

with col_select1:
    selected_filename = valid_samples[st.session_state.current_index]
    st.markdown(
        f"""
        <div class="sample-bar-container" style="display:flex; align-items:center; justify-content:space-between;">
            <div style="font-weight:600; color:#8b949e;">Active Sample:</div>
            <div class="sample-badge">📄 {selected_filename}</div>
            <div style="font-size:0.85rem; color:#8b949e;">(Sample {st.session_state.current_index + 1} of {len(valid_samples)})</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col_select2:
    if st.button("🎲 Random Sample", use_container_width=True):
        st.session_state.current_index = random.randint(0, len(valid_samples) - 1)
        st.rerun()

with col_select3:
    if st.button("➡️ Next Sample", use_container_width=True):
        st.session_state.current_index = (st.session_state.current_index + 1) % len(valid_samples)
        st.rerun()


# ============================================================
# LOAD SAMPLE NUMPY ARRAYS
# ============================================================

noisy_path = os.path.join(noisy_dir_used, selected_filename)
pred_path = os.path.join(PRED_DIR, selected_filename)
gt_path = os.path.join(GT_DIR, selected_filename) if os.path.exists(os.path.join(GT_DIR, selected_filename)) else None

try:
    noisy_arr = np.load(noisy_path).astype(np.float32)
    pred_arr = np.load(pred_path).astype(np.float32)
    gt_arr = np.load(gt_path).astype(np.float32) if gt_path else None
except Exception as e:
    st.error(f"❌ Error loading sample files: {e}")
    st.stop()


# Validate shapes
shape_warnings = []
if noisy_arr.shape != (128, 128):
    shape_warnings.append(f"NoisyLR shape is {noisy_arr.shape}, expected (128, 128)")
if pred_arr.shape != (256, 256):
    shape_warnings.append(f"Joint SAE Output shape is {pred_arr.shape}, expected (256, 256)")
if gt_arr is not None and gt_arr.shape != (256, 256):
    shape_warnings.append(f"Ground Truth shape is {gt_arr.shape}, expected (256, 256)")

for warn in shape_warnings:
    st.warning(f"⚠️ {warn}")


# Helper function to prepare normalized float array for display preview (grayscale)
def normalize_for_display(img_array):
    """Min-max normalize for visual display preview without altering underlying npy array statistics."""
    arr_min = img_array.min()
    arr_max = img_array.max()
    if arr_max - arr_min < 1e-6:
        return np.zeros_like(img_array)
    return (img_array - arr_min) / (arr_max - arr_min)


# Prepare visual display images
noisy_display = normalize_for_display(noisy_arr)
pred_display = normalize_for_display(pred_arr)
gt_display = normalize_for_display(gt_arr) if gt_arr is not None else np.zeros((256, 256), dtype=np.float32)


# ============================================================
# MAIN VISUALIZATION (3 COLUMNS)
# ============================================================

st.markdown('<div class="section-header"><span class="section-header-icon">🖼️</span> Visual Comparison</div>', unsafe_allow_html=True)

col_img1, col_img2, col_img3 = st.columns(3)

# LEFT COLUMN: Degraded Input
with col_img1:
    st.markdown(
        """
        <div class="image-card">
            <div class="image-title">Degraded Input</div>
            <div class="image-subtitle">NoisyLR • 128 × 128 • Low-Res</div>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.image(noisy_display, clamp=True, use_container_width=True)
    
    st.markdown(
        """
        <div class="meta-box">
            <div class="meta-row"><span class="meta-key">Resolution</span><span class="meta-val">128 × 128</span></div>
            <div class="meta-row"><span class="meta-key">Channels</span><span class="meta-val">1 (Grayscale)</span></div>
            <div class="meta-row"><span class="meta-key">Image Type</span><span class="meta-val">NoisyLR</span></div>
            <div class="meta-row"><span class="meta-key">Format</span><span class="meta-val">NumPy (.npy)</span></div>
        </div>
        """,
        unsafe_allow_html=True
    )

# CENTER COLUMN: Joint SAE Restored Output (Emphasized Focus)
with col_img2:
    st.markdown(
        """
        <div class="image-card-center">
            <div class="center-focus-tag">★ Core Model Output</div>
            <div class="image-title" style="color:#58a6ff; margin-top:0.4rem;">Joint SAE Restored Output</div>
            <div class="image-subtitle" style="color:#79c0ff;">Feature Restored • 256 × 256 • 2× Upscale</div>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.image(pred_display, clamp=True, use_container_width=True)
    
    st.markdown(
        """
        <div class="meta-box" style="border-color: rgba(88, 166, 255, 0.4);">
            <div class="meta-row"><span class="meta-key">Resolution</span><span class="meta-val-highlight">256 × 256</span></div>
            <div class="meta-row"><span class="meta-key">Channels</span><span class="meta-val">1 (Grayscale)</span></div>
            <div class="meta-row"><span class="meta-key">Image Type</span><span class="meta-val-highlight">Joint SAE Restored</span></div>
            <div class="meta-row"><span class="meta-key">Upscale Factor</span><span class="meta-val-highlight">2× (128 → 256)</span></div>
        </div>
        """,
        unsafe_allow_html=True
    )

# RIGHT COLUMN: Ground Truth
with col_img3:
    st.markdown(
        """
        <div class="image-card">
            <div class="image-title">Ground Truth</div>
            <div class="image-subtitle">Original Clean • 256 × 256</div>
        </div>
        """,
        unsafe_allow_html=True
    )
    if gt_arr is not None:
        st.image(gt_display, clamp=True, use_container_width=True)
        st.markdown(
            """
            <div class="meta-box">
                <div class="meta-row"><span class="meta-key">Resolution</span><span class="meta-val">256 × 256</span></div>
                <div class="meta-row"><span class="meta-key">Channels</span><span class="meta-val">1 (Grayscale)</span></div>
                <div class="meta-row"><span class="meta-key">Image Type</span><span class="meta-val">Ground Truth</span></div>
                <div class="meta-row"><span class="meta-key">Format</span><span class="meta-val">NumPy (.npy)</span></div>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.info("ℹ️ Ground Truth not available for this test sample.")


# ============================================================
# IMAGE RESTORATION COMPARISON & NUMPY STATISTICS
# ============================================================

st.markdown('<div class="section-header"><span class="section-header-icon">📊</span> Image Restoration Comparison</div>', unsafe_allow_html=True)

gt_min_str = f"{gt_arr.min():.4f}" if gt_arr is not None else "N/A"
gt_max_str = f"{gt_arr.max():.4f}" if gt_arr is not None else "N/A"
gt_mean_str = f"{gt_arr.mean():.4f}" if gt_arr is not None else "N/A"
gt_std_str = f"{gt_arr.std():.4f}" if gt_arr is not None else "N/A"

st.markdown(
    f"""
    <div class="comparison-table-wrapper">
        <table class="comp-table">
            <thead>
                <tr>
                    <th>Feature / Property</th>
                    <th>Degraded Input</th>
                    <th>Joint SAE Output</th>
                    <th>Ground Truth</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td><strong>Spatial Resolution</strong></td>
                    <td><span class="table-mono">128 × 128</span></td>
                    <td><span class="table-mono" style="color:#58a6ff;">256 × 256 (2×)</span></td>
                    <td><span class="table-mono">256 × 256</span></td>
                </tr>
                <tr>
                    <td><strong>Channels</strong></td>
                    <td><span class="table-mono">1</span></td>
                    <td><span class="table-mono">1</span></td>
                    <td><span class="table-mono">1</span></td>
                </tr>
                <tr>
                    <td><strong>File Format</strong></td>
                    <td><span class="table-mono">NumPy .npy</span></td>
                    <td><span class="table-mono">NumPy .npy</span></td>
                    <td><span class="table-mono">NumPy .npy</span></td>
                </tr>
                <tr>
                    <td><strong>Image Type</strong></td>
                    <td>Degraded (NoisyLR)</td>
                    <td><strong style="color:#3fb950;">Restored</strong></td>
                    <td>Original Clean</td>
                </tr>
                <tr>
                    <td><strong>Intensity Min Value</strong></td>
                    <td><span class="table-mono">{noisy_arr.min():.4f}</span></td>
                    <td><span class="table-mono">{pred_arr.min():.4f}</span></td>
                    <td><span class="table-mono">{gt_min_str}</span></td>
                </tr>
                <tr>
                    <td><strong>Intensity Max Value</strong></td>
                    <td><span class="table-mono">{noisy_arr.max():.4f}</span></td>
                    <td><span class="table-mono">{pred_arr.max():.4f}</span></td>
                    <td><span class="table-mono">{gt_max_str}</span></td>
                </tr>
                <tr>
                    <td><strong>Intensity Mean Value</strong></td>
                    <td><span class="table-mono">{noisy_arr.mean():.4f}</span></td>
                    <td><span class="table-mono">{pred_arr.mean():.4f}</span></td>
                    <td><span class="table-mono">{gt_mean_str}</span></td>
                </tr>
                <tr>
                    <td><strong>Intensity Std Deviation</strong></td>
                    <td><span class="table-mono">{noisy_arr.std():.4f}</span></td>
                    <td><span class="table-mono">{pred_arr.std():.4f}</span></td>
                    <td><span class="table-mono">{gt_std_str}</span></td>
                </tr>
            </tbody>
        </table>
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# RESTORATION PIPELINE FLOW DIAGRAM
# ============================================================

st.markdown('<div class="section-header"><span class="section-header-icon">🔄</span> Restoration Pipeline</div>', unsafe_allow_html=True)

st.markdown(
    """
    <div class="pipeline-container">
        <div class="pipeline-node">
            <div class="pipeline-node-title">NoisyLR Input</div>
            <div class="pipeline-node-sub">128 × 128 × 1</div>
        </div>
        <div class="pipeline-arrow">→</div>
        <div class="pipeline-node">
            <div class="pipeline-node-title">Feature Extraction</div>
            <div class="pipeline-node-sub">Conv2D (64 Ch)</div>
        </div>
        <div class="pipeline-arrow">→</div>
        <div class="pipeline-node-sae">
            <div class="pipeline-node-title" style="color:#58a6ff;">Sparse Autoencoder</div>
            <div class="pipeline-node-sub" style="color:#79c0ff;">Latent Dim: 128</div>
        </div>
        <div class="pipeline-arrow">→</div>
        <div class="pipeline-node">
            <div class="pipeline-node-title">Feature Reconstruction</div>
            <div class="pipeline-node-sub">64 Ch Feature Map</div>
        </div>
        <div class="pipeline-arrow">→</div>
        <div class="pipeline-node">
            <div class="pipeline-node-title">2× Upsampling</div>
            <div class="pipeline-node-sub">PixelShuffle / Conv</div>
        </div>
        <div class="pipeline-arrow">→</div>
        <div class="pipeline-node" style="border-color:#3fb950;">
            <div class="pipeline-node-title" style="color:#3fb950;">Restored Output</div>
            <div class="pipeline-node-sub">256 × 256 × 1</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# VALIDATION RESULTS
# ============================================================

st.markdown('<div class="section-header"><span class="section-header-icon">📈</span> Validation Results</div>', unsafe_allow_html=True)

st.markdown(
    """
    <div class="metric-card-group">
        <div class="res-metric-card res-metric-card-baseline">
            <div class="metric-title">Baseline Network</div>
            <div class="metric-main-value">28.3739 dB</div>
            <div class="metric-sub-label">PSNR Score</div>
            <div style="margin-top:0.8rem;" class="metric-main-value">0.7752</div>
            <div class="metric-sub-label">SSIM Score</div>
        </div>
        
        <div class="res-metric-card res-metric-card-joint">
            <div class="metric-title" style="color:#58a6ff;">Joint SAE Model</div>
            <div class="metric-main-value" style="color:#58a6ff;">28.3865 dB</div>
            <div class="metric-sub-label">PSNR Score</div>
            <div style="margin-top:0.8rem;" class="metric-main-value" style="color:#58a6ff;">0.7761</div>
            <div class="metric-sub-label">SSIM Score</div>
        </div>
        
        <div class="res-metric-card res-metric-card-gain">
            <div class="metric-title" style="color:#3fb950;">SAE Improvement</div>
            <div class="metric-gain-value">+0.0125 dB</div>
            <div class="metric-sub-label">PSNR Improvement</div>
            <div style="margin-top:0.8rem;" class="metric-gain-value">+0.0009</div>
            <div class="metric-sub-label">SSIM Improvement</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# ABOUT THE JOINT SAE MODEL (COLLAPSIBLE)
# ============================================================

with st.expander("ℹ️ About the Joint SAE Model"):
    st.markdown(
        """
        ### Joint SAE Semiconductor Image Restoration Model
        
        The **ReSAGE** architecture enhances standard semiconductor image super-resolution by integrating a **Sparse Autoencoder (SAE)** directly into the feature representation space.
        
        #### Model Architecture & Specifications:
        - **Baseline Restoration Network**: Deep residual convolutional neural network with 4 residual blocks.
        - **Intermediate Feature Representation**: 64 channels at 128 × 128 spatial resolution.
        - **Sparse Autoencoder (SAE)**: Latent dimension **128**, imposing sparse feature bottlenecking to eliminate noise artifacts and enhance edge fidelity.
        - **Input Format**: 1-channel grayscale low-resolution noisy image (**128 × 128**).
        - **Output Format**: 1-channel restored high-resolution image (**256 × 256**).
        - **Upscaling Factor**: **2×** super-resolution.
        - **Training Strategy**: Joint fine-tuning of baseline restoration weights and SAE parameters.
        
        *The restored images demonstrated above are loaded directly from the existing Joint SAE prediction outputs.*
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer-text">
        ReSAGE • AI-Based Semiconductor Image Restoration Project Demo • Powered by Streamlit
    </div>
    """,
    unsafe_allow_html=True
)