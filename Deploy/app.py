import streamlit as st
import nibabel as nib
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os
import tempfile
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import time
import gdown
from scipy.ndimage import zoom
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import tensorflow as tf

# Force CPU-only operation to avoid CUDA errors
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Set page config
st.set_page_config(page_title="Glioma Segmentation", layout="wide")

# Initialize scaler
scaler = MinMaxScaler()

# Constants
MODEL_URL = "https://drive.google.com/uc?id=1lV1SgafomQKwgv1NW2cjlpyb4LwZXFwX"
MODEL_DIR = "saved_model"
MODEL_PATH = os.path.join(MODEL_DIR, "3D_unet_100_epochs_2_batch_patch_training.keras")

# Model expects (64, 64, 64, 4) input
TARGET_SHAPE = (64, 64, 64, 4)

# Ensure model directory exists
os.makedirs(MODEL_DIR, exist_ok=True)

# Download model from Google Drive (cache this to avoid repeated downloads)
@st.cache_resource
def download_and_load_model():
    if not os.path.exists(MODEL_PATH):
        st.info("Downloading model from Google Drive... (This may take a few minutes)")
        try:
            gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
            if not os.path.exists(MODEL_PATH):
                raise FileNotFoundError("Model download failed")
        except Exception as e:
            st.error(f"Failed to download model: {str(e)}")
            return None
    
    try:
        tf.get_logger().setLevel('ERROR')
        model = load_model(MODEL_PATH, compile=False)
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Function to process uploaded files
def process_uploaded_files(uploaded_files):
    modalities = {}
    for uploaded_file in uploaded_files:
        file_name = uploaded_file.name.lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix='.nii.gz') as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            tmp_path = tmp_file.name
        
        try:
            img = nib.load(tmp_path)
            img_data = img.get_fdata()
            img_data = scaler.fit_transform(img_data.reshape(-1, img_data.shape[-1])).reshape(img_data.shape)
            
            if 't1n' in file_name:
                modalities['t1n'] = img_data
            elif 't1c' in file_name:
                modalities['t1c'] = img_data
            elif 't2f' in file_name:
                modalities['t2f'] = img_data
            elif 't2w' in file_name:
                modalities['t2w'] = img_data
            elif 'seg' in file_name:
                modalities['mask'] = img_data.astype(np.uint8)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    return modalities

# Function to prepare input for model
def prepare_input(modalities):
    required = ['t1n', 't1c', 't2f', 't2w']
    if not all(m in modalities for m in required):
        return None, None, None
    
    combined = np.stack([
        modalities['t1n'],
        modalities['t1c'],
        modalities['t2f'],
        modalities['t2w']
    ], axis=3)
    
    combined = combined[56:184, 56:184, 13:141, :]
    original_shape = combined.shape
    downsampled = combined[::2, ::2, ::2, :]
    
    return downsampled, original_shape, combined

# Function to make prediction
def make_prediction(model, input_data):
    input_data = np.expand_dims(input_data, axis=0)
    prediction = model.predict(input_data, verbose=0)
    return np.argmax(prediction, axis=4)[0, :, :, :]

# Function to upsample prediction
def upsample_prediction(prediction, target_shape):
    zoom_factors = (
        target_shape[0] / prediction.shape[0],
        target_shape[1] / prediction.shape[1],
        target_shape[2] / prediction.shape[2]
    )
    return zoom(prediction, zoom_factors, order=0)

# Function to create zoomable visualization
def create_zoomable_visualization(original_data, prediction, ground_truth=None):
    # Select slices to display
    slice_indices = [30, 50, 70]
    modality_idx = 1  # Using T1c for display
    
    # Create subplot figure
    rows = 3
    cols = 3 if ground_truth is not None else 2
    fig = make_subplots(rows=rows, cols=cols,
                        subplot_titles=[f"Slice {idx}" for idx in slice_indices]*cols,
                        horizontal_spacing=0.05, vertical_spacing=0.05)
    
    for i, slice_idx in enumerate(slice_indices):
        row = i + 1
        
        # Input image
        img_slice = np.rot90(original_data[:, :, slice_idx, modality_idx])
        fig.add_trace(go.Heatmap(z=img_slice, colorscale='gray', showscale=False),
                     row=row, col=1)
        
        # Prediction
        pred_slice = np.rot90(prediction[:, :, slice_idx])
        fig.add_trace(go.Heatmap(z=pred_slice, showscale=False),
                     row=row, col=2)
        
        # Ground truth if available
        if ground_truth is not None:
            gt_slice = np.rot90(ground_truth[:, :, slice_idx])
            fig.add_trace(go.Heatmap(z=gt_slice, showscale=False),
                         row=row, col=3)
    
    # Update layout for better display
    fig.update_layout(
        height=800,
        width=1000 if ground_truth is not None else 700,
        margin=dict(l=20, r=20, t=50, b=20),
        title_text="Glioma Segmentation Results (Zoomable)",
        title_x=0.5
    )
    
    # Add column titles
    fig.update_annotations(
        text="Input Image", x=0.16, y=1.05, xref="paper", yref="paper", showarrow=False
    )
    fig.update_annotations(
        text="Prediction", x=0.5, y=1.05, xref="paper", yref="paper", showarrow=False
    )
    if ground_truth is not None:
        fig.update_annotations(
            text="Ground Truth", x=0.84, y=1.05, xref="paper", yref="paper", showarrow=False
        )
    
    return fig

def main():
    st.title("3D Glioma Segmentation with U-Net")
    st.write("Upload MRI scans in NIfTI format for glioma segmentation")
    
    with st.expander("How to use this app"):
        st.markdown("""
        1. Upload **all four MRI modalities** (T1n, T1c, T2f, T2w) as NIfTI files (.nii.gz)
        2. Optionally upload a segmentation mask for comparison
        3. Click 'Process and Predict' button
        4. View and interact with the zoomable results
        """)
    
    model = download_and_load_model()
    if model is None:
        return
    
    uploaded_files = st.file_uploader(
        "Upload MRI scans (NIfTI format)",
        type=['nii', 'nii.gz'],
        accept_multiple_files=True
    )
    
    if uploaded_files and len(uploaded_files) >= 4:
        if st.button("Process and Predict"):
            with st.spinner("Processing files..."):
                modalities = process_uploaded_files(uploaded_files)
                input_data, original_shape, original_data = prepare_input(modalities)
                
                if input_data is None:
                    st.error("Could not prepare input data. Please ensure you've uploaded all required modalities.")
                    return
                
                ground_truth = None
                if 'mask' in modalities:
                    ground_truth = modalities['mask'][56:184, 56:184, 13:141]
                    ground_truth[ground_truth == 4] = 3
                
                with st.spinner("Making prediction..."):
                    start_time = time.time()
                    prediction = make_prediction(model, input_data)
                    prediction = upsample_prediction(prediction, original_shape[:3])
                    prediction = prediction.astype(np.int32)
                    elapsed_time = time.time() - start_time
                
                st.success(f"Prediction completed in {elapsed_time:.2f} seconds")
                
                # Create zoomable visualization
                fig = create_zoomable_visualization(original_data, prediction, ground_truth)
                st.plotly_chart(fig, use_container_width=True)
                
                # Download option
                with tempfile.NamedTemporaryFile(suffix='.nii.gz', delete=False) as tmp_file:
                    nib.save(nib.Nifti1Image(prediction, affine=np.eye(4), dtype=np.int32), tmp_file.name)
                    with open(tmp_file.name, 'rb') as f:
                        pred_data = f.read()
                    os.unlink(tmp_file.name)
                
                st.download_button(
                    label="Download Segmentation (NIfTI)",
                    data=pred_data,
                    file_name="glioma_segmentation.nii.gz",
                    mime="application/octet-stream"
                )
    elif uploaded_files and len(uploaded_files) < 4:
        st.warning("Please upload all four modalities (T1n, T1c, T2f, T2w)")

if __name__ == "__main__":
    main()
