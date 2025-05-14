import streamlit as st
import nibabel as nib
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os
import tempfile
from tensorflow.keras.models import load_model
from matplotlib import pyplot as plt
import time
import requests
import gdown
from pathlib import Path

# Set page config
st.set_page_config(page_title="Glioma Segmentation", layout="wide")

# Initialize scaler
scaler = MinMaxScaler()

# Constants
MODEL_URL = "https://drive.google.com/uc?id=1lV1SgafomQKwgv1NW2cjlpyb4LwZXFwX"
MODEL_DIR = "saved_model"
MODEL_PATH = os.path.join(MODEL_DIR, "3D_unet_100_epochs_2_batch_patch_training.keras")

# Ensure model directory exists
os.makedirs(MODEL_DIR, exist_ok=True)

# Download model from Google Drive (cache this to avoid repeated downloads)
@st.cache_resource
def download_and_load_model():
    # Check if model already exists
    if not os.path.exists(MODEL_PATH):
        st.info("Downloading model from Google Drive... (This may take a few minutes)")
        try:
            # Download using gdown
            gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
            
            # Verify download
            if not os.path.exists(MODEL_PATH):
                raise FileNotFoundError("Model download failed")
                
        except Exception as e:
            st.error(f"Failed to download model: {str(e)}")
            return None
    
    # Load the model
    try:
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
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.nii.gz') as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            tmp_path = tmp_file.name
        
        try:
            # Load NIfTI file
            img = nib.load(tmp_path)
            img_data = img.get_fdata()
            
            # Scale the data
            img_data = scaler.fit_transform(img_data.reshape(-1, img_data.shape[-1])).reshape(img_data.shape)
            
            # Determine modality
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
                
        except Exception as e:
            st.error(f"Error processing file {uploaded_file.name}: {str(e)}")
        finally:
            # Clean up temporary file
            os.unlink(tmp_path)
    
    return modalities

# Function to prepare input for model
def prepare_input(modalities):
    # Check we have all required modalities
    required = ['t1n', 't1c', 't2f', 't2w']
    if not all(m in modalities for m in required):
        return None
    
    # Combine modalities
    combined = np.stack([
        modalities['t1n'],
        modalities['t1c'],
        modalities['t2f'],
        modalities['t2w']
    ], axis=3)
    
    # Crop to expected size (128, 128, 128, 4)
    combined = combined[56:184, 56:184, 13:141, :]
    
    return combined

# Function to make prediction
def make_prediction(model, input_data):
    # Add batch dimension
    input_data = np.expand_dims(input_data, axis=0)
    
    # Make prediction
    prediction = model.predict(input_data)
    prediction_argmax = np.argmax(prediction, axis=4)[0, :, :, :]
    
    return prediction_argmax

# Function to visualize results
def visualize_results(input_data, prediction, ground_truth=None):
    # Select a modality to display (using T1c here)
    image_data = input_data[:, :, :, 1]  # T1c is the second channel
    
    # Select some slices to display
    slice_indices = [30, 50, 70]
    
    # Create figure
    fig, axes = plt.subplots(3, 3 if ground_truth is not None else 2, 
                            figsize=(15, 10))
    
    for i, slice_idx in enumerate(slice_indices):
        # Rotate images for better visualization
        img_slice = np.rot90(image_data[:, :, slice_idx])
        pred_slice = np.rot90(prediction[:, :, slice_idx])
        
        # Plot input image
        axes[i, 0].imshow(img_slice, cmap='gray')
        axes[i, 0].set_title(f'Input Image - Slice {slice_idx}')
        axes[i, 0].axis('off')
        
        # Plot prediction
        axes[i, 1].imshow(pred_slice)
        axes[i, 1].set_title(f'Prediction - Slice {slice_idx}')
        axes[i, 1].axis('off')
        
        # Plot ground truth if available
        if ground_truth is not None:
            gt_slice = np.rot90(ground_truth[:, :, slice_idx])
            axes[i, 2].imshow(gt_slice)
            axes[i, 2].set_title(f'Ground Truth - Slice {slice_idx}')
            axes[i, 2].axis('off')
    
    plt.tight_layout()
    return fig

# Main app
def main():
    st.title("3D Glioma Segmentation with U-Net")
    st.write("Upload MRI scans in NIfTI format for glioma segmentation")
    
    with st.expander("How to use this app"):
        st.markdown("""
        1. Upload **all four MRI modalities** (T1n, T1c, T2f, T2w) as NIfTI files (.nii.gz)
        2. Optionally upload a segmentation mask for comparison (must contain 'seg' in filename)
        3. Click 'Process and Predict' button
        4. View the segmentation results
        
        **Note:** The first run will download the model (~100MB) which may take a few minutes.
        """)
    
    # Load model (this will trigger download if needed)
    model = download_and_load_model()
    
    if model is None:
        st.error("Failed to load model. Please check the error message above.")
        return
    
    # File uploader
    uploaded_files = st.file_uploader(
        "Upload MRI scans (NIfTI format)",
        type=['nii', 'nii.gz'],
        accept_multiple_files=True
    )
    
    if uploaded_files and len(uploaded_files) >= 4:
        if st.button("Process and Predict"):
            with st.spinner("Processing files..."):
                # Process uploaded files
                modalities = process_uploaded_files(uploaded_files)
                
                # Prepare input
                input_data = prepare_input(modalities)
                
                if input_data is None:
                    st.error("Could not prepare input data. Please ensure you've uploaded all required modalities.")
                    return
                
                # Get ground truth if available
                ground_truth = modalities.get('mask', None)
                if ground_truth is not None:
                    ground_truth = ground_truth[56:184, 56:184, 13:141]
                    ground_truth[ground_truth == 4] = 3  # Reassign label 4 to 3
                
                # Make prediction
                with st.spinner("Making prediction..."):
                    start_time = time.time()
                    prediction = make_prediction(model, input_data)
                    elapsed_time = time.time() - start_time
                
                st.success(f"Prediction completed in {elapsed_time:.2f} seconds")
                
                # Visualize results
                fig = visualize_results(input_data, prediction, ground_truth)
                st.pyplot(fig)
                
                # Provide download option for prediction
                st.subheader("Download Prediction")
                
                # Create a temporary NIfTI file for download
                with tempfile.NamedTemporaryFile(suffix='.nii.gz', delete=False) as tmp_file:
                    # Convert prediction to NIfTI
                    pred_img = nib.Nifti1Image(prediction, affine=np.eye(4))
                    nib.save(pred_img, tmp_file.name)
                    
                    # Read back the file data
                    with open(tmp_file.name, 'rb') as f:
                        pred_data = f.read()
                    
                    # Clean up
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
