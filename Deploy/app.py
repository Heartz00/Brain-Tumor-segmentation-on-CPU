import streamlit as st
import numpy as np
import nibabel as nib
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import os
import matplotlib.pyplot as plt
import gdown
import zipfile
import tempfile
from tensorflow.keras.utils import to_categorical

# Title of the app
st.title("Brain Tumor Segmentation using 3D U-Net - (Lightweight Architecture on Normal CPUs)")

# Function to download the default model from Google Drive
def download_default_model():
    file_id = "1lV1SgafomQKwgv1NW2cjlpyb4LwZXFwX"  # Replace with your file ID
    output_path = "default_model.keras"
    
    if not os.path.exists(output_path):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output_path, quiet=False)
    
    return output_path

# Load the default model
@st.cache_resource
def load_default_model():
    model_path = download_default_model()
    model = load_model(model_path, compile=False)
    return model

default_model = load_default_model()

# Function to preprocess a NIfTI file
def preprocess_nifti(file_path):
    image = nib.load(file_path).get_fdata()
    scaler = MinMaxScaler()
    image = scaler.fit_transform(image.reshape(-1, image.shape[-1])).reshape(image.shape)
    return image

# Function to combine and crop 4 channels
def combine_channels(t1n, t1c, t2f, t2w):
    combined = np.stack([t1n, t1c, t2f, t2w], axis=-1)  # Shape: (H,W,D,4)
    cropped = combined[56:184, 56:184, 13:141, :]  # Crop to 128x128x128x4
    return cropped

# Function to run segmentation with proper input shaping
def run_segmentation(model, input_image):
    # Add batch dimension and ensure correct shape
    input_image = np.expand_dims(input_image, axis=0)  # Shape: (1,128,128,128,4)
    
    if input_image.shape != (1,128,128,128,4):
        st.error(f"Input shape must be (1,128,128,128,4). Got {input_image.shape}")
        return None
    
    try:
        prediction = model.predict(input_image)
        return np.argmax(prediction, axis=4)[0]  # Remove batch dim
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")
        return None

# Sidebar for model upload
st.sidebar.header("Upload Your Own Model")
uploaded_model = st.sidebar.file_uploader("Upload a Keras model (.keras)", type=["keras"])

# Model selection
model = default_model
if uploaded_model:
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".keras") as tmp:
            tmp.write(uploaded_model.getbuffer())
            model = load_model(tmp.name, compile=False)
        st.sidebar.success("Custom model loaded!")
    except Exception as e:
        st.sidebar.error(f"Invalid model: {str(e)}")
        st.sidebar.info("Using default model")

# Main app
st.header("Upload MRI Scans (ZIP containing T1n, T1c, T2f, T2w)")
uploaded_zip = st.file_uploader("Upload scans", type=["zip"])

if uploaded_zip:
    with tempfile.TemporaryDirectory() as tmpdir:
        # Save and extract zip
        zip_path = os.path.join(tmpdir, "upload.zip")
        with open(zip_path, "wb") as f:
            f.write(uploaded_zip.getbuffer())
        
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(tmpdir)
        
        # Find required files
        required = {"t1n": None, "t1c": None, "t2f": None, "t2w": None, "seg": None}
        for root, _, files in os.walk(tmpdir):
            for f in files:
                for k in required:
                    if k in f.lower() and f.endswith(".nii.gz"):
                        required[k] = os.path.join(root, f)
        
        # Check if all MRI sequences found
        if not all(required.values()):
            st.error("Missing files in ZIP. Need: T1n, T1c, T2f, T2w, and seg")
        else:
            # Load and preprocess each modality
            modalities = {}
            for name, path in required.items():
                if name != "seg":
                    modalities[name] = preprocess_nifti(path)
            
            # Combine channels
            combined = combine_channels(
                modalities["t1n"],
                modalities["t1c"], 
                modalities["t2f"],
                modalities["t2w"]
            )
            
            st.write(f"Input shape: {combined.shape}")
            
            # Run segmentation
            seg_result = run_segmentation(model, combined)
            
            if seg_result is not None:
                # Load ground truth
                mask = nib.load(required["seg"]).get_fdata()
                mask = mask[56:184, 56:184, 13:141]  # Crop to match
                mask[mask == 4] = 3
                
                # Visualization
                fig, axes = plt.subplots(3, 3, figsize=(15, 10))
                slices = [40, 64, 90]  # Example slices
                
                for i, sl in enumerate(slices):
                    # Original (T1c)
                    axes[i,0].imshow(np.rot90(combined[:,:,sl,1]), cmap='gray')
                    axes[i,0].set_title(f"T1c Slice {sl}")
                    
                    # Ground truth
                    axes[i,1].imshow(np.rot90(mask[:,:,sl]))
                    axes[i,1].set_title(f"Ground Truth")
                    
                    # Prediction
                    axes[i,2].imshow(np.rot90(seg_result[:,:,sl]))
                    axes[i,2].set_title(f"Prediction")
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # Save results
                output_path = "prediction.nii.gz"
                nib.save(nib.Nifti1Image(seg_result, np.eye(4)), output_path)
                
                with open(output_path, "rb") as f:
                    st.download_button(
                        "Download Prediction",
                        f,
                        file_name=output_path,
                        mime="application/octet-stream"
                    )
                
                os.remove(output_path)
