import streamlit as st
import numpy as np
import nibabel as nib
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import os
import matplotlib.pyplot as plt
import zipfile
import tempfile
from tensorflow.keras.utils import to_categorical

# Configure app
st.set_page_config(layout="wide")
st.title("Brain Tumor Segmentation using 3D U-Net")

# Constants
TARGET_SHAPE = (128, 128, 128, 4)
CROP_PARAMS = ((56, 184), (56, 184), (13, 141))  # y, x, z cropping

# Load default model
@st.cache_resource
import gdown  # For Google Drive
import os

@st.cache_resource
def load_default_model():
    try:
        # Download model if it doesn't exist
        if not os.path.exists("default_model.keras"):
            # Google Drive example (replace with your link)
            url = "url = "https://drive.google.com/uc?id=1lV1SgafomQKwgv1NW2cjlpyb4LwZXFwX"
            gdown.download(url, "default_model.keras", quiet=False)
        
        return load_model("default_model.keras", compile=False)
    except Exception as e:
        st.error(f"Failed to load default model: {str(e)}")
        return None
        

model = load_default_model()

# File processing functions
def load_and_preprocess_nifti(filepath):
    try:
        img = nib.load(filepath).get_fdata()
        scaler = MinMaxScaler()
        return scaler.fit_transform(img.reshape(-1, img.shape[-1])).reshape(img.shape)
    except Exception as e:
        st.error(f"Error processing {filepath}: {str(e)}")
        return None

def process_uploaded_zip(uploaded_zip):
    with tempfile.TemporaryDirectory() as tmpdir:
        # Save and extract zip
        zip_path = os.path.join(tmpdir, "upload.zip")
        with open(zip_path, "wb") as f:
            f.write(uploaded_zip.getbuffer())
        
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(tmpdir)
        
        # Find required files
        files = {
            't1n': None, 't1c': None, 
            't2f': None, 't2w': None,
            'seg': None
        }
        
        for root, _, filenames in os.walk(tmpdir):
            for f in filenames:
                f_lower = f.lower()
                if f.endswith('.nii.gz'):
                    if 't1n' in f_lower: files['t1n'] = os.path.join(root, f)
                    elif 't1c' in f_lower: files['t1c'] = os.path.join(root, f)
                    elif 't2f' in f_lower: files['t2f'] = os.path.join(root, f)
                    elif 't2w' in f_lower: files['t2w'] = os.path.join(root, f)
                    elif 'seg' in f_lower: files['seg'] = os.path.join(root, f)
        
        return files

# Model prediction
def predict_volume(model, volume):
    try:
        # Add batch dimension
        input_data = np.expand_dims(volume, axis=0)
        
        # Verify input shape
        if input_data.shape[1:] != TARGET_SHAPE:
            st.error(f"Input shape mismatch. Expected {TARGET_SHAPE}, got {input_data.shape[1:]}")
            return None
            
        return model.predict(input_data)[0]  # Remove batch dimension
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")
        return None

# UI Components
def show_results(input_vol, prediction, ground_truth=None):
    slices = [30, 64, 90]  # Representative slices
    
    fig, axes = plt.subplots(len(slices), 3, figsize=(15, 5*len(slices)))
    
    for i, sl in enumerate(slices):
        # Input (T1c channel)
        axes[i,0].imshow(np.rot90(input_vol[:,:,sl,1]), cmap='gray')
        axes[i,0].set_title(f"Input Slice {sl}")
        
        # Ground truth if available
        if ground_truth is not None:
            axes[i,1].imshow(np.rot90(ground_truth[:,:,sl]))
            axes[i,1].set_title("Ground Truth")
        else:
            axes[i,1].axis('off')
        
        # Prediction
        axes[i,2].imshow(np.rot90(np.argmax(prediction, axis=-1)[:,:,sl]))
        axes[i,2].set_title("Prediction")
    
    plt.tight_layout()
    st.pyplot(fig)

# Main app flow
def main():
    # Model upload section
    with st.sidebar:
        st.header("Model Configuration")
        uploaded_model = st.file_uploader("Upload custom model (.keras)", type=['keras'])
        
        if uploaded_model:
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.keras') as tmp:
                    tmp.write(uploaded_model.getbuffer())
                    model = load_model(tmp.name, compile=False)
                st.success("Custom model loaded successfully!")
            except Exception as e:
                st.error(f"Failed to load custom model: {str(e)}")
                st.info("Reverting to default model")
                model = load_default_model()
    
    # Main processing section
    st.header("MRI Volume Upload")
    uploaded_zip = st.file_uploader("Upload MRI scans (ZIP containing T1n, T1c, T2f, T2w)", type=['zip'])
    
    if uploaded_zip and model:
        with st.spinner("Processing scans..."):
            files = process_uploaded_zip(uploaded_zip)
            
            if None in files.values():
                st.error("Missing required scan files in the uploaded ZIP")
                return
            
            # Load and preprocess each modality
            modalities = {}
            for name, path in files.items():
                if name != 'seg':
                    modalities[name] = load_and_preprocess_nifti(path)
                    if modalities[name] is None:
                        return
            
            # Combine and crop channels
            combined = np.stack([
                modalities['t1n'],
                modalities['t1c'],
                modalities['t2f'],
                modalities['t2w']
            ], axis=-1)
            
            # Crop to target size
            combined = combined[
                CROP_PARAMS[0][0]:CROP_PARAMS[0][1],
                CROP_PARAMS[1][0]:CROP_PARAMS[1][1],
                CROP_PARAMS[2][0]:CROP_PARAMS[2][1],
                :
            ]
            
            # Load ground truth if available
            gt = None
            if files['seg']:
                gt = nib.load(files['seg']).get_fdata()
                gt = gt[
                    CROP_PARAMS[0][0]:CROP_PARAMS[0][1],
                    CROP_PARAMS[1][0]:CROP_PARAMS[1][1],
                    CROP_PARAMS[2][0]:CROP_PARAMS[2][1]
                ]
                gt[gt == 4] = 3  # Relabel tumor classes
            
            # Run prediction
            prediction = predict_volume(model, combined)
            
            if prediction is not None:
                st.success("Segmentation complete!")
                show_results(combined, prediction, gt)
                
                # Save results
                output_path = "segmentation_result.nii.gz"
                nib.save(
                    nib.Nifti1Image(
                        np.argmax(prediction, axis=-1).astype(np.float32),
                        np.eye(4)
                    ),
                    output_path
                )
                
                with open(output_path, "rb") as f:
                    st.download_button(
                        "Download Segmentation",
                        f,
                        file_name=output_path,
                        mime="application/octet-stream"
                    )
                
                os.remove(output_path)

if __name__ == "__main__":
    if model is None:
        st.error("Failed to load model. Cannot proceed.")
    else:
        main()
