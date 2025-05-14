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
import gdown
import requests

# Configure app
st.set_page_config(layout="wide")
st.title("Brain Tumor Segmentation using 3D U-Net")

# Constants
TARGET_SHAPE = (128, 128, 128, 4)
CROP_PARAMS = ((56, 184), (56, 184), (13, 141))  # y, x, z cropping

@st.cache_resource
def load_default_model():
    try:
        MODEL_PATH = "default_model.keras"
        MODEL_URL = "https://drive.google.com/uc?id=1lV1SgafomQKwgv1NW2cjlpyb4LwZXFwX"
        
        # Download model if it doesn't exist
        if not os.path.exists(MODEL_PATH):
            with st.spinner("Downloading model (65MB)... This may take a minute..."):
                gdown.download(MODEL_URL, MODEL_PATH, quiet=True)
                
                # Verify download completed
                if not os.path.exists(MODEL_PATH):
                    st.error("Model download failed. Please check your internet connection.")
                    return None
        
        # Load the model with custom objects if needed
        try:
            model = load_model(MODEL_PATH, compile=False)
            st.success("Model loaded successfully!")
            return model
        except Exception as load_error:
            st.error(f"Model loading failed: {str(load_error)}")
            # Try to clean up corrupted download
            if os.path.exists(MODEL_PATH):
                os.remove(MODEL_PATH)
            return None
            
    except Exception as e:
        st.error(f"Failed to initialize model: {str(e)}")
        return None

# Initialize model at the start
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
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save and extract zip
            zip_path = os.path.join(tmpdir, "upload.zip")
            with open(zip_path, "wb") as f:
                f.write(uploaded_zip.getbuffer())
            
            with zipfile.ZipFile(zip_path, 'r') as z:
                z.extractall(tmpdir)
            
            # Initialize files dictionary
            files = {
                't1n': None, 't1c': None, 
                't2f': None, 't2w': None,
                'seg': None
            }
            
            # Search through all extracted files
            for root, _, filenames in os.walk(tmpdir):
                for f in filenames:
                    f_lower = f.lower()
                    if f.endswith('.nii.gz') or f.endswith('.nii'):
                        full_path = os.path.join(root, f)
                        # Verify file actually exists before assigning
                        if os.path.exists(full_path):
                            if '-t1n.' in f_lower: files['t1n'] = full_path
                            elif '-t1c.' in f_lower: files['t1c'] = full_path
                            elif '-t2f.' in f_lower: files['t2f'] = full_path
                            elif '-t2w.' in f_lower: files['t2w'] = full_path
                            elif '-seg.' in f_lower: files['seg'] = full_path
            
            # Debug output
            st.success("Files found:")
            for file_type, path in files.items():
                if path:
                    st.info(f"{file_type.upper()}: {path}")
                else:
                    if file_type != 'seg':  # seg is optional
                        st.warning(f"{file_type.upper()}: Not found")
            
            # Verify required files
            required_files = ['t1n', 't1c', 't2f', 't2w']
            missing = [ft for ft in required_files if not files[ft] or not os.path.exists(files[ft])]
            
            if missing:
                st.error(f"Missing or inaccessible files: {', '.join(missing)}")
                st.info("Full directory structure:")
                for root, dirs, files_in_dir in os.walk(tmpdir):
                    st.info(f"Directory: {root}")
                    for f in files_in_dir:
                        st.info(f" - {f}")
                return None
            
            return files
    except Exception as e:
        st.error(f"Error processing ZIP file: {str(e)}")
        return None
        
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
    global model
    
    # Model upload section (keep this unchanged)
    with st.sidebar:
        st.header("Model Configuration")
        uploaded_model = st.file_uploader("Upload custom model (.keras)", type=['keras'])
        
        if uploaded_model:
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.keras') as tmp:
                    tmp.write(uploaded_model.getbuffer())
                    tmp_path = tmp.name
                
                try:
                    new_model = load_model(tmp_path, compile=False)
                    model = new_model
                    st.success("Custom model loaded successfully!")
                finally:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
            except Exception as e:
                st.error(f"Failed to load custom model: {str(e)}")
                st.info("Reverting to default model")
                model = load_default_model()
    
    # Main processing section - THIS IS THE FIXED PART
    st.header("BRAIN TUMOR SEGMENTATION WITH 3D UNET")
    uploaded_zip = st.file_uploader("Upload MRI scans (ZIP containing T1n, T1c, T2f, T2w)", type=['zip'])
    
    if uploaded_zip:
        if model is None:
            st.error("No model available for prediction. Please try refreshing the page.")
            return
    
        with st.spinner("Processing scans..."):
            files = process_uploaded_zip(uploaded_zip)
            st.info("Resolved file paths:")
            for k, v in files.items():
                if v: st.info(f"{k}: {v} (exists: {os.path.exists(v)})")
        
            # Updated check - only validates required files (excluding SEG)
            required_files = ['t1n', 't1c', 't2f', 't2w']
            if files is None or any(files[ft] is None for ft in required_files):
                st.error("Missing required scan files in the uploaded ZIP")
                return
            
            # Rest of your processing code remains unchanged...
            modalities = {}
            
            for name, path in files.items():
                if name != 'seg' and path:  # Only process if path exists
                    try:
                        modalities[name] = load_and_preprocess_nifti(path)
                        if modalities[name] is None:
                            return
                    except Exception as e:
                        st.error(f"Failed to load {name} from {path}: {str(e)}")
                        return
            
            combined = np.stack([
                modalities['t1n'],
                modalities['t1c'],
                modalities['t2f'],
                modalities['t2w']
            ], axis=-1)
            
            combined = combined[
                CROP_PARAMS[0][0]:CROP_PARAMS[0][1],
                CROP_PARAMS[1][0]:CROP_PARAMS[1][1],
                CROP_PARAMS[2][0]:CROP_PARAMS[2][1],
                :
            ]
            
            # Optional ground truth handling
            gt = None
            if files['seg']:  # This is now safely optional
                try:
                    gt = nib.load(files['seg']).get_fdata()
                    gt = gt[
                        CROP_PARAMS[0][0]:CROP_PARAMS[0][1],
                        CROP_PARAMS[1][0]:CROP_PARAMS[1][1],
                        CROP_PARAMS[2][0]:CROP_PARAMS[2][1]
                    ]
                    gt[gt == 4] = 3
                except:
                    st.warning("Could not load segmentation file (optional)")
            
            prediction = predict_volume(model, combined)
            
            if prediction is not None:
                st.success("Segmentation complete!")
                show_results(combined, prediction, gt)
                
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
                
                if os.path.exists(output_path):
                    os.remove(output_path)
if __name__ == "__main__":
    if model is None:
        st.error("Failed to load model. Please check your internet connection and try again.")
    else:
        main()
