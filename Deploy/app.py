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
import io  # Added for BytesIO
import shutil  # Added for file operations

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

def process_uploaded_zip(uploaded_zip):
    try:
        # Create a mapping of patterns to file types
        patterns = {
            '-t1n.': 't1n',
            '-t1c.': 't1c',
            '-t2f.': 't2f',
            '-t2w.': 't2w',
            '-seg.': 'seg'
        }
        
        # Create dictionary to store found files
        found_files = {ft: None for ft in patterns.values()}
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save the uploaded zip to a temporary file
            zip_path = os.path.join(tmpdir, "upload.zip")
            with open(zip_path, "wb") as f:
                f.write(uploaded_zip.getbuffer())
            
            # Extract all files while flattening the structure
            extracted_files = []
            with zipfile.ZipFile(zip_path, 'r') as z:
                for file_in_zip in z.namelist():
                    if file_in_zip.lower().endswith(('.nii.gz', '.nii')):
                        # Get the base filename
                        filename = os.path.basename(file_in_zip)
                        target_path = os.path.join(tmpdir, filename)
                        
                        # Handle potential name collisions
                        counter = 1
                        while os.path.exists(target_path):
                            name, ext = os.path.splitext(filename)
                            target_path = os.path.join(tmpdir, f"{name}_{counter}{ext}")
                            counter += 1
                        
                        # Extract the file
                        with open(target_path, 'wb') as out_file:
                            out_file.write(z.read(file_in_zip))
                        extracted_files.append(target_path)
            
            # Match files to types
            for filepath in extracted_files:
                filename = os.path.basename(filepath).lower()
                for pattern, file_type in patterns.items():
                    if pattern in filename and found_files[file_type] is None:
                        found_files[file_type] = filepath
                        break
            
            # Verify required files
            required_files = ['t1n', 't1c', 't2f', 't2w']
            missing = [ft for ft in required_files if not found_files[ft]]
            
            if missing:
                st.error(f"Missing required files: {', '.join(missing)}")
                st.info("Files found in ZIP:")
                for ft, path in found_files.items():
                    if path: st.info(f"{ft.upper()}: {os.path.basename(path)}")
                return None
            
            return found_files
            
    except Exception as e:
        st.error(f"ZIP processing failed: {str(e)}")
        return None

def load_and_preprocess_nifti(filepath):
    try:
        # Load using nibabel
        img = nib.load(filepath).get_fdata()
        
        # Preprocessing
        scaler = MinMaxScaler()
        return scaler.fit_transform(img.reshape(-1, img.shape[-1])).reshape(img.shape)
    except Exception as e:
        st.error(f"Error processing {filepath}: {str(e)}")
        return None

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
    plt.close(fig)  # Prevent memory leaks

def main():
    global model
    
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
    
    st.header("BRAIN TUMOR SEGMENTATION WITH 3D UNET")
    st.markdown("""
    ### Upload Instructions:
    1. Prepare a ZIP file containing these 4 scans:
       - T1 Native (filename must contain 't1n')
       - T1 Contrast (filename must contain 't1c')
       - T2 Flair (filename must contain 't2f')
       - T2 Weighted (filename must contain 't2w')
    2. Example valid names:  
       `BraTS-001-t1n.nii.gz`, `patient1_t1c.nii`, `case5_T2_FLAIR.nii.gz`
    """)
    
    uploaded_zip = st.file_uploader("Upload MRI scans ZIP", type=['zip'])
    
    if uploaded_zip:
        if model is None:
            st.error("No model available for prediction. Please try refreshing the page.")
            return
    
        with st.spinner("Processing scans..."):
            # Process the ZIP file
            files = process_uploaded_zip(uploaded_zip)
            if files is None:
                return
            
            # Load and preprocess each modality
            modalities = {}
            for name, path in files.items():
                if name != 'seg' and path:
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
            
            combined = combined[
                CROP_PARAMS[0][0]:CROP_PARAMS[0][1],
                CROP_PARAMS[1][0]:CROP_PARAMS[1][1],
                CROP_PARAMS[2][0]:CROP_PARAMS[2][1],
                :
            ]
            
            # Optional ground truth handling
            gt = None
            if files.get('seg'):
                try:
                    gt = nib.load(files['seg']).get_fdata()
                    gt = gt[
                        CROP_PARAMS[0][0]:CROP_PARAMS[0][1],
                        CROP_PARAMS[1][0]:CROP_PARAMS[1][1],
                        CROP_PARAMS[2][0]:CROP_PARAMS[2][1]
                    ]
                    gt[gt == 4] = 3
                except Exception as e:
                    st.warning(f"Could not load segmentation file: {str(e)}")
            
            # Run prediction
            prediction = predict_volume(model, combined)
            
            if prediction is not None:
                st.success("Segmentation complete!")
                show_results(combined, prediction, gt)
                
                # Save and offer download
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
