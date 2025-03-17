import streamlit as st
import numpy as np
import nibabel as nib
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import os
import matplotlib.pyplot as plt
import gdown  # For downloading files from Google Drive
import zipfile  # To handle folder uploads
import tempfile  # To handle temporary files
from tensorflow.keras.utils import to_categorical

# Title of the app
st.title("Brain Tumor Segmentation using 3D U-Net - (Lightweight Architecture on Normal CPUs)")

# Function to download the default model from Google Drive
def download_default_model():
    # Google Drive file ID for the default model
    file_id = "1lV1SgafomQKwgv1NW2cjlpyb4LwZXFwX"  # Replace with your file ID
    output_path = "default_model.keras"
    
    # Download the file if it doesn't already exist
    if not os.path.exists(output_path):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output_path, quiet=False)
    
    return output_path

# Load the default model
@st.cache_resource  # Cache the model to avoid reloading on every interaction
def load_default_model():
    # Download the model from Google Drive
    model_path = download_default_model()
    model = load_model(model_path, compile=False)
    return model

default_model = load_default_model()

# Function to preprocess a NIfTI file
def preprocess_nifti(file_path):
    # Load the NIfTI file
    image = nib.load(file_path).get_fdata()
    
    # Normalize the image
    scaler = MinMaxScaler()
    image = scaler.fit_transform(image.reshape(-1, image.shape[-1])).reshape(image.shape)
    
    return image

# Function to combine 4 channels into a single 4-channel numpy array
def combine_channels(t1n, t1c, t2f, t2w):
    # Stack the 4 channels along the last axis
    combined_image = np.stack([t1n, t1c, t2f, t2w], axis=3)
    combined_image = combined_image[56:184, 56:184, 13:141]
    return combined_image

# Function to run segmentation
def run_segmentation(model, input_image):
    # Add batch and channel dimensions
    input_image = np.expand_dims(input_image, axis=0)  # Add batch dimension
    #input_image = np.expand_dims(input_image, axis=-1)  # Add channel dimension
    
    # Print the shape of input_image for debugging
    st.write(f"Shape of input_image: {input_image.shape}")
    
    # Ensure the input_image has the correct shape
    if len(input_image.shape) != 5:
        st.error(f"Unexpected shape for input_image: {input_image.shape}. Expected shape: (batch_size, height, width, depth, channels).")
        return None
    
    # Run prediction
    prediction = model.predict(input_image)
    prediction_argmax = np.argmax(prediction, axis=4)[0, :, :, :]
    return prediction_argmax

# Sidebar for model upload
st.sidebar.header("Upload Your Own Model")
uploaded_model = st.sidebar.file_uploader("Upload a Keras model (.keras)", type=["keras"])

# Load the model (default or uploaded)
if uploaded_model is not None:
    # Save the uploaded model temporarily
    with open("temp_model.keras", "wb") as f:
        f.write(uploaded_model.getbuffer())
    
    # Load the uploaded model
    try:
        model = load_model("temp_model.keras", compile=False)
        st.sidebar.success("Custom model loaded successfully!")
    except Exception as e:
        st.sidebar.error(f"Error loading custom model: {e}")
        st.sidebar.info("Using the default model instead.")
        model = default_model
else:
    model = default_model
    st.sidebar.info("Using the default model.")

# Main app: Upload a folder containing NIfTI files
st.header("Upload a Folder Containing NIfTI Files")
uploaded_folder = st.file_uploader("Upload a folder (as a zip file) containing T1n, T1c, T2f, T2w NIfTI files", type=["zip"])

if uploaded_folder is not None:
    # Create a temporary directory to extract the zip file
    with tempfile.TemporaryDirectory() as temp_dir:
        # Save the uploaded zip file
        zip_path = os.path.join(temp_dir, "uploaded_folder.zip")
        with open(zip_path, "wb") as f:
            f.write(uploaded_folder.getbuffer())
        
        # Extract the zip file
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(temp_dir)
        
        # Find the NIfTI files in the extracted folder
        t1n_path = None
        t1c_path = None
        t2f_path = None
        t2w_path = None
        mask_path = None
        
        for root, _, files in os.walk(temp_dir):
            for file in files:
                if file.endswith("t1n.nii.gz"):
                    t1n_path = os.path.join(root, file)
                elif file.endswith("t1c.nii.gz"):
                    t1c_path = os.path.join(root, file)
                elif file.endswith("t2f.nii.gz"):
                    t2f_path = os.path.join(root, file)
                elif file.endswith("t2w.nii.gz"):
                    t2w_path = os.path.join(root, file)
                elif file.endswith("seg.nii.gz"):
                    mask_path = os.path.join(root, file)
        
        # Check if all required files are found
        if t1n_path and t1c_path and t2f_path and t2w_path:
            # Preprocess each channel
            t1n = preprocess_nifti(t1n_path)
            t1c = preprocess_nifti(t1c_path)
            t2f = preprocess_nifti(t2f_path)
            t2w = preprocess_nifti(t2w_path)
            
            # Combine the 4 channels
            combined_image = combine_channels(t1n, t1c, t2f, t2w)
            
            # Print the shape of combined_image for debugging
            st.write(f"Shape of combined_image: {combined_image.shape}")
            
            # Ensure the combined_image has the correct shape
            if len(combined_image.shape) != 4:
                st.error(f"Unexpected shape for combined_image: {combined_image.shape}. Expected shape: (height, width, depth, channels).")
            else:
                # Run segmentation
                st.write("Running segmentation...")
                segmentation_result = run_segmentation(model, combined_image)
                
                # Display the segmentation result
                st.write("Segmentation completed! Displaying results...")
                
                # Load the ground truth mask if available
                if mask_path:
                    mask = nib.load(mask_path).get_fdata()
                    mask = mask.astype(np.uint8)
                    mask[mask == 4] = 3  # Reassign mask values 4 to 3
                    mask_argmax = np.argmax(to_categorical(mask, num_classes=4), axis=3)
                else:
                    mask_argmax = None
                
                # Select slice indices for visualization
                slice_indices = [75, 90, 100]  # Example: 25%, 50%, 75% of depth
                
                # Ensure slice indices are within bounds
                slice_indices = [min(idx, depth - 1) for idx in slice_indices]
                
                # Plotting Results
                fig, ax = plt.subplots(3, 4, figsize=(18, 12))
                
                for i, n_slice in enumerate(slice_indices):
                    # Rotate images to correct orientation
                    test_img_rotated = np.rot90(combined_image[:, :, n_slice, 0])  # Rotating 90 degrees
                    test_prediction_rotated = np.rot90(segmentation_result[:, :, n_slice])
                    
                    # Plotting Results
                    ax[i, 0].imshow(test_img_rotated, cmap='gray')
                    ax[i, 0].set_title(f'Testing Image - Slice {n_slice}')
                    
                    if mask_path:
                        test_mask_rotated = np.rot90(mask_argmax[:, :, n_slice])
                        ax[i, 1].imshow(test_mask_rotated)
                        ax[i, 1].set_title(f'Ground Truth - Slice {n_slice}')
                    else:
                        ax[i, 1].axis('off')  # Hide the ground truth subplot if no mask is available
                    
                    ax[i, 2].imshow(test_prediction_rotated)
                    ax[i, 2].set_title(f'Prediction - Slice {n_slice}')
                    
                    ax[i, 3].imshow(test_img_rotated, cmap='gray')
                    ax[i, 3].imshow(test_prediction_rotated, alpha=0.5)  # Overlay prediction mask
                    ax[i, 3].set_title(f'Overlay - Slice {n_slice}')
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # Save the segmentation result
                output_file = "segmentation_result.nii.gz"
                nib.save(nib.Nifti1Image(segmentation_result.astype(np.float32), np.eye(4)), output_file)
                
                # Provide a download link for the segmentation result
                with open(output_file, "rb") as f:
                    st.download_button(
                        label="Download Segmentation Result",
                        data=f,
                        file_name=output_file,
                        mime="application/octet-stream"
                    )
                
                # Clean up temporary files
                os.remove(output_file)
        else:
            st.error("The uploaded folder does not contain all required NIfTI files (T1n, T1c, T2f, T2w).")
