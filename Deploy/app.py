import streamlit as st
import numpy as np
import nibabel as nib
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import os
import matplotlib.pyplot as plt

# Title of the app
st.title("Brain Tumor Segmentation using 3D U-Net")

# Load the default model
@st.cache_resource  # Cache the model to avoid reloading on every interaction
def load_default_model():
    default_model_path = "saved_model/3D_unet_100_epochs_2_batch_patch_training.keras"
    model = load_model(default_model_path, compile=False)
    return model

default_model = load_default_model()

# Function to preprocess a NIfTI file
def preprocess_nifti(file_path, patch_size=(64, 64, 64)):
    # Load the NIfTI file
    image = nib.load(file_path).get_fdata()
    
    # Normalize the image
    scaler = MinMaxScaler()
    image = scaler.fit_transform(image.reshape(-1, image.shape[-1])).reshape(image.shape)
    
    # Pad or crop the image to match the patch size
    if image.shape != patch_size:
        # Example: Center cropping
        start_x = (image.shape[0] - patch_size[0]) // 2
        start_y = (image.shape[1] - patch_size[1]) // 2
        start_z = (image.shape[2] - patch_size[2]) // 2
        image = image[start_x:start_x + patch_size[0],
                      start_y:start_y + patch_size[1],
                      start_z:start_z + patch_size[2]]
    
    # Expand dimensions for model input
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    image = np.expand_dims(image, axis=-1)  # Add channel dimension
    return image

# Function to run segmentation
def run_segmentation(model, input_image):
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

# Main app: Upload NIfTI file
st.header("Upload a NIfTI File for Segmentation")
uploaded_file = st.file_uploader("Upload a NIfTI file (.nii.gz)", type=["nii.gz"])

if uploaded_file is not None:
    # Save the uploaded file temporarily
    with open("temp_file.nii.gz", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Preprocess the uploaded file
    input_image = preprocess_nifti("temp_file.nii.gz")
    
    # Run segmentation
    st.write("Running segmentation...")
    segmentation_result = run_segmentation(model, input_image)
    
    # Display the segmentation result
    st.write("Segmentation completed! Displaying results...")
    
    # Visualize a random slice
    n_slice = st.slider("Select a slice to visualize", 0, segmentation_result.shape[2] - 1, segmentation_result.shape[2] // 2)
    
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(input_image[0, :, :, n_slice, 0], cmap='gray')
    ax[0].set_title("Input Image")
    ax[1].imshow(segmentation_result[:, :, n_slice], cmap='viridis')
    ax[1].set_title("Segmentation Result")
    st.pyplot(fig)
    
    # Save the segmentation result
    output_file = "segmentation_result.nii.gz"
    nib.save(nib.Nifti1Image(segmentation_result, np.eye(4)), output_file)
    
    # Provide a download link for the segmentation result
    with open(output_file, "rb") as f:
        st.download_button(
            label="Download Segmentation Result",
            data=f,
            file_name=output_file,
            mime="application/octet-stream"
        )
    
    # Clean up temporary files
    os.remove("temp_file.nii.gz")
    os.remove(output_file)
    if uploaded_model is not None:
        os.remove("temp_model.keras")
