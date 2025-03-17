import streamlit as st
import numpy as np
import nibabel as nib
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import os
import matplotlib.pyplot as plt
from scipy.ndimage import rotate
from tensorflow.keras.utils import to_categorical

# Title of the app
st.title("Brain Tumor Segmentation using 3D U-Net (Lightweight Architecture for CPU based systems")

# Load the default model
@st.cache_resource  # Cache the model to avoid reloading on every interaction
def load_default_model():
    default_model_path = "saved_model/3D_unet_100_epochs_2_batch_patch_training.keras"
    model = load_model(default_model_path, compile=False)
    return model

default_model = load_default_model()

# Function to preprocess a NIfTI file
def preprocess_nifti(file_path, mask_path=None, patch_size=(128, 128, 128)):
    # Load the NIfTI file
    image = nib.load(file_path).get_fdata()
    
    # Normalize the image
    scaler = MinMaxScaler()
    image = scaler.fit_transform(image.reshape(-1, image.shape[-1])).reshape(image.shape)
    
    # Load and preprocess the mask if provided
    if mask_path:
        mask = nib.load(mask_path).get_fdata()
        mask = mask.astype(np.uint8)
        mask[mask == 4] = 3  # Reassign mask values 4 to 3
    else:
        mask = None
    
    # Crop to a size divisible by 64 (or any desired size)
    start_x = (image.shape[0] - patch_size[0]) // 2
    start_y = (image.shape[1] - patch_size[1]) // 2
    start_z = (image.shape[2] - patch_size[2]) // 2
    
    image = image[start_x:start_x + patch_size[0],
                  start_y:start_y + patch_size[1],
                  start_z:start_z + patch_size[2]]
    
    if mask is not None:
        mask = mask[start_x:start_x + patch_size[0],
                    start_y:start_y + patch_size[1],
                    start_z:start_z + patch_size[2]]
    
    # Expand dimensions for model input
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    image = np.expand_dims(image, axis=-1)  # Add channel dimension
    
    if mask is not None:
        mask = to_categorical(mask, num_classes=4)
        mask = np.expand_dims(mask, axis=0)  # Add batch dimension
    
    return image, mask

# Function to extract a patch from an image
def extract_patch(image, patch_size):
    img_shape = image.shape[1:4]  # Exclude batch and channel dimensions
    patch_x = np.random.randint(0, max(img_shape[0] - patch_size[0], 1))
    patch_y = np.random.randint(0, max(img_shape[1] - patch_size[1], 1))
    patch_z = np.random.randint(0, max(img_shape[2] - patch_size[2], 1))
    
    return image[:, patch_x:patch_x + patch_size[0], patch_y:patch_y + patch_size[1], patch_z:patch_z + patch_size[2], :]

# Function to augment an image
def augment_image(image):
    # Rotation
    angle = np.random.uniform(-15, 15)
    image = rotate(image, angle, axes=(1, 2), reshape=False, mode='reflect')
    
    # Flipping
    if np.random.rand() > 0.5:
        image = np.flip(image, axis=1)
    if np.random.rand() > 0.5:
        image = np.flip(image, axis=2)
    
    # Brightness Adjustment
    brightness = np.random.uniform(0.9, 1.1)
    image = np.clip(image * brightness, 0, 1)
    
    # Noise Addition (Gaussian noise)
    if np.random.rand() > 0.5:
        noise = np.random.normal(0, 0.02, image.shape)
        image = np.clip(image + noise, 0, 1)
    
    # Gamma Correction
    if np.random.rand() > 0.5:
        gamma = np.random.uniform(0.8, 1.2)
        image = np.clip(image ** gamma, 0, 1)
    
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
uploaded_mask = st.file_uploader("Upload a corresponding mask file (.nii.gz)", type=["nii.gz"])

if uploaded_file is not None:
    # Save the uploaded file temporarily
    with open("temp_file.nii.gz", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    if uploaded_mask is not None:
        with open("temp_mask.nii.gz", "wb") as f:
            f.write(uploaded_mask.getbuffer())
        mask_path = "temp_mask.nii.gz"
    else:
        mask_path = None
    
    # Preprocess the uploaded file
    input_image, input_mask = preprocess_nifti("temp_file.nii.gz", mask_path)
    
    # Extract a random patch and augment the image
    input_image = extract_patch(input_image, patch_size=(64, 64, 64))
    input_image = augment_image(input_image)
    
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
    if uploaded_mask is not None:
        os.remove("temp_mask.nii.gz")
    os.remove(output_file)
    if uploaded_model is not None:
        os.remove("temp_model.keras")
