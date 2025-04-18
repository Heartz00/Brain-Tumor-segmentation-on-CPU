import numpy as np
import nibabel as nib
import glob
import random
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from tifffile import imwrite 
from sklearn.preprocessing import MinMaxScaler
import os
import splitfolders

scaler = MinMaxScaler()

# Define the path to your dataset
TRAIN_DATASET_PATH = 'BraTS-Africa/95_Glioma'

# Load sample images and visualize
image_t1n = nib.load(TRAIN_DATASET_PATH + '/BraTS-SSA-00008-000/BraTS-SSA-00008-000-t1n.nii.gz').get_fdata()
image_t1c = nib.load(TRAIN_DATASET_PATH + '/BraTS-SSA-00008-000/BraTS-SSA-00008-000-t1c.nii.gz').get_fdata()
image_t2f = nib.load(TRAIN_DATASET_PATH + '/BraTS-SSA-00008-000/BraTS-SSA-00008-000-t2f.nii.gz').get_fdata()
image_t2w = nib.load(TRAIN_DATASET_PATH + '/BraTS-SSA-00008-000/BraTS-SSA-00008-000-t2w.nii.gz').get_fdata()
mask = nib.load(TRAIN_DATASET_PATH + '/BraTS-SSA-00008-000/BraTS-SSA-00008-000-seg.nii.gz').get_fdata()

# Scale the images
image_t1n = scaler.fit_transform(image_t1n.reshape(-1, image_t1n.shape[-1])).reshape(image_t1n.shape)
image_t1c = scaler.fit_transform(image_t1c.reshape(-1, image_t1c.shape[-1])).reshape(image_t1c.shape)
image_t2f = scaler.fit_transform(image_t2f.reshape(-1, image_t2f.shape[-1])).reshape(image_t2f.shape)
image_t2w = scaler.fit_transform(image_t2w.reshape(-1, image_t2w.shape[-1])).reshape(image_t2w.shape)

# Convert mask to uint8 and re-label
mask = mask.astype(np.uint8)
np.unique(mask)
mask[mask == 4] = 3  # Reassign mask values 4 to 3

# Visualize a random slice
import numpy as np
import matplotlib.pyplot as plt
import random

n_slice = random.randint(0, mask.shape[2])

plt.figure(figsize=(12, 8))

plt.subplot(231)
plt.imshow(np.rot90(image_t1n[:, :, n_slice]), cmap='gray')
plt.title('Image t1n')

plt.subplot(232)
plt.imshow(np.rot90(image_t1c[:, :, n_slice]), cmap='gray')
plt.title('Image t1c')

plt.subplot(233)
plt.imshow(np.rot90(image_t2f[:, :, n_slice]), cmap='gray')
plt.title('Image t2f')

plt.subplot(234)
plt.imshow(np.rot90(image_t2w[:, :, n_slice]), cmap='gray')
plt.title('Image t2w')

plt.subplot(235)
plt.imshow(np.rot90(mask[:, :, n_slice]))
plt.title('Mask')

plt.show()


# Combine images into a multi-channel array
combined_x = np.stack([image_t1n, image_t1c, image_t2f, image_t2w], axis=3)

# Crop to a size divisible by 64 (or any desired size)
combined_x = combined_x[56:184, 56:184, 13:141]  # Example cropping to 128x128x128x4
mask = mask[56:184, 56:184, 13:141]

# Visualize the combined image
n_slice = random.randint(0, mask.shape[2])
plt.figure(figsize=(12, 8))

plt.subplot(221)
plt.imshow(combined_x[:,:,n_slice, 0], cmap='gray')
plt.title('Image t1n')
plt.subplot(222)
plt.imshow(combined_x[:,:,n_slice, 1], cmap='gray')
plt.title('Image t1c')
plt.subplot(223)
plt.imshow(combined_x[:,:,n_slice, 2], cmap='gray')
plt.title('Image t2f')
plt.subplot(224)
plt.imshow(combined_x[:,:,n_slice, 3], cmap='gray')
plt.title('Image t2w')
plt.show()

# Save the combined image and mask
imwrite('combined_image.tif', combined_x)
np.save('combined_image.npy', combined_x)
mask = to_categorical(mask, num_classes=4)


os.makedirs('glioma/images', exist_ok=True)
os.makedirs('glioma/masks', exist_ok=True)

# Process all images in the dataset
t1n_list = sorted(glob.glob(TRAIN_DATASET_PATH + '/*/*t1n.nii.gz'))
t1c_list = sorted(glob.glob(TRAIN_DATASET_PATH + '/*/*t1c.nii.gz'))
t2f_list = sorted(glob.glob(TRAIN_DATASET_PATH + '/*/*t2f.nii.gz'))
t2w_list = sorted(glob.glob(TRAIN_DATASET_PATH + '/*/*t2w.nii.gz'))
mask_list = sorted(glob.glob(TRAIN_DATASET_PATH + '/*/*seg.nii.gz'))

for img in range(len(t1n_list)):  # Using t1n_list as all lists are of the same size
    print("Now preparing image and masks number: ", img)

    temp_image_t1n = nib.load(t1n_list[img]).get_fdata()
    temp_image_t1n = scaler.fit_transform(temp_image_t1n.reshape(-1, temp_image_t1n.shape[-1])).reshape(temp_image_t1n.shape)

    temp_image_t1c = nib.load(t1c_list[img]).get_fdata()
    temp_image_t1c = scaler.fit_transform(temp_image_t1c.reshape(-1, temp_image_t1c.shape[-1])).reshape(temp_image_t1c.shape)

    temp_image_t2f = nib.load(t2f_list[img]).get_fdata()
    temp_image_t2f = scaler.fit_transform(temp_image_t2f.reshape(-1, temp_image_t2f.shape[-1])).reshape(temp_image_t2f.shape)

    temp_image_t2w = nib.load(t2w_list[img]).get_fdata()
    temp_image_t2w = scaler.fit_transform(temp_image_t2w.reshape(-1, temp_image_t2w.shape[-1])).reshape(temp_image_t2w.shape)

    temp_mask = nib.load(mask_list[img]).get_fdata()
    temp_mask = temp_mask.astype(np.uint8)
    temp_mask[temp_mask == 4] = 3  # Reassign mask values 4 to 3

    temp_combined_images = np.stack([temp_image_t1n, temp_image_t1c, temp_image_t2f, temp_image_t2w], axis=3)

    # Crop to a size divisible by 64 (or any desired size)
    temp_combined_images = temp_combined_images[56:184, 56:184, 13:141]
    temp_mask = temp_mask[56:184, 56:184, 13:141]

    val, counts = np.unique(temp_mask, return_counts=True)

    if (1 - (counts[0] / counts.sum())) > 0.01:  # At least 1% useful volume with labels that are not 0
        print("Save Me")
        temp_mask = to_categorical(temp_mask, num_classes=4)
        np.save('glioma/images/image_' + str(img) + '.npy', temp_combined_images)
        np.save('glioma/masks/mask_' + str(img) + '.npy', temp_mask)
    else:
        print("I am not a good addition to the model")


# Split the data into training and validation sets

os.makedirs('glioma split data', exist_ok=True)

input_folder = 'glioma/'
output_folder = 'glioma split data/'
splitfolders.ratio(input_folder, output=output_folder, seed=42, ratio=(.75, .15, .10), group_prefix=None)
