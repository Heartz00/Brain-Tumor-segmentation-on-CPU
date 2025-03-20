# DICE SCORE CUSTOM FUNCTION 
class DiceScore(tf.keras.metrics.Metric):
    def __init__(self, num_classes, class_weights=None, smooth=1e-6, **kwargs):
        super(DiceScore, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.smooth = smooth
        self.class_weights = class_weights if class_weights is not None else tf.ones(num_classes)  # Default to equal weights
        self.dice_scores = self.add_weight(name='dice_scores', shape=(self.num_classes,), initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Flatten the tensors to ensure computations are class-wise
        y_true = tf.reshape(y_true, [-1])  # Flatten ground truth
        y_pred = tf.reshape(y_pred, [-1])  # Flatten predictions

        # Initialize the dice scores for each class
        dice_scores = []

        for i in range(self.num_classes):
            # Create binary masks for class i
            y_true_class = tf.cast(tf.equal(y_true, i), 'float32')
            y_pred_class = tf.cast(tf.equal(tf.round(y_pred), i), 'float32')

            # Calculate intersection and union
            intersection = tf.reduce_sum(y_true_class * y_pred_class)
            union = tf.reduce_sum(y_true_class) + tf.reduce_sum(y_pred_class)

            # Compute Dice score for the current class
            dice_class = (2. * intersection + self.smooth) / (union + self.smooth)

            # Apply class weight to the Dice score
            weighted_dice_class = dice_class * self.class_weights[i]
            dice_scores.append(weighted_dice_class)

        # Update state by averaging weighted dice scores across all classes
        dice_scores = tf.stack(dice_scores)
        self.dice_scores.assign(dice_scores)

    def result(self):
        # Return the mean of the weighted dice scores
        return tf.reduce_mean(self.dice_scores)

    def reset_states(self):
        # Reset the dice scores at the start of each batch
        self.dice_scores.assign(tf.zeros(self.num_classes))




# MODEL TRAINING 
import time

start = time.time()
# Set up loss function with class weights
wt0, wt1, wt2, wt3 = 0.25, 0.25, 0.25, 0.25
class_weights = np.array([wt0, wt1, wt2, wt3], dtype=np.float32)

dice_loss = sm.losses.DiceLoss(class_weights=class_weights)
focal_loss = sm.losses.CategoricalFocalLoss()
total_loss = 0.2 * dice_loss + 0.2 * focal_loss  # focal loss of 0.2 (perf best now) becomes 0.1

# Define metrics
metrics = ['accuracy', sm.metrics.IOUScore(threshold=0.5), DiceScore(num_classes=4)]

# Learning rate scheduler (Cosine Decay)
lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5 # from 6 to 4(perf better) to 3
)

# Optimizer (use a fixed learning rate)
optimizer = keras.optimizers.Nadam(learning_rate=0.001, clipnorm=1.0) #increase from 0.001(perf best) to 0.01

# Data paths
DATA_ROOT = "glioma split data"
train_img_dir, train_mask_dir = f"{DATA_ROOT}/train/images/", f"{DATA_ROOT}/train/masks/"
val_img_dir, val_mask_dir = f"{DATA_ROOT}/val/images/", f"{DATA_ROOT}/val/masks/"

# Load data
train_img_list, train_mask_list = os.listdir(train_img_dir), os.listdir(train_mask_dir)
val_img_list, val_mask_list = os.listdir(val_img_dir), os.listdir(val_mask_dir)

# Data generators
batch_size = 2
patch_size = (64, 64, 64)
train_data = imageLoader(train_img_dir, train_img_list, train_mask_dir, train_mask_list, batch_size, patch_size)
val_data = imageLoader(val_img_dir, val_img_list, val_mask_dir, val_mask_list, batch_size, patch_size)

# Training parameters
steps_per_epoch = len(train_img_list) // batch_size
val_steps_per_epoch = len(val_img_list) // batch_size
epochs = 100

# Initialize and compile model
model = simple_unet_model(IMG_HEIGHT=64, IMG_WIDTH=64, IMG_DEPTH=64, IMG_CHANNELS=4, num_classes=4)
model.compile(optimizer=optimizer, loss=total_loss, metrics=metrics)
model.summary()

# Define callbacks
checkpoint_path = f"saved_model/3D_unet_{epochs}_epochs_{batch_size}_batch_patch_training.keras"
callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True), #increase patience from 10 to 20
    ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True, mode='min'),
    lr_scheduler  # Learning rate decay callback
]

# Train model
history = model.fit(
    train_data,
    steps_per_epoch=steps_per_epoch,
    epochs=epochs,
    validation_data=val_data,
    validation_steps=val_steps_per_epoch,
    callbacks=callbacks,
    verbose=1
)

# Save training history
history_df = pd.DataFrame(history.history)
os.makedirs("model_history", exist_ok=True)
history_df.to_csv("model_history/training_history.csv", index=False)
print("Training history and best model saved successfully.")

end = time.time()
exec_time = (end - start)/60
print(f'execution time is - {exec_time}')



# EVALUATION
import os
import numpy as np
import keras
from keras.models import load_model
from keras.metrics import MeanIoU
from matplotlib import pyplot as plt

# -------------------- MODEL LOADING --------------------
# Path to trained model
model_path = 'saved_model/3D_unet_100_epochs_2_batch_patch_training.keras'

# Load the trained model without recompiling (for inference only)
my_model = load_model(model_path, compile=False)

# -------------------- DATA LOADING --------------------
# Path to dataset (Modify as needed)
DATA_PATH = "glioma split data"

# Train and Validation Directories
train_img_dir = os.path.join(DATA_PATH, "train/images/")
train_mask_dir = os.path.join(DATA_PATH, "train/masks/")
val_img_dir = os.path.join(DATA_PATH, "val/images/")
val_mask_dir = os.path.join(DATA_PATH, "val/masks/")
test_img_dir = os.path.join(DATA_PATH, "test/images/")
test_mask_dir = os.path.join(DATA_PATH, "test/masks/")

# Get list of images and masks
train_img_list, train_mask_list = os.listdir(train_img_dir), os.listdir(train_mask_dir)
val_img_list, val_mask_list = os.listdir(val_img_dir), os.listdir(val_mask_dir)
test_img_list, test_mask_list = os.listdir(test_img_dir), os.listdir(test_mask_dir)

# Define patch and batch size
patch_size = (64, 64, 64)
batch_size = 2

# -------------------- VALIDATION DATA EVALUATION --------------------
# Create Data Generator for validation set
val_img_loader = imageLoader(val_img_dir, val_img_list, val_mask_dir, val_mask_list, batch_size, patch_size)

# Fetch a batch for evaluation
val_img_batch, val_mask_batch = val_img_loader.__next__()

# Convert masks to argmax format
val_mask_argmax = np.argmax(val_mask_batch, axis=4)

# Model prediction
val_pred_batch = my_model.predict(val_img_batch)
val_pred_argmax = np.argmax(val_pred_batch, axis=4)

# Compute Mean IoU
#sm.metrics.IOUScore(threshold=0.5)
n_classes = 4
iou_metric = MeanIoU(num_classes=n_classes)
iou_metric.update_state(val_pred_argmax, val_mask_argmax)
val_iou_score = iou_metric.result().numpy()

# Compute Dice Score
dice_metric = DiceScore(num_classes=n_classes)
dice_metric.update_state(val_mask_argmax, val_pred_argmax)
val_dice_score = dice_metric.result().numpy()

print(f"Validation Mean IoU: {val_iou_score}")
print(f"Validation Dice Score: {val_dice_score}")

# -------------------- TEST DATA EVALUATION --------------------
# Create Data Generator for test set
test_img_loader = imageLoader(test_img_dir, test_img_list, test_mask_dir, test_mask_list, batch_size, patch_size)

# Fetch a batch for evaluation
test_img_batch, test_mask_batch = test_img_loader.__next__()

# Convert masks to argmax format
test_mask_argmax = np.argmax(test_mask_batch, axis=4)

# Model prediction
test_pred_batch = my_model.predict(test_img_batch)
test_pred_argmax = np.argmax(test_pred_batch, axis=4)

# Compute Mean IoU
iou_metric.update_state(test_pred_argmax, test_mask_argmax)
test_iou_score = iou_metric.result().numpy()

# Compute Dice Score
dice_metric.update_state(test_mask_argmax, test_pred_argmax)
test_dice_score = dice_metric.result().numpy()

print(f"Test Mean IoU: {test_iou_score}")
print(f"Test Dice Score: {test_dice_score}")


# VISUALIZATION
# -------------------- VISUALIZATION --------------------
# Select a random test image for visualization
img_num = 21  # Change index as needed
test_img = np.load(os.path.join(test_img_dir, f"image_{img_num}.npy"))
test_mask = np.load(os.path.join(test_mask_dir, f"mask_{img_num}.npy"))
test_mask_argmax = np.argmax(test_mask, axis=3)

# Expand dimensions for model prediction
test_img_input = np.expand_dims(test_img, axis=0)
test_prediction = my_model.predict(test_img_input)
test_prediction_argmax = np.argmax(test_prediction, axis=4)[0, :, :, :]

# Select slice indices for visualization
slice_indices = [75, 90, 100]  # Change slice indices as needed

# Plotting Results
plt.figure(figsize=(18, 12))

for i, n_slice in enumerate(slice_indices):
    # Rotate images to correct orientation
    test_img_rotated = np.rot90(test_img[:, :, n_slice, 1])  # Rotating 90 degrees
    test_mask_rotated = np.rot90(test_mask_argmax[:, :, n_slice])
    test_prediction_rotated = np.rot90(test_prediction_argmax[:, :, n_slice])

    # Plotting Results
    plt.subplot(3, 4, i*4 + 1)
    plt.title(f'Testing Image - Slice {n_slice}')
    plt.imshow(test_img_rotated, cmap='gray')

    plt.subplot(3, 4, i*4 + 2)
    plt.title(f'Ground Truth - Slice {n_slice}')
    plt.imshow(test_mask_rotated)

    plt.subplot(3, 4, i*4 + 3)
    plt.title(f'Prediction - Slice {n_slice}')
    plt.imshow(test_prediction_rotated)

    plt.subplot(3, 4, i*4 + 4)
    plt.title(f'Overlay - Slice {n_slice}')
    plt.imshow(test_img_rotated, cmap='gray')
    plt.imshow(test_prediction_rotated, alpha=0.5)  # Overlay prediction mask

plt.tight_layout()
plt.show()


