"""
cnn_vgg16_landscape_crossv.py
@author: s231366 Lam Hui Yin (Henry)
"""

#Method inspired from https://www.learndatasci.com/tutorials/hands-on-transfer-learning-keras/
#Code inspired from https://github.com/gabrielcassimiro17/object-detection

# Import the required libraries
import os
import numpy as np
import tensorflow as tf
import keras
from keras.layers import Conv2D, UpSampling2D, Input
from keras.models import Sequential, Model
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import img_to_array, load_img
from skimage.color import rgb2lab, lab2rgb, gray2rgb
from skimage.transform import resize
import matplotlib.pyplot as plt
from tqdm import tqdm
from keras.callbacks import EarlyStopping
from sklearn.model_selection import KFold
from PIL import Image
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import Accuracy

# Import pre-trained model VGG16
from keras.applications.vgg16 import VGG16
vgg16model = VGG16(include_top=False, weights='imagenet')

# Create a new model and add layers from VGG16 up to the 19th layer (feature extraction)
feature_extraction_model = Sequential()
for i, layer in enumerate(vgg16model.layers):
    if i < 19:
        feature_extraction_model.add(layer)

feature_extraction_model.summary()

# Set all layers in the new model to be non-trainable
for layer in feature_extraction_model.layers:
    layer.trainable = False

# Path to the training data
path = '/work3/s231366/color/trainset/'
image_size = (224, 224)
batch_size = 5704

# Define the data generator and normalize the images
train_datagen = ImageDataGenerator(rescale=1. / 255)
train_generator = train_datagen.flow_from_directory(
    path,
    target_size=image_size,
    batch_size=batch_size,
    class_mode=None
)

# Convert images from RGB to Lab color space
L_channel = []
ab_channel = []

# Use tqdm to create a progress bar
with tqdm(total=batch_size, ncols=80) as pbar:
    for i in range(batch_size // batch_size):
        batch_start = i * batch_size
        batch_end = (i + 1) * batch_size
        batch_images = train_generator.next()

        for img in batch_images:
            lab = rgb2lab(img)
            L_channel.append(lab[:, :, 0])  # The L channel (lightness)
            ab_channel.append(lab[:, :, 1:] / 128)  # a and b channels (color components) divided by 128
        pbar.update(1)

# Convert L_channel and ab_channel to numpy arrays
L_channel = np.array(L_channel)
ab_channel = np.array(ab_channel)

# Reshape L_channel to match the dimensions of ab_channel
L_channel = L_channel.reshape(L_channel.shape + (1,))

# Print the shapes of L_channel and ab_channel
print(L_channel.shape)
print(ab_channel.shape)

# To match the input shape expected by VGG16, the single-channel L channel is repeated twice to create a three-channel input for the grayscale image.
vgg16features = []

# Use tqdm to create a progress bar
with tqdm(total=len(L_channel), ncols=80) as pbar:
    for i, sample in enumerate(L_channel):
        sample = gray2rgb(sample)
        sample = sample.reshape((1, 224, 224, 3))
        prediction = feature_extraction_model.predict(sample, verbose=0)
        prediction = prediction.reshape((7, 7, 512))
        vgg16features.append(prediction)

        pbar.update(1)

vgg16features = np.array(vgg16features)
print(vgg16features.shape)

# Define the number of folds (k)
k = 5

# Shuffle the data before cross-validation
indices = np.arange(len(L_channel))
np.random.shuffle(indices)

# Split data into k folds
kf = KFold(n_splits=k)

# Lists to store the history and model instances for each fold
histories = []
models = []

# Perform k-fold cross-validation
for fold_idx, (train_index, val_index) in enumerate(kf.split(indices)):
    print(f"Training on fold {fold_idx+1}/{k}")

    # Split data into training and validation sets for this fold
    L_channel_train, L_channel_val = vgg16features[train_index], vgg16features[val_index]
    ab_channel_train, ab_channel_val = ab_channel[train_index], ab_channel[val_index]

    # Create a new model for each fold (reset the weights)
    input_layer = Input(shape=(7, 7, 512))
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(input_layer)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    output_layer = Conv2D(2, (3, 3), activation='tanh', padding='same')(x)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.summary()

    optimizer = Adam()
    loss = MeanSquaredError()
    metrics = [Accuracy()]

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)


    # Train the model on this fold's data
    history = model.fit(L_channel_train, ab_channel_train, epochs=1000, verbose=1, validation_data=(L_channel_val, ab_channel_val),
                        shuffle=True, callbacks=[EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='auto')])

    # Save the history and model for this fold
    histories.append(history)
    models.append(model)

# Lists to store the training and validation loss and accuracy from each fold
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

# Function to save losses and accuracies to a text file
def save_results_to_txt(train_losses, train_accuracies, val_losses, val_accuracies):
    with open('/work3/s231366/losses_accuracies_newcrossv.txt', 'w') as file:
        file.write("Fold\tTrain Loss\tTrain Accuracy\tValidation Loss\tValidation Accuracy\n")
        for i in range(len(train_losses)):
            file.write(f"{i+1}\t{train_losses[i]:.4f}\t{train_accuracies[i]:.4f}\t{val_losses[i]:.4f}\t{val_accuracies[i]:.4f}\n")
        file.write("Average\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\n".format(
            np.mean(train_losses), np.mean(train_accuracies),
            np.mean(val_losses), np.mean(val_accuracies)))

# Evaluate the model performance on each fold and store the training and validation loss and accuracy
for fold_idx, (model, history) in enumerate(zip(models, histories)):
    print(f"Evaluating fold {fold_idx+1}/{k}")
    evaluation = model.evaluate(vgg16features[train_index], ab_channel_train)
    train_losses.append(evaluation[0])
    train_accuracies.append(evaluation[1])

    evaluation = model.evaluate(vgg16features[val_index], ab_channel_val)
    val_losses.append(evaluation[0])
    val_accuracies.append(evaluation[1])

    print("Training Loss:", evaluation[0])
    print("Training Accuracy:", evaluation[1])

    print("Validation Loss:", evaluation[0])
    print("Validation Accuracy:", evaluation[1])

# Calculate the average training and validation loss and accuracy
avg_train_loss = np.mean(train_losses)
avg_train_accuracy = np.mean(train_accuracies)
avg_val_loss = np.mean(val_losses)
avg_val_accuracy = np.mean(val_accuracies)

print("Average Training Loss:", avg_train_loss)
print("Average Training Accuracy:", avg_train_accuracy)
print("Average Validation Loss:", avg_val_loss)
print("Average Validation Accuracy:", avg_val_accuracy)

# Save the results to a text file
save_results_to_txt(train_losses, train_accuracies, val_losses, val_accuracies)

# Plot training loss and validation loss from history for each fold
plt.figure(figsize=(10, 6))
for fold_idx, history in enumerate(histories):
    plt.plot(history.history['loss'], label=f"Fold {fold_idx+1} Train")
    plt.plot(history.history['val_loss'], label=f"Fold {fold_idx+1} Validation")
plt.title('Training Loss vs Validation Loss for Each Fold')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('/work3/s231366/Training_Validation_Loss_newcrossv.png')
plt.clf()

# Plot training accuracy and validation accuracy from history for each fold
plt.figure(figsize=(10, 6))
for fold_idx, history in enumerate(histories):
    plt.plot(history.history['accuracy'], label=f"Fold {fold_idx+1} Train")
    plt.plot(history.history['val_accuracy'], label=f"Fold {fold_idx+1} Validation")
plt.title('Training Accuracy vs Validation Accuracy for Each Fold')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('/work3/s231366/Training_Validation_Accuracy_newcrossv.png')

# Save the model with the best overall performance based on validation loss
best_fold_idx = np.argmin(val_losses)
best_model = models[best_fold_idx]
best_model.save('/work3/s231366/best_model_colorization_newcrossv.model')


# Visualize the colorized results on the test images
testpath = '/work3/s231366/color/test/'
output_folder = '/work3/s231366/color/test_result/'

os.makedirs(output_folder, exist_ok=True)

files = os.listdir(testpath)
num_files = len(files)

# Create a single progress bar for the entire loop
with tqdm(total=num_files, desc="Colorizing images", unit="image") as pbar:
    for file in files:
        test = Image.open(os.path.join(testpath, file))
        test = test.resize((224, 224))
        test = np.array(test) / 255.0
        lab = rgb2lab(test)
        l = lab[:, :, 0]
        L = gray2rgb(l)
        L = L.reshape((1, 224, 224, 3))

        # Generate the colorized image
        vgg16prediction = feature_extraction_model.predict(L)
        ab = best_model.predict(vgg16prediction)
        ab = ab * 128
        cur = np.zeros((224, 224, 3))
        cur[:, :, 0] = l
        cur[:, :, 1:] = ab

        # Save the colorized image with the same filename as the grayscale input image
        filename_without_extension = os.path.splitext(file)[0]
        colorized_filename = 'result_' + filename_without_extension + '.jpg'
        save_path = os.path.join(output_folder, colorized_filename)
        colorized_image = lab2rgb(cur)
        colorized_image = np.clip(colorized_image, 0, 1)
        colorized_image = (colorized_image * 255).astype(np.uint8)
        colorized_image = Image.fromarray(colorized_image)
        colorized_image.save(save_path)

        # Update the progress bar
        pbar.update(1)

