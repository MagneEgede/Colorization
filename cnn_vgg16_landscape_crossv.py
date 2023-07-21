"""
cnn_vgg16_landscape_crossv.py
@author: Henry
"""

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

# Set the GPU to be used by TensorFlow (optional, if you have GPU available)
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Import pre-trained model VGG16
from keras.applications.vgg16 import VGG16
vggmodel = VGG16(include_top=False, weights='imagenet')

# Create a new model and add layers from VGG16 up to the 19th layer (feature extraction)
newmodel = Sequential()
for i, layer in enumerate(vggmodel.layers):
    if i < 19:
        newmodel.add(layer)

newmodel.summary()

# Set all layers in the new model to be non-trainable
for layer in newmodel.layers:
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
X = []
Y = []

# Use tqdm to create a progress bar
with tqdm(total=batch_size, ncols=80) as pbar:
    for i in range(batch_size // batch_size):
        batch_start = i * batch_size
        batch_end = (i + 1) * batch_size
        batch_images = train_generator.next()

        for img in batch_images:
            try:
                lab = rgb2lab(img)
                X.append(lab[:, :, 0])  # The L channel (lightness)
                Y.append(lab[:, :, 1:] / 128)  # a and b channels (color components) divided by 128
            except:
                print('Error')

            pbar.update(1)

# Convert X and Y to numpy arrays
X = np.array(X)
Y = np.array(Y)

# Reshape X to match the dimensions of Y
X = X.reshape(X.shape + (1,))

# Print the shapes of X and Y
print(X.shape)
print(Y.shape)

# To match the input shape expected by VGG16, the single-channel L channel is repeated twice to create a three-channel input for the grayscale image.
vggfeatures = []

# Use tqdm to create a progress bar
with tqdm(total=len(X), ncols=80) as pbar:
    for i, sample in enumerate(X):
        sample = gray2rgb(sample)
        sample = sample.reshape((1, 224, 224, 3))
        prediction = newmodel.predict(sample, verbose=0)
        prediction = prediction.reshape((7, 7, 512))
        vggfeatures.append(prediction)

        pbar.update(1)

vggfeatures = np.array(vggfeatures)
print(vggfeatures.shape)

# Define the number of folds (k)
k = 5

# Shuffle the data before cross-validation
indices = np.arange(len(X))
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
    X_train, X_val = vggfeatures[train_index], vggfeatures[val_index]
    Y_train, Y_val = Y[train_index], Y[val_index]

    # Create a new model for each fold (reset the weights)
    model = Sequential()
    model.add(Conv2D(256, (3,3), activation='relu', padding='same', input_shape=(7,7,512)))
    model.add(Conv2D(128, (3,3), activation='relu', padding='same'))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(64, (3,3), activation='relu', padding='same'))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(32, (3,3), activation='relu', padding='same'))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(16, (3,3), activation='relu', padding='same'))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(2, (3, 3), activation='tanh', padding='same'))
    model.add(UpSampling2D((2, 2)))
    model.summary()

    model.compile(optimizer='Adam', loss='mse', metrics=['accuracy'])

    # Train the model on this fold's data
    history = model.fit(X_train, Y_train, epochs=1000, verbose=1, validation_data=(X_val, Y_val),
                        shuffle=True, callbacks=[EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='auto')])

    # Save the history and model for this fold
    histories.append(history)
    models.append(model)


# Lists to store the validation loss and accuracy from each fold
val_losses = []
val_accuracies = []

# Evaluate the model performance on each fold and store the validation loss and accuracy
for fold_idx, (model, history) in enumerate(zip(models, histories)):
    print(f"Evaluating fold {fold_idx+1}/{k}")
    evaluation = model.evaluate(vggfeatures[val_index], Y_val)
    val_losses.append(evaluation[0])
    val_accuracies.append(evaluation[1])
    print("Loss:", evaluation[0])
    print("Accuracy:", evaluation[1])

# Calculate the average validation loss and accuracy
avg_val_loss = np.mean(val_losses)
avg_val_accuracy = np.mean(val_accuracies)

print("Average Validation Loss:", avg_val_loss)
print("Average Validation Accuracy:", avg_val_accuracy)

# Plot loss and validation loss from history for each fold
plt.figure(figsize=(10, 6))
for fold_idx, history in enumerate(histories):
    plt.plot(history.history['loss'], label=f"Fold {fold_idx+1}")
plt.title('Training Loss for Each Fold')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('/work3/s231366/Training_Loss.png')
plt.show()

# Plot accuracy and validation accuracy from history for each fold
plt.figure(figsize=(10, 6))
for fold_idx, history in enumerate(histories):
    plt.plot(history.history['accuracy'], label=f"Fold {fold_idx+1}")
plt.title('Training Accuracy for Each Fold')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('/work3/s231366/Training_Accuracy.png')
plt.show()

# Save the model with the best overall performance based on validation loss
best_fold_idx = np.argmin(val_losses)
best_model = models[best_fold_idx]
best_model.save('/work3/s231366/best_model_colorization.model')


# Visualize the colorized results on the test images
testpath = '/work3/s231366/color/test/'
output_folder = '/work3/s231366/color/test_result/'

os.makedirs(output_folder, exist_ok=True)

files = os.listdir(testpath)
for file in tqdm(files, desc="Colorizing images", unit="image"):
    test = Image.open(os.path.join(testpath, file))
    test = test.resize((224, 224))
    test = np.array(test) / 255.0
    lab = rgb2lab(test)
    l = lab[:, :, 0]
    L = gray2rgb(l)
    L = L.reshape((1, 224, 224, 3))
    
    # Generate the colorized image
    vggpred = newmodel.predict(L)
    ab = best_model.predict(vggpred)
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

