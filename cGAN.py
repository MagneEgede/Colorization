# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#%% Load libraries
import matplotlib.pyplot as plt
import numpy as np

from skimage.io import imread_collection
from skimage.color import rgb2lab, lab2rgb

from keras.optimizers import Adam
from keras.models import Sequential, load_model, Model
from keras.layers import Dense, Flatten, Conv2D,Input,ZeroPadding2D, Concatenate
from keras.layers import Conv2DTranspose, LeakyReLU, Dropout, BatchNormalization
from keras.utils import plot_model

import tensorflow as tf
tf.config.run_functions_eagerly(True)
tf.data.experimental.enable_debug_mode()
import os
print(os.getcwd())
#%% Load the data 
# read all images 
img_color = imread_collection('data/color/*.jpg')
img_color = [img for img in img_color]

# Remove all images without 150, 150, 3
img_selection = [img_color[i].shape == (150,150,3)for i in range(len(img_color))]
img_color = np.array(img_color)[img_selection]

img_color = np.array([rgb2lab(img_color[i]) for i in range(len(img_color))])

#%% Load train / test config
train = np.loadtxt("trainset.csv",
                 delimiter=",", dtype=int)
#%%
test = np.where(train[:len(img_color)] == 0)
train = np.where(train[:len(img_color)] == 1)
print(train)

#%% Test and train 
# Separate test and train set
img_l_train = img_color[train][:,:,:,2]
img_l_test = img_color[test][:,:,:,2]

img_real_train = img_color[train]
img_real_test = img_color[test]



# plot 25 color images 
for i in range(4):
	plt.subplot(2, 2, 1 + i)
	plt.axis('off')
	plt.imshow(img_color[i,:,:,0], cmap='gray')
plt.show()


#print(f'Shape of the color image {img_color[0].shape}')
#print(f'Number of color images {len(img_color)}')
del img_color

#%% Generator model 
def define_generator(in_shape):
    # Encoder 
    input_img = Input(shape=in_shape)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = BatchNormalization()(x)
    x = Conv2D(64, (3, 3), activation='relu', strides=(2, 2))(x)
    x = BatchNormalization()(x)
    x = Conv2D(128, (3, 3), activation='relu', strides=(2, 2))(x)
    x = BatchNormalization()(x)
    x = Conv2D(256, (3, 3), activation='relu', strides=(2, 2), padding='same')(x)
    x = BatchNormalization()(x)

    # Decoder
    x = Conv2DTranspose(128, (3, 3), activation='relu', strides=(2, 2), padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2DTranspose(64, (3, 3), activation='relu', strides=(2, 2), padding='same')(x)
    x = BatchNormalization()(x)
    x = ZeroPadding2D(padding=(1,1))(x)
    x = Conv2DTranspose(32, (3, 3), activation='relu', strides=(2, 2), padding='same')(x)
    x = BatchNormalization()(x)
    x = ZeroPadding2D(padding=(1,1))(x)
    decoded = Conv2DTranspose(3, (3, 3), activation='sigmoid', padding='same')(x)
    return Model(input_img, decoded, name = "Generator_CAE")
generator = define_generator((150,150,1))
generator.summary()

#%% Define discriminator 
def define_discriminator(in_shape_src = (150,150,1), input_shape_target = (150,150,3) ):
    
	in_src_image = Input(shape=in_shape_src, name="original_bw")
	in_target_image = Input(shape=input_shape_target,name="evaluate_color")

	merged = Concatenate(axis=3)([in_src_image, in_target_image])
	
	d = Conv2D(128, (3,3), strides=(2,2), padding='same')(merged) #16x16x128
	d = LeakyReLU(alpha=0.2)(d)
	
	d = Conv2D(128, (3,3), strides=(2,2), padding='same')(d) #8x8x128
	d = LeakyReLU(alpha=0.2)(d)
	
	d = Flatten()(d) #shape of 8192
	d = Dropout(0.4)(d)
	output = Dense(1, activation='sigmoid')(d) #shape of 1
    
	model = Model([in_src_image, in_target_image],output, name='Discriminator')
	# compile model
	opt = Adam(learning_rate=0.0002, beta_1=0.5)
	model.compile(loss='binary_crossentropy',
               optimizer=opt)
	return model

discriminator = define_discriminator()
discriminator.summary()

    
#%% Combined GAN
def define_gan(generator, discriminator, img_shape_gen=(150,150,1)):
    # In order to either train generator or discriminator
    discriminator.trainable =  False
 
    in_src_gen = Input(shape=img_shape_gen)
    
    # 
    output_generator = generator(in_src_gen)
    output_discriminator = discriminator([in_src_gen, output_generator])
    
    model = Model(inputs = in_src_gen,outputs = [output_generator,output_discriminator], name = "cGAN")
    # Evaluate the model
    opt = Adam(learning_rate=0.0002, beta_1=0.5)
    model.compile(loss=['binary_crossentropy','mean_absolute_error'],loss_weights = [1,100],
                  optimizer=opt)
    return model
cgan = define_gan(generator, discriminator)
cgan.summary()

#%%
print("Order of Inputs:")
for input_tensor in cgan.inputs:
    print(input_tensor)
    print("\n")

#%% Train function
def train(gan, generator, discriminator, dataset_real, dataset_l, epochs = 20, img_batch = 2):
    batch_per_epoch = int(np.ceil(dataset_real.shape[0]/img_batch))
    half_batch = int(img_batch/2)
    

    for epoch in range(epochs):
        for batch in range(batch_per_epoch):
            # Train the discriminator
            ## Select real images
            imgs = np.random.randint(0,len(dataset_real), half_batch)
            img_gray = dataset_l[imgs] 
            img_color = dataset_real[imgs]
            y_real = np.ones((half_batch,1))
            ## Train discriminator on real images
            cost_d_real = discriminator.train_on_batch([img_gray, img_color], y_real)

            
            ## Select fake images 
            imgs = np.random.randint(0,len(dataset_l), half_batch)
            img_gray = dataset_l[imgs]
            img_generated = generator.predict(img_gray)
            img_fake = [img_gray, img_generated ]
            
            y_fake = np.zeros((half_batch,1))
            ## Train discriminator on fake images
            cost_d_fake = discriminator.train_on_batch([img_gray, img_generated],y_fake)

            
            # Train the cGAN
            imgs = np.random.randint(0,len(dataset_l), img_batch)
            img_gray = dataset_l[imgs].reshape(len(imgs),150,150,1)
            img_real = dataset_real[imgs].reshape(len(imgs),150,150,3)
            y_gan = np.ones((img_batch,1))
            
            with tf.device('/CPU:0'):
                cost_gan = cgan.train_on_batch(img_gray,[img_real,y_gan])

            print('Epoch>%d, Batch %d/%d, d1=%.3f, d2=%.3f g=%.3f' %
                (epoch+1, batch+1, batch_per_epoch, cost_d_real, cost_d_fake, cost_gan[0]))
            if batch % 150 == 0:
                generator.save('generator')
        # Save the generator
        print(epoch)

#%% train model
train(cgan, generator, discriminator, img_real_train, img_l_train)

#%%
plot_model(cgan,show_dtype = True, show_shapes=True, to_file = 'cgan.jpg')
plot_model(generator,show_dtype = True, show_shapes=True,to_file = 'CAE.jpg')
plot_model(discriminator,show_dtype = True, show_shapes=True,to_file = 'CNN.jpg')


#%% Load trained model 
generator = load_model('generator')

#%% Generate Img
img = img_l_test[14,:,:]
img_col = generator.predict(img.reshape(1,150,150,1))
img_col.reshape(150,150,3)

#img_col = lab2rgb(img_col[0])
fig, ax = plt.subplots(1,2)
ax[0].imshow(img,cmap='gray')
ax[1].imshow(img_col[0])
