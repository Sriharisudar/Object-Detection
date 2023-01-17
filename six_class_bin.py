"""importing neccessay lib"""
from numba import njit
import numpy as np
import cv2
import os
import tensorflow as tf
from tensorflow.keras.utils import image_dataset_from_directory

from tensorflow import keras
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint,EarlyStopping,ReduceLROnPlateau
from keras.layers import Conv2D, Flatten, MaxPooling2D,Dense,Dropout,SpatialDropout2D
from keras.models  import Sequential
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img, array_to_img

import random,glob
import matplotlib.pyplot as plt

#@njit
#path of data_set
DATA_DIR = "E:\AI_bin\dataset\min_12"
CLASSES = os.listdir(DATA_DIR)
print(CLASSES)

#resizing image
IMAGE_SIZE = (300, 300)
BATCH_SIZE = 16

#data split up
train_dataset = image_dataset_from_directory(
    DATA_DIR,
    subset= "training",
    validation_split = 0.2,
    seed=1,
    image_size = IMAGE_SIZE,
    batch_size = BATCH_SIZE,
    label_mode = "categorical",
    class_names = CLASSES
)


test_dataset = image_dataset_from_directory(
    DATA_DIR,
    subset= "validation",
    validation_split = 0.2,
    seed=1,
    image_size = IMAGE_SIZE,
    batch_size = BATCH_SIZE,
    label_mode = "categorical",
    class_names = CLASSES
)

print(train_dataset,test_dataset,sep="\n")

#data agumentation
train=ImageDataGenerator(horizontal_flip=True,
                         vertical_flip=True,
                         validation_split=0.1,
                         rescale=1./255,
                         shear_range = 0.1,
                         zoom_range = 0.1,
                         width_shift_range = 0.1,
                         height_shift_range = 0.1,)

test=ImageDataGenerator(rescale=1/255,
                        validation_split=0.1)

train_generator=train.flow_from_directory(DATA_DIR,
                                          target_size=(300,300),
                                          batch_size=16,
                                          class_mode='categorical',
                                          subset='training')

test_generator=test.flow_from_directory(DATA_DIR,
                                        target_size=(300,300),
                                        batch_size=16,
                                        class_mode='categorical',
                                        subset='validation')

labels = (train_generator.class_indices)
print(labels)

labels = dict((v,k) for k,v in labels.items())
print(labels)

print("Image Processing.......Compleated")

print("Building Neural Network.....")
model=Sequential()
#Convolution blocks

model.add(Conv2D(32,(3,3),strides=(1,1), padding='same',input_shape=(300,300,3),activation='relu'))
model.add(MaxPooling2D(pool_size=2)) 
#model.add(SpatialDropout2D(0.5)) # No accuracy

model.add(Conv2D(64,(3,3), padding='same',activation='elu'))
model.add(MaxPooling2D(pool_size=1)) 
#model.add(SpatialDropout2D(0.5))

model.add(Conv2D(32,(3,3), padding='same',activation='leaky_relu'))
model.add(MaxPooling2D(pool_size=2)) 

#Classification layers
model.add(Flatten())

model.add(Dense(64,activation='relu'))
#model.add(SpatialDropout2D(0.5))
model.add(Dropout(0.2))
model.add(Dense(32,activation='relu'))

model.add(Dropout(0.2))
model.add(Dense(len(CLASSES),activation='softmax'))


filepath="trained_model.h5"
checkpoint1 = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint1]
print(callbacks_list)

"""

print("Training cnn")
cnn.fit(x = train_dataset,validation_data = test_dataset, epochs = 5)# validation_data = test_dataset
cnn.save_weights("w.h5")
cnn.load_weights("w.h5")

"""

early_stopping = EarlyStopping(monitor='val_loss',patience=10,verbose=1,mode='min')
model_save = ModelCheckpoint('garbage_detector',save_best_only=True,monitor='val_loss',mode='min')
reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss',factor=0.01,patience=7,verbose=1,min_delta=0.001,mode='min')
optimiser = tf.keras.optimizers.Adam(learning_rate=0.0002,amsgrad=True)

#compile model
model.compile(loss='categorical_crossentropy',optimizer=optimiser,metrics=['acc'])

print("Training cnn")
history = model.fit(train_generator,
                              epochs=5,
                              steps_per_epoch=16,#2276//32
                              validation_data=test_generator,
                              validation_steps=5,#"""251//32""",
                              workers = 4,
                              callbacks=[early_stopping,model_save,reduce_lr_loss])


# serialize the model to disk
print("... saving mask detector model...")
model.save("garbage_detector.model", save_format="h5")

