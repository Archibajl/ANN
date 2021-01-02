import tensorflow as tf
#import tensornetwork as tn
import numpy as np
#import pandas as pd
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator

print(tf.version.VERSION)
(images_train, labels_train), (images_test, labels_test) = tf.keras.datasets.mnist.load_data()
#images_train = ImageDataGenerator(rescale=1./255)
images_train = images_train.astype('float32')
images_train = images_train/255
#images_train = np.reshape(images_train/784, (60000, 28, 28, 1))
images_train = images_train.reshape((60000, 28, 28, 1))
images_test = images_test.astype('float32')
#images_test = np.reshape(images_test/784, (10000, 28, 28, 1))
images_test = images_test.reshape((10000, 28, 28, 1))
labels_train = to_categorical(labels_train)
labels_test = to_categorical(labels_test)
model = tf.keras.models.Sequential()
#model.add(layers.Softmax())
model.add(layers.Conv2D(32, (3, 3), activation = 'sigmoid',
                                 input_shape=(28, 28, 1)))
#model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.MaxPooling2D(2, 2))
#model.add(layers.Dropout(0.5))
model.add(layers.Conv2D(64, (3, 3), activation = 'sigmoid'))
model.add(layers.MaxPooling2D(2, 2))
#model.add(layers.Dropout(0.5))
model.add(layers.Conv2D  (64, (3, 3), activation = 'sigmoid'))
model.add(layers.MaxPooling2D(2, 2))
model.add(layers.Flatten())
#model.add(layers.Dropout(0.5))
model.add(layers.Dense(10, activation='softmax'))
model.compile(optimizer=optimizers.RMSprop(),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(images_train, labels_train,
          epochs=11,
          batch_size=64,
          validation_data=(images_test, labels_test))
#print(labels_train.shape)
#print(labels_test.shape)
#model.evaluate(images_test,  labels_test)
#train_datagen = ImageDataGenerator(
#    rescale=1./255,
#    rotation_range=40,
#    width_shift_range=0.2,
#    height_shift_range=0.2,
#   shear_range=0.2,
#    zoom_range=0.2,
#    horizontal_flip=True)
#test_datagen = ImageDataGenerator(rescale=1./255)
#train_generator=train_datagen(
#    tf.keras.datasets.mnist.load_data(),
#    target_size=(150, 150),
#    batch_size=32,
#    class_mode='categorical')
#validation_generator = test_datagen.flow_from_data(
#    (images_test, labels_test),
#    target_size=(150, 150),
#    batch_size=32,
#    class_mode='binary')
#labels_train = to_categorical(labels_train)
#labels_test = to_categorical(labels_test)
#input_shape = images_train.shape
#model.fit_generator((images_train, labels_train),
#          train_generator,
#          steps_per_epoch=100,
#          epochs=100,
#          validation_data=validation_generator,
#          validation_steps=50)
#results = model.evaluate(images_test, labels_test)
#print(results)

   # x_train = vectorizeSequences(train_data)
   # x_test = vectorizeSequences(test_data)
   # vectorize_sequences((x_train, y_train))
   #model.add(layers.Dense(1, activation = 'sigmoid'))