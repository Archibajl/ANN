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
#import sys
#import os
#import pkg_resources
#from pprint import pprint


(images_train, labels_train), (images_test, labels_test) = tf.keras.datasets.cifar10.load_data()
images_train = images_train.reshape((60000, 28, 28, 1))
images_test = images_test.reshape((10000, 28, 28, 1))
images_train = images_train.astype('float32')
images_train = images_train/255
images_test = images_test.astype('float32')
images_test = images_test/255
labels_train = to_categorical(labels_train)
labels_test = to_categorical(labels_test)
model = tf.keras.models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(32, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(20, activation='relu'))
model.compile(optimizer=optimizers.RMSprop(lr=1e-4),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(images_train, labels_train,
          epochs=10,
          batch_size=30,
          validation_data=(images_test, labels_test),
          verbose=0)
results = model.evaluate(images_test, labels_test)
print(results)

   # x_train = vectorizeSequences(train_data)
   # x_test = vectorizeSequences(test_data)
   # vectorize_sequences((x_train, y_train))
   #model.add(layers.Dense(1, activation = 'sigmoid'))