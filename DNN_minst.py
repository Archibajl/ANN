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


(images_train, labels_train), (images_test, labels_test) = tf.keras.datasets.minst.load_data()
#images_train = images_train.reshape((60000, 28, 28, 1))
images_train = images_train.astype('float32')
images_train = images_train/255
images_test = images_test.astype('float32')
images_test = images_test/255
labels_train = to_categorical(labels_train)
labels_test = to_categorical(labels_test)
#validation_features = x_train[:10000]
#validation_labels = y_train[:10000]
#train_features = np.reshape(x_train, (100000, 4 * 4 * 512))
#validation_features = np.reshape(validation_features, (10000, 4 * 4 * 512))
#test_features = np.reshape(x_test, (1000, 4 * 4 * 512))
model = tf.keras.models.Sequential()
model.add(layers.Flatten())
model.add(layers.Dense(32, activation='elu'))
#model.add(layers.Dense(32, activation='elu'))
model.add(layers.Dense(64, activation='elu'))
model.add(layers.Dense(22, activation='elu'))
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(images_train, labels_train,
          epochs=1,
          batch_size=64,
          validation_data=(images_test, labels_test),
          verbose=2)
test_loss, test_accuracy = model.evaluate(images_test, labels_test)
print("loss is ")
print(test_loss)
print('\n')
print("test accuracy is")
print(test_accuracy)


def vectorize_sequences(sequences, dimension=10000):
   results = np.zeros((len(sequences), dimension))
   for i, sequence in  enumerate(sequences):
      results[i, sequence] = 1.
   return results
