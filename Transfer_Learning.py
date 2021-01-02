import tensorflow as tf
#import tensornetwork as tn
import numpy as np

from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16

def ShrinkArray(array):
    ret_array = 0
    ret_array.resize((array.size[0]/12, ))
    for i in range(0, (array.size[0]/12).asinteger_ratio):
        for j in range(0, 27):
            for k in range(0, 27):
                ret_array[i, j, k, 1] = array[i, j, k]
    return ret_array

(images_train, labels_train), (images_test, labels_test) = tf.keras.datasets.mnist.load_data()
images_train = images_train.astype('float32')
images_train = images_train/255
#print(images_train)
#images_train2 = ShrinkArray(images_train)
#images_temp = images_train.reshape((60000, 28, 28, 1))
images_train = np.resize(images_train, (2000, 224, 224, 3))
images_test = images_test.astype('float32')
images_test = images_test/255
#images_test2 = ShrinkArray(images_test)
#images_temp = images_test.reshape((10000, 28, 28, 1))
images_test = np.resize(images_test, (333, 224, 224, 3))
#print(images_test)
labels_train = to_categorical(labels_train)
labels_test = to_categorical(labels_test)
labels_temp = labels_train
labels_train = labels_temp[0:2000]
labels_temp2 = labels_test
labels_test = labels_temp2[0:333]
VGG16_base = VGG16(weights='imagenet',
                  include_top=True
                 # input_shape=(28, 28, 1)
                           )
VGG16_base.trainable = False;
tl_model = models.Sequential()
tl_model.add(VGG16_base)
tl_model.add(layers.Flatten())
tl_model.add(layers.Dense(100, activation='sigmoid'))
tl_model.add(layers.Dense(10, activation='softmax'))

tl_model.compile(optimizer=optimizers.RMSprop(),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
tl_model.fit(images_train, labels_train,
              epochs=1,
              batch_size=60,
              validation_data=(images_test, labels_test))


