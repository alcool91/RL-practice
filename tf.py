# -*- coding: utf-8 -*-
"""
Created on Sat Dec 12 22:10:28 2020

@author: Allen
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

print(tf.__version__)
print(tf.test.is_built_with_cuda())
print(tf.test.is_gpu_available(cuda_only=False,min_cuda_compute_capability=None))


# mnist = tf.keras.datasets.mnist

# (x_train, y_train), (x_test, y_test) = mnist.load_data()
# x_train = x_train / 255
# x_test = x_test / 255

# model = tf.keras.models.Sequential()
# model.add(tf.keras.layers.Flatten())
# model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
# model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
# model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

# model.compile(optimizer='adam',
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy']
#               )

# model.fit(x_train, y_train, epochs=3)
# val_loss, val_acc = model.evaluate(x_test, y_test)

# print(val_loss, val_acc)
# print(x_train[0])

# plt.imshow(x_train[0])