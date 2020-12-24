# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 01:06:20 2020

@author: Allen
"""

from pygame.locals import *
import random
import pygame
import time
import numpy as np
import pickle
import math
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten
from tensorflow.keras.optimizers import Adam
from collections import deque



with open('test_imitation_data', 'rb') as file:
    data = pickle.load(file)
print(type(data))



model = Sequential()
model.add(Conv2D(32, (3,3), input_shape=(20,20,1)))
model.add(Activation("relu"))
model.add(MaxPooling2D(2,2))
#model.add(Dropout(0.2))

model.add(Conv2D(32, (3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(2,2))
#model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(64))

model.add(Dense(4, activation="linear"))
model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=['accuracy'])


