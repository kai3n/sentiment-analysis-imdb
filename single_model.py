# -*- coding: utf-8 -*-

import numpy as np
import gc
import re

from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.layers.embeddings import Embedding
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Dense
from keras.layers import Flatten

class SingleModel(object):
    def __init__(self):
        self.model = None

    def build(self):
        model = Sequential()
        model.add(Embedding(5000, 32, input_length=500))
        model.add(Flatten())
        model.add(Dense(250, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))

        model.load_weights("model.h5")
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.model = model

        return self


