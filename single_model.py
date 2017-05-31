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
    def __init__(self, filename):
        self.model = None
        self.weights_filename = filename

    def build(self):
        model = Sequential()
        model.add(Embedding(5000, 32, input_length=500))
        model.add(Flatten())
        model.add(Dense(250, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))

        model.load_weights(self.weights_filename)
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        print(model.summary())
        self.model = model

        return self


