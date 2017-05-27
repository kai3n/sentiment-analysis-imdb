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

    def __init__(self, top_words=5000, dimention=32, max_words=500):

        self.model = Sequential()
        self.model.add(Embedding(top_words, dimention, input_length=max_words))
        self.model.add(Flatten())
        self.model.add(Dense(250, activation='relu'))
        self.model.add(Dense(1, activation='sigmoid'))
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.x_train, self.y_train, self.x_text, self.y_text = self._load_imdb()
        print(self.model.summary())

    def _load_imdb(self, top_words=5000, max_words=500):
        """Returns x_train, y_train, x_test, y_text lists"""

        seed = 7
        np.random.seed(seed)
        (self.x_train, self.y_train), (self.x_test, self.y_test) = imdb.load_data(num_words=top_words)
        self.x_train = sequence.pad_sequences(self.x_train, maxlen=max_words)
        self.x_test = sequence.pad_sequences(self.x_test, maxlen=max_words)
        return self.x_train, self.y_train, self.x_test, self.y_test

    def train(self):
        """ trains model"""

        # Fit the model
        self.model.fit(self.x_train, self.y_train, validation_data=(self.x_test, self.y_test),
                       epochs=2, batch_size=128, verbose=2)
        # Final evaluation of the model
        scores = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        print("Accuracy: %.2f%%" % (scores[1]*100))

    def store_model(self):
        """ serializes model"""

        self.model_json = self.model.to_json()
        with open("model.json", "w") as json_file:
            json_file.write(self.model_json)
        # serialize weights to HDF5
        self.model.save_weights("model.h5")
        print("Saved model to disk")

    def load_model(self):
        """load model """

        self.json_file = open('model.json', 'r')
        self.loaded_model_json = self.json_file.read()
        self.json_file.close()
        self.model = model_from_json(self.loaded_model_json)
        # load weights into new model
        self.model.load_weights("model.h5")
        print("Loaded model from disk")
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    def evaluate_model(self):
        """ evaluates model"""

        self.load_model()
        score = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        print("Accuracy: %.2f%%" % (score[1] * 100))
        gc.collect()


if __name__ == "__main__":
    SingleModel()
    SingleModel().evaluate_model()

