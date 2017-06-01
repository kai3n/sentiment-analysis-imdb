from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D


class CNNModel(object):
    def __init__(self):
        self.model = None

    def build(self, max_features=20000, maxlen=500, embedding_dims=32, filters=10,
              kernel_size=3, hidden_dims=10):
        model = Sequential()

        # we start off with an efficient embedding layer which maps
        # our vocab indices into embedding_dims dimensions
        model.add(Embedding(max_features,
                            embedding_dims,
                            input_length=maxlen))
        model.add(Dropout(0.2))

        # we add a Convolution1D, which will learn filters
        # word group filters of size filter_length:
        model.add(Conv1D(filters,
                         kernel_size,
                         padding='valid',
                         activation='relu',
                         strides=1))
        # we use max pooling:
        model.add(GlobalMaxPooling1D())

        # We add a vanilla hidden layer:
        model.add(Dense(hidden_dims))
        model.add(Dropout(0.2))
        model.add(Activation('relu'))

        # We project onto a single unit output layer, and squash it with a sigmoid:
        model.add(Dense(1))
        model.add(Activation('sigmoid'))

        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        print(model.summary())
        self.model = model

        return self