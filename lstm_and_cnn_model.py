from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D


class LSTMAndCNNModel(object):
    def __init__(self):
        self.model = None

    def build(self, max_features=5000, embedding_dims=100, maxlen=500):
        model = Sequential()
        model.add(Embedding(max_features, embedding_dims, input_length=maxlen))
        model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
        model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        print(model.summary())
        self.model = model

        return self