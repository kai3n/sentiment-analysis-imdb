from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM


class LSTMModel(object):
    def __init__(self):
        self.model = None

    def build(self, max_features=20000, embedding_dims=100):
        model = Sequential()
        model.add(Embedding(max_features, embedding_dims))
        model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        print(model.summary())
        self.model = model

        return self