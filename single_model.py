from keras.layers.embeddings import Embedding
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten


class SingleModel(object):
    def __init__(self):
        self.model = None

    def build(self, max_features=20000, maxlen=500, embedding_dims=100, hidden_dims=250):
        model = Sequential()
        model.add(Embedding(max_features, embedding_dims, input_length=maxlen))
        model.add(Flatten())
        model.add(Dense(hidden_dims, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        print(model.summary())
        self.model = model

        return self
