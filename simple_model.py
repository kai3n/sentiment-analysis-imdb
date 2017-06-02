from keras.layers.embeddings import Embedding
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D


class SimpleModel(object):
    def __init__(self):
        self.model = None

    def build(self, embedding_matrix, max_features=20000, maxlen=500, embedding_dims=100):
        model = Sequential()

        print(embedding_matrix)
        model.add(Embedding(max_features,
                            embedding_dims,
                            weights=[embedding_matrix],
                            input_length=maxlen,
                            trainable=True))

        model.add(GlobalAveragePooling1D())
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        print(model.summary())
        self.model = model

        return self