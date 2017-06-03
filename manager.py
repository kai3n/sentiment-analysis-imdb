import os

from keras.models import model_from_json
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from single_model import SingleModel
from cnn_model import CNNModel
from lstm_and_cnn_model import LSTMAndCNNModel
from lstm_model import LSTMModel
from simple_model import SimpleModel
from gru_model import GRUModel

import numpy as np
np.random.seed(1337)  # for reproducibility

import numpy as np
from keras.preprocessing.text import Tokenizer


class Manager(object):
    def __init__(self, ngram_range=1, max_features=20000):
        self.model = None
        self.X_train = []
        self.y_train = []
        self.X_test = []
        self.y_test = []
        self.scores = 0
        self.max_features = max_features
        self.ngram_range = ngram_range
        # create int to word dictionary
        self.intToWord = {}
        self.embedding_matrix = None

    def load_dataset(self, max_len=500, embedding_dims=100):
        """Loads dataset. Before execute this function,
        get dataset and unzip: http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"""

        # read in the train data
        path = 'dataset/train/pos/'
        self.X_train.extend([open(path + f).read() for f in os.listdir(path) if f.endswith('.txt')])
        self.y_train.extend([1 for _ in range(12500)])

        path = 'dataset/train/neg/'
        self.X_train.extend([open(path + f).read() for f in os.listdir(path) if f.endswith('.txt')])
        self.y_train.extend([0 for _ in range(12500)])

        # read in the test data
        path = 'dataset/test/pos/'
        self.X_test.extend([open(path + f).read() for f in os.listdir(path) if f.endswith('.txt')])
        self.y_test.extend([1 for _ in range(12500)])

        path = 'dataset/test/neg/'
        self.X_test.extend([open(path + f).read() for f in os.listdir(path) if f.endswith('.txt')])
        self.y_test.extend([0 for _ in range(12500)])

        # tokenize works to list of integers where each integer is a key to a word
        imdbTokenizer = Tokenizer(num_words=self.max_features)
        imdbTokenizer.fit_on_texts(self.X_train)

        word_index = imdbTokenizer.word_index

        embeddings_index = {}
        f = open(os.path.join('glove.6B/', 'glove.6B.100d.txt'))
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        f.close()

        # prepare embedding matrix
        self.max_features = min(self.max_features, len(word_index))

        embedding_matrix = np.zeros((self.max_features, embedding_dims))
        for word, i in word_index.items():
            if i >= self.max_features:
                continue
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector
        self.embedding_matrix = embedding_matrix

        # add a symbol for null placeholder
        self.intToWord[0] = "!!!NA!!!"

        # convert word strings to integer sequence lists
        self.X_train = imdbTokenizer.texts_to_sequences(self.X_train)
        self.X_test = imdbTokenizer.texts_to_sequences(self.X_test)

        # Censor the data by having a max review length (in number of words)
        print(len(self.X_train), 'train sequences')
        print(len(self.X_test), 'test sequences')

        if self.ngram_range > 1:
            print('Adding {}-gram features'.format(self.ngram_range))
            # Create set of unique n-gram from the training set.
            ngram_set = set()
            for input_list in self.X_train:
                for i in range(2, self.ngram_range + 1):
                    set_of_ngram = self.create_ngram_set(input_list, ngram_value=i)
                    ngram_set.update(set_of_ngram)

            # Dictionary mapping n-gram token to a unique integer.
            # Integer values are greater than max_features in order
            # to avoid collision with existing features.
            start_index = self.max_features + 1
            token_indice = {v: k + start_index for k, v in enumerate(ngram_set)}
            indice_token = {token_indice[k]: k for k in token_indice}

            # max_features is the highest integer that could be found in the dataset.
            self.max_features = np.max(list(indice_token.keys())) + 1

            # prepare embedding matrix

            embedding_matrix = np.zeros((self.max_features, embedding_dims))
            for word, i in word_index.items():
                if i >= self.max_features:
                    continue
                embedding_vector = embeddings_index.get(word)
                if embedding_vector is not None:
                    # words not found in embedding index will be all-zeros.
                    embedding_matrix[i] = embedding_vector
            self.embedding_matrix = embedding_matrix

            # Augmenting x_train and x_test with n-grams features
            self.X_train = self.add_ngram(self.X_train, token_indice, self.ngram_range)
            self.X_test = self.add_ngram(self.X_test, token_indice, self.ngram_range)
            print('Average train sequence length: {}'.format(np.mean(list(map(len, self.X_train)), dtype=int)))
            print('Average test sequence length: {}'.format(np.mean(list(map(len, self.X_test)), dtype=int)))

        print("Pad sequences (samples x time)")
        self.X_train = sequence.pad_sequences(self.X_train, maxlen=max_len)
        self.X_test = sequence.pad_sequences(self.X_test, maxlen=max_len)

        print('X_train shape:', self.X_train.shape)
        print('X_test shape:', self.X_test.shape)

        self.y_train = np.array(self.y_train)
        self.y_test = np.array(self.y_test)

        # double check that word sequences behave/final dimensions are as expected
        print("y distribution:", np.unique(self.y_train, return_counts=True))
        print("max x word:", np.max(self.X_train), "; min x word", np.min(self.X_train))
        print("y distribution test:", np.unique(self.y_test, return_counts=True))
        print("max x word test:", np.max(self.X_test), "; min x word", np.min(self.X_test))

        print("most and least popular words: ")
        print(np.unique(self.X_train, return_counts=True))

    def train(self, model, batch_size=128, epochs=10):
        """ trains model"""
        self.model = model
        # Fit the model
        self.model.fit(self.X_train, self.y_train, validation_data=(self.X_test, self.y_test),
                       epochs=epochs, batch_size=batch_size, verbose=2)
        # Final evaluation of the model
        self.scores = self.model.evaluate(self.X_test, self.y_test, verbose=0)
        print("Accuracy: %.2f%%" % (self.scores[1] * 100))

    def store_model(self, filename):
        """ serializes model"""

        model_json = self.model.to_json()
        with open("model/" + filename + "_" + str(self.scores[1] * 100) + "acc_" + "model.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        self.model.save_weights("model/" + filename + "_" + str(self.scores[1] * 100) + "acc_" + "model.h5")
        print("Saved model to disk")

    def load_model(self, filename):
        """load model. You should put filename without file extention"""

        json_file = open("model/" + filename + ".json", 'rt')
        loaded_model_json = json_file.read()
        json_file.close()
        self.model = model_from_json(loaded_model_json)
        # load weights into new model
        self.model.load_weights("model/" + filename + ".h5")
        print("Loaded model from disk")
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    @staticmethod
    def create_ngram_set(input_list, ngram_value=2):
        return set(zip(*[input_list[i:] for i in range(ngram_value)]))

    @staticmethod
    def add_ngram(sequences, token_indice, ngram_range=2):
        new_sequences = []
        for input_list in sequences:
            new_list = input_list[:]
            for i in range(len(new_list) - ngram_range + 1):
                for ngram_value in range(2, ngram_range + 1):
                    ngram = tuple(new_list[i:i + ngram_value])
                    if ngram in token_indice:
                        new_list.append(token_indice[ngram])
            new_sequences.append(new_list)

        return new_sequences


if __name__ == "__main__":
    # cnn and bi-gram model
    manager = Manager(ngram_range=2)
    manager.load_dataset()
    model = CNNModel().build(embedding_matrix=manager.embedding_matrix,
                             max_features=manager.max_features).model
    manager.train(model)
    manager.store_model("cnn_and_bi-gram")











