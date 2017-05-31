import os

from keras.models import model_from_json
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from single_model import SingleModel
import numpy as np
np.random.seed(1337)  # for reproducibility


class Manager(object):
    def __init__(self, model):
        self.model = model
        self.X_train = []
        self.y_train = []
        self.X_test = []
        self.y_test = []
        self.scores = 0
        # create int to word dictionary
        self.intToWord = {}

    def load_dataset(self, max_features=5000, max_len=500):
        """Loads dataset. Before execute this function,
        get dataset and unzip: http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"""

        # read in the train data
        path = 'dataset/train/pos/'
        self.X_train.extend([open(path + f).read() for f in os.listdir(path) if f.endswith('.txt')])
        self.y_train.extend([1 for _ in range(12500)])

        path = 'dataset/train/neg/'
        self.X_train.extend([open(path + f).read() for f in os.listdir(path) if f.endswith('.txt')])
        self.y_train.extend([0 for _ in range(12500)])

        print('x:')
        print(self.X_train[:1])
        print(self.X_train[-1:])
        print(len(self.X_train))
        print('y:')
        print(self.y_train[:1])
        print(self.y_train[-1:])
        print(len(self.y_train))

        # read in the test data
        path = 'dataset/test/pos/'
        self.X_test.extend([open(path + f).read() for f in os.listdir(path) if f.endswith('.txt')])
        self.y_test.extend([1 for _ in range(12500)])

        path = 'dataset/test/neg/'
        self.X_test.extend([open(path + f).read() for f in os.listdir(path) if f.endswith('.txt')])
        self.y_test.extend([0 for _ in range(12500)])

        print('x:')
        print(self.X_test[:1])
        print(self.X_test[-1:])
        print(len(self.X_test))
        print('y:')
        print(self.y_test[:1])
        print(self.y_test[-1:])
        print(len(self.y_test))

        # tokenize works to list of integers where each integer is a key to a word
        imdbTokenizer = Tokenizer(num_words=max_features)
        imdbTokenizer.fit_on_texts(self.X_train)

        # add a symbol for null placeholder
        self.intToWord[0] = "!!!NA!!!"

        # convert word strings to integer sequence lists
        self.X_train = imdbTokenizer.texts_to_sequences(self.X_train)
        self.X_test = imdbTokenizer.texts_to_sequences(self.X_test)

        # Censor the data by having a max review length (in number of words)
        print(len(self.X_train), 'train sequences')
        print(len(self.X_test), 'test sequences')

        print("Pad sequences (samples x time)")
        self.X_train = sequence.pad_sequences(self.X_train, maxlen=max_len)
        self.X_test = sequence.pad_sequences(self.X_test, maxlen=max_len)

        print('X_train shape:', self.X_train.shape)
        print('X_test shape:', self.X_test.shape)

        self.y_train = np.array(self.y_train)
        self.y_test = np.array(self.y_test)

        # example of a sentence sequence, note that lower integers are words that occur more commonly
        print("x:", self.X_train[0])  # per observation vector of 20000 words
        print("y:", self.y_train[0])  # positive or negative review encoding

        # double check that word sequences behave/final dimensions are as expected
        print("y distribution:", np.unique(self.y_train, return_counts=True))
        print("max x word:", np.max(self.X_train), "; min x word", np.min(self.X_train))
        print("y distribution test:", np.unique(self.y_test, return_counts=True))
        print("max x word test:", np.max(self.X_test), "; min x word", np.min(self.X_test))

        print("most and least popular words: ")
        print(np.unique(self.X_train, return_counts=True))

    def train(self):
        """ trains model"""

        # Fit the model
        self.model.fit(self.X_train, self.y_train, validation_data=(self.X_test, self.y_test),
                       epochs=2, batch_size=128, verbose=2)
        # Final evaluation of the model
        self.scores = self.model.evaluate(self.X_test, self.y_test, verbose=0)
        print("Accuracy: %.2f%%" % (self.scores[1] * 100))

    def store_model(self, filename):
        """ serializes model"""

        model_json = self.model.to_json()
        with open(filename + "_" + str(self.scores[1] * 100) + "acc_" + "model.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        model.save_weights(filename + "_" + str(self.scores[1] * 100) + "acc_" + "model.h5")
        print("Saved model to disk")

    def load_model(self, filename):
        """load model. You should put filename without file extention"""

        json_file = open(filename + ".json", 'rt')
        loaded_model_json = json_file.read()
        json_file.close()
        self.model = model_from_json(loaded_model_json)
        # load weights into new model
        self.model.load_weights(filename + ".h5")
        print("Loaded model from disk")
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


if __name__ == "__main__":
    model = SingleModel().build().model
    manager = Manager(model)
    manager.load_dataset()
    manager.train()
    manager.store_model("single")
