from keras.models import model_from_json
from vocabulary import Vocabulary

import gc


class Classifier(object):
    def __init__(self, filename, max_words=500):
        self.filename = filename
        self.model = None
        self.vocab = Vocabulary()
        self.max_words = max_words

    def build(self):
        self.vocab.build()

        json_file = open("model/" + self.filename + ".json", 'rt')
        loaded_model_json = json_file.read()
        json_file.close()
        self.model = model_from_json(loaded_model_json)
        # load weights into new model
        self.model.load_weights("model/" + self.filename + ".h5")
        print("Loaded model from disk")
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return self

    def predict(self, X):
        X = self.vocab.vectorize(X)
        X = self.model.model.predict(X)
        return X[0]

if __name__ == "__main__":
    a = Classifier(filename="single_84.588acc_model")
    a.build()
    print(a.predict("i love you"))

    gc.collect()
