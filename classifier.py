import gc

from vocabulary import Vocabulary
from single_model import SingleModel

class Classifier(object):
    def __init__(self, model=None, filename=None, max_words=500):
        self.target_model = model(filename)
        self.vocab = Vocabulary()
        self.max_words = max_words

    def build(self):
        self.vocab.build()
        self.target_model.build()
        return self

    def predict(self, X):
        X = self.vocab.vectorize(X)
        X = self.target_model.model.predict(X)
        return X[0]

if __name__ == "__main__":
    a = Classifier(model=SingleModel, filename="single_84.588acc_model.h5")
    a.build()
    print(a.predict("i love you"))

    gc.collect()
