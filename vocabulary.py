import re
import os
import codecs
import numpy as np

from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

MAX_WORDS = 500

class Vocabulary(object):

    def __init__(self):
        pass

    def build(self):
        #  TODO: save this as a file
        self.imdbTokenizer = Tokenizer(num_words=4999)
        self.vocab = []
        path = 'dataset/train/pos/'
        self.vocab.extend([open(path + f).read() for f in os.listdir(path) if f.endswith('.txt')])
        path = 'dataset/train/neg/'
        self.vocab.extend([open(path + f).read() for f in os.listdir(path) if f.endswith('.txt')])
        self.imdbTokenizer.fit_on_texts(self.vocab)
        return self

    def size(self):
        return len(self.vocab)

    def _refine(self, text):
        """Remove impurities from the text"""

        text = re.sub(r"[^A-Za-z0-9!?\'\`]", " ", text)
        text = re.sub(r"it's", " it is", text)
        text = re.sub(r"that's", " that is", text)
        text = re.sub(r"\'s", " 's", text)
        text = re.sub(r"\'ve", " have", text)
        text = re.sub(r"won't", " will not", text)
        text = re.sub(r"don't", " do not", text)
        text = re.sub(r"can't", " can not", text)
        text = re.sub(r"cannot", " can not", text)
        text = re.sub(r"n\'t", " n\'t", text)
        text = re.sub(r"\'re", " are", text)
        text = re.sub(r"\'d", " would", text)
        text = re.sub(r"\'ll", " will", text)
        text = re.sub(r"!", " ! ", text)
        text = re.sub(r"\?", " ? ", text)
        text = re.sub(r"\s{2,}", " ", text)
        return text.lower()

    def vectorize(self, text):
        text = self._refine(text)
        text = self.imdbTokenizer.texts_to_sequences([text])
        text = pad_sequences(text, maxlen=MAX_WORDS)
        return text

if __name__ == "__main__":
    test = Vocabulary().build()
    test.imdbTokenizer.texts_to_sequences(["I love you"])
    test = pad_sequences(test, maxlen=MAX_WORDS)
