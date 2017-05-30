import re
import codecs
import numpy as np

from keras.preprocessing.sequence import pad_sequences

MAX_WORDS = 500

class Vocabulary(object):

    def __init__(self, vocab_path='dataset/imdb.vocab'):
        self.vocab = dict()
        self.vocab_path = vocab_path

    def build(self):
        with codecs.open(self.vocab_path, 'r', 'UTF-8') as trainfile:
            words = [x.strip().rstrip('\n') for x in trainfile.readlines()]
            self.vocab = dict((c, i + 1) for i, c in enumerate(words))
        return self

    def size(self):
        return len(self.vocab)

    def refine(self, text):
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

    def pad(self, X):
        return pad_sequences(X, maxlen=500)

    def tokenize(self, text):
        text = self.refine(text)
        return [x.strip() for x in re.split('(\W+)', text) if x.strip()]

    def vectorize(self, text):
        words = filter(lambda x: x in self.vocab, self.tokenize(text))
        words = [self.vocab[w] for w in words]
        words = np.array(words).reshape((1, len(words)))
        words = pad_sequences(words, maxlen=MAX_WORDS)
        return words

if __name__ == "__main__":
    test = Vocabulary().build()
    print(test.vectorize("i love you"))
