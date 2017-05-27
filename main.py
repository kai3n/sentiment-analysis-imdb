# -*- coding: utf-8 -*-

import numpy as np
import gc
import re

from keras.datasets import imdb
from keras.preprocessing import sequence

from singleNN import SingleModel

TOP_WORDS = 5000
MAX_WORDS = 500

def refine_text(review_text):
    """Remove impurities from the text"""

    review_text = re.sub(r"[^A-Za-z0-9!?\'\`]", " ", review_text)
    review_text = re.sub(r"it's", " it is", review_text)
    review_text = re.sub(r"that's", " that is", review_text)
    review_text = re.sub(r"\'s", " 's", review_text)
    review_text = re.sub(r"\'ve", " have", review_text)
    review_text = re.sub(r"won't", " will not", review_text)
    review_text = re.sub(r"don't", " do not", review_text)
    review_text = re.sub(r"can't", " can not", review_text)
    review_text = re.sub(r"cannot", " can not", review_text)
    review_text = re.sub(r"n\'t", " n\'t", review_text)
    review_text = re.sub(r"\'re", " are", review_text)
    review_text = re.sub(r"\'d", " would", review_text)
    review_text = re.sub(r"\'ll", " will", review_text)
    review_text = re.sub(r"!", " ! ", review_text)
    review_text = re.sub(r"\?", " ? ", review_text)
    review_text = re.sub(r"\s{2,}", " ", review_text)
    return review_text.lower()

def word2idx(word):
    """Returns idx of word"""
    d = imdb.get_word_index()
    return d.get(word, -1)

def idx2word(idx):
    """Returns word of idx"""
    d = {v: k for k, v in imdb.get_word_index().items()}
    return d.get(idx, -1)

def make_vector(review_text):
    """Makes word be vectors"""

    review_text = refine_text(review_text)
    review_vector = []
    for i in review_text.split(' '):
        if word2idx(i) != -1 and word2idx(i) < TOP_WORDS:
            review_vector.append(word2idx(i))
    review_vector = np.array(review_vector).reshape((1, len(review_vector)))
    review_vector = sequence.pad_sequences(review_vector, maxlen=MAX_WORDS)
    gc.collect()
    return review_vector

def main():
    # text = """I gave this film my rare 10 stars.<br /><br />When I first began watching it and realized it would not be a film with a strong plot line I almost turned it off. I am very glad I didn't.<br /><br />This is a character driven film, a true story, which revolves mainly around the life of Rachel "Nanny" Crosby, a strong, beautiful (inside and out)Black woman and how she touched the lives of so many in the community of Lackawanna.<br /><br />Highly interesting not only its strong characterizations of Nanny and the people who lived at her boardinghouse, but also it gives us a look at what life and community were like for African Americans in the 1950's, prior to integration, and the good and bad sides of segregation and how it ultimately affected and changed the Black community.<br /><br />In addition to excellent performances by all members of the cast, there is some fine singing and dancing from that era."""
    s = SingleModel()
    s.load_model()
    while True:
        text = input()
        review_vector = make_vector(text)
        prediction = s.model.predict(review_vector, verbose=0)
        print(prediction[0])
        gc.collect()


if __name__ == "__main__":
    main()
