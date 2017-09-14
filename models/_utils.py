import functools as ft
from collections import namedtuple, OrderedDict

import numpy as np
import simplejson

__all__ = ['lazy_property', 'minibatch']


def lazy_property(function):
    attribute = '_cache_' + function.__name__

    @property
    @ft.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, function(self))
        return getattr(self, attribute)

    return decorator


def minibatch(numpy_array, batch_size=1, min_batch_size=1):
    numpy_array = np.asarray(numpy_array)
    assert 0 < min_batch_size <= batch_size, \
        "batch_size (%d) has to be larger than min_batch_size (%d) and they both have to be greater than zero!" % \
        (batch_size, min_batch_size)
    # go through the first dimension of the input array.
    for i in iter(range((numpy_array.shape[0] // batch_size) + 1)):
        idx = i * batch_size
        data = numpy_array[idx:(idx + batch_size)]
        if data.shape[0] >= min_batch_size:
            yield data


DataSet = namedtuple('DataSet', ['word_idx', 'phrase_lengths', 'labels'])


def load_dataset(path, max_len, vocab_size, retvocab=False):
    with open(path, 'r') as buf:
        raw = simplejson.load(buf)
        data = raw['data']
        vocab = raw['vocab']

    valid_data = [d for d in data.values() if d['text'] is not None]
    phrases = [[word for word in d['text'] if word < vocab_size]
               for d in valid_data]
    phrase_lengths = np.asarray([len(phrase) for phrase in phrases])
    phrase_lengths[phrase_lengths > max_len] = max_len
    labels = np.asarray([d['label'] for d in valid_data])
    word_idx = np.zeros((len(labels), max_len), dtype=np.int32)

    for w, phrase in zip(word_idx, phrases):
        length = min(max_len, len(phrase))
        w[:length] = phrase[:length]

    sel = phrase_lengths > 0
    dataset = DataSet(word_idx=word_idx[sel], phrase_lengths=phrase_lengths[sel],
                      labels=labels[sel])

    vocab = OrderedDict(sorted(((k, v) for k, v in vocab.items() if v < vocab_size),
                               key=lambda x: x[1]))

    return (dataset, vocab) if retvocab else dataset
