import logging as logger
import os
from collections import Counter, OrderedDict
from glob import glob

import nltk
import pypandoc
import simplejson
from tqdm import tqdm as tqdm


FILTERMATH_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               'filtermath.py')

REPLACE_RULES = [('%', r'\%')]


def generate_tokens(raw_text, stemmatize):
    for rule in REPLACE_RULES:
        raw_text = raw_text.replace(*rule)

    text = pypandoc.convert_text(raw_text, 'plain', 'latex',
                                    filters=[FILTERMATH_PATH])
    text = text.lower().encode('ascii', errors='ignore').decode()
    return [stemmatize(token)
            for token in nltk.tokenize.word_tokenize(text)
            if len(token) > 1]


def load_dataset(path):
    with open(path, 'r') as buf:
        return simplejson.load(buf)


def generate_vocab(globpath, stemmatize=None):
    vocab = Counter()

    if stemmatize is None:
        stemmatizer = nltk.stem.PorterStemmer()
        stemmatize = stemmatizer.stem

    for path in tqdm(glob(globpath)):
        data = load_dataset(path)
        try:
            tokens = generate_tokens(data['abstract'], stemmatize)
            vocab.update(tokens)
        except RuntimeError as e:
            logger.error('Error occured during processing of ' + data['id'])
            logger.error(str(e))

    return vocab


def main(globpath, outpath, stemmatize=None):
    vocab = generate_vocab(globpath)
    with open(outpath, 'w') as buf:
        simplejson.dump(OrderedDict(vocab.most_common()), buf)


if __name__ == '__main__':
    main('data/raw/*.json', 'data/vocab.json')
